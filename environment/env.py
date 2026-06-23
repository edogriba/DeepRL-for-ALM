import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment.config import GAMMA, RHO, F, THETA
from environment.scenario import Scenario, DepositBetaLiabilityGenerator, MarkovYieldCurveGenerator
from models.utils import calculate_pv, identity_utility, cara_utility, get_discount_factor, print_holdings_matrix, get_nelson_siegel_yield
from environment.bond import Bond

class DeepALMEnv(gym.Env):
    def __init__(self,  markov_config=None, seed=None, verbose=False, use_dirichlet=False, eval_band=False):
        """
        Args:
            bond_configs: List of dicts, e.g., 
                [{'bond_type': '5Y', 'nominal_value': 100, 'maturity_months': 60, 
                  'coupon_dates': [6, 12, ...], 'annual_yield': 0.04}, ...]
                  
            yield_path = YieldCurveGenerator.generate(self.T, seed=episode_seed)
            liability_vector = LiabilityGenerator.generate(self.T, seed=episode_seed, low=0, high=100)
        """
        super().__init__()
        # Initialize Bond objects for each type k 
        self.bonds = [Bond(**cfg) for cfg in markov_config["bond_configs"]]
        self.K = len(self.bonds)
        # M is the maximum maturity across all bond types 
        self.M = max(b.M for b in self.bonds)
        self.T = markov_config["T"]
        self.scenario = None
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._seed_lo, self._seed_hi = (10_000_000, 20_000_000) if eval_band else (0, 10_000_000)
        self.markov_config = markov_config
        self.use_dirichlet = use_dirichlet
        self.B1_0 = markov_config["B1_0"]
        b0_init = markov_config["BETA0_GRID"][markov_config["i0_init"]]
        b1_init = markov_config["BETA1_GRID"][markov_config["i1_init"]]
        b2_init = markov_config["BETA2"]
        self.b1_issuance_yield = get_nelson_siegel_yield(2.0/12.0, [b0_init, b1_init, b2_init])
        self._k_2m = next((k for k, b in enumerate(self.bonds) if b.M == 2), None) # to place initial B1 in the correct slot if needed
        self.verbose = verbose
        low_bounds  = np.array([ 0.00, -0.20, -0.20], dtype=np.float32) # Lower bounds for components of yield
        high_bounds = np.array([ 0.20,  0.20,  0.20], dtype=np.float32) # Upper bounds for components of yield
        
        # State Space: (C_t, h_t, y_t, L_t) 
        self.observation_space = spaces.Dict({
            "cash": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "holdings": spaces.Box(low=0, high=np.inf, shape=(self.K, self.M)),
            "yields": spaces.Box(low=low_bounds, high=high_bounds, shape=(3,)),
            "liabilities": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "coupons": spaces.Box(low=0, high=np.inf, shape=(self.K, self.M))
        })
        
        if self.use_dirichlet:
            # Action Space: b_t^(k) fractions that sum to 1 (including cash)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.K + 1,), dtype=np.float32)
        else:
            # Action Space: b_t^(k) fractions
            self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.K + 1,), dtype=np.float32)        
        
    def _get_current_nav(self):
        # Project Asset Cash Flows (based on current holdings)
        asset_cfs = self._project_bond_cash_flows()
        
        # Calculate PV(Assets) = Cash + PV(Bond CFs)
        # (self.cash can now be negative, representing debt)
        pv_a = calculate_pv(self.cash, asset_cfs, self.yield_params)
        
        # Calculate PV(Liabilities)
        residual_liabs = self.scenario.liabilities[self.current_step + 1:].flatten()
        if residual_liabs.size > 0:
            pv_l = calculate_pv(0, residual_liabs, self.yield_params)
        else:
            pv_l = 0.0
            
        # Net Asset Value (Penalties are already absorbed by self.cash)
        nav = pv_a - pv_l
        
        return nav

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "scenario" in options:
            self.scenario = options["scenario"]
        else:
            episode_seed = int(self._rng.integers(self._seed_lo, self._seed_hi))
                        
            safe_config = {k: self.markov_config[k] for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"] if k in self.markov_config}

            yield_path = MarkovYieldCurveGenerator.generate(self.T * 2, seed=episode_seed, pure_grid=True, **safe_config)
            
            liability_vector = DepositBetaLiabilityGenerator.generate(yield_path, noise_std=0.0, seed=episode_seed)
            
            liability_vector[self.T + 1:] = 0.0
            
            self.scenario = Scenario(yields=yield_path, liabilities=liability_vector)            
        
        self.current_step = 0
        self.cash = self.markov_config["W0"]
        self.cumulative_penalty = 0.0
        
        self.holdings = np.zeros((self.K, self.M), dtype=np.float32)
        self.coupons  = np.zeros((self.K, self.M), dtype=np.float32)

        if self.B1_0 > 0.0 and self._k_2m is not None:
            r2_at_issuance = 1.0 + self.b1_issuance_yield * (2.0 / 12.0)
            effective_face = self.B1_0 / r2_at_issuance
            units = effective_face / self.bonds[self._k_2m].V
            self.holdings[self._k_2m, 1] = units
            self.coupons [self._k_2m, 1] = self.b1_issuance_yield
        
        self.yield_params = self.scenario.yields[0]
        self.liabilities = self.scenario.liabilities[:self.M].flatten()
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "cash": np.array([self.cash], dtype=np.float32),
            "holdings": self.holdings.astype(np.float32),
            "yields": self.yield_params.flatten().astype(np.float32),
            "liabilities": self.liabilities.flatten().astype(np.float32)[0:1],
            "coupons": self.coupons.astype(np.float32)
        }
        
    def _get_total_inflows(self):
        '''Vectorized calculation of total inflows from all bond holdings at the current step'''
        j_idx = np.arange(self.M)  # maturity buckets
        total_inflows = 0.0

        for k in range(self.K):
            bond = self.bonds[k]

            h = self.holdings[k]
            r = self.coupons[k]

            # remaining life and age calculations
            remaining_life = j_idx + 1
            current_age = bond.M - remaining_life
            next_age = current_age + 1

            # coupon size for each lot
            months_per_payment = (
                bond.coupon_dates[1] - bond.coupon_dates[0]
                if len(bond.coupon_dates) > 1
                else bond.coupon_dates[0]
            )
            coupon_amount = (r * bond.V) * (months_per_payment / 12.0)

            # conditions
            is_coupon = np.isin(next_age, bond.coupon_dates)
            is_maturity = next_age == bond.M

            # inflow for each bucket
            inflow = coupon_amount * is_coupon + bond.V * is_maturity

            total_inflows += np.sum(h * inflow)

        return total_inflows

    def _project_bond_cash_flows(self):
        ''' Vectorized projection of future cash flows from current bond holdings'''
        future_cfs = np.zeros(self.M)

        j_idx = np.arange(self.M)[:, None]
        tau_idx = np.arange(self.M)[None, :]

        for k in range(self.K):
            bond = self.bonds[k]

            h = self.holdings[k]
            r = self.coupons[k]

            # Remaining life for each lot
            remaining_life = j_idx + 1

            # Current age
            current_age = bond.M - remaining_life

            # Future ages grid
            future_age = current_age + (tau_idx + 1)

            # Lot only exists until maturity
            is_alive = tau_idx <= j_idx

            # Coupon size
            months_per_payment = (
                bond.coupon_dates[1] - bond.coupon_dates[0]
                if len(bond.coupon_dates) > 1
                else bond.coupon_dates[0]
            )
            coupon_amount = (r * bond.V) * (months_per_payment / 12.0)

            # Payment conditions
            is_coupon = np.isin(future_age, bond.coupon_dates)
            is_maturity = future_age == bond.M

            inflow = coupon_amount[:, None] * is_coupon + bond.V * is_maturity

            # Apply survival mask
            inflow *= is_alive

            # Multiply by holdings
            future_cfs += np.sum(h[:, None] * inflow, axis=0)

        return future_cfs
    
    def step(self, action):
        
        if self.verbose:
            print(f"Month {self.current_step+1}:")
            for k, bond in enumerate(self.bonds):
                entries = []
                for j in range(self.M):
                    if self.holdings[k, j] > 0:
                        entries.append(f"{j+1}m: {100*self.coupons[k,j]:.2f}%")
                print(f"{bond.M}M Bond Lots: {', '.join(entries) if entries else 'None'}")
            print(f"  Current Market Yields")   
            for m in range(1, self.K + 1):
                print(f"    {self.bonds[m-1].M}M: {get_nelson_siegel_yield(m/12.0, self.yield_params)*100:.2f}%")
            print(f"  Cash Before Action: {self.cash:.2f}")
        
        inflows = self._get_total_inflows()
        liability_due = self.liabilities[0].item()
        liability_due_next = self.liabilities[1].item() if len(self.liabilities) > 1 else 0.0

        # Update cash with inflows and pay liabilities
        self.cash = self.cash + inflows - liability_due
        
        # Early bankruptcy termination
        if self.cash < 0:
            if self.verbose:
                print(f"  [BANKRUPTCY] Cash is {self.cash:.2f}. Terminating episode early.")
            
            # Apply fixed bankruptcy fine
            self.cash -= F
            self.cumulative_penalty += F
            
            # Shift to avoid double counting
            new_h = np.zeros_like(self.holdings)
            new_h[:, :-1] = self.holdings[:, 1:]
            self.holdings = new_h
            
            new_c = np.zeros_like(self.coupons)
            new_c[:, :-1] = self.coupons[:, 1:]
            self.coupons = new_c
            
            # Calculate final metrics immediately
            nav = self._get_current_nav()
            utility = cara_utility(nav)
            
            info = {
                "lcr": 0.0,
                "pv_a": calculate_pv(self.cash, self._project_bond_cash_flows(), self.yield_params),
                "nav": nav,
                "penalty": self.cumulative_penalty,
                "utility": utility
            }
            
            # Return immediately with terminated = True
            return self._get_obs(), utility, True, False, info
            
        # If solvent continue normally
        investment_base = self.cash
        
        if self.use_dirichlet:
            # Dirichlet action are already fractional
            weights = action
        else:
            # Standard Gaussian needs softmax
            shifted_logits = action - np.max(action)
            exp_acts = np.exp(shifted_logits)
            weights = exp_acts / np.sum(exp_acts)

        # First k elements are related to bonds buying
        bond_weights = weights[:-1]
        
        # The last element is cash
        cash_weight = weights[-1]

        # Purchases are done with bond weights
        purchases = np.sum(bond_weights * investment_base)
        self.cash -= purchases
        
        # Roll states
        self._roll_states(bond_weights, investment_base)
        
        # Calculate LCR: HQLA = cash + inflows (coupons and/or face value) arriving NEXT month, discounted at 1M
        hqla = self.cash
        y_1m = get_nelson_siegel_yield(1.0 / 12.0, self.yield_params)
        r1 = 1.0 + y_1m * (1.0 / 12.0)
        
        for k in range(self.K):
            bond = self.bonds[k]
            months_per_pmt = bond.coupon_dates[0] if len(bond.coupon_dates) == 1 else bond.coupon_dates[1] - bond.coupon_dates[0]
            for m in range(self.M):
                age_next = bond.M - m  # age the lot in slot m will have next month
                coupon_next = (self.coupons[k, m] * bond.V * (months_per_pmt / 12.0)) if age_next in bond.coupon_dates else 0.0
                face_next = bond.V if m == 0 else 0.0
                hqla += (self.holdings[k, m] * (face_next + coupon_next)) / r1
                    
        lcr = hqla / (liability_due_next + 1e-8)
        
        # Deduct LCR Penalty directly from cash
        if lcr < 1.0:
            lcr_penalty = RHO * (1.0 - lcr) + THETA
            self.cash -= lcr_penalty
            self.cumulative_penalty += lcr_penalty # Keep tracking for info dict
        
        self.current_step += 1
        
        if self.current_step < self.scenario.T:
            self.liabilities = self.scenario.liabilities[self.current_step : self.current_step + self.M].flatten()
            self.yield_params = self.scenario.yields[self.current_step]
        
        terminated = self.current_step == self.T
        
        if self.verbose:
            print(f"  Inflows: {inflows:.2f}  |  Liability Due: {liability_due:.2f}  |  Next Liability: {liability_due_next:.2f}")
            print(f"  Cash After Action: {self.cash:.2f}")
            print(f"  LCR: {lcr:.4f}  |  Utility: {cara_utility(self._get_current_nav()):.2f}")
            time_buckets = [f"{m}M" for m in range(1, self.M + 1)]
            print_holdings_matrix(self.holdings * np.array([b.V for b in self.bonds])[:, None], self.bonds, time_buckets)
            
        nav = self._get_current_nav()  
        utility = 0.0
        asset_cfs = self._project_bond_cash_flows()
        pv_a = calculate_pv(self.cash, asset_cfs, self.yield_params)
        
        # Terminal MTM Alignment
        if terminated:
            # Fast-forward time by 1 month to match DP's terminal evaluation state
            df = get_discount_factor(1/12, self.yield_params)
            self.cash += self._get_total_inflows() - self.scenario.liabilities[self.T].item() 
            
            # Shift holdings forward one last time so 1M bonds clear out
            new_h = np.zeros_like(self.holdings)
            new_h[:, :-1] = self.holdings[:, 1:]
            self.holdings = new_h
            
            new_c = np.zeros_like(self.coupons)
            new_c[:, :-1] = self.coupons[:, 1:]
            self.coupons = new_c
            
            # Calculate PV of Assets ONLY
            asset_cfs = self._project_bond_cash_flows()
            pv_a = calculate_pv(self.cash, asset_cfs, self.yield_params)
            
            if len(self.liabilities) > 1:
                pv_l = calculate_pv(0, self.liabilities[1:], self.yield_params)
            else:
                pv_l = 0.0
            # NAV is simply pv_a because penalties are already deducted from self.cash
            nav = pv_a -pv_l
            
            # Apply penalty if going bankrupt
            if nav < 0:
                nav -= F
                
            utility = cara_utility(nav)
            
        info = {
            "lcr": lcr,
            "pv_a": pv_a,
            "nav": nav,
            "penalty": self.cumulative_penalty,
            "utility": utility
        }
        
        # Return sparse reward equivalent to utility at the final step
        return self._get_obs(), utility, terminated, False, info


    def _roll_states(self, action, investment_base):
        # Roll holdings
        new_h = np.zeros_like(self.holdings)
        new_h[:, :-1] = self.holdings[:, 1:]
        
        # Roll coupons
        new_coupons = np.zeros_like(self.coupons)
        new_coupons[:, :-1] = self.coupons[:, 1:]
        
        # Handle new purchases at par
        for k in range(self.K):
            bond = self.bonds[k]
            issuance_yield = get_nelson_siegel_yield(bond.M / 12.0, self.yield_params)
            new_units = (action[k] * investment_base) / bond.V
            
            new_h[k, bond.M - 1] += new_units
            new_coupons[k, bond.M - 1] = issuance_yield
            
        self.holdings = new_h
        self.coupons = new_coupons