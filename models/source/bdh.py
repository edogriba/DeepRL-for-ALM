import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from environment.bond import Bond
from models.utils import get_nelson_siegel_yield_batched, build_state
from environment.scenario import DepositBetaLiabilityGenerator, MarkovYieldCurveGenerator
from environment.config import GAMMA, RHO, F, THETA, markov_config

# Architecture

class SoftplusNormalize(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = torch.nn.functional.softplus(x) + self.eps
        return x / x.sum(dim=-1, keepdim=True)

class DeepALMPolicyDH(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.monthly_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                SoftplusNormalize()
            )
            for _ in range(T)
        ])

    def forward(self, t, state, prev_action):
        x = torch.cat([state, prev_action], dim=-1)
        return self.monthly_policies[t](x)
    
    def save(self, path: str):
        torch.save({"model_state_dict": self.state_dict(), "T": self.T}, path)

    def load(self, path: str, map_location=None, evaluation=False):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.T = checkpoint.get("T", self.T)
        if evaluation:
            self.eval()

# Loss and risk metrics

class OCEUtilityExp(nn.Module):
    def __init__(self, gamma: float = GAMMA):
        super().__init__()
        self.gamma = gamma
        self.y = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, X):
        gains = X + self.y
        args = -self.gamma * gains
        max_arg = torch.max(args).detach() 
        
        expectation_stabilized = torch.mean(torch.exp(args - max_arg))
        u = (1.0 - torch.exp(max_arg) * expectation_stabilized) / self.gamma - self.y
        return -u

def entropic_risk_measure_eval(terminal_values: torch.Tensor, gamma: float = GAMMA) -> float:
    args = -gamma * terminal_values
    max_arg = torch.max(args)
    expectation_stabilized = torch.mean(torch.exp(args - max_arg))
    risk = (max_arg + torch.log(expectation_stabilized)) / gamma
    return risk.item()

# Rollout engine

def batched_differentiable_rollout(
    model: torch.nn.Module, yields_batch: torch.Tensor, liabilities_batch: torch.Tensor,
    T: int, bond_maturities: list, bond_coupon_dates: list, bond_nominal: float = 100.0,
    initial_cash: float = markov_config["W0"], initial_holdings: torch.Tensor = None, initial_coupons: torch.Tensor = None,
    log_scenario_idx: int = None
):
    device = yields_batch.device
    batch_size = yields_batch.shape[0]
    K, max_M = len(bond_maturities), max(bond_maturities)

    cash = torch.full((batch_size, 1), initial_cash, device=device)
    holdings = initial_holdings.to(device).clone() if initial_holdings is not None else torch.zeros(batch_size, K, max_M, device=device)
    coupons = initial_coupons.to(device).clone() if initial_coupons is not None else torch.zeros(batch_size, K, max_M, device=device)

    prev_action = torch.zeros(batch_size, K + 1, device=device) 
    cumulative_penalty = torch.zeros(batch_size, 1, device=device) 
    history_log = [] 
    ever_bankrupt = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    is_alive = torch.ones(batch_size, 1, device=device, dtype=torch.bool)

    # Frozen mark-to-market NAV at the step of default.
    early_terminal_nav = torch.zeros(batch_size, 1, device=device)

    for t in range(T):
        betas_t = yields_batch[:, t]
        liability_due = liabilities_batch[:, t].view(-1, 1)

        inflows = torch.zeros(batch_size, 1, device=device)
        for k, M in enumerate(bond_maturities):
            coupon_dates_k = bond_coupon_dates[k]
            coupons_per_year_k = len(coupon_dates_k) * 12.0 / M
            inflows += holdings[:, k, 0:1] * bond_nominal
            for c in coupon_dates_k:
                slot_c = M - c  
                if 0 <= slot_c < max_M:
                    coupon_per_unit = bond_nominal * coupons[:, k, slot_c:slot_c+1] / coupons_per_year_k
                    inflows += holdings[:, k, slot_c:slot_c+1] * coupon_per_unit

        # Liquidity check
        cash_after_liab = cash + inflows - liability_due
        bankruptcy_trigger = (cash_after_liab < 0) & is_alive
        ever_bankrupt = ever_bankrupt | bankruptcy_trigger.squeeze(-1)
        
        y_1m_current = get_nelson_siegel_yield_batched(1.0 / 12.0, betas_t)
        r1_current = 1.0 + y_1m_current * (1.0 / 12.0)
        
        cash = torch.where(bankruptcy_trigger, cash_after_liab - F, cash_after_liab)
        illiquid_penalty = torch.where(bankruptcy_trigger, torch.tensor(float(F), device=device), torch.tensor(0.0, device=device))
        cumulative_penalty += illiquid_penalty

        # Freeze a discounted MTM NAV for paths defaulting this step.
        if bankruptcy_trigger.any():
            # Shift temporary matrices to clear out slot 0 (which was already added to cash)
            h_mtm = torch.zeros_like(holdings)
            c_mtm = torch.zeros_like(coupons)
            h_mtm[:, :, :-1] = holdings[:, :, 1:]
            c_mtm[:, :, :-1] = coupons[:, :, 1:]

            pv_assets = torch.zeros(batch_size, 1, device=device)
            for k, M in enumerate(bond_maturities):
                coupon_dates_k = bond_coupon_dates[k]
                coupons_per_year_k = len(coupon_dates_k) * 12.0 / M if M > 0 else 0
                for m in range(max_M):
                    units = h_mtm[:, k, m]  # Use shifted holdings
                    if units.abs().sum() == 0:
                        continue
                    tau_face = (m + 1) / 12.0
                    y_face = get_nelson_siegel_yield_batched(tau_face, betas_t)
                    pv_assets += (units * bond_nominal).unsqueeze(-1) / (1.0 + y_face * tau_face)
                    for c in coupon_dates_k:
                        age = M - m - 1
                        if c > age:
                            tau_c = (c - age) / 12.0
                            y_c = get_nelson_siegel_yield_batched(tau_c, betas_t)
                            coupon_amount = bond_nominal * c_mtm[:, k, m] / coupons_per_year_k # Use shifted coupons
                            pv_assets += (units * coupon_amount).unsqueeze(-1) / (1.0 + y_c * tau_c)

            pv_liabs = torch.zeros(batch_size, 1, device=device)
            L_total = liabilities_batch.shape[1]
            for j in range(max_M - 1):
                idx = t + 1 + j
                if idx >= L_total:
                    break
                tau_l = (j + 1) / 12.0
                y_l = get_nelson_siegel_yield_batched(tau_l, betas_t)
                pv_liabs += liabilities_batch[:, idx].view(-1, 1) / (1.0 + y_l * tau_l)

            mtm_nav = cash + pv_assets - pv_liabs
            early_terminal_nav = torch.where(bankruptcy_trigger, mtm_nav, early_terminal_nav)

        is_alive = is_alive & ~bankruptcy_trigger
        investable = torch.relu(cash) * is_alive.float()

        # Undiscounted nominal future-cash-flow projection.
        future_cash_projection = torch.zeros(batch_size, max_M, device=device)
        for k, M in enumerate(bond_maturities):
            coupon_dates_k = bond_coupon_dates[k]
            coupons_per_year_k = len(coupon_dates_k) * 12.0 / M if M > 0 else 0
            for m in range(max_M):
                future_cash_projection[:, m] += holdings[:, k, m] * bond_nominal
                for c in coupon_dates_k:
                    s = m + (M - c)
                    if 0 <= s < max_M:
                        future_cash_projection[:, m] += holdings[:, k, s] * (bond_nominal * coupons[:, k, s] / coupons_per_year_k)

        
        state = build_state(cash, future_cash_projection, betas_t, liability_due)
        weights = model(t, state, prev_action)

        if log_scenario_idx is not None and log_scenario_idx < batch_size:
            history_log.append({
                "t": t,
                "real_cash": cash[log_scenario_idx, 0].item(),
                "investable_budget": investable[log_scenario_idx, 0].item(),
                "real_liability": liability_due[log_scenario_idx, 0].item(),
                "network_weights": weights[log_scenario_idx].detach().cpu().numpy()
            })

        new_holdings, new_coupons = torch.zeros_like(holdings), torch.zeros_like(coupons)
        new_holdings[:, :, :-1], new_coupons[:, :, :-1] = holdings[:, :, 1:], coupons[:, :, 1:]

        for k, M in enumerate(bond_maturities):
            amount_k = weights[:, k:k+1] * investable
            issuance_yield = get_nelson_siegel_yield_batched(float(M) / 12.0, betas_t)
            new_holdings[:, k, M-1:M] += amount_k / bond_nominal
            new_coupons[:, k, M-1:M] = issuance_yield

        bond_spend = investable - (weights[:, K:K+1] * investable)
        cash, holdings, coupons, prev_action = cash - bond_spend, new_holdings, new_coupons, weights

        hqla = cash.clone()
        for k, M in enumerate(bond_maturities):
            coupon_dates_k = bond_coupon_dates[k]
            months_per_pmt_k = coupon_dates_k[0] if len(coupon_dates_k) == 1 else coupon_dates_k[1] - coupon_dates_k[0]

            # Final inflow at maturity: face value always, plus the last
            # coupon ONLY if the schedule has a payment at maturity.
            if M > 0:
                if M in coupon_dates_k:
                    coupon_k_final = coupons[:, k, 0:1] * bond_nominal * (months_per_pmt_k / 12.0)
                else:
                    coupon_k_final = torch.zeros_like(coupons[:, k, 0:1])
                hqla += holdings[:, k, 0:1] * (bond_nominal + coupon_k_final) / r1_current

            # Intermediate coupons
            for c in coupon_dates_k:
                slot_c = M - c
                if 0 < slot_c < max_M:
                    coupon_intermedia = coupons[:, k, slot_c:slot_c+1] * bond_nominal * (months_per_pmt_k / 12.0)
                    hqla += (holdings[:, k, slot_c:slot_c+1] * coupon_intermedia) / r1_current
                    
        
        next_liability = liabilities_batch[:, t + 1].view(-1, 1) if t + 1 < liabilities_batch.shape[1] else torch.zeros_like(liability_due).view(-1, 1)
        lcr = hqla / (next_liability + 1e-8)
        
        lcr_mask = (lcr < 1.0) & is_alive
        lcr_penalty = torch.where(lcr_mask, (1.0 - lcr) * RHO + float(THETA), torch.zeros_like(lcr))
        cash -= lcr_penalty
        cumulative_penalty += lcr_penalty


    # Terminal step
    last_betas = yields_batch[:, T]
    liability_t = liabilities_batch[:, T].view(-1, 1)

    terminal_value = cash - liability_t
    for k, M in enumerate(bond_maturities):
        coupon_dates_k = bond_coupon_dates[k]
        coupons_per_year_k = len(coupon_dates_k) * 12.0 / M if M > 0 else 0
        terminal_value = terminal_value + holdings[:, k, 0:1] * bond_nominal
        for c in coupon_dates_k:
            slot_c = M - c
            if 0 <= slot_c < max_M:
                terminal_value = terminal_value + holdings[:, k, slot_c:slot_c+1] * (bond_nominal * coupons[:, k, slot_c:slot_c+1] / coupons_per_year_k)

    new_holdings, new_coupons = torch.zeros_like(holdings), torch.zeros_like(coupons)
    new_holdings[:, :, :-1], new_coupons[:, :, :-1] = holdings[:, :, 1:], coupons[:, :, 1:]
    holdings, coupons = new_holdings, new_coupons

    for k, M in enumerate(bond_maturities):
        coupon_dates_k = bond_coupon_dates[k]
        coupons_per_year_k = len(coupon_dates_k) * 12.0 / M if M > 0 else 0
        for m in range(max_M):
            units = holdings[:, k, m]
            tau_face, issuance_yield = m + 1, coupons[:, k, m]
            y_face = get_nelson_siegel_yield_batched(float(tau_face) / 12.0, last_betas)
            terminal_value = terminal_value + (units * bond_nominal).unsqueeze(-1) * (1.0 / (1.0 + y_face * (tau_face / 12.0)))
            for c in coupon_dates_k:
                if c > M - m - 1:
                    tau_c = c - (M - m - 1)
                    y_c = get_nelson_siegel_yield_batched(float(tau_c) / 12.0, last_betas)
                    terminal_value = terminal_value + (units * bond_nominal * issuance_yield / coupons_per_year_k).unsqueeze(-1) * (1.0 / (1.0 + y_c * (tau_c / 12.0)))

    pv_liabs_term = torch.zeros(batch_size, 1, device=device)
    L_total = liabilities_batch.shape[1]
    for j in range(max_M - 1):
        idx = T + 1 + j
        if idx >= L_total:
            break
        tau_l = (j + 1) / 12.0
        y_l = get_nelson_siegel_yield_batched(tau_l, last_betas)
        pv_liabs_term += liabilities_batch[:, idx].view(-1, 1) / (1.0 + y_l * tau_l)

    terminal_value = terminal_value - pv_liabs_term
    
    # Single terminal insolvency penalty 
    terminal_insolvent = (terminal_value < 0) & is_alive
    term_insolvency_penalty = torch.where(terminal_insolvent, torch.tensor(float(F), device=device), torch.tensor(0.0, device=device))
    terminal_value = terminal_value - term_insolvency_penalty
    cumulative_penalty += term_insolvency_penalty

    # Preserve downstream reporting
    ever_bankrupt = ever_bankrupt | terminal_insolvent.squeeze(-1)

    # Dead paths take their frozen mark-to-market NAV from the default step.
    final_nav = torch.where(~is_alive, early_terminal_nav, terminal_value)

    return final_nav.squeeze(-1), cumulative_penalty.squeeze(-1), history_log, ever_bankrupt

# ==========================================
# TRAINING AND EVALUATION HELPERS
# ==========================================

def train_step(model, optimizer, loss_fn, yields_train, liabilities_train, T, bond_maturities, bond_coupon_dates):
    model.train()
    optimizer.zero_grad()
    
    terminal_value, _, _, _ = batched_differentiable_rollout(
        model, yields_train, liabilities_train, T, bond_maturities, bond_coupon_dates
    )
    
    loss = loss_fn(terminal_value)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def eval_step(model, gamma, yields_val, liabilities_val, T, bond_maturities, bond_coupon_dates):
    model.eval()
    
    terminal_value, _, _, _ = batched_differentiable_rollout(
        model, yields_val, liabilities_val, T, bond_maturities, bond_coupon_dates
    )
    
    return entropic_risk_measure_eval(terminal_value, gamma=gamma)

# ==========================================
# DIAGNOSTICS & BASELINES
# ==========================================

class ForcedActionWrapper(torch.nn.Module):
    def __init__(self, original_model, forced_action_vector, t_force=None):
        super().__init__()
        self.original_model = original_model
        self.forced_action_vector = forced_action_vector
        self.t_force = t_force 

    def forward(self, t, state, prev_action):
        batch_size = state.shape[0]
        return self.forced_action_vector.repeat(batch_size, 1).to(state.device)

class HeuristicPolicyLogger(nn.Module):
    def __init__(self, action_dim, fixed_weights, log_scenario_idx=0):
        super().__init__()
        self.action_dim = action_dim
        self.fixed_weights = torch.tensor(fixed_weights, dtype=torch.float32)
        self.log_scenario_idx = log_scenario_idx

    def forward(self, t, state, prev_action):
        batch_size = state.shape[0]
        device = state.device
        
        if self.log_scenario_idx is not None and self.log_scenario_idx < batch_size:
            cash_approx = state[self.log_scenario_idx, 0].item() * 1000.0 
            liab_approx = state[self.log_scenario_idx, -1].item() * 1000.0
            print(f"  [Step t={t}] | Initial Cash: ${cash_approx:.2f} | Liability Due: ${liab_approx:.2f}")
            print(f"             | Action Ordered (Weights): {self.fixed_weights.tolist()}")
            
        return self.fixed_weights.repeat(batch_size, 1).to(device)

def plot_true_oce_q_function(trained_model, trained_loss_fn, dataloader, action_dim, T, bond_maturities, bond_coupon_dates, bond_idx=0, avail_cash=None):
    print(f"Estimating TRUE optimized metric (OCE Utility) for Bond {bond_idx}...")
    action_grid = np.linspace(0.0, 1.0, 30) 
    oce_utilities = []
    trained_model.eval()
    
    with torch.no_grad():
        for alloc in action_grid:
            forced_weights = torch.zeros(action_dim, dtype=torch.float32)
            forced_weights[bond_idx] = alloc
            forced_weights[-1] = 1.0 - alloc  
            
            wrapped_model = ForcedActionWrapper(trained_model, forced_weights, t_force=0)
            all_terminal_values = []
            
            for batch_yields, batch_liabs in dataloader:
                terminal_assets, penalties, _, _ = batched_differentiable_rollout(
                    model=wrapped_model, yields_batch=batch_yields, liabilities_batch=batch_liabs, 
                    T=T, bond_maturities=bond_maturities, bond_coupon_dates=bond_coupon_dates
                )
                all_terminal_values.append(terminal_assets)
                
            full_terminal_values = torch.cat(all_terminal_values, dim=0)
            true_oce = -trained_loss_fn(full_terminal_values).item()
            oce_utilities.append(true_oce)
            
    plt.figure(figsize=(9, 6))
    cash_for_plot = avail_cash if avail_cash is not None else 1000.0
    x_axis_euros = action_grid * cash_for_plot

    plt.plot(x_axis_euros, oce_utilities, color='purple', marker='o', markersize=5, linewidth=2, label='True OCE Utility')
    best_idx = np.argmax(oce_utilities)
    
    plt.scatter([x_axis_euros[best_idx]], [oce_utilities[best_idx]], color='red', s=100, zorder=5, 
                label=f'Optimal Action (X2 = {x_axis_euros[best_idx]:.2f}€)')
    plt.title(f'Deep ALM Internal Objective (OCE Utility)\nat $t=0$ (Investable Cash ≈ ${cash_for_plot:.1f})', fontsize=14)
    plt.xlabel(r'Action: Dollars invested in Bonds ($X_2$)', fontsize=12)
    plt.ylabel(r'Optimized Certainty Equivalent (OCE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

def evaluate_naive_baseline(loss_fn, dataloader, action_dim, T, bond_maturities, bond_coupon_dates, initial_cash, naive_weights, gamma):
    print("=" * 60)
    print(f"NAIVE BASELINE EVALUATION: {naive_weights}")
    print("=" * 60)
    
    baseline_model = HeuristicPolicyLogger(action_dim=action_dim, fixed_weights=naive_weights, log_scenario_idx=0)
    all_terminal_values = []
    
    with torch.no_grad():
        for i, (batch_yields, batch_liabs) in enumerate(dataloader):
            if i > 0:
                baseline_model.log_scenario_idx = None
                
            terminal_assets, penalties, _, _ = batched_differentiable_rollout(
                model=baseline_model, yields_batch=batch_yields, liabilities_batch=batch_liabs, 
                T=T, bond_maturities=bond_maturities, bond_coupon_dates=bond_coupon_dates, initial_cash=initial_cash
            )
            all_terminal_values.append(terminal_assets)
            
    full_terminal_values = torch.cat(all_terminal_values, dim=0)
    naive_oce = -loss_fn(full_terminal_values).item()
    naive_entropic_risk = entropic_risk_measure_eval(full_terminal_values, gamma)
    
    print("-" * 60)
    print(f"FINAL TEST SET RESULTS:")
    print(f"-> OCE Utility OOS:      {naive_oce:.4f}")
    print(f"-> Entropic Risk OOS:   ${naive_entropic_risk:.2f}")
    print("=" * 60)
    
    return naive_oce, naive_entropic_risk

def analyze_trained_allocations_clean(model, dataloader, T, bond_maturities, bond_coupon_dates, initial_cash=markov_config["W0"], scenario_idx=0):
    print("=" * 95)
    print(f" ACCURATE SIMULATION ENGINE ANALYSIS (SCENARIO {scenario_idx})")
    print("=" * 95)
    model.eval()
    
    batch_yields, batch_liabs = next(iter(dataloader))
    
    _, _, history, _ = batched_differentiable_rollout(
        model=model, yields_batch=batch_yields, liabilities_batch=batch_liabs, 
        T=T, bond_maturities=bond_maturities, bond_coupon_dates=bond_coupon_dates, 
        initial_cash=initial_cash, log_scenario_idx=scenario_idx
    )
    
    for step in history:
        t, cash, investable, liab, weights = step["t"], step["real_cash"], step["investable_budget"], step["real_liability"], step["network_weights"]
        monetary_invested = weights * investable
        allocs_pct_str = ", ".join([f"{a*100:6.1f}%" for a in weights])
        allocs_val_str = ", ".join([f"${v:7.2f}" for v in monetary_invested])
        
        print(f"  [Month t={t:02d}] | Balance Cash: ${cash:8.2f} | Expected Liability: ${liab:8.2f}")
        if investable == 0.0 and cash < 0:
            print(f"               | [BANKRUPTCY] Budget frozen at $0.00! No purchases allowed.")
        else:
            print(f"               | Investable Budget: ${investable:8.2f}")
        print(f"               | Network Choice (%) : [{allocs_pct_str}] (Bonds -> Cash)")
        print(f"               | Actual Invest. ($) : [{allocs_val_str}]")
        print(f"               -------------------------------------------------------------------------")
    print("=" * 95)


@torch.no_grad()
def evaluate_deep_alm_dh(model, markov_config, eval_episodes=1000, device="cpu", seed_offset=1000000, log_scenario_idx=0, gamma=1.0, alpha=0.05):
    """
    Evaluates the DeepALM model, tracking standard metrics, tail risk (VaR/CVaR), and bankruptcy events.
    """
    model.eval()
    T = markov_config["T"]
    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    bond_coupon_dates = [Bond(**cfg).coupon_dates for cfg in markov_config["bond_configs"]]
    W0 = markov_config.get("W0", 1000.0)
    
    safe_config = {k: markov_config[k] for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"] if k in markov_config}

    y_list, l_list = [], []
    for i in range(eval_episodes):
        seed = seed_offset + i
        y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=seed, pure_grid=True, **safe_config)
        l_path = DepositBetaLiabilityGenerator.generate(y_path, noise_std=0.0, seed=seed)
        # Zero out liabilities beyond horizon T
        l_path[T + 1:] = 0.0
        l_list.append(l_path)
        y_list.append(y_path)

    yields = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    liabs = torch.tensor(np.stack(l_list), dtype=torch.float32, device=device)

    terminal_value_tensor, cumulative_penalty, history_log, ever_bankrupt = batched_differentiable_rollout(
        model=model, yields_batch=yields, liabilities_batch=liabs, T=T, 
        bond_maturities=bond_maturities, bond_coupon_dates=bond_coupon_dates, 
        initial_cash=W0, log_scenario_idx=log_scenario_idx
    )

    terminal_values = terminal_value_tensor.detach().cpu().numpy()
    entropic_risk = entropic_risk_measure_eval(terminal_value_tensor, gamma=gamma)
    
    ever_bankrupt_1d = ever_bankrupt if ever_bankrupt.ndim == 1 else ever_bankrupt.any(dim=1)
    bankrupt_indices = torch.nonzero(ever_bankrupt_1d).squeeze(-1).cpu().numpy().tolist()

    var_threshold = float(np.percentile(terminal_values, alpha * 100))
    tail_values = terminal_values[terminal_values <= var_threshold]
    cvar = float(np.mean(tail_values)) if len(tail_values) > 0 else var_threshold
    conf_level = int((1.0 - alpha) * 100)

    metrics = {
        "mean_nav": float(np.mean(terminal_values)),
        "median_nav": float(np.median(terminal_values)),
        "std_nav": float(np.std(terminal_values)),
        "min_nav": float(np.min(terminal_values)),
        "max_nav": float(np.max(terminal_values)),
        "entropic_risk": entropic_risk,
        "default_rate": ever_bankrupt_1d.float().mean().item(),
        f"var_{conf_level}": var_threshold,
        f"cvar_{conf_level}": cvar,
        "bankrupt_indices": bankrupt_indices 
    }
    
    print("\n=== DeepALM DH Evaluation Results ===")
    print(f"Mean NAV:       {metrics['mean_nav']:.2f} ± {metrics['std_nav']:.2f}")
    print(f"Median NAV:     {metrics['median_nav']:.2f}")
    print(f"VaR ({conf_level}%):      {var_threshold:.2f} (Worst {alpha*100:.0f}% cutoff)")
    print(f"CVaR ({conf_level}%):     {cvar:.2f} (Average of worst {alpha*100:.0f}%)")
    print(f"Min / Max NAV:  {metrics['min_nav']:.2f} / {metrics['max_nav']:.2f}")
    print(f"Entropic Risk:  {metrics['entropic_risk']:.4f}")
    print(f"Default Rate:   {metrics['default_rate'] * 100:.2f}%")
    print(f"Total Defaults: {len(bankrupt_indices)} out of {eval_episodes} episodes")

    return metrics, history_log, terminal_values