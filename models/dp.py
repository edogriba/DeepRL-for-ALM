import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from environment.scenario import MarkovYieldCurveGenerator, DepositBetaLiabilityGenerator

# IMPORT CONFIGURATION
from environment.config import GAMMA, RHO, F, THETA, markov_config 
from models.utils import identity_utility, cara_utility

# SETUP PLOTTING & SYSTEM ARGS
show_figures = True
plot_type = "alm_base"

if len(sys.argv) > 1:
    plot_type = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == "donotshowfigs":
        show_figures = False

if not os.path.exists("fig"):
    os.makedirs("fig")

if not os.path.exists("results"):
    os.makedirs("results")
    
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({'font.size': 10}) 

# ENVIRONMENT CONTAINER & FACTORY
class Container:
    pass

def make_env_alm(cfg):
    """
    Builds the environment using the imported markov_config dictionary.
    """
    env = Container()
    
    env.T = cfg["T"]
    env.N_W = cfg["N_W"]
    env.N_B1 = cfg["N_B1"]
    env.W_GRID = cfg["W_GRID"]
    env.B1_GRID = cfg["B1_GRID"]
    
    env.N0, env.N1 = cfg["N0"], cfg["N1"]
    env.N_Y = env.N0 * env.N1
    env.state_map = [(i0, i1) for i0 in range(env.N0) for i1 in range(env.N1)]
    
    # Transition matrix assuming Beta0 and Beta1 are independent
    env.trans_prob = np.zeros((env.N_Y, env.N_Y))
    for s_idx, (i0, i1) in enumerate(env.state_map):
        for s_next_idx, (j0, j1) in enumerate(env.state_map):
            env.trans_prob[s_idx, s_next_idx] = cfg["P0"][i0, j0] * cfg["P1"][i1, j1]
            
    env.L_ALL = cfg["L_ALL"]
    env.R1_M = cfg["R1_M"]
    env.R2_M = cfg["R2_M"]
    
    env.w_step = env.W_GRID[1] - env.W_GRID[0]
    env.b1_step = env.B1_GRID[1] - env.B1_GRID[0]
    
    # Action Grid
    env.X2_GRID = np.arange(0.0, cfg["W0"] + env.w_step, env.w_step)
    env.N_X2 = len(env.X2_GRID)
    
    # Initial Indices
    env.t0  = 0
    env.si0 = int(np.argmin(np.abs(env.W_GRID  - cfg["W0"])))
    env.bi0 = int(np.argmin(np.abs(env.B1_GRID - cfg["B1_0"])))
    env.yi0 = env.state_map.index((cfg["i0_init"], cfg["i1_init"]))
    return env

# CORE DYNAMIC PROGRAMMING ENGINE WITH CORNER BOUNDS
def get_values_with_bounds(env):
    T, N_W, N_B1, N_Y, N_X2 = env.T, env.N_W, env.N_B1, env.N_Y, env.N_X2
    
    # Q_plot tensors are kept float32 only for memory efficiency at save time
    Q_plot_lo = np.full((T, N_W, N_B1, N_Y, N_X2), np.nan, dtype=np.float64)
    Q_plot_hi = np.full((T, N_W, N_B1, N_Y, N_X2), np.nan, dtype=np.float64)

    # Value function bounds: float64 for numerical integrity
    V_lo = np.full((T + 1, N_W, N_B1, N_Y), -1e6, dtype=np.float64)
    V_hi = np.full((T + 1, N_W, N_B1, N_Y), -1e6, dtype=np.float64)

    # Policy: float32 is fine (stores grid-snapped euro amounts)
    pi = np.full((T, N_W, N_B1, N_Y), np.nan, dtype=np.float64)
    
    print(f"Initializing Terminal Value and Bounds at T = {T}...")
    
    # 1. TERMINAL TIME INITIALIZATION
    for s_idx, (i0, i1) in enumerate(env.state_map):
        MTM_B1 = env.B1_GRID[None, :] / env.R1_M[i0, i1]
        nav_terminal = env.W_GRID[:, None] + MTM_B1 - env.L_ALL[i0, i1]
        
        # Applica F in formato vettoriale per la griglia
        nav_terminal = np.where(nav_terminal < 0, nav_terminal - F, nav_terminal)
        
        exact_utility = cara_utility(nav_terminal)
        V_lo[T, :, :, s_idx] = exact_utility
        V_hi[T, :, :, s_idx] = exact_utility

    W_tensor  = env.W_GRID[:, None, None]      
    B1_tensor = env.B1_GRID[None, :, None]     
    X2_tensor = env.X2_GRID[None, None, :]     
        
    for t in range(T - 1, -1, -1):
        print(f"Solving t = {t} (Computing Corner Bounds)...")

        for s_idx, (i0, i1) in enumerate(env.state_map):
            current_L = env.L_ALL[i0, i1]
            avail_cash = W_tensor - current_L
            X1_tensor  = avail_cash - X2_tensor

            r1     = env.R1_M[i0, i1]
            # Apply fixed fine F when in the illiquid regime (avail_cash < 0)
            fixed_fine_penalty = np.where(avail_cash < 0, -F, 0.0)
            W_next_temp = (X1_tensor * r1) + B1_tensor

            Q_lo_acc = np.zeros((N_W, N_B1, N_X2), dtype=np.float64)
            Q_hi_acc = np.zeros((N_W, N_B1, N_X2), dtype=np.float64)

            for s_next_idx, (i0p, i1p) in enumerate(env.state_map):
                prob = env.trans_prob[s_idx, s_next_idx]
                if prob <= 0:
                    continue

                L_next  = env.L_ALL[i0p, i1p]

                hqla_mtm    = W_next_temp / r1
                lcr         = hqla_mtm / (L_next + 1e-8)
                lcr_penalty = np.where(lcr < 1.0, -(RHO * (1.0 - lcr) + THETA), 0.0)

                W_next_s = W_next_temp + lcr_penalty + fixed_fine_penalty
                B1_next  = X2_tensor * env.R2_M[i0, i1]

                idx_w_lo  = np.clip(np.floor((W_next_s  - env.W_GRID[0])  / env.w_step),  0, N_W-1).astype(np.int32)
                idx_w_hi  = np.clip(np.ceil( (W_next_s  - env.W_GRID[0])  / env.w_step),  0, N_W-1).astype(np.int32)
                idx_b1_lo = np.clip(np.floor((B1_next   - env.B1_GRID[0]) / env.b1_step), 0, N_B1-1).astype(np.int32)
                idx_b1_hi = np.clip(np.ceil( (B1_next   - env.B1_GRID[0]) / env.b1_step), 0, N_B1-1).astype(np.int32)

                Q_lo_acc += prob * V_lo[t + 1, idx_w_lo, idx_b1_lo, s_next_idx]
                Q_hi_acc += prob * V_hi[t + 1, idx_w_hi, idx_b1_hi, s_next_idx]

            Q_lo = Q_lo_acc
            Q_hi = Q_hi_acc

            # Regime Solvente: cassa positiva, si può investire X2 fino all'esaurimento della cassa (X1 >= 0)
            mask_solvent = np.broadcast_to(avail_cash >= 0, Q_lo.shape)
            mask_x1_valid = np.broadcast_to(X1_tensor >= 0, Q_lo.shape)
            valid_solvent_actions = mask_solvent & mask_x1_valid

            # Regime Illiquido: deficit, l'unica azione permessa è X2 = 0
            mask_illiquid = np.broadcast_to(avail_cash < 0, Q_lo.shape)
            mask_x2_zero = np.broadcast_to(X2_tensor == 0.0, Q_lo.shape)
            valid_illiquid_actions = mask_illiquid & mask_x2_zero

            # Combinazione delle azioni valide (Equazione 3.18 del PDF)
            valid_action_mask = valid_solvent_actions | valid_illiquid_actions

            # Invalidiamo le azioni non permesse (le penalità implicite gestiranno i punteggi dei deficit)
            Q_lo = np.where(valid_action_mask, Q_lo, -np.inf)
            Q_hi = np.where(valid_action_mask, Q_hi, -np.inf)

            Q_plot_lo[t, :, :, s_idx, :] = Q_lo.astype(np.float64)
            Q_plot_hi[t, :, :, s_idx, :] = Q_hi.astype(np.float64)

            # Maximin policy: maximise the lower bound
            best_action_idx = np.argmax(Q_lo, axis=-1)

            V_lo[t, :, :, s_idx] = np.max(Q_lo, axis=-1)
            V_hi[t, :, :, s_idx] = np.max(Q_hi, axis=-1)
            V_hi[t, :, :, s_idx] = np.maximum(V_hi[t, :, :, s_idx],
                                               V_lo[t, :, :, s_idx])

            # Salvataggio Policy coerente con la nuova maschera
            valid_policy_mask = np.any(valid_action_mask, axis=-1)
            pi[t, :, :, s_idx] = np.where(valid_policy_mask,
                                           env.X2_GRID[best_action_idx], np.nan)

    return Q_plot_lo, Q_plot_hi, V_lo, V_hi, pi


# EXECUTION & PLOTTING PIPELINE
if plot_type == "alm_base":
    print(f"\n--- Starting ALM Dynamic Programming with Corner Bounds ---")
    
    # Environment Initialization
    env = make_env_alm(markov_config)
    
    # Run Dynamic Programming with Corner Bounds
    Q_lo, Q_hi, V_lo, V_hi, pi = get_values_with_bounds(env)
    
    #print("\nNaN/Inf Concentration (Bankruptcy/Invalid States) per t:")              
    #for t in range(env.T):
    #    invalid_c = np.sum((V_lo[t] < 0)) / V_lo[t].size
    #    print(f"t={t}, invalid_states={invalid_c:.4f}")

    # Generate Q-Function Plots (using Robust Lower Bound) for the initial state
    w_idx, b1_idx, y_idx = env.si0, env.bi0, env.yi0
    i0, i1 = env.state_map[y_idx]

    w_val = env.W_GRID[w_idx]
    b1_val = env.B1_GRID[b1_idx]
    avail_cash = w_val - env.L_ALL[i0, i1]
    
    if show_figures:
        for t in range(env.T):
            q_slice_lo = Q_lo[t, w_idx, b1_idx, y_idx, :]
            q_slice_hi = Q_hi[t, w_idx, b1_idx, y_idx, :]
            
            valid_plot_mask = ~np.isinf(q_slice_lo) & ~np.isnan(q_slice_lo) & ~np.isinf(q_slice_hi) & ~np.isnan(q_slice_hi)
            actions_x2_dollar = env.X2_GRID[valid_plot_mask]
            
            q_values_valid_lo = q_slice_lo[valid_plot_mask]
            q_values_valid_hi = q_slice_hi[valid_plot_mask]
            
            plt.figure(figsize=(8, 5))
            if len(actions_x2_dollar) > 0:
                plt.plot(actions_x2_dollar, q_values_valid_lo, label=f'Robust Q-Value $Q^{{lo}}(s,a)$ at t={t}', color='#1f77b4', linewidth=2)
                
                # Filled corridor for uncertainty
                if not np.array_equal(q_values_valid_lo, q_values_valid_hi):
                    plt.fill_between(actions_x2_dollar, q_values_valid_lo, q_values_valid_hi, color='#1f77b4', alpha=0.2, label='Corner Bound Corridor')
                
                opt_idx = np.argmax(q_values_valid_lo)
                best_action = actions_x2_dollar[opt_idx]
                plt.scatter([best_action], [q_values_valid_lo[opt_idx]], 
                            color='red', zorder=5, s=80, label=f'Optimal Action (x2 = ${best_action:.2f})')
            else:
                plt.title(f"No valid actions (Bankruptcy/Insufficient Cash) at t={t}")
                
            plt.xlabel(r'Action: Dollars invested in 2-Month Bonds ($X_2$)') 
            plt.ylabel(r'Expected Future Value Bounds')
            plt.title(f'ALM Robust Q-Function at $t={t}$\n(W=${w_val:.1f}, B1=${b1_val:.1f}, Available Cash=${avail_cash:.1f})')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"fig/alm_robust_q_function_t{t}.pdf")
            plt.close() 
            
        print("\nQ-Function plots saved in 'fig/' directory.")

    # SAVE THE SOLUTION
    dp_solution = {
        "W_GRID":       env.W_GRID,           
        "B1_GRID":      env.B1_GRID,          
        "X2_GRID":      env.X2_GRID,
        "BETA0_GRID":   markov_config["BETA0_GRID"],
        "BETA1_GRID":   markov_config["BETA1_GRID"],
        "BETA2":        markov_config["BETA2"],
        "LAMBDA":       markov_config["LAMBDA"],
        "R1_M":         env.R1_M,             
        "R2_M":         env.R2_M,             
        "L_ALL":        env.L_ALL,            
        "P0":           markov_config["P0"],     
        "P1":           markov_config["P1"],     
        "PI":           pi,                   
        "V_LO":         V_lo,          
        "V_HI":         V_hi,            
        "Q_LO":         Q_lo,        
        "Q_HI":         Q_hi,  
        "T":            env.T,
        "W0":           markov_config["W0"],
        "B1_0":         markov_config["B1_0"],                  
        "i0_init":      markov_config["i0_init"],
        "i1_init":      markov_config["i1_init"],
        "N_W":          env.N_W,
        "N_B1":         env.N_B1,
        "N_X2":         env.N_X2,
    }

    with open("saved_models/dp_ns_solution_bounded.pkl", "wb") as f:
        pickle.dump(dp_solution, f)

    print("Bounded DP solution successfully saved to saved_models/dp_ns_solution_bounded.pkl")


def betas_to_indices(betas_t, BETA0_GRID, BETA1_GRID):
    i0 = int(np.argmin(np.abs(BETA0_GRID - betas_t[0])))
    i1 = int(np.argmin(np.abs(BETA1_GRID - betas_t[1])))
    return i0, i1

def dp_policy_lookup(PI, W_grid, B1_grid, t, i0, i1, W, B1, state_map):
    # This literal NN lookup is still valid for simulation steps
    wi  = int(np.argmin(np.abs(W_grid  - W)))
    bi  = int(np.argmin(np.abs(B1_grid - B1)))
    yi  = state_map.index((i0, i1))
    val = PI[t, wi, bi, yi]
    return float(val) if np.isfinite(val) else 0.0

def run_dp_rollout(dp, y_list, l_list, T):
    PI        = dp['PI']
    W_grid    = np.atleast_1d(np.asarray(dp['W_GRID'],  dtype=float)).ravel()
    B1_grid   = np.atleast_1d(np.asarray(dp['B1_GRID'], dtype=float)).ravel()
    BETA0_GRID = np.atleast_1d(dp['BETA0_GRID'])
    BETA1_GRID = np.atleast_1d(dp['BETA1_GRID'])
    N0, N1    = len(BETA0_GRID), len(BETA1_GRID)
    state_map = [(i0, i1) for i0 in range(N0) for i1 in range(N1)]
    R1_M      = dp['R1_M']
    R2_M      = dp['R2_M']
    W0        = float(dp['W0'])
    B1_0      = float(dp['B1_0'])

    navs = []   
    scores = [] 
    
    for y_path, l_path in zip(y_list, l_list):
        W  = W0
        B1 = B1_0
        
        for t in range(T):
            i0, i1 = betas_to_indices(y_path[t], BETA0_GRID, BETA1_GRID)
            l_t = float(l_path[t, 0]) if np.ndim(l_path) == 2 else float(l_path[t])
            
            investable = W - l_t
            
            if investable >= 0:
                # Regime Solvente
                x2_raw = dp_policy_lookup(PI, W_grid, B1_grid, t, i0, i1, W, B1, state_map)
                x2 = np.clip(x2_raw, 0.0, investable)
                x1 = investable - x2
            else:
                # Regime Illiquido (Deficit)
                x2 = 0.0
                x1 = investable # W_t - L_t < 0 (Debito accumulato)
            
            r1 = R1_M[i0, i1]
            hqla_mtm = (x1 * r1 + B1) / r1
            l_next = float(l_path[t+1, 0]) if np.ndim(l_path) == 2 else float(l_path[t+1])
            lcr = hqla_mtm / (l_next + 1e-8)
            
            lcr_penalty = (RHO * (1.0 - lcr) + THETA) if lcr < 1.0 else 0.0
            fixed_fine = F if investable < 0 else 0.0
            
            # State transition directly absorbs the penalties
            W  = x1 * r1 + B1 - lcr_penalty - fixed_fine
            B1 = x2 * R2_M[i0, i1]
            
        i0_T, i1_T = betas_to_indices(y_path[T], BETA0_GRID, BETA1_GRID)
        l_T = float(l_path[T, 0]) if np.ndim(l_path) == 2 else float(l_path[T])
        
        final_nav = W + B1 / R1_M[i0_T, i1_T] - l_T
        if final_nav < 0:
            final_nav -= F
        navs.append(final_nav)

        # La terminal utility include tutto, non c'è più la flag di 'already_bankrupt'
        terminal_util = cara_utility(np.array([final_nav])).item()
        scores.append(terminal_util) 
        
    return np.array(navs), np.array(scores)


def evaluate_dp_agent(dp, y_eval, l_eval, T, seed=0, verbose=False):
    if not isinstance(dp, dict):
        dp = vars(dp) 
        
    if y_eval is None or l_eval is None:
        safe_config = {k: markov_config[k] for k in
                        ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID",
                        "BETA2", "P0", "P1"] if k in markov_config}
        y_eval = MarkovYieldCurveGenerator.generate(T * 2, seed=seed, pure_grid=True, **safe_config)
        l_eval = DepositBetaLiabilityGenerator.generate(y_eval, noise_std=0.0, seed=seed)
        
    PI        = dp['PI']
    W_grid    = np.atleast_1d(np.asarray(dp['W_GRID'],  dtype=float)).ravel()
    B1_grid   = np.atleast_1d(np.asarray(dp['B1_GRID'], dtype=float)).ravel()
    beta0_grid = np.atleast_1d(dp['BETA0_GRID'])
    beta1_grid = np.atleast_1d(dp['BETA1_GRID'])
    N0, N1    = len(beta0_grid), len(beta1_grid)
    state_map = [(i0, i1) for i0 in range(N0) for i1 in range(N1)]
    R1_M      = dp['R1_M']
    R2_M      = dp['R2_M']
    W0        = float(dp['W0'])
    B1_0      = float(dp['B1_0'])

    if verbose:
        print(f"\n{'='*75}")
        print(f" ROBUST DP AGENT EXECUTION TRACE ".center(75, "="))
        print(f"{'='*75}")
        print(f"{'Month':<7} | {'Start Cash (€)':<16} | {'Liability (€)':<15} | {'Actions: 1M / 2M / Cash (€)'}")
        print("-" * 75)

    actions_history, wealth_history = [], []
    W  = W0
    B1 = B1_0

    for t in range(T):
        i0, i1 = betas_to_indices(y_eval[t], beta0_grid, beta1_grid)
        l_t    = float(l_eval[t, 0]) if np.ndim(l_eval) == 2 else float(l_eval[t])
        
        investable = W - l_t
        wealth_history.append(investable)
        
        if investable >= 0:
            # Regime Solvente
            x2_raw = dp_policy_lookup(PI, W_grid, B1_grid, t, i0, i1, W, B1, state_map)
            x2 = np.clip(x2_raw, 0.0, investable)
            x1 = investable - x2
        else:
            # Regime Illiquido
            x2 = 0.0
            x1 = investable # Debito
            if verbose:
                print(f" [!] t={t:<3} | ATTENZIONE: Deficit di {investable:.2f} € -> LCR Penalty esploderà.")
        
        actions_history.append(np.array([x1, x2, 0.0]))
        
        r1 = R1_M[i0, i1]
        hqla_mtm = (x1 * r1 + B1) / r1
        l_next = float(l_eval[t+1, 0]) if np.ndim(l_eval) == 2 else float(l_eval[t+1])
        
        lcr = hqla_mtm / (l_next + 1e-8)
        lcr_penalty = (RHO * (1.0 - lcr) + THETA) if lcr < 1.0 else 0.0
        fixed_fine = F if investable < 0 else 0.0
        
        if verbose:
             print(f" {t:<5} | {W:<16.2f} | {l_t:<15.2f} | [{x1:.2f}, {x2:.2f}, 0.00]")
        
        # State transition directly absorbs the penalty
        W  = x1 * r1 + B1 - lcr_penalty - fixed_fine
        B1 = x2 * R2_M[i0, i1]

    i0_T, i1_T = betas_to_indices(y_eval[T], beta0_grid, beta1_grid)
    l_T = float(l_eval[T, 0]) if np.ndim(l_eval) == 2 else float(l_eval[T])
    
    final_nav_euros = W + B1 / R1_M[i0_T, i1_T] - l_T
    
    if final_nav_euros < 0:
        final_nav_euros -= F

    terminal_utility = cara_utility(np.array([final_nav_euros])).item()
    final_agent_score = terminal_utility 

    if verbose:
        print("-" * 75)
        print(f"TERMINAL NAV: {final_nav_euros:.2f} €")
        print(f"Final Agent Score (Utility): {final_agent_score:.4f}")
        print(f"{'='*75}\n")
        
    return np.array(actions_history), np.array(wealth_history), float(final_nav_euros), float(final_agent_score)