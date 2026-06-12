import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from environment.config import GAMMA, LAMBDA

# ==========================================
# PyTorch / DH Approach Functions
# ==========================================

def get_nelson_siegel_yield_batched(tau_years: float, betas: torch.Tensor, lmbda: float = LAMBDA) -> torch.Tensor:
    """Batched Nelson-Siegel annual yield."""
    if tau_years == 0.0:
        return betas[:, 0:1] + betas[:, 1:2]

    x = tau_years / lmbda
    x_tensor = torch.tensor(x, dtype=betas.dtype, device=betas.device)

    exp_term = torch.exp(-x_tensor)
    f1 = (1.0 - exp_term) / x_tensor
    f2 = f1 - exp_term

    return betas[:, 0:1] + betas[:, 1:2] * f1 + betas[:, 2:3] * f2

def bond_return_factor(tau_months: float, betas: torch.Tensor, lmbda: float = LAMBDA) -> torch.Tensor:
    """Gross return for a ZERO-COUPON bond with maturity tau_months using simple interest."""
    y = get_nelson_siegel_yield_batched(tau_months / 12.0, betas, lmbda)
    return 1.0 + y * (tau_months / 12.0)

def build_state(shadow_cash, future_cash_tensor, betas_t, liability, CASH_SCALE=1000.0, BETA_SCALE=10.0):
    """Constructs the normalized state tensor for the neural network."""
    return torch.cat([
        shadow_cash / CASH_SCALE,
        future_cash_tensor / CASH_SCALE,
        betas_t * BETA_SCALE,
        liability / CASH_SCALE,
    ], dim=1)

def calculate_terminal_pv_assets(cash, holdings, coupons, bond_maturities, bond_coupon_dates, yields_t, yield_fn, bond_nominal=100.0):
    """Calculates the Mark-to-Market PV of the portfolio's assets using strict simple interest."""
    max_M = holdings.shape[2]
    pv_assets = cash.clone()
    
    for k, M in enumerate(bond_maturities):
        coupon_dates_k = bond_coupon_dates[k]
        n_coupons_k = len(coupon_dates_k)
        coupons_per_year_k = n_coupons_k * 12.0 / M if M > 0 else 0
        
        for m in range(max_M):
            units = holdings[:, k, m]
            
            if units.abs().sum() == 0:
                continue
                
            tau_face = m + 1
            issuance_yield = coupons[:, k, m]
            y_face = yield_fn(float(tau_face) / 12.0, yields_t)
            df_face = 1.0 / (1.0 + y_face * (tau_face / 12.0)) 
            
            pv_assets += (units * bond_nominal).view(-1, 1) * df_face
            
            for c in coupon_dates_k:
                if c > M - m - 1:
                    tau_c = c - (M - m - 1)
                    coupon_cash_per_unit = bond_nominal * issuance_yield / coupons_per_year_k
                    y_c = yield_fn(float(tau_c) / 12.0, yields_t)
                    df_c = 1.0 / (1.0 + y_c * (tau_c / 12.0))
                    
                    pv_assets += (units * coupon_cash_per_unit).view(-1, 1) * df_c
                    
    return pv_assets

# ==========================================
# NumPy / Gym Environment Functions
# ==========================================

def get_nelson_siegel_yield(tau_years, beta, lmbda=LAMBDA):
    """Vectorized Nelson-Siegel annual yield."""
    tau_years = np.asarray(tau_years, dtype=float)
    x = tau_years / lmbda
    small = np.abs(x) < 1e-10

    f1 = np.empty_like(x)
    f2 = np.empty_like(x)

    f1[small] = 1.0
    f2[small] = 0.0

    exp_term = np.exp(-x[~small])
    f1[~small] = (1.0 - exp_term) / x[~small]
    f2[~small] = f1[~small] - exp_term

    return beta[0] + beta[1] * f1 + beta[2] * f2

def get_discount_factor(tau_years, beta, lmbda=LAMBDA):
    """Discount factor using simple interest compounding."""
    y_tau = get_nelson_siegel_yield(tau_years, beta, lmbda)
    return 1.0 / (1.0 + y_tau * tau_years)

def calculate_pv(cash, future_cash_flows, yields):
    """Unified PV for Assets or Liabilities."""
    pv = float(cash)
    for i, cf in enumerate(future_cash_flows):
        tau = i + 1  
        df = get_discount_factor(tau/12.0, yields)
        pv += cf * df
    return pv

# ==========================================
# Utility Functions
# ==========================================

def isoelastic_utility(nav, gamma=1.0):
    """Objective function u(x) — works with scalars and numpy arrays."""
    x = np.maximum(1e-6, nav)   
    if gamma == 1.0:
        return np.log(x)
    return (x ** (1 - gamma) - 1) / (1 - gamma)

def cara_utility(v, gamma=GAMMA, scale=1.0):
    """Constant Absolute Risk Aversion (CARA) utility function."""
    if isinstance(v, torch.Tensor):
        return -torch.exp(-gamma * v / scale)
    return -np.exp(-gamma * v / scale)

def identity_utility(v):
    """Identity utility function (risk-neutral)."""
    return v

# ==========================================
# Display & Formatting Functions
# ==========================================

def print_holdings_matrix(holdings_matrix, bonds, time_buckets):
    """Prints a formatted matrix of holdings per bond type and time bucket."""
    w_name = 15
    w_col = 10
    precision = ".2f"

    header_cols = "".join([f"{bucket:>{w_col}}" for bucket in time_buckets])
    header = f"  | {'Bond Type':<{w_name}} |{header_cols} | {'Total':>{w_col}} |"
    divider = "  " + "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    for i, bond in enumerate(bonds):
        row_data = holdings_matrix[i]
        data_str = "".join([f"{val:>{w_col}{precision}}" for val in row_data])
        row_total = np.sum(row_data)
        print(f"  | {bond.bond_type:<{w_name}} |{data_str} | {row_total:>{w_col}{precision}} |")

    print(divider)
    
    col_totals = np.sum(holdings_matrix, axis=0)
    total_assets = np.sum(col_totals)
    footer_str = "".join([f"{val:>{w_col}{precision}}" for val in col_totals])
    
    print(f"  | {'TOTALS':<{w_name}} |{footer_str} | {total_assets:>{w_col}{precision}} |")
    print(divider)
    
def print_bond_universe(bonds):
    """Prints a formatted table of the bond universe."""
    w_name = 20
    w_mat = 20

    header = f"| {'Bond Type':<{w_name}} | {'Maturity (Months)':<{w_mat}} |"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    for bond in bonds:
        print(f"| {bond.bond_type:<{w_name}} | {bond.M:<{w_mat}} |")

    print(divider)

# ==========================================
# Plotting Functions
# ==========================================

def plot_learning_curve(log_folder, title="Learning Curve"):
    """Reads the monitor.csv file and plots the smoothed learning curve."""
    x, y = results_plotter.ts2xy(
        results_plotter.load_results(log_folder), 'timesteps'
    )

    window_size = 50
    if len(y) < window_size:
        window_size = 1  

    y_smoothed = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    x_smoothed = x[window_size - 1:]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(x_smoothed, y_smoothed, color='blue', linewidth=2, label=f'Moving Avg ({window_size} eps)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_final_wealth(log_folder, title="DeepALM Final Wealth Curve"):
    """Reads the monitor.csv file and plots the smoothed final wealth curve."""
    df = results_plotter.load_results(log_folder)
    
    if 'nav' not in df.columns:
        print("Error: 'nav' column not found! Ensure Monitor info_keywords contains 'nav'.")
        return

    x = np.cumsum(df['l'].values)  
    y = df['nav'].values

    window_size = 50
    if len(y) < window_size:
        window_size = 1  

    y_smoothed = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    x_smoothed = x[window_size - 1:]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, alpha=0.3, color='gray', label='Raw Final Wealth')
    plt.plot(x_smoothed, y_smoothed, color='green', linewidth=2, label=f'Moving Avg ({window_size} eps)')
    plt.xlabel('Timesteps')
    plt.ylabel('Terminal Wealth (NAV)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

