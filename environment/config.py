import numpy as np

VERSION = "2Y"

LAMBDA = 0.1

BETA0_GRID = np.array([0.05, 0.045, 0.04, 0.035, 0.03])
BETA1_GRID = np.array([-0.03, -0.035, -0.04])

N0, N1 = len(BETA0_GRID), len(BETA1_GRID)
BETA2 = 0.0

P0_MATRIX = np.array([
    [0.50, 0.15, 0.20, 0.10, 0.05],
    [0.10, 0.50, 0.25, 0.10, 0.05],
    [0.15, 0.15, 0.40, 0.15, 0.15],
    [0.05, 0.10, 0.25, 0.50, 0.10],
    [0.05, 0.10, 0.20, 0.15, 0.50],
])


P1_MATRIX = np.array([
    [0.60, 0.35, 0.05],
    [0.40, 0.40, 0.20],
    [0.35, 0.55, 0.10],
])

I0_INIT = 2  
I1_INIT = 0  

bond_configs = [
    {'bond_type': '1M_Bill', 'nominal_value': 100,
     'maturity_months': 1, 'coupon_dates': [1]},
    {'bond_type': '3M_ZeroCoupon', 'nominal_value': 100,
     'maturity_months': 3, 'coupon_dates': [3]},
    {'bond_type': '6M_ZeroCoupon', 'nominal_value': 100,
     'maturity_months': 6, 'coupon_dates': [6]},
    {'bond_type': '12M_6Months', 'nominal_value': 100,
     'maturity_months': 12, 'coupon_dates': [6, 12]},
    {'bond_type': '24M_6Months', 'nominal_value': 100,
     'maturity_months': 24, 'coupon_dates': [6, 12, 18, 24]}
] if VERSION == "2Y" else [
    {'bond_type': '1M_Bill', 'nominal_value': 100,
     'maturity_months': 1, 'coupon_dates': [1]},
    {'bond_type': '3M_ZeroCoupon', 'nominal_value': 100,
     'maturity_months': 3, 'coupon_dates': [3]},
    {'bond_type': '6M_ZeroCoupon', 'nominal_value': 100,
     'maturity_months': 6, 'coupon_dates': [6]},
    {'bond_type': '12M_6Months', 'nominal_value': 100,
     'maturity_months': 12, 'coupon_dates': [6, 12]},
    {'bond_type': '60M_6Months', 'nominal_value': 100,
     'maturity_months': 60, 'coupon_dates': [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]},
    {'bond_type': '120M_6Months', 'nominal_value': 100,
     'maturity_months': 120, 'coupon_dates': [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 
                                              66, 72, 78, 84, 90, 96, 102, 108, 
                                              114,120]},
]

GAMMA = 0.001 if VERSION == "2Y" else 0.0001
T = 24 if VERSION == "2Y" else 120
INITIAL_CASH = 2000.0 if VERSION == "2Y" else 10_000.0

B1_0 = 0.0


RHO =0.0# 0.250 * INITIAL_CASH

THETA = 0.250 * INITIAL_CASH#0.0

F = 0.10 * INITIAL_CASH # 0.0

N_W, N_B1, N_X2 = 500, 500, 500
W_GRID  = np.linspace(-1.30 * (INITIAL_CASH + B1_0), 1.30 * (INITIAL_CASH + B1_0), N_W)
B1_GRID = np.linspace(0.0, int(np.round(INITIAL_CASH)), N_B1)

# NELSON-SIEGEL PARAMETERS 
TAU1   = 1.0 / 12.0 
TAU2   = 2.0 / 12.0 

def get_nelson_siegel_yield(tau_years, beta, lmbda=LAMBDA):
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

    yield_rate = beta[0] + beta[1] * f1 + beta[2] * f2
    return yield_rate

# PRECOMPUTED MATRICES
Y1M   = np.array([[get_nelson_siegel_yield(TAU1, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y2M   = np.array([[get_nelson_siegel_yield(TAU2, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y3M   = np.array([[get_nelson_siegel_yield(3.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y4M   = np.array([[get_nelson_siegel_yield(4.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y5M   = np.array([[get_nelson_siegel_yield(5.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y6M   = np.array([[get_nelson_siegel_yield(6.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y7M   = np.array([[get_nelson_siegel_yield(7.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y8M   = np.array([[get_nelson_siegel_yield(8.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y9M   = np.array([[get_nelson_siegel_yield(9.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y10M  = np.array([[get_nelson_siegel_yield(10.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y11M  = np.array([[get_nelson_siegel_yield(11.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y12M  = np.array([[get_nelson_siegel_yield(12.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y13M  = np.array([[get_nelson_siegel_yield(13.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y14M  = np.array([[get_nelson_siegel_yield(14.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)]) 
Y15M  = np.array([[get_nelson_siegel_yield(15.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y16M  = np.array([[get_nelson_siegel_yield(16.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y17M  = np.array([[get_nelson_siegel_yield(17.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y18M  = np.array([[get_nelson_siegel_yield(18.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y19M  = np.array([[get_nelson_siegel_yield(19.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y20M  = np.array([[get_nelson_siegel_yield(20.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y21M  = np.array([[get_nelson_siegel_yield(21.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y22M  = np.array([[get_nelson_siegel_yield(22.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y23M  = np.array([[get_nelson_siegel_yield(23.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])
Y24M  = np.array([[get_nelson_siegel_yield(24.0 / 12.0, [BETA0_GRID[i0], BETA1_GRID[i1], BETA2]) for i1 in range(N1)] for i0 in range(N0)])


R1_M  = 1.0 + Y1M / 12.0           
R2_M  = 1.0 + Y2M * 2.0 / 12.0       
R3_M  = 1.0 + Y3M * 3.0 / 12.0
R4_M  = 1.0 + Y4M * 4.0 / 12.0
R5_M  = 1.0 + Y5M * 5.0 / 12.0
R6_M  = 1.0 + Y6M * 6.0 / 12.0
R7_M  = 1.0 + Y7M * 7.0 / 12.0
R8_M  = 1.0 + Y8M * 8.0 / 12.0
R9_M  = 1.0 + Y9M * 9.0 / 12.0
R10_M = 1.0 + Y10M * 10.0 / 12.0
R11_M = 1.0 + Y11M * 11.0 / 12.0
R12_M = 1.0 + Y12M * 12.0 / 12.0
R13_M = 1.0 + Y13M * 13.0 / 12.0
R14_M = 1.0 + Y14M * 14.0 / 12.0
R15_M = 1.0 + Y15M * 15.0 / 12.0
R16_M = 1.0 + Y16M * 16.0 / 12.0
R17_M = 1.0 + Y17M * 17.0 / 12.0
R18_M = 1.0 + Y18M * 18.0 / 12.0
R19_M = 1.0 + Y19M * 19.0 / 12.0
R20_M = 1.0 + Y20M * 20.0 / 12.0
R21_M = 1.0 + Y21M * 21.0 / 12.0
R22_M = 1.0 + Y22M * 22.0 / 12.0
R23_M = 1.0 + Y23M * 23.0 / 12.0
R24_M = 1.0 + Y24M * 24.0 / 12.0





ALPHA = 0.01*INITIAL_CASH if VERSION == "2Y" else 0.004167*INITIAL_CASH
BETA_DEPOSIT = 1*INITIAL_CASH if VERSION == "2Y" else 0.2083*INITIAL_CASH

L_ALL = ALPHA + BETA_DEPOSIT * Y1M       

# ALM SETUP 
markov_config = {
    "i0_init": I0_INIT,
    "i1_init": I1_INIT,
    "BETA0_GRID": BETA0_GRID,
    "BETA1_GRID": BETA1_GRID,
    "BETA2": BETA2,
    "P0": P0_MATRIX,
    "P1": P1_MATRIX,
    "LAMBDA": LAMBDA,
    "T": T,
    "W0": INITIAL_CASH,
    "yield_1m": Y1M,
    "yield_2m": Y2M,
    "L_ALL": L_ALL,
    "N_W": N_W,
    "N_B1": N_B1,
    "N_X2": N_X2,
    "W_GRID": W_GRID,
    "B1_GRID": B1_GRID,
    "B1_0": B1_0,
    "R1_M": R1_M,
    "R2_M": R2_M,
    "R3_M": R3_M,
    "R5_M": R5_M,
    "R6_M": R6_M,
    "R7_M": R7_M,
    "R8_M": R8_M,
    "R9_M": R9_M,
    "R10_M": R10_M,
    "R11_M": R11_M,
    "R12_M": R12_M,
    "R13_M": R13_M,
    "R14_M": R14_M,
    "R15_M": R15_M,
    "R16_M": R16_M,
    "R17_M": R17_M,
    "R18_M": R18_M,
    "R19_M": R19_M,
    "R20_M": R20_M,
    "R21_M": R21_M,
    "R22_M": R22_M,
    "R23_M": R23_M,
    "R24_M": R24_M,
    "bond_configs": bond_configs,
    "N0": N0,
    "N1": N1,
    "b1_issuance_yield": Y2M[I0_INIT, I1_INIT],
    "GAMMA": GAMMA
}