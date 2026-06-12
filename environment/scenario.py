import numpy as np
from dataclasses import dataclass
from environment.config import (
    LAMBDA, BETA0_GRID, BETA1_GRID, P0_MATRIX, P1_MATRIX, 
    I0_INIT, I1_INIT, BETA2, ALPHA, BETA_DEPOSIT
)

@dataclass
class Scenario:
    """
    Deterministic economic scenario used by DeepALMEnv.
    """
    yields: np.ndarray
    liabilities: np.ndarray

    def __post_init__(self):
        assert self.yields.ndim == 2, "yields must be a 2D array"
        assert self.liabilities.ndim == 2, "liabilities must be a 2D array"
        assert self.yields.shape[0] == self.liabilities.shape[0], "Time dimensions must match"
        
        self.T = self.yields.shape[0]
        self.M = self.liabilities.shape[1]


class DepositBetaLiabilityGenerator:
    """
    Yield-driven liability generator.

    L_t = alpha + beta_deposit * Y_NS(tau_ref; beta0_t, beta1_t, beta2_t) + eps_t
    where eps_t ~ N(0, noise_std^2) is optional i.i.d. noise.
    """

    @staticmethod
    def _ns_yield(tau: float, betas: np.ndarray, lmbda: float) -> np.ndarray:
        """
        Vectorised Nelson-Siegel yield.
        """
        x = tau / lmbda
        if x < 1e-10:
            f1 = 1.0
            f2 = 0.0
        else:
            exp_term = np.exp(-x)
            f1 = (1.0 - exp_term) / x
            f2 = f1 - exp_term

        b0 = betas[..., 0]
        b1 = betas[..., 1]
        b2 = betas[..., 2]
        
        return b0 + b1 * f1 + b2 * f2

    @classmethod
    def generate(
        cls,
        yield_path: np.ndarray,
        alpha: float = ALPHA,
        beta_deposit: float = BETA_DEPOSIT,
        tau_ref: float = 1 / 12,
        lmbda: float = LAMBDA,
        noise_std: float = 0.0,
        seed: int = None,
    ) -> np.ndarray:
        """
        Generates yield-driven liabilities.
        """
        yield_path = np.asarray(yield_path, dtype=float)
        assert yield_path.ndim == 2 and yield_path.shape[1] == 3, "yield_path must be shape (T, 3)"

        y_ref = cls._ns_yield(tau_ref, yield_path, lmbda)
        L = alpha + beta_deposit * y_ref

        if noise_std > 0.0:
            rng = np.random.default_rng(seed)
            L = L + rng.normal(0.0, noise_std, size=L.shape)

        L = np.maximum(L, 0.0)
        return L[:, np.newaxis].astype(np.float32)


class MarkovYieldCurveGenerator:
    """
    Nelson-Siegel beta-path generator driven by Markov chain.
    """

    @staticmethod
    def generate(
        T: int,
        seed: int = None,
        i0_init: int = I0_INIT,
        i1_init: int = I1_INIT,
        BETA0_GRID: np.ndarray = BETA0_GRID,
        BETA1_GRID: np.ndarray = BETA1_GRID,
        BETA2: float = BETA2,
        P0: np.ndarray = P0_MATRIX,
        P1: np.ndarray = P1_MATRIX,
        pure_grid: bool = True,
        jitter_b0: float = 0.003,  
        jitter_b1: float = 0.005, 
    ) -> np.ndarray:
        """
        Generates Markov-driven yield curve paths.
        """
        rng = np.random.default_rng(seed)

        b0g = BETA0_GRID
        b1g = BETA1_GRID
        P0 = P0 if P0 is not None else P0_MATRIX
        P1 = P1 if P1 is not None else P1_MATRIX

        i0, i1 = i0_init, i1_init
        path = []

        for _ in range(T):
            b0 = b0g[i0]
            b1 = b1g[i1]

            if not pure_grid:
                b0 = b0 + rng.normal(0.0, jitter_b0)
                b1 = b1 + rng.normal(0.0, jitter_b1)

            path.append([b0, b1, BETA2])

            # Markov transition
            i0 = rng.choice(len(b0g), p=P0[i0])
            i1 = rng.choice(len(b1g), p=P1[i1])

        return np.array(path, dtype=np.float32)