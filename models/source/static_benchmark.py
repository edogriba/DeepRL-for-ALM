import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

from environment.env import DeepALMEnv
from environment.config import T

class StaticPolicy:
    def __init__(self, K, cash_weight=0.10, bond_weights=None):
        self.K = K
        cash_weight = float(cash_weight)
        assert 0.0 <= cash_weight < 1.0, "cash_weight must be in [0, 1)"

        if bond_weights is None:
            bw = np.ones(K, dtype=np.float64)
        else:
            bw = np.asarray(bond_weights, dtype=np.float64)
            assert bw.shape == (K,), "bond_weights must have length K"
            assert np.all(bw >= 0.0), "bond_weights must be non-negative"

        bw = bw / bw.sum() * (1.0 - cash_weight)

        self.weights = np.zeros(K + 1, dtype=np.float64)
        self.weights[:K] = bw
        self.weights[-1] = cash_weight

    def __call__(self, env, investable):
        return self.weights.copy()


def _run_single_episode(args):
    """Funzione helper isolata per permettere il Multiprocessing su CPU multiple."""
    policy, markov_config, K, seed = args
    env = DeepALMEnv(markov_config=markov_config, seed=seed, verbose=False, use_dirichlet=True)
    obs, _ = env.reset()

    terminal_nav, final_agent_score = 0.0, 0.0
    ever_bankrupt = False
    total_penalty = 0.0
    violations = 0

    for t in range(T):
        inflows = env._get_total_inflows()
        l_t = float(env.liabilities[0])
        projected_cash = env.cash + inflows - l_t

        if projected_cash < 0:
            ever_bankrupt = True

        investable = max(0.0, projected_cash)
        weights = policy(env, investable)

        obs, utility, terminated, _, info = env.step(weights.astype(np.float32))

        # Raccogliamo penalty e violazioni
        step_penalty = info.get('penalty', 0.0)
        total_penalty += step_penalty
        if step_penalty < 0 or projected_cash < 0:
            violations += 1

        if terminated:
            terminal_nav = info.get('nav', 0.0)
            final_agent_score = info.get('utility', 0.0)
            if terminal_nav < 0:
                ever_bankrupt = True
            break

    return terminal_nav, final_agent_score, ever_bankrupt, total_penalty, violations


def evaluate_teichmann_alm(policy, markov_config, K, eval_episodes=100_000, seed_offset=10_000_000, alpha=0.05):
    """
    Versione Multi-Core (Parallela) per eseguire 100k episodi in una frazione del tempo.
    Produce l'esatto output richiesto.
    """
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()

    args_list = [(policy, markov_config, K, seed_offset + ep) for ep in range(eval_episodes)]

    navs, scores, defaults, penalties, violation_rates = [], [], [], [], []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(_run_single_episode, args_list))

    for nav, score, is_default, penalty, episode_violations in results:
        navs.append(nav)
        scores.append(score)
        defaults.append(1.0 if is_default else 0.0)
        penalties.append(penalty)
        violation_rates.append(episode_violations / float(T))

    nav_arr = np.asarray(navs, dtype=float)
    var_q = np.percentile(nav_arr, alpha * 100)
    cvar = float(np.mean(nav_arr[nav_arr <= var_q])) if np.any(nav_arr <= var_q) else float(var_q)

    metrics = {
        "mean_nav":        float(np.mean(nav_arr)),
        "std_nav":         float(np.std(nav_arr)),
        "median_nav":      float(np.median(nav_arr)),
        "min_nav":         float(np.min(nav_arr)),
        "max_nav":         float(np.max(nav_arr)),
        "var_05":          float(var_q),
        "cvar_05":         cvar,
        "mean_penalty":    float(np.mean(penalties)),
        "violation_rate":  float(np.mean(violation_rates)),
        "default_rate":    float(np.mean(defaults)),
    }

    elapsed = time.time() - start_time

    print(f"\n[Completed in {elapsed:.1f} seconds]")
    print("=== Evaluation Results ===")
    print(f"Mean NAV     : {metrics['mean_nav']:.2f} ± {metrics['std_nav']:.2f}")   # <-- ± aggiunto
    print(f"Median NAV   : {metrics['median_nav']:.2f}")
    print(f"Min / Max NAV: {metrics['min_nav']:.2f} / {metrics['max_nav']:.2f}")
    print(f"VaR  (5%)    : {metrics['var_05']:.2f}")
    print(f"CVaR (5%)    : {metrics['cvar_05']:.2f}")
    print(f"Mean Penalty : {metrics['mean_penalty']:.4f}")
    print(f"Violation Rate: {metrics['violation_rate']*100:.2f}%")
    print(f"Default Rate : {metrics['default_rate']*100:.2f}%")
    print("==========================")

    return metrics, nav_arr