"""
optimize_baselines.py — End-to-End Hyperparameter Optimization for ALM Baselines
================================================================================
Optimizes PPO, A2C, and SAC by dynamically testing both standard hyperparameters
and the use of Dirichlet versus Gaussian architectures.

Aligned with the hybrid CMDP tuner (optimize_pnn.py):

  * Selectable objective: OBJECTIVE_METRIC = "mean_nav" (average episode
    return) or "cvar_05" (CVaR of the worst CVAR_ALPHA fraction of episodes).
  * Multi-seed robustness: each configuration is trained over N_SEEDS random
    initializations and the objective is their aggregate (mean, or
    mean - std) -> reduces init noise.
  * A MedianPruner kills clearly-bad configs after the first seed(s), so the
    multi-seed cost is far below the naive N_SEEDS x factor.
  * Common random numbers: every evaluation uses the same EVAL_SEED, so all
    seeds/configs are scored on the same scenario stream and metric
    differences reflect policy quality, not scenario luck.

Note: with gamma = 1.0 the episode return is the undiscounted sum of rewards,
so if the environment pays the (terminal) NAV as reward, "mean_nav"/"cvar_05"
computed on episode returns coincide with the NAV-based metrics of the hybrid
tuner. All metrics are stored as trial user attributes (mean and std across
seeds), so both can always be inspected after the study regardless of which
one was optimized.
"""

import os
import sys
import numpy as np
import optuna

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# ===========================================================================
# 0. SAFE PATH RESOLUTION AND CUSTOM IMPORTS
# ===========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.env import DeepALMEnv
from environment.config import markov_config
from environment.bond import Bond

# Import custom policies (make sure the names match your DirichletPolicy.py file)
from models.DirichletPolicy import DirichletActorCriticPolicy, DirichletSACPolicy

# ===========================================================================
# TUNING CONSTANTS
# ===========================================================================
TOTAL_TIMESTEPS = 400_000   # prima 200_000 (50k steps ensure Q-value propagation up to T=24)
N_SEEDS         = 3         # PNN tuner uses 10; 3 keeps SAC wall-clock manageable
                            # (the pruner stops bad configs after the first seed)
SELECTION       = "mean"    # or "mean_minus_std" for a robustness-penalized objective
EVAL_EPISODES   = 2000      # prima 2000 -> aligned with the PNN tuner
EVAL_SEED       = 123       # common random numbers: same eval scenarios everywhere
CVAR_ALPHA      = 0.05      # tail fraction used by the CVaR metric
BAD_SCORE       = -1e9      # finite sentinel for failed / non-finite seeds

# Switch to "mean_nav" for the standard expected-return objective,
# or keep "cvar_05" for the tail-risk objective.
OBJECTIVE_METRIC = "cvar_05"
VALID_METRICS = ("mean_nav", "cvar_05")

# ===========================================================================
# ENVIRONMENT SETUP FOR STABLE-BASELINES3
# ===========================================================================
def make_env(use_dirichlet_flag, seed=None):
    """
    Creates the vectorized environment by injecting the Dirichlet flag.
    This allows the environment to know whether it should apply Softmax
    (for the Gaussian policy) or leave the weights unchanged
    (for the Dirichlet policy).
    """
    return DummyVecEnv([
        lambda: Monitor(DeepALMEnv(
            markov_config=markov_config,
            seed=seed,
            use_dirichlet=use_dirichlet_flag
        ))
    ])

# ===========================================================================
# EVALUATION METRICS (mean NAV vs CVaR)
# ===========================================================================
def compute_metrics(episode_rewards):
    """
    Turns the per-episode returns into a metrics dictionary. Key names mirror
    evaluate_cmdp_alm() of the hybrid agent so the two tuners can be compared
    directly (names assume the default CVAR_ALPHA = 0.05).
    """
    rewards = np.asarray(episode_rewards, dtype=float)
    k = max(1, int(np.ceil(CVAR_ALPHA * rewards.size)))
    worst_k = np.sort(rewards)[:k]              # the k worst episodes (lower tail)
    return {
        "mean_nav": float(rewards.mean()),
        "std_nav":  float(rewards.std()),
        "var_05":   float(np.quantile(rewards, CVAR_ALPHA)),
        "cvar_05":  float(worst_k.mean()),
    }


def run_one_seed(make_model, use_dirichlet, seed):
    """
    Trains and evaluates ONE seed of a given configuration -> metrics dict.
    The training env and the SB3 model are seeded with `seed`; the eval env
    always uses EVAL_SEED (common random numbers across all seeds/configs).
    """
    env = make_env(use_dirichlet_flag=use_dirichlet, seed=seed)
    model = make_model(env, seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    eval_env = make_env(use_dirichlet_flag=use_dirichlet, seed=EVAL_SEED)
    episode_rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        return_episode_rewards=True,            # per-episode returns -> CVaR
    )
    return compute_metrics(episode_rewards)


def multi_seed_objective(trial, make_model, use_dirichlet):
    """
    Same structure as the PNN tuner: trains N_SEEDS independent inits of the
    sampled configuration, reports the running mean after each seed (so the
    MedianPruner can kill bad configs early), aggregates all metrics across
    seeds as user attributes, and returns the SELECTION aggregate of
    OBJECTIVE_METRIC. A crashing seed scores BAD_SCORE instead of killing
    the whole study.
    """
    seed_scores = []
    per_seed_metrics = []

    for i in range(N_SEEDS):
        # distinct, reproducible seed per (trial, seed-index)
        seed = 1000 * trial.number + i
        try:
            metrics = run_one_seed(make_model, use_dirichlet, seed)
            score = float(metrics[OBJECTIVE_METRIC])
            if not np.isfinite(score):
                score = BAD_SCORE
            per_seed_metrics.append(metrics)
        except Exception as e:  # a crashing seed shouldn't kill the whole study
            print(f"[trial {trial.number} | seed {seed}] failed: {e}")
            score = BAD_SCORE

        seed_scores.append(score)

        # ---- pruning: report the running mean after each seed -------
        running_mean = float(np.mean(seed_scores))
        trial.report(running_mean, step=i)
        if trial.should_prune():
            # record what we have so far, then let Optuna mark it pruned
            trial.set_user_attr("seed_scores", seed_scores)
            raise optuna.TrialPruned()

    scores = np.asarray(seed_scores, dtype=float)
    mean_score = float(scores.mean())
    std_score  = float(scores.std())

    # Aggregate metrics across seeds (mean/std) for inspection
    if per_seed_metrics:
        for key in per_seed_metrics[0].keys():
            vals = [m[key] for m in per_seed_metrics]
            trial.set_user_attr(f"{key}_mean", float(np.mean(vals)))
            trial.set_user_attr(f"{key}_std",  float(np.std(vals)))
    trial.set_user_attr("seed_scores", seed_scores)
    trial.set_user_attr("obj_mean", mean_score)
    trial.set_user_attr("obj_std",  std_score)

    # Final objective
    if SELECTION == "mean_minus_std":
        return mean_score - std_score
    return mean_score

# ===========================================================================
# SB3 OBJECTIVE FUNCTIONS (PPO, A2C, SAC)
# ===========================================================================
def objective_ppo(trial):
    # Structural choice: Dirichlet vs Gaussian (sampled ONCE per trial,
    # shared by all seeds)
    use_dirichlet = trial.suggest_categorical("use_dirichlet", [True, False])
    policy_class = DirichletActorCriticPolicy if use_dirichlet else "MultiInputPolicy"

    # PPO hyperparameter search space
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    ent_coef = trial.suggest_categorical("ent_coef", [1e-4, 1e-3, 5e-3, 1e-2])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])

    if batch_size > n_steps:
        batch_size = n_steps

    def make_model(env, seed):
        return PPO(
            policy=policy_class,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_range,
            gamma=1.0,
            seed=seed,
            verbose=0
        )

    return multi_seed_objective(trial, make_model, use_dirichlet)


def objective_a2c(trial):
    # Structural choice
    use_dirichlet = trial.suggest_categorical("use_dirichlet", [True, False])
    policy_class = DirichletActorCriticPolicy if use_dirichlet else "MultiInputPolicy"

    # A2C hyperparameter search space
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 5e-3, 1e-3])
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_categorical("ent_coef", [1e-4, 1e-3, 5e-3, 1e-2])

    def make_model(env, seed):
        return A2C(
            policy=policy_class,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=ent_coef,
            gamma=1.0,
            seed=seed,
            verbose=0
        )

    return multi_seed_objective(trial, make_model, use_dirichlet)


def objective_sac(trial):
    # Structural choice
    #use_dirichlet = trial.suggest_categorical("use_dirichlet", [True, False])
    use_dirichlet = False
    policy_class = DirichletSACPolicy if use_dirichlet else "MultiInputPolicy"

    # SAC hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    tau = trial.suggest_categorical("tau", [0.005, 0.01, 0.02])

    def make_model(env, seed):
        return SAC(
            policy=policy_class,
            env=env,
            learning_rate=learning_rate,
            batch_size=256,
            tau=tau,
            train_freq=1,
            gamma=1.0,
            ent_coef="auto",
            seed=seed,
            verbose=0
        )

    return multi_seed_objective(trial, make_model, use_dirichlet)

# ===========================================================================
# EXECUTION ENGINE (OPTUNA STUDY)
# ===========================================================================
def make_study(name):
    """Same sampler/pruner as the PNN tuner, for a like-for-like search."""
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # run the first 5 trials fully (build a baseline)
            n_warmup_steps=0,     # then allow pruning from the first seed onward
            n_min_trials=1,
        ),
        study_name=name,
    )


def report_study(study, label):
    print(f"\n=== Best {label} trial ===")
    print(f"Value ({SELECTION} of {OBJECTIVE_METRIC}): {study.best_value:.4f}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Per-seed scores:", study.best_trial.user_attrs.get("seed_scores"))
    print("Aggregated metrics:")
    for k, v in study.best_trial.user_attrs.items():
        if k.endswith("_mean") or k.endswith("_std"):
            print(f"  {k}: {v:.4f}")
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nPruned {n_pruned}/{len(study.trials)} trials early.")


if __name__ == "__main__":
    assert OBJECTIVE_METRIC in VALID_METRICS, \
        f"OBJECTIVE_METRIC must be one of {VALID_METRICS}, got '{OBJECTIVE_METRIC}'"

    print("Starting the ALM Baseline Optimization Suite...")
    desc = ("mean episode return" if OBJECTIVE_METRIC == "mean_nav"
            else f"CVaR of the worst {CVAR_ALPHA:.0%} of episodes")
    print(f"Optimizing for: {OBJECTIVE_METRIC} ({desc})")
    print(f"Seeds per configuration: {N_SEEDS} (aggregate: {SELECTION})")

    # Number of configurations to test for each algorithm.

    TRIALS = 20

    # PPO Optimization
    print("\n" + "=" * 50)
    print("PPO Optimization")
    study_ppo = make_study("ppo_alm_tuning")
    study_ppo.optimize(objective_ppo, n_trials=TRIALS, show_progress_bar=True)
    report_study(study_ppo, "PPO")

    # A2C Optimization
    #print("\n" + "=" * 50)
    #print("A2C Optimization")
    #study_a2c = make_study("a2c_alm_tuning")
    #study_a2c.optimize(objective_a2c, n_trials=TRIALS, show_progress_bar=True)
    #report_study(study_a2c, "A2C")

    # SAC Optimization
    #print("\n" + "=" * 50)
    #print("SAC Optimization")
    #study_sac = make_study("sac_alm_tuning")
    #study_sac.optimize(objective_sac, n_trials=TRIALS, show_progress_bar=True)
    #report_study(study_sac, "SAC")

    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETED!")
    print("=" * 50)