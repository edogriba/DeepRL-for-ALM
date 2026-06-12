"""
optuna_tune_hybrid_alm.py
=========================
Hyperparameter search for the Actor-Critic + Lagrangian CMDP agent.

Two constraints:
  1. lambda_lr is not optimized -> hard-pinned to 0.0 (no adaptive/dual lambda).
     Because lambda_lr == 0, the value passed as lmbda stays constant for the
     whole run, so lmbda itself becomes a static hyperparameter we tune.
  2. ent_coef_init and ent_coef_final are optimized but always share the SAME
     value (no decaying entropy). We sample a single ent_coef and pass it to
     both arguments; the linear-decay formula then collapses to a constant.

Robustness:
  * Each configuration is trained over N_SEEDS random initializations and the
    objective is their aggregate (mean, or mean - std) -> reduces init noise.
  * A MedianPruner kills clearly-bad configs after the first seed(s), so the
    multi-seed cost is far below the naive N_SEEDS x factor.
  * Common random numbers: the global torch/numpy RNGs are re-seeded with
    EVAL_SEED right before every evaluation, so (as long as
    evaluate_cmdp_alm draws its scenarios from the global RNGs) all
    seeds/configs are scored on the same scenario stream -- aligned with
    optimize_baselines.py.
"""
import math
import os
import sys

# --- make project root importable BEFORE any project imports -------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import optuna
import torch

from source.pnncritic import (
    HybridActor,
    ValueCritic,
    train_cmdp_alm,
    evaluate_cmdp_alm,
)
from environment.config import T, markov_config
from environment.scenario import MarkovYieldCurveGenerator
from utils import build_state

# TUNING CONSTANTS

N_TRIALS       = 40        # number of Optuna trials (configurations)
N_SEEDS        = 10
SELECTION      = "mean"
TRIAL_EPOCHS   = 200
TRIAL_BATCH    = 512
EVAL_EPISODES  = 2000
EVAL_SEED      = 123       # common random numbers: same eval scenarios everywhere
GAMMA          = 1.0
LAMBDA_LR      = 0.0
CRITIC_WARMUP  = 0
PENALTY_LIMIT  = 0.0
HIDDEN_DIM     = 64        # default hidden dim for both actor and critic
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# finite sentinel for a failed / non-finite seed (keeps mean & std finite)
BAD_SCORE = -1e9

# Switch to "mean_nav" for the standard expected-return objective
# ("cvar_05" = tail-risk objective)
OBJECTIVE_METRIC = "cvar_05"


# Helper: infer state_dim, K, etc. from the config (robust to n_betas/max_M)
def infer_dims(markov_config: dict, device: str = "cpu"):
    T = markov_config["T"]
    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    K = len(bond_maturities)
    max_M = max(bond_maturities)
    safe_config = {
        k: markov_config[k]
        for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"]
        if k in markov_config
    }

    y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=0, pure_grid=True, **safe_config)
    yields = torch.tensor(np.stack([y_path]), dtype=torch.float32, device=device)
    betas_t = yields[:, 0]

    cash = torch.zeros(1, 1, device=device)
    future_cf = torch.zeros(1, max_M, device=device)
    liab = torch.zeros(1, 1, device=device)
    state = build_state(cash, future_cf, betas_t, liab)
    state_dim = state.shape[-1]
    return state_dim, K


# Train + evaluate ONE seed of a given configuration -> objective scalar + metrics
def run_one_seed(params: dict, seed: int, state_dim: int, action_dim: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    actor  = HybridActor(state_dim, action_dim, hidden_dim=HIDDEN_DIM, T=T)
    critic = ValueCritic(state_dim, hidden_dim=HIDDEN_DIM, T=T)

    actor, critic, _ = train_cmdp_alm(
        actor, critic, markov_config,
        epochs=TRIAL_EPOCHS,
        batch_size=TRIAL_BATCH,
        lr=params["lr"],
        critic_lr=params["critic_lr"],
        lambda_lr=LAMBDA_LR,                 # no adaptive lambda
        lmbda=params["lmbda"],
        ent_coef_init=params["ent_coef_init"],
        ent_coef_final=params["ent_coef_final"],   # no decay
        gamma=GAMMA,
        gae_lambda=params["gae_lambda"],
        critic_warmup=CRITIC_WARMUP,
        penalty_limit=PENALTY_LIMIT,
        log_every=10,
        device=DEVICE,
    )

    # Common random numbers: re-seed the global RNGs so every evaluation
    # (across seeds AND configurations) sees the same scenario stream.
    # Best effort -- effective as long as evaluate_cmdp_alm draws its
    # scenarios from the global torch / numpy RNGs.
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    metrics, _ = evaluate_cmdp_alm(
        actor, critic, markov_config,
        eval_episodes=EVAL_EPISODES,
        device=DEVICE,
    )
    return metrics


# Objective
def make_objective(markov_config: dict):
    state_dim, K = infer_dims(markov_config, device=DEVICE)
    action_dim = K + 1  # K bonds + cash

    def objective(trial: optuna.Trial) -> float:
        # Search space
        params = {
            "lr":         trial.suggest_float("lr", 1e-3, 1.5e-2, log=True),
            "critic_lr":  trial.suggest_float("critic_lr", 5e-4, 5e-3, log=True),
            "lmbda":      trial.suggest_float("lmbda", 2.5, 10.0),
            "ent_coef_init":   trial.suggest_float("ent_coef_init", 1e-3, 1e-2, log=True),
            "ent_coef_final":   trial.suggest_float("ent_coef_final", 1e-3, 1e-2, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 1.0)
        }

        seed_scores = []
        per_seed_metrics = []

        for i in range(N_SEEDS):
            # distinct, reproducible seed per (trial, seed-index)
            seed = 1000 * trial.number + i
            try:
                metrics = run_one_seed(params, seed, state_dim, action_dim)
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

        # Aggregate metrics across seeds (mean) for inspection
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

    return objective


# Run
def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # run the first 5 trials fully (build a baseline)
            n_warmup_steps=0,     # then allow pruning from the first seed onward
            n_min_trials=1,
        ),
        study_name="hybrid_alm_tuning",
    )
    study.optimize(make_objective(markov_config), n_trials=N_TRIALS, show_progress_bar=True)

    print("\n=== Best trial ===")
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

    return study


if __name__ == "__main__":
    main()