"""
optuna_tune_deep_alm_dh.py
==========================
Hyperparameter search for the DeepALM Deep-Hedging (DH) policy, trained
end-to-end by a differentiable rollout against the OCE / entropic-risk
objective (OCEUtilityExp).

Mirrors optuna_tune_hybrid_alm.py:
  * Each configuration is trained over N_SEEDS random initializations and the
    objective is their aggregate (mean, or mean - std) -> reduces init noise.
  * A MedianPruner kills clearly-bad configs after the first seed(s), so the
    multi-seed cost is far below the naive N_SEEDS x factor.
  * Common random numbers: every evaluation uses the SAME fixed seed_offset, so
    (as long as evaluate_deep_alm_dh derives its scenarios from that offset)
    all seeds/configs are scored on the same scenario stream.

What is DIFFERENT from the actor-critic agent
----------------------------------------------
The DH policy is not an actor-critic with a dual/Lagrangian constraint, so
there is no critic_lr, no lmbda, no lambda_lr, no entropy coefficient and no
GAE. The policy is differentiated straight through the rollout, so the things
worth tuning are optimisation-side knobs:

    lr, scheduler_step, scheduler_gamma (StepLR), grad_clip, hidden_dim,
    batch_size

The OCE loss carries its own learnable parameter ``y``; exactly as in the
reference training script it is optimised jointly with the policy parameters.

GAMMA (the entropic risk-aversion) defines the objective itself, so it is held
fixed (taken from environment.config), mirroring how the reference tuner pins
its problem-defining constants. An optional block below shows how to tune it.

Scenario generation is done ONCE (it does not depend on hyperparameters) and
the tensors are reused across every trial and seed; only model init, the
DataLoader shuffle and the optimisation knobs vary per (trial, seed).
"""
import contextlib
import os
import sys

# --- make project root importable BEFORE any project imports -------------
# Ascend from this file until we find the project root (the directory that
# contains both ``environment`` and ``models``); fall back to two levels up.
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(6):
    if os.path.isdir(os.path.join(_root, "environment")) and os.path.isdir(
        os.path.join(_root, "models")
    ):
        break
    _root = os.path.abspath(os.path.join(_root, ".."))
else:
    _root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import optuna
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from environment.bond import Bond
from environment.config import bond_configs, markov_config, T, GAMMA
from environment.scenario import (
    MarkovYieldCurveGenerator,
    DepositBetaLiabilityGenerator,
)
from models.utils import build_state, get_nelson_siegel_yield_batched
from models.source.bdh import (
    DeepALMPolicyDH,
    OCEUtilityExp,
    batched_differentiable_rollout,
    evaluate_deep_alm_dh,
)

# =====================================================================
# TUNING CONSTANTS
# =====================================================================
N_TRIALS        = 20          # number of Optuna trials (configurations)
N_SEEDS         = 5           # random inits per config (DH training is heavy)
SELECTION       = "mean"      # "mean" or "mean_minus_std"

NUM_SCENARIOS   = 15_000       # tuning-time scenario count (full run uses 15_000)
TRAIN_FRAC      = 0.8
TRIAL_EPOCHS    = 60          # tuning-time epochs (full run uses 150)

EVAL_EPISODES   = 2_000
EVAL_SEED_OFFSET = 1_000_000  # common random numbers: same eval scenarios everywhere
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SILENCE_EVAL    = True        # suppress evaluate_deep_alm_dh's per-seed prints

# finite sentinel for a failed / non-finite seed (keeps mean & std finite)
BAD_SCORE = -1e9

# Objective metric, taken from the dict returned by evaluate_deep_alm_dh.
# Options: "mean_nav" (expected terminal NAV), "median_nav", "cvar_95"
# (tail-risk: mean of worst 5%), "entropic_risk" (the trained risk measure),
# "default_rate". The study always *maximises*; metrics where lower is better
# are negated automatically (see HIGHER_IS_BETTER below).
OBJECTIVE_METRIC = "mean_nav"

# Direction table: True if a larger value of the metric is better.
HIGHER_IS_BETTER = {
    "mean_nav": True,
    "median_nav": True,
    "max_nav": True,
    "min_nav": True,
    "var_95": True,
    "cvar_95": True,
    "entropic_risk": False,   # risk measure -> lower is better
    "std_nav": False,
    "default_rate": False,
}


def to_maximize(metric_name: str, value: float) -> float:
    """Convert a raw metric into a value the study should maximise."""
    return value if HIGHER_IS_BETTER.get(metric_name, True) else -value


# =====================================================================
# PROBLEM DIMENSIONS (fixed by config, not tuned)
# =====================================================================
BOND_MATURITIES   = [int(cfg["maturity_months"]) for cfg in bond_configs]
BOND_COUPON_DATES = [Bond(**cfg).coupon_dates for cfg in bond_configs]
K                 = len(bond_configs)
MAX_MATURITY      = int(max(BOND_MATURITIES))
ACTION_DIM        = K + 1     # K bonds + cash
W0                = markov_config.get("W0", 1000.0)

_SAFE_KEYS = ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"]
SAFE_CONFIG = {k: markov_config[k] for k in _SAFE_KEYS if k in markov_config}


# Infer the policy input dimension exactly the way the rollout feeds it,
# i.e. straight from build_state, so the network's first layer always matches.
def infer_state_dim(device: str = "cpu") -> int:
    y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=0, pure_grid=True, **SAFE_CONFIG)
    yields = torch.tensor(np.stack([y_path]), dtype=torch.float32, device=device)
    betas_t = yields[:, 0]
    cash = torch.zeros(1, 1, device=device)
    future_cf = torch.zeros(1, MAX_MATURITY, device=device)
    liab = torch.zeros(1, 1, device=device)
    state = build_state(cash, future_cf, betas_t, liab)
    return state.shape[-1]


# Generate the (yields, liabilities) scenario bank ONCE. Independent of every
# hyperparameter, so it is shared by all trials and seeds.
def build_scenarios(device: str = "cpu"):
    y_list, l_list = [], []
    for i in range(NUM_SCENARIOS):
        y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=i, pure_grid=True, **SAFE_CONFIG)
        l_path = DepositBetaLiabilityGenerator.generate(y_path, noise_std=0.0, seed=i)
        l_path[T + 1:] = 0.0          # liabilities only live within the horizon
        y_list.append(y_path)
        l_list.append(l_path)
    yields = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    liabs = torch.tensor(np.stack(l_list), dtype=torch.float32, device=device)
    return yields, liabs


@contextlib.contextmanager
def _maybe_silence():
    if SILENCE_EVAL:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            yield
    else:
        yield


# =====================================================================
# Train + evaluate ONE seed of a given configuration -> metrics dict
# =====================================================================
def run_one_seed(params: dict, seed: int, state_dim: int,
                 yields: torch.Tensor, liabs: torch.Tensor) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_size = int(TRAIN_FRAC * NUM_SCENARIOS)
    train_ds = TensorDataset(yields[:train_size], liabs[:train_size])

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True, generator=gen
    )

    model = DeepALMPolicyDH(
        state_dim=state_dim, action_dim=ACTION_DIM, T=T,
        hidden_dim=params["hidden_dim"],
    ).to(DEVICE)
    loss_fn = OCEUtilityExp(gamma=GAMMA).to(DEVICE)

    # Jointly optimise the policy and the OCE loss's learnable y (as in the
    # reference training script).
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=params["lr"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params["scheduler_step"], gamma=params["scheduler_gamma"]
    )

    model.train()
    for _epoch in range(TRIAL_EPOCHS):
        for batch_yields, batch_liabs in train_loader:
            batch_yields = batch_yields.to(DEVICE)
            batch_liabs = batch_liabs.to(DEVICE)
            bs = batch_yields.shape[0]

            # Initial holdings/coupons (replicates the reference setup; with
            # random_b1 == 0 these are effectively zero, but kept for fidelity).
            random_b1 = 0.0
            betas_0 = batch_yields[:, 0]
            iy_2m = get_nelson_siegel_yield_batched(2.0 / 12.0, betas_0).squeeze(-1)
            effective_face = random_b1 / (1.0 + iy_2m * (2.0 / 12.0))
            init_h = torch.zeros(bs, K, MAX_MATURITY, device=DEVICE)
            init_c = torch.zeros_like(init_h)
            init_h[:, 0, 1] = effective_face / 100.0
            init_c[:, 0, 1] = iy_2m

            optimizer.zero_grad()
            terminal_value, _, _, _ = batched_differentiable_rollout(
                model=model,
                yields_batch=batch_yields,
                liabilities_batch=batch_liabs,
                T=T,
                bond_maturities=BOND_MATURITIES,
                bond_coupon_dates=BOND_COUPON_DATES,
                initial_holdings=init_h,
                initial_coupons=init_c,
                initial_cash=W0,
            )
            loss = loss_fn(terminal_value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["grad_clip"])
            optimizer.step()
        scheduler.step()

    # Common random numbers: a fixed seed_offset means every seed AND every
    # configuration is evaluated on the same out-of-sample scenario stream.
    with _maybe_silence():
        metrics, _, _ = evaluate_deep_alm_dh(
            model=model,
            markov_config=markov_config,
            eval_episodes=EVAL_EPISODES,
            device=DEVICE,
            seed_offset=EVAL_SEED_OFFSET,
            log_scenario_idx=None,
            gamma=GAMMA,
        )
    return metrics


# =====================================================================
# Objective
# =====================================================================
def make_objective(yields: torch.Tensor, liabs: torch.Tensor):
    state_dim = infer_state_dim(device="cpu")

    def objective(trial: optuna.Trial) -> float:
        # Search space (optimisation-side knobs only)
        params = {
            "lr":              trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            "scheduler_step":  trial.suggest_int("scheduler_step", 20, 80, step=20),
            "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.3, 0.9),
            "grad_clip":       trial.suggest_float("grad_clip", 0.5, 5.0),
            "hidden_dim":      trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "batch_size":      trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }

        seed_scores = []          # in "to-maximize" space (for pruning/objective)
        per_seed_metrics = []

        for i in range(N_SEEDS):
            # distinct, reproducible seed per (trial, seed-index)
            seed = 1000 * trial.number + i
            try:
                metrics = run_one_seed(params, seed, state_dim, yields, liabs)
                raw = float(metrics[OBJECTIVE_METRIC])
                if not np.isfinite(raw):
                    score = BAD_SCORE
                else:
                    score = to_maximize(OBJECTIVE_METRIC, raw)
                    per_seed_metrics.append(metrics)
            except Exception as e:  # a crashing seed shouldn't kill the study
                print(f"[trial {trial.number} | seed {seed}] failed: {e}")
                score = BAD_SCORE

            seed_scores.append(score)

            # ---- pruning: report the running mean after each seed -------
            running_mean = float(np.mean(seed_scores))
            trial.report(running_mean, step=i)
            if trial.should_prune():
                trial.set_user_attr("seed_scores", seed_scores)
                raise optuna.TrialPruned()

        scores = np.asarray(seed_scores, dtype=float)
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        # Aggregate the raw metrics across seeds (mean/std) for inspection.
        if per_seed_metrics:
            for key in per_seed_metrics[0].keys():
                vals = [m[key] for m in per_seed_metrics]
                if all(np.isscalar(v) and isinstance(v, (int, float)) for v in vals):
                    trial.set_user_attr(f"{key}_mean", float(np.mean(vals)))
                    trial.set_user_attr(f"{key}_std", float(np.std(vals)))
        trial.set_user_attr("seed_scores", seed_scores)
        trial.set_user_attr("obj_mean", mean_score)
        trial.set_user_attr("obj_std", std_score)

        # Final objective (study maximises)
        if SELECTION == "mean_minus_std":
            return mean_score - std_score
        return mean_score

    return objective


# =====================================================================
# Run
# =====================================================================
def main():
    print(f"Generating {NUM_SCENARIOS} scenarios on {DEVICE} ...")
    yields, liabs = build_scenarios(device=DEVICE)
    print(f"State dim: {infer_state_dim()} | action dim: {ACTION_DIM} | K bonds: {K}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # run the first 5 trials fully (build a baseline)
            n_warmup_steps=0,     # then allow pruning from the first seed onward
            n_min_trials=1,
        ),
        study_name="deep_alm_dh_tuning",
    )
    study.optimize(
        make_objective(yields, liabs),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n=== Best trial ===")
    print(f"Value ({SELECTION} of {OBJECTIVE_METRIC}, maximise-space): {study.best_value:.4f}")
    raw_best = study.best_trial.user_attrs.get(f"{OBJECTIVE_METRIC}_mean")
    if raw_best is not None:
        print(f"Best raw {OBJECTIVE_METRIC} (mean over seeds): {raw_best:.4f}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Per-seed scores:", study.best_trial.user_attrs.get("seed_scores"))
    print("Aggregated metrics:")
    for k, v in study.best_trial.user_attrs.items():
        if k.endswith("_mean") or k.endswith("_std"):
            try:
                print(f"  {k}: {float(v):.4f}")
            except (TypeError, ValueError):
                pass

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nPruned {n_pruned}/{len(study.trials)} trials early.")

    return study


if __name__ == "__main__":
    main()