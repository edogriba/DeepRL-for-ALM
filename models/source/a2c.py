import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.env import DeepALMEnv
from environment.config import F, T
from models.utils import get_nelson_siegel_yield

def evaluate_a2c_alm(
    model,
    markov_config: dict,
    eval_episodes: int = 1000,
    seed_offset: int = 1000000,
    verbose_first_episode: bool = False,
    alpha: float = 0.05
) -> tuple[dict, list]:
    """
    Evaluates a trained A2C policy on unseen synthetic trajectories.
    The environment now handles early termination and terminal liquidity checks.
    """
    print("\n--- Starting A2C Evaluation ---")
    
    def make_env(seed):
        return DeepALMEnv(markov_config=markov_config, seed=seed, verbose=False)
    
    terminal_navs = []
    terminal_rewards = []
    terminal_penalties = []
    lcr_violation_counts = []
    bankruptcy_flags = []
    raw_results = []

    for ep in range(eval_episodes):
        current_seed = seed_offset + ep
        
        eval_env = DummyVecEnv([lambda: make_env(seed=current_seed)])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
        
        obs = eval_env.reset()
        done = False
        
        ep_reward = 0.0
        ep_lcr_violations = 0
        ep_len = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            
            ep_reward += rewards[0]
            ep_len += 1
            lcr = infos[0].get("lcr", 1.0)
            nav = infos[0].get("nav", 0.0)
            penalty = infos[0].get("penalty", 0.0)
            
            if lcr < 1.0:
                ep_lcr_violations += 1
                
            if verbose_first_episode and ep == 0:
                print(f"Ep 0 | Month {ep_len:2d} | NAV: {nav:8.2f} | LCR: {lcr:5.2f} | Step Reward: {rewards[0]:.2f}")
            
            if dones[0]:
                ep_ever_bankrupt = bool(infos[0].get("bankrupt", (ep_len < T) or (nav < 0)))
                terminal_navs.append(nav)
                terminal_rewards.append(ep_reward)
                terminal_penalties.append(penalty)
                lcr_violation_counts.append(ep_lcr_violations)
                bankruptcy_flags.append(ep_ever_bankrupt)
                
                raw_results.append({
                    "seed": current_seed,
                    "nav": nav,
                    "total_reward": ep_reward,
                    "penalty": penalty,
                    "lcr_violations": ep_lcr_violations,
                    "bankrupt": ep_ever_bankrupt
                })
                done = True
                
        eval_env.close()

    nav_arr = np.array(terminal_navs)
    penalty_arr = np.array(terminal_penalties)
    violations_arr = np.array(lcr_violation_counts)
    bankrupt_arr = np.array(bankruptcy_flags)

    # Compute losses relative to initial NAV or simply negative returns
    # Here assuming NAV itself is the terminal wealth metric
    var_nav = np.percentile(nav_arr, alpha * 100)

    # CVaR = average of worst alpha% outcomes
    cvar_nav = np.mean(nav_arr[nav_arr <= var_nav])

    metrics = {
        "mean_nav": float(np.mean(nav_arr)),
        "median_nav": float(np.median(nav_arr)),
        "std_nav": float(np.std(nav_arr)),
        "min_nav": float(np.min(nav_arr)),
        "max_nav": float(np.max(nav_arr)),
        "var_nav": float(var_nav),
        "cvar_nav": float(cvar_nav),
        "mean_penalty": float(np.mean(penalty_arr)),
        "violation_rate": float(np.mean(penalty_arr > 0.0)),
        "mean_ep_reward": float(np.mean(terminal_rewards)),
        "mean_lcr_violations_per_ep": float(np.mean(violations_arr)),
        "default_rate": float(np.mean(bankrupt_arr))
    }

    print("\n=== A2C Evaluation Results ===")
    print(f"Mean NAV:       {metrics['mean_nav']:.2f} ± {metrics['std_nav']:.2f}")
    print(f"Median NAV:     {metrics['median_nav']:.2f}")
    print(f"Min / Max NAV:  {metrics['min_nav']:.2f} / {metrics['max_nav']:.2f}")
    print(f"VaR  (5%):      {metrics['var_nav']:.2f}")
    print(f"CVaR (5%):      {metrics['cvar_nav']:.2f}")
    print(f"Mean Penalty:   {metrics['mean_penalty']:.4f}")
    print(f"Violation Rate: {metrics['violation_rate'] * 100:.4f}%")
    print(f"Default Rate:   {metrics['default_rate'] * 100:.2f}%")
    print("==============================\n")

    return metrics, raw_results

def evaluate_a2c_agent(a2c_model, markov_config, K, seed=0, verbose=False):
    """
    Single Point of Truth for A2C evaluation, relying on the environment's internal logic.
    """
    env = DeepALMEnv(markov_config=markov_config, seed=seed, verbose=verbose)
    obs, _ = env.reset()

    actions_history = []
    wealth_history  = []
    terminal_nav    = 0.0
    final_agent_score = 0.0 
    ever_bankrupt   = False

    for t in range(T):
        inflows = env._get_total_inflows()
        l_t = float(env.liabilities[0])
        projected_cash = env.cash + inflows - l_t
        
        # STRICT LIQUIDITY CHECK
        if projected_cash < 0:
            ever_bankrupt = True
            
            y_1m = get_nelson_siegel_yield(1.0/12.0, env.yield_params)
            r1 = 1.0 + y_1m * (1.0 / 12.0)
            projected_cash *= r1
            projected_cash -= F
            
        investable = max(0.0, projected_cash)
        wealth_history.append(investable)

        # Predict raw action (logits)
        action, _ = a2c_model.predict(obs, deterministic=True)
        
        # Re-apply the same Softmax as the environment for logging
        shifted_logits = action - np.max(action)
        exp_acts = np.exp(shifted_logits)
        weights = exp_acts / np.sum(exp_acts)
        
        # Absolute values in Euro
        abs_acts = weights * investable
        actions_history.append(abs_acts)
        
        # Pass raw logits to the environment
        obs, utility, terminated, _, info = env.step(action)
        
        if verbose:
            print(f"  Inflows: {inflows:.2f}  |  Liability: {l_t:.2f}")
            if projected_cash < 0:
                print(f"  [BANKRUPTCY] Investable budget frozen at 0.00")
            else:
                print(f"  Investable       : {investable:.2f}")
                
            bond_labels = [cfg['bond_type'] for cfg in markov_config["bond_configs"]] + ["HOLD_CASH"]
            print(f"  Action (Absolute): { {l: f'{w:.2f}€' for l, w in zip(bond_labels, abs_acts)} }")

        if terminated:
            terminal_nav = info.get('nav', 0.0)
            final_agent_score = info.get('utility', 0.0)
            
            # Terminal Liquidity / Insolvency Check
            if terminal_nav < 0:
                ever_bankrupt = True
            
            if verbose:
                print("\n" + "="*60)
                print(f"Terminal Utility: {utility:.2f} €")
                print(f"Terminal NAV: {terminal_nav:.2f} €")   
                print(f"Total Penalty Incurred: {info.get('penalty', 0.0):.2f} €")
                print(f"Defaulted during episode: {'YES' if ever_bankrupt else 'NO'}")             
            
            # Pad histories if episode terminates early
            for _ in range(T - t - 1):
                actions_history.append(np.zeros(K + 1))
                wealth_history.append(0.0)
            break

    return np.array(actions_history), np.array(wealth_history), terminal_nav, final_agent_score