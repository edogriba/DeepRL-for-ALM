import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.env import DeepALMEnv
from environment.config import F, T
from models.utils import get_nelson_siegel_yield

def evaluate_sac_alm(
    model,
    markov_config: dict,
    eval_episodes: int = 1000,
    seed_offset: int = 1000000,
    verbose_first_episode: bool = False
) -> tuple[dict, list]:
    """
    Evaluates a trained SAC policy on unseen synthetic trajectories.
    The environment now handles early termination and terminal liquidity checks.
    """
    print("\n--- Starting SAC Evaluation ---")
    
    def make_env(seed):
        return DeepALMEnv(markov_config=markov_config, seed=seed, verbose=False)
    
    terminal_navs = []
    terminal_rewards = []
    lcr_violation_counts = []
    bankruptcy_flags = []
    raw_results = []

    for ep in range(eval_episodes):
        current_seed = seed_offset + ep
        
        eval_env = DummyVecEnv([lambda: make_env(seed=current_seed)])
        obs = eval_env.reset()
        done = False
        
        ep_reward = 0.0
        ep_lcr_violations = 0
        ep_ever_bankrupt = False
        month = 1
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            
            ep_reward += rewards[0]
            lcr = infos[0].get("lcr", 1.0)
            nav = infos[0].get("nav", 0.0)
            
            if lcr < 1.0:
                ep_lcr_violations += 1
            
            # The environment marks bankruptcy via the nav or internal state
            if nav < 0:
                ep_ever_bankrupt = True
                
            if verbose_first_episode and ep == 0:
                print(f"Ep 0 | Month {month:2d} | NAV: {nav:8.2f} | LCR: {lcr:5.2f} | Step Reward: {rewards[0]:.2f}")
            
            month += 1
            
            if dones[0]:
                terminal_navs.append(nav)
                terminal_rewards.append(ep_reward)
                lcr_violation_counts.append(ep_lcr_violations)
                bankruptcy_flags.append(ep_ever_bankrupt)
                
                raw_results.append({
                    "seed": current_seed,
                    "nav": nav,
                    "total_reward": ep_reward,
                    "lcr_violations": ep_lcr_violations,
                    "bankrupt": ep_ever_bankrupt
                })
                done = True
                
        eval_env.close()

    nav_arr = np.array(terminal_navs)
    violations_arr = np.array(lcr_violation_counts)
    bankrupt_arr = np.array(bankruptcy_flags)

    # VaR / CVaR settings
    alpha = 0.05  # 5% tail risk

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
        "var_5_nav": float(var_nav),
        "cvar_5_nav": float(cvar_nav),
        "mean_ep_reward": float(np.mean(terminal_rewards)),
        "mean_lcr_violations_per_ep": float(np.mean(violations_arr)),
        "episode_violation_rate": float(np.mean(violations_arr > 0)),
        "default_rate": float(np.mean(bankrupt_arr))
    }

    print("\n=== SAC Evaluation Results ===")
    print(f"Mean NAV:       {metrics['mean_nav']:.2f} ± {metrics['std_nav']:.2f}")
    print(f"Median NAV:     {metrics['median_nav']:.2f}")
    print(f"Min / Max NAV:  {metrics['min_nav']:.2f} / {metrics['max_nav']:.2f}")

    print(f"VaR (5%):       {metrics['var_5_nav']:.2f}")
    print(f"CVaR (5%):      {metrics['cvar_5_nav']:.2f}")

    print(f"Default Rate:   {metrics['default_rate'] * 100:.2f}%")

    print("==============================\n")

    return metrics, raw_results

def evaluate_sac_agent(sac_model, markov_config, K, seed=0, verbose=False):
    """
    Single Point of Truth for SAC evaluation, relying on the environment's internal logic.
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
        action, _ = sac_model.predict(obs, deterministic=True)
        
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