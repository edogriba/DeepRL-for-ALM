import numpy as np
import matplotlib.pyplot as plt
from models.source.sac import evaluate_sac_agent
from models.source.pnncritic import evaluate_pnncritic_agent

def compare_allocations(actor_pnn, model_sac, markov_config, K, seed=42):
    """
    Evaluates both agents on the same seed and plots their allocations side-by-side.
    """
    # 1. Run Evaluations
    pnn_acts, pnn_wealth, pnn_nav, pnn_score = evaluate_pnncritic_agent(
        actor_pnn, markov_config, K, seed=seed, verbose=False
    )
    
    sac_acts, sac_wealth, sac_nav, sac_score = evaluate_sac_agent(
        model_sac, markov_config, K, seed=seed, verbose=False
    )

    # 2. Setup Data
    T = markov_config["T"]
    month_labels = [f"Mo {t}" for t in range(T)] 
    time_steps = np.arange(T)
    
    # FIX: Reverted to put HOLD_CASH at the end of the array.
    bond_labels = [cfg['bond_type'] for cfg in markov_config["bond_configs"]] + ["HOLD_CASH"]
    num_assets = K + 1

    print(f"--- Evaluation Results (Seed: {seed}) ---")
    print(f"SAC Terminal NAV: {sac_nav:.2f} € | Utility: {sac_score:.2f}")
    print(f"PNN Terminal NAV: {pnn_nav:.2f} € | Utility: {pnn_score:.2f}\n")

    bar_width = 0.35

    # ==========================================
    # Plot 1: Subplots per Asset (Grouped Bars)
    # ==========================================
    fig1, axes = plt.subplots(num_assets, 1, figsize=(16, 3.5 * num_assets), sharex=True)
    if num_assets == 1:
        axes = [axes]

    for i, label in enumerate(bond_labels):
        ax = axes[i]
        sac_alloc = sac_acts[:, i]
        pnn_alloc = pnn_acts[:, i]

        # Plot side-by-side
        bars_sac = ax.bar(time_steps - bar_width/2, sac_alloc, width=bar_width, 
                          label='SAC', color='#1f77b4', edgecolor='black', linewidth=0.5)
        bars_pnn = ax.bar(time_steps + bar_width/2, pnn_alloc, width=bar_width, 
                          label='PNNCritic', color='#ff7f0e', edgecolor='black', linewidth=0.5)

        ax.set_title(f"Allocation: {label}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Euros (€)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        
        # Expand Y-axis slightly for visual clarity
        ax.set_ylim(0, max(max(sac_alloc), max(pnn_alloc)) * 1.25 + 1e-9)

    axes[-1].set_xlabel("Timeline", fontsize=12)
    axes[-1].set_xticks(time_steps)
    axes[-1].set_xticklabels(month_labels, rotation=45)
    
    fig1.suptitle(f"SAC vs PNNCritic: Asset-by-Asset Allocation (Seed {seed})", fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # ==========================================
    # Plot 2: Full Portfolio Paired Stacked Bars
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(18, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, num_assets))
    
    bottom_sac = np.zeros(T)
    bottom_pnn = np.zeros(T)

    for i, label in enumerate(bond_labels):
        sac_vals = sac_acts[:, i]
        pnn_vals = pnn_acts[:, i]
        
        # SAC stacked segments
        bars_sac = ax2.bar(time_steps - bar_width/2, sac_vals, width=bar_width, 
                           bottom=bottom_sac, color=colors[i], edgecolor='white')
        # PNN stacked segments
        bars_pnn = ax2.bar(time_steps + bar_width/2, pnn_vals, width=bar_width, 
                           bottom=bottom_pnn, color=colors[i], edgecolor='white')
        
        bottom_sac += sac_vals
        bottom_pnn += pnn_vals

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='white', label=label) 
                       for i, label in enumerate(bond_labels)]
    legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Left Bar = SAC'))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Right Bar = PNN'))

    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax2.set_title(f"Portfolio Allocation Over Time (SAC vs PNNCritic)", fontsize=16)
    ax2.set_xlabel("Timeline", fontsize=12)
    ax2.set_ylabel("Total Invested Wealth (€)", fontsize=12)
    
    ax2.set_xticks(time_steps)
    ax2.set_xticklabels(month_labels, rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    