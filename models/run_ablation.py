import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.distributions import Dirichlet

# Safely add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environment.bond import Bond
from models.utils import get_nelson_siegel_yield_batched, build_state, get_nelson_siegel_yield
from environment.scenario import DepositBetaLiabilityGenerator, MarkovYieldCurveGenerator
from environment.config import RHO, F, THETA, markov_config



# Model Achitectures

class HybridActor(nn.Module):
    """ Proposed Model (Full): Includes Temporal Feature and Previous Action """
    def __init__(self, state_dim: int, action_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),                 nn.ELU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, t: int, state: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = False):
        batch_size = state.size(0)
        t_norm = torch.full((batch_size, 1), t / self.T, device=state.device, dtype=torch.float32)
        x = torch.cat([state, prev_action, t_norm], dim=-1)
        logits = self.action_head(self.trunk(x))
        
        alphas = F_nn.softplus(logits) + 1.0e-3
        dist = Dirichlet(alphas)
        action = alphas / alphas.sum(-1, keepdim=True) if deterministic else dist.sample()

        eps = 1e-8
        safe_action = (action + eps) / (action + eps).sum(dim=-1, keepdim=True)
        return safe_action, dist.log_prob(safe_action.detach()), dist.entropy()

class ValueCritic(nn.Module):
    """ Baseline Critic V(s_t) with temporal awareness """
    def __init__(self, state_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.v_trunk = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t: int, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(0)
        t_norm = torch.full((batch_size, 1), t / self.T, device=state.device, dtype=torch.float32)
        return self.v_trunk(torch.cat([state, t_norm], dim=-1))

class HybridActorGaussian(nn.Module):
    """ Ablation A: Standard Continuous Policy (Gaussian + Softmax on the Simplex) """
    def __init__(self, state_dim: int, action_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),                 nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, t: int, state: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = False):
        batch_size = state.size(0)
        t_norm = torch.full((batch_size, 1), t / self.T, device=state.device, dtype=torch.float32)
        x = torch.cat([state, prev_action, t_norm], dim=-1)
        
        mean = self.mean_head(self.trunk(x))
        std = self.log_std.expand_as(mean).exp()
        dist = torch.distributions.Normal(mean, std)

        u = mean if deterministic else dist.rsample()
        action = F_nn.softmax(u, dim=-1)
        return action, dist.log_prob(u).sum(dim=-1, keepdim=True), dist.entropy().sum(dim=-1, keepdim=True)

class HybridActorNoTime(nn.Module):
    """ Ablation B: Removed normalized time (t_norm) """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ELU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, t: int, state: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = False):
        x = torch.cat([state, prev_action], dim=-1)
        logits = self.action_head(self.trunk(x))
        alphas = F_nn.softplus(logits) + 1.0e-3 
        dist = Dirichlet(alphas)
        action = alphas / alphas.sum(-1, keepdim=True) if deterministic else dist.sample()
        eps = 1e-8
        safe_action = (action + eps) / (action + eps).sum(dim=-1, keepdim=True)
        return safe_action, dist.log_prob(safe_action.detach()), dist.entropy()

class ValueCriticNoTime(nn.Module):
    """ Baseline Critic V(s_t) without temporal awareness """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.v_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, t: int, state: torch.Tensor) -> torch.Tensor:
        return self.v_trunk(state)

class HybridActorNoPrevAction(nn.Module):
    """ Ablation C: Removed dependency on the previous action """
    def __init__(self, state_dim: int, action_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ELU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, t: int, state: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = False):
        batch_size = state.size(0)
        t_norm = torch.full((batch_size, 1), t / self.T, device=state.device, dtype=torch.float32)
        x = torch.cat([state, t_norm], dim=-1) 
        logits = self.action_head(self.trunk(x))
        alphas = F_nn.softplus(logits) + 1.0e-3 
        dist = Dirichlet(alphas)
        action = alphas / alphas.sum(-1, keepdim=True) if deterministic else dist.sample()
        eps = 1e-8
        safe_action = (action + eps) / (action + eps).sum(dim=-1, keepdim=True)
        return safe_action, dist.log_prob(safe_action.detach()), dist.entropy()

class HybridActorNoTimeNoPrevAction(nn.Module):
    """ Ablation D: Purely Markovian Model (Without time and without past actions) """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, t: int, state: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = False):
        logits = self.action_head(self.trunk(state))
        alphas = F_nn.softplus(logits) + 1.0e-3
        dist = Dirichlet(alphas)
        action = alphas / alphas.sum(-1, keepdim=True) if deterministic else dist.sample()
        eps = 1e-8
        safe_action = (action + eps) / (action + eps).sum(dim=-1, keepdim=True)
        return safe_action, dist.log_prob(safe_action.detach()), dist.entropy()


# Vectorized rollout simulator

def batched_cmdp_rollout(
    actor, critic, yields_batch, liabilities_batch, T, bond_maturities,
    bond_coupon_dates, bond_nominal: float = 100.0, initial_cash: float = 1000.0,
    initial_holdings=None, initial_coupons=None, deterministic: bool = False,
) -> dict:
    device = yields_batch.device
    batch_size = yields_batch.shape[0]
    K = len(bond_maturities)
    
    bond_maturities = [int(m.item() if hasattr(m, 'item') else m) for m in bond_maturities]
    max_M = max(bond_maturities)

    cash = torch.full((batch_size, 1), initial_cash, device=device)
    is_alive = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
    
    holdings = initial_holdings.to(device).clone() if initial_holdings is not None else torch.zeros(batch_size, K, max_M, device=device)
    coupons = initial_coupons.to(device).clone() if initial_coupons is not None else torch.zeros(batch_size, K, max_M, device=device)
    prev_action = torch.zeros(batch_size, K + 1, device=device)
    
    log_probs_seq, penalties_seq, v_preds_seq, entropies_seq, is_alive_seq = [], [], [], [], []
    early_terminal_nav = torch.zeros(batch_size, 1, device=device)

    for t in range(T):
        betas_t = yields_batch[:, t]
        step_penalty = torch.zeros(batch_size, 1, device=device)

        # Cash Inflows
        inflows = torch.zeros(batch_size, 1, device=device)
        for k, M in enumerate(bond_maturities):
            cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M
            inflows = inflows + holdings[:, k, 0:1] * bond_nominal
            for c in bond_coupon_dates[k]:
                slot_c = M - c
                if 0 <= slot_c < max_M:
                    inflows = inflows + holdings[:, k, slot_c:slot_c+1] * (bond_nominal * coupons[:, k, slot_c:slot_c+1] / cpn_rate)

        # Hard Insolvency Check
        projected_cash = cash + inflows - liabilities_batch[:, t]
        newly_bankrupt = (projected_cash < 0) & is_alive
        
        y_1m = get_nelson_siegel_yield_batched(1.0 / 12.0, betas_t)
        debt_interest = 1.0 + y_1m * (1.0 / 12.0)
        
        cash = torch.where(newly_bankrupt, (projected_cash * debt_interest) - F, projected_cash)
        step_penalty = torch.where(newly_bankrupt, step_penalty + F, step_penalty)
        
        if newly_bankrupt.any():
            pv_assets = torch.zeros(batch_size, 1, device=device)
            for k, M in enumerate(bond_maturities):
                cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M if M > 0 else 0
                for m in range(max_M):
                    units = holdings[:, k, m]
                    if units.abs().sum() == 0: continue
                    tau_face = (m + 1) / 12.0 
                    y_face = get_nelson_siegel_yield_batched(tau_face, betas_t)
                    pv_assets += (units * bond_nominal).unsqueeze(-1) / (1.0 + y_face * tau_face)
                    for c in bond_coupon_dates[k]:
                        age = M - m - 1
                        if c > age:
                            tau_c = (c - age) / 12.0
                            y_c = get_nelson_siegel_yield_batched(tau_c, betas_t)
                            coupon_amount = bond_nominal * coupons[:, k, m] / cpn_rate
                            pv_assets += (units * coupon_amount).unsqueeze(-1) / (1.0 + y_c * tau_c)

            pv_liabs = torch.zeros(batch_size, 1, device=device)
            L_total = liabilities_batch.shape[1]
            for j in range(max_M - 1):
                idx = t + 1 + j
                if idx >= L_total: break
                tau_l = (j + 1) / 12.0
                y_l = get_nelson_siegel_yield_batched(tau_l, betas_t)
                pv_liabs += liabilities_batch[:, idx].view(-1, 1) / (1.0 + y_l * tau_l)

            mtm_nav = cash + pv_assets - pv_liabs
            early_terminal_nav = torch.where(newly_bankrupt, mtm_nav, early_terminal_nav)
            
        is_alive = is_alive & ~newly_bankrupt
        investable = torch.relu(cash) * is_alive.float()

        # State Generation
        future_cf = torch.zeros(batch_size, max_M, device=device)
        for k, M in enumerate(bond_maturities):
            cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M
            for m in range(max_M):
                future_cf[:, m] += holdings[:, k, m] * bond_nominal
                for c in bond_coupon_dates[k]:
                    s = m + (M - c)
                    if 0 <= s < max_M:
                        future_cf[:, m] += holdings[:, k, s] * (bond_nominal * coupons[:, k, s] / cpn_rate)

        state = build_state(cash , future_cf , betas_t, liabilities_batch[:, t].view(-1, 1) ).detach()

        v_pred_t = critic(t, state)
        weights, log_prob_t, entropy_t = actor(t, state, prev_action, deterministic=deterministic)
        
        log_prob_t = log_prob_t.view(-1, 1) * is_alive.float()
        entropy_t  = entropy_t.view(-1, 1) * is_alive.float()

        new_holdings, new_coupons = torch.zeros_like(holdings), torch.zeros_like(coupons)
        new_holdings[:, :, :-1], new_coupons[:, :, :-1] = holdings[:, :, 1:], coupons[:, :, 1:]

        for k, M in enumerate(bond_maturities):
            add_h = (weights[:, k:k+1] * investable) / bond_nominal
            new_holdings[:, k, M-1:M] += add_h
            new_coupons[:, k, M-1:M] = get_nelson_siegel_yield_batched(M / 12.0, betas_t)

        cash = cash - (investable - (weights[:, K:K+1] * investable))
        holdings, coupons, prev_action = new_holdings, new_coupons, weights

        # LCR Check
        hqla = cash.clone()
        for k in range(K):
            dates = bond_coupon_dates[k]
            months = dates[0] if len(dates) == 1 else dates[1] - dates[0]
            bond_cf = bond_nominal + coupons[:, k, :] * bond_nominal * (months / 12.0)
            hqla += holdings[:, k, 0:1] * bond_cf[:, 0:1]

        lcr = hqla / (liabilities_batch[:, t + 1].view(-1, 1) + 1e-8)
        lcr_mask = (lcr < 1.0) & is_alive
        lcr_penalty = torch.where(lcr_mask, (1.0 - lcr) * RHO + float(THETA), torch.zeros_like(lcr))
        
        cash = cash - lcr_penalty
        step_penalty = step_penalty + lcr_penalty

        log_probs_seq.append(log_prob_t)
        penalties_seq.append(step_penalty)
        v_preds_seq.append(v_pred_t)
        entropies_seq.append(entropy_t)
        is_alive_seq.append(is_alive.float())

    # Terminal Valuation
    terminal_value = cash - liabilities_batch[:, T].view(-1, 1)
    last_betas = yields_batch[:, T]

    for k, M in enumerate(bond_maturities):
        cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M
        terminal_value = terminal_value + holdings[:, k, 0:1] * bond_nominal
        for c in bond_coupon_dates[k]:
            slot_c = M - c
            if 0 <= slot_c < max_M:
                terminal_value = terminal_value + holdings[:, k, slot_c:slot_c+1] * (bond_nominal * coupons[:, k, slot_c:slot_c+1] / cpn_rate)
                
    new_holdings, new_coupons = torch.zeros_like(holdings), torch.zeros_like(coupons)
    new_holdings[:, :, :-1], new_coupons[:, :, :-1] = holdings[:, :, 1:], coupons[:, :, 1:]

    for k, M in enumerate(bond_maturities):
        cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M if M > 0 else 0
        for m in range(max_M):
            units = new_holdings[:, k, m]
            if units.abs().sum() == 0: continue
            y_face = get_nelson_siegel_yield_batched((m + 1) / 12.0, last_betas)
            terminal_value = terminal_value + (units * bond_nominal).unsqueeze(-1) / (1.0 + y_face * ((m + 1) / 12.0))
            for c in bond_coupon_dates[k]:
                if c > M - m - 1:
                    tau_c = c - (M - m - 1)
                    y_c = get_nelson_siegel_yield_batched(tau_c / 12.0, last_betas)
                    terminal_value = terminal_value + (units * (bond_nominal * new_coupons[:, k, m] / cpn_rate)).unsqueeze(-1) / (1.0 + y_c * (tau_c / 12.0))

    pv_liabs_term = torch.zeros(batch_size, 1, device=device)
    L_total = liabilities_batch.shape[1]
    for j in range(max_M - 1):
        idx = T + 1 + j
        if idx >= L_total: break
        tau_l = (j + 1) / 12.0
        y_l = get_nelson_siegel_yield_batched(tau_l, last_betas)
        pv_liabs_term += liabilities_batch[:, idx].view(-1, 1) / (1.0 + y_l * tau_l)
    
    terminal_value = terminal_value - pv_liabs_term
    term_penalty = torch.where((terminal_value < 0) & is_alive, torch.tensor(float(F), device=device), torch.tensor(0.0, device=device))
    terminal_value = terminal_value - term_penalty
    penalties_seq[-1] = penalties_seq[-1] + term_penalty
    
    final_nav = torch.where(~is_alive, early_terminal_nav, terminal_value)
    
    return {
        "true_nav": final_nav.squeeze(-1),
        "log_probs": torch.cat(log_probs_seq, dim=1),
        "penalties": torch.cat(penalties_seq, dim=1),
        "v_preds": torch.cat(v_preds_seq, dim=1),
        "entropies": torch.cat(entropies_seq, dim=1),
        "is_alive_seq": torch.cat(is_alive_seq, dim=1)
    }


# Standard training loop

def train_cmdp_alm(
    actor, critic, markov_config, epochs: int = 1000, batch_size: int = 2048,
    lr: float = 5e-4, critic_lr: float = 1e-3, lambda_lr: float = 1e-2,
    log_every: int = 50, device: str = "cpu", critic_warmup: int = 50, 
    penalty_limit: float = 0.0, lmbda: float = 3.0, ent_coef_init: float = 0.01, 
    ent_coef_final: float = 1e-5
):
    T = markov_config["T"]
    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    bond_coupon_dates = [Bond(**cfg).coupon_dates for cfg in markov_config["bond_configs"]]
    safe_config = {k: markov_config[k] for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"] if k in markov_config}
    W0, max_F = markov_config["W0"], float(F)

    actor, critic = actor.to(device), critic.to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    log_lambda = torch.nn.Parameter(torch.tensor(math.log(lmbda), device=device))
    lambda_optim = torch.optim.Adam([log_lambda], lr=lambda_lr)

    for epoch in range(epochs):
        decay_fraction = min(1.0, epoch / (epochs * 0.8)) 
        current_ent_coef = ent_coef_init - decay_fraction * (ent_coef_init - ent_coef_final)
        
        y_list, l_list = [], []
        for i in range(batch_size):
            seed = epoch * batch_size + i
            y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=seed, pure_grid=True, **safe_config)
            l_path = DepositBetaLiabilityGenerator.generate(y_path, noise_std=0.0, seed=seed)
            l_path[T+1:] = 0.0
            l_list.append(l_path)
            y_list.append(y_path)

        yields = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
        liabs = torch.tensor(np.stack(l_list), dtype=torch.float32, device=device)

        res = batched_cmdp_rollout(actor, critic, yields, liabs, T, bond_maturities, bond_coupon_dates, initial_cash=W0, deterministic=False)

        norm_nav = (res["true_nav"] / W0).unsqueeze(-1)
        norm_penalties = res["penalties"] / max_F
        v_preds = res["v_preds"]
        log_probs = res["log_probs"]
        entropies = res["entropies"]
        current_lambda = log_lambda.exp().detach()

        penalties_to_go = torch.flip(torch.cumsum(torch.flip(norm_penalties, dims=[1]), dim=1), dims=[1])
        G_t = norm_nav - (current_lambda * penalties_to_go)
        advantage = G_t - v_preds.detach()
        
        # Advantage Normalization per step
        alive_mask = res["is_alive_seq"]
        valid_elements = alive_mask.sum() + 1e-8
        valid_per_step = alive_mask.sum(dim=0, keepdim=True) + 1e-8
        
        masked_adv = advantage * alive_mask
        adv_mean_per_step = masked_adv.sum(dim=0, keepdim=True) / valid_per_step
        adv_var_per_step = (((advantage - adv_mean_per_step)**2) * alive_mask).sum(dim=0, keepdim=True) / valid_per_step
        adv_std_per_step = torch.sqrt(adv_var_per_step) + 1e-8
        
        scaled_adv = ((advantage - adv_mean_per_step) / adv_std_per_step) * alive_mask

        # Calculate standard loss without splitting
        actor_loss = -(scaled_adv * log_probs).sum() / valid_elements - current_ent_coef * entropies.sum() / valid_elements
        v_loss_unreduced = F_nn.mse_loss(v_preds, G_t.detach(), reduction='none')
        v_loss = (v_loss_unreduced * alive_mask).sum() / valid_elements
        
        total_penalties_per_episode = norm_penalties.sum(dim=1).mean().detach()
        penalty_violation = total_penalties_per_episode - penalty_limit / max_F
        lambda_loss = -log_lambda.exp() * penalty_violation

        # Reset gradients at each step
        actor_optim.zero_grad()
        critic_optim.zero_grad()
        lambda_optim.zero_grad()

        # Critic Optimization
        v_loss.backward()
        critic_optim.step()
        
        # Actor and Multiplier Optimization (after warmup)
        if epoch >= critic_warmup:
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_optim.step()
            
            lambda_loss.backward()
            lambda_optim.step()

        with torch.no_grad():
            log_lambda.clamp_(min=math.log(0.1), max=math.log(30.0))

        if epoch % log_every == 0:
            mean_hard_pen = res['penalties'].sum(dim=1).mean().item()
            print(f"Epoch {epoch:4d} | ActLoss: {actor_loss.item():8.4f} | VLoss: {v_loss.item():.4f} | NAV: {res['true_nav'].mean().item():.2f} | Lambda: {log_lambda.exp().item():.3f} | HardPen: {mean_hard_pen:.4f}")

    return actor, critic


# Evaluation pipeline

def evaluate_cmdp_alm(actor: nn.Module, critic: nn.Module, markov_config: dict, eval_episodes: int = 1000, device: str = "cpu") -> dict:
    actor.eval()
    critic.eval()
    T = markov_config["T"]
    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    bond_coupon_dates = [Bond(**cfg).coupon_dates for cfg in markov_config["bond_configs"]]
    safe_config = {k: markov_config[k] for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"] if k in markov_config}
    W0 = markov_config["W0"]

    y_list, l_list = [], []
    for i in range(eval_episodes):
        seed = 1000000 + i  
        y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=seed, pure_grid=True, **safe_config)
        l_path = DepositBetaLiabilityGenerator.generate(y_path, noise_std=0.0, seed=seed)
        l_path[T+1:] = 0.0
        l_list.append(l_path)
        y_list.append(y_path)

    yields = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    liabs = torch.tensor(np.stack(l_list), dtype=torch.float32, device=device)

    with torch.no_grad():
        res = batched_cmdp_rollout(actor, critic, yields, liabs, T, bond_maturities, bond_coupon_dates, initial_cash=W0, deterministic=True)

    true_nav = res["true_nav"].cpu().numpy()
    hard_penalties = res["penalties"].sum(dim=1).cpu().numpy() 
    var_05 = np.percentile(true_nav, 5)

    return {
        "mean_nav":       float(np.mean(true_nav)),
        "std_nav":        float(np.std(true_nav)),
        "var_05":         float(var_05),
        "violation_rate": float(np.mean(hard_penalties > 0.0)),
    }


# Execution engine

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Ablation Study on device: {device.upper()}\n")

    state_dim = 29   
    action_dim = 6   
    T_horizon = markov_config["T"]

    # Structure: "Model_Name": (Actor_Class, Critic_Class, Custom_Train_Kwargs_Dict)
    ablation_registry = {
        "M1_Proposed_Full": (
            lambda: HybridActor(state_dim, action_dim, T_horizon), 
            lambda: ValueCritic(state_dim, T_horizon),
            {} # Default parameters
        ),
        "Ablation_A_Gaussian": (
            lambda: HybridActorGaussian(state_dim, action_dim, T_horizon), 
            lambda: ValueCritic(state_dim, T_horizon),
            {}
        ),
        "Ablation_B_NoTime": (
            lambda: HybridActorNoTime(state_dim, action_dim), 
            lambda: ValueCriticNoTime(state_dim),
            {}
        ),
        "Ablation_C_NoPrevAction": (
            lambda: HybridActorNoPrevAction(state_dim, action_dim, T_horizon), 
            lambda: ValueCritic(state_dim, T_horizon),
            {}
        ),
        "Ablation_D_NoTimeNoAction": (
            lambda: HybridActorNoTimeNoPrevAction(state_dim, action_dim), 
            lambda: ValueCriticNoTime(state_dim),
            {}
        ),
        "Ablation_E_NoEntropy": (
            lambda: HybridActor(state_dim, action_dim, T_horizon), 
            lambda: ValueCritic(state_dim, T_horizon),
            {"ent_coef_init": 0.0, "ent_coef_final": 0.0} # Kill the entropy term
        )
    }

    global_summary = {}

    for name, model_builders in ablation_registry.items():
        print(f"\n" + "="*70)
        print(f" STARTING TRAINING: {name.upper()}")
        print("="*70)
        
        actor = model_builders[0]()
        critic = model_builders[1]()
        train_kwargs = model_builders[2] 
        
        trained_actor, trained_critic = train_cmdp_alm(
            actor, critic, markov_config, 
            epochs=400,          # Remember to drop to ~200 and warmup ~25 for quick runs!
            batch_size=2048,
            device=device,
            log_every=10,
            critic_warmup=0,
            lambda_lr=0.0,
            lmbda=5.0,
            **train_kwargs
        )
        
        print(f"--> Out-of-Sample Validation for {name}...")
        metrics = evaluate_cmdp_alm(trained_actor, trained_critic, markov_config, eval_episodes=2000, device=device)
        global_summary[name] = metrics

    print("\n\n" + "#"*80)
    print("                ABLATION STUDY COMPARATIVE TABLE")
    print("#"*80)
    print(f"{'Model Configuration':<30} | {'Mean NAV':<12} | {'Std Dev':<10} | {'VaR (5%)':<10} | {'Viol. Rate':<10}")
    print("-" * 80)
    for model_name, data in global_summary.items():
        print(f"{model_name:<30} | {data['mean_nav']:12.2f} | {data['std_nav']:10.2f} | {data['var_05']:10.2f} | {data['violation_rate']*100:8.2f}%")
    print("#"*80)