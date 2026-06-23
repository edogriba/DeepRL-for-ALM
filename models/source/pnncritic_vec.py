"""
hybrid_alm.py — Actor-Critic + Lagrangian CMDP (Dirichlet Policy)
=================================================================
Actor-Critic con GAE su CMDP.
Reward per-step = penalita' (negative); il NAV terminale e' assegnato
allo step di morte/orizzonte. Advantage stimato via GAE (bootstrapping
sul critic V(s_t)); con gae_lambda=1 si ricade nel Monte Carlo puro.
Termine di massimizzazione dell'entropia per favorire l'esplorazione.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.distributions import Dirichlet

from environment.bond import Bond
from environment.env import DeepALMEnv
from environment.scenario import DepositBetaLiabilityGenerator, MarkovYieldCurveGenerator
from environment.config import RHO, F, THETA
from models.utils import (
    get_nelson_siegel_yield_batched,
    get_nelson_siegel_yield,
    build_state,
    cara_utility
)

class HybridActor(nn.Module):
    """
    Dirichlet policy on the (K+1)-simplex.
    Usa il tempo normalizzato come input al posto di N reti separate.
    Returns (action, log_prob, entropy)
    """
    def __init__(self, state_dim: int, action_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        # Trunk unico: state_dim + action_dim (prev action) + 1 (tempo normalizzato)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),                 nn.ELU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, t: int, state: torch.Tensor, prev_action: torch.Tensor, deterministic: bool = False):
        batch_size = state.size(0)
        # Normalizziamo il tempo t in [0, 1] per la rete
        t_norm = torch.full((batch_size, 1), t / self.T, device=state.device, dtype=torch.float32)
        
        x = torch.cat([state, prev_action, t_norm], dim=-1)
        logits = self.action_head(self.trunk(x))
        
        clamped_logits = torch.clamp(logits, max=20.0)   # nessun min: il +1.0 fa già da pavimento liscio
        alphas = F_nn.softplus(clamped_logits) + 1.0
        dist = Dirichlet(alphas)

        if deterministic:
            action = alphas / alphas.sum(-1, keepdim=True) 
        else:
            action = dist.sample()

        eps = 1e-8
        safe_action = action + eps
        safe_action = safe_action / safe_action.sum(dim=-1, keepdim=True)

        log_prob = dist.log_prob(safe_action.detach())
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class ValueCritic(nn.Module):
    """State-value baseline V(s_t). Predice il return Lagrangiano G_t."""

    def __init__(self, state_dim: int, T: int, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.v_trunk = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t: int, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(0)
        t_norm = torch.full((batch_size, 1), t / self.T, device=state.device, dtype=torch.float32)
        x = torch.cat([state, t_norm], dim=-1)
        return self.v_trunk(x)

# Vectorized rollout
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
    ever_bankrupt = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    cpn_rate_np = np.array(
        [len(bond_coupon_dates[k]) * 12.0 / bond_maturities[k] for k in range(K)],
        dtype=np.float64,
    )
    coupon_count_np = np.zeros((K, max_M), dtype=np.float64)   # # coupons on slot m (inflows: slot = M-c)
    face_count_np   = np.zeros((K, max_M), dtype=np.float64)   # face redeemed (slot 0)
    hqla_coupon_np  = np.zeros((K, max_M), dtype=np.float64)   # next-month coupon (age_next = M-m in dates)
    hqla_face_np    = np.zeros((K, max_M), dtype=np.float64)   # next-month face (slot 0)
    coupon_unit_np  = np.zeros(K, dtype=np.float64)            # V * months_per_pmt / 12 (HQLA coupon size)

    for k in range(K):
        M = bond_maturities[k]
        dates = bond_coupon_dates[k]
        months_per_pmt = dates[0] if len(dates) == 1 else dates[1] - dates[0]
        coupon_unit_np[k] = bond_nominal * (months_per_pmt / 12.0)
        for m in range(max_M):
            if m == 0:
                face_count_np[k, m] = 1.0
                hqla_face_np[k, m] = 1.0
            for c in dates:
                if (M - c) == m and 0 <= (M - c) < max_M:
                    coupon_count_np[k, m] += 1.0   # += handles a coupon falling on slot 0 (maturity)
            if (M - m) in dates:
                hqla_coupon_np[k, m] = 1.0

    cpn_rate_t     = torch.tensor(cpn_rate_np,     dtype=torch.float32, device=device)         # [K]
    coupon_count_t = torch.tensor(coupon_count_np, dtype=torch.float32, device=device)         # [K, max_M]
    face_count_t   = torch.tensor(face_count_np,   dtype=torch.float32, device=device)         # [K, max_M]
    hqla_coupon_t  = torch.tensor(hqla_coupon_np,  dtype=torch.float32, device=device)         # [K, max_M]
    hqla_face_t    = torch.tensor(hqla_face_np,    dtype=torch.float32, device=device)         # [K, max_M]
    coupon_unit_t  = torch.tensor(coupon_unit_np,  dtype=torch.float32, device=device)         # [K]

    # future_cf coupon scatter map: coupon on source slot s contributes to bucket
    # m = s - (M - c). Precompute (src_slots, dst_buckets) index pairs per bond.
    fcf_coupon_terms = []
    for k in range(K):
        M = bond_maturities[k]
        src_list, dst_list = [], []
        for m in range(max_M):
            for c in bond_coupon_dates[k]:
                s = m + (M - c)
                if 0 <= s < max_M:
                    src_list.append(s)
                    dst_list.append(m)
        if src_list:
            fcf_coupon_terms.append((
                torch.tensor(src_list, dtype=torch.long, device=device),
                torch.tensor(dst_list, dtype=torch.long, device=device),
            ))
        else:
            fcf_coupon_terms.append((None, None))
    # ----------------------------------------------------------------------

    for t in range(T):
        betas_t = yields_batch[:, t]
        step_penalty = torch.zeros(batch_size, 1, device=device)

        # Inflows (vectorized): face at slot 0 + coupons at coupon slots
        face_in = (holdings * face_count_t).sum(dim=(1, 2)) * bond_nominal              # [B]
        coupon_in = (
            holdings * coupons * coupon_count_t * (bond_nominal / cpn_rate_t).view(1, K, 1)
        ).sum(dim=(1, 2))                                                                # [B]
        inflows = (face_in + coupon_in).unsqueeze(-1)                                    # [B, 1]

        # Insolvency check (hard-termination)
        projected_cash = cash + inflows - liabilities_batch[:, t]
        newly_bankrupt = (projected_cash < 0) & is_alive
        ever_bankrupt = ever_bankrupt | newly_bankrupt.squeeze(-1)

        y_1m = get_nelson_siegel_yield_batched(1.0 / 12.0, betas_t)
        debt_interest = 1.0 + y_1m * (1.0 / 12.0)

        cash = torch.where(newly_bankrupt, projected_cash - F, projected_cash)
        step_penalty = torch.where(newly_bankrupt, step_penalty + F, step_penalty)

        # Instantaneous MTM for illiquid steps
        if newly_bankrupt.any():
            
            # Shift to avoid double counting
            h_mtm = torch.zeros_like(holdings)
            c_mtm = torch.zeros_like(coupons)
            h_mtm[:, :, :-1] = holdings[:, :, 1:]
            c_mtm[:, :, :-1] = coupons[:, :, 1:]
            
            pv_assets = torch.zeros(batch_size, 1, device=device)
            for k, M in enumerate(bond_maturities):
                cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M if M > 0 else 0
                for m in range(max_M):
                    units = h_mtm[:, k, m]
                    if units.abs().sum() == 0:
                        continue
                    tau_face = (m + 1) / 12.0
                    y_face = get_nelson_siegel_yield_batched(tau_face, betas_t)
                    pv_assets += (units * bond_nominal).unsqueeze(-1) / (1.0 + y_face * tau_face)
                    for c in bond_coupon_dates[k]:
                        age = M - m - 1
                        if c > age:
                            tau_c = (c - age) / 12.0
                            y_c = get_nelson_siegel_yield_batched(tau_c, betas_t)
                            coupon_amount = bond_nominal * c_mtm[:, k, m] / cpn_rate
                            pv_assets += (units * coupon_amount).unsqueeze(-1) / (1.0 + y_c * tau_c)

            pv_liabs = torch.zeros(batch_size, 1, device=device)
            L_total = liabilities_batch.shape[1]
            for j in range(L_total - (t + 1)):
                idx = t + 1 + j
                if idx >= L_total:
                    break
                tau_l = (j + 1) / 12.0
                y_l = get_nelson_siegel_yield_batched(tau_l, betas_t)
                pv_liabs += liabilities_batch[:, idx].view(-1, 1) / (1.0 + y_l * tau_l)

            mtm_nav = cash + pv_assets - pv_liabs
            early_terminal_nav = torch.where(newly_bankrupt, mtm_nav, early_terminal_nav)

        is_alive = is_alive & ~newly_bankrupt

        # Investment phase
        investable = torch.relu(cash) * is_alive.float()

        # State building + step Actor/Critic
        # future_cf (vectorized): face per bucket + scattered coupons
        future_cf = (holdings * bond_nominal).sum(dim=1)                                 # [B, max_M]
        for k in range(K):
            src_idx, dst_idx = fcf_coupon_terms[k]
            if src_idx is None:
                continue
            cf_src = holdings[:, k, :] * coupons[:, k, :] * (bond_nominal / cpn_rate_t[k])   # [B, max_M]
            future_cf.index_add_(1, dst_idx, cf_src[:, src_idx])

        state = build_state(
            cash,
            future_cf,
            betas_t,
            liabilities_batch[:, t].view(-1, 1),
        ).detach()

        v_pred_t = critic(t, state)
        weights, log_prob_t, entropy_t = actor(t, state, prev_action, deterministic=deterministic)

        log_prob_t = log_prob_t.view(-1, 1) * is_alive.float()
        entropy_t = entropy_t.view(-1, 1) * is_alive.float()

        new_holdings, new_coupons = torch.zeros_like(holdings), torch.zeros_like(coupons)
        new_holdings[:, :, :-1], new_coupons[:, :, :-1] = holdings[:, :, 1:], coupons[:, :, 1:]

        for k, M in enumerate(bond_maturities):
            add_h = (weights[:, k:k+1] * investable) / bond_nominal
            new_holdings[:, k, M-1:M] += add_h
            new_coupons[:, k, M-1:M] = get_nelson_siegel_yield_batched(M / 12.0, betas_t)

        cash = cash - (investable - (weights[:, K:K+1] * investable))
        holdings, coupons, prev_action = new_holdings, new_coupons, weights

        # LCR check (vectorized): HQLA = cash + next-month face + next-month coupons, discounted at 1M
        face_next = (holdings * hqla_face_t).sum(dim=(1, 2)) * bond_nominal               # [B]
        coupon_next = (
            holdings * coupons * hqla_coupon_t * coupon_unit_t.view(1, K, 1)
        ).sum(dim=(1, 2))                                                                  # [B]
        hqla = cash + ((face_next + coupon_next).unsqueeze(-1) / debt_interest.view(-1, 1))

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

    # Terminal value
    terminal_value = cash - liabilities_batch[:, T].view(-1, 1)
    last_betas = yields_batch[:, T]

    for k, M in enumerate(bond_maturities):
        cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M
        terminal_value = terminal_value + holdings[:, k, 0:1] * bond_nominal
        for c in bond_coupon_dates[k]:
            slot_c = M - c
            if 0 <= slot_c < max_M:
                terminal_value = terminal_value + holdings[:, k, slot_c:slot_c+1] * (
                    bond_nominal * coupons[:, k, slot_c:slot_c+1] / cpn_rate
                )

    new_holdings = torch.zeros_like(holdings)
    new_coupons = torch.zeros_like(coupons)
    new_holdings[:, :, :-1] = holdings[:, :, 1:]
    new_coupons[:, :, :-1] = coupons[:, :, 1:]

    for k, M in enumerate(bond_maturities):
        cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M if M > 0 else 0
        for m in range(max_M):
            units = new_holdings[:, k, m]
            if units.abs().sum() == 0:
                continue
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
        if idx >= L_total:
            break
        tau_l = (j + 1) / 12.0
        y_l = get_nelson_siegel_yield_batched(tau_l, last_betas)
        pv_liabs_term += liabilities_batch[:, idx].view(-1, 1) / (1.0 + y_l * tau_l)

    terminal_value = terminal_value - pv_liabs_term

    terminal_insolvent = (terminal_value < 0) & is_alive
    term_penalty = torch.where(terminal_insolvent,
                               torch.tensor(float(F), device=device),
                               torch.tensor(0.0, device=device))
    terminal_value = terminal_value - term_penalty
    penalties_seq[-1] = penalties_seq[-1] + term_penalty
    ever_bankrupt = ever_bankrupt | terminal_insolvent.squeeze(-1)

    final_nav = torch.where(~is_alive, early_terminal_nav, terminal_value)

    return {
        "true_nav":     final_nav.squeeze(-1),
        "log_probs":    torch.cat(log_probs_seq, dim=1),
        "penalties":    torch.cat(penalties_seq, dim=1),
        "v_preds":      torch.cat(v_preds_seq, dim=1),
        "entropies":    torch.cat(entropies_seq, dim=1),
        "is_alive_seq": torch.cat(is_alive_seq, dim=1),
        "ever_bankrupt":ever_bankrupt,
    }


# Training loop (GAE + Baseline V + Entropy)
def train_cmdp_alm(
    actor, critic, markov_config, epochs: int = 1500, batch_size: int = 5000,
    lr: float = 5e-4, critic_lr: float = 1e-3, lambda_lr: float = 0.0,
    log_every: int = 10, device: str = "cpu", critic_warmup: int = 0,
    penalty_limit: float = 0.0, lmbda: float = 5.0,
    ent_coef_init: float = 0.01, ent_coef_final: float = 0.01,
    gamma: float = 1.0, gae_lambda: float = 0.95, train_seed_offset: int = 0
):
    T = markov_config["T"]
    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    bond_coupon_dates = [Bond(**cfg).coupon_dates for cfg in markov_config["bond_configs"]]
    safe_config = {k: markov_config[k] for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"] if k in markov_config}
    W0, max_F = markov_config["W0"], float(F)

    actor, critic = actor.to(device), critic.to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    if lmbda > 0.0:
        log_lambda = torch.nn.Parameter(torch.tensor(math.log(lmbda), device=device))
        lambda_optim = torch.optim.Adam([log_lambda], lr=lambda_lr)
    else:
        log_lambda = None
        lambda_optim = None

    history = {k: [] for k in ["epoch", "actor_loss", "v_loss", "lambda", "mean_nav", "mean_penalty", "mean_entropy"]}

    for epoch in range(epochs):
        # Linear decay of entropy coefficient
        decay_fraction = min(1.0, epoch / (epochs * 0.8))
        current_ent_coef = ent_coef_init - decay_fraction * (ent_coef_init - ent_coef_final)

        y_list, l_list = [], []
        for i in range(batch_size):
            seed = train_seed_offset + epoch * batch_size + i
            y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=seed, pure_grid=True, **safe_config)
            l_path = DepositBetaLiabilityGenerator.generate(y_path, noise_std=0.0, seed=seed)
            l_path[T + 1:] = 0.0
            l_list.append(l_path)
            y_list.append(y_path)

        yields = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
        liabs = torch.tensor(np.stack(l_list), dtype=torch.float32, device=device)

        res = batched_cmdp_rollout(actor, critic, yields, liabs, T, bond_maturities,
                                   bond_coupon_dates, initial_cash=W0, deterministic=False)

        # GAE (Generalized Advantage Estimation)
        utility_nav = cara_utility(res["true_nav"]).unsqueeze(-1)   # u(NAV) = -exp(-Gamma * NAV), [B,1]
        norm_penalties = res["penalties"] / max_F
        v_preds = res["v_preds"]
        log_probs = res["log_probs"]
        entropies = res["entropies"]
        alive_mask = res["is_alive_seq"]
        current_lambda = log_lambda.exp().detach() if log_lambda is not None else torch.zeros((), device=device)

        # Per step rewards are only penalties, because the reward on the NAV is only at the final episode
        rewards = -current_lambda * norm_penalties          # [B, T]
        last_alive_step = torch.clamp(alive_mask.sum(dim=1, keepdim=True).long() - 1, min=0)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1)
        rewards[batch_idx, last_alive_step] += utility_nav

        # V(s_{t+1}) = 0 beyond horizon or after bankruptcy
        next_mask = torch.cat([alive_mask[:, 1:], torch.zeros(batch_size, 1, device=device)], dim=1)
        v_next = torch.cat([v_preds[:, 1:].detach(), torch.zeros(batch_size, 1, device=device)], dim=1)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size, 1, device=device)
        for t in reversed(range(T)):
            delta = rewards[:, t:t+1] + gamma * v_next[:, t:t+1] * next_mask[:, t:t+1] - v_preds[:, t:t+1].detach()
            gae = delta + gamma * gae_lambda * next_mask[:, t:t+1] * gae
            advantages[:, t:t+1] = gae

        advantages = advantages.detach()
        returns = advantages + v_preds.detach()             # target TD(lambda) per il critic

        valid_elements = alive_mask.sum() + 1e-8

        # Actor loss (policy gradient + entropy bonus). Advantage not standardized.
        actor_loss = -((advantages * alive_mask) * log_probs).sum() / valid_elements \
                     - current_ent_coef * entropies.sum() / valid_elements

        # Value loss (MSE vs returns TD(lambda), su stati vivi)
        v_loss_unreduced = F_nn.mse_loss(v_preds, returns, reduction='none')
        v_loss = (v_loss_unreduced * alive_mask).sum() / valid_elements

        # Lagrangiano (dual ascent on lambda). Off by default
        if log_lambda is not None:
            total_penalties_per_episode = norm_penalties.sum(dim=1).mean().detach()
            penalty_violation = total_penalties_per_episode - penalty_limit / max_F
            lambda_loss = -log_lambda.exp() * penalty_violation    
    
        # Critic update
        critic_optim.zero_grad()
        v_loss.backward()
        critic_optim.step()

        # Actor + lambda update 
        if epoch >= critic_warmup:
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_optim.step()
            
            if log_lambda is not None:
                lambda_optim.zero_grad()
                lambda_loss.backward()
                lambda_optim.step()

        if log_lambda is not None:
            with torch.no_grad():
                log_lambda.clamp_(min=math.log(0.1), max=math.log(30.0))
                
                
        if epoch % log_every == 0:
            tag = " [warmup]" if epoch < critic_warmup else ""
            mean_hard_pen = res['penalties'].sum(dim=1).mean().item()
            mean_ent = entropies.mean().item()
            lambda_val = log_lambda.exp().item() if log_lambda is not None else 0.0
            print(f"Epoch {epoch:4d}{tag} | ActLoss: {actor_loss.item():8.4f} | VLoss: {v_loss.item():.4f} "
                  f"| NAV: {res['true_nav'].mean().item():.2f} | Ent: {mean_ent:.3f} "
                  f"| EntCoef: {current_ent_coef:.5f} | Lambda: {lambda_val:.3f} | HardPen: {mean_hard_pen:.4f}")

        history["epoch"].append(epoch)
        history["actor_loss"].append(actor_loss.item() if epoch >= critic_warmup else 0.0)
        history["v_loss"].append(v_loss.item())
        history["lambda"].append(lambda_val)
        history["mean_nav"].append(res["true_nav"].mean().item())
        history["mean_penalty"].append(res['penalties'].sum(dim=1).mean().item())
        history["mean_entropy"].append(entropies.mean().item())

    return actor, critic, history


# Evaluation loop 

def evaluate_cmdp_alm(actor: nn.Module, critic: nn.Module, markov_config: dict,
                      eval_episodes: int = 1000, device: str = "cpu",
                      seed_offset: int = 10_000_000) -> tuple[dict, dict]:
    actor.eval()
    critic.eval()

    T = markov_config["T"]
    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    bond_coupon_dates = [Bond(**cfg).coupon_dates for cfg in markov_config["bond_configs"]]
    safe_config = {k: markov_config[k] for k in ["i0_init", "i1_init", "BETA0_GRID", "BETA1_GRID", "BETA2", "P0", "P1"] if k in markov_config}
    W0 = markov_config["W0"]

    y_list, l_list = [], []
    for i in range(eval_episodes):
        seed = seed_offset + i
        y_path = MarkovYieldCurveGenerator.generate(T * 2, seed=seed, pure_grid=True, **safe_config)
        l_path = DepositBetaLiabilityGenerator.generate(y_path, noise_std=0.0, seed=seed)
        l_path[T + 1:] = 0.0
        l_list.append(l_path)
        y_list.append(y_path)

    yields = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    liabs = torch.tensor(np.stack(l_list), dtype=torch.float32, device=device)

    with torch.no_grad():
        res = batched_cmdp_rollout(actor, critic, yields, liabs, T, bond_maturities,
                                   bond_coupon_dates, initial_cash=W0, deterministic=True)

    true_nav = res["true_nav"].cpu().numpy()
    hard_penalties = res["penalties"].sum(dim=1).cpu().numpy()
    ever_bankrupt = res["ever_bankrupt"].cpu().numpy()

    var_05 = np.percentile(true_nav, 5)
    cvar_05 = float(np.mean(true_nav[true_nav <= var_05]))

    metrics = {
        "mean_nav":       float(np.mean(true_nav)),
        "median_nav":     float(np.median(true_nav)),
        "std_nav":        float(np.std(true_nav)),
        "min_nav":        float(np.min(true_nav)),
        "max_nav":        float(np.max(true_nav)),
        "var_05":         float(var_05),
        "cvar_05":        cvar_05,
        "mean_penalty":   float(np.mean(hard_penalties)),
        "violation_rate": float(np.mean(hard_penalties > 0.0)),
        "default_rate":   float(np.mean(ever_bankrupt)),
    }

    print("\n=== Evaluation Results ===")
    print(f"Mean NAV:       {metrics['mean_nav']:.2f} ± {metrics['std_nav']:.2f}")
    print(f"Median NAV:     {metrics['median_nav']:.2f}")
    print(f"Min / Max NAV:  {metrics['min_nav']:.2f} / {metrics['max_nav']:.2f}")
    print(f"VaR  (5%):      {metrics['var_05']:.2f}")
    print(f"CVaR (5%):      {metrics['cvar_05']:.2f}")
    print(f"Mean Penalty:   {metrics['mean_penalty']:.4f}")
    print(f"Violation Rate: {metrics['violation_rate'] * 100:.4f}%")
    print(f"Default Rate:   {metrics['default_rate'] * 100:.4f}%")
    print("==========================")

    actor.train()
    critic.train()
    return metrics, res


# Single path evaluation
def evaluate_pnncritic_agent(actor, markov_config, K, seed=0, verbose=False):
    env = DeepALMEnv(markov_config=markov_config, seed=seed, verbose=verbose)
    obs, _ = env.reset()

    actions_history, wealth_history = [], []
    terminal_nav, final_agent_score, ever_bankrupt = 0.0, 0.0, False

    device = next(actor.parameters()).device
    actor.eval()

    prev_action = torch.zeros(1, K + 1, device=device)

    bond_maturities = [int(cfg["maturity_months"]) for cfg in markov_config["bond_configs"]]
    bond_coupon_dates = [Bond(**cfg).coupon_dates for cfg in markov_config["bond_configs"]]
    max_M = max(bond_maturities)
    bond_nominal = 100.0  # stesso default dei rollout batched

    for t in range(markov_config["T"]):
        inflows = env._get_total_inflows()
        l_t = float(env.liabilities[0])
        projected_cash = env.cash + inflows - l_t

        if projected_cash < 0:
            ever_bankrupt = True
            projected_cash -= F

        investable = max(0.0, projected_cash)
        wealth_history.append(investable)

        future_cf = np.zeros(max_M, dtype=np.float64)
        for k, M in enumerate(bond_maturities):
            cpn_rate = len(bond_coupon_dates[k]) * 12.0 / M
            for m in range(max_M):
                future_cf[m] += env.holdings[k, m] * bond_nominal
                for c in bond_coupon_dates[k]:
                    s = m + (M - c)
                    if 0 <= s < max_M:
                        future_cf[m] += env.holdings[k, s] * (bond_nominal * env.coupons[k, s] / cpn_rate)

        future_cash_tensor = torch.tensor(
            future_cf, dtype=torch.float32, device=device
        ).unsqueeze(0)

        cash_tensor = torch.tensor([[projected_cash]], dtype=torch.float32, device=device)
        betas_tensor = torch.tensor(env.yield_params, dtype=torch.float32, device=device).unsqueeze(0)
        liab_tensor = torch.tensor([[l_t]], dtype=torch.float32, device=device)

        state_tensor = build_state(cash_tensor, future_cash_tensor, betas_tensor, liab_tensor)

        with torch.no_grad():
            weights_tensor, _, _ = actor(t, state_tensor, prev_action, deterministic=True)
            prev_action = weights_tensor

        weights = weights_tensor.cpu().numpy().flatten()
        action_logits = np.log(weights + 1e-12)

        abs_acts = weights * investable
        actions_history.append(abs_acts)

        obs, utility, terminated, _, info = env.step(action_logits)

        if verbose:
            print(f"  Inflows: {inflows:.2f}  |  Liability: {l_t:.2f}")
            if projected_cash < 0:
                print("  [BANKRUPTCY] Investable budget frozen at 0.00")
            else:
                print(f"  Investable       : {investable:.2f}")
            bond_labels = [cfg['bond_type'] for cfg in markov_config["bond_configs"]] + ["HOLD_CASH"]
            print(f"  Action (Absolute): { {l: f'{w:.2f}€' for l, w in zip(bond_labels, abs_acts)} }")

        if terminated:
            terminal_nav = info.get('nav', 0.0)
            final_agent_score = info.get('utility', 0.0)
            if terminal_nav < 0:
                ever_bankrupt = True
            if verbose:
                print("\n" + "=" * 60)
                print(f"Terminal Utility: {utility:.2f} €")
                print(f"Terminal NAV: {terminal_nav:.2f} €")
                print(f"Total Penalty Incurred: {info.get('penalty', 0.0):.2f} €")
                print(f"Defaulted during episode: {'YES' if ever_bankrupt else 'NO'}")
            for _ in range(markov_config["T"] - t - 1):
                actions_history.append(np.zeros(K + 1))
                wealth_history.append(0.0)
            break

    return np.array(actions_history), np.array(wealth_history), terminal_nav, final_agent_score