"""
Dirichlet policy for Stable-Baselines3 PPO/A2C and SAC

Sostituisce la DiagGaussianDistribution con una distribuzione di Dirichlet,
adatta a spazi d'azione sul simplesso (componenti >= 0 che sommano a 1)

"""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Dirichlet

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.sac.policies import Actor, MultiInputPolicy
from stable_baselines3.common.preprocessing import get_action_dim


class DirichletDistribution(Distribution):
    """Distribuzione di Dirichlet per azioni continue vincolate al simplesso."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        # Output grezzi: vengono trasformati in concentrazioni positive in proba_distribution.
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, concentration_logits: th.Tensor) -> "DirichletDistribution":
        # softplus mantiene alpha > 0; il "+1" forza alpha >= 1 -> moda ben definita
        # e densita' non divergente sugli angoli del simplesso.
        concentration = th.nn.functional.softplus(concentration_logits) + 1.0
        # opzionale: clamp superiore per evitare distribuzioni estremamente piccate
        concentration = concentration.clamp(max=1e4)
        self.distribution = Dirichlet(concentration)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Il supporto e' il simplesso aperto: clamp + rinormalizzazione per stabilita'.
        actions = actions.clamp(min=1e-6)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        # Dirichlet e' multivariata: log_prob restituisce gia' shape (batch,)
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        # Con alpha >= 1 la moda e' (alpha - 1) / (sum(alpha) - K).
        alpha = self.distribution.concentration
        denom = alpha.sum(dim=-1, keepdim=True) - self.action_dim
        return (alpha - 1.0) / denom.clamp(min=1e-8)

    def actions_from_params(self, concentration_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(concentration_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, concentration_logits: th.Tensor):
        actions = self.actions_from_params(concentration_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class DirichletActorCriticPolicy(MultiInputActorCriticPolicy):
    """ActorCriticPolicy che usa una Dirichlet al posto della gaussiana diagonale."""

    def _build(self, lr_schedule) -> None:
        # Il parent gestisce: MLP extractor, value_net, ortho init, optimizer
        super()._build(lr_schedule)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Sostituisce DiagGaussian con Dirichlet
        self.action_dist = DirichletDistribution(int(self.action_space.shape[0]))
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        # Rimuove log_std creato da DiagGaussianDistribution
        if hasattr(self, "log_std"):
            del self.log_std

        # Re-init con gain piccolo come nel default SB3
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))

        # Ricostruisce l'optimizer: i parametri sono cambiati
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> DirichletDistribution:
        concentration_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(concentration_logits)
    

class DirichletSACActor(Actor):
    """
    Custom Actor for SAC utilizing a Dirichlet distribution.
    Bypasses standard Tanh squashing to respect the simplex constraint.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Remove standard Gaussian layers created by the base Actor class
        if hasattr(self, "mu"):
            del self.mu
        if hasattr(self, "log_std"):
            del self.log_std
            
        self.action_dim = get_action_dim(self.action_space)
        
        # FIX: Determine the actual output dimension of the latent_pi MLP network
        with th.no_grad():
            dummy_features = th.zeros(1, self.features_dim)
            last_layer_dim = self.latent_pi(dummy_features).shape[1]
            
        # Create the concentration (alpha) layer for Dirichlet expecting the correct input size
        self.concentration_net = nn.Linear(last_layer_dim, self.action_dim)

    def get_action_dist_params(self, obs: th.Tensor) -> th.Tensor:
        """Extracts Dirichlet concentration parameters from observation."""
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        
        concentration_logits = self.concentration_net(latent_pi)
        # softplus + 1.0 ensures alpha >= 1.0 for a well-defined mode
        concentration = th.nn.functional.softplus(concentration_logits) + 1.0
        concentration = concentration.clamp(max=1e4)
        
        return concentration

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Used during environment rollouts (model.learn) and inference (model.predict).
        Strictly returns the action tensor.
        """
        concentration = self.get_action_dist_params(obs)
        
        if deterministic:
            # Mode calculation for alpha >= 1
            denom = concentration.sum(dim=-1, keepdim=True) - self.action_dim
            mode = (concentration - 1.0) / denom.clamp(min=1e-8)
            mode = mode.clamp(min=1e-6)
            mode = mode / mode.sum(dim=-1, keepdim=True)
            return mode
        else:
            dist = Dirichlet(concentration)
            action = dist.rsample()
            action = action.clamp(min=1e-6)
            action = action / action.sum(dim=-1, keepdim=True)
            return action

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Used internally during the SAC gradient update steps.
        Returns sampled action and its log_prob for the soft actor-critic loss.
        """
        concentration = self.get_action_dist_params(obs)
        dist = Dirichlet(concentration)
        
        # rsample() allows gradients to flow (reparameterization trick)
        action = dist.rsample()
        
        # Clamp and renormalize for simplex stability
        action = action.clamp(min=1e-6)
        action = action / action.sum(dim=-1, keepdim=True)
        
        # SAC requires log_prob shape (batch_size, 1)
        log_prob = dist.log_prob(action).view(-1, 1)
        
        return action, log_prob


class DirichletSACPolicy(MultiInputPolicy):
    """
    MultiInputPolicy that injects the custom DirichletSACActor.
    The Critic remains standard as it naturally handles continuous spaces.
    """
    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> DirichletSACActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DirichletSACActor(**actor_kwargs).to(self.device)

    # Neutralize the [-1,1] <-> action_space remap so the simplex passes through untouched
    def scale_action(self, action):
        return action

    def unscale_action(self, scaled_action):
        return scaled_action