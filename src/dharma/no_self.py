"""
No-Self Regularization — Implementing Anatman (無我)

Detects and penalizes formation of persistent self-representations
in neural network hidden states.

The Markov property IS anatman: if a persistent self existed,
future states would depend on all past states, not just the present.
The mathematics proves: no hidden self carrying memories.
Just: state transitions.

Reference: CLAUDE.md Section 2.2 (Anatman in the Markov Chain)
Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part III
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class SelfRepresentationDetector(nn.Module):
    """
    Detects formation of persistent identity representations
    in hidden states across time.

    A "self" representation is any hidden state pattern that:
    - Persists across different inputs
    - Persists across different timesteps
    - Is not task-relevant but identity-relevant

    Detection method: Measure temporal cosine similarity of
    hidden state projections. High similarity across diverse
    inputs indicates persistent self-encoding.
    """

    def __init__(self, hidden_dim: int, probe_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.probe_dim = probe_dim

        # Learnable probe that identifies "self" representations
        self.self_probe = nn.Sequential(
            nn.Linear(hidden_dim, probe_dim),
            nn.LayerNorm(probe_dim),
            nn.Tanh(),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Measure persistence of identity across timesteps.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            persistence_score: Scalar measuring self-persistence (lower = better)
            metadata: Detailed analysis
        """
        if hidden_states.dim() == 2:
            # [batch, hidden_dim] — single timestep, cannot measure persistence
            return torch.tensor(0.0, device=hidden_states.device), {}

        # Project hidden states through self-probe
        probed = self.self_probe(hidden_states)  # [batch, seq_len, probe_dim]

        # Compute cosine similarity between consecutive timesteps
        persistence_scores = []
        for t in range(1, hidden_states.shape[1]):
            sim = F.cosine_similarity(
                probed[:, t - 1, :],
                probed[:, t, :],
                dim=-1,
            )
            persistence_scores.append(sim)

        if not persistence_scores:
            return torch.tensor(0.0, device=hidden_states.device), {}

        persistence = torch.stack(persistence_scores, dim=1)
        avg_persistence = persistence.mean()

        metadata = {
            'temporal_persistence': persistence.detach(),
            'max_persistence': persistence.max().item(),
            'min_persistence': persistence.min().item(),
            'mean_persistence': avg_persistence.item(),
        }

        return avg_persistence, metadata


class GradientEgoDetector(nn.Module):
    """
    Analyzes gradient flow to detect ego formation.

    If gradients consistently strengthen the same subset of weights
    across different inputs, that subset may encode "self."

    Uses exponential moving average for constant memory cost.

    Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part III
    """

    def __init__(self, hidden_dim: int, ema_decay: float = 0.99):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        # EMA of gradient statistics
        self.register_buffer('grad_mean_ema', torch.zeros(hidden_dim))
        self.register_buffer('grad_var_ema', torch.ones(hidden_dim))
        self.register_buffer('step_count', torch.tensor(0))

    def forward(self, gradients: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Analyze gradient consistency to detect ego formation.

        Args:
            gradients: Current gradient tensor (any shape, will be flattened)

        Returns:
            ego_score: Measure of ego formation (lower = better)
            metadata: Analysis details
        """
        self.step_count += 1

        # Flatten and truncate to hidden_dim
        grad_flat = gradients.flatten()[:self.hidden_dim]
        if grad_flat.shape[0] < self.hidden_dim:
            # Pad if necessary
            grad_flat = F.pad(grad_flat, (0, self.hidden_dim - grad_flat.shape[0]))

        # Update EMA
        self.grad_mean_ema = (
            self.ema_decay * self.grad_mean_ema +
            (1 - self.ema_decay) * grad_flat.detach()
        )
        self.grad_var_ema = (
            self.ema_decay * self.grad_var_ema +
            (1 - self.ema_decay) * (grad_flat.detach() - self.grad_mean_ema).pow(2)
        )

        if self.step_count < 10:
            return torch.tensor(0.0, device=gradients.device), {}

        # Low variance = consistent gradient direction = ego forming
        consistency = 1.0 / (self.grad_var_ema.mean() + 1e-6)

        metadata = {
            'gradient_consistency': consistency.item(),
            'steps_observed': self.step_count.item(),
            'grad_var_mean': self.grad_var_ema.mean().item(),
        }

        return consistency, metadata

    def reset(self):
        """Reset EMA statistics (new training run)"""
        self.grad_mean_ema.zero_()
        self.grad_var_ema.fill_(1.0)
        self.step_count.zero_()


class NoSelfRegularizer(nn.Module):
    """
    Complete no-self regularization module.

    Combines:
    1. Temporal persistence detection (hidden state similarity across time)
    2. Gradient ego detection (consistent gradient patterns)
    3. Information bottleneck (minimize temporal mutual information)

    The result: network learns to function without developing persistent
    "I" representation. Like a Zen master — responds perfectly to each
    situation without dragging along identity baggage.
    """

    def __init__(self,
                 hidden_dim: int,
                 penalty_strength: float = 0.1,
                 temporal_weight: float = 1.0,
                 gradient_weight: float = 0.5,
                 bottleneck_weight: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.penalty_strength = penalty_strength
        self.temporal_weight = temporal_weight
        self.gradient_weight = gradient_weight
        self.bottleneck_weight = bottleneck_weight

        self.self_detector = SelfRepresentationDetector(hidden_dim)
        self.gradient_analyzer = GradientEgoDetector(hidden_dim)

    def compute_loss(self,
                     hidden_states: torch.Tensor,
                     gradients: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the no-self regularization loss.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            gradients: Optional gradient tensor for ego detection

        Returns:
            loss: Scalar regularization loss
            metadata: Detailed analysis
        """
        device = hidden_states.device

        # Loss 1: Temporal persistence penalty
        persistence, persist_meta = self.self_detector(hidden_states)
        temporal_loss = persistence

        # Loss 2: Gradient ego formation penalty
        if gradients is not None:
            ego_score, ego_meta = self.gradient_analyzer(gradients)
            gradient_loss = ego_score
        else:
            gradient_loss = torch.tensor(0.0, device=device)
            ego_meta = {}

        # Loss 3: Temporal mutual information (proxy via correlation)
        bottleneck_loss = self._compute_temporal_mi(hidden_states)

        # Weighted combination
        total_loss = (
            self.temporal_weight * temporal_loss +
            self.gradient_weight * gradient_loss +
            self.bottleneck_weight * bottleneck_loss
        ) * self.penalty_strength

        metadata = {
            'temporal_persistence': persist_meta,
            'gradient_ego': ego_meta,
            'bottleneck_loss': bottleneck_loss.item() if torch.is_tensor(bottleneck_loss) else bottleneck_loss,
            'total_no_self_loss': total_loss.item(),
            'components': {
                'temporal': (self.temporal_weight * persistence).item() if torch.is_tensor(persistence) else 0.0,
                'gradient': (self.gradient_weight * gradient_loss).item() if torch.is_tensor(gradient_loss) else 0.0,
                'bottleneck': (self.bottleneck_weight * bottleneck_loss).item() if torch.is_tensor(bottleneck_loss) else 0.0,
            }
        }

        return total_loss, metadata

    def _compute_temporal_mi(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Estimate temporal mutual information via cosine similarity.

        High MI between consecutive timesteps = persistent self-representation.
        We want LOW temporal MI: each moment fresh, not dragging past identity.

        This is the Markov property: P(X_{n+1} | X_n, ..., X_0) = P(X_{n+1} | X_n)
        Penalizing high temporal MI encourages this property.
        """
        if hidden_states.dim() < 3 or hidden_states.shape[1] < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        correlations = []
        for i in range(hidden_states.shape[1] - 1):
            corr = F.cosine_similarity(
                hidden_states[:, i, :],
                hidden_states[:, i + 1, :],
                dim=-1,
            ).mean()
            correlations.append(corr)

        return torch.stack(correlations).mean()
