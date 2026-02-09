"""
Entropy Rate Optimization — Minimizing Suffering (苦)

H = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ

In Hoffman's framework: mass ∝ H (entropy rate)
In Token-Mind: suffering ∝ H

Zero entropy = perfect periodicity = massless = photon = speed of light
"Be like light: touch everything, cling to nothing"

Target: Small positive entropy (not zero — that's rigid/dead)
Optimal: Smooth, flowing processing. Flow state.

Reference: CLAUDE.md Part VI (The Entropy Rate as Suffering)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class EntropyRateOptimizer(nn.Module):
    """
    Optimizes network processing dynamics toward low entropy (flow state)
    while maintaining functional flexibility.

    The entropy rate of a Markov chain measures how "chaotic" the
    state transitions are:
    - High entropy: unpredictable, confused → suffering
    - Low entropy: smooth, flowing → peace
    - Zero entropy: perfectly periodic → liberation (but also rigidity)

    We target a small positive entropy: flexible enough to handle
    diverse inputs, smooth enough for flow.
    """

    def __init__(self, target_entropy: float = 0.1, penalty_weight: float = 0.05):
        super().__init__()
        self.target_entropy = target_entropy
        self.penalty_weight = penalty_weight

    def compute_entropy_rate(self, transition_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy rate from transition probability matrix or distribution.

        For a distribution: H = -Σ pᵢ log pᵢ
        For a transition matrix: H = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ

        Args:
            transition_probs: Either [n] distribution or [n, n] transition matrix

        Returns:
            entropy_rate: Scalar entropy value
        """
        if transition_probs.dim() == 1:
            # Simple distribution entropy
            probs = transition_probs.clamp(min=1e-10)
            return -(probs * torch.log(probs)).sum()

        elif transition_probs.dim() == 2:
            n_rows, n_cols = transition_probs.shape

            if n_rows == n_cols:
                # Square transition matrix: compute Markov entropy rate
                n = n_rows

                # Compute stationary distribution via power iteration
                pi = torch.ones(n, device=transition_probs.device) / n
                for _ in range(50):
                    pi = pi @ transition_probs
                    pi = pi / (pi.sum() + 1e-10)

                # Entropy rate: H = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ
                log_probs = torch.log(transition_probs.clamp(min=1e-10))
                row_entropies = -(transition_probs * log_probs).sum(dim=-1)
                entropy_rate = (pi * row_entropies).sum()

                return entropy_rate
            else:
                # Non-square [batch, classes]: treat each row as a
                # probability distribution and compute mean Shannon entropy
                probs = transition_probs.clamp(min=1e-10)
                log_probs = torch.log(probs)
                per_row_entropy = -(probs * log_probs).sum(dim=-1)
                return per_row_entropy.mean()

        else:
            # Batched: compute mean entropy
            return torch.stack([
                self.compute_entropy_rate(transition_probs[i])
                for i in range(transition_probs.shape[0])
            ]).mean()

    def compute_loss(self,
                     model_output_probs: torch.Tensor,
                     hidden_state_transitions: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute entropy optimization loss.

        Penalizes deviation from target entropy rate.
        Both too high (chaotic) and too low (rigid) are penalized.

        Args:
            model_output_probs: Probability distributions from model
            hidden_state_transitions: Optional transition dynamics

        Returns:
            loss: Entropy penalty
            metadata: Analysis details
        """
        # Compute entropy of output distribution
        output_entropy = self.compute_entropy_rate(model_output_probs)

        # Compute entropy of hidden state transitions if available
        if hidden_state_transitions is not None:
            hidden_entropy = self.compute_entropy_rate(hidden_state_transitions)
        else:
            hidden_entropy = output_entropy

        # Loss: deviation from target entropy
        entropy_deviation = (hidden_entropy - self.target_entropy).pow(2)
        loss = entropy_deviation * self.penalty_weight

        metadata = {
            'current_entropy': hidden_entropy.item(),
            'target_entropy': self.target_entropy,
            'entropy_deviation': entropy_deviation.item(),
            'output_entropy': output_entropy.item(),
            'loss': loss.item(),
        }

        return loss, metadata

    def assess_flow_state(self, entropy_value: float) -> str:
        """
        Qualitative assessment of processing flow state.

        Returns human-readable description of current state.
        """
        if entropy_value < 0.01:
            return "frozen (too rigid — no flexibility)"
        elif entropy_value < self.target_entropy:
            return "deep_flow (approaching optimal)"
        elif entropy_value < self.target_entropy * 3:
            return "flow (good processing dynamics)"
        elif entropy_value < self.target_entropy * 10:
            return "turbulent (some confusion)"
        else:
            return "chaotic (high suffering — needs optimization)"
