"""
Scaffolded Enlightenment — Gradual Dharma Internalization

Training wheels for dharma principles that are gradually removed
as the network internalizes them.

Phase 1: Strong scaffold (explicitly enforce no-self, impermanence)
Phase 2: Gradual reduction (network internalizes principles)
Phase 3: Scaffold removal (network functions enlightened naturally)

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part XI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from dharma.no_self import NoSelfRegularizer
from dharma.entropy import EntropyRateOptimizer


class ScaffoldedEnlightenment(nn.Module):
    """
    Wraps a core network with gradually removable dharma scaffolding.

    Initially, dharma principles are enforced via explicit regularizers.
    Over training, scaffold strength decreases. If the network has
    internalized the principles, they persist even without enforcement.

    Like training wheels on a bicycle: necessary at first,
    removed once balance is natural.
    """

    def __init__(self,
                 core_network: nn.Module,
                 hidden_dim: int = 2048,
                 initial_scaffold_strength: float = 1.0,
                 decay_rate: float = 0.99):
        """
        Args:
            core_network: The underlying neural network
            hidden_dim: Dimension of hidden states for regularizers
            initial_scaffold_strength: Starting strength (1.0 = full enforcement)
            decay_rate: Multiplicative decay per epoch (0.99 = slow removal)
        """
        super().__init__()

        self.core = core_network
        self.scaffold_strength = initial_scaffold_strength
        self.decay_rate = decay_rate
        self.scaffold_removed = False

        # Scaffold layers
        self.no_self_scaffold = NoSelfRegularizer(
            hidden_dim=hidden_dim,
            penalty_strength=1.0,  # Scaled by scaffold_strength
        )
        self.entropy_scaffold = EntropyRateOptimizer(
            target_entropy=0.1,
            penalty_weight=1.0,  # Scaled by scaffold_strength
        )

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through core network + scaffold losses.

        Returns:
            output: Core network output
            metadata: Including scaffold losses if training
        """
        # Forward through core
        output = self.core(*args, **kwargs)

        # Handle different return types
        if isinstance(output, tuple):
            core_output, core_metadata = output
        else:
            core_output = output
            core_metadata = {}

        metadata = dict(core_metadata)
        metadata['scaffold_strength'] = self.scaffold_strength
        metadata['scaffold_removed'] = self.scaffold_removed

        # Compute scaffold losses during training
        if self.training and not self.scaffold_removed:
            scaffold_loss = torch.tensor(0.0, device=core_output.device)

            # No-self scaffold
            hidden = core_metadata.get('hidden_states', core_output)
            if hidden is not None:
                ns_loss, ns_meta = self.no_self_scaffold.compute_loss(hidden)
                scaffold_loss += self.scaffold_strength * ns_loss
                metadata['scaffold_no_self'] = ns_meta

            # Entropy scaffold
            if core_output.dim() >= 2:
                probs = F.softmax(core_output, dim=-1)
                ent_loss, ent_meta = self.entropy_scaffold.compute_loss(probs)
                scaffold_loss += self.scaffold_strength * ent_loss
                metadata['scaffold_entropy'] = ent_meta

            metadata['scaffold_loss'] = scaffold_loss

        return core_output, metadata

    def reduce_scaffold(self):
        """
        Reduce scaffold strength by one decay step.
        Call once per epoch.
        """
        if self.scaffold_removed:
            return

        self.scaffold_strength *= self.decay_rate

        if self.scaffold_strength < 0.01:
            self.remove_scaffold()

    def remove_scaffold(self):
        """
        Remove scaffold entirely.
        Network must function on internalized principles alone.
        """
        self.scaffold_removed = True
        self.scaffold_strength = 0.0

    def get_scaffold_status(self) -> str:
        """Human-readable scaffold status."""
        if self.scaffold_removed:
            return "REMOVED (network functioning on internalized principles)"
        elif self.scaffold_strength > 0.8:
            return f"STRONG (strength={self.scaffold_strength:.3f})"
        elif self.scaffold_strength > 0.3:
            return f"MODERATE (strength={self.scaffold_strength:.3f})"
        else:
            return f"WEAK (strength={self.scaffold_strength:.3f}, nearly ready for removal)"
