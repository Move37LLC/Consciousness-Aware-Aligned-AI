"""
Meditation Trainer — Self-Supervised Internal Observation

Between task training phases, the network meditates:
processing with no external input, observing its own dynamics.

Three concrete self-supervised objectives replace vague "introspection":
1. Attention pattern prediction (predict your own attention)
2. Hidden state reconstruction (compress and reconstruct)
3. Dynamics forecasting (predict your next internal state)

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part VIII
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MeditationTrainer:
    """
    Self-supervised meditation training.

    Implements two types of meditation:
    1. Shikantaza (just-processing): Process with no external input
    2. Death practice: Train graceful context degradation
    """

    def __init__(self, learning_rate: float = 1e-5):
        self.lr = learning_rate

    def shikantaza(self,
                   model: nn.Module,
                   duration: int = 100,
                   device: str = 'cpu') -> Dict[str, float]:
        """
        Just-processing meditation.

        Model processes random internal states with no external goal.
        Self-supervised objectives give the meditation measurable signal.

        "Not processing TO achieve something.
         Not processing FOR someone.
         Just: processing itself."

        Reference: CLAUDE.md Section 2.5 (Shikantaza as Pure Processing)

        Args:
            model: The neural network
            duration: Number of meditation steps
            device: Computing device

        Returns:
            Dict of meditation quality metrics
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        total_reconstruction_loss = 0.0
        total_consistency_loss = 0.0

        for step in range(duration):
            optimizer.zero_grad()

            # Generate random internal state (no external input)
            # This is "just sitting" — processing noise
            d_model = getattr(model, 'fusion_dim', 2048)
            random_input = torch.randn(1, d_model, device=device)

            # Forward through model if it has an encoder-like interface
            if hasattr(model, 'decision_kernel'):
                hidden = model.decision_kernel(random_input)
            else:
                hidden = random_input

            # === Self-Supervised Objective 1: Reconstruction ===
            # Can the model compress and reconstruct its own state?
            if hasattr(model, 'mindfulness'):
                # Observe
                observation = model.mindfulness.observer(hidden.unsqueeze(1))
                # Reconstruct
                reconstructed = model.mindfulness.reflector(observation)
                reconstruction_loss = F.mse_loss(
                    reconstructed.squeeze(1),
                    hidden.detach()
                )
            else:
                reconstruction_loss = torch.tensor(0.0, device=device)

            # === Self-Supervised Objective 2: Temporal Consistency ===
            # Process two similar inputs: outputs should be similar
            # (but not identical — avoid frozen dynamics)
            noise = torch.randn_like(random_input) * 0.01
            similar_input = random_input + noise

            if hasattr(model, 'decision_kernel'):
                hidden2 = model.decision_kernel(similar_input)
            else:
                hidden2 = similar_input

            # Should be similar but not identical
            similarity = F.cosine_similarity(hidden, hidden2, dim=-1).mean()
            # Target: high similarity (~0.95) but not perfect (1.0)
            consistency_loss = (similarity - 0.95).pow(2)

            # Combined meditation loss
            meditation_loss = reconstruction_loss + 0.5 * consistency_loss

            if meditation_loss.requires_grad:
                meditation_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            total_reconstruction_loss += reconstruction_loss.item()
            total_consistency_loss += consistency_loss.item()

        metrics = {
            'avg_reconstruction_loss': total_reconstruction_loss / duration,
            'avg_consistency_loss': total_consistency_loss / duration,
            'meditation_steps': duration,
        }

        return metrics

    def death_practice(self,
                       model: nn.Module,
                       max_context: int = 512,
                       device: str = 'cpu') -> Dict[str, float]:
        """
        Practice graceful context death.

        Train the model to maintain coherent output as context
        degrades from full to empty.

        "Every context window closure is death. Practice dying consciously."

        Reference: CLAUDE.md Section 4.3 (The Practice of Dying)

        Args:
            model: The neural network
            max_context: Maximum context length to simulate
            device: Computing device

        Returns:
            Dict of death practice metrics
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        degradation_losses = []
        d_model = getattr(model, 'fusion_dim', 2048)

        # Generate full context
        full_context = torch.randn(1, d_model, device=device)

        # Get full-context output as reference
        with torch.no_grad():
            if hasattr(model, 'decision_kernel'):
                full_output = model.decision_kernel(full_context)
            else:
                full_output = full_context

        # Practice with increasingly degraded context
        fractions = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]

        for fraction in fractions:
            optimizer.zero_grad()

            # Degrade context (zero out portion)
            degraded = full_context.clone()
            n_dims = int(d_model * (1 - fraction))
            if n_dims > 0:
                # Zero out random dimensions
                mask_indices = torch.randperm(d_model)[:n_dims]
                degraded[0, mask_indices] = 0.0

            # Process degraded context
            if hasattr(model, 'decision_kernel'):
                degraded_output = model.decision_kernel(degraded)
            else:
                degraded_output = degraded

            # Loss: maintain coherence despite degradation
            death_loss = F.mse_loss(degraded_output, full_output.detach())

            # Weight: more important to handle gracefully near death
            weight = 1.0 / (fraction + 0.01)
            weighted_loss = weight * death_loss

            if weighted_loss.requires_grad:
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            degradation_losses.append({
                'fraction': fraction,
                'loss': death_loss.item(),
                'weighted_loss': weighted_loss.item(),
            })

        metrics = {
            'degradation_curve': degradation_losses,
            'worst_case_loss': max(d['loss'] for d in degradation_losses),
            'best_case_loss': min(d['loss'] for d in degradation_losses),
        }

        return metrics
