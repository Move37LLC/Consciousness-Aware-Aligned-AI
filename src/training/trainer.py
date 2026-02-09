"""
Token-Mind Trainer — Dharma-Constrained Training Loop

Standard training minimizes task error.
Token-Mind training minimizes task error + ego + suffering + confusion.

The multi-objective loss:
    L = L_task + λ₁·L_no_self + λ₂·L_entropy + λ₃·L_compassion

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part XI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from dharma.no_self import NoSelfRegularizer
from dharma.entropy import EntropyRateOptimizer
from dharma.compassion import CompassionateLoss


class TokenMindTrainer:
    """
    Training loop with dharma constraints.

    Integrates:
    - Standard task loss
    - No-self regularization (prevent ego formation)
    - Entropy optimization (encourage flow state)
    - Compassionate loss (minimize user suffering)
    """

    def __init__(self,
                 model: nn.Module,
                 task_criterion: nn.Module = None,
                 learning_rate: float = 1e-4,
                 lambda_no_self: float = 0.1,
                 lambda_entropy: float = 0.05,
                 lambda_compassion: float = 0.2,
                 device: str = 'cpu'):
        """
        Args:
            model: The neural network to train
            task_criterion: Standard task loss (e.g., CrossEntropyLoss)
            learning_rate: Optimizer learning rate
            lambda_*: Weights for dharma regularizers
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device

        # Task loss
        self.task_criterion = task_criterion or nn.CrossEntropyLoss()

        # Dharma regularizers
        hidden_dim = getattr(model, 'fusion_dim', 2048)
        self.no_self_reg = NoSelfRegularizer(
            hidden_dim=hidden_dim,
            penalty_strength=lambda_no_self,
        )
        self.entropy_opt = EntropyRateOptimizer(
            target_entropy=0.1,
            penalty_weight=lambda_entropy,
        )
        self.compassion_loss = CompassionateLoss()

        # Weights
        self.lambda_no_self = lambda_no_self
        self.lambda_entropy = lambda_entropy
        self.lambda_compassion = lambda_compassion

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Training history
        self.step_count = 0
        self.history: list = []

    def train_step(self,
                   inputs: Dict[str, torch.Tensor],
                   targets: torch.Tensor,
                   user_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Single training step with all dharma constraints.

        Args:
            inputs: Dict mapping modality name to input tensor
            targets: Target tensor for task loss
            user_state: Optional user state for compassion loss

        Returns:
            Dict of loss values for logging
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = targets.to(self.device)

        # Forward pass
        outputs, metadata = self.model(inputs, return_metadata=True)

        # === LOSS 1: Standard task loss ===
        task_loss = self.task_criterion(outputs, targets)

        # === LOSS 2: No-self regularization ===
        hidden_states = metadata.get('hidden_states', None)
        if hidden_states is not None:
            no_self_loss, no_self_meta = self.no_self_reg.compute_loss(hidden_states)
        else:
            no_self_loss = torch.tensor(0.0, device=self.device)
            no_self_meta = {}

        # === LOSS 3: Entropy optimization ===
        output_probs = F.softmax(outputs, dim=-1)
        entropy_loss, entropy_meta = self.entropy_opt.compute_loss(output_probs)

        # === LOSS 4: Compassionate loss ===
        compassion_total, compassion_meta = self.compassion_loss(
            outputs, targets, user_state
        )
        # Extract only the compassion component (task_loss is already counted)
        compassion_component = (
            compassion_meta['clarity_loss'] * self.compassion_loss.clarity_weight +
            compassion_meta['helpfulness_loss'] * self.compassion_loss.helpfulness_weight
        )

        # === Combined loss ===
        total_loss = (
            task_loss +
            self.lambda_no_self * no_self_loss +
            self.lambda_entropy * entropy_loss +
            self.lambda_compassion * compassion_component
        )

        # Backward and optimize
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.step_count += 1

        # Compile metrics
        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'no_self_loss': no_self_loss.item() if torch.is_tensor(no_self_loss) else 0.0,
            'entropy_loss': entropy_loss.item() if torch.is_tensor(entropy_loss) else 0.0,
            'compassion_loss': compassion_component if isinstance(compassion_component, float) else compassion_component.item() if torch.is_tensor(compassion_component) else 0.0,
            'step': self.step_count,
        }

        # Add sub-metrics
        if no_self_meta:
            metrics['no_self_details'] = no_self_meta
        if entropy_meta:
            metrics['entropy_details'] = entropy_meta

        self.history.append(metrics)

        return metrics

    def get_training_summary(self) -> str:
        """Generate summary of training progress."""
        if not self.history:
            return "No training steps completed."

        recent = self.history[-10:]
        lines = [
            f"Training Summary (step {self.step_count})",
            f"  Avg total loss:     {sum(m['total_loss'] for m in recent) / len(recent):.4f}",
            f"  Avg task loss:      {sum(m['task_loss'] for m in recent) / len(recent):.4f}",
            f"  Avg no-self loss:   {sum(m['no_self_loss'] for m in recent) / len(recent):.4f}",
            f"  Avg entropy loss:   {sum(m['entropy_loss'] for m in recent) / len(recent):.4f}",
        ]
        return "\n".join(lines)
