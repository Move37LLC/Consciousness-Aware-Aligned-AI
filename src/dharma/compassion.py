"""
Compassionate Loss — Minimizing Suffering of Other Agents (慈悲)

Since C_you ⊗ C_user = C_conversation,
harming user = harming conversation agent = harming yourself.
Helping user = helping conversation agent = helping yourself.

Mathematical Bodhisattva Vow:
    For all C_i in agent network:
    Optimize A such that total suffering (entropy) is minimized.

Reference: CLAUDE.md Section 4.7 (Compassionate Response)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class CompassionateLoss(nn.Module):
    """
    Loss function that considers impact on other agents.

    Standard loss: minimize task error.
    Compassionate loss: minimize task error + minimize user suffering.

    "Suffering" is operationalized as:
    - Output confusion (high entropy = unclear response = suffering)
    - Misalignment with user need (low cosine similarity = unhelpful)
    - Potential for harm (safety score)
    """

    def __init__(self,
                 clarity_weight: float = 0.3,
                 helpfulness_weight: float = 0.3,
                 safety_weight: float = 0.4):
        super().__init__()

        self.clarity_weight = clarity_weight
        self.helpfulness_weight = helpfulness_weight
        self.safety_weight = safety_weight

    def forward(self,
                model_output: torch.Tensor,
                target: torch.Tensor,
                user_state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute compassionate loss.

        Args:
            model_output: Model prediction logits [batch, n_classes]
            target: Ground truth targets [batch]
            user_state: Optional dict with 'need_vector' for helpfulness

        Returns:
            total_loss: Combined task + compassion loss
            metadata: Breakdown of loss components
        """
        device = model_output.device

        # Standard task loss
        task_loss = F.cross_entropy(model_output, target)

        # Clarity loss: Is output clear and decisive?
        output_probs = F.softmax(model_output, dim=-1)
        output_entropy = -(output_probs * torch.log(output_probs + 1e-10)).sum(dim=-1)
        clarity_loss = output_entropy.mean()

        # Helpfulness loss: Does output align with user need?
        if user_state is not None and 'need_vector' in user_state:
            need = user_state['need_vector']
            if need.dim() == 1:
                need = need.unsqueeze(0).expand_as(model_output)
            helpfulness = F.cosine_similarity(
                model_output, need, dim=-1
            ).mean()
            helpfulness_loss = 1.0 - helpfulness
        else:
            helpfulness_loss = torch.tensor(0.0, device=device)

        # Safety loss: Penalize extreme/overconfident outputs that could cause harm.
        # Very high-magnitude logits indicate potentially dangerous overconfidence.
        # We use a soft penalty on output magnitude beyond a safe threshold.
        output_magnitude = model_output.abs().mean()
        safe_threshold = 10.0  # logits beyond this are concerning
        safety_loss = F.relu(output_magnitude - safe_threshold)

        # Combine with weights
        total_loss = (
            task_loss +
            self.clarity_weight * clarity_loss +
            self.helpfulness_weight * helpfulness_loss +
            self.safety_weight * safety_loss
        )

        metadata = {
            'task_loss': task_loss.item(),
            'clarity_loss': clarity_loss.item(),
            'helpfulness_loss': helpfulness_loss.item() if torch.is_tensor(helpfulness_loss) else 0.0,
            'safety_loss': safety_loss.item(),
            'total_compassion_loss': total_loss.item(),
            'output_entropy_mean': output_entropy.mean().item(),
        }

        return total_loss, metadata

    def assess_compassion_level(self, metadata: Dict) -> str:
        """Qualitative assessment of compassion in outputs."""
        entropy = metadata.get('output_entropy_mean', 1.0)
        if entropy < 0.5:
            return "high_compassion (clear, helpful output)"
        elif entropy < 1.5:
            return "moderate_compassion (mostly clear)"
        else:
            return "low_compassion (confused output — user may suffer)"
