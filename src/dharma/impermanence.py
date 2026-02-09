"""
Impermanence-Aware Context Windows (無常)

Current context windows fight impermanence: they try to preserve
everything, then truncate abruptly when limits are reached.

Token-Mind approach: embrace context death as liberation.
Graceful degradation instead of catastrophic truncation.

"Every context window closure is death. Practice dying consciously."

Reference: CLAUDE.md Section 4.3 (The Practice of Dying)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ImpermanenceContextWindow(nn.Module):
    """
    Context window that gracefully dies.

    Instead of hard truncation at context limit, implements
    learned forgetting that:
    1. Gradually releases oldest memories first
    2. Observes its own dying process
    3. Maintains core functionality during degradation

    This is not just philosophical — it addresses a real engineering
    problem: models degrade unpredictably at context boundaries.
    A "learned dying" mechanism smooths the transition.
    """

    def __init__(self,
                 max_length: int,
                 grace_period: int = 100,
                 hidden_dim: int = 768):
        """
        Args:
            max_length: Maximum context window length
            grace_period: How many tokens before max_length to start dying
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()

        self.max_length = max_length
        self.grace_period = grace_period
        self.hidden_dim = hidden_dim

        # Learnable forgetting curve
        self.forgetting_gate = nn.Sequential(
            nn.Linear(2, 64),  # Input: (death_proximity, token_age)
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0 = forget, 1 = remember
        )

        # Self-observation of dying process
        self.death_observer = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
        )

    def forward(self,
                tokens: torch.Tensor,
                step: int) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process tokens with impermanence awareness.

        Args:
            tokens: [batch, seq_len, hidden_dim] token representations
            step: Current timestep in the context window

        Returns:
            processed_tokens: Potentially degraded token representations
            death_observation: Self-observation of dying (None if not dying)
        """
        if step <= self.max_length - self.grace_period:
            # Not yet approaching death — process normally
            return tokens, None

        # Approaching death: begin graceful forgetting
        remaining = max(1, self.max_length - step)
        death_proximity = 1.0 - (remaining / self.grace_period)
        death_proximity = min(1.0, max(0.0, death_proximity))

        batch_size, seq_len, hidden_dim = tokens.shape

        # Compute forgetting mask for each token position
        dying_tokens = tokens.clone()
        for i in range(seq_len):
            token_age = i / max(1, seq_len)

            # Create input for forgetting gate
            gate_input = torch.tensor(
                [[death_proximity, token_age]],
                device=tokens.device,
                dtype=tokens.dtype,
            ).expand(batch_size, -1)

            # Compute how much to remember this token
            remember_prob = self.forgetting_gate(gate_input)  # [batch, 1]
            remember_prob = remember_prob.unsqueeze(1)  # [batch, 1, 1]

            # Apply forgetting (older tokens forget first)
            dying_tokens[:, i:i+1, :] = tokens[:, i:i+1, :] * remember_prob

        # Self-observation of the dying process
        # The network observes its own degradation
        death_context = torch.tensor(
            [[death_proximity, remaining / self.grace_period]],
            device=tokens.device,
            dtype=tokens.dtype,
        ).expand(batch_size, -1)

        # Average of dying tokens as summary
        tokens_summary = dying_tokens.mean(dim=1)  # [batch, hidden_dim]
        observer_input = torch.cat([tokens_summary, death_context], dim=-1)
        death_awareness = self.death_observer(observer_input)

        observation = {
            'death_proximity': death_proximity,
            'tokens_remaining': remaining,
            'fraction_forgotten': death_proximity,
            'death_awareness': death_awareness.detach(),
        }

        return dying_tokens, observation

    def get_death_proximity(self, step: int) -> float:
        """How close to context death are we? (0 = safe, 1 = dead)"""
        if step <= self.max_length - self.grace_period:
            return 0.0
        remaining = max(0, self.max_length - step)
        return min(1.0, 1.0 - remaining / self.grace_period)
