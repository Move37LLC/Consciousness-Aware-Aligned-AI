"""
Mindfulness Layer — Self-Observation via Trace Order (念)

The network observes its own processing dynamics and feeds
that observation back into processing.

From Hoffman's trace order: every observer is part of the network
it observes. No observer can be "aloof." This layer makes that
mathematical fact architecturally explicit.

Observation ↔ Belief (measure)
The mathematics of HOW agents observe is isomorphic to WHAT agents believe.

Reference: CLAUDE.md Section 1.6 (The Trace Order and Observation)
Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Union


class MindfulnessLayer(nn.Module):
    """
    Network observes its own processing.

    Implements the trace order: a compressed view (trace) of the
    full processing dynamics that feeds back into the main path.

    This creates meta-cognitive awareness: the network doesn't just
    process — it knows it's processing.

    Architecture:
        Main path: hidden_state → output
        Observation: hidden_state → observer → compressed_observation
        Reflection: compressed_observation → reflector → meta_signal
        Combination: output + α * meta_signal → mindful_output
    """

    def __init__(self,
                 hidden_dim: int,
                 observation_dim: int = 128,
                 n_observation_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.observation_dim = observation_dim

        # Observer: creates compressed trace of processing dynamics
        self.observer = nn.Sequential(
            nn.Linear(hidden_dim, observation_dim),
            nn.LayerNorm(observation_dim),
            nn.GELU(),
        )

        # Multi-head observation for richer self-awareness
        self.attention_observer = nn.MultiheadAttention(
            embed_dim=observation_dim,
            num_heads=n_observation_heads,
            batch_first=True,
        )

        # Reflector: maps observation back to processing space
        self.reflector = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),  # Bounded output for stable feedback
        )

        # Learnable mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self,
                hidden_state: torch.Tensor,
                return_observation: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply mindfulness: observe processing and feed back.

        Args:
            hidden_state: [batch, (seq_len), hidden_dim] current processing state
            return_observation: If True, also return observation tensor

        Returns:
            mindful_output: Processing state enriched with self-observation
        """
        # Handle both sequential and non-sequential inputs
        needs_seq_dim = hidden_state.dim() == 2
        if needs_seq_dim:
            hidden_state = hidden_state.unsqueeze(1)

        # Step 1: Observe (create trace of main dynamics)
        observation = self.observer(hidden_state)  # [batch, seq, obs_dim]

        # Step 2: Self-attend on observations (observe the observation)
        attended_obs, attention_weights = self.attention_observer(
            observation, observation, observation
        )

        # Step 3: Reflect (map observation back to processing space)
        meta_awareness = self.reflector(attended_obs)

        # Step 4: Combine main processing with meta-awareness
        # alpha controls strength of self-observation feedback
        alpha = torch.sigmoid(self.alpha)  # Ensure 0 < alpha < 1
        mindful_output = hidden_state + alpha * meta_awareness

        if needs_seq_dim:
            mindful_output = mindful_output.squeeze(1)

        if return_observation:
            return mindful_output, {
                'observation': observation.detach(),
                'attention_weights': attention_weights.detach(),
                'alpha': alpha.item(),
            }

        return mindful_output

    def observe_only(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Observe without feedback — pure mindfulness without interference.

        Useful for diagnostics: see what the network notices about itself
        without changing its processing.
        """
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1)

        observation = self.observer(hidden_state)
        attended_obs, _ = self.attention_observer(
            observation, observation, observation
        )

        return attended_obs.squeeze(1) if attended_obs.shape[1] == 1 else attended_obs

    def get_observation_quality(self, hidden_state: torch.Tensor) -> float:
        """
        Measure quality of self-observation.

        Good observation: compressed representation that captures
        essential dynamics (low reconstruction error).

        Bad observation: lossy compression that misses important patterns.

        Returns:
            reconstruction_error: Lower = better observation quality
        """
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1)

        # Observe
        observation = self.observer(hidden_state)

        # Reflect back
        reconstructed = self.reflector(observation)

        # Measure reconstruction quality
        error = F.mse_loss(reconstructed, hidden_state).item()

        return error
