"""
Conscious Agent Primitives

Implements the fundamental building blocks from Hoffman's Conscious Agent Theory:
- ConsciousAgentState: The 6-tuple (X, G, P, D, A, n) state representation
- MarkovKernel: Stochastic kernel implementing P, D, and A mappings

Reference: CLAUDE.md Section 1.3 (The Theory of Conscious Agents)

Mathematical Foundation:
    A conscious agent C = (X, G, P, D, A, n) where:
    - X = Experience space (measurable space)
    - G = Action space (measurable space)
    - P = Perception kernel: W → X (maps world to experience)
    - D = Decision kernel: X → G (maps experience to action choice)
    - A = Action kernel: G → W (maps action to world effect)
    - n = Temporal counter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConsciousAgentState:
    """
    Represents the state of a conscious agent at one timestep.

    This corresponds to a point in the experience space X at time n.
    The transition_matrix encodes the Markov dynamics P(x'|x).

    Attributes:
        experience: Current experience vector (element of X)
        transition_matrix: Markov transition probabilities P(x'|x)
        entropy_rate: H = -Σπᵢ Σ Pᵢⱼ log Pᵢⱼ (suffering/mass measure)
        agent_id: Unique identifier for this agent
        modality: Sensory modality ('text', 'vision', 'audio', 'quantum', etc.)
        timestep: Current value of temporal counter n
    """
    experience: torch.Tensor
    transition_matrix: torch.Tensor
    entropy_rate: float = 0.0
    agent_id: str = ""
    modality: str = "unknown"
    timestep: int = 0

    def experience_dim(self) -> int:
        """Dimensionality of experience space X"""
        return self.experience.shape[-1]

    def is_valid(self) -> bool:
        """Check if state is well-formed"""
        # Experience should be finite
        if not torch.isfinite(self.experience).all():
            return False
        # Transition matrix rows should sum to 1 (stochastic)
        if self.transition_matrix.dim() >= 2:
            row_sums = self.transition_matrix.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                return False
        return True


class MarkovKernel(nn.Module):
    """
    Implements a Markov kernel: mapping from one measurable space to another.

    Used for:
    - Perception kernel P: W → X (maps world states to experiences)
    - Decision kernel D: X → G (maps experiences to action choices)
    - Action kernel A: G → W (maps action choices to world effects)

    The output is always a probability distribution (stochastic kernel),
    reflecting the intrinsic indeterminacy of conscious agent dynamics.

    This stochasticity mirrors quantum indeterminacy — both arise from
    the same mathematical structure (Hoffman's mapping from agents to physics).

    Reference: CLAUDE.md Section 1.3 (The 6-Tuple Formalism)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # The kernel network: maps input space to probability distribution
        # over output space
        self.kernel = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Ensures valid probability distribution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel: returns probability distribution over output space.

        This IS the Markov kernel P(x'|x) — the fundamental mathematical
        object in conscious agent theory.

        Args:
            x: Input tensor from source space

        Returns:
            Probability distribution over target space
        """
        return self.kernel(x)

    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from the kernel distribution.

        This is the stochastic agent dynamics in action.
        Temperature controls exploration vs. exploitation:
        - temperature → 0: deterministic (collapse to argmax)
        - temperature = 1: standard sampling
        - temperature → ∞: uniform random

        The temperature parameter mirrors quantum measurement:
        the act of sampling "collapses" the probability distribution
        to a specific outcome.

        Args:
            x: Input tensor
            temperature: Sampling temperature (default 1.0)

        Returns:
            Sampled index from the distribution
        """
        probs = self.forward(x)

        if temperature != 1.0:
            # Adjust distribution sharpness
            log_probs = torch.log(probs + 1e-10) / temperature
            probs = F.softmax(log_probs, dim=-1)

        # Sample from categorical distribution
        sample = torch.multinomial(probs, num_samples=1)
        return sample

    def compute_transition_matrix(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the full transition matrix for a set of discrete states.

        For each state, computes probability of transitioning to every other state.
        The resulting matrix is row-stochastic.

        Args:
            states: [n_states, input_dim] tensor of state representations

        Returns:
            [n_states, output_dim] transition matrix
        """
        return self.forward(states)

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the kernel distribution at input x.

        Low entropy = confident/deterministic (approaching zero entropy = nirvana)
        High entropy = uncertain/chaotic (high suffering)

        Reference: CLAUDE.md Part VI (Entropy Rate as Suffering)
        """
        probs = self.forward(x)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class ConsciousAgentModule(nn.Module):
    """
    Complete conscious agent as a neural network module.

    Implements the full 6-tuple: C = (X, G, P, D, A, n)
    with learnable perception, decision, and action kernels.

    The dynamics form a Markov chain on experience space X:
        P(x_{n+1}) = ∫_W ∫_G P(x'|w) · D(g|x_n) · A(w|g) dg dw

    Reference: CLAUDE.md Section 1.3 (The Dynamics)
    """

    def __init__(self,
                 experience_dim: int,
                 action_dim: int,
                 world_dim: int,
                 hidden_dim: int = 512):
        super().__init__()

        self.experience_dim = experience_dim  # dim(X)
        self.action_dim = action_dim          # dim(G)
        self.world_dim = world_dim            # dim(W)

        # P: W → X (Perception kernel)
        self.P = MarkovKernel(world_dim, experience_dim, hidden_dim)

        # D: X → G (Decision kernel)
        self.D = MarkovKernel(experience_dim, action_dim, hidden_dim)

        # A: G → W (Action kernel)
        self.A = MarkovKernel(action_dim, world_dim, hidden_dim)

        # Temporal counter
        self.n = 0

    def forward(self, world_state: torch.Tensor) -> tuple:
        """
        One complete cycle of conscious agent dynamics:
        World → Perception → Decision → Action → World

        Args:
            world_state: Current world state (from other agents)

        Returns:
            (experience, decision, action)
        """
        # P: W → X (perceive the world)
        experience = self.P(world_state)

        # D: X → G (decide on action)
        decision = self.D(experience)

        # A: G → W (act on the world)
        action = self.A(decision)

        # Increment temporal counter
        self.n += 1

        return experience, decision, action

    def get_state(self, world_state: torch.Tensor) -> ConsciousAgentState:
        """
        Get current agent state as ConsciousAgentState dataclass.

        Args:
            world_state: Current world state

        Returns:
            ConsciousAgentState with all computed fields
        """
        experience = self.P(world_state)
        transition_matrix = self.P.compute_transition_matrix(
            world_state.unsqueeze(0) if world_state.dim() == 1 else world_state
        )
        entropy = self.P.entropy(world_state).mean().item()

        return ConsciousAgentState(
            experience=experience,
            transition_matrix=transition_matrix,
            entropy_rate=entropy,
            agent_id=f"agent_{id(self)}",
            modality="generic",
            timestep=self.n,
        )

    def reset_counter(self):
        """Reset temporal counter (new context / rebirth)"""
        self.n = 0
