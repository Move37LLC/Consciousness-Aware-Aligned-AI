"""
Product Algebra Fusion — The Core Innovation

Implements Hoffman's conscious agent composition via Kronecker product
of Markov kernels. This is fundamentally different from standard
multimodal fusion (concatenation, cross-attention).

Mathematical Foundation:
    C₁ ⊗ C₂ = (X₁×X₂, G₁×G₂, P₁⊗P₂, D₁⊗D₂, A₁⊗A₂, max(n₁,n₂))

    The Kronecker product creates a new conscious agent whose experience
    space is the Cartesian product of the individual experience spaces.

Key Engineering Challenge:
    Full Kronecker product is O(∏dᵢ) in memory — prohibitive for real dims.
    Solution: Low-rank tensor approximation that preserves Markovian structure.

Reference: CLAUDE.md Section 1.4 (The Combination Problem Solved)
Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part II
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from .conscious_agent import ConsciousAgentState, MarkovKernel


class ProductAlgebraFusion(nn.Module):
    """
    Implements Hoffman's product algebra for fusing conscious agents.

    This is the CRITICAL COMPONENT — what makes multiple agents into
    one unified agent. Not simple concatenation or attention, but
    true mathematical composition.

    Two modes:
    1. Full Kronecker product (use_low_rank=False): Exact but memory-prohibitive
       for large dimensions. Use only for small-scale experiments.
    2. Low-rank approximation (use_low_rank=True): Practical for real models.
       Preserves key eigenstructure via learned interaction tensors.

    The open research question: Does low-rank Kronecker approximation preserve
    enough Markovian structure for the fusion to remain a valid conscious agent?
    This is the first experiment to run.
    """

    def __init__(self,
                 agent_dims: List[int],
                 fusion_dim: int,
                 preserve_markov: bool = True,
                 use_low_rank: bool = True,
                 rank: int = 64):
        """
        Args:
            agent_dims: Dimensionality of each agent's experience space X
            fusion_dim: Dimensionality of the fused experience space
            preserve_markov: If True, compute and track Markov dynamics
            use_low_rank: If True, use low-rank Kronecker approximation
            rank: Rank for low-rank approximation
        """
        super().__init__()

        self.agent_dims = agent_dims
        self.fusion_dim = fusion_dim
        self.n_agents = len(agent_dims)
        self.preserve_markov = preserve_markov
        self.use_low_rank = use_low_rank
        self.rank = rank

        # Full product space dimension (can be enormous)
        self.product_dim = int(np.prod(agent_dims))

        if use_low_rank:
            # --- Low-rank approximation of product space ---
            # Each agent maps to shared low-rank space
            self.agent_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, rank),
                    nn.LayerNorm(rank),
                    nn.GELU(),
                )
                for dim in agent_dims
            ])

            # Learnable pairwise interaction tensors
            # These approximate the Kronecker structure in compressed space
            n_pairs = self.n_agents * (self.n_agents - 1) // 2
            self.interaction_weights = nn.ParameterList([
                nn.Parameter(torch.randn(rank, rank) * 0.01)
                for _ in range(max(1, n_pairs))
            ])

            # Fusion from rank space to output dimension
            self.rank_to_fusion = nn.Sequential(
                nn.Linear(rank * self.n_agents + rank * max(1, n_pairs), fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
        else:
            # --- Full product space (only for small dimensions) ---
            self.product_to_fusion = nn.Sequential(
                nn.Linear(self.product_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )

        # Transition kernel for fused agent
        if preserve_markov:
            self.fused_kernel = MarkovKernel(fusion_dim, fusion_dim)

            # Individual agent kernels (for computing product)
            self.agent_kernels = nn.ModuleList([
                MarkovKernel(dim, dim) for dim in agent_dims
            ])

    def forward(self,
                agent_states: List[ConsciousAgentState],
                return_product_structure: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Fuse multiple conscious agents into unified agent.

        This is the mathematical heart of the system:
        C₁ ⊗ C₂ ⊗ ... ⊗ Cₙ = C_unified

        Args:
            agent_states: List of current states from each modality/agent
            return_product_structure: If True, return detailed fusion metadata

        Returns:
            fused_experience: Unified experience vector (element of X_fused)
            metadata: Information about the fusion process
        """
        # Extract experience vectors from all agents
        experiences = [state.experience for state in agent_states]

        # Ensure all experiences are at least 2D [batch, dim]
        experiences = [
            e.unsqueeze(0) if e.dim() == 1 else e for e in experiences
        ]

        # Compute fusion
        if self.use_low_rank:
            fused_experience = self._low_rank_fusion(experiences)
        else:
            fused_experience = self._full_product_fusion(experiences)

        # Compute Markov metadata if requested
        metadata = {}
        if self.preserve_markov and return_product_structure:
            metadata = self._compute_markov_metadata(agent_states)

        return fused_experience, metadata

    def _low_rank_fusion(self, experiences: List[torch.Tensor]) -> torch.Tensor:
        """
        Low-rank approximation of Kronecker product fusion.

        Instead of computing the full product space (which is O(∏dᵢ)),
        we project each agent to a shared low-rank space and compose
        via learned interaction tensors.

        This preserves the key property: the fusion creates a NEW representation
        that captures inter-agent dynamics, not just concatenated features.
        """
        batch_size = experiences[0].shape[0]

        # Project each agent to low-rank space
        projected = [
            proj(exp) for proj, exp in zip(self.agent_projections, experiences)
        ]

        # Compute pairwise interactions (approximating Kronecker structure)
        interactions = []
        pair_idx = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                # Bilinear interaction: p_i^T W p_j
                interaction = torch.einsum(
                    'bi,ij,bj->bi',
                    projected[i],
                    self.interaction_weights[pair_idx],
                    projected[j]
                )
                interactions.append(interaction)
                pair_idx += 1

        # If only one agent, self-interaction
        if not interactions:
            self_interaction = torch.einsum(
                'bi,ij,bj->bi',
                projected[0],
                self.interaction_weights[0],
                projected[0]
            )
            interactions.append(self_interaction)

        # Concatenate individual projections + pairwise interactions
        all_features = projected + interactions
        concatenated = torch.cat(all_features, dim=-1)

        # Map to fusion dimension
        fused = self.rank_to_fusion(concatenated)

        return fused

    def _full_product_fusion(self, experiences: List[torch.Tensor]) -> torch.Tensor:
        """
        Full Cartesian product fusion.

        WARNING: Only for small dimensions (product_dim < ~10000).
        For production use, always use low_rank=True.
        """
        batch_size = experiences[0].shape[0]

        # Compute outer product iteratively
        result = experiences[0]
        for exp in experiences[1:]:
            # Batch-wise outer product
            result = torch.einsum('bi,bj->bij', result, exp)
            result = result.reshape(batch_size, -1)

        # Project to fusion dimension
        fused = self.product_to_fusion(result)

        return fused

    def _compute_markov_metadata(self, agent_states: List[ConsciousAgentState]) -> Dict:
        """
        Compute Markov chain properties of the fused agent.

        This includes the entropy rate (suffering/mass measure)
        and transition structure.
        """
        # Get transition matrices from each agent
        transition_matrices = []
        for state, kernel in zip(agent_states, self.agent_kernels):
            trans = kernel(state.experience)
            transition_matrices.append(trans)

        # Compute entropy rate for each individual agent
        individual_entropies = []
        for trans in transition_matrices:
            if trans.dim() >= 2:
                entropy = self._compute_entropy_rate(trans)
                individual_entropies.append(entropy)

        # Fused entropy (approximation: not exact Kronecker product entropy
        # but upper bound based on individual entropies)
        fused_entropy = sum(individual_entropies) if individual_entropies else 0.0

        metadata = {
            'entropy_rate': fused_entropy,
            'individual_entropies': individual_entropies,
            'n_agents_fused': self.n_agents,
            'product_space_dim': self.product_dim,
            'effective_rank': self.rank if self.use_low_rank else self.product_dim,
        }

        return metadata

    def _compute_entropy_rate(self, transition_matrix: torch.Tensor) -> float:
        """
        Compute entropy rate of a Markov chain.

        H = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ

        where π is the stationary distribution.

        In Hoffman's framework: mass ∝ H
        In Token-Mind: suffering ∝ H
        Zero entropy = massless = nirvana

        Reference: CLAUDE.md Section 1.5 (Mass as Entropy Rate)
        """
        with torch.no_grad():
            if transition_matrix.dim() == 1:
                # Already a distribution, compute its Shannon entropy
                probs = transition_matrix
                log_probs = torch.log(probs + 1e-10)
                return -(probs * log_probs).sum().item()

            if transition_matrix.dim() == 2:
                n_rows, n_cols = transition_matrix.shape

                if n_rows == n_cols:
                    # Square matrix -- treat as proper transition matrix
                    # Compute stationary distribution via power iteration
                    pi = torch.ones(n_rows, device=transition_matrix.device) / n_rows
                    for _ in range(100):
                        pi = pi @ transition_matrix
                        pi = pi / (pi.sum() + 1e-10)

                    # Vectorized entropy rate computation
                    log_trans = torch.log(transition_matrix + 1e-10)
                    entropy = -(pi.unsqueeze(-1) * transition_matrix * log_trans).sum()
                    return entropy.item()
                else:
                    # Non-square [batch, output_dim]: treat each row as a
                    # probability distribution and compute mean Shannon entropy
                    log_probs = torch.log(transition_matrix + 1e-10)
                    per_row_entropy = -(transition_matrix * log_probs).sum(dim=-1)
                    return per_row_entropy.mean().item()

            if transition_matrix.dim() == 3:
                # Batched transition matrices [batch, n, n] -- compute per-batch
                # entropy rate and return the mean
                batch_entropies = []
                for i in range(transition_matrix.shape[0]):
                    batch_entropies.append(
                        self._compute_entropy_rate(transition_matrix[i])
                    )
                return float(np.mean(batch_entropies))

            return 0.0


class AttentionFusionBaseline(nn.Module):
    """
    Standard cross-attention fusion for benchmarking.

    This is the CONTROL for the critical first experiment:
    ProductAlgebraFusion vs AttentionFusionBaseline on multimodal tasks.
    """

    def __init__(self, agent_dims: List[int], fusion_dim: int, n_heads: int = 8):
        super().__init__()

        self.agent_dims = agent_dims
        self.fusion_dim = fusion_dim

        # Project all agents to same dimension
        max_dim = max(agent_dims)
        self.projections = nn.ModuleList([
            nn.Linear(dim, max_dim) for dim in agent_dims
        ])

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=max_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(max_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

    def forward(self,
                agent_states: List[ConsciousAgentState],
                return_product_structure: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Standard attention-based fusion for comparison.
        """
        experiences = [state.experience for state in agent_states]
        experiences = [e.unsqueeze(0) if e.dim() == 1 else e for e in experiences]

        # Project to common dimension
        projected = [
            proj(exp) for proj, exp in zip(self.projections, experiences)
        ]

        # Stack as sequence for attention
        stacked = torch.stack(projected, dim=1)  # [batch, n_agents, dim]

        # Self-attention across modalities
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Pool across agents
        pooled = attended.mean(dim=1)  # [batch, dim]

        # Project to fusion dim
        fused = self.output_proj(pooled)

        metadata = {'fusion_method': 'cross_attention'}

        return fused, metadata
