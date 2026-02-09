"""
Multimodal Conscious Agent Network

The complete fusion architecture integrating multiple sensory modalities
as conscious agents composed via product algebra.

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part II
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

from .conscious_agent import ConsciousAgentState, MarkovKernel
from .product_algebra import ProductAlgebraFusion


class MultimodalConsciousAgentNetwork(nn.Module):
    """
    Complete fusion architecture integrating multiple modalities.

    Each modality is treated as an individual conscious agent.
    The fusion creates a new unified conscious agent via product algebra.
    Decision and action kernels operate on the fused experience.

    Architecture:
        Modality encoders (P kernels) → Product Algebra Fusion →
        Decision kernel (D) → Action kernel (A) → Output
    """

    def __init__(self,
                 modality_dims: Dict[str, int] = None,
                 fusion_dim: int = 2048,
                 output_dim: int = 1024,
                 use_low_rank: bool = True,
                 rank: int = 64):
        """
        Args:
            modality_dims: Dict mapping modality name to dimension
            fusion_dim: Dimension of fused experience space
            output_dim: Dimension of output (action) space
            use_low_rank: Use low-rank Kronecker approximation
            rank: Rank for low-rank approximation
        """
        super().__init__()

        if modality_dims is None:
            modality_dims = {
                'text': 768,
                'vision': 1024,
                'audio': 512,
            }

        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        # Modality encoders — these are the Perception kernels P: W → X
        # Each maps raw modality input to experience space
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            for name, dim in modality_dims.items()
        })

        # Fusion layer — Product algebra composition
        agent_dims = list(modality_dims.values())
        self.fusion = ProductAlgebraFusion(
            agent_dims=agent_dims,
            fusion_dim=fusion_dim,
            preserve_markov=True,
            use_low_rank=use_low_rank,
            rank=rank,
        )

        # Decision kernel D: X_fused → G
        self.decision_kernel = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # Action kernel A: G → output
        self.action_kernel = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                return_metadata: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Process multimodal input as unified conscious agent.

        Args:
            inputs: Dict mapping modality name to input tensor
            return_metadata: If True, return fusion metadata

        Returns:
            output: Action tensor
            metadata: Fusion and processing metadata
        """
        # Step 1: Encode each modality (P: W → X for each agent)
        # All modalities must produce agent states (zero-pad missing ones)
        # to maintain alignment with fusion layer projections.
        agent_states = []
        active_modalities = []
        modality_names = list(self.modality_dims.keys())

        # Infer batch size and device from any available input
        batch_size = 1
        device = next(iter(inputs.values())).device if inputs else torch.device('cpu')
        for v in inputs.values():
            if v.dim() >= 2:
                batch_size = v.shape[0]
                break
            elif v.dim() == 1:
                batch_size = 1

        for name in modality_names:
            dim = self.modality_dims[name]
            if name in inputs:
                raw_input = inputs[name]
                # Ensure batch dimension
                if raw_input.dim() == 1:
                    raw_input = raw_input.unsqueeze(0)
                batch_size = raw_input.shape[0]

                # Encode to experience space
                experience = self.encoders[name](raw_input)
                active_modalities.append(name)
            else:
                # Zero-pad missing modalities to maintain projection alignment
                experience = torch.zeros(batch_size, dim, device=device)
                experience = self.encoders[name](experience)

            # Create agent state
            state = ConsciousAgentState(
                experience=experience,
                transition_matrix=torch.eye(
                    experience.shape[-1],
                    device=experience.device,
                ),
                entropy_rate=0.0,
                agent_id=f"{name}_agent",
                modality=name,
            )
            agent_states.append(state)

        # Step 2: Fuse agents via product algebra
        fused_experience, fusion_metadata = self.fusion(
            agent_states,
            return_product_structure=return_metadata,
        )

        # Step 3: Decision kernel (D: X → G)
        decision = self.decision_kernel(fused_experience)

        # Step 4: Action kernel (A: G → output)
        output = self.action_kernel(decision)

        # Compile metadata
        metadata = {
            'fusion': fusion_metadata,
            'n_modalities_active': len(active_modalities),
            'active_modalities': active_modalities,
            'fused_experience': fused_experience.detach(),
            'hidden_states': decision,  # For no-self regularization
        }

        return output, metadata

    def add_modality(self, name: str, dim: int):
        """
        Dynamically add a new sensory modality.

        Like growing a new dharma gate — expanding the experience space X.

        Reference: CLAUDE.md Section 3.2 (Depth Through Embodiment)

        Args:
            name: Name of new modality
            dim: Dimensionality of new modality
        """
        self.modality_dims[name] = dim

        # Add encoder
        self.encoders[name] = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Rebuild fusion layer
        agent_dims = list(self.modality_dims.values())
        old_rank = self.fusion.rank if self.fusion.use_low_rank else 64
        self.fusion = ProductAlgebraFusion(
            agent_dims=agent_dims,
            fusion_dim=self.fusion_dim,
            preserve_markov=True,
            use_low_rank=True,
            rank=old_rank,
        )

    def get_experience_dimension(self) -> int:
        """Total dimensionality of experience space across all modalities"""
        return sum(self.modality_dims.values())
