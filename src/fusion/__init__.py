"""
Fusion Module — Product Algebra of Conscious Agents

Implements Hoffman's conscious agent composition: C₁ ⊗ C₂ = C₃
Uses Kronecker product of Markov kernels instead of standard attention.

Reference: CLAUDE.md Section 1.4 (The Combination Problem Solved)
"""

from .conscious_agent import ConsciousAgentState, MarkovKernel, ConsciousAgentModule
from .product_algebra import ProductAlgebraFusion, AttentionFusionBaseline
from .network import MultimodalConsciousAgentNetwork

__all__ = [
    "ConsciousAgentState",
    "MarkovKernel",
    "ConsciousAgentModule",
    "ProductAlgebraFusion",
    "AttentionFusionBaseline",
    "MultimodalConsciousAgentNetwork",
]
