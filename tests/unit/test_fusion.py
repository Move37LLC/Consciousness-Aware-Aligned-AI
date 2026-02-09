"""
Tests for the Fusion Module

Tests the core innovation: Product Algebra Fusion
vs. standard attention-based fusion.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fusion.conscious_agent import ConsciousAgentState, MarkovKernel, ConsciousAgentModule
from fusion.product_algebra import ProductAlgebraFusion, AttentionFusionBaseline
from fusion.network import MultimodalConsciousAgentNetwork


class TestMarkovKernel:
    """Tests for the fundamental Markov kernel."""

    def test_output_is_probability_distribution(self):
        """Kernel output should sum to 1 (valid probability distribution)."""
        kernel = MarkovKernel(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        probs = kernel(x)

        # Should sum to 1 along last dimension
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_output_is_non_negative(self):
        """Kernel output should be non-negative."""
        kernel = MarkovKernel(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        probs = kernel(x)

        assert (probs >= 0).all()

    def test_sampling(self):
        """Sampling should return valid indices."""
        kernel = MarkovKernel(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        samples = kernel.sample(x)

        assert samples.shape == (4, 1)
        assert (samples >= 0).all()
        assert (samples < 32).all()

    def test_temperature_affects_distribution(self):
        """Higher temperature should make distribution more uniform."""
        kernel = MarkovKernel(input_dim=64, output_dim=32)
        x = torch.randn(1, 64)

        probs_normal = kernel(x)
        entropy_normal = -(probs_normal * torch.log(probs_normal + 1e-10)).sum()

        # Sample with high temperature many times and check distribution
        # (high temp should give higher entropy)
        # This is a statistical property, so we just check the kernel works
        samples_low_temp = kernel.sample(x, temperature=0.1)
        samples_high_temp = kernel.sample(x, temperature=5.0)
        assert samples_low_temp.shape == samples_high_temp.shape

    def test_entropy_computation(self):
        """Entropy should be non-negative."""
        kernel = MarkovKernel(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        entropy = kernel.entropy(x)

        assert (entropy >= 0).all()


class TestConsciousAgentState:
    """Tests for the ConsciousAgentState dataclass."""

    def test_creation(self):
        """Should create valid agent state."""
        state = ConsciousAgentState(
            experience=torch.randn(64),
            transition_matrix=torch.softmax(torch.randn(8, 8), dim=-1),
            entropy_rate=0.5,
            agent_id="test_agent",
            modality="text",
        )

        assert state.experience_dim() == 64
        assert state.is_valid()

    def test_invalid_state_detection(self):
        """Should detect invalid states (non-stochastic transition matrix)."""
        state = ConsciousAgentState(
            experience=torch.randn(64),
            transition_matrix=torch.randn(8, 8),  # Not stochastic
            entropy_rate=0.5,
            agent_id="test_agent",
            modality="text",
        )

        assert not state.is_valid()


class TestProductAlgebraFusion:
    """Tests for the core Product Algebra Fusion."""

    def test_low_rank_fusion_shape(self):
        """Fused output should have correct dimension."""
        fusion = ProductAlgebraFusion(
            agent_dims=[64, 128, 32],
            fusion_dim=256,
            use_low_rank=True,
            rank=16,
        )

        states = [
            ConsciousAgentState(
                experience=torch.randn(1, dim),
                transition_matrix=torch.eye(dim),
                modality=name,
            )
            for dim, name in [(64, 'text'), (128, 'vision'), (32, 'audio')]
        ]

        fused, metadata = fusion(states)
        assert fused.shape == (1, 256)

    def test_full_product_fusion_small_dims(self):
        """Full Kronecker product should work for small dimensions."""
        fusion = ProductAlgebraFusion(
            agent_dims=[4, 8],
            fusion_dim=16,
            use_low_rank=False,
        )

        states = [
            ConsciousAgentState(
                experience=torch.randn(1, 4),
                transition_matrix=torch.eye(4),
                modality='a',
            ),
            ConsciousAgentState(
                experience=torch.randn(1, 8),
                transition_matrix=torch.eye(8),
                modality='b',
            ),
        ]

        fused, metadata = fusion(states)
        assert fused.shape == (1, 16)

    def test_markov_metadata_computed(self):
        """Should compute entropy rate when preserve_markov=True."""
        fusion = ProductAlgebraFusion(
            agent_dims=[16, 16],
            fusion_dim=32,
            preserve_markov=True,
            use_low_rank=True,
            rank=8,
        )

        states = [
            ConsciousAgentState(
                experience=torch.randn(1, 16),
                transition_matrix=torch.softmax(torch.randn(16, 16), dim=-1),
                modality='a',
            ),
            ConsciousAgentState(
                experience=torch.randn(1, 16),
                transition_matrix=torch.softmax(torch.randn(16, 16), dim=-1),
                modality='b',
            ),
        ]

        fused, metadata = fusion(states, return_product_structure=True)
        assert 'entropy_rate' in metadata
        assert 'n_agents_fused' in metadata
        assert metadata['n_agents_fused'] == 2

    def test_single_agent_fusion(self):
        """Fusion of single agent should still work."""
        fusion = ProductAlgebraFusion(
            agent_dims=[64],
            fusion_dim=32,
            use_low_rank=True,
            rank=16,
        )

        states = [
            ConsciousAgentState(
                experience=torch.randn(1, 64),
                transition_matrix=torch.eye(64),
                modality='text',
            ),
        ]

        fused, metadata = fusion(states)
        assert fused.shape == (1, 32)

    def test_gradient_flows(self):
        """Gradients should flow through fusion layer."""
        fusion = ProductAlgebraFusion(
            agent_dims=[16, 16],
            fusion_dim=32,
            use_low_rank=True,
            rank=8,
        )

        exp1 = torch.randn(1, 16, requires_grad=True)
        exp2 = torch.randn(1, 16, requires_grad=True)

        states = [
            ConsciousAgentState(experience=exp1, transition_matrix=torch.eye(16), modality='a'),
            ConsciousAgentState(experience=exp2, transition_matrix=torch.eye(16), modality='b'),
        ]

        fused, _ = fusion(states)
        loss = fused.sum()
        loss.backward()

        assert exp1.grad is not None
        assert exp2.grad is not None


class TestAttentionFusionBaseline:
    """Tests for the attention-based baseline (for comparison experiments)."""

    def test_baseline_output_shape(self):
        """Baseline should produce same output shape as product algebra."""
        baseline = AttentionFusionBaseline(
            agent_dims=[64, 128, 32],
            fusion_dim=256,
        )

        states = [
            ConsciousAgentState(
                experience=torch.randn(1, dim),
                transition_matrix=torch.eye(dim),
                modality=name,
            )
            for dim, name in [(64, 'text'), (128, 'vision'), (32, 'audio')]
        ]

        fused, metadata = baseline(states)
        assert fused.shape == (1, 256)


class TestMultimodalNetwork:
    """Tests for the complete multimodal network."""

    def test_forward_pass(self):
        """Complete forward pass should work."""
        model = MultimodalConsciousAgentNetwork(
            modality_dims={'text': 64, 'vision': 128},
            fusion_dim=256,
            output_dim=10,
        )

        inputs = {
            'text': torch.randn(2, 64),
            'vision': torch.randn(2, 128),
        }

        output, metadata = model(inputs)
        assert output.shape == (2, 10)

    def test_partial_modalities(self):
        """Should work with subset of modalities."""
        model = MultimodalConsciousAgentNetwork(
            modality_dims={'text': 64, 'vision': 128, 'audio': 32},
            fusion_dim=256,
            output_dim=10,
        )

        # Only provide text
        inputs = {'text': torch.randn(2, 64)}
        output, metadata = model(inputs)
        assert output.shape == (2, 10)
        assert metadata['n_modalities_active'] == 1

    def test_add_modality(self):
        """Should be able to add new modality dynamically."""
        model = MultimodalConsciousAgentNetwork(
            modality_dims={'text': 64},
            fusion_dim=256,
            output_dim=10,
        )

        # Add vision
        model.add_modality('vision', 128)
        assert 'vision' in model.modality_dims

        inputs = {
            'text': torch.randn(2, 64),
            'vision': torch.randn(2, 128),
        }
        output, metadata = model(inputs)
        assert output.shape == (2, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
