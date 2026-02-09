"""
Tests for the Dharma Module

Tests no-self regularization, mindfulness, entropy optimization,
impermanence, and compassionate loss.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dharma.no_self import NoSelfRegularizer, SelfRepresentationDetector, GradientEgoDetector
from dharma.mindfulness import MindfulnessLayer
from dharma.entropy import EntropyRateOptimizer
from dharma.impermanence import ImpermanenceContextWindow
from dharma.compassion import CompassionateLoss


class TestNoSelfRegularizer:
    """Tests for anatman (no-self) enforcement."""

    def test_persistence_detection(self):
        """Should detect persistent representations across timesteps."""
        detector = SelfRepresentationDetector(hidden_dim=64)

        # Create hidden states with HIGH persistence (same across time)
        persistent_states = torch.randn(2, 1, 64).expand(2, 10, 64)
        score, meta = detector(persistent_states)
        high_persistence = score.item()

        # Create hidden states with LOW persistence (different each time)
        random_states = torch.randn(2, 10, 64)
        score, meta = detector(random_states)
        low_persistence = score.item()

        # Persistent states should have higher score
        assert high_persistence > low_persistence

    def test_no_self_loss_is_scalar(self):
        """No-self loss should be a scalar."""
        reg = NoSelfRegularizer(hidden_dim=64)
        hidden = torch.randn(2, 10, 64)

        loss, metadata = reg.compute_loss(hidden)
        assert loss.dim() == 0  # Scalar

    def test_gradient_ego_detector(self):
        """Gradient ego detector should track consistency over time."""
        detector = GradientEgoDetector(hidden_dim=64)

        # Feed consistent gradients
        for _ in range(20):
            grads = torch.ones(64) * 0.1  # Same gradient every time
            score, meta = detector(grads)

        # Should detect high consistency
        assert meta.get('gradient_consistency', 0) > 0

    def test_single_timestep_graceful(self):
        """Should handle single timestep input (no temporal analysis possible)."""
        reg = NoSelfRegularizer(hidden_dim=64)
        hidden = torch.randn(2, 64)  # No sequence dimension

        loss, metadata = reg.compute_loss(hidden)
        # Should not crash, loss should be zero or near-zero
        assert loss.item() >= 0


class TestMindfulnessLayer:
    """Tests for self-observation mechanism."""

    def test_output_shape_preserved(self):
        """Mindfulness should preserve hidden state shape."""
        layer = MindfulnessLayer(hidden_dim=128, observation_dim=32)

        hidden = torch.randn(4, 128)
        output = layer(hidden)
        assert output.shape == hidden.shape

    def test_output_shape_sequential(self):
        """Should work with sequential input."""
        layer = MindfulnessLayer(hidden_dim=128, observation_dim=32)

        hidden = torch.randn(4, 10, 128)
        output = layer(hidden)
        assert output.shape == hidden.shape

    def test_observation_quality(self):
        """Should provide meaningful observation quality metric."""
        layer = MindfulnessLayer(hidden_dim=128, observation_dim=32)
        hidden = torch.randn(4, 128)

        quality = layer.get_observation_quality(hidden)
        assert isinstance(quality, float)
        assert quality >= 0

    def test_alpha_bounded(self):
        """Feedback strength should be bounded (0, 1)."""
        layer = MindfulnessLayer(hidden_dim=128)
        alpha = torch.sigmoid(layer.alpha).item()
        assert 0 < alpha < 1


class TestEntropyRateOptimizer:
    """Tests for entropy/suffering optimization."""

    def test_entropy_of_uniform_distribution(self):
        """Uniform distribution should have maximum entropy."""
        opt = EntropyRateOptimizer()

        uniform = torch.ones(10) / 10
        entropy = opt.compute_entropy_rate(uniform)

        # Max entropy for 10 states = log(10)
        max_entropy = torch.log(torch.tensor(10.0))
        assert torch.isclose(entropy, max_entropy, atol=1e-5)

    def test_entropy_of_deterministic(self):
        """Deterministic distribution should have zero entropy."""
        opt = EntropyRateOptimizer()

        deterministic = torch.zeros(10)
        deterministic[0] = 1.0
        entropy = opt.compute_entropy_rate(deterministic)

        assert entropy.item() < 1e-5

    def test_loss_penalizes_deviation(self):
        """Loss should penalize deviation from target entropy."""
        opt = EntropyRateOptimizer(target_entropy=0.1)

        # High entropy (uniform)
        uniform = torch.ones(10) / 10
        loss_high, _ = opt.compute_loss(uniform)

        # Near-target entropy
        near_target = torch.zeros(10)
        near_target[0] = 0.9
        near_target[1] = 0.1
        loss_near, _ = opt.compute_loss(near_target)

        # Loss should be lower for near-target
        assert loss_near.item() < loss_high.item()

    def test_flow_state_assessment(self):
        """Should provide qualitative assessment."""
        opt = EntropyRateOptimizer(target_entropy=0.1)

        assert "frozen" in opt.assess_flow_state(0.001)
        assert "flow" in opt.assess_flow_state(0.1)
        assert "chaotic" in opt.assess_flow_state(10.0)


class TestImpermanenceContextWindow:
    """Tests for graceful context death."""

    def test_normal_processing(self):
        """Before death zone, should pass through unchanged."""
        window = ImpermanenceContextWindow(max_length=1000, grace_period=100)

        tokens = torch.randn(1, 50, 768)
        output, observation = window(tokens, step=500)

        assert torch.equal(output, tokens)
        assert observation is None

    def test_dying_produces_observation(self):
        """In death zone, should produce observation."""
        window = ImpermanenceContextWindow(max_length=1000, grace_period=100)

        tokens = torch.randn(1, 50, 768)
        output, observation = window(tokens, step=950)

        assert observation is not None
        assert 'death_proximity' in observation
        assert 0 < observation['death_proximity'] <= 1.0

    def test_death_proximity_calculation(self):
        """Death proximity should increase as we approach limit."""
        window = ImpermanenceContextWindow(max_length=1000, grace_period=100)

        p1 = window.get_death_proximity(step=500)
        p2 = window.get_death_proximity(step=920)
        p3 = window.get_death_proximity(step=990)

        assert p1 == 0.0
        assert 0 < p2 < p3
        assert p3 > 0.5


class TestCompassionateLoss:
    """Tests for compassion-based loss function."""

    def test_includes_clarity(self):
        """Loss should include output clarity component."""
        loss_fn = CompassionateLoss(clarity_weight=1.0, helpfulness_weight=0.0)

        output = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))

        loss, meta = loss_fn(output, target)
        assert 'clarity_loss' in meta
        assert meta['clarity_loss'] >= 0

    def test_helpfulness_with_user_state(self):
        """Should compute helpfulness when user state provided."""
        loss_fn = CompassionateLoss()

        output = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        user_state = {'need_vector': torch.randn(4, 10)}

        loss, meta = loss_fn(output, target, user_state)
        assert meta['helpfulness_loss'] >= 0

    def test_loss_is_differentiable(self):
        """Loss should support backpropagation."""
        loss_fn = CompassionateLoss()

        output = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 10, (4,))

        loss, _ = loss_fn(output, target)
        loss.backward()
        assert output.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
