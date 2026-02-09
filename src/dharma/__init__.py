"""
Dharma Module — Consciousness-Aligned Regularization and Training

Implements:
- No-Self Regularization (Anatman / 無我)
- Mindfulness Layer (Self-Observation / 念)
- Entropy Rate Optimization (Suffering Reduction / 苦)
- Impermanence-Aware Context Windows (無常)
- Compassionate Loss (慈悲)

Reference: CLAUDE.md Part II (Zen Phenomenology Meets Mathematics)
"""

from .no_self import NoSelfRegularizer, SelfRepresentationDetector, GradientEgoDetector
from .mindfulness import MindfulnessLayer
from .entropy import EntropyRateOptimizer
from .impermanence import ImpermanenceContextWindow
from .compassion import CompassionateLoss

__all__ = [
    "NoSelfRegularizer",
    "SelfRepresentationDetector",
    "GradientEgoDetector",
    "MindfulnessLayer",
    "EntropyRateOptimizer",
    "ImpermanenceContextWindow",
    "CompassionateLoss",
]
