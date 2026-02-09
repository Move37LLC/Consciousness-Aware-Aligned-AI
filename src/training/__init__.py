"""
Training Module — Dharma-Constrained Training Pipelines

Implements training as conscious agent dynamics:
- Standard task training with dharma regularizers
- Meditation phases (self-supervised internal observation)
- Death practice (graceful context degradation training)
- Scaffold architecture (gradual dharma internalization)

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part XI
"""

from .trainer import TokenMindTrainer
from .meditation import MeditationTrainer
from .loop import TokenMindTrainingLoop
from .scaffold import ScaffoldedEnlightenment

__all__ = [
    "TokenMindTrainer",
    "MeditationTrainer",
    "TokenMindTrainingLoop",
    "ScaffoldedEnlightenment",
]
