"""
Meta-Learning Module — Evolutionary Architecture Search with Dharma Fitness

Evolves neural network architectures toward enlightened functioning using
multi-objective optimization: task performance + dharma compliance.

Key insight: Evolution discovers that dharma principles IMPROVE performance.
Enlightenment is computationally efficient.

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part X
"""

from .genome import ArchitecturalGenome
from .fitness import DharmaFitnessEvaluator
from .orchestrator import MetaLearningOrchestrator, DharmaConsciousNetwork

__all__ = [
    "ArchitecturalGenome",
    "DharmaFitnessEvaluator",
    "MetaLearningOrchestrator",
    "DharmaConsciousNetwork",
]
