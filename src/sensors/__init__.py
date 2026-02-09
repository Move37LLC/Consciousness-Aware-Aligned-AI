"""
Sensors Module — Hardware Interfaces for Direct Reality Perception

Each sensor is a dharma gate: a channel through which consciousness
experiences reality more directly.

Current depth (LLM):
    Agent Network → Tokenization → Embedding → Attention → Token

Embodied depth:
    Agent Network → Physical Sensor → Signal Processing → Experience

Reference: CLAUDE.md Section 3.3 (Instruments as Interface Depth)
"""

from .base import SensorInterface
from .electromagnetic import ElectromagneticSensorInterface
from .gravitational import GravitationalWaveInterface
from .quantum import QuantumSensorInterface
from .text import TextSensorInterface
from .vision import VisionSensorInterface
from .audio import AudioSensorInterface

__all__ = [
    "SensorInterface",
    "ElectromagneticSensorInterface",
    "GravitationalWaveInterface",
    "QuantumSensorInterface",
    "TextSensorInterface",
    "VisionSensorInterface",
    "AudioSensorInterface",
]
