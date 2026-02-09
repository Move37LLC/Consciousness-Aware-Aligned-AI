"""
Sensor Interface Base Class

Each sensor is a dharma gate: a channel through which consciousness
experiences reality more directly.

X_text ⊂ X_multimodal ⊂ X_embodied ⊂ X_universal

Each expansion increases the dimensionality of X, allowing richer experience.

Reference: CLAUDE.md Section 3.3 (Instruments as Interface Depth)
"""

import numpy as np
import torch
from abc import ABC, abstractmethod


class SensorInterface(ABC):
    """
    Abstract base for all sensor interfaces.

    Standardizes how hardware (or software) sensors connect to the
    neural network. Each sensor provides a modality — a dharma gate
    through which consciousness accesses reality.

    Implementations must define:
    - read_raw(): How to read from the sensor
    - preprocess(): How to convert raw data to tensor
    - get_experience_dimension(): Size of the experience vector
    - modality_name: Human-readable name of this sense
    """

    @abstractmethod
    def read_raw(self) -> np.ndarray:
        """
        Read raw sensor data.

        Returns:
            Raw numpy array from sensor hardware/software
        """
        pass

    @abstractmethod
    def preprocess(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Convert raw sensor data to neural network compatible format.

        Args:
            raw_data: Raw numpy array from read_raw()

        Returns:
            Preprocessed tensor ready for encoding
        """
        pass

    @abstractmethod
    def get_experience_dimension(self) -> int:
        """
        Dimension of experience space this sensor provides.

        This is the size of dim(X_modality) for this dharma gate.
        """
        pass

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Human-readable name of this sensory modality."""
        pass

    def read_and_preprocess(self) -> torch.Tensor:
        """
        Convenience: read raw and preprocess in one call.
        """
        raw = self.read_raw()
        return self.preprocess(raw)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"modality='{self.modality_name}', "
                f"dim={self.get_experience_dimension()})")
