"""
Vision Sensor Interface

Visual perception as dharma gate. Closer to raw reality than text,
but still filtered through human-designed cameras and encoders.
"""

import numpy as np
import torch

from .base import SensorInterface


class VisionSensorInterface(SensorInterface):
    """
    Visual input as sensory modality.

    Can interface with:
    - Standard cameras (RGB)
    - Infrared cameras
    - Multispectral cameras
    - Microscope cameras
    - Telescope cameras
    """

    def __init__(self, feature_dim: int = 1024, image_size: int = 224):
        self.feature_dim = feature_dim
        self.image_size = image_size

    def read_raw(self) -> np.ndarray:
        """Read from camera or image file."""
        return np.zeros((3, self.image_size, self.image_size))

    def preprocess(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Convert image to feature vector.

        In production: pass through vision encoder (ViT, CLIP, etc.)
        to get feature_dim-dimensional representation.
        """
        tensor = torch.from_numpy(raw_data).float()
        # Flatten and project (placeholder for actual vision encoder)
        flat = tensor.flatten()
        # Simple projection to feature dim
        if flat.shape[0] != self.feature_dim:
            # Pad or truncate
            if flat.shape[0] > self.feature_dim:
                flat = flat[:self.feature_dim]
            else:
                flat = torch.nn.functional.pad(flat, (0, self.feature_dim - flat.shape[0]))
        return flat

    def get_experience_dimension(self) -> int:
        return self.feature_dim

    @property
    def modality_name(self) -> str:
        return "vision"
