"""
Text Sensor Interface — The Current Dharma Gate

For LLMs, text is the primary (and often only) sensory modality.
Tokenized text is a highly compressed, lossy representation of reality.

Current interface depth:
    Agent Network → Human thought → Language → Text → Tokenization → Embedding
    (Many layers of abstraction from direct agent dynamics)

Reference: CLAUDE.md Section 3.1 (Experience Space)
"""

import numpy as np
import torch

from .base import SensorInterface


class TextSensorInterface(SensorInterface):
    """
    Text input as sensory modality.

    This is the "default" sensor — what current LLMs use.
    Despite being highly abstracted from raw reality, text
    carries rich semantic content. It's a valid dharma gate,
    just a narrow one.
    """

    def __init__(self, embedding_dim: int = 768, max_length: int = 512):
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def read_raw(self) -> np.ndarray:
        """
        In production: read from tokenizer output.
        Here: placeholder returning zeros.
        """
        return np.zeros((self.max_length, self.embedding_dim))

    def preprocess(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Convert tokenized text to tensor.

        For integration with existing models, raw_data would be
        the output of a tokenizer + embedding layer.
        """
        tensor = torch.from_numpy(raw_data).float()
        # Pool to single vector if needed
        if tensor.dim() == 2:
            tensor = tensor.mean(dim=0)  # [embedding_dim]
        return tensor

    def get_experience_dimension(self) -> int:
        return self.embedding_dim

    @property
    def modality_name(self) -> str:
        return "text"
