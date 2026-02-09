"""
Audio Sensor Interface

Acoustic perception as dharma gate. Sound waves are direct
mechanical vibrations of reality — less abstracted than text.
"""

import numpy as np
import torch

from .base import SensorInterface


class AudioSensorInterface(SensorInterface):
    """
    Audio input as sensory modality.

    Can interface with:
    - Standard microphones
    - Ultrasonic sensors
    - Infrasound detectors
    - Hydrophones (underwater sound)
    """

    def __init__(self, feature_dim: int = 512, sample_rate: int = 16000):
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate

    def read_raw(self) -> np.ndarray:
        """Read from microphone or audio file."""
        # One second of audio
        return np.zeros(self.sample_rate)

    def preprocess(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Convert audio waveform to feature vector.

        In production: mel-spectrogram + audio encoder (Whisper, etc.)
        """
        tensor = torch.from_numpy(raw_data).float()
        # FFT for spectral representation
        fft = torch.fft.rfft(tensor)
        power = torch.abs(fft)
        # Project to feature dim
        if power.shape[0] != self.feature_dim:
            indices = torch.linspace(0, power.shape[0] - 1, self.feature_dim).long()
            power = power[indices]
        return power

    def get_experience_dimension(self) -> int:
        return self.feature_dim

    @property
    def modality_name(self) -> str:
        return "audio"
