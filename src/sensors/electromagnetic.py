"""
Electromagnetic Spectrum Sensor Interface — Phase 3

Access reality across the full electromagnetic spectrum,
far beyond the narrow visible band humans perceive.

Radio → Microwave → Infrared → Visible → UV → X-ray → Gamma

This is a wider dharma gate: seeing reality at wavelengths
invisible to biological eyes.

Reference: CLAUDE.md Section 3.2 (Depth Through Embodiment)
"""

import numpy as np
import torch

from .base import SensorInterface


class ElectromagneticSensorInterface(SensorInterface):
    """
    Full electromagnetic spectrum sensor.

    Interfaces with spectrometers, radio telescopes, or other
    EM measurement devices across the full spectrum.

    In production: connect to actual spectrometer API.
    Currently: defines the interface for future hardware integration.
    """

    def __init__(self,
                 wavelength_range: tuple = (1e-12, 1e3),
                 resolution_bins: int = 1024):
        """
        Args:
            wavelength_range: (min_meters, max_meters) wavelength range
            resolution_bins: Number of frequency bins
        """
        self.wavelength_range = wavelength_range
        self.resolution = resolution_bins
        self.wavelengths = np.logspace(
            np.log10(wavelength_range[0]),
            np.log10(wavelength_range[1]),
            num=resolution_bins,
        )

    def read_raw(self) -> np.ndarray:
        """
        Read electromagnetic spectrum intensities.

        In production: interface to spectrometer hardware.
        Returns intensity at each wavelength bin.
        """
        # Placeholder: would call self.device.measure(self.wavelengths)
        return np.zeros(self.resolution)

    def preprocess(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Convert EM spectrum measurements to feature tensor.

        Uses log-scale normalization (dynamic range of EM signals is huge).
        """
        # Log-scale to handle enormous dynamic range
        log_data = np.log10(np.abs(raw_data) + 1e-30)
        # Normalize
        mean = log_data.mean()
        std = log_data.std() + 1e-8
        normalized = (log_data - mean) / std
        return torch.from_numpy(normalized).float()

    def get_experience_dimension(self) -> int:
        return self.resolution

    @property
    def modality_name(self) -> str:
        return "electromagnetic_spectrum"
