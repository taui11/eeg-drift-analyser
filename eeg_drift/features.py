"""
Bandpass + Hilbert-transform feature extraction.

Direct Python port of the reference MATLAB `extracthilbert.m` inner loop:
    inst_freq  = Fs/(2*pi) * diff(unwrap(angle(hilbert(x))))
    inst_power = abs(hilbert(x)).^2
both smoothed with a moving-average window.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, matching MATLAB's filtfilt(butter(...))."""
    nyq = sfreq / 2
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)


def instantaneous_frequency(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Instantaneous frequency via Hilbert transform.
    Output is padded (edge-repeat) back to the original length, matching
    the MATLAB reference's `inst_freq = [inst_freq inst_freq(end)]`.
    """
    analytic = hilbert(data, axis=-1)
    phase = np.unwrap(np.angle(analytic), axis=-1)
    inst_freq = sfreq / (2 * np.pi) * np.diff(phase, axis=-1)
    pad_width = [(0, 0)] * (inst_freq.ndim - 1) + [(0, 1)]
    inst_freq = np.pad(inst_freq, pad_width, mode="edge")
    return inst_freq


def instantaneous_power(data: np.ndarray) -> np.ndarray:
    """Instantaneous power via Hilbert transform: |hilbert(x)|^2."""
    analytic = hilbert(data, axis=-1)
    return np.abs(analytic) ** 2


def smooth_moving_average(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Centered moving-average smoothing, matching MATLAB's smoothdata(x, 'movmean', window)."""
    window_samples = max(1, int(window_samples))
    if window_samples == 1:
        return data

    kernel = np.ones(window_samples) / window_samples

    def _smooth_1d(x):
        return np.convolve(x, kernel, mode="same")

    return np.apply_along_axis(_smooth_1d, axis=-1, arr=data)


def extract_band_features(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    filter_order: int = 4,
    smooth_window_ms: float = 100.0,
) -> dict[str, np.ndarray]:
    """
    Full feature-extraction chain for one band, one array of shape
    (n_channels, n_samples) or (n_samples,).

    Returns dict with 'inst_freq' and 'inst_power', both smoothed,
    same shape as input.
    """
    filtered = bandpass_filter(data, sfreq, fmin, fmax, order=filter_order)

    inst_freq = instantaneous_frequency(filtered, sfreq)
    inst_power = instantaneous_power(filtered)

    window_samples = int(sfreq * smooth_window_ms / 1000.0)
    inst_freq = smooth_moving_average(inst_freq, window_samples)
    inst_power = smooth_moving_average(inst_power, window_samples)

    return {"inst_freq": inst_freq, "inst_power": inst_power}
