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
    """
    Centered moving-average smoothing, matching MATLAB's smoothdata(x, 'movmean', window).

    Edge-pads with the boundary value before convolving (equivalent to
    plain np.convolve(..., mode="same") everywhere except the first/last
    ~window_samples/2 samples). Plain "same"-mode convolution implicitly
    zero-pads outside the array, which pulls those edge samples toward
    zero - e.g. for an ~10 Hz signal, the very first smoothed sample comes
    out as roughly half the true value instead of ~10 Hz. That artifact
    survives averaging across subjects (it's systematic, not noise) and,
    since it sits at the very edge of the time range, is a high-leverage
    outlier for the drift slope regression - worth fixing at the source
    rather than trimming it out downstream.
    """
    window_samples = max(1, int(window_samples))
    if window_samples == 1:
        return data

    kernel = np.ones(window_samples) / window_samples
    pad_left = window_samples // 2
    pad_right = window_samples - 1 - pad_left

    def _smooth_1d(x):
        x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
        return np.convolve(x_padded, kernel, mode="valid")

    return np.apply_along_axis(_smooth_1d, axis=-1, arr=data)


# Hilbert-transform-based instantaneous frequency is unreliable right at
# the filter/transform boundary - checked across 30 real subjects and
# found occasional physically-nonsensical values (tens of Hz, or negative)
# within the first/last ~20 raw samples (~130 ms @ 160 Hz). This is a
# property of the boundary samples themselves (near-zero analytic-signal
# amplitude makes phase, and hence its derivative, ill-defined there), not
# something smoothing's edge handling can fix - averaging can only work
# with what's there, and edge-padding a genuinely bad boundary sample just
# replicates it into the output. Trimmed symmetrically with margin.
EDGE_TRIM_SEC = 0.2


def extract_band_features(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    filter_order: int = 4,
    smooth_window_ms: float = 100.0,
    edge_trim_sec: float = EDGE_TRIM_SEC,
) -> dict[str, np.ndarray]:
    """
    Full feature-extraction chain for one band, one array of shape
    (n_channels, n_samples) or (n_samples,).

    Returns dict with 'inst_freq' and 'inst_power' (smoothed, and trimmed
    by edge_trim_sec at each end - see EDGE_TRIM_SEC), plus
    'n_trimmed_start' (samples dropped from the start) so callers that
    build their own time axis from the original data can stay aligned,
    e.g. `times[n_trimmed_start : len(times) - n_trimmed_start]`.
    """
    filtered = bandpass_filter(data, sfreq, fmin, fmax, order=filter_order)

    inst_freq = instantaneous_frequency(filtered, sfreq)
    inst_power = instantaneous_power(filtered)

    window_samples = int(sfreq * smooth_window_ms / 1000.0)
    inst_freq = smooth_moving_average(inst_freq, window_samples)
    inst_power = smooth_moving_average(inst_power, window_samples)

    n_trim = int(round(sfreq * edge_trim_sec))
    if n_trim > 0:
        inst_freq = inst_freq[..., n_trim:-n_trim]
        inst_power = inst_power[..., n_trim:-n_trim]

    return {"inst_freq": inst_freq, "inst_power": inst_power, "n_trimmed_start": n_trim}
