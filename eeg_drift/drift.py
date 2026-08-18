"""
Slope estimation over time for instantaneous-frequency / instantaneous-power
traces, using robust linear regression (Tukey bisquare weights), matching
the `robustfit` approach referenced in the paper.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

# Analysis window from the reference MATLAB code: samples 20000-245000 @
# 160 Hz (~min 2 to min 25, skips the startup transient). Expressed in
# seconds so it's correct even if a cleaned file ends up at a different
# sample rate. Shared by run_drift_analysis.py and build_reports.py so the
# group topomaps and the per-subject trace plots analyze the same window.
ANALYSIS_WINDOW_SAMPLES = (20_000, 245_000)
REFERENCE_FS = 160.0

# Rate traces are decimated to before plotting/fitting (matches
# fit_drift_slope's own default) - shared so a plotted line and its fitted
# slope always come from the exact same decimated samples.
TRACE_DECIMATE_HZ = 1.0


def analysis_window_seconds() -> tuple[float, float]:
    return (ANALYSIS_WINDOW_SAMPLES[0] / REFERENCE_FS, ANALYSIS_WINDOW_SAMPLES[1] / REFERENCE_FS)


def fit_drift_slope(
    trace: np.ndarray,
    sfreq: float,
    decimate_to_hz: float | None = 1.0,
) -> dict[str, float]:
    """
    Fit a robust linear trend to a single 1-D time series (e.g. one
    channel's smoothed instantaneous-frequency trace) and return the slope
    in units-per-hour.

    decimate_to_hz: if set, the trace is downsampled (simple striding) to
    roughly this rate before fitting, since drift is a slow (minutes-hours)
    process and fitting at full sample rate is wasted compute.
    """
    n = len(trace)
    t_sec = np.arange(n) / sfreq

    if decimate_to_hz is not None and decimate_to_hz < sfreq:
        step = max(1, int(round(sfreq / decimate_to_hz)))
        trace = trace[::step]
        t_sec = t_sec[::step]

    t_hours = t_sec / 3600.0

    X = sm.add_constant(t_hours)
    model = sm.RLM(trace, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()

    return {
        "slope_per_hour": float(result.params[1]),
        "intercept": float(result.params[0]),
        "p_value": float(result.pvalues[1]),
        "n_points_fit": int(len(trace)),
    }


def fit_drift_slopes_all_channels(
    traces: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    decimate_to_hz: float | None = 1.0,
) -> dict[str, dict[str, float]]:
    """
    traces: shape (n_channels, n_samples).
    Returns {channel_name: fit_drift_slope(...) result}.
    """
    assert traces.shape[0] == len(ch_names), "traces/ch_names length mismatch"

    return {
        ch_name: fit_drift_slope(traces[i], sfreq, decimate_to_hz=decimate_to_hz)
        for i, ch_name in enumerate(ch_names)
    }
