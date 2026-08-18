"""
Topomap and summary plots: average slope topomap, % subjects with
significant positive slope, group slope distributions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np


def _pick_info_subset(info: mne.Info, ch_names: list[str]) -> mne.Info:
    sel = mne.pick_channels(info["ch_names"], include=ch_names, ordered=True)
    return mne.pick_info(info, sel)


def plot_slope_topomap(channel_slopes: dict[str, float], info: mne.Info, title: str = ""):
    """Scalp topomap of per-channel slope values (Hz/hour), diverging colormap centered at 0."""
    ch_names = [ch for ch in info["ch_names"] if ch in channel_slopes]
    values = np.array([channel_slopes[ch] for ch in ch_names])
    info_subset = _pick_info_subset(info, ch_names)

    vlim = float(np.abs(values).max()) if len(values) else 1.0

    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = mne.viz.plot_topomap(
        values, info_subset, axes=ax, show=False, cmap="RdBu_r", vlim=(-vlim, vlim)
    )
    fig.colorbar(im, ax=ax, label="Slope (Hz/hour)")
    ax.set_title(title)
    return fig


def plot_pct_significant_topomap(channel_stats: dict[str, dict], info: mne.Info, title: str = ""):
    """
    Scalp topomap of % subjects with a positive slope per channel, using
    the "pct_positive" field from stats.group_test_slopes (point-estimate
    sign only, not a per-subject significance test).
    """
    ch_names = [ch for ch in info["ch_names"] if ch in channel_stats]
    values = np.array([channel_stats[ch]["pct_positive"] for ch in ch_names])
    info_subset = _pick_info_subset(info, ch_names)

    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = mne.viz.plot_topomap(
        values, info_subset, axes=ax, show=False, cmap="RdBu_r", vlim=(0, 100)
    )
    fig.colorbar(im, ax=ax, label="% subjects with positive slope")
    ax.set_title(title)
    return fig


def plot_inst_freq_traces(
    times_sec: np.ndarray,
    traces: dict[str, np.ndarray],
    fits: dict[str, dict],
    band_name: str,
    title: str = "",
):
    """
    One instantaneous-frequency-over-time line per channel (already
    decimated/smoothed - this is not meant for raw full-rate data) with its
    fitted robust-regression drift line overlaid, in the same style as the
    paper's own per-channel drift figures. Meant for a handful of
    representative channels (config/bands.yaml's per-band `channels`), not
    all 64 - that's what the group topomaps are for.

    traces: {ch_name: inst_freq array, aligned with times_sec}
    fits: {ch_name: fit_drift_slope(...) result} for the same channels
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    times_min = times_sec / 60.0

    for i, (ch_name, trace) in enumerate(traces.items()):
        color = colors[i % len(colors)]
        ax.plot(times_min, trace, linewidth=0.6, alpha=0.5, color=color)

        fit = fits[ch_name]
        fit_line = fit["intercept"] + fit["slope_per_hour"] * (times_sec / 3600.0)
        ax.plot(
            times_min, fit_line, linewidth=2, color=color,
            label=f"{ch_name}: {fit['slope_per_hour']:+.3f} Hz/hour",
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Inst. frequency (Hz)")
    ax.set_title(title or f"{band_name}: instantaneous frequency drift")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
