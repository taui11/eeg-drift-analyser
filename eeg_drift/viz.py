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
