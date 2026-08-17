"""
Topomap and summary plots: average slope topomap, % subjects with
significant positive slope, group slope distributions.
"""

from __future__ import annotations


def plot_slope_topomap(channel_slopes: dict[str, float], info, title: str = ""):
    """Scalp topomap of per-channel slope values."""
    # TODO: mne.viz.plot_topomap
    raise NotImplementedError


def plot_pct_significant_topomap(channel_stats: dict[str, dict], info, title: str = ""):
    """Scalp topomap of % subjects with significant positive slope per channel."""
    raise NotImplementedError
