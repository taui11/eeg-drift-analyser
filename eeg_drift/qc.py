"""
Per-subject QC metrics, aggregated across subjects into results/qc/qc_summary.csv.
Automatable checks only (flat channels, duration, ICA rejection counts,
residual line noise) - not a substitute for manual review, but enough to
flag outlier subjects worth a closer look.
"""

from __future__ import annotations

import mne


def compute_qc_row(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    ica,
    subject: str,
    expected_runs: int = 14,
    n_runs_loaded: int | None = None,
) -> dict:
    """
    Compute one row of QC metrics for a single subject.
    Returns a flat dict suitable for appending to qc_summary.csv.
    """
    # TODO: flat/dead channel count, duration, n components rejected,
    #       per-class rejection percentages, residual 60Hz power,
    #       % data retained
    raise NotImplementedError


def flag_outliers(qc_summary, n_std: float = 2.0):
    """Flag subjects whose automatable QC metrics fall outside n_std of the group mean."""
    # TODO: operate on a pandas DataFrame of qc rows
    raise NotImplementedError
