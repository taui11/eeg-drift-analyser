"""
Per-subject QC metrics, aggregated across subjects into results/qc/qc_summary.csv.
Automatable checks only (flat channels, duration, ICA rejection counts,
residual line noise) - not a substitute for manual review, but enough to
flag outlier subjects worth a closer look.
"""

from __future__ import annotations

import numpy as np
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
    data_before = raw_before.get_data()
    flat_channels = int(np.sum(np.ptp(data_before, axis=1) < 1e-13))

    n_components = ica.n_components_
    n_excluded = len(ica.exclude)

    psd = raw_after.compute_psd(fmin=55, fmax=65, verbose=False)
    freqs = psd.freqs
    mean_power = psd.get_data().mean(axis=0)
    residual_60hz_power = float(mean_power[np.argmin(np.abs(freqs - 60.0))])

    return {
        "subject": subject,
        "expected_runs": expected_runs,
        "n_runs_loaded": n_runs_loaded,
        "n_runs_missing": (expected_runs - n_runs_loaded) if n_runs_loaded is not None else None,
        "duration_sec": float(raw_before.times[-1]),
        "n_channels": len(raw_before.ch_names),
        "n_flat_channels": flat_channels,
        "n_ica_components": n_components,
        "n_components_excluded": n_excluded,
        "pct_components_excluded": (100.0 * n_excluded / n_components) if n_components else None,
        "excluded_component_idx": list(ica.exclude),
        "residual_60hz_power": residual_60hz_power,
    }


def flag_outliers(qc_summary, n_std: float = 2.0):
    """Flag subjects whose automatable QC metrics fall outside n_std of the group mean."""
    # TODO: operate on a pandas DataFrame of qc rows
    raise NotImplementedError
