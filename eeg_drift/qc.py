"""
Per-subject QC metrics, aggregated across subjects into results/qc/qc_summary.csv.
Automatable checks only (flat channels, duration, ICA rejection counts,
residual line noise) - not a substitute for manual review, but enough to
flag outlier subjects worth a closer look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mne

# Columns that are identifiers/counts-of-context rather than "quality"
# metrics - flagging a subject as an outlier because they have a different
# subject ID or a different expected_runs count would be meaningless.
_NON_METRIC_COLUMNS = {"subject", "expected_runs"}


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


def flag_outliers(qc_summary: pd.DataFrame, n_std: float = 2.0) -> pd.DataFrame:
    """
    Flag subjects whose automatable QC metrics fall outside n_std of the
    group mean. Returns a copy of qc_summary with one added boolean column
    per numeric metric ("<metric>_outlier") plus "any_outlier" - input is
    not mutated.
    """
    result = qc_summary.copy()

    numeric_cols = [
        col
        for col in qc_summary.columns
        if col not in _NON_METRIC_COLUMNS and pd.api.types.is_numeric_dtype(qc_summary[col])
    ]

    outlier_cols = []
    for col in numeric_cols:
        mean = qc_summary[col].mean()
        std = qc_summary[col].std()
        flag_col = f"{col}_outlier"

        if not std or pd.isna(std):
            result[flag_col] = False
        else:
            result[flag_col] = (qc_summary[col] - mean).abs() > n_std * std
        outlier_cols.append(flag_col)

    result["any_outlier"] = result[outlier_cols].any(axis=1) if outlier_cols else False
    return result
