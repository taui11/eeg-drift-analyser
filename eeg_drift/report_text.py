"""
Templated (not AI-generated) bullet-point summaries for per-subject QC
reports. Fixed sentence structures filled in with computed QC numbers -
deterministic, scannable, and defensible.
"""

from __future__ import annotations

import math


def _is_true(value) -> bool:
    """Handles both real booleans (dict built from a DataFrame row) and 'True'/'False' strings (from a CSV)."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _truthy_number(value) -> bool:
    """
    bool(value) for a number, but False for None/NaN - a missing value read
    back through pandas becomes float('nan'), and bool(nan) is True, which
    would otherwise render as a bogus "nan run(s) missing" warning.
    """
    if value is None:
        return False
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
    except TypeError:
        pass
    return bool(value)


def subject_summary_bullets(qc: dict) -> list[str]:
    """
    qc is one row from the qc_summary table (see qc.compute_qc_row),
    optionally merged with qc.flag_outliers()'s "<metric>_outlier" columns
    (if present, an extra bullet calls out which metrics were flagged).

    NOTE: no "prior discussion" thread was available to match against - this
    is a reasonable default structure. Tell me if you had something more
    specific in mind and I'll adjust it.
    """
    bullets = []

    duration_min = qc["duration_sec"] / 60.0
    bullets.append(f"Recording duration: {duration_min:.1f} min")

    n_runs_missing = qc.get("n_runs_missing")
    if _truthy_number(n_runs_missing):
        bullets.append(f"WARNING: {int(n_runs_missing)} run(s) missing (expected {qc['expected_runs']})")

    bullets.append(f"Channels: {qc['n_channels']} ({qc['n_flat_channels']} flat)")
    if _truthy_number(qc["n_flat_channels"]):
        bullets.append(f"WARNING: {int(qc['n_flat_channels'])} flat channel(s) detected")

    bullets.append(
        f"ICA: {qc['n_components_excluded']}/{qc['n_ica_components']} components excluded "
        f"({float(qc['pct_components_excluded']):.1f}%)"
    )

    bullets.append(f"Residual 60 Hz power: {float(qc['residual_60hz_power']):.2e}")

    outlier_metrics = sorted(
        col[: -len("_outlier")]
        for col, value in qc.items()
        if col.endswith("_outlier") and col != "any_outlier" and _is_true(value)
    )
    if outlier_metrics:
        bullets.append("WARNING: outlier vs. group mean on " + ", ".join(outlier_metrics))

    return bullets
