"""
Templated (not AI-generated) bullet-point summaries for per-subject QC
reports. Fixed sentence structures filled in with computed QC numbers -
deterministic, scannable, and defensible.
"""

from __future__ import annotations


def subject_summary_bullets(qc: dict) -> list[str]:
    """qc is one row from the qc_summary table (see qc.compute_qc_row)."""
    # TODO: fixed f-string bullets + warning-flagged outliers, see prior
    #       discussion for the intended structure
    raise NotImplementedError
