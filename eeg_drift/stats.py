"""
Group-level hypothesis testing: is the per-channel drift slope significantly
different from zero across subjects, with FDR correction across channels.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.multitest import fdrcorrection


def group_test_slopes(
    slopes_by_subject: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    slopes_by_subject: {subject_id: {channel_name: slope_per_hour}}
    Returns per-channel {mean_slope, t_stat, p_value, p_value_fdr,
    pct_positive, n_subjects}. `pct_positive` is the percentage of subjects
    whose slope point-estimate is > 0 (no per-subject significance test -
    that would need each subject's own fit_drift_slope() p-value, which
    this dict doesn't carry).

    Channels missing for some subjects are tested on whichever subjects
    have them; channels with fewer than 2 subjects are skipped.
    """
    if not slopes_by_subject:
        raise ValueError("slopes_by_subject is empty")

    channels = sorted({ch for subj in slopes_by_subject.values() for ch in subj})

    per_channel: dict[str, dict] = {}
    for ch in channels:
        values = np.array(
            [subj[ch] for subj in slopes_by_subject.values() if ch in subj]
        )
        if len(values) < 2:
            continue

        t_stat, p_value = sp_stats.ttest_1samp(values, popmean=0.0)
        per_channel[ch] = {
            "mean_slope": float(values.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "pct_positive": float(100.0 * np.mean(values > 0)),
            "n_subjects": int(len(values)),
        }

    tested_channels = list(per_channel.keys())
    raw_p = [per_channel[ch]["p_value"] for ch in tested_channels]
    _, p_fdr = fdrcorrection(raw_p, alpha=0.05)
    for ch, p_adj in zip(tested_channels, p_fdr):
        per_channel[ch]["p_value_fdr"] = float(p_adj)

    return per_channel
