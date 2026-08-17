"""
Group-level hypothesis testing: is the per-channel drift slope significantly
different from zero across subjects, with FDR correction across channels.
"""

from __future__ import annotations

import numpy as np


def group_test_slopes(
    slopes_by_subject: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    slopes_by_subject: {subject_id: {channel_name: slope_per_hour}}
    Returns per-channel {mean_slope, t_stat, p_value, p_value_fdr}.
    """
    # TODO: one-sample t-test per channel across subjects,
    #       statsmodels.stats.multitest.fdrcorrection across channels
    raise NotImplementedError
