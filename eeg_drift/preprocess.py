"""
Preprocessing: notch filter, high-pass filter, ICA, ICLabel-based artifact
rejection with per-class thresholds (ported from the reference MATLAB code
and refined against our own manually-labeled components, see
scripts/run_ica_review.py and scripts/derive_thresholds.py).
"""

from __future__ import annotations

import mne
from mne.preprocessing import ICA

# Per-class ICLabel probability thresholds - starting point taken from the
# reference MATLAB code (idx > 0.48 eye, idx2 > 0.77 muscle/other, idx3 >
# 0.95 line noise). Update these once derive_thresholds.py has run against
# our own manually-labeled components.
ICLABEL_THRESHOLDS = {
    "eye blink": 0.50,          # supervisor: >50%, can go as low as 30% if needed
    "muscle artifact": 0.69,    # supervisor: >69% (not 0.77/0.90 — final call after discussion)
    "channel noise": 0.95,      # supervisor: >95% (detection unreliable below 90%)
}


def notch_and_highpass(
    raw: mne.io.BaseRaw,
    notch_freq: float = 60.0,
    highpass_freq: float = 0.5,
) -> mne.io.BaseRaw:
    """Port of removepowerline2.m (bandstop notch) + the 0.5 Hz high-pass filter."""
    # TODO: raw.notch_filter(freqs=[notch_freq], ...)
    # TODO: raw.filter(l_freq=highpass_freq, h_freq=None, ...)
    raise NotImplementedError


def fit_ica(
    raw: mne.io.BaseRaw,
    random_state: int = 42,
) -> ICA:
    """Fit ICA using Picard with extended-Infomax-equivalent settings."""
    ica = ICA(
        n_components=None,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw)
    return ica


def label_and_reject_components(
    raw: mne.io.BaseRaw,
    ica: ICA,
    thresholds: dict[str, float] = ICLABEL_THRESHOLDS,
) -> ICA:
    """
    Run ICLabel, mark components exceeding any per-class threshold as
    bad, and return the ICA object with .exclude populated (not yet
    applied to the data - call ica.apply(raw) separately).
    """
    # TODO: from mne_icalabel import label_components
    # TODO: ic_labels = label_components(raw, ica, method="iclabel")
    # TODO: apply per-class thresholds from `thresholds`, set ica.exclude
    raise NotImplementedError


def preprocess_subject(
    raw: mne.io.BaseRaw,
    montage_name: str = "standard_1005",
) -> mne.io.BaseRaw:
    """Full preprocessing chain for one subject's concatenated raw data."""
    # TODO: set montage, notch_and_highpass, fit_ica,
    #       label_and_reject_components, ica.apply(raw)
    raise NotImplementedError
