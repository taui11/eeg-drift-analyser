"""
Preprocessing: notch filter, high-pass filter, ICA, ICLabel-based artifact
rejection with per-class thresholds (ported from the reference MATLAB code
and refined against our own manually-labeled components, see
scripts/run_ica_review.py and scripts/derive_thresholds.py).
"""

from __future__ import annotations

import warnings

import mne
from mne.preprocessing import ICA

# Per-class ICLabel probability thresholds, derived empirically by the
# supervisor from a manual review of S001's components (see
# scripts/run_ica_review.py history / email thread with Paul). These
# supersede the reference MATLAB code's idx>0.48/idx2>0.77/idx3>0.95 values.
ICLABEL_THRESHOLDS = {
    "eye blink": 0.50,          # supervisor: >50%, can go as low as 30% if needed for recall
    "muscle artifact": 0.69,    # supervisor: >69% (not the MATLAB reference's 0.77)
    "channel noise": 0.95,      # supervisor: >95% (detection unreliable below 90%, don't lower casually)
}

# OPEN QUESTION (unresolved as of the supervisor email thread): the reference
# MATLAB code also rejected a `line noise` class at >0.95, which doesn't
# appear in the supervisor's revised set above. Rather than silently guessing
# a threshold or silently dropping the class, label_and_reject_components
# warns at runtime if it sees a confidently-predicted "line noise" component
# so it surfaces during review instead of being invisibly kept. Confirm with
# the supervisor/teammate whether it should be added to ICLABEL_THRESHOLDS.
LINE_NOISE_WARN_PROBA = 0.5


def notch_and_highpass(
    raw: mne.io.BaseRaw,
    notch_freq: float = 60.0,
    highpass_freq: float = 0.5,
) -> mne.io.BaseRaw:
    """Port of removepowerline2.m (bandstop notch) + the 0.5 Hz high-pass filter.

    Uses method="iir" with a Butterworth design to match the reference
    MATLAB filtfilt(butter(...)) behaviour (zero-phase, forward-backward).
    """
    raw = raw.copy()

    raw.notch_filter(
        freqs=[notch_freq],
        notch_widths=2.0,  # +/-1 Hz half-power points, matching removepowerline2.m
        method="iir",
        iir_params=dict(order=2, ftype="butter"),
        verbose=False,
    )
    raw.filter(
        l_freq=highpass_freq,
        h_freq=None,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )

    return raw


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
    from mne_icalabel import label_components

    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    exclude = []
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        if label in thresholds and prob > thresholds[label]:
            exclude.append(idx)
        elif label == "line noise" and prob > LINE_NOISE_WARN_PROBA:
            warnings.warn(
                f"IC{idx:03d} predicted 'line noise' (p={prob:.2f}) but no "
                "threshold is defined for this class - not excluded. See "
                "the OPEN QUESTION note above ICLABEL_THRESHOLDS."
            )

    ica.exclude = sorted(exclude)
    return ica


def preprocess_subject(
    raw: mne.io.BaseRaw,
    montage_name: str = "standard_1005",
    random_state: int = 42,
) -> tuple[mne.io.BaseRaw, ICA]:
    """
    Full preprocessing chain for one subject's concatenated raw data:
    montage -> notch/high-pass -> ICA fit -> ICLabel rejection -> apply.

    Returns (cleaned_raw, ica) - the ica object is returned alongside the
    cleaned data so callers (e.g. scripts/run_preprocess.py) can compute
    QC metrics (rejection counts/labels) via eeg_drift.qc.compute_qc_row
    without having to refit ICA. `raw` itself is never mutated.
    """
    raw = raw.copy()
    raw.rename_channels(lambda name: name.strip("."))  # PhysioNet EDF channel names have trailing dots
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    raw = notch_and_highpass(raw)

    ica = fit_ica(raw, random_state=random_state)
    label_and_reject_components(raw, ica)

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean, ica
