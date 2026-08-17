"""
BIDS -> mne.Raw loading. Everything downstream of BIDS conversion goes
through here, so it only ever depends on a valid BIDS tree existing at
some root path - not on how that tree was produced.
"""

from __future__ import annotations

from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids


def load_subject_raw(
    bids_root: Path,
    subject: str,
    task: str,
    run: str,
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Load a single subject/run as an mne.Raw object from a BIDS tree."""
    bp = BIDSPath(subject=subject, task=task, run=run, datatype="eeg", root=bids_root)
    raw = read_raw_bids(bp, verbose=False)
    if preload:
        raw.load_data()
    return raw


def list_subjects(bids_root: Path) -> list[str]:
    """Return sorted subject IDs (without 'sub-' prefix) found in a BIDS tree."""
    # TODO: glob bids_root for sub-* directories
    raise NotImplementedError


def concatenate_subject_runs(
    bids_root: Path,
    subject: str,
    task: str,
    runs: list[str],
) -> mne.io.BaseRaw:
    """Load and concatenate multiple runs for one subject into one continuous Raw."""
    # TODO: load_subject_raw per run, mne.concatenate_raws
    raise NotImplementedError
