"""
BIDS -> mne.Raw loading. Everything downstream of BIDS conversion goes
through here, so it only ever depends on a valid BIDS tree existing at
some root path - not on how that tree was produced.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

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
    return sorted(
        p.name.removeprefix("sub-")
        for p in Path(bids_root).glob("sub-*")
        if p.is_dir()
    )


def concatenate_subject_runs(
    bids_root: Path,
    subject: str,
    task: str,
    runs: list[str],
) -> mne.io.BaseRaw:
    """
    Load and concatenate multiple runs for one subject into one continuous
    Raw. Runs missing from the BIDS tree are skipped with a warning rather
    than failing the whole subject, matching the reference MATLAB
    helpfun(i)'s "missing file, skip" behaviour.
    """
    raws = []
    for run in runs:
        try:
            raws.append(load_subject_raw(bids_root, subject, task, run))
        except FileNotFoundError:
            warnings.warn(f"sub-{subject} run-{run} (task-{task}) not found under {bids_root}, skipping.")

    if not raws:
        raise FileNotFoundError(f"No runs found for subject {subject} under {bids_root}")

    # concatenate_raws only returns a (raw, events) tuple when events_list is
    # passed; we never do, so this is always a plain Raw at runtime, but
    # mne's stubs aren't precise enough for the type checker to narrow that.
    return cast(mne.io.BaseRaw, mne.concatenate_raws(raws))
