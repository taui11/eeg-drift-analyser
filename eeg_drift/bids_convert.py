"""
Raw EDF (PhysioNet) -> BIDS conversion.

Owned by: [teammate]

Everything downstream (io.py onward) only depends on there being a valid
BIDS tree at data/bids/ - it doesn't care how it got there. Feel free to
build this out with argparse however you like; the only contract is the
output layout (standard BIDS, see mne_bids.write_raw_bids).

Suggested CLI shape (see scripts/run_bids_convert.py):
    python scripts/run_bids_convert.py --raw-dir data/raw --bids-root data/bids
"""

from __future__ import annotations

from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids

READERS = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".fif": mne.io.read_raw_fif,
    ".set": mne.io.read_raw_eeglab,
}


def convert_file_to_bids(
    raw_path: Path,
    bids_root: Path,
    subject: str,
    task: str,
    run: str,
) -> BIDSPath:
    """Detect filetype, load, and write into BIDS if not already there."""
    bids_path = BIDSPath(subject=subject, task=task, run=run, datatype="eeg", root=bids_root)

    if bids_path.fpath.exists():
        return bids_path  # already converted

    ext = raw_path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported file type: {ext}")

    raw = READERS[ext](raw_path, preload=False)
    write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format="EDF")
    return bids_path


def convert_dataset(raw_dir: Path, bids_root: Path) -> None:
    """Convert an entire PhysioNet-style raw directory to BIDS. TODO: build out."""
    raise NotImplementedError
