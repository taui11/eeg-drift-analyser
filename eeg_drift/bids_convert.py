"""
Raw EDF (PhysioNet) -> BIDS conversion.

Everything downstream (io.py onward) only depends on there being a valid
BIDS tree at data/bids/ - it doesn't care how it got there.

V1 implemented against the reference MATLAB code's usable-subjects list and
run_preprocess.py's task/run naming convention (task="motorimagery", runs
"01".."14"). Validate the actual output with `bids-validator data/bids`
before trusting it - this hasn't been run through that tool.
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

# Matches scripts/run_preprocess.py's DEFAULT_TASK/DEFAULT_RUNS - keep these
# in sync if either changes.
DEFAULT_TASK = "motorimagery"
N_RUNS = 14

# Usable subjects per the reference MATLAB code (excludes corrupted
# recordings / wrong sample rate), from the README's Quick Reference:
# [1:32, 35, 36, 38, 39, 40:60, 62:71, 73, 75:78, 80:88, 90:92, 94:100, 100:109]
USABLE_SUBJECTS: list[int] = sorted(
    set(
        list(range(1, 33))
        + [35, 36, 38, 39]
        + list(range(40, 61))
        + list(range(62, 72))
        + [73]
        + list(range(75, 79))
        + list(range(80, 89))
        + list(range(90, 93))
        + list(range(94, 101))
        + list(range(100, 110))
    )
)


def subject_folder_name(subject_id: int) -> str:
    """PhysioNet's raw layout: S001, ..., S010, ..., S109 (always 3 digits, zero-padded)."""
    return f"S{subject_id:03d}"


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


def convert_dataset(
    raw_dir: Path,
    bids_root: Path,
    subjects: list[int] | None = None,
    task: str = DEFAULT_TASK,
) -> None:
    """
    Convert an entire PhysioNet-style raw directory
    (raw_dir/S001/S001R01.edf ... raw_dir/S109/S109R14.edf) into a BIDS
    tree at bids_root.

    subjects: which subject IDs (ints) to convert; defaults to
    USABLE_SUBJECTS. Missing subject directories/runs are skipped with a
    message rather than failing the whole conversion (matches the
    reference MATLAB code's own "missing file, skip" behaviour).
    """
    subjects = USABLE_SUBJECTS if subjects is None else subjects

    for subject_id in subjects:
        folder = subject_folder_name(subject_id)
        subject_dir = raw_dir / folder
        if not subject_dir.exists():
            print(f"Missing subject directory, skipping: {subject_dir}")
            continue

        bids_subject = f"{subject_id:03d}"

        for run in range(1, N_RUNS + 1):
            edf_path = subject_dir / f"{folder}R{run:02d}.edf"
            if not edf_path.exists():
                print(f"Missing run, skipping: {edf_path}")
                continue

            bids_path = convert_file_to_bids(edf_path, bids_root, subject=bids_subject, task=task, run=f"{run:02d}")
            print(f"sub-{bids_subject} run-{run:02d} -> {bids_path.fpath}")
