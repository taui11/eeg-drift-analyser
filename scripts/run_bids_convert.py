#!/usr/bin/env python3
"""
CLI entry point for raw -> BIDS conversion.

Example:
    python scripts/run_bids_convert.py --raw-dir data/raw --bids-root data/bids
    python scripts/run_bids_convert.py --subjects 1 2 3   # subset, for testing

After running, validate the output is real BIDS before trusting it:
    bids-validator data/bids
"""

import argparse
from pathlib import Path

from eeg_drift.bids_convert import DEFAULT_TASK, convert_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert raw EDF data to BIDS")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="subject IDs, e.g. 1 2 3; default: all usable subjects (eeg_drift.bids_convert.USABLE_SUBJECTS)",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()

    convert_dataset(args.raw_dir, args.bids_root, subjects=args.subjects, task=args.task)


if __name__ == "__main__":
    main()
