#!/usr/bin/env python3
"""
CLI entry point for preprocessing: BIDS -> cleaned .fif per subject.

Example:
    python scripts/run_preprocess.py --bids-root data/bids --out data/derivatives
"""

import argparse
from pathlib import Path

from eeg_drift.io import concatenate_subject_runs
from eeg_drift.preprocess import preprocess_subject


def main():
    parser = argparse.ArgumentParser(description="Preprocess subjects from BIDS")
    parser.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    parser.add_argument("--out", type=Path, default=Path("data/derivatives"))
    parser.add_argument("--subjects", nargs="+", default=None, help="subset, e.g. 001 002")
    args = parser.parse_args()

    # TODO: loop over subjects, concatenate_subject_runs, preprocess_subject,
    #       save cleaned raw to args.out, compute + append qc row


if __name__ == "__main__":
    main()
