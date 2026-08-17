#!/usr/bin/env python3
"""
CLI entry point for raw -> BIDS conversion.

Owned by: [teammate]

Example:
    python scripts/run_bids_convert.py --raw-dir data/raw --bids-root data/bids
"""

import argparse
from pathlib import Path

from eeg_drift.bids_convert import convert_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert raw EDF data to BIDS")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    # TODO: add whatever else is useful (--subjects, --overwrite, ...)
    args = parser.parse_args()

    convert_dataset(args.raw_dir, args.bids_root)


if __name__ == "__main__":
    main()
