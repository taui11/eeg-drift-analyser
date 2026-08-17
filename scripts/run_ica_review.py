#!/usr/bin/env python3
"""
Manual ICA component review tool: fit ICA on one subject, show per-component
plots + ICLabel probabilities, log keep/remove (+ reason) decisions to
results/ica_labels/manual_labels.csv.

Used to build a small ground-truth set for scripts/derive_thresholds.py.

Example:
    python scripts/run_ica_review.py --subject 001 --bids-root data/bids
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Manually review ICA components")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    parser.add_argument("--out", type=Path, default=Path("results/ica_labels"))
    args = parser.parse_args()

    # TODO: load + preprocess up to ICA fit, run ICLabel, save per-component
    #       plots, prompt for keep/remove + reason, append to manual_labels.csv


if __name__ == "__main__":
    main()
