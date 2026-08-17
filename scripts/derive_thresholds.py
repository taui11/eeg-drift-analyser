#!/usr/bin/env python3
"""
Derive per-class ICLabel thresholds from results/ica_labels/manual_labels.csv
(built via scripts/run_ica_review.py), e.g. via ROC / Youden's J per class.

Example:
    python scripts/derive_thresholds.py --labels results/ica_labels/manual_labels.csv
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Derive ICLabel thresholds from manual labels")
    parser.add_argument("--labels", type=Path, default=Path("results/ica_labels/manual_labels.csv"))
    args = parser.parse_args()

    # TODO: per class, roc_curve + Youden's J, print/save derived thresholds


if __name__ == "__main__":
    main()
