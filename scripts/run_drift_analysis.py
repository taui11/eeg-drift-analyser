#!/usr/bin/env python3
"""
Full drift analysis: cleaned .fif -> band features -> per-channel slopes ->
group statistics -> topomaps, for a configurable band.

Example:
    python scripts/run_drift_analysis.py --band mu --deriv-root data/derivatives --out results/drift/mu
"""

import argparse
from pathlib import Path

import yaml

from eeg_drift.features import extract_band_features
from eeg_drift.drift import fit_drift_slopes_all_channels
from eeg_drift.stats import group_test_slopes


def main():
    parser = argparse.ArgumentParser(description="Run drift analysis for one band")
    parser.add_argument("--band", required=True, help="key in config/bands.yaml")
    parser.add_argument("--config", type=Path, default=Path("config/bands.yaml"))
    parser.add_argument("--deriv-root", type=Path, default=Path("data/derivatives"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        band_cfg = yaml.safe_load(f)[args.band]

    out_dir = args.out or Path("results/drift") / args.band
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: loop over cleaned subjects in deriv_root, extract_band_features,
    #       fit_drift_slopes_all_channels, collect into slopes_by_subject,
    #       group_test_slopes, save CSV + call viz


if __name__ == "__main__":
    main()
