#!/usr/bin/env python3
"""
Build per-subject HTML QC reports (mne.Report) from results/qc/qc_summary.csv
and derivative data.

Example:
    python scripts/build_reports.py --qc results/qc/qc_summary.csv --out reports/
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build per-subject QC reports")
    parser.add_argument("--qc", type=Path, default=Path("results/qc/qc_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    # TODO: for each subject row, build mne.Report + report_text bullets,
    #       save HTML to args.out


if __name__ == "__main__":
    main()
