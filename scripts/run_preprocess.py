#!/usr/bin/env python3
"""
CLI entry point for preprocessing: BIDS -> cleaned .fif per subject.

Example:
    python scripts/run_preprocess.py --bids-root data/bids --out data/derivatives
"""

import argparse
import csv
from pathlib import Path

from eeg_drift.io import concatenate_subject_runs, list_subjects
from eeg_drift.preprocess import preprocess_subject
from eeg_drift.qc import compute_qc_row

# NOTE: task/run naming isn't finalized yet - convert_dataset() (teammate's
# batch BIDS conversion loop) is still a stub, so this is a placeholder
# convention, not a confirmed contract. Confirm task label + zero-padding
# with whoever builds out bids_convert.convert_dataset before running this
# for real, and update the defaults below (or just pass --task/--runs).
DEFAULT_TASK = "motorimagery"
DEFAULT_RUNS = [f"{i:02d}" for i in range(1, 15)]
EXPECTED_RUNS = len(DEFAULT_RUNS)


def main():
    parser = argparse.ArgumentParser(description="Preprocess subjects from BIDS")
    parser.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    parser.add_argument("--out", type=Path, default=Path("data/derivatives"))
    parser.add_argument("--subjects", nargs="+", default=None, help="subset, e.g. 001 002")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--qc-out", type=Path, default=Path("results/qc/qc_summary.csv"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list_subjects(args.bids_root)
    if not subjects:
        print(f"No subjects found under {args.bids_root}")
        return

    qc_rows = []
    for subject in subjects:
        print(f"[{subject}] loading + concatenating runs...")
        try:
            raw = concatenate_subject_runs(args.bids_root, subject, task=args.task, runs=args.runs)
        except FileNotFoundError as exc:
            print(f"[{subject}] skipping: {exc}")
            continue

        n_runs_loaded = None  # concatenate_subject_runs doesn't currently report this count

        print(f"[{subject}] preprocessing (notch/HP filter + ICA)...")
        raw_clean, ica = preprocess_subject(raw)

        out_path = args.out / f"sub-{subject}_clean_raw.fif"
        raw_clean.save(out_path, overwrite=True)
        print(f"[{subject}] saved cleaned data to {out_path} ({len(ica.exclude)} components excluded)")

        qc_rows.append(
            compute_qc_row(
                raw_before=raw,
                raw_after=raw_clean,
                ica=ica,
                subject=subject,
                expected_runs=EXPECTED_RUNS,
                n_runs_loaded=n_runs_loaded,
            )
        )

    if qc_rows:
        with open(args.qc_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=qc_rows[0].keys())
            writer.writeheader()
            writer.writerows(qc_rows)
        print(f"Wrote QC summary for {len(qc_rows)} subjects to {args.qc_out}")


if __name__ == "__main__":
    main()
