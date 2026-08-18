#!/usr/bin/env python3
"""
Build per-subject HTML QC reports (mne.Report) from results/qc/qc_summary.csv
and derivative data.

Example:
    python scripts/build_reports.py --qc results/qc/qc_summary.csv --out reports/
"""

import argparse
from pathlib import Path

import mne
import pandas as pd

from eeg_drift.qc import flag_outliers
from eeg_drift.report_text import subject_summary_bullets


def build_subject_report(subject: str, qc_row: dict, deriv_root: Path, out_dir: Path) -> Path:
    report = mne.Report(title=f"QC report - sub-{subject}")

    bullets = subject_summary_bullets(qc_row)
    html = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    report.add_html(html, title="QC summary", tags=("qc",))

    fif_path = deriv_root / f"sub-{subject}_clean_raw.fif"
    if fif_path.exists():
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        report.add_raw(raw, title="Cleaned data", psd=True, tags=("raw",))
    else:
        report.add_html(
            f"<p>No cleaned data found at {fif_path}</p>", title="Cleaned data", tags=("raw",)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{subject}_qc_report.html"
    report.save(out_path, overwrite=True, open_browser=False, verbose=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build per-subject QC reports")
    parser.add_argument("--qc", type=Path, default=Path("results/qc/qc_summary.csv"))
    parser.add_argument("--deriv-root", type=Path, default=Path("data/derivatives"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if not args.qc.exists():
        raise FileNotFoundError(f"{args.qc} not found - run scripts/run_preprocess.py first.")

    qc_df = pd.read_csv(args.qc, dtype={"subject": str})
    if qc_df.empty:
        print(f"{args.qc} is empty - nothing to build.")
        return

    qc_df = flag_outliers(qc_df)

    for _, row in qc_df.iterrows():
        qc_row = row.to_dict()
        subject = qc_row["subject"]
        print(f"[{subject}] building QC report...")
        out_path = build_subject_report(subject, qc_row, args.deriv_root, args.out)
        print(f"[{subject}] saved {out_path}")


if __name__ == "__main__":
    main()
