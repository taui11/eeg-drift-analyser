#!/usr/bin/env python3
"""
Build one combined HTML report summarizing results across ALL processed
subjects (group-level QC + group-level drift statistics per band) - the
group-level counterpart to build_reports.py's per-subject reports, mirroring
what Kostoglou et al. report at the group level rather than per-subject.

Needs scripts/run_drift_analysis.py to have already been run for the bands
you want included (it writes results/drift/<band>/channel_stats.csv +
topomap PNGs, which this script only reads).

Example:
    python scripts/build_group_report.py --qc results/qc/qc_summary.csv --drift-root results/drift --out reports/group_report.html
"""

import argparse
from pathlib import Path

import mne
import pandas as pd
import yaml

from eeg_drift.qc import flag_outliers


def _qc_summary_html(qc_df: pd.DataFrame) -> str:
    html = f"<p>{len(qc_df)} subject(s) processed.</p>"
    html += qc_df.to_html(index=False, float_format=lambda x: f"{x:.3g}")

    if "any_outlier" in qc_df.columns:
        flagged = qc_df.loc[qc_df["any_outlier"] == True, "subject"]  # noqa: E712 (pandas bool mask)
        if len(flagged):
            html += (
                f"<p><b>WARNING:</b> {len(flagged)} subject(s) flagged as outliers on at least one "
                f"QC metric (&gt;2 std from the group mean): {', '.join(flagged.astype(str))}</p>"
            )
    return html


def _band_section(band: str, drift_root: Path) -> tuple[str, list[Path]]:
    band_dir = drift_root / band
    stats_csv = band_dir / "channel_stats.csv"
    if not stats_csv.exists():
        return (
            f"<p>No group channel_stats.csv for band '{band}' - run "
            f"scripts/run_drift_analysis.py with &gt;=2 subjects first.</p>",
            [],
        )

    stats_df = pd.read_csv(stats_csv)
    html = stats_df.to_html(index=False, float_format=lambda x: f"{x:.3g}")

    images = [band_dir / name for name in ("mean_slope_topomap.png", "pct_positive_topomap.png")]
    return html, [p for p in images if p.exists()]


def main():
    parser = argparse.ArgumentParser(description="Build a combined all-subjects group report")
    parser.add_argument("--qc", type=Path, default=Path("results/qc/qc_summary.csv"))
    parser.add_argument("--drift-root", type=Path, default=Path("results/drift"))
    parser.add_argument("--config", type=Path, default=Path("config/bands.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/group_report.html"))
    args = parser.parse_args()

    if not args.qc.exists():
        raise FileNotFoundError(f"{args.qc} not found - run scripts/run_preprocess.py first.")

    qc_df = pd.read_csv(args.qc, dtype={"subject": str})
    if qc_df.empty:
        print(f"{args.qc} is empty - nothing to build.")
        return
    qc_df = flag_outliers(qc_df)

    with open(args.config) as f:
        band_names = list(yaml.safe_load(f).keys())

    report = mne.Report(title=f"Group QC + drift report ({len(qc_df)} subjects)")
    report.add_html(_qc_summary_html(qc_df), title="QC summary (all subjects)", tags=("qc",))

    for band in band_names:
        html, images = _band_section(band, args.drift_root)
        report.add_html(html, title=f"{band}: channel stats", tags=("drift", band))
        for img_path in images:
            report.add_image(img_path, title=f"{band}: {img_path.stem}", tags=("drift", band))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.save(args.out, overwrite=True, open_browser=False, verbose=False)
    print(f"Saved group report to {args.out}")


if __name__ == "__main__":
    main()
