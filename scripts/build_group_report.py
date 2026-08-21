#!/usr/bin/env python3
"""
Build one combined HTML report summarizing results across ALL processed
subjects (group-level QC + group-level drift statistics per band) - the
group-level counterpart to build_reports.py's per-subject reports, mirroring
what Kostoglou et al. report at the group level rather than per-subject.

Needs scripts/run_drift_analysis.py to have already been run for the bands
you want included (it writes results/drift/<band>/*.csv + topomap/trace
PNGs, which this script only reads).

Example:
    python scripts/build_group_report.py --qc results/qc/qc_summary.csv --drift-root results/drift --out reports/group_report.html
"""

import argparse
from pathlib import Path

import mne
import pandas as pd
import yaml

from eeg_drift.qc import flag_outliers

CHANNEL_STATS_EXPLANATION = (
    "<p>One-sample t-test of each channel's slope (Hz/hour) against 0 across subjects, "
    "Benjamini-Hochberg FDR-corrected across channels (p_value_fdr). pct_positive is the "
    "share of subjects with slope &gt; 0 (point estimate only, not a per-subject significance test).</p>"
)


def _qc_summary_html(qc_df: pd.DataFrame) -> str:
    html = (
        "<p>Automated QC metrics per subject, with columns flagged (&lt;metric&gt;_outlier) "
        "when a subject sits more than 2 standard deviations from the group mean on that metric.</p>"
    )
    html += f"<p>{len(qc_df)} subject(s) processed.</p>"
    html += qc_df.to_html(index=False, float_format=lambda x: f"{x:.3g}")

    if "any_outlier" in qc_df.columns:
        flagged = qc_df.loc[qc_df["any_outlier"] == True, "subject"]  # noqa: E712 (pandas bool mask)
        if len(flagged):
            html += (
                f"<p><b>WARNING:</b> {len(flagged)} subject(s) flagged as outliers on at least one "
                f"QC metric (&gt;2 std from the group mean): {', '.join(flagged.astype(str))}</p>"
            )
    return html


def _band_images(band: str, band_dir: Path) -> list[tuple[Path, str, str]]:
    """Returns (path, title_suffix, caption) for whichever of this band's plots exist."""
    candidates = [
        (
            "avg_inst_freq_trace.png", "group-average drift",
            "Group-average instantaneous frequency per representative channel (mean trace across "
            "all processed subjects), with a drift line fit directly to that averaged trace.",
        ),
        (
            "mean_slope_topomap.png", "mean slope topomap",
            "Average per-channel drift slope (Hz/hour) across subjects, all 64 channels. "
            "Red = accelerating, blue = slowing.",
        ),
        (
            "pct_positive_topomap.png", "% positive topomap",
            "Percentage of subjects with a positive (accelerating) slope per channel "
            "(point estimate only, see channel_stats table for the FDR-corrected group test).",
        ),
    ]
    return [(band_dir / fname, title, caption) for fname, title, caption in candidates if (band_dir / fname).exists()]


def _band_stats_html(band: str, band_dir: Path) -> str:
    stats_csv = band_dir / "channel_stats.csv"
    if not stats_csv.exists():
        return (
            "<p>No group channel_stats.csv for this band - run scripts/run_drift_analysis.py "
            "with &gt;=2 subjects first (a single subject can't fit a t-test).</p>"
        )
    stats_df = pd.read_csv(stats_csv)
    return CHANNEL_STATS_EXPLANATION + stats_df.to_html(index=False, float_format=lambda x: f"{x:.3g}")


def build_group_report(qc_path: Path, drift_root: Path, config_path: Path, out_path: Path) -> None:
    if not qc_path.exists():
        raise FileNotFoundError(f"{qc_path} not found - run scripts/run_preprocess.py first.")

    qc_df = pd.read_csv(qc_path, dtype={"subject": str})
    if qc_df.empty:
        print(f"{qc_path} is empty - nothing to build.")
        return
    qc_df = flag_outliers(qc_df)

    with open(config_path) as f:
        band_names = list(yaml.safe_load(f).keys())

    report = mne.Report(title=f"Group QC + drift report ({len(qc_df)} subjects)")
    report.add_html(_qc_summary_html(qc_df), title="QC summary (all subjects)", tags=("qc",))

    for band in band_names:
        band_dir = drift_root / band

        for img_path, title_suffix, caption in _band_images(band, band_dir):
            report.add_image(img_path, title=f"{band}: {title_suffix}", caption=caption, tags=("drift", band))

        report.add_html(_band_stats_html(band, band_dir), title=f"{band}: channel stats", tags=("drift", band))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(out_path, overwrite=True, open_browser=False, verbose=False)
    print(f"Saved group report to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build a combined all-subjects group report")
    parser.add_argument("--qc", type=Path, default=Path("results/qc/qc_summary.csv"))
    parser.add_argument("--drift-root", type=Path, default=Path("results/drift"))
    parser.add_argument("--config", type=Path, default=Path("config/bands.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/group_report.html"))
    args = parser.parse_args()

    build_group_report(args.qc, args.drift_root, args.config, args.out)


if __name__ == "__main__":
    main()
