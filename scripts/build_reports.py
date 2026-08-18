#!/usr/bin/env python3
"""
Build per-subject HTML QC reports (mne.Report) from results/qc/qc_summary.csv
and derivative data: QC bullets, a PSD overview (no full-length multichannel
time series - unreadable at 64 channels), and one instantaneous-frequency
drift plot per configured band (config/bands.yaml's representative
`channels`), matching the paper's own per-channel drift figures.

Example:
    python scripts/build_reports.py --qc results/qc/qc_summary.csv --out reports/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml

from eeg_drift.drift import analysis_window_seconds, fit_drift_slope
from eeg_drift.features import bandpass_filter, instantaneous_frequency, smooth_moving_average
from eeg_drift.qc import flag_outliers
from eeg_drift.report_text import subject_summary_bullets
from eeg_drift.viz import plot_inst_freq_traces

DECIMATE_TO_HZ = 1.0  # matches drift.fit_drift_slope's default - trace and fit line stay consistent


def _band_trace_figure(raw: mne.io.BaseRaw, band_name: str, band_cfg: dict):
    channels = [ch for ch in band_cfg.get("channels", []) if ch in raw.ch_names]
    if not channels:
        return None

    data = raw.get_data(picks=channels)
    sfreq = raw.info["sfreq"]

    filtered = bandpass_filter(data, sfreq, band_cfg["fmin"], band_cfg["fmax"], order=band_cfg.get("filter_order", 4))
    inst_freq = instantaneous_frequency(filtered, sfreq)
    window_samples = int(sfreq * band_cfg.get("smooth_window_ms", 100.0) / 1000.0)
    inst_freq = smooth_moving_average(inst_freq, window_samples)

    step = max(1, int(round(sfreq / DECIMATE_TO_HZ)))
    times_sec = raw.times[::step]
    inst_freq = inst_freq[:, ::step]

    traces, fits = {}, {}
    for i, ch in enumerate(channels):
        traces[ch] = inst_freq[i]
        fits[ch] = fit_drift_slope(inst_freq[i], sfreq=DECIMATE_TO_HZ, decimate_to_hz=None)

    return plot_inst_freq_traces(times_sec, traces, fits, band_name)


def build_subject_report(subject: str, qc_row: dict, deriv_root: Path, bands_cfg: dict, out_dir: Path) -> Path:
    report = mne.Report(title=f"QC report - sub-{subject}")

    bullets = subject_summary_bullets(qc_row)
    html = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    report.add_html(html, title="QC summary", tags=("qc",))

    fif_path = deriv_root / f"sub-{subject}_clean_raw.fif"
    if fif_path.exists():
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        report.add_raw(raw, title="Cleaned data overview", psd=True, butterfly=False, tags=("raw",))

        tmin, tmax = analysis_window_seconds()
        if tmax <= raw.times[-1]:
            raw_cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
            for band_name, band_cfg in bands_cfg.items():
                fig = _band_trace_figure(raw_cropped, band_name, band_cfg)
                if fig is not None:
                    report.add_figure(fig, title=f"{band_name}: drift", tags=("drift", band_name))
                    plt.close(fig)
        else:
            report.add_html(
                f"<p>Recording ({raw.times[-1]:.1f}s) shorter than the analysis window "
                f"({tmax:.1f}s) - no drift plots.</p>",
                title="Drift", tags=("drift",),
            )
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
    parser.add_argument("--config", type=Path, default=Path("config/bands.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if not args.qc.exists():
        raise FileNotFoundError(f"{args.qc} not found - run scripts/run_preprocess.py first.")

    qc_df = pd.read_csv(args.qc, dtype={"subject": str})
    if qc_df.empty:
        print(f"{args.qc} is empty - nothing to build.")
        return

    qc_df = flag_outliers(qc_df)

    with open(args.config) as f:
        bands_cfg = yaml.safe_load(f)

    for _, row in qc_df.iterrows():
        qc_row = row.to_dict()
        subject = qc_row["subject"]
        print(f"[{subject}] building QC report...")
        out_path = build_subject_report(subject, qc_row, args.deriv_root, bands_cfg, args.out)
        print(f"[{subject}] saved {out_path}")


if __name__ == "__main__":
    main()
