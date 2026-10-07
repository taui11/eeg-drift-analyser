#!/usr/bin/env python3
"""
Full drift analysis: cleaned .fif -> band features -> per-channel slopes ->
group statistics -> topomaps, for every band configured in config/bands.yaml
(or a single one via --band).

Example:
    python scripts/run_drift_analysis.py                       # all bands, results/drift/<band>/
    python scripts/run_drift_analysis.py --band mu_alpha        # just one band
"""

import argparse
import csv
import gc
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from scipy.stats import trim_mean

from eeg_drift.drift import TRACE_DECIMATE_HZ, analysis_window_seconds, fit_drift_slope, fit_drift_slopes_all_channels
from eeg_drift.features import extract_band_features
from eeg_drift.stats import group_test_slopes
from eeg_drift.viz import plot_inst_amp_traces, plot_inst_freq_traces, plot_pct_significant_topomap, plot_slope_topomap

SUBJECT_FILE_RE = re.compile(r"^sub-(?P<subject>.+)_clean_raw\.fif$")


def _iter_cleaned_subjects(deriv_root: Path):
    for fif_path in sorted(deriv_root.glob("sub-*_clean_raw.fif")):
        m = SUBJECT_FILE_RE.match(fif_path.name)
        if m:
            yield m.group("subject"), fif_path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _subjectwise_car_demean(slopes: dict[str, float]) -> dict[str, float]:
    """Remove the across-channel mean slope from each channel within one subject."""
    if not slopes:
        return {}
    mean_slope = float(np.mean(list(slopes.values())))
    return {ch: float(slope - mean_slope) for ch, slope in slopes.items()}


def run_for_band(band_name: str, band_cfg: dict, deriv_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmin, tmax = analysis_window_seconds()

    trace_channels = [ch for ch in band_cfg.get("channels", [])]
    trace_arrays: dict[str, list[np.ndarray]] = {ch: [] for ch in trace_channels}
    amp_trace_arrays: dict[str, list[np.ndarray]] = {ch: [] for ch in trace_channels}
    times_sec_ref = None

    slopes_by_subject: dict[str, dict[str, float]] = {}
    amp_slopes_by_subject: dict[str, dict[str, float]] = {}
    last_info = None  # assumes a consistent 64-ch montage across subjects (true for this pipeline)

    for subject, fif_path in _iter_cleaned_subjects(deriv_root):
        print(f"[{band_name}/{subject}] loading {fif_path.name}...")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

        if tmax > raw.times[-1]:
            print(f"[{band_name}/{subject}] recording too short ({raw.times[-1]:.1f}s), skipping")
            continue
        raw.crop(tmin=tmin, tmax=tmax)

        features = extract_band_features(
            raw.get_data(),
            sfreq=raw.info["sfreq"],
            fmin=band_cfg["fmin"],
            fmax=band_cfg["fmax"],
            filter_order=band_cfg.get("filter_order", 4),
            smooth_window_ms=band_cfg.get("smooth_window_ms", 100.0),
        )

        freq_slopes = fit_drift_slopes_all_channels(
            features["inst_freq"], sfreq=raw.info["sfreq"], ch_names=raw.ch_names
        )
        amp_slopes = fit_drift_slopes_all_channels(
            features["inst_amp"], sfreq=raw.info["sfreq"], ch_names=raw.ch_names
        )
        slopes_by_subject[subject] = {ch: r["slope_per_hour"] for ch, r in freq_slopes.items()}
        amp_slopes_by_subject[subject] = {ch: r["slope_per_hour"] for ch, r in amp_slopes.items()}
        last_info = raw.info

        n_trim = features["n_trimmed_start"]
        times_trimmed = raw.times[n_trim : len(raw.times) - n_trim] if n_trim else raw.times

        step = max(1, int(round(raw.info["sfreq"] / TRACE_DECIMATE_HZ)))
        if times_sec_ref is None:
            times_sec_ref = times_trimmed[::step]
        for ch in trace_channels:
            if ch in raw.ch_names:
                # .copy() is essential here: step-slicing a numpy array returns a
                # VIEW, not a copy, so without it this decimated ~11KB array would
                # keep the entire underlying (64, n_samples) inst_freq/inst_amp
                # buffer (~230MB combined) alive for the rest of the band's loop,
                # once per subject - OOM-killed a 97-subject run on this machine.
                trace_arrays[ch].append(features["inst_freq"][raw.ch_names.index(ch)][::step].copy())
                amp_trace_arrays[ch].append(features["inst_amp"][raw.ch_names.index(ch)][::step].copy())

        del raw, features, freq_slopes, amp_slopes
        gc.collect()

    if not slopes_by_subject:
        print(f"[{band_name}] no cleaned subjects found (or none long enough) - nothing to analyze.")
        return

    if times_sec_ref is not None and any(trace_arrays.values()):
        avg_traces, avg_fits = {}, {}
        for ch, arrs in trace_arrays.items():
            if not arrs:
                continue
            min_len = min(len(a) for a in arrs)
            stacked = np.stack([a[:min_len] for a in arrs])
            avg_traces[ch] = trim_mean(stacked, proportiontocut=0.2, axis=0) if len(arrs) >= 5 else stacked.mean(axis=0)
            avg_fits[ch] = fit_drift_slope(avg_traces[ch], sfreq=TRACE_DECIMATE_HZ, decimate_to_hz=None)

        if avg_traces:
            common_len = min(len(t) for t in avg_traces.values())
            fig = plot_inst_freq_traces(
                times_sec_ref[:common_len],
                {ch: t[:common_len] for ch, t in avg_traces.items()},
                avg_fits,
                band_name,
                title=f"{band_name}: group-average inst. freq. drift (n={len(slopes_by_subject)}, {'trimmed mean ±20%' if len(slopes_by_subject) >= 5 else 'mean'})",
            )
            fig.savefig(out_dir / "avg_inst_freq_trace.png", dpi=150)
            plt.close(fig)
            print(f"[{band_name}] saved group-average frequency trace plot to {out_dir}")

    if times_sec_ref is not None and any(amp_trace_arrays.values()):
        avg_amp_traces, avg_amp_fits = {}, {}
        for ch, arrs in amp_trace_arrays.items():
            if not arrs:
                continue
            min_len = min(len(a) for a in arrs)
            stacked = np.stack([a[:min_len] for a in arrs])
            avg_amp_traces[ch] = trim_mean(stacked, proportiontocut=0.2, axis=0) if len(arrs) >= 5 else stacked.mean(axis=0)
            avg_amp_fits[ch] = fit_drift_slope(avg_amp_traces[ch], sfreq=TRACE_DECIMATE_HZ, decimate_to_hz=None)

        if avg_amp_traces:
            common_len = min(len(t) for t in avg_amp_traces.values())
            fig = plot_inst_amp_traces(
                times_sec_ref[:common_len],
                {ch: t[:common_len] for ch, t in avg_amp_traces.items()},
                avg_amp_fits,
                band_name,
                title=f"{band_name}: group-average inst. amplitude drift (n={len(amp_slopes_by_subject)}, {'trimmed mean ±20%' if len(amp_slopes_by_subject) >= 5 else 'mean'})",
            )
            fig.savefig(out_dir / "avg_inst_amp_trace.png", dpi=150)
            plt.close(fig)
            print(f"[{band_name}] saved group-average amplitude trace plot to {out_dir}")

    channels = sorted({ch for subj in slopes_by_subject.values() for ch in subj})
    amp_channels = sorted({ch for subj in amp_slopes_by_subject.values() for ch in subj})
    _write_csv(
        out_dir / "slopes_by_subject.csv",
        fieldnames=["subject", *channels],
        rows=[
            {"subject": subject, **{ch: slopes.get(ch, "") for ch in channels}}
            for subject, slopes in slopes_by_subject.items()
        ],
    )
    print(f"[{band_name}] wrote frequency slopes to {out_dir}")

    _write_csv(
        out_dir / "amplitude_slopes_by_subject.csv",
        fieldnames=["subject", *amp_channels],
        rows=[
            {"subject": subject, **{ch: slopes.get(ch, "") for ch in amp_channels}}
            for subject, slopes in amp_slopes_by_subject.items()
        ],
    )
    print(f"[{band_name}] wrote amplitude slopes to {out_dir}")

    demeaned_slopes_by_subject = {
        subject: _subjectwise_car_demean(slopes) for subject, slopes in slopes_by_subject.items()
    }
    demeaned_channels = sorted({ch for subj in demeaned_slopes_by_subject.values() for ch in subj})
    _write_csv(
        out_dir / "demeaned_slopes_by_subject.csv",
        fieldnames=["subject", *demeaned_channels],
        rows=[
            {"subject": subject, **{ch: slopes.get(ch, "") for ch in demeaned_channels}}
            for subject, slopes in demeaned_slopes_by_subject.items()
        ],
    )
    print(f"[{band_name}] wrote subject-wise CAR-demeaned frequency slopes to {out_dir}")

    demeaned_amp_slopes_by_subject = {
        subject: _subjectwise_car_demean(slopes) for subject, slopes in amp_slopes_by_subject.items()
    }
    demeaned_amp_channels = sorted({ch for subj in demeaned_amp_slopes_by_subject.values() for ch in subj})
    _write_csv(
        out_dir / "demeaned_amplitude_slopes_by_subject.csv",
        fieldnames=["subject", *demeaned_amp_channels],
        rows=[
            {"subject": subject, **{ch: slopes.get(ch, "") for ch in demeaned_amp_channels}}
            for subject, slopes in demeaned_amp_slopes_by_subject.items()
        ],
    )
    print(f"[{band_name}] wrote subject-wise CAR-demeaned amplitude slopes to {out_dir}")

    mean_slopes = {
        ch: float(np.mean([slopes_by_subject[s][ch] for s in slopes_by_subject if ch in slopes_by_subject[s]]))
        for ch in channels
    }
    fig = plot_slope_topomap(
        mean_slopes, last_info, title=f"{band_name}: mean slope (Hz/hour, n={len(slopes_by_subject)})"
    )
    fig.savefig(out_dir / "mean_slope_topomap.png", dpi=150)
    plt.close(fig)
    print(f"[{band_name}] saved mean frequency slope topomap to {out_dir}")

    mean_amp_slopes = {
        ch: float(np.mean([amp_slopes_by_subject[s][ch] for s in amp_slopes_by_subject if ch in amp_slopes_by_subject[s]]))
        for ch in amp_channels
    }
    fig = plot_slope_topomap(
        mean_amp_slopes,
        last_info,
        title=f"{band_name}: mean amplitude slope (a.u./hour, n={len(amp_slopes_by_subject)})",
        unit="a.u./hour",
    )
    fig.savefig(out_dir / "mean_amplitude_slope_topomap.png", dpi=150)
    plt.close(fig)
    print(f"[{band_name}] saved mean amplitude slope topomap to {out_dir}")

    demeaned_mean_slopes = {
        ch: float(
            np.mean(
                [demeaned_slopes_by_subject[s][ch] for s in demeaned_slopes_by_subject if ch in demeaned_slopes_by_subject[s]]
            )
        )
        for ch in demeaned_channels
    }
    fig = plot_slope_topomap(
        demeaned_mean_slopes,
        last_info,
        title=f"{band_name}: mean slope after subject-wise CAR demeaning (Hz/hour, n={len(demeaned_slopes_by_subject)})",
    )
    fig.savefig(out_dir / "mean_slope_car_topomap.png", dpi=150)
    plt.close(fig)
    print(f"[{band_name}] saved CAR-demeaned frequency slope topomap to {out_dir}")

    demeaned_mean_amp_slopes = {
        ch: float(
            np.mean(
                [demeaned_amp_slopes_by_subject[s][ch] for s in demeaned_amp_slopes_by_subject if ch in demeaned_amp_slopes_by_subject[s]]
            )
        )
        for ch in demeaned_amp_channels
    }
    fig = plot_slope_topomap(
        demeaned_mean_amp_slopes,
        last_info,
        title=f"{band_name}: mean amplitude slope after subject-wise CAR demeaning (a.u./hour, n={len(demeaned_amp_slopes_by_subject)})",
        unit="a.u./hour",
    )
    fig.savefig(out_dir / "mean_amplitude_slope_car_topomap.png", dpi=150)
    plt.close(fig)
    print(f"[{band_name}] saved CAR-demeaned amplitude slope topomap to {out_dir}")

    if len(slopes_by_subject) < 2:
        print(
            f"[{band_name}] only {len(slopes_by_subject)} subject(s) processed - group_test_slopes "
            "needs >=2 to fit a t-test per channel, so channel_stats.csv/the pct-positive topomaps are skipped."
        )
        return

    print(f"[{band_name}] running group stats on {len(slopes_by_subject)} subject(s)...")
    channel_stats = group_test_slopes(slopes_by_subject)
    _write_csv(
        out_dir / "channel_stats.csv",
        fieldnames=["channel", *next(iter(channel_stats.values())).keys()],
        rows=[{"channel": ch, **row} for ch, row in channel_stats.items()],
    )
    print(f"[{band_name}] wrote frequency channel stats to {out_dir}")

    amp_channel_stats = group_test_slopes(amp_slopes_by_subject)
    _write_csv(
        out_dir / "amplitude_channel_stats.csv",
        fieldnames=["channel", *next(iter(amp_channel_stats.values())).keys()],
        rows=[{"channel": ch, **row} for ch, row in amp_channel_stats.items()],
    )
    print(f"[{band_name}] wrote amplitude channel stats to {out_dir}")

    fig = plot_pct_significant_topomap(
        channel_stats, last_info, title=f"{band_name}: % subjects with positive frequency slope"
    )
    fig.savefig(out_dir / "pct_positive_topomap.png", dpi=150)
    plt.close(fig)
    print(f"[{band_name}] saved pct-positive frequency topomap to {out_dir}")

    amp_fig = plot_pct_significant_topomap(
        amp_channel_stats, last_info, title=f"{band_name}: % subjects with positive amplitude slope"
    )
    amp_fig.savefig(out_dir / "pct_positive_amplitude_topomap.png", dpi=150)
    plt.close(amp_fig)
    print(f"[{band_name}] saved pct-positive amplitude topomap to {out_dir}")


def run_drift_analysis_all(deriv_root: Path, out_root: Path, config_path: Path, band: str | None = None) -> None:
    """Run run_for_band() for `band`, or every band in config_path if band is None."""
    with open(config_path) as f:
        all_bands_cfg = yaml.safe_load(f)

    bands_to_run = [band] if band else list(all_bands_cfg.keys())

    for band_name in bands_to_run:
        run_for_band(band_name, all_bands_cfg[band_name], deriv_root, out_root / band_name)


def main():
    parser = argparse.ArgumentParser(description="Run drift analysis for one band, or all configured bands")
    parser.add_argument("--band", default=None, help="key in config/bands.yaml; omit to run every band")
    parser.add_argument("--config", type=Path, default=Path("config/bands.yaml"))
    parser.add_argument("--deriv-root", type=Path, default=Path("data/derivatives"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/drift"),
        help="root dir - each band's output goes to <out>/<band>/",
    )
    args = parser.parse_args()

    run_drift_analysis_all(args.deriv_root, args.out, args.config, band=args.band)


if __name__ == "__main__":
    main()
