#!/usr/bin/env python3
"""
Single-command, end-to-end pipeline: raw EDFs -> BIDS -> preprocessing ->
drift analysis -> per-subject + group reports. All paths/settings come
from one YAML config (default config/pipeline.yaml) instead of juggling
CLI flags across every scripts/run_*.py / build_*.py individually - those
still work standalone (e.g. to rerun just one stage), this just chains
them using the same config.

Must be run as `python scripts/run_pipeline.py` from the repo root (relies
on sibling-script imports resolving via that directory being on sys.path,
same as every other script here).

Example:
    python scripts/run_pipeline.py                     # config/pipeline.yaml
    python scripts/run_pipeline.py --config my_run.yaml
"""

import argparse
from pathlib import Path

import yaml

from eeg_drift.bids_convert import convert_dataset

from build_group_report import build_group_report
from build_reports import build_all_subject_reports
from run_drift_analysis import run_drift_analysis_all
from run_preprocess import run_preprocess


def run_pipeline(cfg: dict) -> None:
    raw_dir = Path(cfg["raw_dir"])
    bids_root = Path(cfg["bids_root"])
    deriv_root = Path(cfg["deriv_root"])
    qc_out = Path(cfg["qc_out"])
    drift_root = Path(cfg["drift_root"])
    reports_dir = Path(cfg["reports_dir"])
    group_report_out = Path(cfg["group_report_out"])
    bands_config = Path(cfg["bands_config"])

    task = cfg.get("task", "motorimagery")
    runs = cfg.get("runs") or [f"{i:02d}" for i in range(1, 15)]
    subject_ids = cfg.get("subjects")  # list of ints, or None = default

    # bids_convert wants PhysioNet subject numbers (ints); everything
    # downstream discovers subjects from what's actually in data/bids/ or
    # data/derivatives/ unless explicitly restricted here too.
    downstream_subjects = [f"{s:03d}" for s in subject_ids] if subject_ids else None

    print(f"=== [1/5] BIDS conversion ({raw_dir} -> {bids_root}) ===")
    convert_dataset(raw_dir, bids_root, subjects=subject_ids, task=task)

    print(f"\n=== [2/5] Preprocessing ({bids_root} -> {deriv_root}) ===")
    run_preprocess(bids_root, deriv_root, task, runs, qc_out, subjects=downstream_subjects)

    print(f"\n=== [3/5] Drift analysis ({deriv_root} -> {drift_root}) ===")
    run_drift_analysis_all(deriv_root, drift_root, bands_config, band=None)

    print(f"\n=== [4/5] Per-subject reports ({reports_dir}) ===")
    build_all_subject_reports(qc_out, deriv_root, bands_config, reports_dir)

    print(f"\n=== [5/5] Group report ({group_report_out}) ===")
    build_group_report(qc_out, drift_root, bands_config, group_report_out)

    print(f"\nPipeline complete. Per-subject reports in {reports_dir}/, group report at {group_report_out}")


def main():
    parser = argparse.ArgumentParser(description="Run the full eeg-drift pipeline from one config file")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
