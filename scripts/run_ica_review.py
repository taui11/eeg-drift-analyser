#!/usr/bin/env python3
"""
Manual ICA component review tool: fit ICA on one subject, show per-component
plots + ICLabel probabilities, log keep/remove (+ reason) decisions to
results/ica_labels/manual_labels.csv.

Used to build a small ground-truth set for scripts/derive_thresholds.py.

Progress is saved incrementally (one CSV row per decision, flushed
immediately), and already-labeled (subject, component) pairs are skipped on
rerun, so a review session can be stopped ('quit') and resumed later without
losing work.

Example:
    python scripts/run_ica_review.py --subject 001 --bids-root data/bids
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne_icalabel import label_components

from eeg_drift.io import concatenate_subject_runs
from eeg_drift.preprocess import fit_ica, notch_and_highpass

# Column order matches ica.labels_scores_ (set as a side effect of
# label_components(..., method="iclabel") below) - see
# mne_icalabel.config.ICALABEL_METHODS_NUMERICAL_TO_STRING["iclabel"].
ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]

CSV_FIELDNAMES = [
    "subject",
    "component_idx",
    "predicted_label",
    "predicted_proba",
    *[f"proba_{cls.replace(' ', '_')}" for cls in ICLABEL_CLASSES],
    "decision",
    "reason",
    "plot_path",
]

DECISION_ALIASES = {"k": "keep", "r": "remove", "s": "skip", "q": "quit"}


def _load_reviewed_keys(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(row["subject"], int(row["component_idx"])) for row in csv.DictReader(f)}


def save_component_plot(raw: mne.io.BaseRaw, ica, idx: int, out_dir: Path, label_slug: str) -> Path:
    """Same EEGLAB/ICLabel-style rendering (jet topomap, red spectrum line) as the earlier S001 review."""
    figs = ica.plot_properties(raw, picks=idx, show=False, verbose=False, topomap_args=dict(cmap="jet"))
    fig = figs[0]

    spectrum_ax = fig.axes[3]
    for line in spectrum_ax.get_lines():
        line.set_color("red")
        line.set_linewidth(2)
    spectrum_ax.grid(True, alpha=0.4)

    fig.set_size_inches(10, 7.5)
    fig.suptitle(f"IC{idx:03d} - {label_slug}", y=1.0)
    fig.subplots_adjust(top=0.90)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"IC{idx:03d}_{label_slug}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def prompt_decision() -> str:
    while True:
        raw_input_ = input("  keep / remove / skip / quit? ").strip().lower()
        decision = DECISION_ALIASES.get(raw_input_, raw_input_)
        if decision in {"keep", "remove", "skip", "quit"}:
            return decision
        print("  please answer keep/remove/skip/quit (or k/r/s/q)")


def main():
    parser = argparse.ArgumentParser(description="Manually review ICA components")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    parser.add_argument("--task", default="motorimagery", help="must match your BIDS conversion's task label")
    parser.add_argument("--runs", nargs="+", default=[f"{i:02d}" for i in range(1, 15)])
    parser.add_argument("--out", type=Path, default=Path("results/ica_labels"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels_csv = args.out / "manual_labels.csv"
    already_reviewed = _load_reviewed_keys(labels_csv)

    print(f"[{args.subject}] loading + concatenating runs...")
    raw = concatenate_subject_runs(args.bids_root, args.subject, task=args.task, runs=args.runs)

    raw.rename_channels(lambda name: name.strip("."))
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw = notch_and_highpass(raw)

    print(f"[{args.subject}] fitting ICA (picard, this can take a while)...")
    ica = fit_ica(raw)

    print(f"[{args.subject}] running ICLabel...")
    ic_labels = label_components(raw, ica, method="iclabel")
    predicted_labels = ic_labels["labels"]
    predicted_probas = ic_labels["y_pred_proba"]
    all_class_probas = ica.labels_scores_  # (n_components, 7) - side effect of label_components() above

    plot_dir = args.out / f"sub-{args.subject}"
    file_exists = labels_csv.exists()

    with open(labels_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        n_components = ica.n_components_
        for idx in range(n_components):
            if (args.subject, idx) in already_reviewed:
                continue

            label = predicted_labels[idx]
            proba = predicted_probas[idx]
            label_slug = label.replace(" ", "-")

            plot_path = save_component_plot(raw, ica, idx, plot_dir, label_slug)

            print(f"\nIC{idx:03d}/{n_components - 1} - predicted: {label} (p={proba:.2f})")
            print(f"  plot: {plot_path}")

            decision = prompt_decision()
            if decision == "quit":
                print("Stopping - progress saved, rerun the same command to resume.")
                break
            if decision == "skip":
                continue

            reason = input("  reason (optional): ").strip()

            row = {
                "subject": args.subject,
                "component_idx": idx,
                "predicted_label": label,
                "predicted_proba": f"{proba:.4f}",
                "decision": decision,
                "reason": reason,
                "plot_path": str(plot_path),
            }
            for cls, p in zip(ICLABEL_CLASSES, all_class_probas[idx]):
                row[f"proba_{cls.replace(' ', '_')}"] = f"{p:.4f}"

            writer.writerow(row)
            f.flush()

    print(f"\nSaved manual labels to {labels_csv}")


if __name__ == "__main__":
    main()
