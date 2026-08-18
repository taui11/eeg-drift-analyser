#!/usr/bin/env python3
"""
Derive per-class ICLabel thresholds from results/ica_labels/manual_labels.csv
(built via scripts/run_ica_review.py), via ROC / Youden's J per class.

For each ICLabel class, the manual "remove" decision is used as ground
truth and that class's predicted probability (proba_<class>, present for
every reviewed component regardless of which class ICLabel actually
predicted) as the score - so this is a real per-class ROC, not just
computed over the components where that class happened to win the argmax.

Example:
    python scripts/derive_thresholds.py --labels results/ica_labels/manual_labels.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]


def load_manual_labels(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def derive_threshold_for_class(rows: list[dict], cls: str) -> dict | None:
    """ROC/Youden's J for one class. Returns None if there's no class balance to test against."""
    proba_key = f"proba_{cls.replace(' ', '_')}"
    y_true = np.array([1 if row["decision"] == "remove" else 0 for row in rows])
    y_score = np.array([float(row[proba_key]) for row in rows])

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))

    return {
        "class": cls,
        "threshold": float(thresholds[best_idx]),
        "youden_j": float(youden_j[best_idx]),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "n_samples": len(rows),
        "n_removed": int(y_true.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Derive ICLabel thresholds from manual labels")
    parser.add_argument("--labels", type=Path, default=Path("results/ica_labels/manual_labels.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/ica_labels/derived_thresholds.csv"))
    args = parser.parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(
            f"{args.labels} not found - run scripts/run_ica_review.py first to build a labeled set."
        )

    rows = load_manual_labels(args.labels)
    if not rows:
        print(f"{args.labels} is empty - nothing to derive.")
        return

    print(f"Loaded {len(rows)} manually-labeled components.")

    results = []
    for cls in ICLABEL_CLASSES:
        result = derive_threshold_for_class(rows, cls)
        if result is None:
            print(f"{cls}: skipped (need both kept and removed examples)")
            continue
        results.append(result)
        print(
            f"{cls}: threshold={result['threshold']:.3f}  "
            f"(TPR={result['tpr']:.2f}, FPR={result['fpr']:.2f}, "
            f"J={result['youden_j']:.2f}, n={result['n_samples']})"
        )

    if not results:
        print("\nNo class had both kept and removed examples - nothing to save.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote derived thresholds to {args.out}")


if __name__ == "__main__":
    main()
