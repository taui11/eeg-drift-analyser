######################################################
# One-off ICA component review for subject S001
#
# Dataset (from README.md):
#   PhysioNet EEG Motor Movement/Imagery Database
#   109 subjects, 64 channels, 160 Hz, motor execution + imagery tasks.
#   64 electrodes from the international 10-10 system, excluding
#   Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, P10.
#   Signal indices are 0-63; the montage figure numbers electrodes 1-64.
#
# Fits ICA on subject S001 (runs 1-14, concatenated) the same way as
# preprocess_physionet.py, then saves one PNG per component with
# topomap, epochs image, ERP, spectrum and epoch-variance panels
# (mne's ICA.plot_properties) so they can be reviewed and annotated
# by hand before sending to the supervisor.
#
# Hardcoded to S001 on purpose - not meant to be pipelined yet.
######################################################

import os
import glob

import matplotlib
matplotlib.use("Agg")

import mne
from mne.preprocessing import ICA

try:
    from mne_icalabel import label_components
    HAS_ICLABEL = True
except ImportError:
    HAS_ICLABEL = False
    print("Warning: mne_icalabel not installed. Components will be plotted without label hints.")

from preprocess_physionet import (
    clean_channel_names,
    preprocess_raw_before_ica,
)

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_DIR = os.path.join(SCRIPT_DIR, "data", "S001")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "ica_review_S001")
MONTAGE_NAME = "standard_1005"
FS = 160
RANDOM_STATE = 43


def load_subject_s001():
    edf_paths = sorted(glob.glob(os.path.join(SUBJECT_DIR, "S001R*.edf")))

    raws = []
    for path in edf_paths:
        print(f"Loading: {path}")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        if raw.info["sfreq"] != FS:
            print(f"Warning: Expected {FS} Hz, got {raw.info['sfreq']} Hz.")

        clean_channel_names(raw)
        montage = mne.channels.make_standard_montage(MONTAGE_NAME)
        raw.set_montage(montage, match_case=False, on_missing="ignore")

        raw = preprocess_raw_before_ica(raw)
        raws.append(raw)

    if not raws:
        raise FileNotFoundError(f"No EDF files found in {SUBJECT_DIR}")

    print("Concatenating runs...")
    return mne.concatenate_raws(raws)


def fit_ica(raw: mne.io.BaseRaw):
    ica = ICA(
        n_components=None,  # all available components, not just top variance
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    print("Running ICA (this can take a few minutes)...")
    ica.fit(raw)
    return ica


def label_hints(raw: mne.io.BaseRaw, ica: ICA):
    """Best-effort ICLabel predictions to sanity-check your own calls against."""
    if not HAS_ICLABEL:
        return [None] * ica.n_components_, [None] * ica.n_components_

    ic_labels = label_components(raw, ica, method="iclabel")
    return ic_labels["labels"], ic_labels["y_pred_proba"]


def save_component_plots(raw: mne.io.BaseRaw, ica: ICA, labels, probs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(OUTPUT_DIR, "IC*.png")):
        os.remove(stale)

    for idx in range(ica.n_components_):
        label = labels[idx] if labels[idx] is not None else "unlabeled"
        prob = probs[idx].max() if probs[idx] is not None else float("nan")
        label_slug = label.replace(" ", "-")

        print(f"Plotting IC{idx:03d} ({label}, p={prob:.2f})...")
        figs = ica.plot_properties(
            raw, picks=idx, show=False, verbose=False,
            topomap_args=dict(cmap="jet"),
        )
        fig = figs[0]

        # EEGLAB/ICLabel-style spectrum: thick red line + gridlines
        spectrum_ax = fig.axes[3]
        for line in spectrum_ax.get_lines():
            line.set_color("red")
            line.set_linewidth(2)
        spectrum_ax.grid(True, alpha=0.4)

        fig.set_size_inches(10, 7.5)
        fig.suptitle(f"S001 - IC{idx:03d} - ICLabel: {label} (p={prob:.2f})", y=1.0)
        fig.subplots_adjust(top=0.90)

        filename = f"IC{idx:03d}_{label_slug}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        matplotlib.pyplot.close(fig)


def main():
    raw = load_subject_s001()
    ica = fit_ica(raw)
    labels, probs = label_hints(raw, ica)
    save_component_plots(raw, ica, labels, probs)
    print(f"Saved {ica.n_components_} component plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
