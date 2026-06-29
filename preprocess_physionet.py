######################################################
# Non-invasive Brain-Computer Interfaces, KU [709.028]
# MATLAB helpfun(i) translated to Python/MNE
######################################################

import os
import numpy as np
import mne
from scipy.io import savemat
from scipy.signal import butter, filtfilt, iirnotch
from mne.preprocessing import ICA

try:
    from mne_icalabel import label_components
    HAS_ICLABEL = True
except ImportError:
    HAS_ICLABEL = False
    print("Warning: mne_icalabel not installed. ICA labels will be skipped.")


# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
FS = 160
PHYSIONET_ROOT = "/home/kkostoglou/Desktop/Fatigue/Physionet"
OUTPUT_DIR = "."
MONTAGE_NAME = "standard_1005"


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def subject_folder_name(subject_id: int) -> str:
    """
    MATLAB paths:
    i < 10      -> S00i
    10 <= i<100 -> S0i
    i >= 100   -> Si
    """
    if subject_id < 10:
        return f"S00{subject_id}"
    elif subject_id < 100:
        return f"S0{subject_id}"
    else:
        return f"S{subject_id}"


def edf_file_path(subject_id: int, run: int, root: str = PHYSIONET_ROOT) -> str:
    """
    Recreates MATLAB filename logic:
    S001/S001R01.edf, ...
    S010/S010R10.edf, ...
    """
    folder = subject_folder_name(subject_id)

    if run < 10:
        filename = f"{folder}R0{run}.edf"
    else:
        filename = f"{folder}R{run}.edf"

    return os.path.join(root, folder, filename)


def highpass_filter(data: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 4):
    """
    MATLAB:
    [b,a] = butter(4, [0.5]/(160/2), 'high');
    DATA = filtfilt(b,a,...)
    """
    b, a = butter(order, cutoff / (fs / 2), btype="highpass")
    return filtfilt(b, a, data, axis=-1)


def remove_powerline(data: np.ndarray, fs: float, freq: float = 60.0, bandwidth: float = 2.0):
    """
    Python equivalent of MATLAB removepowerline2:

    d = designfilt('bandstopiir','FilterOrder',2,
        'HalfPowerFrequency1',F-1,'HalfPowerFrequency2',F+1,
        'DesignMethod','butter','SampleRate',Fs);
    s = filtfilt(d,s);

    Here implemented as a Butterworth bandstop.
    """
    low = freq - bandwidth / 2
    high = freq + bandwidth / 2

    b, a = butter(
        N=2,
        Wn=[low, high],
        btype="bandstop",
        fs=fs
    )

    return filtfilt(b, a, data, axis=-1)


def preprocess_raw_before_ica(raw: mne.io.BaseRaw):
    """
    Similar to the MATLAB preprocessing before ICA:
    - 60 Hz powerline removal
    - 0.5 Hz high-pass
    - average reference
    """
    raw = raw.copy()

    data = raw.get_data()

    data = remove_powerline(data, fs=raw.info["sfreq"], freq=60.0, bandwidth=2.0)
    data = highpass_filter(data, fs=raw.info["sfreq"], cutoff=0.5, order=4)

    raw._data = data

    raw.set_eeg_reference("average", projection=False)

    return raw


def remove_ica_artifacts(raw: mne.io.BaseRaw, random_state: int = 43):
    """
    Replacement for MATLAB:
    EEG = pop_runica(...)
    EEG = pop_iclabel(...)
    idx  = Eye > 0.48
    idx2 = Muscle > 0.77
    idx3 = Channel noise > 0.95
    EEG = pop_subcomp(...)
    """
    raw = raw.copy()

    ica = ICA(
        n_components=0.99,
        method="fastica",
        random_state=random_state,
        max_iter="auto"
    )

    ica.fit(raw)

    excluded_components = []
    labels = None
    probs = None

    if HAS_ICLABEL:
        ic_labels = label_components(raw, ica, method="iclabel")

        labels = ic_labels["labels"]
        probs = ic_labels["y_pred_proba"]

        for comp_idx, label in enumerate(labels):
            prob = probs[comp_idx].max()

            # Conservative version inspired by your MATLAB thresholds
            if label == "eye blink" or label == "eye":
                if prob > 0.48:
                    excluded_components.append(comp_idx)

            elif label == "muscle artifact" or label == "muscle":
                if prob > 0.77:
                    excluded_components.append(comp_idx)

            elif label == "channel noise":
                if prob > 0.95:
                    excluded_components.append(comp_idx)

            # Alternative from Assignment 1:
            # exclude all non-brain/non-other with probability >= 0.8
            elif label not in ["brain", "other"] and prob >= 0.8:
                excluded_components.append(comp_idx)

    else:
        print("No ICLabel available. ICA fitted but no components excluded automatically.")

    ica.exclude = sorted(set(excluded_components))
    print("Excluded ICA components:", ica.exclude)

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean, ica, labels, probs


def helpfun_python(subject_id: int):
    """
    Python translation of MATLAB helpfun(i).

    Loads EDF runs 1-14 for one PhysioNet subject,
    concatenates them,
    removes powerline + high-pass filters,
    runs ICA,
    removes likely artifacts,
    and saves cleaned DATA to DATA<subject_id>.mat.
    """

    raws = []

    for run in range(1, 15):
        path = edf_file_path(subject_id, run)

        if not os.path.exists(path):
            print(f"Missing file, skipping: {path}")
            continue

        print(f"Loading: {path}")

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        # Force expected sampling rate check
        if raw.info["sfreq"] != FS:
            print(f"Warning: Expected {FS} Hz, got {raw.info['sfreq']} Hz.")

        # Set montage if possible
        montage = mne.channels.make_standard_montage(MONTAGE_NAME)
        raw.set_montage(montage, match_case=False, on_missing="ignore")

        # MATLAB used EEG.data, usually channels x samples
        raw = preprocess_raw_before_ica(raw)

        raws.append(raw)

    if len(raws) == 0:
        raise FileNotFoundError(f"No EDF files found for subject {subject_id}")

    print("Concatenating runs...")
    raw_concat = mne.concatenate_raws(raws)

    print("Running ICA...")
    raw_clean, ica, labels, probs = remove_ica_artifacts(raw_concat)

    DATA = raw_clean.get_data()

    output_path = os.path.join(OUTPUT_DIR, f"DATA{subject_id}.mat")

    savemat(
        output_path,
        {
            "DATA": DATA,
            "ica_exclude": np.array(ica.exclude),
            "ic_labels": np.array(labels, dtype=object) if labels is not None else np.array([]),
            "ic_probs": np.array(probs) if probs is not None else np.array([]),
        }
    )

    print(f"Saved cleaned data to: {output_path}")

    return DATA, raw_clean, ica


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------
if __name__ == "__main__":
    subject = 1
    helpfun_python(subject)