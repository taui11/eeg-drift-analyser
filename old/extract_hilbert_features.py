######################################################
# Non-invasive Brain-Computer Interfaces, KU [709.028]
# MATLAB extracthilbert translated to Python
######################################################

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, resample_poly, hilbert
from scipy.ndimage import uniform_filter1d


# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
FS_ORIGINAL = 160

SUBJECTS = list(range(1, 33)) + [35, 36, 38, 39] + list(range(40, 61)) + \
           list(range(62, 72)) + [73] + list(range(75, 79)) + \
           list(range(80, 89)) + [90, 91, 92, 94, 95, 96, 97, 98, 99] + \
           list(range(100, 110))

START_SAMPLE_ORIGINAL = 20000
END_SAMPLE_ORIGINAL = 245000

MIN_REQUIRED_SAMPLE_ORIGINAL = 245000

N_CHANNELS = 64


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def moving_average(x: np.ndarray, window_size: int):
    """
    MATLAB smoothdata(..., 'movmean', window)
    """
    if window_size < 1:
        window_size = 1

    return uniform_filter1d(x, size=window_size, axis=-1, mode="nearest")


def bandpass_filter(data: np.ndarray, fmin: float, fmax: float, fs: float, order: int):
    """
    Butterworth bandpass filter.

    Input:
        data shape: channels x samples
    """
    b, a = butter(
        N=order,
        Wn=[fmin, fmax],
        btype="bandpass",
        fs=fs
    )

    return filtfilt(b, a, data, axis=-1)


def resample_data(data: np.ndarray, fs_old: int, fs_new: int):
    """
    Replacement for MATLAB resample().
    Uses polyphase resampling.

    Input:
        data shape: channels x samples
    """
    from math import gcd

    g = gcd(fs_new, fs_old)
    up = fs_new // g
    down = fs_old // g

    return resample_poly(data, up, down, axis=-1)


def extract_instantaneous_frequency_and_power(signal_1d: np.ndarray, fs: float, smooth_window: int):
    """
    Compute Hilbert instantaneous frequency and power.

    Instantaneous frequency:
        f_inst = fs / (2*pi) * diff(unwrap(angle(hilbert(x))))

    Power:
        abs(hilbert(x))^2
    """
    analytic_signal = hilbert(signal_1d)

    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = fs / (2 * np.pi) * np.diff(phase)

    # MATLAB pads with last value to keep same length
    inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])

    inst_power = np.abs(analytic_signal) ** 2

    inst_freq_smooth = moving_average(inst_freq, smooth_window)
    inst_power_smooth = moving_average(inst_power, smooth_window)

    return inst_freq_smooth, inst_power_smooth


def extract_hilbert(
    fmin: float,
    fmax: float,
    fs_new: int,
    order: int,
    data_dir: str = ".",
    output_path: str | None = None
):
    """
    Python translation of MATLAB:

    function [pval,tval,pvalf,tvalf]=extracthilbert(fmin,fmax,Fsnew,ord)

    The MATLAB function name suggests p-values/t-values,
    but the shown code actually computes:
        powf = instantaneous frequency
        pow  = instantaneous power

    Returns:
        pow_power: shape subjects x channels x samples
        pow_freq:  shape subjects x channels x samples
        valid_subjects
    """

    pow_power = []
    pow_freq = []
    valid_subjects = []

    smooth_window = int(fs_new / 10)  # 100 ms window, same as MATLAB Fsnew/10

    start_idx = round(START_SAMPLE_ORIGINAL * fs_new / FS_ORIGINAL)
    end_idx = round(END_SAMPLE_ORIGINAL * fs_new / FS_ORIGINAL)

    min_required_samples = round(MIN_REQUIRED_SAMPLE_ORIGINAL * fs_new / FS_ORIGINAL)

    for subject in SUBJECTS:
        mat_path = os.path.join(data_dir, f"DATA{subject}.mat")

        if not os.path.exists(mat_path):
            print(f"Missing file, skipping subject {subject}: {mat_path}")
            continue

        print(f"\nLoading subject {subject}")
        mat = loadmat(mat_path)

        if "DATA" not in mat:
            print(f"DATA variable not found in {mat_path}, skipping.")
            continue

        DATA = np.asarray(mat["DATA"], dtype=float)

        # Expected shape: channels x samples
        if DATA.shape[0] > DATA.shape[1]:
            print("Warning: DATA seems transposed. Transposing automatically.")
            DATA = DATA.T

        # Bandpass at original sampling rate
        DATA_filtered = bandpass_filter(
            DATA,
            fmin=fmin,
            fmax=fmax,
            fs=FS_ORIGINAL,
            order=order
        )

        # Resample to fs_new
        DATA_resampled = resample_data(
            DATA_filtered,
            fs_old=FS_ORIGINAL,
            fs_new=fs_new
        )

        if DATA_resampled.shape[1] < min_required_samples:
            print(f"Subject {subject} too short, skipping.")
            continue

        print(f"Processing subject {subject}")
        valid_subjects.append(subject)

        subject_power = []
        subject_freq = []

        for ch in range(N_CHANNELS):
            print(f"Subject {subject}, channel {ch + 1}")

            segment = DATA_resampled[ch, start_idx:end_idx]

            inst_freq, inst_power = extract_instantaneous_frequency_and_power(
                segment,
                fs=fs_new,
                smooth_window=smooth_window
            )

            subject_freq.append(inst_freq)
            subject_power.append(inst_power)

        pow_freq.append(np.stack(subject_freq, axis=0))
        pow_power.append(np.stack(subject_power, axis=0))

    pow_freq = np.stack(pow_freq, axis=0)
    pow_power = np.stack(pow_power, axis=0)
    valid_subjects = np.array(valid_subjects)

    if output_path is not None:
        savemat(
            output_path,
            {
                "pow": pow_power,
                "powf": pow_freq,
                "valid_subjects": valid_subjects,
                "fmin": fmin,
                "fmax": fmax,
                "fs_new": fs_new,
                "order": order,
            }
        )
        print(f"\nSaved Hilbert features to: {output_path}")

    return pow_power, pow_freq, valid_subjects


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------
if __name__ == "__main__":
    pow_power, pow_freq, valid_subjects = extract_hilbert(
        fmin=8,
        fmax=13,
        fs_new=160,
        order=4,
        data_dir=".",
        output_path="hilbert_features_alpha.mat"
    )

    print("pow_power shape:", pow_power.shape)
    print("pow_freq shape:", pow_freq.shape)
    print("valid subjects:", valid_subjects)