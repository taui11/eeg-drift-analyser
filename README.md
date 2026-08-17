> ## 🚨 @Spyros — YOUR PART: `eeg_drift/bids_convert.py`
>
> `convert_dataset()` is still `raise NotImplementedError` — that's the
> batch loop over `data/raw/` → `data/bids/`, wired up via
> `scripts/run_bids_convert.py` with argparse (your call on the flags).
>
> `convert_file_to_bids()` (single file) is already done, use it as the
> building block.
>
> **Use AI / dark magic for this** — Claude Code, Copilot, whatever, this
> part is pure plumbing and not worth doing by hand. Just check the output
> is actually valid BIDS (`bids-validator`) before you trust it.
>
> Ping me once `data/bids/` exists so I can start on preprocessing.
>
> **Prompt for Claude Code / Copilot** — paste this in to get started:
>
> ```
> We're building eeg-drift-tracker, a pipeline measuring within-session
> frequency drift in EEG bands (PhysioNet EEG Motor Movement/Imagery DB,
> 109 subjects, 64 channels, 160 Hz, 14 EDF runs/subject: R01 rest-eyes-
> open, R02 rest-eyes-closed, R03-R14 motor execution/imagery tasks).
>
> My teammate has already implemented everything downstream of BIDS:
> - eeg_drift/preprocess.py — notch/high-pass filter, Picard ICA, ICLabel
>   rejection (preprocess_subject(raw) -> (clean_raw, ica))
> - eeg_drift/io.py — list_subjects(bids_root), concatenate_subject_runs(
>   bids_root, subject, task, runs) -> Raw (loads via mne_bids.read_raw_bids)
> - scripts/run_preprocess.py — CLI that loops subjects, preprocesses,
>   saves cleaned .fif to data/derivatives/, writes
>   results/qc/qc_summary.csv
>
> Everything downstream ONLY depends on a valid BIDS tree existing at
> data/bids/ - it doesn't care how it got there. That's my part.
>
> My job: eeg_drift/bids_convert.py's convert_dataset(raw_dir, bids_root)
> - the batch loop over PhysioNet's raw layout (data/raw/S001/S001R01.edf
> ... S109/S109R14.edf) into a BIDS tree, and wiring
> scripts/run_bids_convert.py (argparse) to call it.
>
> Already implemented for me, reuse it:
> convert_file_to_bids(raw_path, bids_root, subject, task, run) -> BIDSPath
> - handles one file, skips if already converted, writes via
> mne_bids.write_raw_bids(..., format="EDF") (needs the edfio package -
> already in pyproject.toml).
>
> IMPORTANT - task/run naming isn't finalized yet. run_preprocess.py
> currently assumes, as an unconfirmed placeholder:
>   task = "motorimagery", runs = "01".."14" (zero-padded)
> Either match that convention exactly, or if you pick something else,
> update DEFAULT_TASK/DEFAULT_RUNS at the top of scripts/run_preprocess.py
> to match and tell me.
>
> Usable subjects only (skip the rest - corrupted/wrong sample rate per
> the reference MATLAB code):
> [1:32, 35, 36, 38, 39, 40:60, 62:71, 73, 75:78, 80:88, 90:92, 94:100, 100:109]
>
> Validate the output is real BIDS (bids-validator data/bids) before
> trusting it - don't just eyeball the folder structure.
> ```


# eeg-drift-analyser

Seminar project — detecting within-session frequency drift in EEG bands.  
Based on [Kostoglou & Müller-Putz (2026)](https://doi.org/10.1371/journal.pcbi.1014112).

**TL;DR:** Take any EEG frequency band, check if the instantaneous frequency drifts over the course of a recording. The paper says mu speeds up over motor cortex and alpha slows down elsewhere. We replicate this in Python and make it work for arbitrary bands.

## Assignments
- Clean data
- no need for resampling (160Hz)
- Highpass filtering (0.5Hz) (Bandpass?)
- remove 60Hz with Notch filter
- screenshot of 1 Subjects (frontal) ICA. Include, which should be eliminated. Before and After(???)
- With the clean data; Bandpass for alpha, theta(4-7Hz), mu(8-12Hz) and beta(13-30Hz)
- Correlate Bands to eachother
- generally show correlation of magnitude and frequency w.r.t time
- ### No generalization needed 

## Data

[PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/) — 109 subjects, 64 channels, 160 Hz, motor execution + imagery tasks.

The EEG montage includes 64 electrodes from the international 10-10 system, excluding Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10. Signal indices in the recordings are numbered from 0 to 63, while the associated montage figure numbers electrodes from 1 to 64.

## TODO

### Setup
- [ ] Python env (MNE, scipy, statsmodels, mne-icalabel)
- [ ] Load 1 subject's EDFs in MNE, apply channel locations
- [ ] Port notch + HP filter from MATLAB code
- [ ] First end-to-end on 1 subject (no ICA): load → filter → bandpass [8 12] → Hilbert → plot inst. frequency

### Preprocessing
- [ ] ICA (extended infomax) on concatenated data
- [ ] ICLabel auto-rejection (eye > 0.48, line > 0.77, muscle > 0.95)
- [ ] Wrap into `preprocess.py`, cache cleaned data as `.fif`
- [ ] Run on all usable subjects (~95), spot-check removed components

### Drift analysis
- [ ] `features.py`: configurable bandpass + Hilbert + smoothing (100 ms moving avg)
- [ ] `drift.py`: robust linear regression per channel → slope in Hz/hour
- [ ] `stats.py`: one-sample t-test of slopes vs 0, Benjamini-Hochberg FDR
- [ ] `viz.py`: topomap of avg slope + topomap of % subjects with positive slope
- [ ] Run for mu [8 12] and compare to paper Fig 3a

### Generalize
- [ ] Make band fully configurable (theta, alpha, beta, gamma via YAML or CLI args)
- [ ] Run for at least 3 bands, compare to paper Fig 7
- [ ] Synthetic chirp test (known drift → verify slope estimator recovers it)
- [ ] Sensitivity check: vary filter order, smoothing window
- [ ] Final figures + results summary

### Nice-to-have
- [ ] Second dataset (Dreyer2023 from Zenodo)
- [ ] EKF tracker instead of Hilbert (the paper's main method)
- [ ] Notebook that regenerates all comparison figures

## Project structure

```
eeg-drift-tracker/
├── config/bands.yaml
├── eeg_drift/
│   ├── io.py            # loading EDFs, montage
│   ├── preprocess.py    # filtering, ICA, artifact rejection
│   ├── features.py      # bandpass, Hilbert, smoothing
│   ├── drift.py         # robust regression, slopes
│   ├── stats.py         # group tests, FDR
│   ├── viz.py           # topomaps
│   └── run.py           # CLI entry point
├── data/                # not in git
├── results/             # not in git
├── locs.ced
└── README.md
```

## Quick reference

- **Usable subjects:** `[1:32, 35, 36, 38, 39, 40:60, 62:71, 73, 75:78, 80:88, 90:92, 94:100, 100:109]`
- **Analysis window:** samples 20k–245k @ 160 Hz (≈ min 2 to min 25)
- **Inst. frequency:** `Fs/(2π) · diff(unwrap(angle(hilbert(x))))`
- **Inst. power:** `|hilbert(x)|²`
- **Slope method:** robust linear regression, bisquare weights (`statsmodels.RLM`)
- **Stats:** one-sample t-test per channel, BH-FDR correction

## Notes

- The MATLAB code we got uses the Hilbert approach, NOT the EKF from the paper — that's fine, the paper shows Hilbert recovers the same spatial pattern (S3/S6 Figs)
- Some subjects have 128 Hz instead of 160 Hz — exclude them
- ICA will be slow on full datasets, run overnight / use `picard` method
- Ask Kostoglou for one cleaned `DATA{i}.mat` to validate against if possible
