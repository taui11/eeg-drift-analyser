> ## 💡 Future improvements (not yet implemented)
>
> - **Correlation analysis** — already listed under Assignments below
>   ("correlate bands to each other", "correlation of magnitude and
>   frequency w.r.t. time") but not built yet.
>   `eeg_drift.features.extract_band_features` already computes
>   `inst_power` alongside `inst_freq`, so the data's there - just needs a
>   correlation/plotting module to actually compare them.
> - **Parallelization** — a real 5-subject run took ~21 min wall-clock
>   (avg ~4 min/subject, ICA-dominated), so all ~100 usable subjects would
>   be several hours sequential. Subjects are fully independent through
>   preprocessing, so multiprocessing across subjects would help - but
>   picard/numpy already used ~400% CPU for a single subject in that run,
>   so worker count needs to account for that (or cap threads per worker
>   via `OMP_NUM_THREADS` etc.) rather than just spawning
>   `min(n_subjects, n_cpus)` workers and hoping for a clean multiplier.
>
> A few other gaps noticed while building this, not tracked anywhere else:
> - **Resumable preprocessing** — `run_preprocess()` always refits ICA for
>   every requested subject, even if
>   `data/derivatives/sub-XXX_clean_raw.fif` already exists.
>   `bids_convert.convert_file_to_bids` already skips already-converted
>   files - `run_preprocess` should do the same, so reruns of
>   `run_pipeline.py` don't redo the most expensive step every time.
> - **`n_runs_loaded`/`n_runs_missing`** — always blank in
>   `qc_summary.csv`; `concatenate_subject_runs` would need to report how
>   many runs it actually found for these to mean anything.
> - **Validate the ICLabel thresholds for real** — `run_ica_review.py` +
>   `derive_thresholds.py` exist and were smoke-tested, but never run on
>   an actual manually-labeled set to check/refine `ICLABEL_THRESHOLDS`.
> - **The line-noise ICLabel class** — flagged as an open question in
>   `preprocess.py` (warns at runtime if seen, never actually handled).
> - **Correlate against Kostoglou's published slope values** — deferred
>   earlier for lack of her actual per-channel/per-band numbers; ask her
>   for those and wire up the comparison plot.


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
