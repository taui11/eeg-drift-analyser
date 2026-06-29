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
