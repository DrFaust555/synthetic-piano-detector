# eGeMAPS baseline under protocol v2 (format-matched, isolated notes)

Generated 2026-10-01 11:30 by one command, from the project root:

```
python Stattia/revision/egemaps/run_egemaps.py
```

Models: Python 3.10.7, numpy 1.26.4, pandas 1.5.2, scikit-learn 1.4.1.post1, scipy 1.11.1, xgboost 3.2.0 (n_jobs=1, random_state=0) -- the protocol-v2 environment. Extraction (separate interpreter, `egemaps_extract.py`): opensmile 2.6.0, audinterface 1.4.0, audiofile 1.6.1, pandas 2.3.3 (needed by audinterface), numpy 1.26.4 (system). opensmile was installed with `pip install --target` into a scratch folder; the system Python is unchanged. Runtime 170 s (extraction 13 s incl. cache hits, CV 119 s). Determinism check vs the previous run (out/checksums.txt): 3 of 3 output CSVs byte-identical.

Scripts: `Stattia/revision/egemaps/{run_egemaps, egemaps_extract}.py`. Outputs (`Stattia/revision/egemaps/out/`): `egemaps_comparisons.csv` (every AUC row, incl. CCMusic false alarms), `egemaps_scores.csv` (per-file scores, every run), `egemaps_clip_scores.csv`, `key_numbers_egemaps.json`, `cache/egemaps_{main,sr44k}.csv` (88 functionals per file and loader; delete to re-extract), `cache/lead_in.csv`.

## 0. Key numbers: eGeMAPS vs the 47-feature XGBoost (RESULTS_v2)

**Summary (all numbers from the tables below).**

- CCMusic notes vs algorithmic notes: eGeMAPS XGB 0.918 [0.812, 0.994], det 92.5% (T1), 0.943 [0.822, 1.000], det 92.6% (T2); 47-feature XGB 0.904 [0.825, 0.963] (T1), 0.917 [0.842, 0.971] (T2), 0.908 [0.820, 0.970] (I-only). Submitted eGeMAPS value 0.950.
- CCMusic notes vs library notes (pooled, mf layer): eGeMAPS XGB 0.874 [0.714, 0.987], det 81.2% (T1), 0.743 [0.519, 0.899], det 25.0% (T2); eGeMAPS LR 0.864 [0.716, 0.971] (T1), 0.324 [0.071, 0.628] (T2); 47-feature XGB 0.433 [0.277, 0.601] (T1), 0.530 [0.370, 0.708] (T2), 0.551 [0.384, 0.723] (I-only). Submitted eGeMAPS 'sample-based' value 0.065 (concatenated 14-note takes, not per-note units).
- Position of the pooled library AUC relative to chance (95% CI above / including / below 0.5): XGB T1 mf: above (0.874); XGB T1 all layers: above (0.888); XGB T2 mf: above (0.743); XGB T2 all layers: includes (0.698); LR T1 mf: above (0.864); LR T1 all layers: above (0.830); LR T2 mf: includes (0.324); LR T2 all layers: includes (0.295); XGB T1 clip format mf: above (0.709); XGB T1 clip format all layers: above (0.692); XGB T2 clip format mf: includes (0.719); XGB T2 clip format all layers: includes (0.698); LR T1 clip format mf: includes (0.638); LR T1 clip format all layers: includes (0.656); LR T2 clip format mf: includes (0.657); LR T2 clip format all layers: includes (0.679).
- Native files: the three eGeMAPS features with the largest XGBoost gain are loudness_sma3_meanRisingSlope, loudnessPeaksPerSec, loudness_sma3_stddevRisingSlope (together 0.65 of the total gain). Median lead-in from file start to the attack: CCMusic 0.021 s, SoundFont renders 0.000 s, Otey 0.002 s, libraries Salamander 0.011 s, Iowa 0.250 s, K18 0.005 s, FreePats 0.000 s; median durations and levels differ too (section 1). In the clip format (section 3.2), where lead-in, duration, level and sample rate are equalised, the top feature is MeanUnvoicedSegmentLength (gain share 0.309; median CCMusic 0.035, SoundFont 1.31, Otey 0.124, library mf 0); the SoundFont renders (median 1.8 s) are zero-padded to 3 s in this format, so they end in digital silence.

Cells: AUC [95% CI]; det = share of synthetic files with P(synthetic) > 0.5. The eGeMAPS detector is trained on isolated notes only (CCMusic vs SoundFont renders + Otey, plus library notes under T1). Its T2 is therefore the analogue of the 47-feature **I-only** detector (isolated notes only, libraries zero-shot); the 47-feature T1 / T2 columns come from the whole-corpus models (performances in training; under T1 the libraries were in training as 14-note takes, not as single notes).

| CCMusic notes vs | eGeMAPS XGB T1 | eGeMAPS XGB T2 | 47-feat XGB T1 | 47-feat XGB T2 | 47-feat XGB I-only |
|---|---|---|---|---|---|
| SoundFont renders | 0.941 [0.848, 0.998]; det 96.3% | 0.956 [0.837, 1.000]; det 96.3% | 0.924 [0.856, 0.969]; det 88.7% | 0.934 [0.870, 0.976]; det 88.3% | 0.929 [0.856, 0.975]; det 87.0% |
| Otey notes | 0.773 [0.639, 0.890]; det 68.3% | 0.863 [0.669, 0.974]; det 69.2% | 0.779 [0.667, 0.879]; det 52.5% | 0.807 [0.696, 0.900]; det 50.0% | 0.779 [0.640, 0.891]; det 38.3% |
| algorithmic (SoundFont + Otey) | 0.918 [0.812, 0.994]; det 92.5% | 0.943 [0.822, 1.000]; det 92.6% | 0.904 [0.825, 0.963]; det 83.8% | 0.917 [0.842, 0.971]; det 83.1% | 0.908 [0.820, 0.970]; det 80.4% |
| library notes, all 4 libraries, mf layer | 0.874 [0.714, 0.987]; det 81.2% | 0.743 [0.519, 0.899]; det 25.0% | 0.433 [0.277, 0.601]; det 13.4% | 0.530 [0.370, 0.708]; det 13.4% | 0.551 [0.384, 0.723]; det 11.2% |
| library notes, all 4 libraries, all layers | 0.888 [0.756, 0.984]; det 85.0% | 0.698 [0.480, 0.894]; det 19.8% | 0.466 [0.302, 0.601]; det 15.1% | 0.570 [0.389, 0.720]; det 14.0% | 0.600 [0.403, 0.763]; det 13.1% |

Same, logistic regression:

| CCMusic notes vs | eGeMAPS LR T1 | eGeMAPS LR T2 | 47-feat LR T1 | 47-feat LR T2 | 47-feat LR I-only |
|---|---|---|---|---|---|
| SoundFont renders | 0.862 [0.773, 0.932]; det 83.2% | 0.886 [0.694, 0.982]; det 88.2% | 0.814 [0.730, 0.880]; det 90.3% | 0.835 [0.756, 0.900]; det 89.9% | 0.921 [0.849, 0.977]; det 92.3% |
| Otey notes | 0.582 [0.429, 0.737]; det 31.7% | 0.726 [0.499, 0.873]; det 54.2% | 0.715 [0.640, 0.781]; det 76.7% | 0.727 [0.653, 0.800]; det 76.7% | 0.811 [0.700, 0.919]; det 66.7% |
| algorithmic (SoundFont + Otey) | 0.824 [0.709, 0.920]; det 76.2% | 0.864 [0.672, 0.975]; det 83.6% | 0.800 [0.719, 0.873]; det 88.4% | 0.820 [0.744, 0.891]; det 88.1% | 0.906 [0.826, 0.970]; det 88.8% |
| library notes, all 4 libraries, mf layer | 0.864 [0.716, 0.971]; det 80.8% | 0.324 [0.071, 0.628]; det 16.5% | 0.484 [0.362, 0.615]; det 36.2% | 0.532 [0.422, 0.653]; det 36.6% | 0.535 [0.380, 0.699]; det 16.1% |
| library notes, all 4 libraries, all layers | 0.830 [0.683, 0.971]; det 75.0% | 0.295 [0.127, 0.569]; det 16.2% | 0.506 [0.367, 0.626]; det 40.9% | 0.540 [0.414, 0.655]; det 38.4% | 0.522 [0.367, 0.698]; det 17.4% |

Format-equalised sensitivity (section 3.2): every unit (CCMusic, SoundFont, Otey, library notes) passed through the listening-clip transform (44.1 kHz mono, onset-trimmed, 3 s, RMS-normalised) before eGeMAPS:

| CCMusic notes vs | eGeMAPS XGB T1, clip format | eGeMAPS XGB T2, clip format | eGeMAPS LR T1, clip format | eGeMAPS LR T2, clip format |
|---|---|---|---|---|
| SoundFont renders | 0.927 [0.839, 0.984]; det 93.6% | 0.951 [0.827, 1.000]; det 93.0% | 0.881 [0.723, 0.964]; det 92.0% | 0.923 [0.756, 0.993]; det 94.0% |
| Otey notes | 0.776 [0.633, 0.893]; det 72.5% | 0.702 [0.480, 0.839]; det 28.3% | 0.571 [0.401, 0.712]; det 25.0% | 0.746 [0.557, 0.842]; det 50.0% |
| algorithmic (SoundFont + Otey) | 0.907 [0.815, 0.977]; det 90.7% | 0.917 [0.770, 0.998]; det 84.2% | 0.839 [0.674, 0.955]; det 82.9% | 0.899 [0.740, 0.990]; det 88.0% |
| library notes, all 4 libraries, mf layer | 0.709 [0.548, 0.847]; det 62.9% | 0.719 [0.466, 0.884]; det 17.0% | 0.638 [0.470, 0.771]; det 46.0% | 0.657 [0.438, 0.830]; det 16.5% |
| library notes, all 4 libraries, all layers | 0.692 [0.528, 0.835]; det 59.6% | 0.698 [0.455, 0.859]; det 10.7% | 0.656 [0.494, 0.773]; det 49.5% | 0.679 [0.466, 0.810]; det 15.5% |

Listening clips (LOGO, XGB): eGeMAPS detects Iowa 4/7, K18 7/7, Salamander 4/6, SoundFont (Casio CA-110) 5/5, Otey 4/5; AUC recorded vs algorithmic 0.900 [0.700, 1.000] (47-feature: Iowa 2/7, K18 4/7, Salamander 3/6, SoundFont (Casio CA-110) 3/5, Otey 3/5; AUC 0.665).

Submitted paper (Table 2 row 3, `quality/opensmile_baseline.py`, different units and folds): eGeMAPS 0.950 [0.854, 1.000] real vs algorithmic, 0.065 [0.001, 0.194] real vs sample-based (8 concatenated 14-note takes per library, in training as synthetic).

## 1. Units, folds and audio loading

File lists = protocol v2 section 3: 668 CCMusic note files, 762 SoundFont single-note renders, 120 Otey notes (rows of `analysis/out/folds.csv`, with their CV group and FIXED fold), 999 library note files (224 in the mf-equivalent layer; the rows of `analysis/out/pernote_library_scores.csv`, checked identical to `protocol_v2.library_note_files()`). Every unit is one struck note in its own file. Library notes take their library's fold from folds.csv (FreePats, Iowa: fold 1; K18, Salamander: fold 2), so under T1 each library is scored by the fold model that never saw that library. CCMusic = 6 brand groups (protocol v2), not the 8 filename-prefix groups of the submitted baseline.

**Feature extraction** = eGeMAPSv02 functionals (88), `opensmile.Smile(feature_set=eGeMAPSv02, feature_level=Functionals)` with default options (channels=0, mixdown=False, no resampling), i.e. the call of `quality/opensmile_baseline.py`. Audio loading is the baseline's: single-note corpus files and clips via `smile.process_file(path)` (baseline l.59); library audio via `detune.segment.load_mono` (mean of channels, native rate) + `smile.process_signal` (baseline l.95/101, used there for the concatenated takes).

| source | files | errors | sample rate | channels | loader | median duration (s) | median lead-in before the attack (s) | median peak |x| | files with |x| >= 1 | NaN rate (88 features) |
|---|---|---|---|---|---|---|---|---|---|---|
| ccmusic | 668 | 0 | 44100 | 1 | file | 4.3 | 0.021 | 0.82 | 0 | 0.000 |
| soundfont_single | 762 | 0 | 44100 | 1 | file | 1.8 | 0.000 | 0.90 | 0 | 0.000 |
| otey_single | 120 | 0 | 44100 | 1 | file | 3.0 | 0.002 | 0.16 | 0 | 0.000 |
| library Salamander (all layers) | 480 | 0 | 48000 | 2 | mono | 14.8 | 0.011 | 0.21 | 0 | 0.000 |
| library Iowa (all layers) | 261 | 0 | 44100 | 2 | mono | 30.9 | 0.250 | 0.04 | 0 | 0.000 |
| library K18 (all layers) | 180 | 0 | 48000 | 2 | mono | 19.8 | 0.005 | 0.12 | 0 | 0.000 |
| library FreePats (all layers) | 78 | 0 | 44100 | 2 | mono | 7.1 | 0.000 | 0.84 | 0 | 0.000 |
| listening clips | 30 | 0 | 44100 | 1 | file | 3.0 | 0.010 | 0.78 | 0 | 0.000 |
| clip source files | 30 | 0 | 44100/48000 | 1/2 | file, mono | 16.0 | 0.008 | 0.19 | 0 | 0.000 |
| sensitivity sr44k: library notes | 999 | 0 | 44100 (48000 resampled) | 2 | mono + resample | 15.8 | as native | 0.14 | 0 | 0.000 |
| sensitivity clip format: corpus notes | 1550 | 0 | 44100 | 1 | clip | 3.0 | 0.010 | 0.81 | 0 | 0.000 |
| sensitivity clip format: library notes | 999 | 0 | 44100 | 1 | clip | 3.0 | 0.010 | 0.86 | 0 | 0.000 |
| sensitivity clip format: clip sources | 30 | 0 | 44100 | 1 | clip | 3.0 | 0.010 | 0.78 | 0 | 0.000 |

Lead-in = time from the file start to the first sample above 2% of the file peak (the onset rule of the listening-clip transform); clip-format units start 10 ms before that point. Median peak of the library notes relative to CCMusic: Salamander -12 dB, Iowa -27 dB, K18 -17 dB, FreePats +0 dB.

Differences from the submitted baseline and checks:

1. **Units**: library notes are single-note files (format-matched to CCMusic / SoundFont / Otey), not concatenations of 14 random notes; nothing else is concatenated.
2. **Loader equivalence**: all CCMusic, SoundFont, Otey and clip files are mono, so the two loading routes coincide for them: 60 files (first 20 of each source) extracted both ways differ by max |d| = 0 (0 NaN mismatches). The library files are stereo: as in the baseline they are mixed to mono (mean of channels); `process_file` would have used the left channel only. Iowa .aiff files cannot be read by `process_file` in this environment (audiofile needs mediainfo/ffmpeg for AIFF); the baseline never did either.
3. **Sample rate**: K18 and Salamander are 48 kHz, everything else 44.1 kHz; eGeMAPS runs at the native rate (as the baseline and the 47-feature pipeline). Sensitivity: library notes resampled to 44.1 kHz with opensmile's own resampler (`sampling_rate=44100, resample=True`; section 3.1). Check, max |d| between the resampled and the native features: FreePats 0, Iowa 0 (44.1 kHz already, so unchanged as expected); K18 3.26e+03, Salamander 4.71e+03 (resampled).
4. **int16 conversion**: the opensmile wrapper feeds `(x * 32768).astype(int16)` to openSMILE, which wraps around for |x| >= 1; the column "files with |x| >= 1" counts the affected files (same behaviour as the baseline).
5. **Errors** are recorded, not silently dropped (the baseline's `except: pass`): 0 file(s) with an extraction error.
6. **Folds / model / CI**: fixed protocol-v2 folds instead of `GroupKFold` (tie-order dependent); XGBoost hyper-parameters as the baseline but n_jobs=1, random_state=0; scale_pos_weight = global real/synthetic ratio of the training population; stratified grouped bootstrap (real and synthetic groups resampled separately, B = 2000, seed 0) instead of the baseline's unstratified group bootstrap.

## 2. Per-note isolated regime with eGeMAPS (out-of-fold / zero-shot)

"det." = share of synthetic files with P(synthetic) > 0.5; "mean P" = mean P(synthetic) of the synthetic files. CCMusic false alarms (share of CCMusic files with P > 0.5) and mean P are given below each table.

### 2.1 XGBoost, 88 eGeMAPS functionals

| CCMusic notes vs | files/groups | T1 AUC [95% CI] | T1 det. | T1 mean P | T2 AUC [95% CI] | T2 det. | T2 mean P |
|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.941 [0.848, 0.998] | 96.3% | 0.947 | 0.956 [0.837, 1.000] | 96.3% | 0.953 |
| Otey notes | 668/6 vs 120/1 | 0.773 [0.639, 0.890] | 68.3% | 0.672 | 0.863 [0.669, 0.974] | 69.2% | 0.667 |
| algorithmic (SoundFont + Otey) | 668/6 vs 882/11 | 0.918 [0.812, 0.994] | 92.5% | 0.910 | 0.943 [0.822, 1.000] | 92.6% | 0.914 |
| library notes, all 4 libraries, mf layer | 668/6 vs 224/4 | 0.874 [0.714, 0.987] | 81.2% | 0.804 | 0.743 [0.519, 0.899] | 25.0% | 0.273 |
| library notes, all 4 libraries, all layers | 668/6 vs 999/4 | 0.888 [0.756, 0.984] | 85.0% | 0.843 | 0.698 [0.480, 0.894] | 19.8% | 0.205 |
| library notes, Salamander, mf layer | 668/6 vs 30/1 | 0.849 [0.703, 0.964] | 73.3% | 0.735 | 0.593 [0.376, 0.797] | 3.3% | 0.038 |
| library notes, Iowa, mf layer | 668/6 vs 86/1 | 0.962 [0.893, 1.000] | 100.0% | 0.992 | 0.773 [0.530, 0.932] | 4.7% | 0.139 |
| library notes, K18, mf layer | 668/6 vs 30/1 | 0.927 [0.810, 0.996] | 93.3% | 0.927 | 0.784 [0.551, 0.940] | 30.0% | 0.308 |
| library notes, FreePats, mf layer | 668/6 vs 78/1 | 0.766 [0.609, 0.920] | 59.0% | 0.575 | 0.752 [0.548, 0.878] | 53.8% | 0.499 |
| library notes, Salamander, all layers | 668/6 vs 480/1 | 0.854 [0.709, 0.966] | 77.9% | 0.774 | 0.617 [0.398, 0.813] | 9.0% | 0.086 |
| library notes, Iowa, all layers | 668/6 vs 261/1 | 0.969 [0.903, 1.000] | 100.0% | 0.996 | 0.769 [0.534, 0.923] | 21.5% | 0.246 |
| library notes, K18, all layers | 668/6 vs 180/1 | 0.917 [0.794, 0.995] | 93.3% | 0.921 | 0.790 [0.562, 0.942] | 31.7% | 0.333 |
| library notes, FreePats, all layers | 668/6 vs 78/1 | 0.766 [0.609, 0.920] | 59.0% | 0.575 | 0.752 [0.548, 0.878] | 53.8% | 0.499 |

CCMusic false alarms: T1 21.9% (mean P 0.226); T2 10.0% (mean P 0.110).

### 2.2 Logistic regression, 88 eGeMAPS functionals

| CCMusic notes vs | files/groups | T1 AUC [95% CI] | T1 det. | T1 mean P | T2 AUC [95% CI] | T2 det. | T2 mean P |
|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.862 [0.773, 0.932] | 83.2% | 0.804 | 0.886 [0.694, 0.982] | 88.2% | 0.851 |
| Otey notes | 668/6 vs 120/1 | 0.582 [0.429, 0.737] | 31.7% | 0.337 | 0.726 [0.499, 0.873] | 54.2% | 0.459 |
| algorithmic (SoundFont + Otey) | 668/6 vs 882/11 | 0.824 [0.709, 0.920] | 76.2% | 0.741 | 0.864 [0.672, 0.975] | 83.6% | 0.798 |
| library notes, all 4 libraries, mf layer | 668/6 vs 224/4 | 0.864 [0.716, 0.971] | 80.8% | 0.784 | 0.324 [0.071, 0.628] | 16.5% | 0.159 |
| library notes, all 4 libraries, all layers | 668/6 vs 999/4 | 0.830 [0.683, 0.971] | 75.0% | 0.733 | 0.295 [0.127, 0.569] | 16.2% | 0.136 |
| library notes, Salamander, mf layer | 668/6 vs 30/1 | 0.736 [0.616, 0.847] | 66.7% | 0.593 | 0.286 [0.179, 0.392] | 3.3% | 0.073 |
| library notes, Iowa, mf layer | 668/6 vs 86/1 | 0.960 [0.930, 0.984] | 95.3% | 0.941 | 0.030 [0.021, 0.037] | 2.3% | 0.018 |
| library notes, K18, mf layer | 668/6 vs 30/1 | 0.979 [0.945, 0.996] | 100.0% | 0.982 | 0.491 [0.338, 0.586] | 46.7% | 0.334 |
| library notes, FreePats, mf layer | 668/6 vs 78/1 | 0.763 [0.636, 0.883] | 62.8% | 0.608 | 0.599 [0.404, 0.754] | 25.6% | 0.281 |
| library notes, Salamander, all layers | 668/6 vs 480/1 | 0.718 [0.593, 0.838] | 56.7% | 0.554 | 0.278 [0.165, 0.392] | 6.0% | 0.065 |
| library notes, Iowa, all layers | 668/6 vs 261/1 | 0.968 [0.943, 0.986] | 96.9% | 0.954 | 0.067 [0.047, 0.078] | 6.5% | 0.047 |
| library notes, K18, all layers | 668/6 vs 180/1 | 0.959 [0.919, 0.984] | 97.2% | 0.946 | 0.538 [0.378, 0.632] | 53.3% | 0.391 |
| library notes, FreePats, all layers | 668/6 vs 78/1 | 0.763 [0.636, 0.883] | 62.8% | 0.608 | 0.599 [0.404, 0.754] | 25.6% | 0.281 |

CCMusic false alarms: T1 26.6% (mean P 0.294); T2 16.2% (mean P 0.185).

### 2.3 Per-library AUC next to the 47-feature detector (XGBoost)

| CCMusic notes vs | eGeMAPS T1 | eGeMAPS T2 | 47-feat T1 | 47-feat T2 | 47-feat I-only |
|---|---|---|---|---|---|
| library notes, all 4 libraries, mf layer | 0.874 [0.714, 0.987] | 0.743 [0.519, 0.899] | 0.433 [0.277, 0.601] | 0.530 [0.370, 0.708] | 0.551 [0.384, 0.723] |
| library notes, all 4 libraries, all layers | 0.888 [0.756, 0.984] | 0.698 [0.480, 0.894] | 0.466 [0.302, 0.601] | 0.570 [0.389, 0.720] | 0.600 [0.403, 0.763] |
| library notes, Salamander, mf layer | 0.849 [0.703, 0.964] | 0.593 [0.376, 0.797] | 0.526 [0.395, 0.643] | 0.659 [0.514, 0.786] | 0.665 [0.505, 0.802] |
| library notes, Iowa, mf layer | 0.962 [0.893, 1.000] | 0.773 [0.530, 0.932] | 0.287 [0.188, 0.376] | 0.388 [0.276, 0.501] | 0.431 [0.293, 0.567] |
| library notes, K18, mf layer | 0.927 [0.810, 0.996] | 0.784 [0.551, 0.940] | 0.529 [0.423, 0.624] | 0.627 [0.512, 0.727] | 0.642 [0.510, 0.760] |
| library notes, FreePats, mf layer | 0.766 [0.609, 0.920] | 0.752 [0.548, 0.878] | 0.522 [0.405, 0.625] | 0.600 [0.475, 0.711] | 0.604 [0.467, 0.725] |
| library notes, Salamander, all layers | 0.854 [0.709, 0.966] | 0.617 [0.398, 0.813] | 0.523 [0.398, 0.635] | 0.638 [0.496, 0.763] | 0.675 [0.516, 0.811] |
| library notes, Iowa, all layers | 0.969 [0.903, 1.000] | 0.769 [0.534, 0.923] | 0.305 [0.212, 0.389] | 0.395 [0.287, 0.504] | 0.426 [0.298, 0.553] |
| library notes, K18, all layers | 0.917 [0.794, 0.995] | 0.790 [0.562, 0.942] | 0.522 [0.410, 0.621] | 0.628 [0.505, 0.736] | 0.649 [0.511, 0.769] |
| library notes, FreePats, all layers | 0.766 [0.609, 0.920] | 0.752 [0.548, 0.878] | 0.522 [0.405, 0.625] | 0.600 [0.475, 0.711] | 0.604 [0.467, 0.725] |

## 3. Sensitivity analyses

### 3.1 Which library notes are in training (T1) and the sample rate

**T1mf** = only the 224 mf-layer library notes in training (other layers scored by the fold model of their library, which never saw that library). **sr44k** = library notes resampled to 44.1 kHz before extraction (K18 and Salamander change; FreePats and Iowa are 44.1 kHz already). Cells: AUC [95% CI]; det.

**XGBoost**

| CCMusic notes vs | T1 (all layers in training) | T1mf | T1 sr44k | T2 | T2 sr44k |
|---|---|---|---|---|---|
| SoundFont renders | 0.941 [0.848, 0.998]; det 96.3% | 0.948 [0.850, 0.999]; det 97.2% | 0.943 [0.855, 0.997]; det 96.2% | 0.956 [0.837, 1.000]; det 96.3% | 0.956 [0.837, 1.000]; det 96.3% |
| Otey notes | 0.773 [0.639, 0.890]; det 68.3% | 0.781 [0.641, 0.892]; det 68.3% | 0.778 [0.649, 0.897]; det 68.3% | 0.863 [0.669, 0.974]; det 69.2% | 0.863 [0.669, 0.974]; det 69.2% |
| algorithmic (SoundFont + Otey) | 0.918 [0.812, 0.994]; det 92.5% | 0.925 [0.817, 0.998]; det 93.3% | 0.920 [0.819, 0.995]; det 92.4% | 0.943 [0.822, 1.000]; det 92.6% | 0.943 [0.822, 1.000]; det 92.6% |
| library notes, all 4 libraries, mf layer | 0.874 [0.714, 0.987]; det 81.2% | 0.841 [0.673, 0.976]; det 79.5% | 0.875 [0.718, 0.986]; det 80.8% | 0.743 [0.519, 0.899]; det 25.0% | 0.742 [0.519, 0.897]; det 25.0% |
| library notes, all 4 libraries, all layers | 0.888 [0.756, 0.984]; det 85.0% | 0.859 [0.704, 0.978]; det 83.5% | 0.890 [0.761, 0.984]; det 85.5% | 0.698 [0.480, 0.894]; det 19.8% | 0.696 [0.478, 0.893]; det 19.7% |
| library notes, Salamander, mf layer | 0.849 [0.703, 0.964]; det 73.3% | 0.812 [0.641, 0.946]; det 70.0% | 0.849 [0.707, 0.966]; det 73.3% | 0.593 [0.376, 0.797]; det 3.3% | 0.584 [0.370, 0.786]; det 3.3% |
| library notes, Iowa, mf layer | 0.962 [0.893, 1.000]; det 100.0% | 0.914 [0.772, 0.999]; det 100.0% | 0.962 [0.896, 1.000]; det 100.0% | 0.773 [0.530, 0.932]; det 4.7% | 0.773 [0.530, 0.932]; det 4.7% |
| library notes, K18, mf layer | 0.927 [0.810, 0.996]; det 93.3% | 0.914 [0.775, 0.997]; det 93.3% | 0.928 [0.817, 0.996]; det 93.3% | 0.784 [0.551, 0.940]; det 30.0% | 0.783 [0.552, 0.938]; det 30.0% |
| library notes, FreePats, mf layer | 0.766 [0.609, 0.920]; det 59.0% | 0.743 [0.573, 0.900]; det 55.1% | 0.767 [0.614, 0.925]; det 57.7% | 0.752 [0.548, 0.878]; det 53.8% | 0.752 [0.548, 0.878]; det 53.8% |
| library notes, Salamander, all layers | 0.854 [0.709, 0.966]; det 77.9% | 0.827 [0.656, 0.956]; det 75.8% | 0.855 [0.712, 0.969]; det 79.0% | 0.617 [0.398, 0.813]; det 9.0% | 0.613 [0.394, 0.808]; det 8.8% |
| library notes, Iowa, all layers | 0.969 [0.903, 1.000]; det 100.0% | 0.925 [0.791, 1.000]; det 99.6% | 0.971 [0.913, 1.000]; det 100.0% | 0.769 [0.534, 0.923]; det 21.5% | 0.769 [0.534, 0.923]; det 21.5% |
| library notes, K18, all layers | 0.917 [0.794, 0.995]; det 93.3% | 0.901 [0.754, 0.997]; det 92.8% | 0.918 [0.802, 0.995]; det 93.9% | 0.790 [0.562, 0.942]; det 31.7% | 0.788 [0.560, 0.940]; det 31.7% |
| library notes, FreePats, all layers | 0.766 [0.609, 0.920]; det 59.0% | 0.743 [0.573, 0.900]; det 55.1% | 0.767 [0.614, 0.925]; det 57.7% | 0.752 [0.548, 0.878]; det 53.8% | 0.752 [0.548, 0.878]; det 53.8% |

**Logistic regression**

| CCMusic notes vs | T1 (all layers in training) | T1mf | T1 sr44k | T2 | T2 sr44k |
|---|---|---|---|---|---|
| SoundFont renders | 0.862 [0.773, 0.932]; det 83.2% | 0.879 [0.776, 0.949]; det 86.2% | 0.864 [0.775, 0.932]; det 82.9% | 0.886 [0.694, 0.982]; det 88.2% | 0.886 [0.694, 0.982]; det 88.2% |
| Otey notes | 0.582 [0.429, 0.737]; det 31.7% | 0.619 [0.432, 0.802]; det 39.2% | 0.587 [0.438, 0.737]; det 31.7% | 0.726 [0.499, 0.873]; det 54.2% | 0.726 [0.499, 0.873]; det 54.2% |
| algorithmic (SoundFont + Otey) | 0.824 [0.709, 0.920]; det 76.2% | 0.844 [0.710, 0.941]; det 79.8% | 0.827 [0.712, 0.920]; det 76.0% | 0.864 [0.672, 0.975]; det 83.6% | 0.864 [0.672, 0.975]; det 83.6% |
| library notes, all 4 libraries, mf layer | 0.864 [0.716, 0.971]; det 80.8% | 0.780 [0.597, 0.938]; det 66.5% | 0.871 [0.724, 0.974]; det 79.9% | 0.324 [0.071, 0.628]; det 16.5% | 0.324 [0.069, 0.631]; det 17.0% |
| library notes, all 4 libraries, all layers | 0.830 [0.683, 0.971]; det 75.0% | 0.811 [0.656, 0.945]; det 70.2% | 0.835 [0.691, 0.970]; det 76.3% | 0.295 [0.127, 0.569]; det 16.2% | 0.296 [0.127, 0.572]; det 16.3% |
| library notes, Salamander, mf layer | 0.736 [0.616, 0.847]; det 66.7% | 0.753 [0.599, 0.880]; det 63.3% | 0.750 [0.618, 0.871]; det 56.7% | 0.286 [0.179, 0.392]; det 3.3% | 0.265 [0.157, 0.374]; det 3.3% |
| library notes, Iowa, mf layer | 0.960 [0.930, 0.984]; det 95.3% | 0.871 [0.784, 0.948]; det 76.7% | 0.963 [0.935, 0.985]; det 96.5% | 0.030 [0.021, 0.037]; det 2.3% | 0.030 [0.021, 0.037]; det 2.3% |
| library notes, K18, mf layer | 0.979 [0.945, 0.996]; det 100.0% | 0.980 [0.935, 1.000]; det 100.0% | 0.981 [0.952, 0.996]; det 100.0% | 0.491 [0.338, 0.586]; det 46.7% | 0.512 [0.355, 0.606]; det 50.0% |
| library notes, FreePats, mf layer | 0.763 [0.636, 0.883]; det 62.8% | 0.614 [0.457, 0.771]; det 43.6% | 0.774 [0.646, 0.890]; det 62.8% | 0.599 [0.404, 0.754]; det 25.6% | 0.599 [0.404, 0.754]; det 25.6% |
| library notes, Salamander, all layers | 0.718 [0.593, 0.838]; det 56.7% | 0.741 [0.588, 0.880]; det 57.3% | 0.727 [0.602, 0.842]; det 59.4% | 0.278 [0.165, 0.392]; det 6.0% | 0.279 [0.165, 0.396]; det 5.4% |
| library notes, Iowa, all layers | 0.968 [0.943, 0.986]; det 96.9% | 0.896 [0.813, 0.960]; det 83.1% | 0.970 [0.948, 0.987]; det 97.3% | 0.067 [0.047, 0.078]; det 6.5% | 0.067 [0.047, 0.078]; det 6.5% |
| library notes, K18, all layers | 0.959 [0.919, 0.984]; det 97.2% | 0.964 [0.910, 0.993]; det 97.2% | 0.954 [0.918, 0.979]; det 96.7% | 0.538 [0.378, 0.632]; det 53.3% | 0.544 [0.384, 0.635]; det 55.6% |
| library notes, FreePats, all layers | 0.763 [0.636, 0.883]; det 62.8% | 0.614 [0.457, 0.771]; det 43.6% | 0.774 [0.646, 0.890]; det 62.8% | 0.599 [0.404, 0.754]; det 25.6% | 0.599 [0.404, 0.754]; det 25.6% |

### 3.2 Format-equalised units (listening-clip format)

Every unit -- CCMusic, SoundFont, Otey and library notes -- is first passed through `quality/build_listening_kit.process`, the transform that produced the 30 listening clips: librosa load at 44.1 kHz mono, start 10 ms before the first sample above 2% of the peak, 3.0 s (zero-padded if shorter), 10 ms fades, RMS scaled to 0.1 (-20 dBFS), peak limited to 0.97. This removes the differences in duration, lead-in before the attack, level and sample rate between the sources (section 1 table; "3.0 s" also cuts the long library decays, and pads the 1.8 s SoundFont renders with digital silence); it is not part of the submitted baseline. Same folds, models and conditions. Cells: AUC [95% CI]; det.

**XGBoost**

| CCMusic notes vs | T1 (native files) | T1, clip format | T2 (native files) | T2, clip format |
|---|---|---|---|---|
| SoundFont renders | 0.941 [0.848, 0.998]; det 96.3% | 0.927 [0.839, 0.984]; det 93.6% | 0.956 [0.837, 1.000]; det 96.3% | 0.951 [0.827, 1.000]; det 93.0% |
| Otey notes | 0.773 [0.639, 0.890]; det 68.3% | 0.776 [0.633, 0.893]; det 72.5% | 0.863 [0.669, 0.974]; det 69.2% | 0.702 [0.480, 0.839]; det 28.3% |
| algorithmic (SoundFont + Otey) | 0.918 [0.812, 0.994]; det 92.5% | 0.907 [0.815, 0.977]; det 90.7% | 0.943 [0.822, 1.000]; det 92.6% | 0.917 [0.770, 0.998]; det 84.2% |
| library notes, all 4 libraries, mf layer | 0.874 [0.714, 0.987]; det 81.2% | 0.709 [0.548, 0.847]; det 62.9% | 0.743 [0.519, 0.899]; det 25.0% | 0.719 [0.466, 0.884]; det 17.0% |
| library notes, all 4 libraries, all layers | 0.888 [0.756, 0.984]; det 85.0% | 0.692 [0.528, 0.835]; det 59.6% | 0.698 [0.480, 0.894]; det 19.8% | 0.698 [0.455, 0.859]; det 10.7% |
| library notes, Salamander, mf layer | 0.849 [0.703, 0.964]; det 73.3% | 0.686 [0.518, 0.843]; det 63.3% | 0.593 [0.376, 0.797]; det 3.3% | 0.642 [0.406, 0.804]; det 0.0% |
| library notes, Iowa, mf layer | 0.962 [0.893, 1.000]; det 100.0% | 0.703 [0.542, 0.849]; det 59.3% | 0.773 [0.530, 0.932]; det 4.7% | 0.671 [0.426, 0.827]; det 4.7% |
| library notes, K18, mf layer | 0.927 [0.810, 0.996]; det 93.3% | 0.684 [0.526, 0.840]; det 53.3% | 0.784 [0.551, 0.940]; det 30.0% | 0.758 [0.500, 0.903]; det 36.7% |
| library notes, FreePats, mf layer | 0.766 [0.609, 0.920]; det 59.0% | 0.735 [0.573, 0.875]; det 70.5% | 0.752 [0.548, 0.878]; det 53.8% | 0.787 [0.527, 0.927]; det 29.5% |
| library notes, Salamander, all layers | 0.854 [0.709, 0.966]; det 77.9% | 0.691 [0.525, 0.845]; det 58.8% | 0.617 [0.398, 0.813]; det 9.0% | 0.675 [0.429, 0.834]; det 3.8% |
| library notes, Iowa, all layers | 0.969 [0.903, 1.000]; det 100.0% | 0.716 [0.560, 0.854]; det 65.1% | 0.769 [0.534, 0.923]; det 21.5% | 0.693 [0.438, 0.846]; det 6.1% |
| library notes, K18, all layers | 0.917 [0.794, 0.995]; det 93.3% | 0.644 [0.488, 0.802]; det 48.9% | 0.790 [0.562, 0.942]; det 31.7% | 0.730 [0.479, 0.874]; det 27.8% |
| library notes, FreePats, all layers | 0.766 [0.609, 0.920]; det 59.0% | 0.735 [0.573, 0.875]; det 70.5% | 0.752 [0.548, 0.878]; det 53.8% | 0.787 [0.527, 0.927]; det 29.5% |

CCMusic false alarms, clip format: T1 33.1% (mean P 0.344); T2 13.2% (mean P 0.137).

**Logistic regression**

| CCMusic notes vs | T1 (native files) | T1, clip format | T2 (native files) | T2, clip format |
|---|---|---|---|---|
| SoundFont renders | 0.862 [0.773, 0.932]; det 83.2% | 0.881 [0.723, 0.964]; det 92.0% | 0.886 [0.694, 0.982]; det 88.2% | 0.923 [0.756, 0.993]; det 94.0% |
| Otey notes | 0.582 [0.429, 0.737]; det 31.7% | 0.571 [0.401, 0.712]; det 25.0% | 0.726 [0.499, 0.873]; det 54.2% | 0.746 [0.557, 0.842]; det 50.0% |
| algorithmic (SoundFont + Otey) | 0.824 [0.709, 0.920]; det 76.2% | 0.839 [0.674, 0.955]; det 82.9% | 0.864 [0.672, 0.975]; det 83.6% | 0.899 [0.740, 0.990]; det 88.0% |
| library notes, all 4 libraries, mf layer | 0.864 [0.716, 0.971]; det 80.8% | 0.638 [0.470, 0.771]; det 46.0% | 0.324 [0.071, 0.628]; det 16.5% | 0.657 [0.438, 0.830]; det 16.5% |
| library notes, all 4 libraries, all layers | 0.830 [0.683, 0.971]; det 75.0% | 0.656 [0.494, 0.773]; det 49.5% | 0.295 [0.127, 0.569]; det 16.2% | 0.679 [0.466, 0.810]; det 15.5% |
| library notes, Salamander, mf layer | 0.736 [0.616, 0.847]; det 66.7% | 0.649 [0.475, 0.770]; det 50.0% | 0.286 [0.179, 0.392]; det 3.3% | 0.673 [0.450, 0.791]; det 3.3% |
| library notes, Iowa, mf layer | 0.960 [0.930, 0.984]; det 95.3% | 0.576 [0.422, 0.695]; det 34.9% | 0.030 [0.021, 0.037]; det 2.3% | 0.544 [0.354, 0.653]; det 4.7% |
| library notes, K18, mf layer | 0.979 [0.945, 0.996]; det 100.0% | 0.696 [0.527, 0.817]; det 50.0% | 0.491 [0.338, 0.586]; det 46.7% | 0.743 [0.525, 0.855]; det 26.7% |
| library notes, FreePats, mf layer | 0.763 [0.636, 0.883]; det 62.8% | 0.681 [0.507, 0.803]; det 55.1% | 0.599 [0.404, 0.754]; det 25.6% | 0.742 [0.521, 0.856]; det 30.8% |
| library notes, Salamander, all layers | 0.718 [0.593, 0.838]; det 56.7% | 0.655 [0.482, 0.779]; det 48.3% | 0.278 [0.165, 0.392]; det 6.0% | 0.685 [0.463, 0.802]; det 12.3% |
| library notes, Iowa, all layers | 0.968 [0.943, 0.986]; det 96.9% | 0.681 [0.520, 0.791]; det 56.3% | 0.067 [0.047, 0.078]; det 6.5% | 0.615 [0.408, 0.730]; det 7.7% |
| library notes, K18, all layers | 0.959 [0.919, 0.984]; det 97.2% | 0.611 [0.449, 0.734]; det 40.6% | 0.538 [0.378, 0.632]; det 53.3% | 0.726 [0.512, 0.838]; det 28.9% |
| library notes, FreePats, all layers | 0.763 [0.636, 0.883]; det 62.8% | 0.681 [0.507, 0.803]; det 55.1% | 0.599 [0.404, 0.754]; det 25.6% | 0.742 [0.521, 0.856]; det 30.8% |

CCMusic false alarms, clip format: T1 28.7% (mean P 0.327); T2 13.0% (mean P 0.152).

## 4. What the eGeMAPS XGBoost uses

Top 10 features by share of total gain, XGBoost fitted on all 1550 isolated corpus files (T2 population); univariate AUC (feature value as score; < 0.5 = lower in the synthetic class) and medians.

**Native files** (as the baseline):

| feature | gain share | univariate AUC CCMusic vs algorithmic | univariate AUC CCMusic vs library notes (mf) | median CCMusic | median algorithmic | median library (mf) |
|---|---|---|---|---|---|---|
| loudness_sma3_meanRisingSlope | 0.418 | 0.085 | 0.167 | 12.1 | -0.17 | 0.972 |
| loudnessPeaksPerSec | 0.119 | 0.192 | 0.093 | 0.234 | 0 | 0.0385 |
| loudness_sma3_stddevRisingSlope | 0.110 | 0.093 | 0.168 | 9.27 | 0 | 0.734 |
| mfcc1_sma3_stddevNorm | 0.089 | 0.372 | 0.673 | 0.178 | 0.103 | 0.228 |
| spectralFluxV_sma3nz_stddevNorm | 0.031 | 0.394 | 0.521 | 1.81 | 1.5 | 1.9 |
| VoicedSegmentsPerSec | 0.024 | 0.537 | 0.243 | 0.487 | 0.581 | 0.14 |
| jitterLocal_sma3nz_amean | 0.023 | 0.157 | 0.329 | 0.0148 | 0.00118 | 0.00509 |
| F0semitoneFrom27.5Hz_sma3nz_stddevNorm | 0.021 | 0.129 | 0.239 | 0.128 | 0.000541 | 0.00569 |
| slopeV0-500_sma3nz_stddevNorm | 0.019 | 0.447 | 0.504 | 0.304 | 0.136 | 0.337 |
| slopeV0-500_sma3nz_amean | 0.013 | 0.524 | 0.563 | 0.0117 | 0.0198 | 0.0197 |

**Clip format** (section 3.2):

| feature | gain share | univariate AUC CCMusic vs algorithmic | univariate AUC CCMusic vs library notes (mf) | median CCMusic | median algorithmic | median library (mf) |
|---|---|---|---|---|---|---|
| MeanUnvoicedSegmentLength | 0.309 | 0.898 | 0.438 | 0.035 | 1.29 | 0 |
| loudness_sma3_stddevRisingSlope | 0.211 | 0.069 | 0.315 | 12.8 | 0 | 7.57 |
| mfcc1_sma3_stddevNorm | 0.067 | 0.733 | 0.317 | 0.182 | 0.821 | 0.102 |
| loudness_sma3_meanRisingSlope | 0.060 | 0.072 | 0.326 | 16.7 | -0.0507 | 11 |
| jitterLocal_sma3nz_amean | 0.048 | 0.185 | 0.364 | 0.0126 | 0.000994 | 0.00485 |
| spectralFluxUV_sma3nz_amean | 0.041 | 0.397 | 0.302 | 0.136 | 0.0161 | 0 |
| slopeV0-500_sma3nz_amean | 0.032 | 0.534 | 0.474 | 0.00464 | 0.0186 | 0.0123 |
| equivalentSoundLevel_dBp | 0.021 | 0.351 | 0.426 | -20 | -20 | -20 |
| F0semitoneFrom27.5Hz_sma3nz_stddevNorm | 0.020 | 0.170 | 0.319 | 0.117 | 0.000567 | 0.00571 |
| loudnessPeaksPerSec | 0.018 | 0.134 | 0.389 | 0.334 | 0 | 0.334 |

## 5. Listening-test clips scored out of sample (leave-own-group-out, as RESULTS_v2 section 4)

eGeMAPS of each of the 30 clips the listeners heard (`listening_kit/web/clip_XX.wav`, 3 s, onset-trimmed, RMS-normalised; `process_file`) and of their unprocessed source files. Each clip is scored by a model fitted on all T2 training files (CCMusic + SoundFont renders + Otey; no library) with its own group removed: Otey clips by a model without Otey, Casio CA-110 clips by a model without that SoundFont, library clips by the full model. Unlike RESULTS_v2 s.4, the training set has no performances (eGeMAPS was extracted for isolated notes only). Detection = P(synthetic) > 0.5. The 47-feature rows are recomputed from `analysis/out/stimuli_detector_scores.csv` with the same code. "Trained on clip-format units" = the same LOGO scheme with every training file in the clip format of section 3.2, so that training and test units have the same format. Check: the clip WAVs and the clip transform re-applied to their source files give eGeMAPS values with median relative difference 0.00037 (99th percentile 0.19) and scores differing by max |dP| = 0.025 (XGB) / 0.040 (LR) (the WAVs are PCM_16).

| scoring | Iowa det. | K18 det. | Salamander det. | SoundFont det. | Otey det. | AUC recorded(20) vs algorithmic(10) [95% CI, clips resampled within class] | AUC Iowa+K18(14) vs algorithmic(10) | AUC Salamander(6) vs Iowa+K18(14) |
|---|---|---|---|---|---|---|---|---|
| eGeMAPS XGB, clip, LOGO (primary) | 4/7 | 7/7 | 4/6 | 5/5 | 4/5 | 0.900 [0.700, 1.000] | 0.900 | 0.286 |
| eGeMAPS XGB, clip, one model w/o Otey+CA-110 | 0/7 | 2/7 | 0/6 | 5/5 | 4/5 | 0.935 [0.805, 1.000] | 0.936 | 0.476 |
| eGeMAPS LR, clip, LOGO | 2/7 | 2/7 | 2/6 | 5/5 | 4/5 | 0.930 [0.840, 1.000] | 0.936 | 0.655 |
| eGeMAPS XGB, unprocessed source file, LOGO | 0/7 | 2/7 | 0/6 | 5/5 | 5/5 | 0.990 [0.960, 1.000] | 0.986 | 0.071 |
| eGeMAPS LR, unprocessed source file, LOGO | 0/7 | 5/7 | 1/6 | 4/5 | 5/5 | 0.725 [0.610, 0.850] | 0.643 | 0.631 |
| eGeMAPS XGB trained on clip-format units, clip, LOGO | 0/7 | 3/7 | 0/6 | 5/5 | 0/5 | 0.840 [0.670, 0.975] | 0.814 | 0.512 |
| eGeMAPS LR trained on clip-format units, clip, LOGO | 1/7 | 1/7 | 0/6 | 5/5 | 2/5 | 0.885 [0.760, 0.990] | 0.893 | 0.595 |
| 47-feature XGB all47, clip, LOGO (RESULTS_v2 s.4) | 2/7 | 4/7 | 3/6 | 3/5 | 3/5 | 0.665 [0.470, 0.845] | 0.679 | 0.512 |
| 47-feature LR all47, clip, LOGO (RESULTS_v2 s.4) | 4/7 | 6/7 | 3/6 | 1/5 | 5/5 | 0.410 [0.210, 0.625] | 0.379 | 0.381 |
| 47-feature XGB all47, source file, LOGO (RESULTS_v2 s.4) | 1/7 | 2/7 | 1/6 | 4/5 | 3/5 | 0.850 [0.715, 0.955] | 0.821 | 0.619 |

Per-clip P(synthetic):

| clip | class | note | eGeMAPS XGB LOGO | eGeMAPS LR LOGO | eGeMAPS XGB src LOGO | eGeMAPS XGB LOGO, clip-format training | 47-feat XGB LOGO |
|---|---|---|---|---|---|---|---|
| clip_01 | Salamander | D#3 | 0.558 | 0.430 | 0.001 | 0.004 | 0.095 |
| clip_02 | Iowa | D3 | 0.980 | 0.222 | 0.022 | 0.047 | 0.250 |
| clip_03 | K18 | D#3 | 0.980 | 0.425 | 0.005 | 0.825 | 0.300 |
| clip_04 | Salamander | C4 | 0.746 | 0.818 | 0.001 | 0.011 | 0.861 |
| clip_05 | Salamander | D#4 | 0.542 | 0.173 | 0.001 | 0.004 | 0.684 |
| clip_06 | Iowa | B4 | 0.035 | 0.622 | 0.001 | 0.004 | 0.889 |
| clip_07 | Otey | B4 | 0.997 | 0.991 | 0.985 | 0.295 | 0.981 |
| clip_08 | Salamander | A3 | 0.184 | 0.069 | 0.000 | 0.002 | 0.008 |
| clip_09 | Iowa | F4 | 0.169 | 0.109 | 0.003 | 0.001 | 0.555 |
| clip_10 | SoundFont (Casio CA-110) | C4 | 0.998 | 0.983 | 1.000 | 1.000 | 0.998 |
| clip_11 | Salamander | F#4 | 0.913 | 0.574 | 0.003 | 0.018 | 0.010 |
| clip_12 | Iowa | B3 | 0.771 | 0.023 | 0.052 | 0.001 | 0.279 |
| clip_13 | Otey | C5 | 0.998 | 0.995 | 0.989 | 0.143 | 0.800 |
| clip_14 | Iowa | G3 | 0.948 | 0.061 | 0.027 | 0.002 | 0.008 |
| clip_15 | K18 | C5 | 0.986 | 0.943 | 0.990 | 0.907 | 0.998 |
| clip_16 | K18 | C3 | 0.975 | 0.067 | 0.004 | 0.002 | 0.000 |
| clip_17 | K18 | F#3 | 0.858 | 0.084 | 0.016 | 0.010 | 0.714 |
| clip_18 | K18 | D#4 | 0.661 | 0.129 | 0.006 | 0.004 | 0.306 |
| clip_19 | Otey | E4 | 0.011 | 0.321 | 0.995 | 0.001 | 0.936 |
| clip_20 | Salamander | D#5 | 0.256 | 0.488 | 0.002 | 0.003 | 1.000 |
| clip_21 | SoundFont (Casio CA-110) | C3 | 0.998 | 0.973 | 1.000 | 1.000 | 0.999 |
| clip_22 | K18 | A3 | 0.835 | 0.017 | 0.008 | 0.002 | 0.995 |
| clip_23 | K18 | D#5 | 0.737 | 0.890 | 0.981 | 0.914 | 0.640 |
| clip_24 | SoundFont (Casio CA-110) | A4 | 0.997 | 0.984 | 0.999 | 0.999 | 0.132 |
| clip_25 | Otey | G4 | 0.997 | 0.650 | 0.996 | 0.063 | 0.320 |
| clip_26 | SoundFont (Casio CA-110) | A3 | 0.995 | 0.998 | 0.998 | 0.998 | 0.083 |
| clip_27 | Iowa | D5 | 0.407 | 0.534 | 0.006 | 0.042 | 0.010 |
| clip_28 | Otey | E3 | 0.997 | 0.871 | 0.994 | 0.065 | 0.378 |
| clip_29 | Iowa | F3 | 0.839 | 0.113 | 0.002 | 0.003 | 0.017 |
| clip_30 | SoundFont (Casio CA-110) | C5 | 0.997 | 0.999 | 0.996 | 1.000 | 0.566 |
