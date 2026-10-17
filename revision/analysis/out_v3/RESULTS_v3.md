# RESULTS v3 -- protocol v3 (after the independent code audit)

Generated 2026-10-01 18:49 by one command, from the project root:

```
python Stattia/revision/analysis/run_v3.py
```

Environment: Python 3.10.7, numpy 1.26.4, pandas 1.5.2, xgboost 3.2.0 (n_jobs=1, random_state=0). Scripts: `Stattia/revision/analysis/{run_v3, v3_lib, v3_sf2, v3_render, v3_controls}.py` (+ v2 modules for the corpus). Runtime 104 s. Determinism check vs the previous run: 7 of 7 output CSVs byte-identical.

Fixes against v2 (CODE_AUDIT.md): A1 own-fold scoring of every never-trained row (v2 rule as secondary column); A2 cleaned real class + video-id grouping; A3 SoundFont velocity copies removed; A4 one-group synthetic sides get "real-side only" intervals. Outputs: `out_v3/` (folds.csv, table7_v3.csv, controls_scores.csv, c2_soundfont_categories.csv, c4_phaseogram_c3.csv, c5_loop_index.csv, excluded_real_v3.csv, key_numbers_v3.json, cache/).

## 0. Key numbers (T2, own-fold scoring, cleaned real class)

| comparison | all 47 | 16 Group-A medians |
|---|---|---|
| isolated: CCMusic vs SoundFont | 0.937 [0.870, 0.983]; det 84.1% | 0.877 [0.773, 0.958]; det 76.8% |
| isolated: CCMusic vs Otey | 0.821 (real-side only [0.703, 0.926]); det 34.2% | 0.675 (real-side only [0.574, 0.779]); det 23.3% |
| isolated: CCMusic vs library notes (mf) | 0.621 [0.482, 0.753]; det 15.2% | 0.528 [0.419, 0.648]; det 17.0% |
| performance: real vs algorithmic renders | 0.943 [0.907, 0.978]; det 36.7% | 0.723 [0.622, 0.839]; det 9.2% |
| performance: real vs sampled digital pianos | 0.871 [0.843, 0.891]; det 7.0% | 0.562 [0.525, 0.622]; det 1.6% |
| performance: real vs modelled digital | 0.854 (real-side only [0.845, 0.863]); det 9.5% | 0.583 (real-side only [0.570, 0.596]); det 0.0% |
| performance: real vs Korg M1 | 0.928 (real-side only [0.922, 0.934]); det 23.3% | 0.874 (real-side only [0.866, 0.882]); det 16.7% |

Control table B in brief (I-only, all 47, T2, AUC vs CCMusic):

| source | original | C1a 1.2 s | C1b 4.0 s | C3 | C1a+C3 | C1b+C3 |
|---|---|---|---|---|---|---|
| SoundFont | 0.937 | 0.819 | 0.840 | 0.923 | 0.861 | 0.940 |
| SoundFont, unaltered sample | 0.931 | 0.818 | 0.826 | 0.912 | 0.850 | 0.929 |
| Otey | 0.821 | 0.645 | 0.780 | 0.699 | 0.759 | 0.831 |
| library notes, mf layer | 0.621 | 0.658 | 0.621 | 0.585 | 0.624 | 0.642 |


## 1. Corpus v3

Real class cleaned: 21 non-acoustic files of CODE_AUDIT B1 removed (missing from the corpus: none); audit/real_class_exclusions.csv: 105 file(s) with decision exclude (84 removed); 0 byte-identical duplicate copies removed (). Real performances regrouped by YouTube video id: 2318 -> 2275 groups. SoundFont velocity copies (identical 47-feature vectors within an instrument): 84 files flagged and left out of the isolated analyses (sensitivity row keeps them). Folds: v2 rule re-applied to the v3 groups (`out_v3/folds.csv`). Corpus: 4355 files; algo_performance 120, algo_single 882, digital_other 1, m1 30, modelled_digital 21, real_isolated 668, real_performance 2471, sampled_digital_performance 128, sampled_library 32, sound_module 2.

C1b re-renders: CP80_16_44100 max|d| 0.0, CTK_230_SoundFont max|d| 0.0, CTK_533_Piano1_00_Casio max|d| 0.0, Casio_PX_860_Concert_Grand_Piano max|d| 0.0, Casio_CA_110_Piano_FM max|d| 0.0, JV1080_Nice_Piano_m max|d| 0.0, U20PIANO max|d| 0.0, YAMAHA_PSR400_500 max|d| 0.0, Yamaha_PSRF50 max|d| 0.0, Yamaha_S90_ES_Natural_Grand max|d| 0.0; Otey reproduction of out_full at 3.0 s: max|d| 0.0.

## 2. Table 7, protocol v3: one detector per format (own-fold scoring)

### 2.1 Isolated notes: I-only detector

Training (T2): 668 CCMusic notes (6 brand groups) vs 678 SoundFont renders (velocity copies removed) + 120 Otey notes. Library notes (999 files) never trained (T2), scored by the model of their library's fold (own-fold). T1: library notes added to training labelled synthetic, scored out-of-fold. Cells: AUC [95% CI]; det = share of the synthetic notes with P > 0.5; P = mean P(synthetic). "real-side only" = one synthetic group: the interval resamples real groups only and does not generalise to other engines.

| CCMusic notes vs | notes/groups | all47 T2 | med16 T2 | all47 T1 | med16 T1 |
|---|---|---|---|---|---|
| SoundFont single notes | 678/10 | 0.937 [0.870, 0.983]; det 84.1%; P 0.82 | 0.877 [0.773, 0.958]; det 76.8%; P 0.73 | 0.867 [0.787, 0.945]; det 90.4%; P 0.84 | 0.799 [0.700, 0.897]; det 85.0%; P 0.76 |
| Otey single notes | 120/1 | 0.821 (real-side only [0.703, 0.926]); det 34.2%; P 0.42 | 0.675 (real-side only [0.574, 0.779]); det 23.3%; P 0.32 | 0.760 (real-side only [0.657, 0.867]); det 74.2%; P 0.64 | 0.657 (real-side only [0.548, 0.760]); det 58.3%; P 0.57 |
| algorithmic (both) | 798/11 | 0.920 [0.840, 0.978]; det 76.6%; P 0.76 | 0.846 [0.731, 0.944]; det 68.8%; P 0.67 | 0.851 [0.758, 0.936]; det 88.0%; P 0.81 | 0.778 [0.666, 0.883]; det 81.0%; P 0.74 |
| library notes, 4 libraries, mf layer | 224/4 | 0.621 [0.482, 0.753]; det 15.2%; P 0.18 | 0.528 [0.419, 0.648]; det 17.0%; P 0.21 | 0.714 [0.613, 0.813]; det 62.5%; P 0.59 | 0.633 [0.534, 0.731]; det 59.8%; P 0.57 |
| library notes, 4 libraries, all layers | 999/4 | 0.645 [0.484, 0.783]; det 19.3%; P 0.22 | 0.592 [0.440, 0.711]; det 26.8%; P 0.29 | 0.691 [0.607, 0.781]; det 58.0%; P 0.57 | 0.662 [0.567, 0.751]; det 63.1%; P 0.60 |
| Salamander, mf layer | 30/1 | 0.699 (real-side only [0.575, 0.819]); det 30.0%; P 0.28 | 0.615 (real-side only [0.525, 0.709]); det 23.3%; P 0.31 | 0.653 (real-side only [0.571, 0.741]); det 46.7%; P 0.51 | 0.569 (real-side only [0.495, 0.639]); det 50.0%; P 0.51 |
| Salamander, all layers | 480/1 | 0.714 (real-side only [0.590, 0.834]); det 25.6%; P 0.29 | 0.657 (real-side only [0.567, 0.748]); det 36.5%; P 0.37 | 0.692 (real-side only [0.608, 0.781]); det 56.7%; P 0.56 | 0.671 (real-side only [0.581, 0.753]); det 63.3%; P 0.61 |
| Iowa, mf layer | 86/1 | 0.608 (real-side only [0.468, 0.746]); det 9.3%; P 0.12 | 0.461 (real-side only [0.371, 0.556]); det 12.8%; P 0.16 | 0.757 (real-side only [0.673, 0.847]); det 65.1%; P 0.65 | 0.667 (real-side only [0.594, 0.734]); det 66.3%; P 0.62 |
| Iowa, all layers | 261/1 | 0.576 (real-side only [0.441, 0.708]); det 10.0%; P 0.13 | 0.461 (real-side only [0.367, 0.559]); det 10.0%; P 0.14 | 0.704 (real-side only [0.623, 0.791]); det 59.8%; P 0.59 | 0.643 (real-side only [0.559, 0.720]); det 62.5%; P 0.59 |
| K18, mf layer | 30/1 | 0.555 (real-side only [0.440, 0.667]); det 16.7%; P 0.18 | 0.559 (real-side only [0.464, 0.657]); det 23.3%; P 0.26 | 0.699 (real-side only [0.612, 0.792]); det 63.3%; P 0.59 | 0.709 (real-side only [0.600, 0.813]); det 70.0%; P 0.66 |
| K18, all layers | 180/1 | 0.568 (real-side only [0.453, 0.682]); det 17.8%; P 0.20 | 0.627 (real-side only [0.530, 0.727]); det 30.0%; P 0.31 | 0.671 (real-side only [0.591, 0.756]); det 55.6%; P 0.55 | 0.697 (real-side only [0.599, 0.791]); det 67.8%; P 0.64 |
| FreePats, mf layer | 78/1 | 0.629 (real-side only [0.506, 0.750]); det 15.4%; P 0.21 | 0.557 (real-side only [0.461, 0.659]); det 16.7%; P 0.22 | 0.694 (real-side only [0.604, 0.790]); det 65.4%; P 0.57 | 0.590 (real-side only [0.494, 0.682]); det 52.6%; P 0.51 |
| FreePats, all layers | 78/1 | 0.629 (real-side only [0.506, 0.750]); det 15.4%; P 0.21 | 0.557 (real-side only [0.461, 0.659]); det 16.7%; P 0.22 | 0.694 (real-side only [0.604, 0.790]); det 65.4%; P 0.57 | 0.590 (real-side only [0.494, 0.682]); det 52.6%; P 0.51 |

CCMusic false alarms (P > 0.5): all47 T2 11.5%, med16 T2 17.7%, all47 T1 29.0%, med16 T1 37.0%. Sensitivity: SoundFont incl. the 84 velocity copies (all47 T2): 0.942 [0.878, 0.984]; det 86.2%. Library notes with the v2 rule (mean of 5 fold models), mf layer: all47 0.605 [0.449, 0.773]; det 12.5%, med16 0.508 [0.356, 0.697]; det 10.7%.

### 2.2 Performances: performance-only detector

Training (T2): 2471 real performances (cleaned; grouped by v2 group + YouTube video id) vs 120 SoundFont / Otey performance renders (each with its engine's group). Digital pianos, M1, SC-55 and GP-3 never trained (T2), scored own-fold (the model of their group's fold); T1: trained, labelled synthetic. Digital pianos grouped by brand. Last column: the v2 scoring rule (mean of the 5 fold models) for comparison.

| id | real performances vs | files/groups | all47 T2 | med16 T2 | all47 T1 | med16 T1 | all47 T2, v2 rule (mean of 5) |
|---|---|---|---|---|---|---|---|
| P1 | algo_performance | 120/11 | 0.943 [0.907, 0.978]; det 36.7%; P 0.41 | 0.723 [0.622, 0.839]; det 9.2%; P 0.13 | 0.954 [0.926, 0.977]; det 58.3%; P 0.61 | 0.758 [0.679, 0.835]; det 28.3%; P 0.32 | 0.943 [0.907, 0.978]; det 36.7% |
| P1a | ... vs SoundFont performance renders (sf2_perf) | 100/10 | 0.947 [0.899, 0.983]; det 44.0%; P 0.48 | 0.756 [0.643, 0.858]; det 11.0%; P 0.16 | 0.949 [0.918, 0.975]; det 55.0%; P 0.59 | 0.751 [0.662, 0.840]; det 27.0%; P 0.31 | 0.947 [0.899, 0.983]; det 44.0% |
| P1b | ... vs Otey performance renders (otey_perf) | 20/1 | 0.923 (real-side only [0.915, 0.931]); det 0.0%; P 0.05 | 0.558 (real-side only [0.545, 0.572]); det 0.0%; P 0.01 | 0.979 (real-side only [0.976, 0.983]); det 75.0%; P 0.73 | 0.796 (real-side only [0.787, 0.805]); det 35.0%; P 0.37 | 0.923 (real-side only [0.915, 0.931]); det 0.0% |
| P2 | sampled digital pianos (old demos + digital_v2) | 128/7 | 0.871 [0.843, 0.891]; det 7.0%; P 0.10 | 0.562 [0.525, 0.622]; det 1.6%; P 0.03 | 0.953 [0.939, 0.965]; det 60.2%; P 0.58 | 0.844 [0.798, 0.870]; det 32.0%; P 0.39 | 0.901 [0.864, 0.926]; det 7.8% |
| P2a | ... old demos only (SC-55 module excluded) | 9/4 | 0.916 [0.887, 0.948]; det 0.0%; P 0.07 | 0.621 [0.468, 0.858]; det 0.0%; P 0.02 | 0.955 [0.917, 0.978]; det 44.4%; P 0.50 | 0.860 [0.794, 0.955]; det 55.6%; P 0.48 | 0.936 [0.900, 0.962]; det 11.1% |
| P2d | ... digital_v2 only | 119/7 | 0.868 [0.839, 0.888]; det 7.6%; P 0.10 | 0.558 [0.525, 0.611]; det 1.7%; P 0.03 | 0.953 [0.939, 0.966]; det 61.3%; P 0.59 | 0.843 [0.793, 0.872]; det 30.3%; P 0.38 | 0.898 [0.862, 0.923]; det 7.6% |
| P2d-line | ... digital_v2, line capture | 12/6 | 0.913 [0.847, 0.954]; det 8.3%; P 0.13 | 0.594 [0.474, 0.718]; det 0.0%; P 0.03 | 0.979 [0.965, 0.991]; det 66.7%; P 0.72 | 0.895 [0.860, 0.934]; det 33.3%; P 0.40 | 0.953 [0.912, 0.979]; det 8.3% |
| P2d-mic | ... digital_v2, microphone capture | 2/2 | 0.964 [0.947, 0.980]; det 0.0%; P 0.05 | 0.495 [0.135, 0.857]; det 0.0%; P 0.01 | 0.986 [0.974, 0.996]; det 100.0%; P 0.78 | 0.845 [0.760, 0.926]; det 0.0%; P 0.28 | 0.981 [0.965, 0.995]; det 0.0% |
| P2d-unk | ... digital_v2, capture unknown | 105/7 | 0.861 [0.829, 0.886]; det 7.6%; P 0.10 | 0.555 [0.524, 0.600]; det 1.9%; P 0.03 | 0.949 [0.935, 0.964]; det 60.0%; P 0.57 | 0.837 [0.781, 0.868]; det 30.5%; P 0.38 | 0.890 [0.851, 0.917]; det 7.6% |
| P2d-lt | ... digital_v2, long_tails flag | 10/6 | 0.864 [0.843, 0.905]; det 10.0%; P 0.10 | 0.507 [0.231, 0.652]; det 0.0%; P 0.01 | 0.976 [0.966, 0.993]; det 70.0%; P 0.69 | 0.861 [0.742, 0.948]; det 50.0%; P 0.42 | 0.886 [0.850, 0.939]; det 0.0% |
| P2d-nolt | ... digital_v2, no long_tails flag | 109/7 | 0.868 [0.837, 0.892]; det 7.3%; P 0.10 | 0.563 [0.527, 0.622]; det 1.8%; P 0.03 | 0.951 [0.936, 0.965]; det 60.6%; P 0.58 | 0.841 [0.796, 0.868]; det 28.4%; P 0.37 | 0.899 [0.859, 0.929]; det 8.3% |
| P3 | modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 21/1 | 0.854 (real-side only [0.845, 0.863]); det 9.5%; P 0.10 | 0.583 (real-side only [0.570, 0.596]); det 0.0%; P 0.01 | 0.870 (real-side only [0.864, 0.876]); det 57.1%; P 0.52 | 0.766 (real-side only [0.756, 0.776]); det 19.0%; P 0.29 | 0.873 (real-side only [0.865, 0.881]); det 4.8% |
| P3d | ... digital_v2 modelled only | 19/1 | 0.863 (real-side only [0.854, 0.872]); det 10.5%; P 0.10 | 0.592 (real-side only [0.580, 0.606]); det 0.0%; P 0.01 | 0.891 (real-side only [0.885, 0.897]); det 57.9%; P 0.52 | 0.788 (real-side only [0.779, 0.798]); det 21.1%; P 0.32 | 0.878 (real-side only [0.870, 0.885]); det 5.3% |
| P4 | Korg M1 | 30/1 | 0.928 (real-side only [0.922, 0.934]); det 23.3%; P 0.28 | 0.874 (real-side only [0.866, 0.882]); det 16.7%; P 0.22 | 0.874 (real-side only [0.866, 0.882]); det 43.3%; P 0.43 | 0.875 (real-side only [0.868, 0.882]); det 53.3%; P 0.51 | 0.920 (real-side only [0.913, 0.926]); det 20.0% |
| P5 | digital_other (Roland GP-3, hybrid / unclear) | 1/1 | 0.991 (real-side only [0.987, 0.995]); det 0.0%; P 0.22 | 0.860 (real-side only [0.847, 0.874]); det 0.0%; P 0.03 | 0.987 (real-side only [0.983, 0.991]); det 100.0%; P 0.78 | 0.825 (real-side only [0.810, 0.839]); det 0.0%; P 0.18 | 0.972 (real-side only [0.965, 0.978]); det 0.0% |
| P6 | Roland SC-55 sound module (not a digital piano) | 2/1 | 1.000 (real-side only [0.999, 1.000]); det 100.0%; P 0.85 | 0.783 (real-side only [0.768, 0.799]); det 0.0%; P 0.01 | 0.992 (real-side only [0.989, 0.994]); det 100.0%; P 0.86 | 0.915 (real-side only [0.908, 0.923]); det 50.0%; P 0.55 | 0.999 (real-side only [0.999, 1.000]); det 100.0% |

Real-performance false alarms (P > 0.5): all47 T2 0.2%, all47 T1 3.4%, med16 T2 1.7%, med16 T1 5.9%.

Feature-group ablation (performance-only, T2, own-fold):

| features | P2 sampled digital | P1 algorithmic perf | P3 modelled |
|---|---|---|---|
| all47 | 0.871 [0.843, 0.891]; det 7.0% | 0.943 [0.907, 0.978]; det 36.7% | 0.854 (real-side only [0.845, 0.863]); det 9.5% |
| only_A_med | 0.562 [0.525, 0.622]; det 1.6% | 0.723 [0.622, 0.839]; det 9.2% | 0.583 (real-side only [0.570, 0.596]); det 0.0% |
| only_A_iqr | 0.678 [0.609, 0.730]; det 3.9% | 0.819 [0.729, 0.906]; det 30.0% | 0.710 (real-side only [0.698, 0.721]); det 9.5% |
| only_B | 0.680 [0.593, 0.754]; det 7.8% | 0.732 [0.621, 0.839]; det 20.8% | 0.667 (real-side only [0.657, 0.677]); det 9.5% |
| only_C | 0.791 [0.766, 0.813]; det 16.4% | 0.738 [0.649, 0.813]; det 23.3% | 0.656 (real-side only [0.646, 0.667]); det 4.8% |

## 3. Controls for the isolated-note claim (I-only XGBoost, T2, own-fold scoring)

I-only XGBoost (CCMusic notes vs SoundFont + Otey single notes; library notes zero-shot, own-fold scoring), all47 and med16, T2, re-extracted features under:   orig    : the stored renders / recordings   C1a     : every isolated note cut to 1.2 s after its onset + 0.1-s raised-cosine fade (before any             SoundFont note-off at 1.4 s)   C1b     : SoundFont notes re-rendered with the key held 5.0 s and Otey notes rendered for 5.0 s             (v3_render.py), every source cut at 4.0 s after onset + 0.1-s fade; real notes (CCMusic,             library notes) kept only if they still sound at 4.0 s (frame level >= -60 dB re peak)   C3      : every note convolved with a measured room IR (MIT IR Survey, one IR drawn per file) + white             noise at 40 dB SNR re the note RMS; real notes too   C1a+C3, C1b+C3 : C3 applied to the full note, then the cut 

Room IRs: 270 measured IRs of the MIT McDermott IR Survey (`audio/_ir/mit_survey/Audio/`, Audio.zip from mcdermottlab.mit.edu/Reverb/IR_Survey.html; the archive holds 270 wav files, the survey page lists 271), 32 kHz, resampled to each file's rate; the convolved note is rescaled to the original peak; RNG per file = crc32(relative path) (IR choice and noise). C1b real-note filter: CCMusic 380 of 668 and library notes 874 of 999 still sound at 4.0 s (Iowa 261/261, K18 120/180, Salamander 443/480, FreePats 50/78). Synthetic notes below -60 dB at 4.0 s (sample ended or decayed; kept): SoundFont 304 of 678, Otey 43 of 120. SoundFont = 678 renders after removing velocity copies.

**Control table B, all 47 features** (AUC vs CCMusic notes [95% CI]; det = share detected at P > 0.5):

| CCMusic notes vs | original | C1a 1.2 s | C1b 4.0 s | C3 room+noise | C1a+C3 | C1b+C3 |
|---|---|---|---|---|---|---|
| SoundFont | 0.937 [0.870, 0.983]; det 84.1% (n=678) | 0.819 [0.731, 0.900]; det 65.9% (n=678) | 0.840 [0.770, 0.897]; det 73.0% (n=674) | 0.923 [0.837, 0.978]; det 88.8% (n=677) | 0.861 [0.783, 0.933]; det 81.3% (n=675) | 0.940 [0.907, 0.968]; det 92.6% (n=678) |
| SoundFont, unaltered sample | 0.931 [0.862, 0.981]; det 83.3% (n=168) | 0.818 [0.734, 0.897]; det 67.3% (n=171) | 0.826 [0.718, 0.902]; det 65.3% (n=150) | 0.912 [0.827, 0.975]; det 85.6% (n=167) | 0.850 [0.765, 0.927]; det 79.4% (n=170) | 0.929 [0.888, 0.968]; det 90.2% (n=153) |
| Otey | 0.821 (real-side only [0.703, 0.926]); det 34.2% (n=120) | 0.645 (real-side only [0.575, 0.726]); det 40.7% (n=118) | 0.780 (real-side only [0.702, 0.810]); det 63.3% (n=120) | 0.699 (real-side only [0.551, 0.825]); det 34.5% (n=119) | 0.759 (real-side only [0.669, 0.848]); det 64.2% (n=120) | 0.831 (real-side only [0.773, 0.881]); det 75.2% (n=117) |
| library notes, all layers | 0.645 [0.484, 0.783]; det 19.3% (n=999) | 0.710 [0.517, 0.833]; det 48.4% (n=999) | 0.683 [0.529, 0.798]; det 45.8% (n=874) | 0.584 [0.439, 0.714]; det 22.8% (n=999) | 0.663 [0.486, 0.793]; det 47.6% (n=999) | 0.670 [0.561, 0.754]; det 44.9% (n=874) |
| library notes, mf layer | 0.621 [0.482, 0.753]; det 15.2% (n=224) | 0.658 [0.557, 0.769]; det 35.7% (n=224) | 0.621 [0.524, 0.770]; det 34.4% (n=183) | 0.585 [0.440, 0.716]; det 21.9% (n=224) | 0.624 [0.501, 0.745]; det 41.1% (n=224) | 0.642 [0.562, 0.747]; det 39.3% (n=183) |
| CCMusic false alarms | 11.5% (n=668) | 21.7% (n=667) | 22.9% (n=380) | 20.1% (n=667) | 25.1% (n=666) | 20.8% (n=380) |

**Control table B, 16 Group-A medians** (AUC vs CCMusic notes [95% CI]; det = share detected at P > 0.5):

| CCMusic notes vs | original | C1a 1.2 s | C1b 4.0 s | C3 room+noise | C1a+C3 | C1b+C3 |
|---|---|---|---|---|---|---|
| SoundFont | 0.877 [0.773, 0.958]; det 76.8% (n=678) | 0.735 [0.642, 0.825]; det 53.8% (n=678) | 0.731 [0.644, 0.807]; det 62.0% (n=674) | 0.883 [0.768, 0.959]; det 86.3% (n=677) | 0.768 [0.688, 0.845]; det 73.0% (n=675) | 0.914 [0.888, 0.942]; det 89.1% (n=678) |
| SoundFont, unaltered sample | 0.867 [0.761, 0.952]; det 75.0% (n=168) | 0.758 [0.670, 0.844]; det 56.7% (n=171) | 0.730 [0.583, 0.832]; det 60.0% (n=150) | 0.864 [0.752, 0.956]; det 82.0% (n=167) | 0.736 [0.639, 0.832]; det 66.5% (n=170) | 0.910 [0.868, 0.951]; det 88.2% (n=153) |
| Otey | 0.675 (real-side only [0.574, 0.779]); det 23.3% (n=120) | 0.573 (real-side only [0.537, 0.608]); det 40.7% (n=118) | 0.670 (real-side only [0.631, 0.698]); det 52.5% (n=120) | 0.520 (real-side only [0.401, 0.622]); det 22.7% (n=119) | 0.654 (real-side only [0.576, 0.729]); det 52.5% (n=120) | 0.820 (real-side only [0.809, 0.830]); det 76.9% (n=117) |
| library notes, all layers | 0.592 [0.440, 0.711]; det 26.8% (n=999) | 0.645 [0.503, 0.747]; det 43.0% (n=999) | 0.624 [0.436, 0.756]; det 49.0% (n=874) | 0.559 [0.445, 0.666]; det 28.9% (n=999) | 0.644 [0.486, 0.761]; det 52.7% (n=999) | 0.619 [0.543, 0.682]; det 45.1% (n=874) |
| library notes, mf layer | 0.528 [0.419, 0.648]; det 17.0% (n=224) | 0.590 [0.505, 0.685]; det 33.5% (n=224) | 0.542 [0.400, 0.752]; det 36.6% (n=183) | 0.566 [0.429, 0.685]; det 27.2% (n=224) | 0.591 [0.497, 0.691]; det 44.2% (n=224) | 0.590 [0.513, 0.721]; det 39.3% (n=183) |
| CCMusic false alarms | 17.7% (n=668) | 23.5% (n=667) | 29.5% (n=380) | 21.9% (n=667) | 29.4% (n=666) | 27.6% (n=380) |

**Detection rate per source and condition** (all 47, share with P > 0.5; CCMusic = false alarms):

| source | original | C1a 1.2 s | C1b 4.0 s | C3 room+noise | C1a+C3 | C1b+C3 |
|---|---|---|---|---|---|---|
| CCMusic | 11.5% (668) | 21.7% (667) | 22.9% (380) | 20.1% (667) | 25.1% (666) | 20.8% (380) |
| CP80_16_44100 | 94.3% (53) | 90.6% (53) | 84.9% (53) | 92.5% (53) | 88.7% (53) | 98.1% (53) |
| CTK_230_SoundFont | 100.0% (78) | 96.2% (78) | 100.0% (78) | 94.9% (78) | 87.2% (78) | 93.6% (78) |
| CTK_533_Piano1_00_Casio | 100.0% (39) | 100.0% (39) | 97.4% (39) | 87.2% (39) | 89.7% (39) | 94.9% (39) |
| Casio_CA_110_Piano_FM | 80.5% (77) | 76.6% (77) | 45.5% (77) | 93.5% (77) | 92.1% (76) | 100.0% (77) |
| Casio_PX_860_Concert_Grand_Piano | 76.9% (78) | 62.8% (78) | 79.5% (78) | 70.5% (78) | 62.8% (78) | 83.3% (78) |
| JV1080_Nice_Piano_m | 92.3% (78) | 55.1% (78) | 73.1% (78) | 79.5% (78) | 62.8% (78) | 92.3% (78) |
| U20PIANO | 93.2% (44) | 68.2% (44) | 88.6% (44) | 93.2% (44) | 93.2% (44) | 97.7% (44) |
| YAMAHA_PSR400_500 | 86.7% (75) | 76.0% (75) | 72.0% (75) | 98.7% (75) | 91.9% (74) | 96.0% (75) |
| Yamaha_PSRF50 | 42.3% (78) | 11.5% (78) | 38.5% (78) | 92.3% (78) | 87.2% (78) | 93.6% (78) |
| Yamaha_S90_ES_Natural_Grand | 89.7% (78) | 48.7% (78) | 73.0% (74) | 88.3% (77) | 70.1% (77) | 82.1% (78) |
| Otey | 34.2% (120) | 40.7% (118) | 63.3% (120) | 34.5% (119) | 64.2% (120) | 75.2% (117) |
| Iowa | 10.0% (261) | 28.7% (261) | 21.5% (261) | 12.3% (261) | 23.0% (261) | 29.9% (261) |
| K18 | 17.8% (180) | 21.7% (180) | 31.7% (120) | 32.2% (180) | 35.0% (180) | 36.7% (120) |
| Salamander | 25.6% (480) | 72.1% (480) | 63.9% (443) | 24.0% (480) | 65.4% (480) | 54.4% (443) |
| FreePats | 15.4% (78) | 30.8% (78) | 46.0% (50) | 29.5% (78) | 50.0% (78) | 58.0% (50) |

**C2 -- SoundFont notes by synthesis method** (from the SF2 data: sample root key, active loop mode (generator 54) and loop start; "transposed" = |key - root| >= 0.5 semitone, "looped" = the loop start is reached within the analysed window: 1.8 s original / 1.3 s C1a / 4.1 s C1b). All 47, I-only T2; AUC vs all CCMusic notes; each cell: AUC; det (n notes).

| category | original | C1a 1.2 s | C1b 4.0 s | C3 room+noise | C1a+C3 | C1b+C3 |
|---|---|---|---|---|---|---|
| unaltered sample | 0.931; det 83.3% (168, 9 SF) | 0.818; det 67.3% (171, 9 SF) | 0.826; det 65.3% (150, 8 SF) | 0.912; det 85.6% (167, 9 SF) | 0.850; det 79.4% (170, 9 SF) | 0.929; det 90.2% (153, 8 SF) |
| transposed only | 0.927; det 80.4% (378, 8 SF) | 0.790; det 58.2% (390, 9 SF) | 0.818; det 69.0% (335, 7 SF) | 0.918; det 87.6% (378, 8 SF) | 0.858; det 79.7% (389, 9 SF) | 0.936; det 92.3% (336, 7 SF) |
| looped only | 0.977; det 100.0% (18, 3 SF) | 0.951; det 100.0% (15, 2 SF) | 0.858; det 78.8% (33, 3 SF) | 0.959; det 100.0% (18, 3 SF) | 0.924; det 93.3% (15, 2 SF) | 0.939; det 93.9% (33, 3 SF) |
| transposed + looped | 0.974; det 94.7% (114, 2 SF) | 0.912; det 88.2% (102, 2 SF) | 0.895; det 87.8% (156, 3 SF) | 0.950; det 95.6% (114, 2 SF) | 0.883; det 89.1% (101, 2 SF) | 0.960; det 95.5% (156, 3 SF) |

Per SoundFont (SF2 data for the rendered notes): CP80_16_44100: |transp.| median 3.0 st, loop mode on 0%, loop start median 0.00 s, sample end median 10.55 s; CTK_230_SoundFont: |transp.| median 5.5 st, loop mode on 100%, loop start median 0.28 s, sample end median 0.28 s; CTK_533_Piano1_00_Casio: |transp.| median 0.0 st, loop mode on 0%, loop start median 0.00 s, sample end median 5.04 s; Casio_CA_110_Piano_FM: |transp.| median 6.0 st, loop mode on 0%, loop start median 0.00 s, sample end median 3.06 s; Casio_PX_860_Concert_Grand_Piano: |transp.| median 1.0 st, loop mode on 0%, loop start median 0.00 s, sample end median 28.40 s; JV1080_Nice_Piano_m: |transp.| median 1.0 st, loop mode on 100%, loop start median 5.32 s, sample end median 5.73 s; U20PIANO: |transp.| median 4.5 st, loop mode on 0%, loop start median 0.00 s, sample end median 1.47 s; YAMAHA_PSR400_500: |transp.| median 2.0 st, loop mode on 100%, loop start median 1.02 s, sample end median 1.15 s; Yamaha_PSRF50: |transp.| median 3.0 st, loop mode on 0%, loop start median 0.00 s, sample end median 4.81 s; Yamaha_S90_ES_Natural_Grand: |transp.| median 1.0 st, loop mode on 0%, loop start median 3.19 s, sample end median 3.77 s.

**C4 -- phaseogram drift irregularity after C3** (the figure analysis' primary 190 notes: MIDI 48-76, pitch check passed, 50 per class stratified by source, all 40 Otey; `figures/phaseo_lib.py` phaseogram + drift_summary; samples per cycle):

| class | notes | median irregularity (original) | median irregularity (C3) | P(class > real), original | P(class > real), C3 |
|---|---|---|---|---|---|
| real | 50 | 1.355 | 1.868 |  |  |
| samplelib | 50 | 0.769 | 0.632 | 0.32 (p=0.0022) | 0.28 (p=0.00013) |
| soundfont | 50 | 0.288 | 0.383 | 0.09 (p=1.1e-12) | 0.23 (p=3.5e-06) |
| otey | 40 | 0.054 | 0.183 | 0.05 (p=1.9e-13) | 0.12 (p=7.4e-10) |

Recomputed original irregularity vs `drift_notes_all.csv`: max |difference| = 4.98e-07.

**C5 -- loop / exact-repetition index on the C1b notes** (window 0.5-4.0 s after onset; max over 50-ms reference segments every 0.1 s of the normalised cross-correlation with the signal at lags 20 ms-1.5 s; each segment mean-removed and normalised, so the slow envelope is removed):

| source | notes | median r_max | 90th pct | r > 0.999 | r > 0.9999 | median lag of r > 0.999 (s) | SF2: loop reached by 4.1 s / sample ended | SF2 loop length median (s) |
|---|---|---|---|---|---|---|---|---|
| CCMusic (sounding at 4 s) | 380 | 0.9849 | 0.9970 | 3% | 0% | 0.023 |  | - |
| library Iowa (sounding at 4 s) | 261 | 0.9894 | 0.9990 | 10% | 0% | 0.023 |  | - |
| library K18 (sounding at 4 s) | 120 | 0.9908 | 0.9996 | 20% | 2% | 0.024 |  | - |
| library Salamander (sounding at 4 s) | 443 | 0.9886 | 0.9977 | 0% | 0% | - |  | - |
| library FreePats (sounding at 4 s) | 50 | 0.9841 | 0.9974 | 2% | 0% | 0.021 |  | - |
| Otey | 120 | 0.9963 | 0.9998 | 32% | 6% | 0.025 |  | - |
| SF CP80_16_44100 | 53 | 0.9997 | 0.9999 | 77% | 11% | 0.109 | 0% / ended 9% | - |
| SF CTK_230_SoundFont | 78 | 1.0000 | 1.0000 | 100% | 100% | 0.467 | 100% / ended 100% | 0.004 |
| SF CTK_533_Piano1_00_Casio | 39 | 0.9999 | 1.0000 | 100% | 46% | 0.333 | 0% / ended 15% | - |
| SF Casio_CA_110_Piano_FM | 77 | 1.0000 | 1.0000 | 88% | 84% | 0.050 | 0% / ended 65% | - |
| SF Casio_PX_860_Concert_Grand_Piano | 78 | 0.9960 | 0.9992 | 17% | 0% | 0.026 | 0% / ended 0% | - |
| SF JV1080_Nice_Piano_m | 78 | 0.9961 | 0.9999 | 19% | 12% | 1.295 | 46% / ended 38% | 0.358 |
| SF U20PIANO | 44 | 0.9748 | 0.9884 | 0% | 0% | - | 0% / ended 100% | - |
| SF YAMAHA_PSR400_500 | 75 | 1.0000 | 1.0000 | 88% | 84% | 0.942 | 100% / ended 96% | 0.013 |
| SF Yamaha_PSRF50 | 78 | 0.9995 | 0.9996 | 96% | 0% | 0.070 | 0% / ended 31% | - |
| SF Yamaha_S90_ES_Natural_Grand | 78 | 0.9889 | 0.9995 | 13% | 0% | 0.169 | 0% / ended 65% | - |
