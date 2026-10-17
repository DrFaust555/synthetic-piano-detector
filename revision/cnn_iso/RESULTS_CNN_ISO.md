# Phaseogram CNN and fusion, FORMAT-MATCHED (isolated notes only)

Generated 2026-10-01 16:03 by one command, from the project root:

```
python Stattia/revision/cnn_iso/run_all.py
```

(builds the phaseogram cache, trains every model that is not yet in `cnn_iso/jobs/` -- resumable -- and evaluates. The matched-XGBoost control additionally needs xgboost 3.2.0 importable, as protocol v2.)

Environment: Python 3.10.7, torch 2.0.1+cpu (CPU only), numpy 1.26.4, pandas 1.5.2. Scripts: `Stattia/revision/cnn_iso/{run_all, build_phaseograms, train_cnn_iso, evaluate_cnn_iso, cnn_iso_lib}.py`. Outputs: `RESULTS_CNN_ISO.md`, `oof_scores_cnn_iso.csv` (per-file scores of every model/condition), `comparisons_cnn_iso.csv`, `stimuli_cnn_iso_scores.csv`, `cache/` (phaseograms + `phaseo_meta.csv`), `jobs/` (one .npz per trained model), `logs/`.

## Reading (all 69 fits complete)

**1. Algorithmic notes, format-matched.**
- The paper reported 0.981 (CNN) and 0.985 (fusion) for real vs algorithmic, against a real class made mostly of performances. With CCMusic isolated notes as the real class, the CNN and the fusion score about the same as XGBoost on the same notes.

  T2 AUCs (T1 within 0.05: CNN 0.893, fusion-47 0.913, fusion-16 0.894 for SoundFont + Otey):

  | | CNN | Fusion-47 | Fusion-16 | XGB, same notes |
  |---|---|---|---|---|
  | SoundFont + Otey | 0.892 | 0.923 | 0.907 | 0.908 |
  | SoundFont renders | 0.910 | 0.940 | 0.924 | 0.929 |
  | Otey notes | 0.776 | 0.817 | 0.797 | 0.779 |

- CCMusic false alarms:

  | | CNN | Fusion-47 | XGB |
  |---|---|---|---|
  | T2 | 18% | 16% | 15% |
  | T1 | 34% | 26% | -- |

  The paper's observation that the networks flag almost every isolated note was a property of training against performances. It disappears when the real class is isolated notes too.

**2. Library notes, format-matched.**
- In the paper the CNN separated the libraries at AUC 0.984 and the fusion at 0.986, with 81% and 78% of library takes detected. Here, pooled over the four libraries (mf layer):

  | | T1, paper protocol | T2, never seen |
  |---|---|---|
  | CNN | 0.597 [0.382, 0.848], 51% detected at 34% CCMusic false alarms | 0.713 [0.548, 0.877], 36% at 18% |
  | Fusion-47 | 0.558 [0.305, 0.848], 39% at 26% | 0.619 [0.427, 0.819], 18% at 16% |
  | Fusion-16 | 0.483 | 0.609 |
  | XGB, same notes | 0.424 | 0.551 |

- With the detector's own f0 restricted to MIDI 48-76, the CNN gives 0.761 [0.562, 0.936] (T1) and 0.622 [0.382, 0.852] (T2). Only two CNN intervals exclude 0.5: constrained-f0 T2 (lower bound 0.548) and MIDI 48-76 T1 (0.562). Detection rates at P > 0.5 rise and fall with the CCMusic false-alarm rate.
- So the near-perfect library separation of the paper does not survive format matching. **The "recording-format shortcut" explanation holds for the bulk of the paper's number.**

**3. A residual remains, and it is library-specific.**
- Under every model and condition, the CNN separates Salamander, K18 and FreePats notes from CCMusic notes at AUC of about 0.7-0.94. Iowa stays at or below chance: 0.56 (T2), 0.37 (T1), 0.53 (48-76, T1), 0.44 (48-76, T2). Iowa is also the least detected library in each of the 8 individual CNN models (s.3.5).
- Training on the other three libraries as synthetic (T1) does not consistently help on the held-out one. It lowers the pooled AUC in the constrained-f0 variant (0.713 to 0.597) and raises it in the MIDI 48-76 variant (0.622 to 0.761); in both, Iowa stays at or below chance.
- That is not what a learned "sampled-note" cue would do. It fits differences between particular recording chains and the CCMusic recordings.
- The residual is not explained by:
  - the f0 choice (constrained vs detector f0, MIDI 48-76, |f0 - nominal| < 0.5 semitone);
  - zero padding;
  - where the 48-cycle window starts relative to the attack (s.2).
- The paper should therefore not claim either that the CNN detects sample libraries, or that its library detection is purely a format effect. Format matched, the CNN's library separation is weak, library-dependent and unstable across training conditions.

**4. Listening clips.**
- The format-matched CNN flags almost every clip: 19/20 recorded and 10/10 algorithmic with the LOGO models; 18/20 and 10/10 with the fold models. Its recorded-vs-algorithmic AUC is therefore uninterpretable: 0.140 with LOGO, 0.905 with the fold models, depending only on which saturated model scores which class.
- The same models flag only 2/7 (LOGO) or 1/7 (folds) of the unprocessed Iowa source notes. The clip processing, not the instrument, drives the CNN's verdict on these stimuli. The clip's onset trim and fade-in move the window about 40 ms later on the Iowa notes, and the phaseogram changes accordingly (s.5).
- The fusion models do not saturate but do not separate the classes either:

  | Model (LOGO) | Recorded detected | Algorithmic detected | AUC |
  |---|---|---|---|
  | Fusion-47 | 7/20 | 7/10 | 0.600 [0.415, 0.820] |
  | Fusion-16 | -- | -- | 0.655 |
  | XGB (RESULTS_v2 s.4) | -- | -- | 0.665 |
  | XGB on isolated notes only | -- | -- | 0.605 |

**Deviations from the paper's run (all deliberate, all class-independent).**
- One phaseogram per file, the primary segment, instead of up to 15 notes per file.
- Each fit is seeded `torch.manual_seed(0)` at its start, instead of seeding once per `run()`.
- CPU, 2 torch threads per fit; the paper's run was probably on a GPU.
- T1 trains on the mf-equivalent library layer only.
- The fusion's 47 features keep the detector's unconstrained f0; only the phaseograms use the constrained f0.
- The LOGO clip models are trained on the isolated-note T2 population (RESULTS_v2 s.4 also included performances).

## 0. Key numbers (constrained-f0 phaseograms, all registers)

Cells: AUC [95% CI, stratified grouped bootstrap]; det = share of synthetic files with P(synthetic) > 0.5. XGB columns are copied from `analysis/out/` (RESULTS_v2 s.2-3): T1/T2 = corpus-level training (performances included), I-only = trained on isolated notes only (CCMusic vs SoundFont + Otey, same folds; library notes by the mean of the 5 fold models -- the XGBoost counterpart of this study's T2; shown from the matched control of s.3.4, which reproduces the RESULTS_v2 I-only AUCs exactly and adds the "algorithmic" row). RESULTS_v2 version read: generated 2026-10-01 14:00. Its T1 detector is trained on the whole v2 corpus, including any data_v2 rows present (e.g. the digital_v2 sampled-digital performances labelled synthetic), so its T1 numbers move when data_v2 grows; its T2 and I-only numbers do not involve those rows.

**T1** (library mf-layer notes in training, labelled synthetic; each library note scored by the fold model whose test fold holds its library)

| CCMusic notes vs | CNN only T1 | Fusion, CNN + 47 features T1 | Fusion, CNN + 16 Group-A medians T1 | XGB all47 T1 | XGB med16 T1 |
|---|---|---|---|---|---|
| SoundFont renders | 0.907 [0.846, 0.968]; det 95.7% | 0.930 [0.857, 0.988]; det 94.9% | 0.901 [0.811, 0.976]; det 96.6% | 0.924 [0.856, 0.970]; det 88.5% | 0.817 [0.728, 0.898]; det 76.9% |
| Otey notes | 0.801 [0.704, 0.909]; det 75.0% | 0.806 [0.677, 0.943]; det 73.3% | 0.845 [0.741, 0.952]; det 88.3% | 0.809 [0.697, 0.898]; det 61.7% | 0.638 [0.548, 0.749]; det 33.3% |
| algorithmic (SoundFont + Otey) | 0.893 [0.819, 0.964]; det 92.9% | 0.913 [0.828, 0.985]; det 92.0% | 0.894 [0.806, 0.973]; det 95.5% | 0.908 [0.833, 0.966]; det 84.8% | 0.793 [0.692, 0.888]; det 71.0% |
| library notes: all 4 libraries, mf layer | 0.597 [0.382, 0.848]; det 51.3% | 0.558 [0.305, 0.848]; det 38.8% | 0.483 [0.245, 0.776]; det 36.6% | 0.440 [0.282, 0.627]; det 12.9% | 0.345 [0.204, 0.517]; det 9.8% |
| library notes: all 4 libraries, all layers | 0.688 [0.403, 0.875]; det 64.7% | 0.616 [0.313, 0.844]; det 49.5% | 0.561 [0.263, 0.803]; det 47.3% | 0.506 [0.307, 0.656]; det 17.9% | 0.410 [0.239, 0.550]; det 16.1% |
| (CCMusic false alarms, P > 0.5) | 34.4% | 26.0% | 33.7% | 19.9% | 27.2% |

**T2** (libraries excluded from training; library notes = mean of the 5 fold models)

| CCMusic notes vs | CNN only T2 | Fusion, CNN + 47 features T2 | Fusion, CNN + 16 Group-A medians T2 | XGB all47 I-only | XGB all47 T2 | XGB med16 I-only |
|---|---|---|---|---|---|---|
| SoundFont renders | 0.910 [0.832, 0.970]; det 87.8% | 0.940 [0.862, 0.988]; det 93.3% | 0.924 [0.831, 0.982]; det 88.5% | 0.929 [0.856, 0.975]; det 87.0% | 0.934 [0.870, 0.976]; det 88.3% | 0.849 [0.744, 0.931]; det 75.6% |
| Otey notes | 0.776 [0.679, 0.861]; det 50.8% | 0.817 [0.686, 0.933]; det 52.5% | 0.797 [0.664, 0.909]; det 48.3% | 0.779 [0.640, 0.891]; det 38.3% | 0.807 [0.696, 0.900]; det 50.0% | 0.606 [0.478, 0.726]; det 24.2% |
| algorithmic (SoundFont + Otey) | 0.892 [0.810, 0.961]; det 82.8% | 0.923 [0.835, 0.983]; det 87.8% | 0.907 [0.806, 0.977]; det 83.0% | 0.908 [0.820, 0.970]; det 80.4% | 0.917 [0.842, 0.971]; det 83.1% | 0.816 [0.695, 0.919]; det 68.6% |
| library notes: all 4 libraries, mf layer | 0.713 [0.548, 0.877]; det 36.2% | 0.619 [0.427, 0.819]; det 18.3% | 0.609 [0.410, 0.835]; det 17.0% | 0.551 [0.384, 0.723]; det 11.2% | 0.530 [0.370, 0.708]; det 13.4% | 0.476 [0.316, 0.666]; det 12.1% |
| library notes: all 4 libraries, all layers | 0.756 [0.574, 0.888]; det 45.8% | 0.646 [0.439, 0.832]; det 17.9% | 0.667 [0.419, 0.863]; det 20.9% | 0.600 [0.403, 0.763]; det 13.1% | 0.570 [0.389, 0.720]; det 14.0% | 0.560 [0.360, 0.715]; det 22.7% |
| (CCMusic false alarms, P > 0.5) | 18.1% | 16.2% | 15.9% | 15.4% | 16.0% | 22.6% |

Listening clips (out of sample): CNN only, LOGO (primary): recorded det 95% / algorithmic det 100%, AUC 0.140; CNN only, CV fold models (T2): recorded det 90% / algorithmic det 100%, AUC 0.905; CNN only, CV fold models (T2), unprocessed source note files: recorded det 60% / algorithmic det 90%, AUC 0.890; Fusion, CNN + 47 features, LOGO (primary): recorded det 35% / algorithmic det 70%, AUC 0.600; Fusion, CNN + 47 features, CV fold models (T2): recorded det 35% / algorithmic det 60%, AUC 0.590; Fusion, CNN + 47 features, CV fold models (T2), unprocessed source note files: recorded det 30% / algorithmic det 90%, AUC 0.900; Fusion, CNN + 16 Group-A medians, LOGO (primary): recorded det 55% / algorithmic det 70%, AUC 0.655; Fusion, CNN + 16 Group-A medians, CV fold models (T2): recorded det 45% / algorithmic det 60%, AUC 0.610; Fusion, CNN + 16 Group-A medians, CV fold models (T2), unprocessed source note files: recorded det 20% / algorithmic det 90%, AUC 0.950; XGB all47, LOGO, T2 population incl. performances (RESULTS_v2 s.4): recorded det 45% / algorithmic det 60%, AUC 0.665.

## 1. Data, phaseograms, models, protocol

| source | files | CV groups | fold(s) | files with nominal MIDI 48-76 |
|---|---|---|---|---|
| CCMusic notes (real) | 668 | 6 | 0, 1, 3, 4 | 220 |
| SoundFont renders | 762 | 10 | 0, 1, 2, 3, 4 | 300 |
| Otey notes | 120 | 1 | 2 | 40 |
| library Salamander (all layers / mf layer) | 480 / 30 | 1 | 2 | 160 |
| library Iowa (all layers / mf layer) | 261 / 86 | 1 | 1 | 87 |
| library K18 (all layers / mf layer) | 180 / 30 | 1 | 2 | 60 |
| library FreePats (all layers / mf layer) | 78 / 78 | 1 | 1 | 26 |
| listening clips | 30 | 5 | - | 30 |

- **One file = one struck note, one phaseogram per file**, in every class: the primary segment (largest energy among segments with 24 < f0 < 4500 Hz) of `detune.segment.build_segments`, the rule the detector uses for the sample-library notes (`quality/extract_samplelib_features.py`). Construction otherwise as the detector (`quality/phasefeat.compute_phaseogram`, P = 256, C = 48; spectral-flux onset, -45 dB trim, 30-ms skip, inharmonicity-corrected f0 from `refine_f0_multipartial`). Check: for the three files the figure analysis used (S90ES C4 v64, Otey C4 v75, Otey G3 v127) the unconstrained phaseograms equal the detector's own cached arrays in `quality/cache/phase/` (max |difference| = 0).
- **Constrained f0 (variant `constr`, primary).** `detune.segment.estimate_pitches` is replaced, in the worker process only, by the same spectrum and harmonic-sum salience evaluated only at MIDI nominal-1, nominal, nominal+1, then the same parabolic peak refinement limited to [f(nominal-1), f(nominal+1)]; the sub-octave guard is dropped (it can only leave the window). The detector's B estimate and `refine_f0_multipartial` then run unchanged; if the refined f0 leaves the window the window-limited peak is kept ('fallback', counted below). Identical for all classes. Because the pitch also decides how the segmenter merges repeated onsets of the same note, constraining it can merge spurious re-onsets into one longer ring; in files where the detector's own f0 was already right this changes the image in 11% of files (f0 shift median < 1 cent).
- **Unconstrained detector f0 (variant `det48`).** The detector unchanged; all classes restricted to nominal MIDI 48-76, where its f0 is mostly right; trained and evaluated on that subset only.
- **Nominal pitch.** CCMusic: the HuggingFace dataset's `pitch` class label (`ccmusic-database/pianos`, config `8_class`, 88 Helmholtz names, e.g. `c1` = C4, `C` = C2, `A2` = A0), which the WAV export wrote into each file name (`<Piano>_<pitch>_<split>_<index>.wav`); parsed with `pv2_lib.helmholtz_to_midi` (sharps take the first spelling, `c1s-d1b` = C#4). The convention is the one the figure analysis established; its own f0 agrees with it in 211 of 220 mid-register files. SoundFont and Otey renders: note name in the file name (`C4_v127.wav`); libraries: note name in the file name (Iowa `Piano.mf.C4.aiff`, Salamander `C4v8.ogg`, K18 `C4.mf.wav`, FreePats `C4.wav`); clips: `listening_kit/answer_key.csv` column `note`. All via `pv2_lib.nominal_midi` / `sci_to_midi`.
- **Models** = `quality.realness_net._train_fold` / `_predict` unchanged (PhaseCNN 16-32-64-128 + BN, emb 128; head 128-ReLU-Dropout 0.4-1; harmonic EQ, phase roll, per-cycle gain jitter, noise 0.02, row masking; BCE(pos_weight = N_real/N_synth of the training fold) + 1.0 x MSE consistency of two views; Adam lr 1e-3, weight decay 1e-4; batch 24 files; 40 epochs, final weights; torch Generator seed 0). (a) CNN only; (b) fusion with the 47 features (median-imputed, standardised on the training fold); (c) fusion with the 16 Group-A medians (the head's input size comes from `len(realness_net.FEATURE_COLS)`, which is set to the 16 names during that call). Each fit starts from `torch.manual_seed(0)`, `np.random.seed(0)`.
- **File pooling.** Kept the code's [mean; std] pooling; every file has one note, so the code's own single-note branch makes std = 0 for every file of every class (the head sees [embedding; 0]). Nothing class-dependent.
- **47 features** for fusion = the stored rows: `quality/cache/features_cal.csv` (CCMusic), `synth_features.csv` (SoundFont, Otey); library notes and clips from `analysis/out/cache/{library_notes, stimuli_clips}.csv` (same single-file pipeline). They use the detector's unconstrained f0.
- **Folds** = `analysis/out/folds.csv` (CCMusic, SoundFont, Otey and the four library groups). **T2**: train = CCMusic + SoundFont + Otey notes outside the test fold; library notes = mean of the 5 fold models. **T1**: as T2 plus the libraries' mf-equivalent-layer notes outside the test fold (Salamander v8, Iowa mf, K18 mf, FreePats), labelled synthetic; each library note (mf or any layer) scored by the fold model whose test fold holds its library (FreePats, Iowa: fold 1; K18, Salamander: fold 2). Fold 2 holds no CCMusic group, so its real notes never enter the comparisons; the same holds for the XGB numbers.

## 2. f0 and phaseogram diagnostics per source

| source | files | detector f0 off >= 1 st (primary) | ... in MIDI 48-76 | constrained: fallback | constrained: abs(f0 - nominal) >= 0.5 st | image identical det = constr | piece mode (several onsets) | segments det / constr (median) | primary ring length (s, median) | all-zero rows of 48 (mean, constr) |
|---|---|---|---|---|---|---|---|---|---|---|
| ccmusic | 668 | 329 | 9/220 | 45 | 47 | 0.46 | 0.42 | 3 / 1 | 2.57 | 2.3 |
| soundfont | 762 | 135 | 2/300 | 14 | 4 | 0.72 | 0.85 | 1 / 1 | 1.72 | 1.4 |
| otey | 120 | 18 | 0/40 | 0 | 0 | 0.85 | 0.26 | 1 / 1 | 2.93 | 0.0 |
| library Salamander | 480 | 228 | 1/160 | 65 | 53 | 0.40 | 0.91 | 5 / 3 | 2.47 | 5.8 |
| library Iowa | 261 | 136 | 12/87 | 11 | 20 | 0.48 | 0.12 | 2 / 1 | 23.52 | 1.5 |
| library K18 | 180 | 88 | 1/60 | 11 | 6 | 0.51 | 0.39 | 2 / 1 | 5.45 | 4.4 |
| library FreePats | 78 | 35 | 0/26 | 6 | 5 | 0.45 | 0.87 | 2 / 1 | 2.47 | 6.1 |
| clip | 30 | 0 | 0/30 | 1 | 0 | 0.87 | 0.87 | 1 / 1 | 2.47 | 0.0 |

All-zero rows = cycles past the end of the primary ring (zero padding of `compute_phaseogram`): in piece mode a spurious later onset ends the ring early. Used alone as a score, the number of zero rows separates CCMusic notes from the other sources at AUC SoundFont 0.49, Otey 0.46, library notes (all layers) 0.53 (0.5 = no information), so the padding is not a usable format cue here. The detector-f0 failure counts reproduce the f0 audit of RESULTS_v2 s.3.6 exactly (CCMusic 329/668, SoundFont 135, Otey 18, Salamander 228, Iowa 136, K18 88, FreePats 35).

**Level-ratio proxy ("SNR").** RMS of the phaseogram window (onset + 30 ms, 48 periods) over the 5th-percentile RMS of 20-ms frames of the whole file (the figure analysis' SNR definition, a lower bound; for a long decaying note it also measures how far the file decays, e.g. into digital silence). "Alone" = AUC of the ratio as a score, CCMusic notes vs that library (higher = more synthetic). rho = Spearman correlation between the ratio and the model's P(synthetic) WITHIN the source (T2 scores).

| source | files | median ratio (dB) | ratio alone: AUC vs CCMusic | rho(ratio, P) CNN T2 | rho(ratio, P) fusion-47 T2 |
|---|---|---|---|---|---|
| CCMusic notes | 668 | 35.6 | - | -0.11 | -0.07 |
| SoundFont renders | 762 | 40.4 | - | +0.32 | +0.22 |
| Otey notes | 120 | 37.2 | - | +0.18 | +0.56 |
| library Salamander (all layers) | 480 | 46.7 | 0.75 | +0.53 | +0.25 |
| library Iowa (all layers) | 261 | 30.7 | 0.37 | +0.20 | +0.54 |
| library K18 (all layers) | 180 | 85.2 | 1.00 | +0.64 | +0.12 |
| library FreePats (all layers) | 78 | 58.0 | 0.92 | +0.28 | +0.45 |

**Where the 48-cycle window starts.** row 0 minus the attack (first sample with |x| > 0.02 x peak, the listening kit's rule), in ms; the detector puts row 0 30 ms after its -45 dB trim point, so pre-attack noise above -45 dB moves the window towards the attack. "Alone" = AUC of this offset as a score vs CCMusic. Last column: share with P(synthetic) > 0.5 (CNN only, T2) for windows starting < 30 / 30-60 / >= 60 ms after the attack (for CCMusic = false alarms).

| source | files | row 0 - attack, ms: median (10-90%) | alone: AUC vs CCMusic | CNN T2 P > 0.5 by window start |
|---|---|---|---|---|
| CCMusic notes | 668 | 54 (13-230) | - | 24% (n=213) / 10% (n=134) / 17% (n=321) |
| SoundFont renders | 762 | 76 (53-122) | 0.60 | 100% (n=6) / 79% (n=154) / 90% (n=602) |
| Otey notes | 120 | 49 (29-226) | 0.49 | 43% (n=51) / 48% (n=21) / 60% (n=48) |
| library Salamander (all layers) | 480 | 66 (49-127) | 0.58 | 0% (n=6) / 61% (n=124) / 68% (n=350) |
| library Iowa (all layers) | 261 | 15 (2-310) | 0.38 | 10% (n=153) / 0% (n=7) / 5% (n=101) |
| library K18 (all layers) | 180 | 60 (29-87) | 0.49 | 40% (n=25) / 26% (n=69) / 60% (n=86) |
| library FreePats (all layers) | 78 | 75 (53-122) | 0.58 | 0% (n=1) / 53% (n=19) / 55% (n=58) |
| listening clips | 30 | 55 (43-79) | - | - |

## 3. Full results, constrained-f0 phaseograms (all registers)

### 3.1 CNN only

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T1 mean P | T2 AUC [CI] | T2 det. | T2 mean P | XGB all47 T1 (v2) | XGB all47 T2 (v2) | XGB all47 I-only (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.907 [0.846, 0.968] | 95.7% | 0.914 | 0.910 [0.832, 0.970] | 87.8% | 0.795 | 0.924 [0.856, 0.970]; det 88.5% | 0.934 [0.870, 0.976]; det 88.3% | 0.929 [0.856, 0.975]; det 87.0% |
| Otey notes | 668/6 vs 120/1 | 0.801 [0.704, 0.909] | 75.0% | 0.737 | 0.776 [0.679, 0.861] | 50.8% | 0.501 | 0.809 [0.697, 0.898]; det 61.7% | 0.807 [0.696, 0.900]; det 50.0% | 0.779 [0.640, 0.891]; det 38.3% |
| algorithmic (SoundFont + Otey) | 668/6 vs 882/11 | 0.893 [0.819, 0.964] | 92.9% | 0.890 | 0.892 [0.810, 0.961] | 82.8% | 0.755 | 0.908 [0.833, 0.966]; det 84.8% | 0.917 [0.842, 0.971]; det 83.1% | n/a |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.597 [0.382, 0.848] | 51.3% | 0.499 | 0.713 [0.548, 0.877] | 36.2% | 0.346 | 0.440 [0.282, 0.627]; det 12.9% | 0.530 [0.370, 0.708]; det 13.4% | 0.551 [0.384, 0.723]; det 11.2% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.688 [0.403, 0.875] | 64.7% | 0.616 | 0.756 [0.574, 0.888] | 45.8% | 0.439 | 0.506 [0.307, 0.656]; det 17.9% | 0.570 [0.389, 0.720]; det 14.0% | 0.600 [0.403, 0.763]; det 13.1% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.827 [0.735, 0.929] | 83.3% | 0.788 | 0.835 [0.741, 0.919] | 66.7% | 0.607 | 0.615 [0.477, 0.730]; det 26.7% | 0.659 [0.514, 0.786]; det 20.0% | 0.665 [0.505, 0.802]; det 16.7% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.370 [0.244, 0.544] | 23.3% | 0.230 | 0.562 [0.438, 0.687] | 5.8% | 0.096 | 0.282 [0.184, 0.369]; det 1.2% | 0.388 [0.276, 0.501]; det 2.3% | 0.431 [0.293, 0.567]; det 0.0% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.766 [0.660, 0.888] | 73.3% | 0.682 | 0.782 [0.677, 0.879] | 46.7% | 0.441 | 0.537 [0.427, 0.630]; det 16.7% | 0.627 [0.512, 0.727]; det 20.0% | 0.642 [0.510, 0.760]; det 20.0% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.694 [0.589, 0.814] | 61.5% | 0.613 | 0.806 [0.698, 0.908] | 53.8% | 0.484 | 0.511 [0.394, 0.613]; det 19.2% | 0.600 [0.475, 0.711]; det 20.5% | 0.604 [0.467, 0.725]; det 17.9% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.819 [0.727, 0.921] | 84.2% | 0.784 | 0.832 [0.737, 0.917] | 65.6% | 0.603 | 0.607 [0.473, 0.717]; det 24.4% | 0.638 [0.496, 0.763]; det 14.6% | 0.675 [0.516, 0.811]; det 14.4% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.389 [0.265, 0.555] | 24.1% | 0.255 | 0.576 [0.452, 0.697] | 8.0% | 0.113 | 0.306 [0.213, 0.389]; det 5.0% | 0.395 [0.287, 0.504]; det 4.6% | 0.426 [0.298, 0.553]; det 3.4% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.768 [0.658, 0.893] | 72.8% | 0.694 | 0.795 [0.689, 0.894] | 44.4% | 0.457 | 0.524 [0.411, 0.623]; det 18.9% | 0.628 [0.505, 0.736]; det 23.3% | 0.649 [0.511, 0.769]; det 21.7% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.694 [0.589, 0.814] | 61.5% | 0.613 | 0.806 [0.698, 0.908] | 53.8% | 0.484 | 0.511 [0.394, 0.613]; det 19.2% | 0.600 [0.475, 0.711]; det 20.5% | 0.604 [0.467, 0.725]; det 17.9% |

CCMusic false alarms (P > 0.5) / mean P: T1 34.4% / 0.353; T2 18.1% / 0.178.

### 3.2 Fusion, CNN + 47 features

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T1 mean P | T2 AUC [CI] | T2 det. | T2 mean P | XGB all47 T1 (v2) | XGB all47 T2 (v2) | XGB all47 I-only (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.930 [0.857, 0.988] | 94.9% | 0.944 | 0.940 [0.862, 0.988] | 93.3% | 0.912 | 0.924 [0.856, 0.970]; det 88.5% | 0.934 [0.870, 0.976]; det 88.3% | 0.929 [0.856, 0.975]; det 87.0% |
| Otey notes | 668/6 vs 120/1 | 0.806 [0.677, 0.943] | 73.3% | 0.704 | 0.817 [0.686, 0.933] | 52.5% | 0.513 | 0.809 [0.697, 0.898]; det 61.7% | 0.807 [0.696, 0.900]; det 50.0% | 0.779 [0.640, 0.891]; det 38.3% |
| algorithmic (SoundFont + Otey) | 668/6 vs 882/11 | 0.913 [0.828, 0.985] | 92.0% | 0.911 | 0.923 [0.835, 0.983] | 87.8% | 0.857 | 0.908 [0.833, 0.966]; det 84.8% | 0.917 [0.842, 0.971]; det 83.1% | n/a |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.558 [0.305, 0.848] | 38.8% | 0.370 | 0.619 [0.427, 0.819] | 18.3% | 0.194 | 0.440 [0.282, 0.627]; det 12.9% | 0.530 [0.370, 0.708]; det 13.4% | 0.551 [0.384, 0.723]; det 11.2% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.616 [0.313, 0.844] | 49.5% | 0.473 | 0.646 [0.439, 0.832] | 17.9% | 0.211 | 0.506 [0.307, 0.656]; det 17.9% | 0.570 [0.389, 0.720]; det 14.0% | 0.600 [0.403, 0.763]; det 13.1% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.750 [0.598, 0.917] | 66.7% | 0.610 | 0.686 [0.527, 0.841] | 16.7% | 0.189 | 0.615 [0.477, 0.730]; det 26.7% | 0.659 [0.514, 0.786]; det 20.0% | 0.665 [0.505, 0.802]; det 16.7% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.308 [0.162, 0.526] | 4.7% | 0.063 | 0.476 [0.333, 0.628] | 1.2% | 0.028 | 0.282 [0.184, 0.369]; det 1.2% | 0.388 [0.276, 0.501]; det 2.3% | 0.431 [0.293, 0.567]; det 0.0% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.764 [0.627, 0.908] | 80.0% | 0.713 | 0.746 [0.605, 0.882] | 40.0% | 0.377 | 0.537 [0.427, 0.630]; det 16.7% | 0.627 [0.512, 0.727]; det 20.0% | 0.642 [0.510, 0.760]; det 20.0% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.682 [0.538, 0.842] | 50.0% | 0.486 | 0.702 [0.559, 0.842] | 29.5% | 0.310 | 0.511 [0.394, 0.613]; det 19.2% | 0.600 [0.475, 0.711]; det 20.5% | 0.604 [0.467, 0.725]; det 17.9% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.730 [0.592, 0.879] | 66.0% | 0.619 | 0.701 [0.546, 0.852] | 18.5% | 0.230 | 0.607 [0.473, 0.717]; det 24.4% | 0.638 [0.496, 0.763]; det 14.6% | 0.675 [0.516, 0.811]; det 14.4% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.294 [0.157, 0.502] | 7.7% | 0.080 | 0.449 [0.316, 0.593] | 4.2% | 0.050 | 0.306 [0.213, 0.389]; det 5.0% | 0.395 [0.287, 0.504]; det 4.6% | 0.426 [0.298, 0.553]; det 3.4% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.751 [0.614, 0.899] | 66.1% | 0.649 | 0.762 [0.619, 0.899] | 31.1% | 0.350 | 0.524 [0.411, 0.623]; det 18.9% | 0.628 [0.505, 0.736]; det 23.3% | 0.649 [0.511, 0.769]; det 21.7% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.682 [0.538, 0.842] | 50.0% | 0.486 | 0.702 [0.559, 0.842] | 29.5% | 0.310 | 0.511 [0.394, 0.613]; det 19.2% | 0.600 [0.475, 0.711]; det 20.5% | 0.604 [0.467, 0.725]; det 17.9% |

CCMusic false alarms (P > 0.5) / mean P: T1 26.0% / 0.272; T2 16.2% / 0.164.

### 3.3 Fusion, CNN + 16 Group-A medians

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T1 mean P | T2 AUC [CI] | T2 det. | T2 mean P | XGB med16 T1 (v2) | XGB med16 T2 (v2) | XGB med16 I-only (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.901 [0.811, 0.976] | 96.6% | 0.951 | 0.924 [0.831, 0.982] | 88.5% | 0.856 | 0.817 [0.728, 0.898]; det 76.9% | 0.834 [0.748, 0.907]; det 75.3% | 0.849 [0.744, 0.931]; det 75.6% |
| Otey notes | 668/6 vs 120/1 | 0.845 [0.741, 0.952] | 88.3% | 0.869 | 0.797 [0.664, 0.909] | 48.3% | 0.473 | 0.638 [0.548, 0.749]; det 33.3% | 0.684 [0.598, 0.774]; det 33.3% | 0.606 [0.478, 0.726]; det 24.2% |
| algorithmic (SoundFont + Otey) | 668/6 vs 882/11 | 0.894 [0.806, 0.973] | 95.5% | 0.940 | 0.907 [0.806, 0.977] | 83.0% | 0.804 | 0.793 [0.692, 0.888]; det 71.0% | 0.813 [0.723, 0.898]; det 69.6% | n/a |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.483 [0.245, 0.776] | 36.6% | 0.335 | 0.609 [0.410, 0.835] | 17.0% | 0.202 | 0.345 [0.204, 0.517]; det 9.8% | 0.445 [0.299, 0.613]; det 12.9% | 0.476 [0.316, 0.666]; det 12.1% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.561 [0.263, 0.803] | 47.3% | 0.449 | 0.667 [0.419, 0.863] | 20.9% | 0.249 | 0.410 [0.239, 0.550]; det 16.1% | 0.518 [0.333, 0.654]; det 21.2% | 0.560 [0.360, 0.715]; det 22.7% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.630 [0.464, 0.827] | 50.0% | 0.467 | 0.730 [0.587, 0.867] | 23.3% | 0.251 | 0.424 [0.344, 0.517]; det 13.3% | 0.552 [0.465, 0.643]; det 20.0% | 0.609 [0.482, 0.732]; det 26.7% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.264 [0.118, 0.477] | 7.0% | 0.074 | 0.413 [0.297, 0.537] | 0.0% | 0.036 | 0.196 [0.131, 0.274]; det 1.2% | 0.281 [0.217, 0.350]; det 1.2% | 0.331 [0.230, 0.439]; det 0.0% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.732 [0.591, 0.888] | 86.7% | 0.751 | 0.751 [0.611, 0.882] | 26.7% | 0.348 | 0.503 [0.414, 0.614]; det 23.3% | 0.590 [0.507, 0.675]; det 33.3% | 0.613 [0.491, 0.728]; det 26.7% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.573 [0.410, 0.777] | 44.9% | 0.412 | 0.724 [0.581, 0.855] | 29.5% | 0.309 | 0.418 [0.328, 0.528]; det 12.8% | 0.528 [0.441, 0.619]; det 15.4% | 0.532 [0.410, 0.650]; det 14.1% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.660 [0.504, 0.838] | 56.9% | 0.551 | 0.763 [0.619, 0.897] | 27.7% | 0.315 | 0.468 [0.381, 0.574]; det 20.4% | 0.588 [0.501, 0.680]; det 26.0% | 0.644 [0.519, 0.762]; det 32.3% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.249 [0.109, 0.452] | 6.5% | 0.070 | 0.394 [0.289, 0.504] | 2.3% | 0.049 | 0.226 [0.159, 0.304]; det 3.1% | 0.308 [0.241, 0.379]; det 5.0% | 0.360 [0.255, 0.469]; det 1.9% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.745 [0.613, 0.892] | 82.2% | 0.741 | 0.781 [0.636, 0.914] | 26.1% | 0.339 | 0.518 [0.429, 0.627]; det 25.0% | 0.631 [0.547, 0.719]; det 34.4% | 0.638 [0.515, 0.755]; det 31.1% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.573 [0.410, 0.777] | 44.9% | 0.412 | 0.724 [0.581, 0.855] | 29.5% | 0.309 | 0.418 [0.328, 0.528]; det 12.8% | 0.528 [0.441, 0.619]; det 15.4% | 0.532 [0.410, 0.650]; det 14.1% |

CCMusic false alarms (P > 0.5) / mean P: T1 33.7% / 0.337; T2 15.9% / 0.159.

### 3.4 Control: XGBoost on exactly these isolated-note populations and folds (computed here)

Same rows, folds, T1/T2 definitions and scoring rules as the CNN; protocol-v2 XGBoost settings. T2 here is the I-only detector of RESULTS_v2 (identical training set and scoring).

| CCMusic notes vs | XGB all47 T1 | XGB all47 T2 | XGB med16 T1 | XGB med16 T2 |
|---|---|---|---|---|
| SoundFont renders | 0.867 [0.785, 0.951]; det 87.8% | 0.929 [0.856, 0.975]; det 87.0% | 0.777 [0.656, 0.908]; det 81.1% | 0.849 [0.744, 0.931]; det 75.6% |
| Otey notes | 0.687 [0.551, 0.838]; det 43.3% | 0.779 [0.640, 0.891]; det 38.3% | 0.522 [0.375, 0.710]; det 32.5% | 0.606 [0.478, 0.726]; det 24.2% |
| algorithmic (SoundFont + Otey) | 0.842 [0.736, 0.943]; det 81.7% | 0.908 [0.820, 0.970]; det 80.4% | 0.742 [0.598, 0.883]; det 74.5% | 0.816 [0.695, 0.919]; det 68.6% |
| library notes: all 4 libraries, mf layer | 0.424 [0.236, 0.664]; det 19.6% | 0.551 [0.384, 0.723]; det 11.2% | 0.342 [0.173, 0.579]; det 17.0% | 0.476 [0.316, 0.666]; det 12.1% |
| library notes: all 4 libraries, all layers | 0.508 [0.272, 0.709]; det 28.4% | 0.600 [0.403, 0.763]; det 13.1% | 0.441 [0.210, 0.651]; det 29.5% | 0.560 [0.360, 0.715]; det 22.7% |
| library notes: Salamander, mf layer | 0.560 [0.418, 0.727]; det 36.7% | 0.665 [0.505, 0.802]; det 16.7% | 0.453 [0.327, 0.625]; det 33.3% | 0.609 [0.482, 0.732]; det 26.7% |
| library notes: Iowa, mf layer | 0.263 [0.142, 0.444]; det 1.2% | 0.431 [0.293, 0.567]; det 0.0% | 0.210 [0.103, 0.382]; det 3.5% | 0.331 [0.230, 0.439]; det 0.0% |
| library notes: K18, mf layer | 0.620 [0.494, 0.761]; det 46.7% | 0.642 [0.510, 0.760]; det 20.0% | 0.567 [0.430, 0.738]; det 46.7% | 0.613 [0.491, 0.728]; det 26.7% |
| library notes: FreePats, mf layer | 0.473 [0.345, 0.634]; det 23.1% | 0.604 [0.467, 0.725]; det 17.9% | 0.359 [0.233, 0.542]; det 14.1% | 0.532 [0.410, 0.650]; det 14.1% |
| library notes: Salamander, all layers | 0.598 [0.458, 0.757]; det 37.7% | 0.675 [0.516, 0.811]; det 14.4% | 0.515 [0.381, 0.690]; det 37.7% | 0.644 [0.519, 0.762]; det 32.3% |
| library notes: Iowa, all layers | 0.291 [0.174, 0.460]; det 4.2% | 0.426 [0.298, 0.553]; det 3.4% | 0.231 [0.120, 0.406]; det 3.8% | 0.360 [0.255, 0.469]; det 1.9% |
| library notes: K18, all layers | 0.596 [0.469, 0.743]; det 41.1% | 0.649 [0.511, 0.769]; det 21.7% | 0.584 [0.457, 0.745]; det 51.7% | 0.638 [0.515, 0.755]; det 31.1% |
| library notes: FreePats, all layers | 0.473 [0.345, 0.634]; det 23.1% | 0.604 [0.467, 0.725]; det 17.9% | 0.359 [0.233, 0.542]; det 14.1% | 0.532 [0.410, 0.650]; det 14.1% |
| (CCMusic false alarms) | 28.7% | 15.4% | 37.4% | 22.6% |

### 3.5 Library notes under each individual model trained on the T2 population (constrained f0)

All library notes are out of sample for all 8 models (5 CV fold models + 3 LOGO models). det = share of mf-layer library notes with P > 0.5, per library and pooled; AUC = CCMusic notes of the model's own test fold vs all mf-layer library notes (fold models only; fold 2 holds no CCMusic).

| model | fit | library mf det. | Salamander det. | Iowa det. | K18 det. | FreePats det. | AUC own-fold CCMusic vs library mf | own-fold CCMusic false alarms |
|---|---|---|---|---|---|---|---|---|
| CNN only | f0 | 42% | 73% | 15% | 57% | 54% | 0.626 | 27% (n=176) |
| CNN only | f1 | 37% | 63% | 12% | 47% | 50% | 0.752 | 3% (n=140) |
| CNN only | f2 | 41% | 73% | 8% | 50% | 60% | - | - |
| CNN only | f3 | 29% | 53% | 3% | 37% | 46% | 0.640 | 12% (n=176) |
| CNN only | f4 | 36% | 70% | 5% | 47% | 53% | 0.615 | 27% (n=176) |
| CNN only | logo_full | 51% | 80% | 21% | 63% | 69% | - | - |
| CNN only | logo_noOtey | 40% | 67% | 14% | 57% | 51% | - | - |
| CNN only | logo_noCasio | 23% | 53% | 5% | 33% | 27% | - | - |
| Fusion, CNN + 47 features | f0 | 27% | 13% | 9% | 57% | 41% | 0.513 | 25% (n=176) |
| Fusion, CNN + 47 features | f1 | 15% | 13% | 1% | 30% | 26% | 0.767 | 0% (n=140) |
| Fusion, CNN + 47 features | f2 | 15% | 13% | 0% | 33% | 26% | - | - |
| Fusion, CNN + 47 features | f3 | 14% | 10% | 0% | 30% | 26% | 0.589 | 4% (n=176) |
| Fusion, CNN + 47 features | f4 | 24% | 40% | 2% | 37% | 37% | 0.463 | 32% (n=176) |
| Fusion, CNN + 47 features | logo_full | 19% | 13% | 2% | 37% | 32% | - | - |
| Fusion, CNN + 47 features | logo_noOtey | 22% | 17% | 2% | 43% | 37% | - | - |
| Fusion, CNN + 47 features | logo_noCasio | 16% | 10% | 1% | 33% | 28% | - | - |
| Fusion, CNN + 16 Group-A medians | f0 | 30% | 30% | 12% | 70% | 36% | 0.554 | 23% (n=176) |
| Fusion, CNN + 16 Group-A medians | f1 | 22% | 27% | 3% | 30% | 38% | 0.764 | 1% (n=140) |
| Fusion, CNN + 16 Group-A medians | f2 | 17% | 20% | 0% | 30% | 31% | - | - |
| Fusion, CNN + 16 Group-A medians | f3 | 12% | 10% | 0% | 33% | 18% | 0.553 | 6% (n=176) |
| Fusion, CNN + 16 Group-A medians | f4 | 16% | 30% | 1% | 17% | 26% | 0.436 | 31% (n=176) |
| Fusion, CNN + 16 Group-A medians | logo_full | 17% | 23% | 1% | 23% | 31% | - | - |
| Fusion, CNN + 16 Group-A medians | logo_noOtey | 33% | 33% | 10% | 47% | 51% | - | - |
| Fusion, CNN + 16 Group-A medians | logo_noCasio | 18% | 27% | 2% | 23% | 31% | - | - |

## 4. Register and f0 sensitivity

Rows: `constr` = constrained-f0 models (all registers) evaluated on all files, on nominal MIDI 48-76 only (both classes), or on files whose constrained f0 lies within 0.5 semitone of the nominal pitch; `det48` = models trained and evaluated on MIDI 48-76 with the detector's unconstrained f0. XGB rows: RESULTS_v2 s.3.6 (corpus-level training, evaluation restricted to MIDI 48-76) and the matched control restricted to 48-76.

**T1**

| model / restriction | CCMusic files | SoundFont renders | Otey notes | algorithmic (SoundFont + Otey) | library notes: all 4 libraries, mf layer | library notes: all 4 libraries, all layers | CCMusic false alarms |
|---|---|---|---|---|---|---|---|
| CNN only, constr, all | 668 | 0.907 [0.846, 0.968]; det 95.7% | 0.801 [0.704, 0.909]; det 75.0% | 0.893 [0.819, 0.964]; det 92.9% | 0.597 [0.382, 0.848]; det 51.3% | 0.688 [0.403, 0.875]; det 64.7% | 34.4% |
| CNN only, constr, MIDI 48-76 | 220 | 0.923 [0.855, 0.988]; det 100.0% | 0.898 [0.822, 0.982]; det 100.0% | 0.920 [0.849, 0.989]; det 100.0% | 0.721 [0.479, 0.947]; det 74.7% | 0.776 [0.483, 0.953]; det 81.4% | 41.4% |
| CNN only, constr, abs(f0 - nominal) < 0.5 st | 621 | 0.906 [0.841, 0.970]; det 96.2% | 0.795 [0.691, 0.909]; det 75.0% | 0.891 [0.812, 0.967]; det 93.3% | 0.608 [0.387, 0.864]; det 54.0% | 0.701 [0.411, 0.894]; det 68.4% | 35.3% |
| CNN only, det48 (unconstrained f0, MIDI 48-76) | 220 | 0.889 [0.803, 0.949]; det 86.7% | 0.913 [0.863, 0.954]; det 92.5% | 0.892 [0.815, 0.950]; det 87.4% | 0.761 [0.562, 0.936]; det 68.0% | 0.778 [0.584, 0.907]; det 74.8% | 24.1% |
| Fusion, CNN + 47 features, constr, all | 668 | 0.930 [0.857, 0.988]; det 94.9% | 0.806 [0.677, 0.943]; det 73.3% | 0.913 [0.828, 0.985]; det 92.0% | 0.558 [0.305, 0.848]; det 38.8% | 0.616 [0.313, 0.844]; det 49.5% | 26.0% |
| Fusion, CNN + 47 features, constr, MIDI 48-76 | 220 | 0.936 [0.837, 0.998]; det 99.3% | 0.755 [0.548, 0.959]; det 70.0% | 0.915 [0.797, 0.997]; det 95.9% | 0.589 [0.292, 0.878]; det 44.0% | 0.633 [0.322, 0.887]; det 57.1% | 31.4% |
| Fusion, CNN + 47 features, constr, abs(f0 - nominal) < 0.5 st | 621 | 0.927 [0.848, 0.989]; det 95.0% | 0.798 [0.654, 0.944]; det 73.3% | 0.910 [0.819, 0.986]; det 92.0% | 0.561 [0.306, 0.861]; det 38.9% | 0.623 [0.316, 0.860]; det 50.6% | 27.2% |
| Fusion, CNN + 47 features, det48 (unconstrained f0, MIDI 48-76) | 220 | 0.916 [0.815, 0.981]; det 97.0% | 0.778 [0.599, 0.916]; det 87.5% | 0.900 [0.786, 0.975]; det 95.9% | 0.580 [0.333, 0.809]; det 37.3% | 0.543 [0.345, 0.779]; det 41.7% | 35.9% |
| Fusion, CNN + 16 Group-A medians, constr, all | 668 | 0.901 [0.811, 0.976]; det 96.6% | 0.845 [0.741, 0.952]; det 88.3% | 0.894 [0.806, 0.973]; det 95.5% | 0.483 [0.245, 0.776]; det 36.6% | 0.561 [0.263, 0.803]; det 47.3% | 33.7% |
| Fusion, CNN + 16 Group-A medians, constr, MIDI 48-76 | 220 | 0.894 [0.788, 0.986]; det 99.7% | 0.734 [0.560, 0.926]; det 80.0% | 0.875 [0.757, 0.982]; det 97.4% | 0.529 [0.259, 0.818]; det 44.0% | 0.545 [0.286, 0.818]; det 45.0% | 40.0% |
| Fusion, CNN + 16 Group-A medians, constr, abs(f0 - nominal) < 0.5 st | 621 | 0.899 [0.805, 0.978]; det 96.8% | 0.838 [0.721, 0.953]; det 88.3% | 0.891 [0.796, 0.975]; det 95.7% | 0.490 [0.244, 0.786]; det 37.9% | 0.566 [0.274, 0.814]; det 48.7% | 34.8% |
| Fusion, CNN + 16 Group-A medians, det48 (unconstrained f0, MIDI 48-76) | 220 | 0.922 [0.829, 0.978]; det 99.0% | 0.704 [0.534, 0.855]; det 67.5% | 0.896 [0.787, 0.974]; det 95.3% | 0.656 [0.401, 0.864]; det 58.7% | 0.579 [0.405, 0.788]; det 48.3% | 36.8% |
| XGB all47 T1 (RESULTS_v2 s.3.6), MIDI 48-76 | 220 | 0.937 [0.846, 0.985]; det 95.3% | 0.759 [0.620, 0.861]; det 55.0% | 0.916 [0.818, 0.979]; det 90.6% | 0.400 [0.233, 0.582]; det 16.0% | 0.423 [0.259, 0.583]; det 15.9% | 23.6% |
| XGB all47 T1, matched control, MIDI 48-76 | 220 | 0.874 [0.774, 0.973]; det 93.3% | 0.595 [0.415, 0.803]; det 42.5% | 0.841 [0.708, 0.964]; det 87.4% | 0.388 [0.189, 0.637]; det 16.0% | 0.417 [0.213, 0.655]; det 18.3% | 35.9% |
| XGB med16 T1, matched control, MIDI 48-76 | 220 | 0.794 [0.660, 0.932]; det 88.0% | 0.488 [0.329, 0.685]; det 32.5% | 0.758 [0.599, 0.915]; det 81.5% | 0.324 [0.136, 0.570]; det 14.7% | 0.343 [0.164, 0.566]; det 17.4% | 40.0% |

**T2**

| model / restriction | CCMusic files | SoundFont renders | Otey notes | algorithmic (SoundFont + Otey) | library notes: all 4 libraries, mf layer | library notes: all 4 libraries, all layers | CCMusic false alarms |
|---|---|---|---|---|---|---|---|
| CNN only, constr, all | 668 | 0.910 [0.832, 0.970]; det 87.8% | 0.776 [0.679, 0.861]; det 50.8% | 0.892 [0.810, 0.961]; det 82.8% | 0.713 [0.548, 0.877]; det 36.2% | 0.756 [0.574, 0.888]; det 45.8% | 18.1% |
| CNN only, constr, MIDI 48-76 | 220 | 0.876 [0.752, 0.976]; det 90.0% | 0.880 [0.769, 0.970]; det 82.5% | 0.876 [0.759, 0.977]; det 89.1% | 0.712 [0.490, 0.948]; det 56.0% | 0.769 [0.513, 0.952]; det 65.2% | 27.7% |
| CNN only, constr, abs(f0 - nominal) < 0.5 st | 621 | 0.910 [0.828, 0.973]; det 88.3% | 0.772 [0.669, 0.863]; det 50.8% | 0.891 [0.803, 0.965]; det 83.1% | 0.723 [0.551, 0.895]; det 38.4% | 0.766 [0.575, 0.904]; det 49.5% | 18.5% |
| CNN only, det48 (unconstrained f0, MIDI 48-76) | 220 | 0.832 [0.700, 0.926]; det 95.3% | 0.714 [0.529, 0.862]; det 77.5% | 0.819 [0.678, 0.923]; det 93.2% | 0.622 [0.382, 0.852]; det 60.0% | 0.662 [0.412, 0.852]; det 71.2% | 42.7% |
| Fusion, CNN + 47 features, constr, all | 668 | 0.940 [0.862, 0.988]; det 93.3% | 0.817 [0.686, 0.933]; det 52.5% | 0.923 [0.835, 0.983]; det 87.8% | 0.619 [0.427, 0.819]; det 18.3% | 0.646 [0.439, 0.832]; det 17.9% | 16.2% |
| Fusion, CNN + 47 features, constr, MIDI 48-76 | 220 | 0.944 [0.850, 0.997]; det 97.0% | 0.783 [0.586, 0.954]; det 50.0% | 0.925 [0.814, 0.994]; det 91.5% | 0.612 [0.386, 0.815]; det 22.7% | 0.604 [0.388, 0.810]; det 18.6% | 21.4% |
| Fusion, CNN + 47 features, constr, abs(f0 - nominal) < 0.5 st | 621 | 0.937 [0.854, 0.989]; det 93.5% | 0.809 [0.668, 0.936]; det 52.5% | 0.920 [0.825, 0.984]; det 87.9% | 0.625 [0.426, 0.827]; det 19.4% | 0.650 [0.441, 0.839]; det 19.1% | 17.1% |
| Fusion, CNN + 47 features, det48 (unconstrained f0, MIDI 48-76) | 220 | 0.936 [0.857, 0.984]; det 97.3% | 0.671 [0.459, 0.861]; det 15.0% | 0.905 [0.788, 0.978]; det 87.6% | 0.585 [0.353, 0.789]; det 22.7% | 0.515 [0.326, 0.754]; det 22.5% | 24.5% |
| Fusion, CNN + 16 Group-A medians, constr, all | 668 | 0.924 [0.831, 0.982]; det 88.5% | 0.797 [0.664, 0.909]; det 48.3% | 0.907 [0.806, 0.977]; det 83.0% | 0.609 [0.410, 0.835]; det 17.0% | 0.667 [0.419, 0.863]; det 20.9% | 15.9% |
| Fusion, CNN + 16 Group-A medians, constr, MIDI 48-76 | 220 | 0.923 [0.796, 0.996]; det 93.0% | 0.837 [0.672, 0.976]; det 60.0% | 0.913 [0.789, 0.995]; det 89.1% | 0.644 [0.432, 0.862]; det 21.3% | 0.668 [0.423, 0.909]; det 21.0% | 17.3% |
| Fusion, CNN + 16 Group-A medians, constr, abs(f0 - nominal) < 0.5 st | 621 | 0.925 [0.824, 0.987]; det 88.9% | 0.796 [0.662, 0.915]; det 48.3% | 0.907 [0.800, 0.981]; det 83.4% | 0.620 [0.420, 0.846]; det 18.0% | 0.676 [0.431, 0.879]; det 22.0% | 16.1% |
| Fusion, CNN + 16 Group-A medians, det48 (unconstrained f0, MIDI 48-76) | 220 | 0.919 [0.770, 0.988]; det 97.7% | 0.732 [0.518, 0.885]; det 57.5% | 0.897 [0.745, 0.984]; det 92.9% | 0.600 [0.378, 0.801]; det 32.0% | 0.590 [0.390, 0.782]; det 32.7% | 32.7% |
| XGB all47 T2 (RESULTS_v2 s.3.6), MIDI 48-76 | 220 | 0.946 [0.858, 0.991]; det 95.7% | 0.746 [0.594, 0.860]; det 45.0% | 0.923 [0.821, 0.987]; det 89.7% | 0.504 [0.337, 0.671]; det 17.3% | 0.495 [0.335, 0.661]; det 16.2% | 19.1% |
| XGB all47 T2, matched control, MIDI 48-76 | 220 | 0.948 [0.867, 0.990]; det 94.7% | 0.751 [0.599, 0.869]; det 45.0% | 0.925 [0.827, 0.985]; det 88.8% | 0.527 [0.356, 0.692]; det 13.3% | 0.534 [0.354, 0.694]; det 13.2% | 15.0% |
| XGB med16 T2 (RESULTS_v2 s.3.6), MIDI 48-76 | 220 | 0.874 [0.757, 0.947]; det 87.3% | 0.684 [0.562, 0.775]; det 40.0% | 0.852 [0.736, 0.938]; det 81.8% | 0.411 [0.241, 0.599]; det 12.0% | 0.427 [0.267, 0.592]; det 13.8% | 21.4% |
| XGB med16 T2, matched control, MIDI 48-76 | 220 | 0.895 [0.783, 0.964]; det 87.0% | 0.621 [0.485, 0.727]; det 35.0% | 0.863 [0.739, 0.953]; det 80.9% | 0.481 [0.311, 0.659]; det 12.0% | 0.496 [0.315, 0.650]; det 14.4% | 20.9% |

## 5. Listening-test clips scored out of sample

The 30 clips the listeners heard (`listening_kit/web/clip_XX.wav`), phaseogram with the constrained f0 (nominal note from `answer_key.csv`), 47 features from `analysis/out/cache/stimuli_clips.csv`. LOGO (primary, as RESULTS_v2 s.4): models trained on the whole T2 isolated-note population (CCMusic + SoundFont + Otey) minus the clip's own group -- Otey clips by a model without Otey, Casio CA-110 clips by a model without that SoundFont, library clips (Iowa, K18, Salamander) by the full model. CV-fold rows: the fold models (Otey clips by the fold-2 model, Casio clips by the fold-0 model, library clips by the mean of the 5). Detection = P > 0.5; AUC CI = clips resampled within each of the 5 classes (B = 2000, seed 0). "Unprocessed source note files" = the same models applied to the 30 original note files the clips were cut from (`answer_key.csv` column `src`; out of sample under the same rule). This isolates the effect of the clip processing (resampling to 44.1 kHz, onset trim with 10-ms pre-roll, 3-s cut, 10-ms fades, RMS normalisation, 16-bit; facts_C s.9.3).

| scoring | Iowa det. | K18 det. | Salamander det. | SoundFont det. | Otey det. | AUC recorded(20) vs algorithmic(10) [95% CI] | AUC Iowa+K18 vs algorithmic | AUC Salamander vs Iowa+K18 |
|---|---|---|---|---|---|---|---|---|
| CNN only, LOGO (primary) | 7/7 | 6/7 | 6/6 | 5/5 | 5/5 | 0.140 [0.040, 0.265] | 0.193 | 0.750 |
| CNN only, LOGO, unprocessed source note files | 2/7 | 5/7 | 6/6 | 5/5 | 5/5 | 0.430 [0.270, 0.580] | 0.593 | 0.845 |
| CNN only, CV fold models (T2) | 6/7 | 6/7 | 6/6 | 5/5 | 5/5 | 0.905 [0.805, 0.980] | 0.893 | 0.560 |
| CNN only, CV fold models (T2), unprocessed source note files | 1/7 | 5/7 | 6/6 | 5/5 | 4/5 | 0.890 [0.780, 0.980] | 0.914 | 0.774 |
| Fusion, CNN + 47 features, LOGO (primary) | 1/7 | 4/7 | 2/6 | 2/5 | 5/5 | 0.600 [0.415, 0.820] | 0.586 | 0.500 |
| Fusion, CNN + 47 features, LOGO, unprocessed source note files | 0/7 | 4/7 | 2/6 | 5/5 | 5/5 | 0.915 [0.810, 0.995] | 0.907 | 0.512 |
| Fusion, CNN + 47 features, CV fold models (T2) | 1/7 | 4/7 | 2/6 | 2/5 | 4/5 | 0.590 [0.370, 0.835] | 0.579 | 0.512 |
| Fusion, CNN + 47 features, CV fold models (T2), unprocessed source note files | 0/7 | 4/7 | 2/6 | 5/5 | 4/5 | 0.900 [0.805, 0.975] | 0.900 | 0.500 |
| Fusion, CNN + 16 Group-A medians, LOGO (primary) | 3/7 | 4/7 | 4/6 | 2/5 | 5/5 | 0.655 [0.475, 0.875] | 0.643 | 0.631 |
| Fusion, CNN + 16 Group-A medians, LOGO, unprocessed source note files | 0/7 | 2/7 | 2/6 | 5/5 | 5/5 | 0.970 [0.910, 1.000] | 0.971 | 0.571 |
| Fusion, CNN + 16 Group-A medians, CV fold models (T2) | 1/7 | 4/7 | 4/6 | 2/5 | 4/5 | 0.610 [0.390, 0.855] | 0.607 | 0.631 |
| Fusion, CNN + 16 Group-A medians, CV fold models (T2), unprocessed source note files | 0/7 | 2/7 | 2/6 | 5/5 | 4/5 | 0.950 [0.870, 1.000] | 0.943 | 0.571 |
| CNN only, MIDI 48-76 variant, CV fold models (T2) | 7/7 | 7/7 | 6/6 | 5/5 | 4/5 | 0.740 [0.550, 0.910] | 0.807 | 0.762 |
| Fusion, CNN + 47 features, MIDI 48-76 variant, CV fold models (T2) | 1/7 | 6/7 | 3/6 | 2/5 | 1/5 | 0.335 [0.100, 0.595] | 0.321 | 0.488 |
| Fusion, CNN + 16 Group-A medians, MIDI 48-76 variant, CV fold models (T2) | 6/7 | 7/7 | 5/6 | 5/5 | 4/5 | 0.810 [0.635, 0.965] | 0.814 | 0.560 |
| XGB all47, LOGO, T2 population incl. performances (RESULTS_v2 s.4) | 2/7 | 4/7 | 3/6 | 3/5 | 3/5 | 0.665 [0.470, 0.845] | 0.679 | 0.512 |
| XGB med16, LOGO, T2 population incl. performances (RESULTS_v2 s.4) | 1/7 | 3/7 | 2/6 | 2/5 | 2/5 | 0.670 [0.470, 0.845] | 0.671 | 0.512 |
| XGB all47, LOGO, isolated-note population (computed here) | 2/7 | 3/7 | 3/6 | 2/5 | 3/5 | 0.605 [0.375, 0.820] | 0.607 | 0.536 |
| XGB med16, LOGO, isolated-note population (computed here) | 0/7 | 3/7 | 2/6 | 2/5 | 2/5 | 0.590 [0.365, 0.810] | 0.579 | 0.464 |

**Clip vs its source note.** Shift of the phaseogram window (row 0 relative to the attack, clip minus source, ms) and correlation of the two constrained-f0 phaseograms at the best common circular phase shift, median per class: Iowa +42 ms, r = 0.93; K18 +1 ms, r = 0.99; Otey -0 ms, r = 1.00; Salamander +6 ms, r = 1.00; SoundFont (Casio CA-110) -0 ms, r = 1.00. A large shift and a low correlation mean that the CNN sees a different stretch of the note in the clip than in the source file. In the Iowa source files the noise floor lies within 45 dB of the peak, so the -45 dB trim starts before the attack and the 30-ms skip ends near it; the clip's onset trim and 10-ms fade-in remove that noise and push the window about 40 ms later.

Per-clip P(synthetic):

| clip | class | MIDI | constr_cnn_LOGO | constr_cnn_LOGOsrc | constr_fus47_LOGO | constr_fus47_LOGOsrc | constr_fus16_LOGO | constr_cnn_T2 | constr_cnn_T2src | xgb_all47_clip_LOGO_v2 | xgbmatched_all47_LOGO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_01 | Salamander | 51 | 0.990 | 0.980 | 0.143 | 0.000 | 0.914 | 0.801 | 0.746 | 0.095 | 0.079 |
| clip_02 | Iowa | 50 | 0.937 | 0.003 | 0.233 | 0.007 | 0.734 | 0.456 | 0.002 | 0.250 | 0.184 |
| clip_03 | K18 | 51 | 0.995 | 0.991 | 0.450 | 0.000 | 0.220 | 0.873 | 0.842 | 0.300 | 0.266 |
| clip_04 | Salamander | 60 | 0.999 | 0.998 | 0.614 | 0.000 | 0.954 | 0.940 | 0.936 | 0.861 | 0.937 |
| clip_05 | Salamander | 63 | 0.997 | 0.997 | 0.386 | 0.909 | 0.836 | 0.933 | 0.951 | 0.684 | 0.713 |
| clip_06 | Iowa | 71 | 0.996 | 0.991 | 0.664 | 0.305 | 0.719 | 0.953 | 0.904 | 0.889 | 0.715 |
| clip_07 | Otey | 71 | 0.993 | 0.986 | 0.996 | 0.994 | 0.997 | 0.972 | 0.964 | 0.981 | 0.973 |
| clip_08 | Salamander | 57 | 0.997 | 0.996 | 0.050 | 0.000 | 0.417 | 0.889 | 0.831 | 0.008 | 0.002 |
| clip_09 | Iowa | 65 | 0.997 | 0.006 | 0.161 | 0.000 | 0.332 | 0.962 | 0.005 | 0.555 | 0.448 |
| clip_10 | SoundFont (Casio CA-110) | 60 | 0.882 | 0.882 | 0.527 | 0.601 | 0.939 | 0.983 | 0.983 | 0.998 | 0.999 |
| clip_11 | Salamander | 66 | 0.999 | 0.999 | 0.326 | 0.001 | 0.483 | 0.971 | 0.974 | 0.010 | 0.020 |
| clip_12 | Iowa | 59 | 0.958 | 0.004 | 0.002 | 0.000 | 0.139 | 0.528 | 0.001 | 0.279 | 0.562 |
| clip_13 | Otey | 72 | 0.967 | 0.837 | 0.998 | 0.998 | 1.000 | 0.941 | 0.376 | 0.800 | 0.882 |
| clip_14 | Iowa | 55 | 0.972 | 0.680 | 0.243 | 0.020 | 0.838 | 0.618 | 0.153 | 0.008 | 0.006 |
| clip_15 | K18 | 72 | 0.997 | 0.997 | 0.995 | 0.999 | 0.989 | 0.952 | 0.970 | 0.998 | 0.993 |
| clip_16 | K18 | 48 | 0.033 | 0.017 | 0.000 | 0.000 | 0.000 | 0.162 | 0.137 | 0.000 | 0.000 |
| clip_17 | K18 | 54 | 0.995 | 0.994 | 0.993 | 0.766 | 0.993 | 0.925 | 0.930 | 0.714 | 0.410 |
| clip_18 | K18 | 63 | 0.999 | 0.999 | 0.356 | 0.102 | 0.049 | 0.976 | 0.971 | 0.306 | 0.191 |
| clip_19 | Otey | 64 | 0.973 | 0.973 | 0.927 | 0.935 | 0.998 | 0.952 | 0.952 | 0.936 | 0.748 |
| clip_20 | Salamander | 75 | 0.993 | 0.994 | 0.974 | 0.828 | 0.838 | 0.902 | 0.919 | 1.000 | 0.997 |
| clip_21 | SoundFont (Casio CA-110) | 48 | 0.922 | 0.922 | 1.000 | 1.000 | 0.998 | 0.988 | 0.988 | 0.999 | 1.000 |
| clip_22 | K18 | 57 | 0.997 | 0.318 | 0.999 | 0.968 | 0.995 | 0.913 | 0.131 | 0.995 | 0.994 |
| clip_23 | K18 | 75 | 0.991 | 0.980 | 0.991 | 0.723 | 0.972 | 0.912 | 0.847 | 0.640 | 0.748 |
| clip_24 | SoundFont (Casio CA-110) | 69 | 0.906 | 0.906 | 0.000 | 1.000 | 0.000 | 0.981 | 0.981 | 0.132 | 0.028 |
| clip_25 | Otey | 67 | 0.981 | 0.981 | 0.895 | 0.712 | 0.962 | 0.972 | 0.971 | 0.320 | 0.067 |
| clip_26 | SoundFont (Casio CA-110) | 57 | 0.919 | 0.919 | 0.000 | 1.000 | 0.000 | 0.988 | 0.988 | 0.083 | 0.006 |
| clip_27 | Iowa | 74 | 0.993 | 0.097 | 0.000 | 0.000 | 0.000 | 0.914 | 0.025 | 0.010 | 0.007 |
| clip_28 | Otey | 52 | 0.988 | 0.988 | 0.995 | 0.993 | 0.997 | 0.952 | 0.952 | 0.378 | 0.124 |
| clip_29 | Iowa | 53 | 0.987 | 0.297 | 0.020 | 0.000 | 0.373 | 0.905 | 0.154 | 0.017 | 0.014 |
| clip_30 | SoundFont (Casio CA-110) | 72 | 0.894 | 0.905 | 0.000 | 1.000 | 0.000 | 0.964 | 0.982 | 0.566 | 0.431 |

## 6. Runtime

| variant | model | condition | CPU-process minutes (sum over fits) | training files per fit |
|---|---|---|---|---|
| constr | cnn | LOGO | 118 | 1550, 1430, 1472 |
| constr | cnn | T1 | 228 | 1442, 1314, 1438, 1445, 1457 |
| constr | cnn | T2 | 196 | 1218, 1254, 1274, 1221, 1233 |
| constr | fus16 | LOGO | 115 | 1550, 1430, 1472 |
| constr | fus16 | T1 | 220 | 1442, 1314, 1438, 1445, 1457 |
| constr | fus16 | T2 | 184 | 1218, 1254, 1274, 1221, 1233 |
| constr | fus47 | LOGO | 117 | 1550, 1430, 1472 |
| constr | fus47 | T1 | 221 | 1442, 1314, 1438, 1445, 1457 |
| constr | fus47 | T2 | 195 | 1218, 1254, 1274, 1221, 1233 |
| det48 | cnn | T1 | 70 | 517, 474, 515, 517, 517 |
| det48 | cnn | T2 | 58 | 442, 454, 460, 442, 442 |
| det48 | fus16 | T1 | 56 | 517, 474, 515, 517, 517 |
| det48 | fus16 | T2 | 61 | 442, 454, 460, 442, 442 |
| det48 | fus47 | T1 | 68 | 517, 474, 515, 517, 517 |
| det48 | fus47 | T2 | 59 | 442, 454, 460, 442, 442 |

Sum over all fits: 32.7 h of per-process time (6 worker processes x 2 torch threads on an 8-core i9-9900K, CPU only). Training log `logs/train.log`: first entry 2026-10-01 10:24:39, last entry 2026-10-01 16:01:32 (5.61 h wall clock). Evaluation: 66 s.
