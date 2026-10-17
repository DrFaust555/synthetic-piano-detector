# S2. Phaseogram CNN and fusion: construction, training and diagnostics

This supplement documents the isolated-note phaseogram CNN and the fusion detectors of Sec. 4.1 and 5.4 (Table 3). Everything below is taken from the code and the stored outputs of the format-matched study (`Stattia/revision/cnn_iso/`: `build_phaseograms.py`, `train_cnn_iso.py`, `cnn_iso_lib.py`, `evaluate_cnn_iso.py`, `RESULTS_CNN_ISO.md`, `key_numbers_cnn_iso.json`, `comparisons_cnn_iso.csv`, `diag/CNN_DIAG.md`) and from the model code they call unchanged (`quality/phasefeat.py`, `quality/realness_net.py`, `detune/segment.py`). "Not recorded" marks facts that no stored file contains.

## S2.1 Phaseogram

| Step | Specification |
|---|---|
| Unit | One struck note per file; one phaseogram per file. |
| Note window | Detector segmentation (`detune.segment.build_segments`): onset by spectral flux; the ring is trimmed where the signal first comes within 45 dB of its maximum; the window starts 30 ms later. Of the segments of a file, the one with the largest energy and 24 Hz < $f_0$ < 4500 Hz is used ("primary segment"). The segment is peak-normalized. |
| $f_0$ (primary variant, "constrained") | Detector pitch search restricted to the nominal note ±1 semitone: harmonic-sum salience evaluated at MIDI nominal−1, nominal, nominal+1, parabolic peak refinement limited to [f(nominal−1), f(nominal+1)]; then the detector's inharmonicity estimate $B$ and its multi-partial refinement (magnitude-weighted median of the $f_0$ implied by partials $k = 1 \dots 8$ under $f_k = k f_0 \sqrt{1 + B k^2}$). If the refined value leaves the ±1-semitone window, the window-limited peak is kept (fallbacks: CCMusic 45 of 668 files, SoundFont 14 of 762, Otey 0 of 120, Salamander 65 of 480, Iowa 11 of 261, K18 11 of 180, FreePats 6 of 78). |
| Nominal pitch | CCMusic: dataset pitch label (written into the file name); SoundFont, Otey and library notes: note name in the file name. |
| Resampling | Linear interpolation so that one period spans exactly 256 samples (period = sr/$f_0$ source samples). |
| Image | The first 48 consecutive periods from the window start, stacked as rows: 48 × 256. The image therefore covers 48/$f_0$ s (e.g. 0.18 s at C4, 1.75 s at A0, 0.023 s at C7). Samples past the end of the segment are set to zero (mean number of all-zero rows: CCMusic 2.3, SoundFont 1.4, Otey 0.0, Salamander 5.8, Iowa 1.5, K18 4.4, FreePats 6.1). |
| Normalization | Mean of each row removed (per-period DC); whole image divided by its absolute peak. |
| Secondary variant ("det48") | Detector $f_0$ unchanged, only notes with nominal MIDI 48–76 (C3–E5); models trained and evaluated on that subset. |
| Where row 0 falls | Row 0 minus attack (first sample above 2% of the peak), median (10–90%): CCMusic 54 ms (13–230), SoundFont 76 (53–122), Otey 49 (29–226), Salamander 66 (49–127), Iowa 15 (2–310), K18 60 (29–87), FreePats 75 (53–122). |
| Files | 668 CCMusic, 762 SoundFont (including 84 velocity copies that protocol v3 removes from the XGBoost analyses), 120 Otey, 999 library notes (224 mf layer), 30 listening-test clips (Test 1). No file failed (build log: 0 errors). |

The irregularity measure of Table 7 is computed from the same phaseogram construction by the figure analysis (`figures/phaseo_lib.py`, 190 notes C3–E5); it is not an output of the CNN.

## S2.2 Network (Table 3 in full)

| Part | Specification | Parameters |
|---|---|---|
| Input | 1 × 48 × 256 phaseogram per note | – |
| Block 1 | Conv 3 × 3, padding 1, 1→16 channels; BatchNorm; ReLU; max pool 2 × 2 → 16 × 24 × 128 | 160 + 32 |
| Block 2 | Conv 3 × 3, 16→32; BatchNorm; ReLU; max pool → 32 × 12 × 64 | 4 640 + 64 |
| Block 3 | Conv 3 × 3, 32→64; BatchNorm; ReLU; max pool → 64 × 6 × 32 | 18 496 + 128 |
| Block 4 | Conv 3 × 3, 64→128; BatchNorm; ReLU; max pool → 128 × 3 × 16 | 73 856 + 256 |
| Embedding | Global average pool → 128; linear 128→128 | 16 512 |
| CNN total | | 114 144 |
| File pooling | Concatenation of the mean and the (population) standard deviation of the note embeddings → 256. A file with one note gets SD = 0. In this study every file holds one note, so the SD half is zero for every file of every class. | – |
| Head, CNN only | Linear 256→128, ReLU, dropout 0.4, linear 128→1, logistic | 33 025 (total 147 169) |
| Head, fusion-47 | Input = [pooled embedding (256); 47 features] = 303; same layers | 39 041 (total 153 185) |
| Head, fusion-16 | Input = [256; 16 Group-A medians] = 272; same layers | 35 073 (total 149 217) |

CNN-only totals are logged by the diagnostic runs (147 169, CNN 114 144); fusion totals are computed from the architecture.

## S2.3 Training protocol

| Item | Value |
|---|---|
| Classes | 0 = CCMusic notes; 1 = SoundFont and Otey notes. Library notes are never trained in the primary condition (T2). |
| Cross-validation | 5 folds, grouped by instrument, one fixed assignment: protocol-v2 greedy GroupKFold (`analysis/out/folds.csv`). Fold 0: CCMusic Kawai (both pianos), Casio CA-110, Yamaha PSR-400/500. Fold 1: CCMusic Steinway (both pianos), Casio PX-860, Yamaha CP-80, FreePats, Iowa. Fold 2: Otey, Casio CTK-230, Yamaha S90 ES, K18, Salamander (no CCMusic). Fold 3: CCMusic Yamaha, Young Chang, Roland JV-1080, Roland U-20. Fold 4: CCMusic Hsinghai, Pearl River, Casio CTK-533, Yamaha PSR-F50. The CNN was not repeated over the ten fold assignments used for XGBoost. |
| Training files per fit (T2) | 1218, 1254, 1274, 1221, 1233 (folds 0–4). |
| Other conditions | T1: as T2 plus the mf-layer library notes outside the test fold, labelled synthetic. LOGO: three models on the whole T2 population (all; without Otey; without Casio CA-110), used only to score listening-test clips. |
| Fits | 69 = 3 models (CNN, fusion-47, fusion-16) × [T2, T1 × 5 folds + 3 LOGO, constrained $f_0$] + 3 models × 2 conditions × 5 folds (det48). |
| Loss | $0.5\,[\mathrm{BCE}(z_1) + \mathrm{BCE}(z_2)] + 1.0 \cdot \mathrm{MSE}(\sigma(z_1), \sigma(z_2))$, where $z_1, z_2$ are the logits of two independently augmented views of the same batch. |
| Class weighting | BCE with positive-class weight = N(real) / N(synthetic) of the training fold. |
| Augmentation (each view, training only) | (1) Smooth EQ along the within-period axis (FFT over the 256 samples): tilt U(−6, 6) dB × (normalized frequency − 0.5) plus two Gaussian bumps (centre U(0, 1), width U(0.05, 0.30), gain U(−5, 5) dB), clipped to ±10 dB; one curve per note, applied to all 48 rows. (2) Circular shift along the within-period axis by U{0, …, 255} samples (one shift per view and batch). (3) Per-row gain × (1 + 0.08·N(0, 1)). (4) Additive Gaussian noise, SD 0.02 of the input peak (input images have peak 1); the resulting SNR depends on the waveform's crest factor and was not computed. (5) Peak re-normalization. (6) Two masks, each of 0–7 consecutive rows set to zero (same rows for the whole batch). At inference only the peak normalization is applied. |
| Features (fusion) | 47 features (or the 16 Group-A medians) from the stored detector feature tables, computed with the detector's own (unconstrained) $f_0$; median imputation and standardization fitted on the training fold. |
| Optimizer | Adam, learning rate $10^{-3}$ (constant, no schedule), weight decay $10^{-4}$. |
| Batch, epochs | 24 files per batch, shuffled every epoch; 40 epochs. |
| Seeds | Each fit: `torch.manual_seed(0)`, `np.random.seed(0)`, and a `torch.Generator` seeded 0 for shuffling and augmentation. One seed per fit; no averaging over seeds. |
| Early stopping | None. No validation split. |
| Scored checkpoint | The final (epoch-40) weights, in evaluation mode (no augmentation, dropout off). No weights are stored; each job file holds P(synthetic) for every usable file. |
| Scoring | Corpus notes by the model of their own fold (out-of-fold). Library notes: `RESULTS_CNN_ISO.md` uses the mean of the five fold models (protocol v2); the values in the paper (Sec. 5.4) use the model of the library's own fold (protocol v3, `FINAL_NUMBERS.md`). |
| Intervals | Stratified grouped bootstrap (real and synthetic instrument groups resampled separately), B = 2000, seed 0. |

**Fusion training.** The fusion models are trained end to end by the same procedure: CNN and head are initialized together and trained jointly on the same loss and augmented views; the standardized features are concatenated to the pooled embedding before the head and are not augmented. There is no separate pre-training of the CNN and no stacking.

**Caveat on the Otey AUC.** Otey (one group, 120 notes) lies entirely in fold 2, which holds no CCMusic notes. Its AUC compares the fold-2 model's Otey scores with CCMusic scores from the other four fold models.

## S2.4 Diagnostics

**Training vs held-out AUC** (stored epoch-40 scores, constrained $f_0$, T2; mean and range over folds):

| Model | Comparison | Training AUC | Within-fold test AUC | Pooled out-of-fold AUC |
|---|---|---|---|---|
| CNN | CCMusic vs SoundFont | 0.984 (0.973–0.992) | 0.924 (0.873–0.957), 4 folds | 0.910 |
| CNN | CCMusic vs Otey | 0.983 (0.976–0.991), 4 folds | – | 0.776 |
| Fusion-47 | CCMusic vs SoundFont | 1.000 (0.99994–1.000) | 0.963 (0.928–0.987), 4 folds | 0.940 |
| Fusion-47 | CCMusic vs Otey | 1.000 (0.9997–1.000), 4 folds | – | 0.817 |

**Longer training and a wider network** (one fold, fold 0, 120 epochs at constant learning rate; same code, seeds and augmentation; evaluation every 2 epochs). Values: mean ± SD over the evaluation points in each epoch window; the SD is the epoch-to-epoch variation.

| Run | Epochs | Training AUC, SoundFont | Test AUC, SoundFont (Casio CA-110, Yamaha PSR-400/500) | Test AUC, Otey (unseen) | Clean training BCE |
|---|---|---|---|---|---|
| A: base net, fold 0 (1218 training files) | 30–40 | 0.977 ± 0.004 | 0.915 ± 0.020 | – | 0.200 |
| | 50–70 | 0.987 ± 0.005 | 0.928 ± 0.008 | – | 0.118 |
| | 80–120 | 0.994 ± 0.004 | 0.937 ± 0.014 | – | 0.082 |
| B: base net, Otey removed from training (1098 files) | 30–40 | 0.982 ± 0.003 | 0.937 ± 0.015 | 0.785 ± 0.024 | 0.180 |
| | 50–70 | 0.991 ± 0.004 | 0.949 ± 0.014 | 0.795 ± 0.022 | 0.107 |
| | 80–120 | 0.996 ± 0.002 | 0.951 ± 0.010 | 0.799 ± 0.022 | 0.075 |
| C: wide net (32-64-128-256 channels; CNN 421 696, total 454 721 parameters = 3.7 × base), Otey removed | 30–40 | 0.983 ± 0.002 | 0.933 ± 0.018 | 0.791 ± 0.023 | 0.170 |
| | 50–70 | 0.991 ± 0.004 | 0.939 ± 0.018 | 0.790 ± 0.021 | 0.116 |
| | 80–120 | 0.997 ± 0.002 | 0.935 ± 0.011 | 0.790 ± 0.022 | 0.081 |

Reproduction check of run A at epoch 40 against the stored fold-0 fit: Spearman 0.989 over 1550 files, mean |ΔP| 0.037 (CPU thread nondeterminism). Longer training changes held-out AUC by +0.01 to +0.02 (SoundFont 0.915→0.937 in A, 0.937→0.951 in B; Otey 0.785→0.799), the size of the epoch-to-epoch SD (0.008–0.024). The wide net does not improve on the base net (SoundFont 0.935 vs 0.951; Otey 0.790 vs 0.799, epochs 80–120).

**Sensitivity to the window position (listening-test clips, Test 1).** The clip processing (onset trim with 10-ms pre-roll, 10-ms fade-in) moves the phaseogram window relative to the attack. Median shift clip − source and correlation of the two phaseograms (best circular shift): Iowa +42 ms, r = 0.93; K18 +1 ms, r = 0.99; Salamander +6 ms, r = 1.00; SoundFont (Casio CA-110) 0 ms, r = 1.00; Otey 0 ms, r = 1.00. In the Iowa source files the recording noise lies within 45 dB of the peak, so the trim point precedes the attack; the clip removes that noise and the window starts about 40 ms later. Clips flagged (P > 0.5):

| CNN only | Iowa | K18 | Salamander | SoundFont | Otey | AUC recorded (20) vs algorithmic (10) |
|---|---|---|---|---|---|---|
| LOGO models, clips | 7/7 | 6/7 | 6/6 | 5/5 | 5/5 | 0.140 [0.040, 0.265] |
| LOGO models, unprocessed source notes | 2/7 | 5/7 | 6/6 | 5/5 | 5/5 | 0.430 [0.270, 0.580] |
| Fold models, clips | 6/7 | 6/7 | 6/6 | 5/5 | 5/5 | 0.905 [0.805, 0.980] |
| Fold models, unprocessed source notes | 1/7 | 5/7 | 6/6 | 5/5 | 4/5 | 0.890 [0.780, 0.980] |

The verdict on five of seven Iowa notes changes with the window shift. Fusion-47 (LOGO) on the clips: 7/20 recorded and 7/10 algorithmic flagged, AUC 0.600 [0.415, 0.820].

## S2.5 Compute

| Item | Value |
|---|---|
| Hardware | CPU only: Intel Core i9-9900K (8 cores). No GPU. |
| Software | Python 3.10.7, torch 2.0.1+cpu, numpy 1.26.4, pandas 1.5.2. |
| Phaseogram cache | 2579 files, 12 worker processes, 92 s. |
| Training (69 fits) | 6 worker processes × 2 torch threads; 2026-10-01 10:24–15:56 (5.52 h wall clock); 32.7 h of summed per-process time. A constrained-$f_0$ T2 fit took about 38 min. Summed per-process minutes: constrained CNN T2 196, T1 228, LOGO 118; fusion-47 T2 195, T1 221, LOGO 117; fusion-16 T2 184, T1 220, LOGO 115; det48 models 56–70 per model and condition. |
| Evaluation | 66 s. |
| Diagnostic runs (120 epochs, one fold) | Run A 73 min, run B 69 min, run C 96 min (CPU; thread count per run not recorded beyond the script default of 5). |
| Memory use | Not recorded. |
