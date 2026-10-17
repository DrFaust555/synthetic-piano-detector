# FINAL NUMBERS -- protocol v3, final run

Command: `python Stattia/revision/analysis/final_v3.py` (runs run_v3.py first). Own-fold scoring throughout; T2 unless stated. PRIMARY = CODE_AUDIT B1 files and every "exclude" row of audit/real_class_exclusions.csv removed, real performances grouped by group_id / video id (audit/real_class_exclusions.csv: 105 file(s) with decision exclude; 2275 real-performance groups). SENSITIVITY = also every "review" row removed (audit/real_class_exclusions.csv: 294 file(s) with decision exclude or review; 2089 groups). `*` = one synthetic group (no general interval). Runtime 1084 s.

Caveat: the screen changes only real performances, but the greedy GroupKFold assignment is recomputed over the whole corpus, so 5 of the 11 synthetic single-note groups (and the libraries tied to their folds) moved to other folds relative to the pre-screen v3 run (and differ again in the sensitivity corpus). All isolated-note changes versus pre-screen v3 (e.g. library mf 0.507 -> 0.621, Iowa mf 0.363 -> 0.608) are therefore fold-assignment variability, not an effect of the screen; they lie inside the grouped-bootstrap intervals.

## Table 7 (detection by format)

| row | files/groups | 47 feat. primary | 47 feat. sensitivity | 16 med. primary | 16 med. sensitivity | 47 feat. T1 primary |
|---|---|---|---|---|---|---|
| Iso: SoundFont | 678/10 | 0.937 [0.87, 0.98]; det 84.1% | 0.924 [0.84, 0.97]; det 79.9% | 0.877 | 0.852 | 0.867 |
| Iso: Otey | 120/1 | 0.821*; det 34.2% | 0.773*; det 31.7% | 0.675* | 0.587* | 0.760* |
| Iso: algorithmic (both) | 798/11 | 0.920 [0.84, 0.98]; det 76.6% | 0.901 [0.81, 0.97]; det 72.7% | 0.846 | 0.812 | 0.851 |
| Iso: library notes, mf (224) | 224/4 | 0.621 [0.48, 0.75]; det 15.2% | 0.579 [0.43, 0.72]; det 15.6% | 0.528 | 0.509 | 0.714 |
| Iso: library notes, all layers | 999/4 | 0.645 [0.48, 0.78]; det 19.3% | 0.620 [0.44, 0.77]; det 17.1% | 0.592 | 0.555 | 0.691 |
| Iso: Salamander mf | 30/1 | 0.699*; det 30.0% | 0.705*; det 16.7% | 0.615* | 0.568* | 0.653* |
| Iso: Iowa mf | 86/1 | 0.608*; det 9.3% | 0.501*; det 5.8% | 0.461* | 0.417* | 0.757* |
| Iso: K18 mf | 30/1 | 0.555*; det 16.7% | 0.602*; det 23.3% | 0.559* | 0.606* | 0.699* |
| Iso: FreePats mf | 78/1 | 0.629*; det 15.4% | 0.607*; det 23.1% | 0.557* | 0.550* | 0.694* |
| Perf: SoundFont renders (P1a) | 100/10 | 0.947 [0.90, 0.98]; det 44.0% | 0.950 [0.91, 0.98]; det 48.0% | 0.756 | 0.768 | 0.949 |
| Perf: Otey renders (P1b) | 20/1 | 0.923*; det 0.0% | 0.951*; det 5.0% | 0.558* | 0.604* | 0.979* |
| Perf: algorithmic renders (P1) | 120/11 | 0.943 [0.91, 0.98]; det 36.7% | 0.951 [0.92, 0.98]; det 40.8% | 0.723 | 0.741 | 0.954 |
| Perf: sampled digital (P2) | 128/7 | 0.871 [0.84, 0.89]; det 7.0% | 0.884 [0.84, 0.92]; det 12.5% | 0.562 | 0.580 | 0.953 |
| Perf: sampled digital, long tails | 10/6 | 0.864 [0.84, 0.90]; det 10.0% | 0.848 [0.75, 0.93]; det 10.0% | 0.507 | 0.494 | 0.976 |
| Perf: sampled digital, no long tails | 109/7 | 0.868 [0.84, 0.89]; det 7.3% | 0.887 [0.84, 0.92]; det 12.8% | 0.563 | 0.583 | 0.951 |
| Perf: digital, line capture | 12/6 | 0.913 [0.85, 0.95]; det 8.3% | 0.886 [0.72, 0.98]; det 16.7% | 0.594 | 0.580 | 0.979 |
| Perf: modelled digital (P3) | 21/1 | 0.854*; det 9.5% | 0.856*; det 9.5% | 0.583* | 0.642* | 0.870* |
| Perf: Korg M1 (P4) | 30/1 | 0.928*; det 23.3% | 0.925*; det 30.0% | 0.874* | 0.806* | 0.874* |

False alarms: CCMusic 11.5% (sens. 14.2%), med16 17.7%; T1 29.0%. Real performances 0.2% (sens. 0.5%), med16 1.7%. Real performances: 2471 files primary, 2282 sensitivity.
Formats mixed (corpus-level detector, T2): real performances vs algorithmic single notes 0.995 [0.99, 1.00]; det 85.3% (sens. 0.993 [0.99, 1.00]); med16 0.958 (sens. 0.952).

## 5.1 Format

SoundFont single-note files with all 16 IQR = 0: 555 of 762. Real performances: 7.3% of IQR values zero (2885 of 39536); files with all 16 zero: 0; files with an undefined Group B value: 725 (1321 values); with an undefined Group C value: 3 (9 values). Sensitivity: 7.3% zero, 0 all-zero, 666 / 2 files with undefined B / C.

## 5.2 Isolated notes (extras)

Instrument level (median P per instrument, exact Mann-Whitney): 8 pianos vs 4 libraries p = 0.214 (P = 0.75); 8 pianos vs 10 SoundFonts p = 4.6e-05 (P = 1.00).

f0 / register restriction (evaluation restricted to files with a correct primary f0 and nominal MIDI 48-76; remaining: CCMusic 211, SoundFont 246, Otey 40, library notes 319): SoundFont: 0.937 -> 0.943 (change +0.006); Otey: 0.821* -> 0.789* (change -0.032); library mf: 0.621 -> 0.624 (change +0.004); library all: 0.645 -> 0.599 (change -0.046).

eGeMAPS (88 features, same XGBoost, v3 folds): algorithmic 0.948 [0.83, 1.00]; library notes mf own-fold 0.704 [0.47, 0.89] (v2 rule 0.759); all layers own-fold 0.617 [0.39, 0.91] (v2 rule 0.727).

## 5.3 Controls (Table 8; primary corpus)

| source | original | 1.2 s | 4 s | room+noise | 1.2 s+room | 4 s+room |
|---|---|---|---|---|---|---|
| SoundFont | 0.937 (84.1%) | 0.819 (65.9%) | 0.840 (73.0%) | 0.923 (88.8%) | 0.861 (81.3%) | 0.940 (92.6%) |
| SoundFont, unaltered sample | 0.931 (83.3%) | 0.818 (67.3%) | 0.826 (65.3%) | 0.912 (85.6%) | 0.850 (79.4%) | 0.929 (90.2%) |
| Otey | 0.821 (34.2%) | 0.645 (40.7%) | 0.780 (63.3%) | 0.699 (34.5%) | 0.759 (64.2%) | 0.831 (75.2%) |
| library notes, mf layer | 0.621 (15.2%) | 0.658 (35.7%) | 0.621 (34.4%) | 0.585 (21.9%) | 0.624 (41.1%) | 0.642 (39.3%) |
| false alarms | 11.5% | 21.7% | 22.9% | 20.1% | 25.1% | 20.8% |
| Otey, 16 medians | 0.675 | 0.573 | 0.670 | 0.520 | 0.654 | 0.820 |

Loop index (C5, share r > 0.9999): CCMusic (sounding at 4 s) 0%, library Iowa (sounding at 4 s) 0%, library K18 (sounding at 4 s) 2%, library Salamander (sounding at 4 s) 0%, library FreePats (sounding at 4 s) 0%, Otey 6%, SF CP80_16_44100 11%, SF CTK_230_SoundFont 100%, SF CTK_533_Piano1_00_Casio 46%, SF Casio_CA_110_Piano_FM 84%, SF Casio_PX_860_Concert_Grand_Piano 0%, SF JV1080_Nice_Piano_m 12%, SF U20PIANO 0%, SF YAMAHA_PSR400_500 84%, SF Yamaha_PSRF50 0%, SF Yamaha_S90_ES_Natural_Grand 0%.

## 5.4 Phaseogram and CNN (Table 9)

| source | notes | irregularity median [IQR] | with room+noise | row-to-row r |
|---|---|---|---|---|
| real | 50 | 1.355 [0.66, 2.80] | 1.87 | 0.980 |
| samplelib | 50 | 0.769 [0.49, 1.25] | 0.63 | 0.993 |
| soundfont | 50 | 0.288 [0.20, 0.42] | 0.38 | 0.993 |
| otey | 40 | 0.054 [0.03, 0.23] | 0.18 | 0.998 |

Real vs Otey AUC 0.95 (p = 2e-13); vs SoundFont 0.91; vs libraries 0.68 (p = 0.0022). Real early / late half: 1.78 / 0.62. After room+noise: P(class > real) SoundFont 0.23, Otey 0.12.

CNN / fusion (isolated training, T2, constrained f0), own-fold: cnn: algorithmic 0.892, library mf 0.698 [0.53, 0.87] (v2 rule 0.713); fus47: algorithmic 0.923, library mf 0.543 [0.37, 0.73] (v2 rule 0.619); fus16: algorithmic 0.907, library mf 0.567 [0.37, 0.78] (v2 rule 0.609). CNN per library (mf, own-fold): Salamander 0.86, FreePats 0.78, K18 0.79, Iowa 0.54.

## 5.5 Performances (extras)

Bass fundamental share: real performances 0.027, sampled digital 0.083, renders 0.081, CCMusic 0.112; bass log(h2/h1): digital 0.06 vs real top-10% / bottom quarter 0.47 / 0.61. Detuning: digital_v2 5.67 vs top-10%-rated real 6.71; tuning-only performance detector vs top-10% real: 0.668 [0.62, 0.71]. Dry renders effect (corpus detector, T2, own-fold): P2 0.644 [0.58, 0.68] -> 0.860 [0.82, 0.89] (sens. 0.646 -> 0.871).

## 5.6 Listening clips (Table 10 detector column; isolated-note detector, leave-own-group-out)

all47: Iowa 3/7, K18 4/7, Salamander 3/6, SoundFont 2/5, Otey 3/5, AUC 0.575; med16: Iowa 0/7, K18 3/7, Salamander 2/6, SoundFont 2/5, Otey 2/5, AUC 0.605. Listener numbers: unchanged (RESULTS_v2 section 6).

## Over 10 fold assignments (primary corpus)

Seeds 0-9 of the same greedy GroupKFold rule (largest group first, ties broken in a seed-permuted order); seed 0 = the assignment used everywhere above. Each cell: median [min-max] over the seeds; the seed-0 column gives the value with its grouped-bootstrap 95% CI. Per-seed values: `out_v3/fold_seed_sweep.csv`. The listening-clip detector (section 5.6) is trained on all isolated notes except the clip's own group, so it does not use folds: its values are the same for every seed.

### Table 7

| row | 47 feat., seed 0 [CI] | 47 feat., median [min-max] | 16 med., seed 0 | 16 med., median [min-max] |
|---|---|---|---|---|
| Iso: SoundFont | 0.937 [0.87, 0.98] | 0.930 [0.919-0.937] | 0.877 | 0.862 [0.849-0.877] |
| Iso: Otey | 0.821* | 0.813 [0.783-0.831] | 0.675 | 0.631 [0.580-0.685] |
| Iso: algorithmic (both) | 0.920 [0.84, 0.98] | 0.913 [0.900-0.920] | 0.846 | 0.826 [0.815-0.847] |
| Iso: library notes, mf | 0.621 [0.48, 0.75] | 0.545 [0.519-0.621] | 0.528 | 0.473 [0.412-0.528] |
| Iso: library notes, all layers | 0.645 [0.48, 0.78] | 0.614 [0.572-0.645] | 0.592 | 0.574 [0.543-0.607] |
| Iso: Salamander mf | 0.699* | 0.700 [0.630-0.746] | 0.615 | 0.632 [0.578-0.687] |
| Iso: Iowa mf | 0.608* | 0.387 [0.359-0.608] | 0.461 | 0.264 [0.218-0.461] |
| Iso: K18 mf | 0.555* | 0.627 [0.555-0.638] | 0.559 | 0.599 [0.559-0.636] |
| Iso: FreePats mf | 0.629* | 0.613 [0.575-0.673] | 0.557 | 0.556 [0.451-0.638] |
| Perf: P1a | 0.947 [0.90, 0.98] | 0.945 [0.923-0.956] | 0.756 | 0.758 [0.730-0.786] |
| Perf: P1b | 0.923* | 0.942 [0.864-0.967] | 0.558 | 0.710 [0.505-0.787] |
| Perf: P1 | 0.943 [0.91, 0.98] | 0.944 [0.913-0.958] | 0.723 | 0.750 [0.692-0.783] |
| Perf: P2 | 0.871 [0.84, 0.89] | 0.863 [0.848-0.877] | 0.562 | 0.561 [0.514-0.601] |
| Perf: P2d-lt | 0.864 [0.84, 0.90] | 0.857 [0.784-0.872] | 0.507 | 0.493 [0.397-0.532] |
| Perf: P2d-nolt | 0.868 [0.84, 0.89] | 0.862 [0.842-0.874] | 0.563 | 0.562 [0.518-0.603] |
| Perf: P2d-line | 0.913 [0.85, 0.95] | 0.914 [0.857-0.951] | 0.594 | 0.599 [0.507-0.681] |
| Perf: P3 | 0.854* | 0.850 [0.781-0.890] | 0.583 | 0.637 [0.516-0.679] |
| Perf: P4 | 0.928* | 0.909 [0.875-0.928] | 0.874 | 0.848 [0.712-0.878] |
| False alarms, CCMusic | 11.5% | 12.6% [10.3%-13.0%] | 17.7% | 18.0% [16.0%-20.8%] |
| False alarms, real performances | 0.2% | 0.3% [0.2%-0.6%] | 1.7% | 1.7% [1.2%-2.0%] |

### Table 8 controls (47 features), median [min-max]

| source | original | 1.2 s | 4 s | room+noise | 1.2 s+room | 4 s+room |
|---|---|---|---|---|---|---|
| SoundFont | 0.930 [0.919-0.937] | 0.814 [0.784-0.824] | 0.822 [0.788-0.840] | 0.913 [0.901-0.923] | 0.847 [0.834-0.863] | 0.924 [0.915-0.940] |
| SoundFont, unaltered sample | 0.925 [0.911-0.935] | 0.788 [0.730-0.818] | 0.800 [0.771-0.826] | 0.895 [0.873-0.912] | 0.822 [0.798-0.850] | 0.907 [0.882-0.929] |
| Otey | 0.813 [0.783-0.831] | 0.641 [0.615-0.668] | 0.755 [0.737-0.783] | 0.671 [0.625-0.702] | 0.733 [0.700-0.759] | 0.796 [0.767-0.831] |
| library notes, mf layer | 0.545 [0.519-0.621] | 0.560 [0.547-0.658] | 0.537 [0.475-0.621] | 0.548 [0.516-0.589] | 0.559 [0.548-0.644] | 0.607 [0.569-0.642] |
| library notes, all layers | 0.614 [0.572-0.645] | 0.648 [0.612-0.710] | 0.653 [0.602-0.683] | 0.562 [0.525-0.584] | 0.635 [0.608-0.665] | 0.652 [0.629-0.670] |
| CCMusic FA | 12.6% [10.3%-13.0%] | 22.0% [20.7%-25.2%] | 24.7% [21.3%-29.5%] | 20.3% [19.6%-23.4%] | 25.5% [23.4%-28.1%] | 23.6% [20.8%-26.1%] |

### Instrument-level tests, eGeMAPS, clips

| number | seed 0 | median [min-max] |
|---|---|---|
| 8 pianos vs 4 libraries | p = 0.214, P = 0.75 | p 0.46 [0.21-0.57]; P 0.656 [0.625-0.750] |
| 8 pianos vs 10 SoundFonts | p = 4.57e-05, P = 1.00 | p 4.6e-05 [4.6e-05-9.1e-05]; P 1.000 [0.988-1.000] |
| eGeMAPS: library notes, mf | 0.704 [0.47, 0.89] | 0.694 [0.670-0.729] |
| eGeMAPS: library notes, all layers | 0.617 [0.39, 0.91] | 0.662 [0.615-0.733] |
| eGeMAPS: algorithmic | 0.948 [0.83, 1.00] | 0.943 [0.937-0.948] |
| Listening clips, detector AUC (all47) | 0.575 | fold-independent (same for every seed) |
| Listening clips, detector AUC (med16) | 0.605 | fold-independent (same for every seed) |
