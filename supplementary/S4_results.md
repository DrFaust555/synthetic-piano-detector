# S4. All result tables with intervals and per-source values

## S4.0 Conventions

- **Protocol.** Protocol v3 (final run, `analysis/out_v3/FINAL_NUMBERS.md`, `final_numbers.json`, `RESULTS_v3.md`), protocol v4 (enlarged recorded class, `analysis/v4/RESULTS_v4.md`), the format-matched CNN study (`cnn_iso/`) and the eGeMAPS baseline (`egemaps/`). Values from protocol v2 (`analysis/out/RESULTS_v2.md`) appear only where v3 has no replacement and are marked "v2".
- **Detector.** XGBoost, 400 trees, depth 4, learning rate 0.05, subsample 0.8, column subsample 0.8, `scale_pos_weight` = real/synthetic ratio of the training population. One detector per format. T2 (primary): sample-based and separate sources never in training. T1: those sources added to training, labelled synthetic (shown for reference only).
- **Scoring.** Every file is scored by the model of its own fold ("own-fold"). Sources never trained are assigned to one fold with their group.
- **AUC** = real files of the same format vs the source (0.5 = chance; < 0.5 = scored as more real than real). **Det.** = share of the source with P(synthetic) > 0.5. **FA** = share of real files with P > 0.5.
- **Ten fold assignments.** Seeds 0–9 of the same greedy GroupKFold rule; "median [min–max]" is over these ten. Seed 0 is the assignment used for every bootstrap interval and every detection rate; detection rates were recorded only for seed 0 (the ten-assignment sweep stores AUC and FA only).
- **Bootstrap 95% interval.** Stratified grouped bootstrap over instrument groups (real and synthetic groups resampled separately), B = 2000, seed 0. **\*** = one synthetic group: the interval resamples only the real groups and does not generalize to other instruments of that kind.
- **Corpora.** Primary: audit exclusions removed, 2471 real performances in 2275 video groups. Sensitivity: also every "review" row removed, 2282 real performances in 2089 groups.
- Paper tables report the ten-assignment medians rounded to two decimals.

## S4.1 Format

Mixed-format detector (corpus level, T2, seed 0): real performances vs algorithmic single notes.

| Feature set | Primary corpus | Sensitivity corpus |
|---|---|---|
| 47 features | 0.995 [0.991, 0.999]; det. 85.3%; FA 0.2% | 0.993 [0.989, 0.997]; det. 80.4%; FA 0.3% |
| 16 medians | 0.958 [0.924, 0.987]; det. 76.3%; FA 1.9% | 0.952 [0.913, 0.986]; det. 71.0%; FA 1.7% |

Feature availability: 555 of 762 SoundFont single-note files have all 16 inter-quartile ranges equal to zero. In real performances 7.3% of the inter-quartile values are zero (2885 of 39 536), no file has all 16 zero, 725 files have at least one undefined Group-B value (1321 values) and 3 files an undefined Group-C value (9 values).

## S4.2 Detection by format (Table 4 extended)

Table S4.1. Isolated notes (668 CCMusic notes, 6 groups). AUC median [min–max] over ten fold assignments; seed-0 AUC [bootstrap 95% interval]; det. (seed 0). T1 = libraries in training labelled synthetic.

| CCMusic notes vs | Files/groups | 47 feat.: median [range] | 47 feat.: seed 0 [CI] | Det. | 16 med.: median [range] | 16 med.: seed 0 [CI] | Det. | 47 feat. T1: seed 0 [CI]; det. |
|---|---|---|---|---|---|---|---|---|
| SoundFont (10 instruments) | 678/10 | 0.930 [0.919–0.937] | 0.937 [0.87, 0.98] | 84.1% | 0.862 [0.849–0.877] | 0.877 [0.77, 0.96] | 76.8% | 0.867 [0.79, 0.94]; 90.4% |
| Otey model | 120/1 | 0.813 [0.783–0.831] | 0.821 [0.70, 0.93]\* | 34.2% | 0.631 [0.580–0.685] | 0.675 [0.57, 0.78]\* | 23.3% | 0.760 [0.66, 0.87]; 74.2% |
| Algorithmic (both) | 798/11 | 0.913 [0.900–0.920] | 0.920 [0.84, 0.98] | 76.6% | 0.826 [0.815–0.847] | 0.846 [0.73, 0.94] | 68.8% | 0.851 [0.76, 0.94]; 88.0% |
| Library notes, mf layer | 224/4 | 0.545 [0.519–0.621] | 0.621 [0.48, 0.75] | 15.2% | 0.473 [0.412–0.528] | 0.528 [0.42, 0.65] | 17.0% | 0.714 [0.61, 0.81]; 62.5% |
| Library notes, all layers | 999/4 | 0.614 [0.572–0.645] | 0.645 [0.48, 0.78] | 19.3% | 0.574 [0.543–0.607] | 0.592 [0.44, 0.71] | 26.8% | 0.691 [0.61, 0.78]; 58.0% |
| Salamander, mf | 30/1 | 0.700 [0.630–0.746] | 0.699 [0.57, 0.82]\* | 30.0% | 0.632 [0.578–0.687] | 0.615 [0.52, 0.71]\* | 23.3% | 0.653 [0.57, 0.74]; 46.7% |
| Iowa, mf | 86/1 | 0.387 [0.359–0.608] | 0.608 [0.47, 0.75]\* | 9.3% | 0.264 [0.218–0.461] | 0.461 [0.37, 0.56]\* | 12.8% | 0.757 [0.67, 0.85]; 65.1% |
| K18, mf | 30/1 | 0.627 [0.555–0.638] | 0.555 [0.44, 0.67]\* | 16.7% | 0.599 [0.559–0.636] | 0.559 [0.46, 0.66]\* | 23.3% | 0.699 [0.61, 0.79]; 63.3% |
| FreePats (single layer) | 78/1 | 0.613 [0.575–0.673] | 0.629 [0.51, 0.75]\* | 15.4% | 0.556 [0.451–0.638] | 0.557 [0.46, 0.66]\* | 16.7% | 0.694 [0.60, 0.79]; 65.4% |
| Salamander, all layers | 480/1 | 0.709 [0.639–0.738] | 0.714 [0.59, 0.83]\* | 25.6% | 0.678 [0.622–0.717] | 0.657 [0.57, 0.75]\* | 36.5% | 0.692 [0.61, 0.78]; 56.7% |
| Iowa, all layers | 261/1 | 0.402 [0.379–0.576] | 0.576 [0.44, 0.71]\* | 10.0% | 0.323 [0.283–0.478] | 0.461 [0.37, 0.56]\* | 10.0% | 0.704 [0.62, 0.79]; 59.8% |
| K18, all layers | 180/1 | 0.638 [0.568–0.661] | 0.568 [0.45, 0.68]\* | 17.8% | 0.650 [0.606–0.678] | 0.627 [0.53, 0.73]\* | 30.0% | 0.671 [0.59, 0.76]; 55.6% |
| FA on CCMusic notes | 668/6 | 12.6% [10.3–13.0] | 11.5% | | 18.0% [16.0–20.8] | 17.7% | | 29.0% |

SoundFont including the 84 velocity copies (47 features): median 0.931 [0.919–0.942]; seed 0 0.942 [0.88, 0.98], det. 86.2%.

Table S4.2. Performances (2471 real performances, 2275 groups). Same columns. Digital pianos are grouped by brand.

| Real performances vs | Files/groups | 47 feat.: median [range] | 47 feat.: seed 0 [CI] | Det. | 16 med.: median [range] | 16 med.: seed 0 [CI] | Det. | 47 feat. T1: seed 0 [CI]; det. |
|---|---|---|---|---|---|---|---|---|
| SoundFont renders | 100/10 | 0.945 [0.923–0.956] | 0.947 [0.90, 0.98] | 44.0% | 0.758 [0.730–0.786] | 0.756 [0.64, 0.86] | 11.0% | 0.949 [0.92, 0.97]; 55.0% |
| Otey renders | 20/1 | 0.942 [0.864–0.967] | 0.923 [0.91, 0.93]\* | 0.0% | 0.710 [0.505–0.787] | 0.558 [0.54, 0.57]\* | 0.0% | 0.979 [0.98, 0.98]; 75.0% |
| Algorithmic renders (both) | 120/11 | 0.944 [0.913–0.958] | 0.943 [0.91, 0.98] | 36.7% | 0.750 [0.692–0.783] | 0.723 [0.62, 0.84] | 9.2% | 0.954 [0.93, 0.98]; 58.3% |
| Digital pianos, sampled (all) | 128/7 | 0.863 [0.848–0.877] | 0.871 [0.84, 0.89] | 7.0% | 0.561 [0.514–0.601] | 0.562 [0.53, 0.62] | 1.6% | 0.953 [0.94, 0.97]; 60.2% |
| — older demonstrations | 9/4 | 0.921 [0.882–0.938] | 0.916 [0.89, 0.95] | 0.0% | 0.623 [0.531–0.675] | 0.621 [0.47, 0.86] | 0.0% | 0.955 [0.92, 0.98]; 44.4% |
| — recent demonstrations | 119/7 | 0.859 [0.842–0.872] | 0.868 [0.84, 0.89] | 7.6% | 0.556 [0.507–0.596] | 0.558 [0.53, 0.61] | 1.7% | 0.953 [0.94, 0.97]; 61.3% |
| — recent, line output | 12/6 | 0.914 [0.857–0.951] | 0.913 [0.85, 0.95] | 8.3% | 0.599 [0.507–0.681] | 0.594 [0.47, 0.72] | 0.0% | 0.979 [0.97, 0.99]; 66.7% |
| — recent, microphone | 2/2 | 0.968 [0.946–0.976] | 0.964 [0.95, 0.98] | 0.0% | 0.789 [0.495–0.919] | 0.495 [0.13, 0.86] | 0.0% | 0.986 [0.97, 1.00]; 100% |
| — recent, capture unknown | 105/7 | 0.849 [0.831–0.864] | 0.861 [0.83, 0.89] | 7.6% | 0.547 [0.507–0.589] | 0.555 [0.52, 0.60] | 1.9% | 0.949 [0.93, 0.96]; 60.0% |
| — recent, long reverberant tails | 10/6 | 0.857 [0.784–0.872] | 0.864 [0.84, 0.90] | 10.0% | 0.493 [0.397–0.532] | 0.507 [0.23, 0.65] | 0.0% | 0.976 [0.97, 0.99]; 70.0% |
| — recent, no long tails | 109/7 | 0.862 [0.842–0.874] | 0.868 [0.84, 0.89] | 7.3% | 0.562 [0.518–0.603] | 0.563 [0.53, 0.62] | 1.8% | 0.951 [0.94, 0.96]; 60.6% |
| Digital pianos, modelled (Roland) | 21/1 | 0.850 [0.781–0.890] | 0.854 [0.85, 0.86]\* | 9.5% | 0.637 [0.516–0.679] | 0.583 [0.57, 0.60]\* | 0.0% | 0.870 [0.86, 0.88]; 57.1% |
| — recent demonstrations | 19/1 | 0.856 [0.786–0.894] | 0.863 [0.85, 0.87]\* | 10.5% | 0.635 [0.509–0.671] | 0.592 [0.58, 0.61]\* | 0.0% | 0.891 [0.88, 0.90]; 57.9% |
| Korg M1 | 30/1 | 0.909 [0.875–0.928] | 0.928 [0.92, 0.93]\* | 23.3% | 0.848 [0.712–0.878] | 0.874 [0.87, 0.88]\* | 16.7% | 0.874 [0.87, 0.88]; 43.3% |
| Roland GP-3 (hybrid; not in Table 1) | 1/1 | 0.955 [0.870–0.991] | 0.991 [0.99, 0.99]\* | 0.0% | 0.895 [0.435–0.968] | 0.860 [0.85, 0.87]\* | 0.0% | 0.987 [0.98, 0.99]; 100% |
| Roland SC-55 module (not in Table 1) | 2/1 | 0.998 [0.987–1.000] | 1.000 [1.00, 1.00]\* | 100% | 0.821 [0.669–0.898] | 0.783 [0.77, 0.80]\* | 0.0% | 0.992 [0.99, 0.99]; 100% |
| FA on real performances | 2471/2275 | 0.3% [0.2–0.6] | 0.2% | | 1.7% [1.2–2.0] | 1.7% | | 3.4% |

Table S4.3. Primary vs sensitivity corpus (review rows also removed): seed-0 AUC [CI]; det.

| Row | Primary, 47 feat. | Sensitivity, 47 feat. | Primary, 16 med. | Sensitivity, 16 med. |
|---|---|---|---|---|
| Iso: SoundFont | 0.937 [0.87, 0.98]; 84.1% | 0.924 [0.84, 0.97]; 79.9% | 0.877 | 0.852 |
| Iso: Otey | 0.821\*; 34.2% | 0.773\*; 31.7% | 0.675\* | 0.587\* |
| Iso: algorithmic | 0.920 [0.84, 0.98]; 76.6% | 0.901 [0.81, 0.97]; 72.7% | 0.846 | 0.812 |
| Iso: library mf | 0.621 [0.48, 0.75]; 15.2% | 0.579 [0.43, 0.72]; 15.6% | 0.528 | 0.509 |
| Iso: library all layers | 0.645 [0.48, 0.78]; 19.3% | 0.620 [0.44, 0.77]; 17.1% | 0.592 | 0.555 |
| Iso: Salamander / Iowa / K18 / FreePats mf | 0.699 / 0.608 / 0.555 / 0.629 | 0.705 / 0.501 / 0.602 / 0.607 | 0.615 / 0.461 / 0.559 / 0.557 | 0.568 / 0.417 / 0.606 / 0.550 |
| Perf: SoundFont | 0.947 [0.90, 0.98]; 44.0% | 0.950 [0.91, 0.98]; 48.0% | 0.756 | 0.768 |
| Perf: Otey | 0.923\*; 0.0% | 0.951\*; 5.0% | 0.558\* | 0.604\* |
| Perf: algorithmic | 0.943 [0.91, 0.98]; 36.7% | 0.951 [0.92, 0.98]; 40.8% | 0.723 | 0.741 |
| Perf: digital, sampled | 0.871 [0.84, 0.89]; 7.0% | 0.884 [0.84, 0.92]; 12.5% | 0.562 | 0.580 |
| Perf: digital, long tails | 0.864 [0.84, 0.90]; 10.0% | 0.848 [0.75, 0.93]; 10.0% | 0.507 | 0.494 |
| Perf: digital, no long tails | 0.868 [0.84, 0.89]; 7.3% | 0.887 [0.84, 0.92]; 12.8% | 0.563 | 0.583 |
| Perf: digital, line output | 0.913 [0.85, 0.95]; 8.3% | 0.886 [0.72, 0.98]; 16.7% | 0.594 | 0.580 |
| Perf: digital, modelled | 0.854\*; 9.5% | 0.856\*; 9.5% | 0.583\* | 0.642\* |
| Perf: Korg M1 | 0.928\*; 23.3% | 0.925\*; 30.0% | 0.874\* | 0.806\* |
| FA, CCMusic / real performances | 11.5% / 0.2% | 14.2% / 0.5% | 17.7% / 1.7% | not recorded |

The sensitivity corpus was evaluated for seed 0 only; ten-assignment ranges: not computed.

## S4.3 Instrument-level tests and register restriction

Table S4.4. Instrument level: median P(synthetic) per instrument, exact two-sided Mann–Whitney test; P = probability that the second set ranks above the pianos.

| Comparison | Seed 0 | Median [min–max] over ten assignments |
|---|---|---|
| 8 CCMusic pianos vs 4 libraries | p = 0.214, P = 0.75 | p = 0.46 [0.21–0.57]; P = 0.656 [0.625–0.750] |
| 8 CCMusic pianos vs 10 SoundFonts | p = 4.6 × 10⁻⁵, P = 1.00 | p = 4.6 × 10⁻⁵ [4.6 × 10⁻⁵–9.1 × 10⁻⁵]; P = 1.000 [0.988–1.000] |

Table S4.5. f₀/register restriction (seed 0, 47 features): evaluation restricted to files with correct primary f₀ and nominal MIDI 48–76 (C3–E5); same models.

| CCMusic notes vs | All files: n, AUC [CI], det. | Restricted: n real / n source, AUC [CI], det. | Change |
|---|---|---|---|
| SoundFont | 668/678; 0.937 [0.87, 0.98]; 84.1% | 211/246; 0.943 [0.86, 0.99]; 88.2% | +0.006 |
| Otey | 668/120; 0.821 [0.70, 0.93]\*; 34.2% | 211/40; 0.789 [0.65, 0.92]\*; 40.0% | −0.032 |
| Library notes, mf | 668/224; 0.621 [0.48, 0.75]; 15.2% | 211/72; 0.624 [0.47, 0.78]; 19.4% | +0.004 |
| Library notes, all layers | 668/999; 0.645 [0.48, 0.78]; 19.3% | 211/319; 0.599 [0.45, 0.75]; 16.6% | −0.046 |

FA on CCMusic: 11.5% (all), 11.8% (restricted). Ten-assignment ranges: not computed.

## S4.4 Controls (isolated notes)

Conditions: original; C1a = cut to 1.2 s (+0.1-s fade); C1b = key held 5 s, cut to 4 s (real notes kept only if still sounding at 4 s: CCMusic 380/668, libraries 874/999); C3 = measured room response (MIT IR Survey, 270 responses) + white noise 40 dB below the note RMS; C1a+C3, C1b+C3 = room first, then the cut.

Table S4.6. AUC against CCMusic notes, 47 features: median [min–max] over ten assignments.

| Source | Original | 1.2 s | 4 s | Room | 1.2 s + room | 4 s + room |
|---|---|---|---|---|---|---|
| SoundFont | 0.930 [0.919–0.937] | 0.814 [0.784–0.824] | 0.822 [0.788–0.840] | 0.913 [0.901–0.923] | 0.847 [0.834–0.863] | 0.924 [0.915–0.940] |
| — unaltered sample | 0.925 [0.911–0.935] | 0.788 [0.730–0.818] | 0.800 [0.771–0.826] | 0.895 [0.873–0.912] | 0.822 [0.798–0.850] | 0.907 [0.882–0.929] |
| Otey | 0.813 [0.783–0.831] | 0.641 [0.615–0.668] | 0.755 [0.737–0.783] | 0.671 [0.625–0.702] | 0.733 [0.700–0.759] | 0.796 [0.767–0.831] |
| Library notes, mf | 0.545 [0.519–0.621] | 0.560 [0.547–0.658] | 0.537 [0.475–0.621] | 0.548 [0.516–0.589] | 0.559 [0.548–0.644] | 0.607 [0.569–0.642] |
| Library notes, all layers | 0.614 [0.572–0.645] | 0.648 [0.612–0.710] | 0.653 [0.602–0.683] | 0.562 [0.525–0.584] | 0.635 [0.608–0.665] | 0.652 [0.629–0.670] |
| FA on CCMusic | 12.6% [10.3–13.0] | 22.0% [20.7–25.2] | 24.7% [21.3–29.5] | 20.3% [19.6–23.4] | 25.5% [23.4–28.1] | 23.6% [20.8–26.1] |

Table S4.7. Same, seed 0: AUC [bootstrap CI]; det. (n). 47 features.

| Source | Original | 1.2 s | 4 s | Room | 1.2 s + room | 4 s + room |
|---|---|---|---|---|---|---|
| SoundFont | 0.937 [0.870, 0.983]; 84.1% (678) | 0.819 [0.731, 0.900]; 65.9% (678) | 0.840 [0.770, 0.897]; 73.0% (674) | 0.923 [0.837, 0.978]; 88.8% (677) | 0.861 [0.783, 0.933]; 81.3% (675) | 0.940 [0.907, 0.968]; 92.6% (678) |
| — unaltered sample | 0.931 [0.862, 0.981]; 83.3% (168) | 0.818 [0.734, 0.897]; 67.3% (171) | 0.826 [0.718, 0.902]; 65.3% (150) | 0.912 [0.827, 0.975]; 85.6% (167) | 0.850 [0.765, 0.927]; 79.4% (170) | 0.929 [0.888, 0.968]; 90.2% (153) |
| Otey | 0.821 [0.703, 0.926]\*; 34.2% (120) | 0.645 [0.575, 0.726]\*; 40.7% (118) | 0.780 [0.702, 0.810]\*; 63.3% (120) | 0.699 [0.551, 0.825]\*; 34.5% (119) | 0.759 [0.669, 0.848]\*; 64.2% (120) | 0.831 [0.773, 0.881]\*; 75.2% (117) |
| Library notes, all layers | 0.645 [0.484, 0.783]; 19.3% (999) | 0.710 [0.517, 0.833]; 48.4% (999) | 0.683 [0.529, 0.798]; 45.8% (874) | 0.584 [0.439, 0.714]; 22.8% (999) | 0.663 [0.486, 0.793]; 47.6% (999) | 0.670 [0.561, 0.754]; 44.9% (874) |
| Library notes, mf | 0.621 [0.482, 0.753]; 15.2% (224) | 0.658 [0.557, 0.769]; 35.7% (224) | 0.621 [0.524, 0.770]; 34.4% (183) | 0.585 [0.440, 0.716]; 21.9% (224) | 0.624 [0.501, 0.745]; 41.1% (224) | 0.642 [0.562, 0.747]; 39.3% (183) |
| FA on CCMusic | 11.5% (668) | 21.7% (667) | 22.9% (380) | 20.1% (667) | 25.1% (666) | 20.8% (380) |

Table S4.8. Same, 16 medians, seed 0: AUC [CI]; det.

| Source | Original | 1.2 s | 4 s | Room | 1.2 s + room | 4 s + room |
|---|---|---|---|---|---|---|
| SoundFont | 0.877 [0.773, 0.958]; 76.8% | 0.735 [0.642, 0.825]; 53.8% | 0.731 [0.644, 0.807]; 62.0% | 0.883 [0.768, 0.959]; 86.3% | 0.768 [0.688, 0.845]; 73.0% | 0.914 [0.888, 0.942]; 89.1% |
| — unaltered sample | 0.867 [0.761, 0.952]; 75.0% | 0.758 [0.670, 0.844]; 56.7% | 0.730 [0.583, 0.832]; 60.0% | 0.864 [0.752, 0.956]; 82.0% | 0.736 [0.639, 0.832]; 66.5% | 0.910 [0.868, 0.951]; 88.2% |
| Otey | 0.675\*; 23.3% | 0.573\*; 40.7% | 0.670\*; 52.5% | 0.520\*; 22.7% | 0.654\*; 52.5% | 0.820\*; 76.9% |
| Library notes, all layers | 0.592 [0.440, 0.711]; 26.8% | 0.645 [0.503, 0.747]; 43.0% | 0.624 [0.436, 0.756]; 49.0% | 0.559 [0.445, 0.666]; 28.9% | 0.644 [0.486, 0.761]; 52.7% | 0.619 [0.543, 0.682]; 45.1% |
| Library notes, mf | 0.528 [0.419, 0.648]; 17.0% | 0.590 [0.505, 0.685]; 33.5% | 0.542 [0.400, 0.752]; 36.6% | 0.566 [0.429, 0.685]; 27.2% | 0.591 [0.497, 0.691]; 44.2% | 0.590 [0.513, 0.721]; 39.3% |
| FA on CCMusic | 17.7% | 23.5% | 29.5% | 21.9% | 29.4% | 27.6% |

Table S4.9. Per instrument under the controls (seed 0, 47 features): AUC against all CCMusic notes / detection rate. Library rows: all velocity layers. Values for 16 medians: `S4_controls_per_instrument.csv`.

| Instrument (n notes) | Original | 1.2 s | 4 s | Room | 1.2 s + room | 4 s + room |
|---|---|---|---|---|---|---|
| Yamaha CP-80 (53) | 0.976 / 94.3% | 0.931 / 90.6% | 0.913 / 84.9% | 0.956 / 92.5% | 0.929 / 88.7% | 0.953 / 98.1% |
| Casio CTK-230 (78) | 0.986 / 100% | 0.929 / 96.2% | 0.928 / 100% | 0.950 / 94.9% | 0.871 / 87.2% | 0.958 / 93.6% |
| Casio CTK-533 (39) | 0.975 / 100% | 0.921 / 100% | 0.946 / 97.4% | 0.934 / 87.2% | 0.872 / 89.7% | 0.950 / 94.9% |
| Casio CA-110, FM piano (77) | 0.933 / 80.5% | 0.913 / 76.6% | 0.753 / 45.5% | 0.944 / 93.5% | 0.927 / 92.1% | 0.991 / 100% |
| Casio PX-860 (78) | 0.893 / 76.9% | 0.749 / 62.8% | 0.857 / 79.5% | 0.853 / 70.5% | 0.787 / 62.8% | 0.901 / 83.3% |
| Roland JV-1080 (78) | 0.948 / 92.3% | 0.775 / 55.1% | 0.837 / 73.1% | 0.878 / 79.5% | 0.777 / 62.8% | 0.910 / 92.3% |
| Roland U-20 (44) | 0.941 / 93.2% | 0.821 / 68.2% | 0.888 / 88.6% | 0.936 / 93.2% | 0.903 / 93.2% | 0.974 / 97.7% |
| Yamaha PSR-400/500 file (75) | 0.947 / 86.7% | 0.859 / 76.0% | 0.841 / 72.0% | 0.963 / 98.7% | 0.907 / 91.9% | 0.954 / 96.0% |
| Yamaha PSR-F50 (78) | 0.865 / 42.3% | 0.590 / 11.5% | 0.706 / 38.5% | 0.917 / 92.3% | 0.869 / 87.2% | 0.949 / 93.6% |
| Yamaha S90 ES (78) | 0.943 / 89.7% | 0.792 / 48.7% | 0.824 / 73.0% | 0.924 / 88.3% | 0.818 / 70.1% | 0.881 / 82.1% |
| Otey (120) | 0.821 / 34.2% | 0.645 / 40.7% | 0.780 / 63.3% | 0.699 / 34.5% | 0.759 / 64.2% | 0.831 / 75.2% |
| Salamander (480) | 0.714 / 25.6% | 0.840 / 72.1% | 0.797 / 63.9% | 0.607 / 24.0% | 0.771 / 65.4% | 0.729 / 54.4% |
| Iowa (261) | 0.576 / 10.0% | 0.628 / 28.7% | 0.539 / 21.5% | 0.518 / 12.3% | 0.526 / 23.0% | 0.565 / 29.9% |
| K18 (180; 120 at 4 s) | 0.568 / 17.8% | 0.517 / 21.7% | 0.582 / 31.7% | 0.607 / 32.2% | 0.573 / 35.0% | 0.644 / 36.7% |
| FreePats (78; 50 at 4 s) | 0.629 / 15.4% | 0.625 / 30.8% | 0.676 / 46.0% | 0.616 / 29.5% | 0.662 / 50.0% | 0.746 / 58.0% |

Per-instrument AUCs were computed for this supplement from the stored seed-0 scores (`controls_scores.csv`); they reproduce the pooled seed-0 values of Table S4.7. Ten-assignment ranges per instrument: not computed.

Table S4.10. SoundFont notes by what the renderer does to the sample within the analysed window (seed 0, 47 features, AUC vs all CCMusic notes; det.; n notes, n instruments). Transposed: |key − sample root| ≥ 0.5 semitone; looped: loop start reached within 1.8 s (original), 1.3 s (1.2 s cut) or 4.1 s (4 s cut).

| Category | Original | 1.2 s | 4 s | Room | 1.2 s + room | 4 s + room |
|---|---|---|---|---|---|---|
| Neither transposed nor looped | 0.931; 83.3% (168, 9) | 0.818; 67.3% (171, 9) | 0.826; 65.3% (150, 8) | 0.912; 85.6% (167, 9) | 0.850; 79.4% (170, 9) | 0.929; 90.2% (153, 8) |
| Transposed only | 0.927; 80.4% (378, 8) | 0.790; 58.2% (390, 9) | 0.818; 69.0% (335, 7) | 0.918; 87.6% (378, 8) | 0.858; 79.7% (389, 9) | 0.936; 92.3% (336, 7) |
| Looped only | 0.977; 100% (18, 3) | 0.951; 100% (15, 2) | 0.858; 78.8% (33, 3) | 0.959; 100% (18, 3) | 0.924; 93.3% (15, 2) | 0.939; 93.9% (33, 3) |
| Transposed and looped | 0.974; 94.7% (114, 2) | 0.912; 88.2% (102, 2) | 0.895; 87.8% (156, 3) | 0.950; 95.6% (114, 2) | 0.883; 89.1% (101, 2) | 0.960; 95.5% (156, 3) |

Table S4.11. Sample properties of the rendered SoundFont notes (from the SF2 data): median |transposition|, share with active loop mode, median loop start and sample end.

| SoundFont | Median |transp.| (semitones) | Loop mode on | Loop start (s) | Sample end (s) |
|---|---|---|---|---|---|
| Yamaha CP-80 | 3.0 | 0% | – | 10.55 |
| Casio CTK-230 | 5.5 | 100% | 0.28 | 0.28 |
| Casio CTK-533 | 0.0 | 0% | – | 5.04 |
| Casio CA-110 FM | 6.0 | 0% | – | 3.06 |
| Casio PX-860 | 1.0 | 0% | – | 28.40 |
| Roland JV-1080 | 1.0 | 100% | 5.32 | 5.73 |
| Roland U-20 | 4.5 | 0% | – | 1.47 |
| Yamaha PSR-400/500 file | 2.0 | 100% | 1.02 | 1.15 |
| Yamaha PSR-F50 | 3.0 | 0% | – | 4.81 |
| Yamaha S90 ES | 1.0 | 0% | 3.19 (inactive) | 3.77 |

## S4.5 Loop index

Table S4.12. Loop index on the 4-s notes (window 0.5–4.0 s after onset; largest normalized correlation of 50-ms segments with the same note 20 ms–1.5 s later, envelope removed). Recorded sources: notes still sounding at 4 s.

| Source | Notes | Median r_max | 90th pct. | r > 0.999 | r > 0.9999 | Median lag of r > 0.999 (s) | SF2: loop reached by 4.1 s / sample ended | SF2 loop length (s) |
|---|---|---|---|---|---|---|---|---|
| CCMusic | 380 | 0.9849 | 0.9970 | 3% | 0% | 0.023 | – | – |
| Iowa | 261 | 0.9894 | 0.9990 | 10% | 0% | 0.023 | – | – |
| K18 | 120 | 0.9908 | 0.9996 | 20% | 2% | 0.024 | – | – |
| Salamander | 443 | 0.9886 | 0.9977 | 0% | 0% | – | – | – |
| FreePats | 50 | 0.9841 | 0.9974 | 2% | 0% | 0.021 | – | – |
| Otey | 120 | 0.9963 | 0.9998 | 32% | 6% | 0.025 | – | – |
| Yamaha CP-80 | 53 | 0.9997 | 0.9999 | 77% | 11% | 0.109 | 0% / 9% | – |
| Casio CTK-230 | 78 | 1.0000 | 1.0000 | 100% | 100% | 0.467 | 100% / 100% | 0.004 |
| Casio CTK-533 | 39 | 0.9999 | 1.0000 | 100% | 46% | 0.333 | 0% / 15% | – |
| Casio CA-110 FM | 77 | 1.0000 | 1.0000 | 88% | 84% | 0.050 | 0% / 65% | – |
| Casio PX-860 | 78 | 0.9960 | 0.9992 | 17% | 0% | 0.026 | 0% / 0% | – |
| Roland JV-1080 | 78 | 0.9961 | 0.9999 | 19% | 12% | 1.295 | 46% / 38% | 0.358 |
| Roland U-20 | 44 | 0.9748 | 0.9884 | 0% | 0% | – | 0% / 100% | – |
| Yamaha PSR-400/500 file | 75 | 1.0000 | 1.0000 | 88% | 84% | 0.942 | 100% / 96% | 0.013 |
| Yamaha PSR-F50 | 78 | 0.9995 | 0.9996 | 96% | 0% | 0.070 | 0% / 31% | – |
| Yamaha S90 ES | 78 | 0.9889 | 0.9995 | 13% | 0% | 0.169 | 0% / 65% | – |

## S4.6 Phaseogram irregularity

Table S4.13. Irregularity (SD of the period-to-period shift after linear detrending, samples per period; 256 samples = one period), notes C3–E5 with correct pitch; 50 notes per class stratified by source, all 40 Otey notes. Room = MIT room response + noise 40 dB below the note.

| Class | Notes | Median [IQR] | With room | Row-to-row r | P(class > real), p: original | P(class > real), p: with room |
|---|---|---|---|---|---|---|
| Real (CCMusic) | 50 | 1.355 [0.66, 2.80] | 1.868 | 0.980 | – | – |
| Sample libraries | 50 | 0.769 [0.49, 1.25] | 0.632 | 0.993 | 0.32, p = 0.0022 | 0.28, p = 0.00013 |
| SoundFont | 50 | 0.288 [0.20, 0.42] | 0.383 | 0.993 | 0.09, p = 1.1 × 10⁻¹² | 0.23, p = 3.5 × 10⁻⁶ |
| Otey | 40 | 0.054 [0.03, 0.23] | 0.183 | 0.998 | 0.05, p = 1.9 × 10⁻¹³ | 0.12, p = 7.4 × 10⁻¹⁰ |

AUC of irregularity, real vs class (= 1 − P): libraries 0.68, SoundFont 0.91, Otey 0.95. Real notes, early / late half of the 48 periods: median 1.78 / 0.62.

Table S4.14. Irregularity per source (median; IQR for n ≥ 6), computed for this supplement from `c4_phaseogram_c3.csv`.

| Source | n | Original: median [IQR] | With room |
|---|---|---|---|
| CCMusic Hsinghai upright | 7 | 5.48 [3.64, 9.66] | 4.57 |
| CCMusic Kawai grand | 7 | 0.41 [0.26, 0.71] | 0.45 |
| CCMusic Kawai upright | 6 | 3.31 [1.63, 4.99] | 2.58 |
| CCMusic Pearl River upright | 6 | 1.14 [0.59, 1.60] | 0.48 |
| CCMusic Steinway (Grand Theatre) | 6 | 1.84 [1.39, 2.66] | 4.87 |
| CCMusic Steinway grand | 6 | 1.35 [0.64, 2.29] | 4.44 |
| CCMusic Yamaha upright | 6 | 0.77 [0.68, 1.08] | 1.42 |
| CCMusic Young Chang upright | 6 | 1.97 [1.21, 7.95] | 1.34 |
| FreePats | 13 | 0.75 [0.51, 1.21] | 0.88 |
| Iowa (Steinway B) | 13 | 1.25 [0.73, 2.18] | 0.77 |
| K18 (upright) | 12 | 0.78 [0.49, 1.19] | 0.59 |
| Salamander (Yamaha C5) | 12 | 0.42 [0.21, 0.81] | 0.27 |
| Yamaha CP-80 | 5 | 0.24 | 0.36 |
| Casio CTK-230 | 5 | 0.34 | 0.33 |
| Casio CTK-533 | 5 | 0.22 | 0.34 |
| Casio CA-110 FM | 5 | 0.031 | 0.28 |
| Casio PX-860 | 5 | 0.73 | 0.70 |
| Roland JV-1080 | 5 | 0.61 | 1.67 |
| Roland U-20 | 5 | 0.43 | 0.52 |
| Yamaha PSR-400/500 file | 5 | 0.20 | 0.61 |
| Yamaha PSR-F50 | 5 | 0.28 | 1.19 |
| Yamaha S90 ES | 5 | 0.29 | 0.32 |
| Otey | 40 | 0.054 [0.029, 0.230] | 0.18 |

## S4.7 Phaseogram CNN and fusion

Single fold assignment (protocol-v2 folds; see S2), constrained f₀, all registers; 762 SoundFont renders (velocity copies included). AUC [bootstrap CI]; det. Ten-assignment ranges: not computed.

Table S4.15. Corpus sources (out-of-fold) and library notes (mean of the five fold models, v2 rule).

| CCMusic notes vs | Files/groups | CNN T2 | Fusion-47 T2 | Fusion-16 T2 | XGBoost 47, same notes and folds | CNN T1 | Fusion-47 T1 |
|---|---|---|---|---|---|---|---|
| SoundFont | 762/10 | 0.910 [0.832, 0.970]; 87.8% | 0.940 [0.862, 0.988]; 93.3% | 0.924 [0.831, 0.982]; 88.5% | 0.929 [0.856, 0.975]; 87.0% | 0.907 [0.846, 0.968]; 95.7% | 0.930 [0.857, 0.988]; 94.9% |
| Otey | 120/1 | 0.776 [0.679, 0.861]\*; 50.8% | 0.817 [0.686, 0.933]\*; 52.5% | 0.797 [0.664, 0.909]\*; 48.3% | 0.779 [0.640, 0.891]\*; 38.3% | 0.801 [0.704, 0.909]\*; 75.0% | 0.806 [0.677, 0.943]\*; 73.3% |
| Algorithmic | 882/11 | 0.892 [0.810, 0.961]; 82.8% | 0.923 [0.835, 0.983]; 87.8% | 0.907 [0.806, 0.977]; 83.0% | 0.908 [0.820, 0.970]; 80.4% | 0.893 [0.819, 0.964]; 92.9% | 0.913 [0.828, 0.985]; 92.0% |
| Library notes, mf | 224/4 | 0.713 [0.548, 0.877]; 36.2% | 0.619 [0.427, 0.819]; 18.3% | 0.609 [0.410, 0.835]; 17.0% | 0.551 [0.384, 0.723]; 11.2% | 0.597 [0.382, 0.848]; 51.3% | 0.558 [0.305, 0.848]; 38.8% |
| Library notes, all layers | 999/4 | 0.756 [0.574, 0.888]; 45.8% | 0.646 [0.439, 0.832]; 17.9% | 0.667 [0.419, 0.863]; 20.9% | 0.600 [0.403, 0.763]; 13.1% | 0.688 [0.403, 0.875]; 64.7% | 0.616 [0.313, 0.844]; 49.5% |
| Salamander, mf | 30/1 | 0.835 [0.741, 0.919]\* | 0.686 [0.527, 0.841]\* | 0.730 [0.587, 0.867]\* | 0.665 [0.505, 0.802]\* | 0.827\* | 0.750\* |
| Iowa, mf | 86/1 | 0.562 [0.438, 0.687]\* | 0.476 [0.333, 0.628]\* | 0.413 [0.297, 0.537]\* | 0.431 [0.293, 0.567]\* | 0.370\* | 0.308\* |
| K18, mf | 30/1 | 0.782 [0.677, 0.879]\* | 0.746 [0.605, 0.882]\* | 0.751 [0.611, 0.882]\* | 0.642 [0.510, 0.760]\* | 0.766\* | 0.764\* |
| FreePats | 78/1 | 0.806 [0.698, 0.908]\* | 0.702 [0.559, 0.842]\* | 0.724 [0.581, 0.855]\* | 0.604 [0.467, 0.725]\* | 0.694\* | 0.682\* |
| FA on CCMusic | 668/6 | 18.1% | 16.2% | 15.9% | 15.4% | 34.4% | 26.0% |

Table S4.16. Library notes scored by the model of the library's own fold (v3 rule; values used in Sec. 5.4), T2.

| CCMusic notes vs | CNN | Fusion-47 | Fusion-16 |
|---|---|---|---|
| Library notes, mf | 0.698 [0.533, 0.874]; 38.4% | 0.543 [0.365, 0.731]; 15.6% | 0.567 [0.367, 0.777]; 21.4% |
| Library notes, all layers | 0.756 [0.557, 0.890]; 52.2% | 0.567 [0.379, 0.746]; 17.9% | 0.601 [0.384, 0.787]; 23.2% |
| Salamander, mf | 0.856 [0.768, 0.932]\*; 73.3% | 0.584\*; 13.3% | 0.628\*; 20.0% |
| Iowa, mf | 0.536 [0.421, 0.652]\*; 11.6% | 0.417\*; 1.2% | 0.377\*; 3.5% |
| K18, mf | 0.789 [0.689, 0.879]\*; 50.0% | 0.696\*; 33.3% | 0.683\*; 30.0% |
| FreePats | 0.779 [0.673, 0.878]\*; 50.0% | 0.607\*; 25.6% | 0.710\*; 38.5% |
| FA on CCMusic | 18.1% | 16.2% | 15.9% |

Table S4.17. CNN and fusion-47, T2, by evaluation subset: AUC; det. (`constr` = constrained-f₀ models; `det48` = trained and evaluated on MIDI 48–76 with the detector f₀).

| Model / subset | CCMusic files | SoundFont | Otey | Library mf | FA |
|---|---|---|---|---|---|
| CNN, all | 668 | 0.910; 87.8% | 0.776; 50.8% | 0.713; 36.2% | 18.1% |
| CNN, MIDI 48–76 | 220 | 0.876; 90.0% | 0.880; 82.5% | 0.712; 56.0% | 27.7% |
| CNN, |f₀ − nominal| < 0.5 semitone | 621 | 0.910; 88.3% | 0.772; 50.8% | 0.723; 38.4% | 18.5% |
| CNN, det48 | 220 | 0.832; 95.3% | 0.714; 77.5% | 0.622; 60.0% | 42.7% |
| Fusion-47, all | 668 | 0.940; 93.3% | 0.817; 52.5% | 0.619; 18.3% | 16.2% |
| Fusion-47, MIDI 48–76 | 220 | 0.944; 97.0% | 0.783; 50.0% | 0.612; 22.7% | 21.4% |
| Fusion-47, |f₀ − nominal| < 0.5 semitone | 621 | 0.937; 93.5% | 0.809; 52.5% | 0.625; 19.4% | 17.1% |
| Fusion-47, det48 | 220 | 0.936; 97.3% | 0.671; 15.0% | 0.585; 22.7% | 24.5% |

## S4.8 eGeMAPS

88 eGeMAPSv02 functionals (openSMILE 2.6.0), same XGBoost, isolated notes, T2.

Table S4.18. Protocol v3 folds, own-fold scoring.

| CCMusic notes vs | Median [min–max] over ten assignments | Seed 0 [CI]; det. |
|---|---|---|
| Algorithmic (798 notes) | 0.943 [0.937–0.948] | 0.948 [0.83, 1.00]; 92.6% |
| Library notes, mf | 0.694 [0.670–0.729] | 0.704 [0.47, 0.89]; 27.7% |
| Library notes, all layers | 0.662 [0.615–0.733] | 0.617 [0.39, 0.91]; 21.7% |
| FA on CCMusic | not computed | 10.3% |

Mean-of-five-models scoring (v2 rule), seed 0: library mf 0.759, all layers 0.727.

Table S4.19. v2 folds and scoring (no v3 replacement): per library and with all files in the listening-clip format (44.1 kHz mono, onset-trimmed, 3 s, RMS −20 dBFS). XGBoost T2, AUC [CI]; det.

| CCMusic notes vs | Native files | Clip format |
|---|---|---|
| SoundFont | 0.956 [0.837, 1.000]; 96.3% | 0.951 [0.827, 1.000]; 93.0% |
| Otey | 0.863 [0.669, 0.974]\*; 69.2% | 0.702 [0.480, 0.839]\*; 28.3% |
| Algorithmic | 0.943 [0.822, 1.000]; 92.6% | 0.917 [0.770, 0.998]; 84.2% |
| Library notes, mf | 0.743 [0.519, 0.899]; 25.0% | 0.719 [0.466, 0.884]; 17.0% |
| Library notes, all layers | 0.698 [0.480, 0.894]; 19.8% | 0.698 [0.455, 0.859]; 10.7% |
| Salamander, mf | 0.593 [0.376, 0.797]\*; 3.3% | 0.642 [0.406, 0.804]\*; 0.0% |
| Iowa, mf | 0.773 [0.530, 0.932]\*; 4.7% | 0.671 [0.426, 0.827]\*; 4.7% |
| K18, mf | 0.784 [0.551, 0.940]\*; 30.0% | 0.758 [0.500, 0.903]\*; 36.7% |
| FreePats | 0.752 [0.548, 0.878]\*; 53.8% | 0.787 [0.527, 0.927]\*; 29.5% |
| FA on CCMusic | 10.0% | 13.2% |

Largest XGBoost gain shares, native files: loudness_sma3_meanRisingSlope 0.418, loudnessPeaksPerSec 0.119, loudness_sma3_stddevRisingSlope 0.110 (together 0.65). Median lead-in before the attack: CCMusic 0.021 s, SoundFont 0.000, Otey 0.002, Salamander 0.011, Iowa 0.250, K18 0.005, FreePats 0.000 s. Clip format: MeanUnvoicedSegmentLength 0.309.

## S4.9 Enlarged recorded class (protocol v4)

13 recorded instruments in 13 groups: 8 CCMusic pianos (one group each), 4 libraries, one Yamaha Disklavier (MAPS ENSTDkCl and ENSTDkAm, one group). Grouped 5-fold CV over 24 groups (10 SoundFonts and Otey added), ten fold assignments; LOGO = leave one group out. Room augmentation: each training note replaced with probability 0.5 by one of two copies convolved with a random MIT room response plus white noise at U(30, 50) dB SNR. Channel test: every test note through one random copy.

Table S4.20. 47 features, all recorded notes vs source: median [min–max] over ten assignments.

| Recorded notes vs | CCMusic only (v3) | 13 instruments, all layers | 13 instruments, mf | + room in training, clean test | + room in training and testing | No room in training, room in testing | LOGO: no room / room in training |
|---|---|---|---|---|---|---|---|
| SoundFont | 0.930 [0.919–0.937] | 0.942 [0.923–0.955] | 0.930 [0.909–0.949] | 0.954 [0.942–0.961] | 0.941 [0.937–0.954] | 0.891 [0.871–0.905] | 0.940 / 0.946 |
| Otey | 0.813 [0.783–0.831] | 0.826 [0.783–0.860] | 0.824 [0.781–0.861] | 0.855 [0.827–0.880] | 0.778 [0.744–0.822] | 0.740 [0.590–0.797] | 0.807 / 0.846 |
| Algorithmic | 0.913 [0.900–0.920] | 0.925 [0.902–0.940] | 0.913 [0.890–0.935] | 0.937 [0.927–0.945] | 0.919 [0.909–0.931] | 0.867 [0.831–0.882] | 0.920 / 0.931 |
| FA, recorded notes | 12.6% [10.3–13.0] | 6.7% [6.1–7.7] | 8.7% [7.2–10.5] | 8.3% [7.4–9.9] | 8.8% [7.8–9.6] | 2.6% [1.8–3.8] | 7.7% / 8.3% |
| Det. SoundFont | – | 75.7% [69.8–78.5] | 76.8% [71.1–78.8] | 84.8% [82.6–89.8] | 82.0% [80.2–85.6] | 40.3% [35.7–44.0] | – |
| Det. Otey | – | 23.8% [16.7–31.7] | 27.5% [23.3–34.2] | 39.2% [28.3–49.2] | 32.5% [22.5–35.0] | 8.8% [1.7–20.0] | – |

Seed 0 with bootstrap CI (all layers): SoundFont 0.937 [0.897, 0.971], det. 72.9%; Otey 0.844 [0.792, 0.904]\*, det. 27.5%; algorithmic 0.923 [0.877, 0.966]. With room in training, clean test: SoundFont 0.946 [0.903, 0.978], Otey 0.865 [0.820, 0.916]\*; room in testing: SoundFont 0.937 [0.894, 0.969], Otey 0.777 [0.721, 0.835]\*.

Table S4.21. Same, 16 medians: median [min–max].

| Recorded notes vs | CCMusic only (v3) | 13 instr., all layers | 13 instr., mf | + room, clean test | + room, room test | No room, room test |
|---|---|---|---|---|---|---|
| SoundFont | 0.862 [0.849–0.877] | 0.862 [0.843–0.885] | 0.871 [0.846–0.898] | 0.865 [0.848–0.889] | 0.890 [0.885–0.913] | 0.788 [0.767–0.817] |
| Otey | 0.631 [0.580–0.685] | 0.703 [0.665–0.756] | 0.673 [0.625–0.713] | 0.615 [0.554–0.660] | 0.599 [0.545–0.662] | 0.621 [0.463–0.699] |
| FA, recorded notes | 18.0% [16.0–20.8] | 10.4% [9.6–12.7] | 13.0% [11.0–14.7] | 15.9% [14.1–19.7] | 12.9% [10.8–14.6] | 1.8% [1.2–2.2] |

Table S4.22. False alarms per recorded instrument (47 features, all layers; median [min–max] over ten assignments).

| Instrument | No room | + room, clean test | + room, room test | LOGO |
|---|---|---|---|---|
| CCMusic Hsinghai | 1.1% [1.1–2.3] | 1.1% [0.0–5.7] | 6.2% [2.3–9.1] | 1.1% |
| CCMusic Kawai | 4.5% [1.1–10.2] | 4.5% [2.3–9.1] | 5.1% [2.3–11.4] | 5.7% |
| CCMusic Kawai-G | 1.1% [0.0–6.8] | 2.8% [0.0–8.0] | 4.5% [0.0–6.8] | 3.4% |
| CCMusic Pearl River | 26.7% [21.6–38.6] | 34.7% [31.8–44.3] | 48.9% [40.9–58.0] | 33.0% |
| CCMusic Steinway | 0.0% [0.0–0.0] | 1.9% [0.0–5.8] | 1.9% [0.0–3.8] | 0.0% |
| CCMusic Steinway-T | 5.7% [3.4–8.0] | 5.1% [0.0–9.1] | 5.7% [3.4–11.4] | 6.8% |
| CCMusic Yamaha | 0.0% [0.0–2.3] | 0.0% [0.0–2.3] | 0.6% [0.0–3.4] | 0.0% |
| CCMusic Young Chang | 4.0% [1.1–8.0] | 2.3% [0.0–4.5] | 2.3% [0.0–3.4] | 4.5% |
| Salamander | 8.5% [8.5–8.5] | 11.5% [9.8–12.3] | 8.9% [7.5–10.0] | 9.2% |
| Iowa | 1.5% [0.8–2.3] | 2.1% [0.8–3.4] | 0.8% [0.0–1.5] | 1.5% |
| K18 | 12.2% [10.6–15.0] | 10.8% [8.3–18.3] | 8.1% [5.6–14.4] | 13.3% |
| FreePats | 10.3% [5.1–12.8] | 18.6% [11.5–25.6] | 16.0% [12.8–21.8] | 10.3% |
| Disklavier, close (ENSTDkCl) | 5.4% [4.2–9.2] | 6.7% [3.3–11.7] | 10.8% [7.5–13.3] | 6.7% |
| Disklavier, ambient (ENSTDkAm) | 6.1% [3.0–12.1] | 8.7% [5.3–11.4] | 10.6% [6.1–12.1] | 8.3% |

Table S4.23. Recorded collections among themselves. (a) Detector trained with all 13 instruments recorded: AUC that the collection is scored as more synthetic than CCMusic (median [min–max]). (b) Dedicated classifier, collection vs CCMusic (pooled out-of-fold AUC; CCMusic grouped by piano, target by pitch).

| Collection | (a) 47 feat. | (a) 16 med. | (a) + room, clean | (b) 47 feat. | (b) 47 feat., room | (b) 16 med. |
|---|---|---|---|---|---|---|
| Salamander | 0.700 [0.673–0.723] | 0.682 [0.658–0.708] | 0.649 [0.619–0.692] | 0.946 | 0.879 | 0.906 |
| Iowa | 0.410 [0.336–0.423] | 0.308 [0.257–0.334] | 0.407 [0.344–0.459] | 0.892 | 0.787 | 0.879 |
| K18 | 0.616 [0.593–0.660] | 0.598 [0.538–0.629] | 0.566 [0.501–0.639] | 0.712 | 0.695 | 0.742 |
| FreePats | 0.602 [0.559–0.653] | 0.597 [0.524–0.638] | 0.630 [0.569–0.682] | 0.774 | 0.775 | 0.721 |
| Disklavier (both rooms) | 0.508 [0.408–0.583] | 0.482 [0.408–0.551] | 0.522 [0.465–0.551] | 0.843 | 0.824 | 0.832 |

Reference rows of (b), median by kind (47 features; room in brackets; 16 medians): free libraries 0.833 (0.781; 0.810); Disklavier 0.843 (0.824; 0.832); one CCMusic piano vs the other seven 0.871 (0.802; 0.853), range 0.769–0.973; SoundFonts 0.972 (0.924; 0.931); Otey 0.976 (0.898; 0.940); Disklavier ambient vs close room 0.678.

## S4.10 Performances: what carries the separation

Table S4.24. Feature-group ablation (performance detector, T2, seed 0): AUC [CI]; det.

| Features | Sampled digital pianos | Algorithmic renders | Modelled digital pianos |
|---|---|---|---|
| All 47 | 0.871 [0.843, 0.891]; 7.0% | 0.943 [0.907, 0.978]; 36.7% | 0.854\*; 9.5% |
| Group A medians (16) | 0.562 [0.525, 0.622]; 1.6% | 0.723 [0.622, 0.839]; 9.2% | 0.583\*; 0.0% |
| Group A inter-quartile ranges (16) | 0.678 [0.609, 0.730]; 3.9% | 0.819 [0.729, 0.906]; 30.0% | 0.710\*; 9.5% |
| Group B, register (8) | 0.680 [0.593, 0.754]; 7.8% | 0.732 [0.621, 0.839]; 20.8% | 0.667\*; 9.5% |
| Group C, tuning (7) | 0.791 [0.766, 0.813]; 16.4% | 0.738 [0.649, 0.813]; 23.3% | 0.656\*; 4.8% |

Table S4.25. Recording-chain and tuning indicators (medians per file).

| Quantity | Real performances | Sampled digital pianos | Algorithmic renders | CCMusic (studio, isolated) |
|---|---|---|---|---|
| Fundamental share, bass notes (feature 33) | 0.027 | 0.083 | 0.081 | 0.112 |
| Integral detuning index (feature 41) | 11.6 (v2) | 5.67 (recent demonstrations) | 6.35 / 5.73 (SoundFont / Otey, v2) | 8.97 (v2) |

Table S4.26. Dry renders (seed 0).

| Comparison | AUC [CI] | Det. | FA |
|---|---|---|---|
| Mixed-format detector, sampled digital pianos, without algorithmic performance renders in training | 0.644 [0.583, 0.682] | not recorded | not recorded |
| Same, with the 120 dry renders in training | 0.860 [0.820, 0.891] | 3.1% | 0.2% |
| Sensitivity corpus: without / with renders | 0.646 [0.581, 0.692] / 0.871 [0.820, 0.911] | – / 7.8% | – / 0.3% |

Table S4.27. Sampled digital pianos by brand (v2: real class before the title screen, mean of the five fold models; no v3 replacement). AUC vs real performances, 47 features, T2; median P(synthetic) and median detuning index.

| Brand | Files | Models | AUC | Median P | Detuning index |
|---|---|---|---|---|---|
| Casio | 16 | 6 | 0.80 | 0.006 | 5.76 |
| Dexibell | 12 | 5 | 0.90 | 0.016 | 5.66 |
| Kawai | 31 | 11 | 0.89 | 0.024 | 5.75 |
| Korg | 14 | 5 | 0.92 | 0.031 | 5.24 |
| Kurzweil | 8 | 3 | 0.88 | 0.017 | 4.33 |
| Nord | 10 | 5 | 0.81 | 0.023 | 8.33 |
| Yamaha | 28 | 10 | 0.93 | 0.102 | 4.94 |
| Roland (modelled) | 19 | 7 | not computed | 0.022 | 7.19 |

## S4.11 Machine-readable copies

`S4_detection_by_format.csv` (Tables S4.1–S4.2: seed-0 AUC, bootstrap limits, det., FA; ten-assignment median, min, max; T1), `S4_controls_ten_assignments.csv` (Table S4.6 with seed-0 values), `S4_controls_per_instrument.csv` (Table S4.9, both feature sets, all conditions).
