# RESULTS v2 -- clean re-analysis for the JAES revision

Generated 2026-10-01 16:14 by one command, from the project root:

```
python Stattia/revision/analysis/run_all.py
```

Environment: Python 3.10.7, numpy 1.26.4, pandas 1.5.2, scikit-learn 1.4.1.post1, scipy 1.11.1, xgboost 3.2.0 (n_jobs=1, random_state=0). Scripts: `Stattia/revision/analysis/{run_all, protocol_v2, listening_v2, followup_v2, level_test, formats_v2, pv2_lib}.py`. Total runtime 554 s. Determinism check vs the previous run (out/checksums.txt): 19 of 20 output CSVs byte-identical.

Outputs (all in `Stattia/revision/analysis/out/`): `folds.csv` (fixed fold per file), `oof_scores.csv` (per-file scores for every model x feature set x condition), `comparisons.csv` (every AUC row incl. real false alarms), `pernote_*.csv`, `oof_scores_isolated_only.csv`, `stimuli_detector_scores.csv`, `listening_*.csv`, `excluded_non_acoustic.csv`, `old_demos_regroup_used.csv`, `f0_audit.csv`, `f0_sensitivity.csv`, `fold_seed_sensitivity.csv`, `followup_*.csv`, `digital_pianos.csv`, `level_test_*.csv`, `format_specific.csv`, `key_numbers.json`, `cache/` (extracted features of library notes, corpus single-note files and stimuli).

## 0. Key numbers (auto-generated from the tables below)

| comparison | XGB all47 T1 | XGB all47 T2 | XGB med16 T1 | XGB med16 T2 |
|---|---|---|---|---|
| isolated: CCMusic vs SoundFont renders | 0.924 [0.856, 0.970]; det 88.5% | 0.934 [0.870, 0.976]; det 88.3% | 0.817 [0.728, 0.898]; det 76.9% | 0.834 [0.748, 0.907]; det 75.3% |
| isolated: CCMusic vs Otey | 0.809 [0.697, 0.898]; det 61.7% | 0.807 [0.696, 0.900]; det 50.0% | 0.638 [0.548, 0.749]; det 33.3% | 0.684 [0.598, 0.774]; det 33.3% |
| isolated: CCMusic vs algorithmic (both) | 0.908 [0.833, 0.966]; det 84.8% | 0.917 [0.842, 0.971]; det 83.1% | 0.793 [0.692, 0.888]; det 71.0% | 0.813 [0.723, 0.898]; det 69.6% |
| isolated: CCMusic vs library takes | 0.607 [0.400, 0.807]; det 18.8% | 0.410 [0.204, 0.648]; det 0.0% | 0.412 [0.299, 0.541]; det 9.4% | 0.379 [0.262, 0.520]; det 6.2% |
| performance: real vs SoundFont performance renders | 0.951 [0.924, 0.975]; det 56.0% | 0.956 [0.921, 0.986]; det 42.0% | 0.756 [0.648, 0.855]; det 25.0% | 0.759 [0.625, 0.876]; det 22.0% |
| performance: real vs Otey performance renders | 0.951 [0.946, 0.956]; det 65.0% | 0.953 [0.948, 0.958]; det 15.0% | 0.686 [0.674, 0.698]; det 10.0% | 0.748 [0.736, 0.760]; det 0.0% |
| performance: real vs sampled digital pianos (old + v2) | 0.915 [0.892, 0.936]; det 34.4% | 0.887 [0.845, 0.917]; det 3.1% | 0.707 [0.669, 0.770]; det 11.7% | 0.676 [0.621, 0.709]; det 0.0% |
| performance: real vs old digital demos | 0.915 [0.876, 0.970]; det 22.2% | 0.918 [0.869, 0.948]; det 0.0% | 0.786 [0.672, 0.978]; det 33.3% | 0.800 [0.762, 0.901]; det 0.0% |
| performance: real vs digital_v2 sample-based | 0.915 [0.892, 0.937]; det 35.3% | 0.885 [0.843, 0.915]; det 3.4% | 0.702 [0.663, 0.763]; det 10.1% | 0.666 [0.607, 0.702]; det 0.0% |
| performance: real vs digital_v2 sampled, line capture | 0.950 [0.894, 0.983]; det 58.3% | 0.937 [0.881, 0.975]; det 8.3% | 0.710 [0.664, 0.796]; det 8.3% | 0.748 [0.622, 0.816]; det 0.0% |
| performance: real vs modelled digital | 0.865 [0.859, 0.871]; det 42.9% | 0.865 [0.857, 0.873]; det 0.0% | 0.812 [0.803, 0.821]; det 28.6% | 0.685 [0.674, 0.697]; det 4.8% |
| performance: real vs Korg M1 | 0.833 [0.824, 0.842]; det 20.0% | 0.891 [0.884, 0.898]; det 6.7% | 0.803 [0.796, 0.811]; det 30.0% | 0.889 [0.884, 0.893]; det 43.3% |
| format-crossed: real performances vs single-note algorithmic | 0.982 [0.973, 0.991]; det 84.8% | 0.994 [0.990, 0.998]; det 83.1% | 0.927 [0.888, 0.964]; det 71.0% | 0.960 [0.934, 0.983]; det 69.6% |
| pooled: all real vs algorithmic | 0.961 [0.938, 0.980]; det 81.5% | 0.970 [0.947, 0.988]; det 77.6% | 0.876 [0.822, 0.928]; det 65.2% | 0.903 [0.858, 0.946]; det 63.5% |
| pooled: all real vs sample-based | 0.845 [0.793, 0.897]; det 29.5% | 0.796 [0.722, 0.864]; det 3.2% | 0.660 [0.596, 0.724]; det 14.2% | 0.642 [0.555, 0.725]; det 7.9% |
| per-note: CCMusic notes vs library notes (mf layer) | 0.440 [0.282, 0.627]; det 12.9% | 0.530 [0.370, 0.708]; det 13.4% | 0.345 [0.204, 0.517]; det 9.8% | 0.445 [0.299, 0.613]; det 12.9% |

f0 audit (CCMusic, primary segment / all segments off by >= 1 semitone): MIDI<48 117/205 files; 465/604 seg.; 48-76 9/220 files; 198/438 seg.; >76 203/243 files; 682/743 seg.; all 329/668 files; 1345/1785 seg..

Isolated regime under f0 / register restrictions (XGB all47, T2; CCMusic vs ...):

| restriction | SoundFont renders | Otey notes | library notes (mf) |
|---|---|---|---|
| baseline | 0.934 [0.870, 0.976] | 0.807 [0.696, 0.900] | 0.530 [0.370, 0.708] |
| f0-valid & 48-76 | 0.944 [0.855, 0.991] | 0.736 [0.582, 0.855] | 0.500 [0.334, 0.665] |
| f0-clean features (retrained) | 0.932 [0.869, 0.983] | 0.798 [0.693, 0.906] | 0.549 [0.395, 0.729] |
| f0-clean features & 48-76 | 0.944 [0.872, 0.993] | 0.803 [0.670, 0.932] | 0.585 [0.413, 0.766] |

Listening stimuli, out-of-sample XGB all47 (LOGO, clips): detected Iowa 2/7, K18 4/7, Salamander 3/6, SoundFont (Casio CA-110) 3/5, Otey 3/5; AUC recorded vs algorithmic 0.665. Old in-sample model (paper): Iowa 1/7, K18 3/7, Salamander 2/6, SoundFont (Casio CA-110) 5/5, Otey 5/5; AUC 0.965.

Listeners, judged synthetic: Iowa 43% (Wilcoxon p>0.5 = 0.97), K18 35% (Wilcoxon p>0.5 = 1), Salamander 33% (Wilcoxon p>0.5 = 1), SoundFont (Casio CA-110) 67% (Wilcoxon p>0.5 = 0.0059), Otey 63% (Wilcoxon p>0.5 = 0.0037), Iowa+K18 (paper's "real") 39% (Wilcoxon p>0.5 = 1), RECORDED (all 3 libraries) 37% (Wilcoxon p>0.5 = 1), ALGORITHMIC (SoundFont + Otey) 65% (Wilcoxon p>0.5 = 0.00048).
Listener AUC recorded vs algorithmic: mean 0.665 (CI 0.597-0.734), panel 0.983; mean d' 0.75.

Level test (section 8), median |feature change| / inter-file IQR by group: A medians x0.5 0.000, x0.1 0.054, pcm16 0.000; A IQRs x0.5 0.000, x0.1 0.030, pcm16 0.000; B x0.5 0.002, x0.1 0.058, pcm16 0.000; C x0.5 0.000, x0.1 0.028, pcm16 0.000.

## 1. Corpus (protocol v2)

Built from `quality/cache/features_cal.csv`, `synth_features.csv`, `samplelib_features.csv` (md5 64ce01f6, 4801bee5, 241abe83; identical to the released CSVs per facts_A). Rows with fewer than 30 non-missing features are dropped, as in `train_realness._load` (0 dropped).

Corrections: **25 non-acoustic files removed from the real class** (list: `out/excluded_non_acoustic.csv`); the 14 old digital-piano demos regrouped by brand, the Roland SC-55 sound module as its own group (`out/old_demos_regroup_used.csv`; source: data_v2/old_demos_regroup.csv): casio 1 file(s) [sample: 1]; korg 4 file(s) [sample: 4]; kurzweil 1 file(s) [sample: 1]; roland 3 file(s) [modelling: 2, hybrid_unclear: 1]; roland_sc55 2 file(s) [sample: 2]; yamaha 3 file(s) [sample: 3]. sound_method decides the kind: sample -> sampled_digital_performance, modelling -> modelled_digital, anything else (e.g. hybrid_unclear) -> digital_other; all three are excluded from training under T2. data_v2: {"digital_v2_features.csv": {"rows": 138, "sound_method": {"sample": 120, "modelling": 19}, "dropped_excluded_by_manifest": 0, "capture_source": "digital_v2_manifest.csv", "excluded_protocol": {"Casio_PXS1000_0Si1TBz9b6I.wav": "Casio PX-S1000 demo with EXTERNAL convolution reverb mixed in (manifest flag external_reverb_added)"}, "capture": {"unknown": 123, "line": 13, "mic": 2}}, "sf2_perf_features.csv": {"rows": 100, "groups": 10, "groups_shared_with_single_note_renders": 100}, "otey_perf_features.csv": {"rows": 20}}.

| kind | subkind | files | CV groups | label |
|---|---|---|---|---|
| real_performance | manifest | 432 | 276 | 0 |
| real_performance | youtube_extra | 2144 | 2144 | 0 |
| real_isolated | ccmusic | 668 | 6 | 0 |
| algo_single | soundfont_single | 762 | 10 | 1 |
| algo_single | otey_single | 120 | 1 | 1 |
| sampled_digital_performance | old_demo | 9 | 4 | 1 |
| digital_other | old_demo | 1 | 1 | 1 |
| modelled_digital | old_demo | 2 | 1 | 1 |
| sound_module | old_demo | 2 | 1 | 1 |
| m1 | m1 | 30 | 1 | 1 |
| sampled_library | library:Salamander | 8 | 1 | 1 |
| sampled_library | library:K18 | 8 | 1 | 1 |
| sampled_library | library:FreePats | 8 | 1 | 1 |
| sampled_library | library:Iowa | 8 | 1 | 1 |
| sampled_digital_performance | digital_v2 | 119 | 7 | 1 |
| modelled_digital | modelled_v2 | 19 | 1 | 1 |
| algo_performance | sf2_perf | 100 | 10 | 1 |
| algo_performance | otey_perf | 20 | 1 | 1 |

Total 4460 files: real 3244 (2426 groups), synthetic 1216 (25 groups).

**Fixed fold assignment** (`out/folds.csv`; greedy GroupKFold rule, tie order = groups sorted by name and permuted with seed 0; compared with previous out/folds.csv: 4460 common files, 0 changed fold):

| fold | files | real files | CCMusic files | CCMusic groups | synthetic groups |
|---|---|---|---|---|---|
| 0 | 861 | 685 | 176 | 1 | sf2:casioca110pianofm, sf2:yamahapsr400500 |
| 1 | 877 | 668 | 140 | 1 | casio, lib:FreePats, lib:Iowa, sf2:casiopx860concertgrandpiano, sf2:cp801644100 |
| 2 | 953 | 511 | 0 | 0 | kawai, korg, yamaha, lib:K18, lib:Salamander, m1, otey, sf2:ctk230soundfont, sf2:yamahas90esnaturalgrand |
| 3 | 899 | 681 | 176 | 2 | dexibell, kurzweil, roland, roland_sc55, sf2:jv1080nicepianom, sf2:u20piano |
| 4 | 870 | 699 | 176 | 2 | nord, sf2:ctk533piano100casio, sf2:yamahapsrf50 |

**Format diagnostics** (why file format is visible to the model): share of the 16 IQR features exactly 0, share of files with ALL 16 IQR features = 0, NaN rate of the 15 Group B+C features, NaN rate of `global_scatter` and `bass_fund_strength`, overall NaN rate.

| subkind | files | IQR==0 (features) | all 16 IQR = 0 (files) | NaN Group B+C | NaN global_scatter | NaN bass_fund_strength | NaN all |
|---|---|---|---|---|---|---|---|
| manifest | 432 | 0.08 | 0.00 | 0.04 | 0.00 | 0.01 | 0.02 |
| youtube_extra | 2144 | 0.07 | 0.00 | 0.04 | 0.00 | 0.01 | 0.02 |
| ccmusic | 668 | 0.34 | 0.27 | 0.32 | 0.29 | 0.29 | 0.11 |
| soundfont_single | 762 | 0.78 | 0.73 | 0.54 | 0.77 | 0.60 | 0.18 |
| otey_single | 120 | 0.76 | 0.71 | 0.53 | 0.88 | 0.67 | 0.17 |
| old_demo | 9 | 0.06 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| old_demo | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| old_demo | 2 | 0.06 | 0.00 | 0.03 | 0.00 | 0.00 | 0.01 |
| old_demo | 2 | 0.03 | 0.00 | 0.07 | 0.00 | 0.00 | 0.02 |
| m1 | 30 | 0.06 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| library:Salamander | 8 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| library:K18 | 8 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| library:FreePats | 8 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| library:Iowa | 8 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| digital_v2 | 119 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| modelled_v2 | 19 | 0.06 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| sf2_perf | 100 | 0.08 | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 |
| otey_perf | 20 | 0.06 | 0.00 | 0.02 | 0.00 | 0.00 | 0.01 |

Format signature in one line: files with all 16 IQR features = 0 / NaN rate of Group B+C features -- manifest 0% / 4%; youtube_extra 0% / 4%; ccmusic 27% / 32%; soundfont_single 73% / 54%; otey_single 71% / 53%; old_demo 0% / 7%; m1 0% / 1%; library:Salamander 0% / 0%; library:K18 0% / 0%; library:FreePats 0% / 0%; library:Iowa 0% / 0%; digital_v2 0% / 0%; modelled_v2 0% / 1%; sf2_perf 0% / 1%; otey_perf 0% / 2%. A model can read "single-note file" from these two signatures alone; the 16 Group-A medians (feature set med16) contain neither, which is why every comparison is also reported for med16.

## 2. Detector results, corpus level (out-of-fold)

T1 = paper protocol (every synthetic kind labelled synthetic and in training). T2 = sample-based kinds (libraries, digital demos, modelled digital, M1) excluded from training; their scores are the mean over the 5 fold models (zero-shot). Algorithmic and real rows are out-of-fold in both. AUC CI = stratified grouped bootstrap (real groups and synthetic groups resampled separately, B = 2000, seed 0). "det." = share of synthetic files with P(synthetic) > 0.5; "mean P" = mean P(synthetic) of the synthetic files. Real false-alarm rates are in `out/comparisons.csv` (`fa_real`).

**Caveat for P1 (algorithmic performances):** the SoundFont and Otey performance renders are dry digital audio (no room, microphone or codec), while the real performances are YouTube recordings. P1 removes the single-note-vs-performance FORMAT confound but not the CHANNEL confound. Each performance render shares the CV group of the same SoundFont's (or Otey's) single-note renders, so no engine is in both the training and the test folds.

### 2.1 XGBoost, all 47 features (primary)

| id | comparison | files/groups real vs syn | T1 AUC [95% CI] | T1 det. | T1 mean P | T2 AUC [95% CI] | T2 det. | T2 mean P |
|---|---|---|---|---|---|---|---|---|
| I1 | real_isolated (CCMusic) vs algo_single (SoundFont renders + Otey) | 668/6 vs 882/11 | 0.908 [0.833, 0.966] | 84.8% | 0.813 | 0.917 [0.842, 0.971] | 83.1% | 0.813 |
| I1a | ... vs SoundFont single-note renders | 668/6 vs 762/10 | 0.924 [0.856, 0.970] | 88.5% | 0.849 | 0.934 [0.870, 0.976] | 88.3% | 0.861 |
| I1b | ... vs Otey single notes | 668/6 vs 120/1 (1 syn. group) | 0.809 [0.697, 0.898] | 61.7% | 0.586 | 0.807 [0.696, 0.900] | 50.0% | 0.506 |
| I2 | real_isolated (CCMusic) vs sampled_library (14-note takes; unit mismatch, see per-note table) | 668/6 vs 32/4 | 0.607 [0.400, 0.807] | 18.8% | 0.250 | 0.410 [0.204, 0.648] | 0.0% | 0.039 |
| P1 | real_performance vs algo_performance | 2576/2420 vs 120/11 | 0.951 [0.928, 0.970] | 57.5% | 0.564 | 0.955 [0.925, 0.980] | 37.5% | 0.391 |
| P1a | ... vs SoundFont performance renders (sf2_perf) | 2576/2420 vs 100/10 | 0.951 [0.924, 0.975] | 56.0% | 0.569 | 0.956 [0.921, 0.986] | 42.0% | 0.429 |
| P1b | ... vs Otey performance renders (otey_perf) | 2576/2420 vs 20/1 (1 syn. group) | 0.951 [0.946, 0.956] | 65.0% | 0.542 | 0.953 [0.948, 0.958] | 15.0% | 0.201 |
| P2 | real_performance vs sampled digital pianos (old demos + digital_v2) | 2576/2420 vs 128/7 | 0.915 [0.892, 0.936] | 34.4% | 0.359 | 0.887 [0.845, 0.917] | 3.1% | 0.089 |
| P2a | ... old demos only (SC-55 module excluded) | 2576/2420 vs 9/4 | 0.915 [0.876, 0.970] | 22.2% | 0.301 | 0.918 [0.869, 0.948] | 0.0% | 0.096 |
| P2d | ... digital_v2 only | 2576/2420 vs 119/7 | 0.915 [0.892, 0.937] | 35.3% | 0.364 | 0.885 [0.843, 0.915] | 3.4% | 0.088 |
| P2d-line | ... digital_v2, line capture | 2576/2420 vs 12/6 | 0.950 [0.894, 0.983] | 58.3% | 0.477 | 0.937 [0.881, 0.975] | 8.3% | 0.133 |
| P2d-mic | ... digital_v2, microphone capture | 2576/2420 vs 2/2 | 0.968 [0.950, 0.984] | 50.0% | 0.437 | 0.926 [0.861, 0.986] | 0.0% | 0.079 |
| P2d-unk | ... digital_v2, capture unknown | 2576/2420 vs 105/7 | 0.910 [0.883, 0.937] | 32.4% | 0.349 | 0.878 [0.835, 0.909] | 2.9% | 0.083 |
| P2d-lt | ... digital_v2, long_tails flag | 2576/2420 vs 10/6 | 0.916 [0.903, 0.940] | 10.0% | 0.257 | 0.803 [0.665, 0.877] | 0.0% | 0.111 |
| P2d-nolt | ... digital_v2, no long_tails flag | 2576/2420 vs 109/7 | 0.915 [0.890, 0.940] | 37.6% | 0.373 | 0.892 [0.849, 0.927] | 3.7% | 0.086 |
| P2d-rv | ... digital_v2, reverb stated (flag or description) | 2576/2420 vs 3/3 | 0.976 [0.961, 0.988] | 66.7% | 0.546 | 0.942 [0.904, 0.989] | 0.0% | 0.109 |
| P2d-norv | ... digital_v2, no reverb statement | 2576/2420 vs 116/7 | 0.914 [0.889, 0.936] | 34.5% | 0.359 | 0.883 [0.840, 0.914] | 3.4% | 0.088 |
| P3 | real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 2576/2420 vs 21/1 (1 syn. group) | 0.865 [0.859, 0.871] | 42.9% | 0.461 | 0.865 [0.857, 0.873] | 0.0% | 0.073 |
| P3d | ... digital_v2 modelled only | 2576/2420 vs 19/1 (1 syn. group) | 0.874 [0.869, 0.880] | 42.1% | 0.461 | 0.867 [0.860, 0.875] | 0.0% | 0.071 |
| P4 | real_performance vs Korg M1 | 2576/2420 vs 30/1 (1 syn. group) | 0.833 [0.824, 0.842] | 20.0% | 0.235 | 0.891 [0.884, 0.898] | 6.7% | 0.141 |
| P5 | real_performance vs digital_other (Roland GP-3, hybrid / unclear) | 2576/2420 vs 1/1 (1 syn. group) | 0.970 [0.964, 0.977] | 0.0% | 0.459 | 0.984 [0.979, 0.989] | 0.0% | 0.164 |
| P6 | real_performance vs Roland SC-55 sound module (not a digital piano) | 2576/2420 vs 2/1 (1 syn. group) | 0.998 [0.996, 1.000] | 100.0% | 0.947 | 1.000 [0.999, 1.000] | 100.0% | 0.785 |
| X1 | all real vs all algorithmic | 3244/2426 vs 1002/11 | 0.961 [0.938, 0.980] | 81.5% | 0.783 | 0.970 [0.947, 0.988] | 77.6% | 0.762 |
| X2 | all real vs sample-based (libraries + sampled digital + M1) | 3244/2426 vs 190/12 | 0.845 [0.793, 0.897] | 29.5% | 0.321 | 0.796 [0.722, 0.864] | 3.2% | 0.089 |
| X3 | all real vs all synthetic | 3244/2426 vs 1216/25 | 0.941 [0.910, 0.965] | 72.7% | 0.706 | 0.940 [0.901, 0.968] | 64.6% | 0.645 |
| C1 | real_performance vs algo_single (format-crossed) | 2576/2420 vs 882/11 | 0.982 [0.973, 0.991] | 84.8% | 0.813 | 0.994 [0.990, 0.998] | 83.1% | 0.813 |
| C2 | real_isolated vs sampled_digital_performance (format-crossed) | 668/6 vs 128/7 | 0.692 [0.553, 0.815] | 34.4% | 0.359 | 0.536 [0.384, 0.677] | 3.1% | 0.089 |
| C3 | real_isolated vs Korg M1 (format-crossed) | 668/6 vs 30/1 (1 syn. group) | 0.574 [0.441, 0.685] | 20.0% | 0.235 | 0.565 [0.426, 0.686] | 6.7% | 0.141 |

### 2.2 XGBoost, 16 Group-A medians only (per-note evidence)

| id | comparison | files/groups real vs syn | T1 AUC [95% CI] | T1 det. | T1 mean P | T2 AUC [95% CI] | T2 det. | T2 mean P |
|---|---|---|---|---|---|---|---|---|
| I1 | real_isolated (CCMusic) vs algo_single (SoundFont renders + Otey) | 668/6 vs 882/11 | 0.793 [0.692, 0.888] | 71.0% | 0.663 | 0.813 [0.723, 0.898] | 69.6% | 0.654 |
| I1a | ... vs SoundFont single-note renders | 668/6 vs 762/10 | 0.817 [0.728, 0.898] | 76.9% | 0.702 | 0.834 [0.748, 0.907] | 75.3% | 0.694 |
| I1b | ... vs Otey single notes | 668/6 vs 120/1 (1 syn. group) | 0.638 [0.548, 0.749] | 33.3% | 0.417 | 0.684 [0.598, 0.774] | 33.3% | 0.406 |
| I2 | real_isolated (CCMusic) vs sampled_library (14-note takes; unit mismatch, see per-note table) | 668/6 vs 32/4 | 0.412 [0.299, 0.541] | 9.4% | 0.175 | 0.379 [0.262, 0.520] | 6.2% | 0.113 |
| P1 | real_performance vs algo_performance | 2576/2420 vs 120/11 | 0.744 [0.652, 0.833] | 22.5% | 0.311 | 0.757 [0.631, 0.857] | 18.3% | 0.223 |
| P1a | ... vs SoundFont performance renders (sf2_perf) | 2576/2420 vs 100/10 | 0.756 [0.648, 0.855] | 25.0% | 0.334 | 0.759 [0.625, 0.876] | 22.0% | 0.251 |
| P1b | ... vs Otey performance renders (otey_perf) | 2576/2420 vs 20/1 (1 syn. group) | 0.686 [0.674, 0.698] | 10.0% | 0.198 | 0.748 [0.736, 0.760] | 0.0% | 0.085 |
| P2 | real_performance vs sampled digital pianos (old demos + digital_v2) | 2576/2420 vs 128/7 | 0.707 [0.669, 0.770] | 11.7% | 0.226 | 0.676 [0.621, 0.709] | 0.0% | 0.065 |
| P2a | ... old demos only (SC-55 module excluded) | 2576/2420 vs 9/4 | 0.786 [0.672, 0.978] | 33.3% | 0.359 | 0.800 [0.762, 0.901] | 0.0% | 0.096 |
| P2d | ... digital_v2 only | 2576/2420 vs 119/7 | 0.702 [0.663, 0.763] | 10.1% | 0.216 | 0.666 [0.607, 0.702] | 0.0% | 0.063 |
| P2d-line | ... digital_v2, line capture | 2576/2420 vs 12/6 | 0.710 [0.664, 0.796] | 8.3% | 0.209 | 0.748 [0.622, 0.816] | 0.0% | 0.068 |
| P2d-mic | ... digital_v2, microphone capture | 2576/2420 vs 2/2 | 0.700 [0.562, 0.835] | 0.0% | 0.152 | 0.712 [0.644, 0.779] | 0.0% | 0.039 |
| P2d-unk | ... digital_v2, capture unknown | 2576/2420 vs 105/7 | 0.701 [0.659, 0.764] | 10.5% | 0.218 | 0.656 [0.592, 0.699] | 0.0% | 0.063 |
| P2d-lt | ... digital_v2, long_tails flag | 2576/2420 vs 10/6 | 0.750 [0.685, 0.843] | 10.0% | 0.207 | 0.569 [0.317, 0.709] | 0.0% | 0.054 |
| P2d-nolt | ... digital_v2, no long_tails flag | 2576/2420 vs 109/7 | 0.697 [0.656, 0.760] | 10.1% | 0.217 | 0.675 [0.630, 0.703] | 0.0% | 0.064 |
| P2d-rv | ... digital_v2, reverb stated (flag or description) | 2576/2420 vs 3/3 | 0.722 [0.585, 0.822] | 0.0% | 0.156 | 0.762 [0.513, 0.914] | 0.0% | 0.093 |
| P2d-norv | ... digital_v2, no reverb statement | 2576/2420 vs 116/7 | 0.701 [0.660, 0.768] | 10.3% | 0.218 | 0.664 [0.605, 0.702] | 0.0% | 0.062 |
| P3 | real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 2576/2420 vs 21/1 (1 syn. group) | 0.812 [0.803, 0.821] | 28.6% | 0.354 | 0.685 [0.674, 0.697] | 4.8% | 0.110 |
| P3d | ... digital_v2 modelled only | 2576/2420 vs 19/1 (1 syn. group) | 0.820 [0.811, 0.829] | 31.6% | 0.369 | 0.693 [0.681, 0.706] | 5.3% | 0.104 |
| P4 | real_performance vs Korg M1 | 2576/2420 vs 30/1 (1 syn. group) | 0.803 [0.796, 0.811] | 30.0% | 0.393 | 0.889 [0.884, 0.893] | 43.3% | 0.450 |
| P5 | real_performance vs digital_other (Roland GP-3, hybrid / unclear) | 2576/2420 vs 1/1 (1 syn. group) | 0.855 [0.841, 0.868] | 0.0% | 0.265 | 0.961 [0.953, 0.969] | 0.0% | 0.327 |
| P6 | real_performance vs Roland SC-55 sound module (not a digital piano) | 2576/2420 vs 2/1 (1 syn. group) | 0.901 [0.891, 0.912] | 0.0% | 0.374 | 0.861 [0.849, 0.873] | 0.0% | 0.106 |
| X1 | all real vs all algorithmic | 3244/2426 vs 1002/11 | 0.876 [0.822, 0.928] | 65.2% | 0.621 | 0.903 [0.858, 0.946] | 63.5% | 0.603 |
| X2 | all real vs sample-based (libraries + sampled digital + M1) | 3244/2426 vs 190/12 | 0.660 [0.596, 0.724] | 14.2% | 0.244 | 0.642 [0.555, 0.725] | 7.9% | 0.134 |
| X3 | all real vs all synthetic | 3244/2426 vs 1216/25 | 0.841 [0.780, 0.887] | 56.4% | 0.557 | 0.857 [0.797, 0.902] | 53.6% | 0.520 |
| C1 | real_performance vs algo_single (format-crossed) | 2576/2420 vs 882/11 | 0.927 [0.888, 0.964] | 71.0% | 0.663 | 0.960 [0.934, 0.983] | 69.6% | 0.654 |
| C2 | real_isolated vs sampled_digital_performance (format-crossed) | 668/6 vs 128/7 | 0.494 [0.382, 0.630] | 11.7% | 0.226 | 0.388 [0.291, 0.483] | 0.0% | 0.065 |
| C3 | real_isolated vs Korg M1 (format-crossed) | 668/6 vs 30/1 (1 syn. group) | 0.613 [0.522, 0.724] | 30.0% | 0.393 | 0.693 [0.613, 0.778] | 43.3% | 0.450 |

### 2.3 Logistic regression, all 47 features

| id | comparison | files/groups real vs syn | T1 AUC [95% CI] | T1 det. | T1 mean P | T2 AUC [95% CI] | T2 det. | T2 mean P |
|---|---|---|---|---|---|---|---|---|
| I1 | real_isolated (CCMusic) vs algo_single (SoundFont renders + Otey) | 668/6 vs 882/11 | 0.798 [0.726, 0.866] | 91.0% | 0.818 | 0.820 [0.744, 0.891] | 88.1% | 0.823 |
| I1a | ... vs SoundFont single-note renders | 668/6 vs 762/10 | 0.808 [0.731, 0.872] | 91.1% | 0.828 | 0.835 [0.756, 0.900] | 89.9% | 0.842 |
| I1b | ... vs Otey single notes | 668/6 vs 120/1 (1 syn. group) | 0.739 [0.675, 0.796] | 90.8% | 0.753 | 0.727 [0.653, 0.800] | 76.7% | 0.702 |
| I2 | real_isolated (CCMusic) vs sampled_library (14-note takes; unit mismatch, see per-note table) | 668/6 vs 32/4 | 0.521 [0.319, 0.731] | 43.8% | 0.415 | 0.470 [0.318, 0.626] | 12.5% | 0.250 |
| P1 | real_performance vs algo_performance | 2576/2420 vs 120/11 | 0.828 [0.743, 0.898] | 46.7% | 0.481 | 0.758 [0.635, 0.865] | 31.7% | 0.347 |
| P1a | ... vs SoundFont performance renders (sf2_perf) | 2576/2420 vs 100/10 | 0.877 [0.844, 0.907] | 55.0% | 0.532 | 0.829 [0.775, 0.880] | 38.0% | 0.402 |
| P1b | ... vs Otey performance renders (otey_perf) | 2576/2420 vs 20/1 (1 syn. group) | 0.585 [0.574, 0.597] | 5.0% | 0.224 | 0.404 [0.392, 0.416] | 0.0% | 0.070 |
| P2 | real_performance vs sampled digital pianos (old demos + digital_v2) | 2576/2420 vs 128/7 | 0.825 [0.805, 0.859] | 37.5% | 0.440 | 0.763 [0.713, 0.813] | 15.6% | 0.283 |
| P2a | ... old demos only (SC-55 module excluded) | 2576/2420 vs 9/4 | 0.874 [0.843, 0.969] | 33.3% | 0.460 | 0.865 [0.812, 0.905] | 0.0% | 0.301 |
| P2d | ... digital_v2 only | 2576/2420 vs 119/7 | 0.821 [0.802, 0.855] | 37.8% | 0.439 | 0.756 [0.706, 0.805] | 16.8% | 0.281 |
| P2d-line | ... digital_v2, line capture | 2576/2420 vs 12/6 | 0.842 [0.765, 0.899] | 50.0% | 0.458 | 0.773 [0.616, 0.858] | 16.7% | 0.290 |
| P2d-mic | ... digital_v2, microphone capture | 2576/2420 vs 2/2 | 0.929 [0.874, 0.980] | 50.0% | 0.555 | 0.841 [0.700, 0.974] | 50.0% | 0.331 |
| P2d-unk | ... digital_v2, capture unknown | 2576/2420 vs 105/7 | 0.817 [0.794, 0.855] | 36.2% | 0.434 | 0.752 [0.707, 0.803] | 16.2% | 0.279 |
| P2d-lt | ... digital_v2, long_tails flag | 2576/2420 vs 10/6 | 0.755 [0.671, 0.895] | 30.0% | 0.357 | 0.666 [0.524, 0.805] | 10.0% | 0.194 |
| P2d-nolt | ... digital_v2, no long_tails flag | 2576/2420 vs 109/7 | 0.828 [0.809, 0.857] | 38.5% | 0.446 | 0.764 [0.711, 0.817] | 17.4% | 0.289 |
| P2d-rv | ... digital_v2, reverb stated (flag or description) | 2576/2420 vs 3/3 | 0.739 [0.510, 0.997] | 33.3% | 0.414 | 0.609 [0.278, 0.998] | 33.3% | 0.330 |
| P2d-norv | ... digital_v2, no reverb statement | 2576/2420 vs 116/7 | 0.824 [0.803, 0.857] | 37.9% | 0.439 | 0.759 [0.717, 0.803] | 16.4% | 0.280 |
| P3 | real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 2576/2420 vs 21/1 (1 syn. group) | 0.799 [0.789, 0.808] | 28.6% | 0.427 | 0.689 [0.679, 0.700] | 9.5% | 0.232 |
| P3d | ... digital_v2 modelled only | 2576/2420 vs 19/1 (1 syn. group) | 0.814 [0.805, 0.823] | 31.6% | 0.445 | 0.708 [0.698, 0.719] | 10.5% | 0.245 |
| P4 | real_performance vs Korg M1 | 2576/2420 vs 30/1 (1 syn. group) | 0.695 [0.687, 0.703] | 33.3% | 0.375 | 0.713 [0.704, 0.722] | 23.3% | 0.290 |
| P5 | real_performance vs digital_other (Roland GP-3, hybrid / unclear) | 2576/2420 vs 1/1 (1 syn. group) | 0.976 [0.969, 0.982] | 100.0% | 0.695 | 0.817 [0.802, 0.832] | 0.0% | 0.215 |
| P6 | real_performance vs Roland SC-55 sound module (not a digital piano) | 2576/2420 vs 2/1 (1 syn. group) | 0.970 [0.965, 0.974] | 100.0% | 0.746 | 0.926 [0.919, 0.932] | 50.0% | 0.585 |
| X1 | all real vs all algorithmic | 3244/2426 vs 1002/11 | 0.897 [0.859, 0.936] | 85.7% | 0.777 | 0.895 [0.848, 0.937] | 81.3% | 0.766 |
| X2 | all real vs sample-based (libraries + sampled digital + M1) | 3244/2426 vs 190/12 | 0.728 [0.659, 0.797] | 37.9% | 0.426 | 0.692 [0.632, 0.751] | 16.3% | 0.278 |
| X3 | all real vs all synthetic | 3244/2426 vs 1216/25 | 0.868 [0.824, 0.907] | 77.3% | 0.716 | 0.859 [0.808, 0.901] | 69.8% | 0.680 |
| C1 | real_performance vs algo_single (format-crossed) | 2576/2420 vs 882/11 | 0.944 [0.918, 0.970] | 91.0% | 0.818 | 0.945 [0.918, 0.974] | 88.1% | 0.823 |
| C2 | real_isolated vs sampled_digital_performance (format-crossed) | 668/6 vs 128/7 | 0.556 [0.469, 0.643] | 37.5% | 0.440 | 0.517 [0.416, 0.627] | 15.6% | 0.283 |
| C3 | real_isolated vs Korg M1 (format-crossed) | 668/6 vs 30/1 (1 syn. group) | 0.513 [0.428, 0.599] | 33.3% | 0.375 | 0.511 [0.411, 0.621] | 23.3% | 0.290 |

### 2.4 Logistic regression, 16 Group-A medians only

| id | comparison | files/groups real vs syn | T1 AUC [95% CI] | T1 det. | T1 mean P | T2 AUC [95% CI] | T2 det. | T2 mean P |
|---|---|---|---|---|---|---|---|---|
| I1 | real_isolated (CCMusic) vs algo_single (SoundFont renders + Otey) | 668/6 vs 882/11 | 0.683 [0.608, 0.749] | 67.0% | 0.595 | 0.693 [0.617, 0.761] | 65.2% | 0.582 |
| I1a | ... vs SoundFont single-note renders | 668/6 vs 762/10 | 0.676 [0.598, 0.747] | 65.4% | 0.587 | 0.688 [0.611, 0.761] | 63.8% | 0.575 |
| I1b | ... vs Otey single notes | 668/6 vs 120/1 (1 syn. group) | 0.728 [0.705, 0.765] | 77.5% | 0.645 | 0.729 [0.698, 0.769] | 74.2% | 0.627 |
| I2 | real_isolated (CCMusic) vs sampled_library (14-note takes; unit mismatch, see per-note table) | 668/6 vs 32/4 | 0.602 [0.483, 0.713] | 62.5% | 0.509 | 0.599 [0.483, 0.724] | 43.8% | 0.469 |
| P1 | real_performance vs algo_performance | 2576/2420 vs 120/11 | 0.587 [0.477, 0.692] | 33.3% | 0.430 | 0.573 [0.457, 0.683] | 30.0% | 0.414 |
| P1a | ... vs SoundFont performance renders (sf2_perf) | 2576/2420 vs 100/10 | 0.532 [0.453, 0.607] | 25.0% | 0.403 | 0.515 [0.435, 0.593] | 23.0% | 0.386 |
| P1b | ... vs Otey performance renders (otey_perf) | 2576/2420 vs 20/1 (1 syn. group) | 0.863 [0.856, 0.870] | 75.0% | 0.564 | 0.863 [0.856, 0.871] | 65.0% | 0.556 |
| P2 | real_performance vs sampled digital pianos (old demos + digital_v2) | 2576/2420 vs 128/7 | 0.747 [0.718, 0.784] | 39.8% | 0.475 | 0.736 [0.714, 0.761] | 26.6% | 0.454 |
| P2a | ... old demos only (SC-55 module excluded) | 2576/2420 vs 9/4 | 0.796 [0.689, 0.943] | 55.6% | 0.497 | 0.791 [0.682, 0.923] | 33.3% | 0.476 |
| P2d | ... digital_v2 only | 2576/2420 vs 119/7 | 0.743 [0.717, 0.781] | 38.7% | 0.474 | 0.731 [0.710, 0.758] | 26.1% | 0.453 |
| P2d-line | ... digital_v2, line capture | 2576/2420 vs 12/6 | 0.748 [0.665, 0.828] | 50.0% | 0.472 | 0.748 [0.694, 0.817] | 16.7% | 0.455 |
| P2d-mic | ... digital_v2, microphone capture | 2576/2420 vs 2/2 | 0.213 [0.073, 0.359] | 0.0% | 0.304 | 0.235 [0.109, 0.367] | 0.0% | 0.297 |
| P2d-unk | ... digital_v2, capture unknown | 2576/2420 vs 105/7 | 0.753 [0.719, 0.799] | 38.1% | 0.477 | 0.739 [0.712, 0.773] | 27.6% | 0.456 |
| P2d-lt | ... digital_v2, long_tails flag | 2576/2420 vs 10/6 | 0.706 [0.623, 0.898] | 20.0% | 0.480 | 0.738 [0.670, 0.893] | 20.0% | 0.459 |
| P2d-nolt | ... digital_v2, no long_tails flag | 2576/2420 vs 109/7 | 0.747 [0.720, 0.779] | 40.4% | 0.473 | 0.731 [0.708, 0.754] | 26.6% | 0.452 |
| P2d-rv | ... digital_v2, reverb stated (flag or description) | 2576/2420 vs 3/3 | 0.787 [0.619, 0.956] | 33.3% | 0.488 | 0.794 [0.669, 0.930] | 33.3% | 0.471 |
| P2d-norv | ... digital_v2, no reverb statement | 2576/2420 vs 116/7 | 0.742 [0.716, 0.782] | 38.8% | 0.473 | 0.730 [0.707, 0.757] | 25.9% | 0.452 |
| P3 | real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 2576/2420 vs 21/1 (1 syn. group) | 0.697 [0.687, 0.708] | 33.3% | 0.464 | 0.643 [0.633, 0.653] | 14.3% | 0.427 |
| P3d | ... digital_v2 modelled only | 2576/2420 vs 19/1 (1 syn. group) | 0.702 [0.692, 0.712] | 31.6% | 0.466 | 0.648 [0.638, 0.659] | 15.8% | 0.429 |
| P4 | real_performance vs Korg M1 | 2576/2420 vs 30/1 (1 syn. group) | 0.820 [0.814, 0.826] | 70.0% | 0.533 | 0.812 [0.806, 0.818] | 73.3% | 0.532 |
| P5 | real_performance vs digital_other (Roland GP-3, hybrid / unclear) | 2576/2420 vs 1/1 (1 syn. group) | 0.965 [0.958, 0.972] | 100.0% | 0.593 | 0.961 [0.954, 0.968] | 100.0% | 0.588 |
| P6 | real_performance vs Roland SC-55 sound module (not a digital piano) | 2576/2420 vs 2/1 (1 syn. group) | 0.595 [0.585, 0.605] | 50.0% | 0.443 | 0.542 [0.531, 0.553] | 0.0% | 0.394 |
| X1 | all real vs all algorithmic | 3244/2426 vs 1002/11 | 0.730 [0.653, 0.792] | 63.0% | 0.575 | 0.729 [0.644, 0.796] | 61.0% | 0.562 |
| X2 | all real vs sample-based (libraries + sampled digital + M1) | 3244/2426 vs 190/12 | 0.721 [0.665, 0.778] | 48.4% | 0.490 | 0.711 [0.656, 0.759] | 36.8% | 0.469 |
| X3 | all real vs all synthetic | 3244/2426 vs 1216/25 | 0.728 [0.662, 0.785] | 60.2% | 0.560 | 0.724 [0.653, 0.785] | 56.3% | 0.545 |
| C1 | real_performance vs algo_single (format-crossed) | 2576/2420 vs 882/11 | 0.769 [0.676, 0.840] | 67.0% | 0.595 | 0.766 [0.661, 0.845] | 65.2% | 0.582 |
| C2 | real_isolated vs sampled_digital_performance (format-crossed) | 668/6 vs 128/7 | 0.569 [0.524, 0.628] | 39.8% | 0.475 | 0.584 [0.534, 0.647] | 26.6% | 0.454 |
| C3 | real_isolated vs Korg M1 (format-crossed) | 668/6 vs 30/1 (1 syn. group) | 0.626 [0.594, 0.677] | 70.0% | 0.533 | 0.656 [0.620, 0.703] | 73.3% | 0.532 |

**Real-class false alarms** (share of real files with P > 0.5, mean P):

| run | real kind | files | false alarms | mean P |
|---|---|---|---|---|
| xgb_all47_T1 | real_isolated | 668 | 19.9% | 0.221 |
| xgb_all47_T1 | real_performance | 2576 | 2.5% | 0.048 |
| xgb_all47_T2 | real_isolated | 668 | 16.0% | 0.180 |
| xgb_all47_T2 | real_performance | 2576 | 0.3% | 0.011 |
| xgb_med16_T1 | real_isolated | 668 | 27.2% | 0.307 |
| xgb_med16_T1 | real_performance | 2576 | 5.5% | 0.129 |
| xgb_med16_T2 | real_isolated | 668 | 22.6% | 0.258 |
| xgb_med16_T2 | real_performance | 2576 | 1.9% | 0.056 |
| lr_all47_T1 | real_isolated | 668 | 41.5% | 0.410 |
| lr_all47_T1 | real_performance | 2576 | 7.1% | 0.181 |
| lr_all47_T2 | real_isolated | 668 | 38.3% | 0.375 |
| lr_all47_T2 | real_performance | 2576 | 3.4% | 0.123 |
| lr_med16_T1 | real_isolated | 668 | 40.6% | 0.402 |
| lr_med16_T1 | real_performance | 2576 | 13.5% | 0.392 |
| lr_med16_T2 | real_isolated | 668 | 37.9% | 0.381 |
| lr_med16_T2 | real_performance | 2576 | 11.9% | 0.377 |

## 2b. Digital pianos (old demos + data_v2)

CV groups and the primary bootstrap use the BRAND for every digital piano (engines are shared within a brand; the SC-55 module is its own group); "CI by model" re-runs the same bootstrap with one group per model (sensitivity; same scores). Excluded: Casio_PXS1000_0Si1TBz9b6I.wav: Casio PX-S1000 demo with EXTERNAL convolution reverb mixed in (manifest flag external_reverb_added). T2 = digital pianos never in training (mean of the 5 fold models); T1 = in training with label synthetic (brand-grouped folds). Capture and flags come from `data_v2/digital_v2_manifest.csv` (old demos have no capture information).

| id | comparison | files / brands / models | XGB all47 T2 AUC [CI by brand] | CI by model | det. | XGB med16 T2 AUC [CI by brand] | CI by model | det. | XGB all47 T1 AUC [CI by brand] | CI by model | det. | XGB med16 T1 AUC [CI by brand] | CI by model | det. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | real_performance vs sampled digital pianos (old demos + digital_v2) | 128 / 7 / 53 | 0.887 [0.845, 0.917] | [0.858, 0.911] | 3.1% | 0.676 [0.621, 0.709] | [0.632, 0.720] | 0.0% | 0.915 [0.892, 0.936] | [0.897, 0.932] | 34.4% | 0.707 [0.669, 0.770] | [0.667, 0.750] | 11.7% |
| P2a | ... old demos only (SC-55 module excluded) | 9 / 4 / 9 | 0.918 [0.869, 0.948] | [0.868, 0.966] | 0.0% | 0.800 [0.762, 0.901] | [0.729, 0.874] | 0.0% | 0.915 [0.876, 0.970] | [0.872, 0.958] | 22.2% | 0.786 [0.672, 0.978] | [0.624, 0.916] | 33.3% |
| P2d | ... digital_v2 only | 119 / 7 / 45 | 0.885 [0.843, 0.915] | [0.857, 0.911] | 3.4% | 0.666 [0.607, 0.702] | [0.619, 0.712] | 0.0% | 0.915 [0.892, 0.937] | [0.897, 0.933] | 35.3% | 0.702 [0.663, 0.763] | [0.662, 0.744] | 10.1% |
| P2d-line | ... digital_v2, line capture | 12 / 6 / 11 | 0.937 [0.881, 0.975] | [0.896, 0.967] | 8.3% | 0.748 [0.622, 0.816] | [0.664, 0.822] | 0.0% | 0.950 [0.894, 0.983] | [0.914, 0.977] | 58.3% | 0.710 [0.664, 0.796] | [0.614, 0.787] | 8.3% |
| P2d-mic | ... digital_v2, microphone capture | 2 / 2 / 2 | 0.926 [0.861, 0.986] | [0.861, 0.986] | 0.0% | 0.712 [0.644, 0.779] | [0.644, 0.779] | 0.0% | 0.968 [0.950, 0.984] | [0.950, 0.984] | 50.0% | 0.700 [0.562, 0.835] | [0.562, 0.835] | 0.0% |
| P2d-unk | ... digital_v2, capture unknown | 105 / 7 / 45 | 0.878 [0.835, 0.909] | [0.848, 0.906] | 2.9% | 0.656 [0.592, 0.699] | [0.604, 0.706] | 0.0% | 0.910 [0.883, 0.937] | [0.890, 0.929] | 32.4% | 0.701 [0.659, 0.764] | [0.659, 0.746] | 10.5% |
| P2d-lt | ... digital_v2, long_tails flag | 10 / 6 / 9 | 0.803 [0.665, 0.877] | [0.705, 0.888] | 0.0% | 0.569 [0.317, 0.709] | [0.388, 0.714] | 0.0% | 0.916 [0.903, 0.940] | [0.882, 0.948] | 10.0% | 0.750 [0.685, 0.843] | [0.668, 0.826] | 10.0% |
| P2d-nolt | ... digital_v2, no long_tails flag | 109 / 7 / 45 | 0.892 [0.849, 0.927] | [0.863, 0.919] | 3.7% | 0.675 [0.630, 0.703] | [0.630, 0.719] | 0.0% | 0.915 [0.890, 0.940] | [0.896, 0.933] | 37.6% | 0.697 [0.656, 0.760] | [0.653, 0.743] | 10.1% |
| P2d-rv | ... digital_v2, reverb stated (flag or description) | 3 / 3 / 3 | 0.942 [0.904, 0.989] | [0.904, 0.989] | 0.0% | 0.762 [0.513, 0.914] | [0.513, 0.914] | 0.0% | 0.976 [0.961, 0.988] | [0.961, 0.988] | 66.7% | 0.722 [0.585, 0.822] | [0.585, 0.822] | 0.0% |
| P2d-norv | ... digital_v2, no reverb statement | 116 / 7 / 45 | 0.883 [0.840, 0.914] | [0.855, 0.910] | 3.4% | 0.664 [0.605, 0.702] | [0.617, 0.711] | 0.0% | 0.914 [0.889, 0.936] | [0.895, 0.931] | 34.5% | 0.701 [0.660, 0.768] | [0.660, 0.744] | 10.3% |
| P3 | real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 21 / 1 / 9 | 0.865 [0.857, 0.873] | [0.755, 0.943] | 0.0% | 0.685 [0.674, 0.697] | [0.583, 0.783] | 4.8% | 0.865 [0.859, 0.871] | [0.715, 0.970] | 42.9% | 0.812 [0.803, 0.821] | [0.707, 0.899] | 28.6% |
| P3d | ... digital_v2 modelled only | 19 / 1 / 7 | 0.867 [0.860, 0.875] | [0.738, 0.948] | 0.0% | 0.693 [0.681, 0.706] | [0.589, 0.782] | 5.3% | 0.874 [0.869, 0.880] | [0.702, 0.980] | 42.1% | 0.820 [0.811, 0.829] | [0.704, 0.913] | 31.6% |
| P5 | real_performance vs digital_other (Roland GP-3, hybrid / unclear) | 1 / 1 / 1 | 0.984 [0.979, 0.989] | [0.979, 0.989] | 0.0% | 0.961 [0.953, 0.969] | [0.953, 0.969] | 0.0% | 0.970 [0.964, 0.977] | [0.964, 0.977] | 0.0% | 0.855 [0.841, 0.868] | [0.841, 0.868] | 0.0% |
| P6 | real_performance vs Roland SC-55 sound module (not a digital piano) | 2 / 1 / 1 | 1.000 [0.999, 1.000] | [0.999, 1.000] | 100.0% | 0.861 [0.849, 0.873] | [0.849, 0.873] | 0.0% | 0.998 [0.996, 1.000] | [0.996, 1.000] | 100.0% | 0.901 [0.891, 0.912] | [0.891, 0.912] | 0.0% |

Real performances: 2576 files; their false-alarm rate is in section 2.

By brand (digital_v2 sample-based, XGB all47 T2): casio 16 files AUC 0.80, dexibell 12 files AUC 0.90, kawai 31 files AUC 0.89, korg 14 files AUC 0.92, kurzweil 8 files AUC 0.88, nord 10 files AUC 0.81, yamaha 28 files AUC 0.93.

### 2b.1 Level convention: raw vs peak-normalised digital-piano audio

The real corpus was peak-normalised to -1 dBFS and stored as PCM-16; the digital-piano excerpts were analysed as float at their downloaded level. `data_v2/*_peaknorm.csv` are the same excerpts after -1 dBFS PCM-16 normalisation (all 152 digital rows change). Same folds and models; under T2 only the scored features change, under T1 the training rows change too.

| id | files | XGB all47 T2: raw -> peaknorm [CI] | XGB med16 T2 | XGB all47 T1 | XGB med16 T1 |
|---|---|---|---|---|---|
| P2 | 128 | 0.887 -> 0.889 [0.848, 0.915]; det 3.1% -> 3.1% | 0.676 -> 0.680 [0.618, 0.718]; det 0.0% -> 0.0% | 0.915 -> 0.909 [0.885, 0.932]; det 34.4% -> 32.8% | 0.707 -> 0.720 [0.673, 0.783]; det 11.7% -> 14.1% |
| P2a | 9 | 0.918 -> 0.923 [0.878, 0.950]; det 0.0% -> 0.0% | 0.800 -> 0.761 [0.686, 0.918]; det 0.0% -> 0.0% | 0.915 -> 0.898 [0.863, 0.955]; det 22.2% -> 11.1% | 0.786 -> 0.750 [0.569, 0.972]; det 33.3% -> 33.3% |
| P2d | 119 | 0.885 -> 0.887 [0.846, 0.914]; det 3.4% -> 3.4% | 0.666 -> 0.674 [0.607, 0.713]; det 0.0% -> 0.0% | 0.915 -> 0.910 [0.884, 0.934]; det 35.3% -> 34.5% | 0.702 -> 0.718 [0.676, 0.778]; det 10.1% -> 12.6% |
| P2d-line | 12 | 0.937 -> 0.941 [0.878, 0.976]; det 8.3% -> 0.0% | 0.748 -> 0.735 [0.598, 0.805]; det 0.0% -> 0.0% | 0.950 -> 0.945 [0.878, 0.978]; det 58.3% -> 41.7% | 0.710 -> 0.701 [0.628, 0.804]; det 8.3% -> 8.3% |
| P2d-unk | 105 | 0.878 -> 0.879 [0.835, 0.908]; det 2.9% -> 3.8% | 0.656 -> 0.667 [0.591, 0.715]; det 0.0% -> 0.0% | 0.910 -> 0.905 [0.876, 0.933]; det 32.4% -> 32.4% | 0.701 -> 0.722 [0.673, 0.791]; det 10.5% -> 13.3% |
| P3 | 21 | 0.865 -> 0.864 [0.856, 0.872]; det 0.0% -> 0.0% | 0.685 -> 0.683 [0.672, 0.696]; det 4.8% -> 4.8% | 0.865 -> 0.856 [0.851, 0.861]; det 42.9% -> 47.6% | 0.812 -> 0.783 [0.774, 0.793]; det 28.6% -> 28.6% |
| P6 | 2 | 1.000 -> 1.000 [0.999, 1.000]; det 100.0% -> 100.0% | 0.861 -> 0.890 [0.879, 0.901]; det 0.0% -> 0.0% | 0.998 -> 0.999 [0.998, 1.000]; det 100.0% -> 100.0% | 0.901 -> 0.974 [0.968, 0.980]; det 0.0% -> 100.0% |

Largest per-feature changes raw -> peaknorm on the digital rows (median |change| in units of the feature's inter-file IQR over the whole corpus): ref_offset_abs 0.08, centroid_var_med 0.04, centroid_var_iqr 0.04, hf_decay_excess_med 0.03, hf_decay_excess_iqr 0.03, med_spread_cents 0.03, bright_decay_med 0.03, attack_time_med 0.03, log_t60_iqr 0.03, bright_decay_iqr 0.03. Median over all 47 features: 0.020 (controlled level test: section 8).

**Algorithmic single notes, detection by SoundFont and by Otey velocity** (out-of-fold, P > 0.5; the listening test used Casio CA-110 at v100 and Otey at v127):

| unit | files | fold | XGB all47 T1 | XGB all47 T2 | XGB med16 T2 |
|---|---|---|---|---|---|
| Otey v127 | 30 | 2 | 33.3% | 23.3% | 26.7% |
| Otey v25 | 30 | 2 | 83.3% | 76.7% | 43.3% |
| Otey v50 | 30 | 2 | 70.0% | 60.0% | 30.0% |
| Otey v75 | 30 | 2 | 60.0% | 40.0% | 33.3% |
| SoundFont casioca110pianofm | 78 | 0 | 73.1% | 65.4% | 80.8% |
| SoundFont casiopx860concertgrandpiano | 78 | 1 | 79.5% | 83.3% | 67.9% |
| SoundFont cp801644100 | 78 | 1 | 84.6% | 84.6% | 84.6% |
| SoundFont ctk230soundfont | 78 | 2 | 100.0% | 100.0% | 88.5% |
| SoundFont ctk533piano100casio | 63 | 4 | 100.0% | 100.0% | 100.0% |
| SoundFont jv1080nicepianom | 78 | 3 | 97.4% | 98.7% | 82.1% |
| SoundFont u20piano | 75 | 3 | 80.0% | 88.0% | 76.0% |
| SoundFont yamahapsr400500 | 78 | 0 | 96.2% | 96.2% | 84.6% |
| SoundFont yamahapsrf50 | 78 | 4 | 96.2% | 88.5% | 42.3% |
| SoundFont yamahas90esnaturalgrand | 78 | 2 | 79.5% | 80.8% | 51.3% |

## 3. Per-note isolated regime (note-for-note comparison)

Feasibility: per-note descriptors are cached only for real files (`quality/cache/notes_cal/`, incl. the 668 CCMusic files); single-note renders, Otey notes and library notes have no per-note cache, but every one of them is a single-note audio file present locally, and the single-file pipeline (`note_raw_descriptors` + `aggregate_file`, `register_trends_cal.json`) re-extracts the stored corpus rows exactly (check below). So the evaluation is feasible: each library note file is featurised exactly like a CCMusic note file, a SoundFont render or an Otey render (one file = one note). CCMusic, SoundFont and Otey rows are the corpus rows (OOF scores from section 2); library notes are new rows, never seen in training: under **T1** each library note is scored by the fold model whose test fold holds that library (it was trained with the other three libraries' 14-note takes labelled synthetic, as in the paper); under **T2** by the mean of the 5 fold models (no library in training). **I-only** = a detector trained only on isolated notes (CCMusic vs SoundFont renders + Otey) with the same fixed folds; library notes scored by the mean of its 5 fold models.

| library | note files | valid (>=30 features) | mf-equivalent layer files | median duration (s) | median segments found |
|---|---|---|---|---|---|
| FreePats | 78 | 78 | 78 | 7.1 | 2 |
| Iowa | 261 | 261 | 86 | 30.9 | 2 |
| K18 | 180 | 180 | 30 | 19.8 | 2 |
| Salamander | 480 | 480 | 30 | 14.8 | 5 |

mf-equivalent layer = Salamander v8 (of 16), Iowa mf, K18 mf, FreePats single layer (the layers used in the listening test). CCMusic note files: median 4.3 s; SoundFont renders 1.8 s; Otey 3.0 s.

**Reproducibility of the single-file pipeline:** all 1550 isolated-note corpus files (CCMusic, SoundFont renders, Otey) were re-extracted from audio with the current code: max |difference| to the stored CSV rows = 5.0e-07 (CSV rounding), 0 missing-value mismatches, 0 errors.

**Segmentation of single-note files** (the onset detector of `detune/segment.py` decides between "single/chord" mode, one window from the onset, and "piece" mode, several windows of at most 2.5 s; "segments" = per-note descriptor sets that enter the median/IQR). One audio file = one struck note in every row, but long library notes are often split into several windows, so their IQR features are less often 0:

| source | files | share single/chord mode | median segments | mean segments | max segments | share with <= 1 segment |
|---|---|---|---|---|---|---|
| ccmusic | 668 | 0.58 | 3 | 2.7 | 36 | 0.27 |
| soundfont_single | 762 | 0.15 | 1 | 1.6 | 11 | 0.75 |
| otey_single | 120 | 0.74 | 1 | 1.4 | 3 | 0.72 |
| library Salamander (all layers) | 480 | 0.09 | 5 | 11.0 | 80 | 0.12 |
| library Iowa (all layers) | 261 | 0.88 | 2 | 2.1 | 16 | 0.44 |
| library K18 (all layers) | 180 | 0.61 | 2 | 3.5 | 54 | 0.47 |
| library FreePats (all layers) | 78 | 0.13 | 2 | 4.0 | 53 | 0.40 |
| listening clips (3 s) | 30 | 0.13 | 1 | 1.8 | 7 | 0.53 |

### 3.1 XGBoost, all 47 features

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T2 AUC [CI] | T2 det. | I-only AUC [CI] | I-only det. |
|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.924 [0.856, 0.970] | 88.5% | 0.934 [0.870, 0.976] | 88.3% | 0.929 [0.856, 0.975] | 87.0% |
| Otey notes | 668/6 vs 120/1 | 0.809 [0.697, 0.898] | 61.7% | 0.807 [0.696, 0.900] | 50.0% | 0.779 [0.640, 0.891] | 38.3% |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.440 [0.282, 0.627] | 12.9% | 0.530 [0.370, 0.708] | 13.4% | 0.551 [0.384, 0.723] | 11.2% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.506 [0.307, 0.656] | 17.9% | 0.570 [0.389, 0.720] | 14.0% | 0.600 [0.403, 0.763] | 13.1% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.615 [0.477, 0.730] | 26.7% | 0.659 [0.514, 0.786] | 20.0% | 0.665 [0.505, 0.802] | 16.7% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.282 [0.184, 0.369] | 1.2% | 0.388 [0.276, 0.501] | 2.3% | 0.431 [0.293, 0.567] | 0.0% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.537 [0.427, 0.630] | 16.7% | 0.627 [0.512, 0.727] | 20.0% | 0.642 [0.510, 0.760] | 20.0% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.511 [0.394, 0.613] | 19.2% | 0.600 [0.475, 0.711] | 20.5% | 0.604 [0.467, 0.725] | 17.9% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.607 [0.473, 0.717] | 24.4% | 0.638 [0.496, 0.763] | 14.6% | 0.675 [0.516, 0.811] | 14.4% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.306 [0.213, 0.389] | 5.0% | 0.395 [0.287, 0.504] | 4.6% | 0.426 [0.298, 0.553] | 3.4% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.524 [0.411, 0.623] | 18.9% | 0.628 [0.505, 0.736] | 23.3% | 0.649 [0.511, 0.769] | 21.7% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.511 [0.394, 0.613] | 19.2% | 0.600 [0.475, 0.711] | 20.5% | 0.604 [0.467, 0.725] | 17.9% |

### 3.2 XGBoost, 16 Group-A medians

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T2 AUC [CI] | T2 det. | I-only AUC [CI] | I-only det. |
|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.817 [0.728, 0.898] | 76.9% | 0.834 [0.748, 0.907] | 75.3% | 0.849 [0.744, 0.931] | 75.6% |
| Otey notes | 668/6 vs 120/1 | 0.638 [0.548, 0.749] | 33.3% | 0.684 [0.598, 0.774] | 33.3% | 0.606 [0.478, 0.726] | 24.2% |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.345 [0.204, 0.517] | 9.8% | 0.445 [0.299, 0.613] | 12.9% | 0.476 [0.316, 0.666] | 12.1% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.410 [0.239, 0.550] | 16.1% | 0.518 [0.333, 0.654] | 21.2% | 0.560 [0.360, 0.715] | 22.7% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.424 [0.344, 0.517] | 13.3% | 0.552 [0.465, 0.643] | 20.0% | 0.609 [0.482, 0.732] | 26.7% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.196 [0.131, 0.274] | 1.2% | 0.281 [0.217, 0.350] | 1.2% | 0.331 [0.230, 0.439] | 0.0% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.503 [0.414, 0.614] | 23.3% | 0.590 [0.507, 0.675] | 33.3% | 0.613 [0.491, 0.728] | 26.7% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.418 [0.328, 0.528] | 12.8% | 0.528 [0.441, 0.619] | 15.4% | 0.532 [0.410, 0.650] | 14.1% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.468 [0.381, 0.574] | 20.4% | 0.588 [0.501, 0.680] | 26.0% | 0.644 [0.519, 0.762] | 32.3% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.226 [0.159, 0.304] | 3.1% | 0.308 [0.241, 0.379] | 5.0% | 0.360 [0.255, 0.469] | 1.9% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.518 [0.429, 0.627] | 25.0% | 0.631 [0.547, 0.719] | 34.4% | 0.638 [0.515, 0.755] | 31.1% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.418 [0.328, 0.528] | 12.8% | 0.528 [0.441, 0.619] | 15.4% | 0.532 [0.410, 0.650] | 14.1% |

### 3.3 Logistic regression, all 47 features

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T2 AUC [CI] | T2 det. | I-only AUC [CI] | I-only det. |
|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.808 [0.731, 0.872] | 91.1% | 0.835 [0.756, 0.900] | 89.9% | 0.921 [0.849, 0.977] | 92.3% |
| Otey notes | 668/6 vs 120/1 | 0.739 [0.675, 0.796] | 90.8% | 0.727 [0.653, 0.800] | 76.7% | 0.811 [0.700, 0.919] | 66.7% |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.499 [0.382, 0.636] | 39.7% | 0.532 [0.422, 0.653] | 36.6% | 0.535 [0.380, 0.699] | 16.1% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.542 [0.380, 0.654] | 46.6% | 0.540 [0.414, 0.655] | 38.4% | 0.522 [0.367, 0.698] | 17.4% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.639 [0.572, 0.704] | 63.3% | 0.607 [0.517, 0.703] | 43.3% | 0.473 [0.338, 0.627] | 10.0% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.386 [0.317, 0.453] | 19.8% | 0.451 [0.370, 0.538] | 26.7% | 0.463 [0.337, 0.602] | 5.8% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.601 [0.524, 0.672] | 60.0% | 0.621 [0.532, 0.710] | 56.7% | 0.676 [0.557, 0.804] | 36.7% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.530 [0.455, 0.607] | 44.9% | 0.559 [0.468, 0.657] | 37.2% | 0.583 [0.456, 0.721] | 21.8% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.603 [0.531, 0.675] | 52.5% | 0.566 [0.476, 0.663] | 36.9% | 0.507 [0.376, 0.654] | 15.6% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.371 [0.307, 0.434] | 26.1% | 0.424 [0.346, 0.508] | 27.6% | 0.424 [0.309, 0.553] | 8.8% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.630 [0.555, 0.700] | 61.7% | 0.628 [0.541, 0.718] | 58.9% | 0.680 [0.552, 0.816] | 32.8% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.530 [0.455, 0.607] | 44.9% | 0.559 [0.468, 0.657] | 37.2% | 0.583 [0.456, 0.721] | 21.8% |

### 3.4 Logistic regression, 16 Group-A medians

| CCMusic notes vs | files/groups | T1 AUC [CI] | T1 det. | T2 AUC [CI] | T2 det. | I-only AUC [CI] | I-only det. |
|---|---|---|---|---|---|---|---|
| SoundFont renders | 668/6 vs 762/10 | 0.676 [0.598, 0.747] | 65.4% | 0.688 [0.611, 0.761] | 63.8% | 0.845 [0.710, 0.941] | 86.6% |
| Otey notes | 668/6 vs 120/1 | 0.728 [0.705, 0.765] | 77.5% | 0.729 [0.698, 0.769] | 74.2% | 0.768 [0.609, 0.892] | 67.5% |
| library notes: all 4 libraries, mf layer | 668/6 vs 224/4 | 0.464 [0.329, 0.645] | 33.9% | 0.486 [0.357, 0.674] | 33.9% | 0.482 [0.302, 0.696] | 19.6% |
| library notes: all 4 libraries, all layers | 668/6 vs 999/4 | 0.522 [0.347, 0.645] | 41.8% | 0.545 [0.368, 0.666] | 43.2% | 0.537 [0.315, 0.719] | 28.0% |
| library notes: Salamander, mf layer | 668/6 vs 30/1 | 0.567 [0.533, 0.621] | 50.0% | 0.604 [0.569, 0.654] | 50.0% | 0.551 [0.419, 0.693] | 26.7% |
| library notes: Iowa, mf layer | 668/6 vs 86/1 | 0.302 [0.256, 0.379] | 10.5% | 0.331 [0.283, 0.406] | 12.8% | 0.319 [0.207, 0.450] | 2.3% |
| library notes: K18, mf layer | 668/6 vs 30/1 | 0.675 [0.646, 0.720] | 66.7% | 0.706 [0.674, 0.746] | 70.0% | 0.686 [0.534, 0.824] | 46.7% |
| library notes: FreePats, mf layer | 668/6 vs 78/1 | 0.521 [0.482, 0.586] | 41.0% | 0.527 [0.479, 0.595] | 37.2% | 0.555 [0.405, 0.715] | 25.6% |
| library notes: Salamander, all layers | 668/6 vs 480/1 | 0.583 [0.549, 0.641] | 49.2% | 0.610 [0.570, 0.664] | 52.7% | 0.601 [0.465, 0.747] | 34.2% |
| library notes: Iowa, all layers | 668/6 vs 261/1 | 0.314 [0.275, 0.380] | 15.7% | 0.337 [0.296, 0.404] | 16.9% | 0.306 [0.202, 0.428] | 3.4% |
| library notes: K18, all layers | 668/6 vs 180/1 | 0.663 [0.633, 0.709] | 60.6% | 0.682 [0.648, 0.727] | 58.9% | 0.693 [0.547, 0.824] | 48.3% |
| library notes: FreePats, all layers | 668/6 vs 78/1 | 0.521 [0.482, 0.586] | 41.0% | 0.527 [0.479, 0.595] | 37.2% | 0.555 [0.405, 0.715] | 25.6% |

**Does the verdict depend on segmentation?** Share of files with P(synthetic) > 0.5 among files in which the segmenter found 1 segment vs >= 2 segments (one segment => all 16 IQR features are 0):

| source | files: 1 seg / >=2 seg | XGB all47 T1: 1 seg / >=2 | XGB all47 T2: 1 seg / >=2 | XGB med16 T2: 1 seg / >=2 |
|---|---|---|---|---|
| CCMusic notes (real; share = false alarms) | 183 / 485 | 36.1% / 13.8% | 33.9% / 9.3% | 23.5% / 22.3% |
| SoundFont renders | 572 / 190 | 95.6% / 66.8% | 95.5% / 66.8% | 75.7% / 74.2% |
| Otey notes | 86 / 34 | 61.6% / 61.8% | 51.2% / 47.1% | 40.7% / 14.7% |
| library notes, mf layer (zero-shot / held-out) | 88 / 136 | 19.3% / 8.8% | 21.6% / 8.1% | 10.2% / 14.7% |
| library notes, all layers | 288 / 711 | 25.0% / 15.0% | 27.1% / 8.7% | 18.4% / 22.4% |

Segment-stratified AUC (CCMusic notes vs the source, both restricted to the same segment stratum; CCMusic: 183 single-segment, 485 multi-segment files):

| CCMusic vs | XGB all47 T2, 1 seg | XGB all47 T2, >=2 seg | XGB med16 T2, 1 seg | XGB med16 T2, >=2 seg |
|---|---|---|---|---|
| SoundFont renders | 0.911 [0.805, 0.981] | 0.906 [0.829, 0.962] | 0.833 [0.699, 0.922] | 0.825 [0.727, 0.915] |
| Otey notes | 0.664 [0.507, 0.812] | 0.853 [0.747, 0.932] | 0.662 [0.529, 0.759] | 0.682 [0.574, 0.801] |
| library notes, mf layer (zero-shot / held-out) | 0.385 [0.204, 0.672] | 0.583 [0.413, 0.735] | 0.371 [0.191, 0.602] | 0.476 [0.332, 0.628] |
| library notes, all layers | 0.428 [0.235, 0.672] | 0.640 [0.424, 0.771] | 0.414 [0.232, 0.629] | 0.557 [0.364, 0.691] |

I-only detector, CCMusic false alarms (P > 0.5): xgb_all47_Ionly 15.4%; xgb_med16_Ionly 22.6%; lr_all47_Ionly 19.6%; lr_med16_Ionly 28.6%.
CCMusic groups per fold: {0: 1, 1: 1, 3: 2, 4: 2} (folds without CCMusic contribute only synthetic OOF scores).

### 3.6 f0 audit of the 47-feature pipeline on isolated notes, and register / f0 sensitivity

The figure analysis found that the phaseogram pipeline's f0 is off by >= 1 semitone for many isolated CCMusic notes outside MIDI 48-76. The 47-feature pipeline uses the same segmenter (`detune/segment.build_segments`): its per-segment f0 sets the partial grid for every per-note descriptor (partial energies, B, centroid/f0). Nominal pitch: CCMusic Helmholtz names / SoundFont, Otey and library note names (same parser as `figures/phaseo_lib.py`). "Primary" = the most energetic segment (the figure analysis' criterion); "segments" = all per-note descriptor sets that enter a file's median/IQR (for a single strike the segmenter can return up to 3 pitch candidates of the same window). Failure = |estimated - nominal| >= 1 semitone.

| source | nominal MIDI < 48 | MIDI 48-76 | MIDI > 76 | all |
|---|---|---|---|---|
| CCMusic (real) | 117/205 files; 465/604 seg. | 9/220 files; 198/438 seg. | 203/243 files; 682/743 seg. | 329/668 files; 1345/1785 seg. |
| SoundFont renders | 69/252 files; 111/328 seg. | 2/300 files; 84/388 seg. | 64/210 files; 326/510 seg. | 135/762 files; 521/1226 seg. |
| Otey | 0/36 files; 0/55 seg. | 0/40 files; 0/40 seg. | 18/44 files; 45/71 seg. | 18/120 files; 45/166 seg. |
| library Salamander (all layers) | 83/144 files; 1985/3156 seg. | 1/160 files; 921/1324 seg. | 144/176 files; 945/1007 seg. | 228/480 files; 3851/5487 seg. |
| library Iowa (all layers) | 32/78 files; 109/167 seg. | 12/87 files; 26/108 seg. | 92/96 files; 264/275 seg. | 136/261 files; 399/550 seg. |
| library K18 (all layers) | 33/54 files; 338/396 seg. | 1/60 files; 4/71 seg. | 54/66 files; 152/171 seg. | 88/180 files; 494/638 seg. |
| library FreePats (all layers) | 13/22 files; 118/204 seg. | 0/26 files; 2/33 seg. | 22/30 files; 62/76 seg. | 35/78 files; 182/313 seg. |
| listening clips | - | 0/30 files; 15/55 seg. | - | 0/30 files; 15/55 seg. |

CCMusic primary-f0 failures by size (semitones, rounded): +12: 73, -12: 26, -31: 22, -19: 15, -67: 8, +24: 8. CCMusic files with no f0-valid segment at all: 291 of 668. Per-file details: `out/f0_audit.csv`.

Sensitivity of the isolated-regime comparisons (CCMusic notes vs ...). "f0-valid files" / "MIDI 48-76" restrict BOTH classes at evaluation (same out-of-fold / zero-shot scores as sections 2-3). "f0-clean features" re-aggregates every isolated-note file (CCMusic, SoundFont, Otey, library notes) from its f0-valid segments only and retrains the CV with the same folds (files left without any f0-valid segment are dropped: CCMusic 291, SoundFont 82, Otey 18; library notes 373). Library 14-note takes cannot be restricted this way and are not shown. Cells: AUC [95% CI] (CCMusic files vs synthetic files).

**XGBoost all 47, T2**

| restriction | SoundFont renders | Otey notes | algorithmic (both) | library notes, mf layer | library notes, all layers |
|---|---|---|---|---|---|
| baseline | 0.934 [0.870, 0.976] (668/762) | 0.807 [0.696, 0.900] (668/120) | 0.917 [0.842, 0.971] (668/882) | 0.530 [0.370, 0.708] (668/224) | 0.570 [0.389, 0.720] (668/999) |
| f0-valid files | 0.929 [0.844, 0.981] (339/627) | 0.768 [0.623, 0.891] (339/102) | 0.907 [0.808, 0.976] (339/729) | 0.492 [0.327, 0.676] (339/115) | 0.510 [0.342, 0.674] (339/512) |
| MIDI 48-76 | 0.946 [0.858, 0.991] (220/300) | 0.746 [0.594, 0.860] (220/40) | 0.923 [0.821, 0.987] (220/340) | 0.504 [0.337, 0.671] (220/75) | 0.495 [0.335, 0.661] (220/333) |
| f0-valid & 48-76 | 0.944 [0.855, 0.991] (211/298) | 0.736 [0.582, 0.855] (211/40) | 0.919 [0.816, 0.986] (211/338) | 0.500 [0.334, 0.665] (211/72) | 0.488 [0.333, 0.651] (211/319) |
| f0-clean features (retrained) | 0.932 [0.869, 0.983] (377/680) | 0.798 [0.693, 0.906] (377/102) | 0.914 [0.838, 0.978] (377/782) | 0.549 [0.395, 0.729] (377/135) | 0.585 [0.415, 0.742] (377/626) |
| f0-clean features & 48-76 | 0.944 [0.872, 0.993] (212/300) | 0.803 [0.670, 0.932] (212/40) | 0.927 [0.839, 0.990] (212/340) | 0.585 [0.413, 0.766] (212/74) | 0.573 [0.398, 0.745] (212/326) |

**XGBoost all 47, T1**

| restriction | SoundFont renders | Otey notes | algorithmic (both) | library notes, mf layer | library notes, all layers |
|---|---|---|---|---|---|
| baseline | 0.924 [0.856, 0.970] (668/762) | 0.809 [0.697, 0.898] (668/120) | 0.908 [0.833, 0.966] (668/882) | 0.440 [0.282, 0.627] (668/224) | 0.506 [0.307, 0.656] (668/999) |
| f0-valid files | 0.918 [0.832, 0.974] (339/627) | 0.775 [0.644, 0.884] (339/102) | 0.898 [0.804, 0.967] (339/729) | 0.406 [0.252, 0.583] (339/115) | 0.446 [0.272, 0.592] (339/512) |
| MIDI 48-76 | 0.937 [0.846, 0.985] (220/300) | 0.759 [0.620, 0.861] (220/40) | 0.916 [0.818, 0.979] (220/340) | 0.400 [0.233, 0.582] (220/75) | 0.423 [0.259, 0.583] (220/333) |
| f0-valid & 48-76 | 0.935 [0.844, 0.984] (211/298) | 0.750 [0.608, 0.856] (211/40) | 0.913 [0.811, 0.979] (211/338) | 0.394 [0.222, 0.573] (211/72) | 0.416 [0.255, 0.570] (211/319) |
| f0-clean features (retrained) | 0.928 [0.861, 0.982] (377/680) | 0.822 [0.722, 0.923] (377/102) | 0.914 [0.839, 0.977] (377/782) | 0.462 [0.285, 0.669] (377/135) | 0.510 [0.324, 0.673] (377/626) |
| f0-clean features & 48-76 | 0.945 [0.872, 0.994] (212/300) | 0.817 [0.686, 0.940] (212/40) | 0.930 [0.846, 0.991] (212/340) | 0.460 [0.274, 0.672] (212/74) | 0.468 [0.291, 0.645] (212/326) |

**XGBoost 16 Group-A medians, T2**

| restriction | SoundFont renders | Otey notes | algorithmic (both) | library notes, mf layer | library notes, all layers |
|---|---|---|---|---|---|
| baseline | 0.834 [0.748, 0.907] (668/762) | 0.684 [0.598, 0.774] (668/120) | 0.813 [0.723, 0.898] (668/882) | 0.445 [0.299, 0.613] (668/224) | 0.518 [0.333, 0.654] (668/999) |
| f0-valid files | 0.839 [0.720, 0.923] (339/627) | 0.660 [0.532, 0.766] (339/102) | 0.814 [0.691, 0.912] (339/729) | 0.413 [0.249, 0.585] (339/115) | 0.449 [0.273, 0.600] (339/512) |
| MIDI 48-76 | 0.874 [0.757, 0.947] (220/300) | 0.684 [0.562, 0.775] (220/40) | 0.852 [0.736, 0.938] (220/340) | 0.411 [0.241, 0.599] (220/75) | 0.427 [0.267, 0.592] (220/333) |
| f0-valid & 48-76 | 0.870 [0.754, 0.945] (211/298) | 0.678 [0.556, 0.776] (211/40) | 0.847 [0.730, 0.937] (211/338) | 0.410 [0.235, 0.592] (211/72) | 0.425 [0.262, 0.582] (211/319) |
| f0-clean features (retrained) | 0.862 [0.772, 0.938] (377/680) | 0.734 [0.645, 0.823] (377/102) | 0.845 [0.751, 0.927] (377/782) | 0.460 [0.308, 0.625] (377/135) | 0.510 [0.338, 0.644] (377/626) |
| f0-clean features & 48-76 | 0.915 [0.828, 0.974] (212/300) | 0.816 [0.718, 0.893] (212/40) | 0.904 [0.816, 0.968] (212/340) | 0.451 [0.301, 0.628] (212/74) | 0.470 [0.319, 0.617] (212/326) |


## 4. Listening-test stimuli scored OUT OF SAMPLE

Features were extracted from each of the 30 **clips the listeners heard** (`listening_kit/web/clip_XX.wav`, 3 s, onset-trimmed, RMS-normalised) with the single-file pipeline used for isolated-note files. Each clip is scored by an XGBoost (or LR) model fitted on the T2 training population (real + algorithmic; sample-based kinds absent) with the clip's own group removed: Otey clips by a model without the Otey engine, Casio CA-110 clips by a model without that SoundFont, library clips (Iowa, K18, Salamander) by the full T2 model (no library in training). "LOGO" = this leave-own-group-out scheme; "noOteyCasio" = one model without both groups for all 30 clips. Detection = P(synthetic) > 0.5.

| scoring | Iowa det. | K18 det. | Salamander det. | SoundFont det. | Otey det. | AUC recorded(20) vs algorithmic(10) [95% CI, clips resampled within class] | AUC Iowa+K18(14) vs algorithmic(10) | AUC Salamander(6) vs Iowa+K18(14) |
|---|---|---|---|---|---|---|---|---|
| XGB all47, clip, LOGO (primary) | 2/7 | 4/7 | 3/6 | 3/5 | 3/5 | 0.665 [0.470, 0.845] | 0.679 | 0.512 |
| XGB all47, clip, one model w/o Otey+CA-110 | 1/7 | 3/7 | 2/6 | 2/5 | 3/5 | 0.695 [0.515, 0.865] | 0.700 | 0.476 |
| XGB med16, clip, LOGO | 1/7 | 3/7 | 2/6 | 2/5 | 2/5 | 0.670 [0.470, 0.845] | 0.671 | 0.512 |
| LR all47, clip, LOGO | 4/7 | 6/7 | 3/6 | 1/5 | 5/5 | 0.410 [0.210, 0.625] | 0.379 | 0.381 |
| LR med16, clip, LOGO | 0/7 | 5/7 | 3/6 | 4/5 | 3/5 | 0.780 [0.595, 0.925] | 0.814 | 0.643 |
| XGB all47, unprocessed source file, LOGO | 1/7 | 2/7 | 1/6 | 4/5 | 3/5 | 0.850 [0.715, 0.955] | 0.821 | 0.619 |
| OLD in-sample model (realness_xgb_pre_hard), source file, as in the paper | 1/7 | 3/7 | 2/6 | 5/5 | 5/5 | 0.965 [0.890, 1.000] | 0.950 | 0.643 |
| OLD in-sample model, clip | 3/7 | 6/7 | 3/6 | 5/5 | 5/5 | 0.910 [0.800, 0.990] | 0.929 | 0.548 |

The stored paper values (`critic_on_stimuli.json`) are reproduced by `realness_xgb_pre_hard.json` on the source files to max |dP| = 0.0000. That model was fitted on all 4227 corpus rows: the 10 algorithmic source files are training rows, and Iowa/K18/Salamander were in training as synthetic 14-note takes.

Per-clip scores (P(synthetic)):

| clip | class | note | XGB all47 LOGO | XGB med16 LOGO | LR all47 LOGO | XGB all47 src LOGO | old in-sample (paper) | segments found in clip |
|---|---|---|---|---|---|---|---|---|
| clip_01 | Salamander | D#3 | 0.095 | 0.266 | 0.475 | 0.001 | 0.002 | 2 |
| clip_02 | Iowa | D3 | 0.250 | 0.016 | 0.873 | 0.001 | 0.003 | 2 |
| clip_03 | K18 | D#3 | 0.300 | 0.220 | 0.315 | 0.001 | 0.001 | 1 |
| clip_04 | Salamander | C4 | 0.861 | 0.029 | 0.800 | 0.003 | 0.037 | 2 |
| clip_05 | Salamander | D#4 | 0.684 | 0.587 | 0.523 | 0.026 | 0.590 | 2 |
| clip_06 | Iowa | B4 | 0.889 | 0.190 | 0.569 | 0.532 | 0.802 | 2 |
| clip_07 | Otey | B4 | 0.981 | 0.207 | 0.757 | 0.648 | 0.992 | 1 |
| clip_08 | Salamander | A3 | 0.008 | 0.015 | 0.134 | 0.010 | 0.078 | 2 |
| clip_09 | Iowa | F4 | 0.555 | 0.626 | 0.086 | 0.000 | 0.000 | 1 |
| clip_10 | SoundFont (Casio CA-110) | C4 | 0.998 | 0.973 | 0.373 | 0.998 | 0.998 | 1 |
| clip_11 | Salamander | F#4 | 0.010 | 0.088 | 0.450 | 0.001 | 0.039 | 2 |
| clip_12 | Iowa | B3 | 0.279 | 0.158 | 0.905 | 0.000 | 0.002 | 3 |
| clip_13 | Otey | C5 | 0.800 | 0.194 | 0.680 | 0.970 | 0.998 | 1 |
| clip_14 | Iowa | G3 | 0.008 | 0.013 | 0.777 | 0.000 | 0.004 | 1 |
| clip_15 | K18 | C5 | 0.998 | 0.976 | 0.920 | 0.997 | 0.996 | 1 |
| clip_16 | K18 | C3 | 0.000 | 0.002 | 0.725 | 0.000 | 0.002 | 1 |
| clip_17 | K18 | F#3 | 0.714 | 0.935 | 0.867 | 0.474 | 0.509 | 1 |
| clip_18 | K18 | D#4 | 0.306 | 0.105 | 0.857 | 0.002 | 0.016 | 1 |
| clip_19 | Otey | E4 | 0.936 | 0.906 | 0.718 | 0.936 | 0.997 | 1 |
| clip_20 | Salamander | D#5 | 1.000 | 0.957 | 0.962 | 0.874 | 0.902 | 2 |
| clip_21 | SoundFont (Casio CA-110) | C3 | 0.999 | 0.988 | 0.998 | 0.999 | 1.000 | 1 |
| clip_22 | K18 | A3 | 0.995 | 0.919 | 0.966 | 0.989 | 0.983 | 1 |
| clip_23 | K18 | D#5 | 0.640 | 0.162 | 0.969 | 0.418 | 0.483 | 1 |
| clip_24 | SoundFont (Casio CA-110) | A4 | 0.132 | 0.339 | 0.000 | 0.546 | 0.994 | 4 |
| clip_25 | Otey | G4 | 0.320 | 0.023 | 0.948 | 0.241 | 0.977 | 1 |
| clip_26 | SoundFont (Casio CA-110) | A3 | 0.083 | 0.126 | 0.000 | 0.253 | 0.991 | 3 |
| clip_27 | Iowa | D5 | 0.010 | 0.003 | 0.226 | 0.001 | 0.000 | 3 |
| clip_28 | Otey | E3 | 0.378 | 0.690 | 0.835 | 0.240 | 0.986 | 1 |
| clip_29 | Iowa | F3 | 0.017 | 0.278 | 0.217 | 0.001 | 0.001 | 3 |
| clip_30 | SoundFont (Casio CA-110) | C5 | 0.566 | 0.407 | 0.000 | 0.895 | 0.993 | 7 |

## 4b. Effect of adding data_v2 (XGBoost; same folds)

Sections 2-4 train on the base corpus PLUS the data_v2 rows present (digital_v2_features.csv: 138 rows, sf2_perf_features.csv: 100 rows, otey_perf_features.csv: 20 rows). Below, the same protocol on the base corpus only (data_v2 rows absent from training and evaluation).

| id | all47 T1 (base -> +v2) | all47 T2 | med16 T1 | med16 T2 |
|---|---|---|---|---|
| I1a | 0.921 -> 0.924 | 0.922 -> 0.934 | 0.799 -> 0.817 | 0.820 -> 0.834 |
| I1b | 0.786 -> 0.809 | 0.804 -> 0.807 | 0.600 -> 0.638 | 0.678 -> 0.684 |
| I2 | 0.610 -> 0.607 | 0.282 -> 0.410 | 0.337 -> 0.412 | 0.360 -> 0.379 |
| P2a | 0.816 -> 0.915 | 0.699 -> 0.918 | 0.693 -> 0.786 | 0.791 -> 0.800 |
| P4 | 0.852 -> 0.833 | 0.855 -> 0.891 | 0.816 -> 0.803 | 0.929 -> 0.889 |
| X1 | 0.977 -> 0.961 | 0.980 -> 0.970 | 0.934 -> 0.876 | 0.949 -> 0.903 |
| X2 | 0.806 -> 0.845 | 0.730 -> 0.796 | 0.682 -> 0.660 | 0.752 -> 0.642 |
| C1 | 0.997 -> 0.982 | 0.999 -> 0.994 | 0.976 -> 0.927 | 0.987 -> 0.960 |
| libnotes_mf | 0.445 -> 0.440 | 0.564 -> 0.530 | 0.340 -> 0.345 | 0.461 -> 0.445 |

## 5. Continuity with the submitted paper (XGBoost, all 47, T1)

"Uncorrected" = the paper's 4227-file corpus and groups (25 non-acoustic files kept as real, demos grouped by case-sensitive filename token), run with this protocol's fixed folds and n_jobs=1. The paper reported 0.964 (all), 0.978 [0.957, 0.995] (algorithmic), 0.806 [0.727, 0.911] (sample-based), with unstored, platform-dependent folds; facts_A §4.3 shows 0.960-0.965 / 0.975-0.978 / 0.783-0.827 across tie orders.

| id | comparison | files (uncorrected) | AUC uncorrected | files (corrected, no data_v2) | AUC corrected, no data_v2 |
|---|---|---|---|---|---|
| X3 | all real vs all synthetic | 3269 vs 958 | 0.962 [0.930, 0.985] | 3244 vs 958 | 0.964 [0.932, 0.988] |
| X1 | all real vs all algorithmic | 3269 vs 882 | 0.977 [0.957, 0.994] | 3244 vs 882 | 0.977 [0.957, 0.994] |
| X2 | all real vs sample-based (libraries + sampled digital + M1) | 3269 vs 76 | 0.789 [0.706, 0.893] | 3244 vs 71 | 0.806 [0.716, 0.918] |
| I1 | real_isolated (CCMusic) vs algo_single (SoundFont renders + Otey) | 668 vs 882 | 0.906 [0.825, 0.962] | 668 vs 882 | 0.903 [0.823, 0.962] |
| I2 | real_isolated (CCMusic) vs sampled_library (14-note takes; unit mismatch, see per-note table) | 668 vs 32 | 0.545 [0.352, 0.737] | 668 vs 32 | 0.610 [0.402, 0.799] |
| P2 | real_performance vs sampled digital pianos (old demos + digital_v2) | 2601 vs 14 | 0.847 [0.757, 0.918] | 2576 vs 9 | 0.816 [0.759, 0.903] |
| P4 | real_performance vs Korg M1 | 2601 vs 30 | 0.839 [0.831, 0.848] | 2576 vs 30 | 0.852 [0.842, 0.862] |
| C1 | real_performance vs algo_single (format-crossed) | 2601 vs 882 | 0.996 [0.993, 0.998] | 2576 vs 882 | 0.997 [0.995, 0.999] |

### 5.1 Sensitivity to the fold assignment (XGBoost, all 47)

The fold map above uses seed 0. Point AUCs (no bootstrap) with the same rule and seeds 0-4; per-note library rows as in section 3 (mf layer).

| id | T1: seed 0 (range seeds 0-4) | T2: seed 0 (range seeds 0-4) |
|---|---|---|
| I1a | 0.924 (0.911-0.937) | 0.934 (0.925-0.946) |
| I1b | 0.809 (0.779-0.843) | 0.807 (0.789-0.850) |
| I2 | 0.607 (0.523-0.612) | 0.410 (0.410-0.459) |
| P2a | 0.915 (0.915-0.967) | 0.918 (0.912-0.931) |
| P4 | 0.833 (0.784-0.841) | 0.891 (0.889-0.893) |
| X1 | 0.961 (0.954-0.965) | 0.970 (0.965-0.974) |
| X2 | 0.845 (0.845-0.863) | 0.796 (0.796-0.807) |
| C1 | 0.982 (0.977-0.983) | 0.994 (0.992-0.994) |
| libnotes_mf | 0.440 (0.417-0.458) | 0.530 (0.519-0.564) |

(protocol_v2 runtime 257 s; xgboost 3.2.0)

## 6. Listening test re-analysed (classes as they really are)

32 listeners x 30 clips = 960 judgments (`listening_kit/results_anon.csv`; self-reported experience: {'lay': 17, 'amateur': 14, 'musician': 1}). The paper's "real" class (14 clips) is Iowa (7) + K18 (7) sample-library notes; its "sample library" class is Salamander (6). All three are recorded acoustic notes from sample libraries. "Judged synthetic" = answer 3 or 4.

### 6.1 Judged-synthetic rate per class

| class | clips | judged synth. | rate | 95% CI (two-way boot.) | binom. p (>0.5) | binom. p (two-sided) | listeners >/=/< 0.5 | Wilcoxon p (>0.5) | Wilcoxon p (two-sided) | mean rating (1-4) |
|---|---|---|---|---|---|---|---|---|---|---|
| Iowa | 7 | 96/224 | 42.9% | [27.2, 58.0] | 0.99 | 0.038 | 15/0/17 | 0.97 | 0.062 | 2.27 |
| K18 | 7 | 78/224 | 34.8% | [24.1, 44.6] | 1 | 6.5e-06 | 7/0/25 | 1 | 9.1e-05 | 2.18 |
| Salamander | 6 | 64/192 | 33.3% | [21.9, 45.3] | 1 | 4.5e-06 | 2/10/20 | 1 | 0.00023 | 2.18 |
| SoundFont (Casio CA-110) | 5 | 107/160 | 66.9% | [51.2, 81.2] | 1.2e-05 | 2.4e-05 | 21/0/11 | 0.0059 | 0.012 | 2.86 |
| Otey | 5 | 101/160 | 63.1% | [48.1, 77.5] | 0.00056 | 0.0011 | 23/0/9 | 0.0037 | 0.0073 | 2.75 |
| Iowa+K18 (paper's "real") | 14 | 174/448 | 38.8% | [29.2, 48.4] | 1 | 2.7e-06 | 8/3/21 | 1 | 0.0019 | 2.23 |
| RECORDED (all 3 libraries) | 20 | 238/640 | 37.2% | [30.0, 44.4] | 1 | 9.3e-11 | 3/3/26 | 1 | 4.4e-05 | 2.21 |
| ALGORITHMIC (SoundFont + Otey) | 10 | 208/320 | 65.0% | [54.7, 74.7] | 4.4e-08 | 8.8e-08 | 24/1/7 | 0.00048 | 0.00095 | 2.81 |

The pooled binomial treats the judgments as independent; the listener-level Wilcoxon accounts for clustering within listeners but not for the small number of clips per class; the two-way bootstrap CI accounts for both.

### 6.2 Salamander vs Iowa / K18, paired across listeners

| difference | metric | mean over listeners | 95% CI (two-way boot.) | listeners +/0/- | Wilcoxon p (two-sided) | sign test p |
|---|---|---|---|---|---|---|
| Salamander - Iowa+K18 | judged-synth rate | -0.055 | [-0.210, +0.100] | 12/2/18 | 0.207 | 0.362 |
| Salamander - Iowa+K18 | mean rating | -0.048 | [-0.339, +0.249] | 14/0/18 | 0.612 | 0.597 |
| Salamander - Iowa | judged-synth rate | -0.095 | [-0.292, +0.103] | 13/1/18 | 0.139 | 0.473 |
| Salamander - Iowa | mean rating | -0.095 | [-0.472, +0.295] | 13/1/18 | 0.427 | 0.473 |
| Salamander - K18 | judged-synth rate | -0.015 | [-0.182, +0.154] | 16/1/15 | 0.62 | 1 |
| Salamander - K18 | mean rating | -0.001 | [-0.294, +0.286] | 17/0/15 | 0.978 | 0.86 |
| Iowa - K18 | judged-synth rate | +0.080 | [-0.098, +0.250] | 15/10/7 | 0.0998 | 0.134 |
| Iowa - K18 | mean rating | +0.094 | [-0.241, +0.415] | 17/4/11 | 0.163 | 0.345 |

### 6.3 Listener-level discrimination: recorded vs algorithmic

| comparison (positive = second class) | mean listener AUC [95% CI] | median | listeners >/=/< 0.5 | Wilcoxon p (AUC>0.5) | mean d' | pooled AUC (all judgments) | panel AUC (per-clip mean rating) [95% CI] |
|---|---|---|---|---|---|---|---|
| recorded (20) vs algorithmic (10) | 0.665 [0.597, 0.734] | 0.690 | 27/0/5 | 7.5e-07 | 0.75 (median 0.79) | 0.665 | 0.983 [0.800, 1.000] |
| Iowa+K18 (14) vs algorithmic (10)  [paper's real vs synthetic-generated] | 0.662 [0.589, 0.742] | 0.689 | 26/0/6 | 2e-05 | 0.72 (median 0.78) | 0.660 | 0.975 [0.764, 1.000] |
| Salamander (6) vs algorithmic (10) | 0.672 [0.595, 0.740] | 0.654 | 30/0/2 | 4.8e-08 | 0.82 (median 0.75) | 0.677 | 1.000 [0.817, 1.000] |
| Iowa+K18 (14) vs Salamander (6)  [positive = Salamander] | 0.484 [0.403, 0.564] | 0.461 | 14/0/18 | 0.78 | -0.10 (median -0.20) | 0.486 | 0.417 [0.119, 0.780] |

Listener AUC uses the 1-4 rating as the score (ties = 1/2); d' uses the binary judgment with the log-linear correction (+0.5 / +1). CIs: two-way bootstrap over listeners and clips (B = 2000; 500 for the mean listener AUC).

The single self-reported musician (L01): judged synthetic Iowa 2/7, K18 1/7, Salamander 6/6, SoundFont (Casio CA-110) 4/5, Otey 4/5.

### 6.4 Effective sounding duration of the clips (duration cue)

Frame RMS (10 ms window, 5 ms hop) of each 3.00-s clip; t40 / t60 = end of the last frame within 40 / 60 dB of the clip's peak frame, measured from the clip start (the onset is at about 10 ms). Values >= 2.97 s are censored (the note is still above the threshold when the 10-ms fade-out ends the clip).

| class | clips | t40 | t60 | last non-zero sample | judged synth. |
|---|---|---|---|---|---|
| Iowa | 7 | 7/7 censored | 7/7 censored | 3.00-3.00 s | 43% |
| K18 | 7 | 5/7 censored; others 2.38-2.65 s | 7/7 censored | 3.00-3.00 s | 35% |
| Salamander | 6 | 6/6 censored | 6/6 censored | 3.00-3.00 s | 33% |
| SoundFont (Casio CA-110) | 5 | 0/5 censored; others 0.95-1.59 s | 0/5 censored; others 1.38-1.75 s | 1.80-1.80 s | 67% |
| Otey | 5 | 1/5 censored; others 2.04-2.87 s | 5/5 censored | 3.00-3.00 s | 63% |

Spearman between t40 and the per-clip judged-synthetic rate: all 30 clips rho = -0.57 (p = 0.001); recorded clips only rho = 0.23 (p = 0.329). Within the design, duration is confounded with class (only the SoundFont clips stop early), so these correlations cannot separate a duration cue from a timbre cue.

Per-clip listener data (`out/listening_per_clip.csv`):

| clip | class | note | t40 (s) | t60 (s) | judged synth. (of 32) | mean rating | detector P, XGB all47 LOGO |
|---|---|---|---|---|---|---|---|
| clip_01 | Salamander | D#3 | 3.00+ | 3.00+ | 9 | 2.22 | 0.095 |
| clip_02 | Iowa | D3 | 3.00+ | 3.00+ | 20 | 2.66 | 0.250 |
| clip_03 | K18 | D#3 | 3.00+ | 3.00+ | 14 | 2.38 | 0.300 |
| clip_04 | Salamander | C4 | 3.00+ | 3.00+ | 16 | 2.34 | 0.861 |
| clip_05 | Salamander | D#4 | 3.00+ | 3.00+ | 8 | 1.97 | 0.684 |
| clip_06 | Iowa | B4 | 2.99+ | 3.00+ | 3 | 1.59 | 0.889 |
| clip_07 | Otey | B4 | 2.18 | 3.00+ | 27 | 3.12 | 0.981 |
| clip_08 | Salamander | A3 | 3.00+ | 3.00+ | 12 | 2.38 | 0.008 |
| clip_09 | Iowa | F4 | 3.00+ | 3.00+ | 16 | 2.41 | 0.555 |
| clip_10 | SoundFont (Casio CA-110) | C4 | 1.59 | 1.74 | 22 | 3.03 | 0.998 |
| clip_11 | Salamander | F#4 | 3.00+ | 3.00+ | 9 | 2.25 | 0.010 |
| clip_12 | Iowa | B3 | 3.00+ | 3.00+ | 14 | 2.44 | 0.279 |
| clip_13 | Otey | C5 | 2.04 | 3.00+ | 18 | 2.75 | 0.800 |
| clip_14 | Iowa | G3 | 3.00+ | 3.00+ | 15 | 2.38 | 0.008 |
| clip_15 | K18 | C5 | 2.99+ | 3.00+ | 12 | 2.12 | 0.998 |
| clip_16 | K18 | C3 | 3.00+ | 3.00+ | 11 | 2.22 | 0.000 |
| clip_17 | K18 | F#3 | 2.65 | 3.00+ | 12 | 2.19 | 0.714 |
| clip_18 | K18 | D#4 | 3.00+ | 3.00+ | 12 | 2.22 | 0.306 |
| clip_19 | Otey | E4 | 2.58 | 3.00+ | 22 | 2.75 | 0.936 |
| clip_20 | Salamander | D#5 | 3.00+ | 3.00+ | 10 | 1.91 | 1.000 |
| clip_21 | SoundFont (Casio CA-110) | C3 | 1.59 | 1.75 | 25 | 3.09 | 0.999 |
| clip_22 | K18 | A3 | 2.38 | 3.00+ | 10 | 2.09 | 0.995 |
| clip_23 | K18 | D#5 | 3.00+ | 3.00+ | 7 | 2.03 | 0.640 |
| clip_24 | SoundFont (Casio CA-110) | A4 | 1.13 | 1.51 | 22 | 2.91 | 0.132 |
| clip_25 | Otey | G4 | 2.87 | 3.00+ | 19 | 2.69 | 0.320 |
| clip_26 | SoundFont (Casio CA-110) | A3 | 1.14 | 1.49 | 17 | 2.56 | 0.083 |
| clip_27 | Iowa | D5 | 3.00+ | 3.00+ | 10 | 1.94 | 0.010 |
| clip_28 | Otey | E3 | 3.00+ | 3.00+ | 15 | 2.44 | 0.378 |
| clip_29 | Iowa | F3 | 3.00+ | 3.00+ | 18 | 2.50 | 0.017 |
| clip_30 | SoundFont (Casio CA-110) | C5 | 0.95 | 1.38 | 21 | 2.72 | 0.566 |
("+" = censored at the clip end.)

### 6.5 Listeners vs the out-of-sample detector, same 30 clips

| class | clips | listeners: judged synth. | clips judged synth. by >50% of listeners | detector XGB all47 (out of sample) | detector XGB med16 (out of sample) | OLD in-sample detector (paper) |
|---|---|---|---|---|---|---|
| Iowa | 7 | 43% | 2/7 | 2/7 | 1/7 | 1/7 |
| K18 | 7 | 35% | 0/7 | 4/7 | 3/7 | 3/7 |
| Salamander | 6 | 33% | 0/6 | 3/6 | 2/6 | 2/6 |
| SoundFont (Casio CA-110) | 5 | 67% | 5/5 | 3/5 | 2/5 | 5/5 |
| Otey | 5 | 63% | 4/5 | 3/5 | 2/5 | 5/5 |
| AUC recorded vs algorithmic | 20 vs 10 | panel 0.983 | mean listener 0.665 | 0.665 | 0.670 | 0.965 |


## 7. Follow-up: what carries the performance-regime separations (P1, P2)?

### 7.1 Feature-group ablation (XGBoost, same fixed folds)

Groups: A medians = per-note descriptors 1-16 (median over the file's notes); A IQRs = 17-32; B = register descriptors 33-40 (bass / treble / deep-bass, not detrended); C = tuning 41-47 (file level). "decay/sustain medians" = log_t60_med, sustain_db_1s_med, hf_decay_excess_med, bright_decay_med. Comparison ids as in section 2 (P1 real performances vs algorithmic performance renders; P1a SoundFont, P1b Otey; P2 vs sample-based digital pianos (old demos + digital_v2), P2a old demos only, P2d digital_v2 only; P3 modelled digital pianos (Roland); P4 Korg M1; I1a/I1b CCMusic notes vs SoundFont / Otey single notes; libnotes_mf = CCMusic notes vs library notes, mf layer, as section 3). n (real / synthetic files): P1 2576/120, P1a 2576/100, P1b 2576/20, P2 2576/128, P2a 2576/9, P2d 2576/119, P3 2576/21, P4 2576/30, I1a 668/762, I1b 668/120, libnotes_mf 668/224.

**T1: AUC [95% CI]**

| feature set | P1 | P1a | P1b | P2 | P2a | P2d | P3 | P4 | I1a | I1b | libnotes_mf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 47 | 0.951 [0.928, 0.970] | 0.951 [0.924, 0.975] | 0.951 [0.946, 0.956] | 0.915 [0.892, 0.936] | 0.915 [0.876, 0.970] | 0.915 [0.892, 0.937] | 0.865 [0.859, 0.871] | 0.833 [0.824, 0.842] | 0.924 [0.856, 0.970] | 0.809 [0.697, 0.898] | 0.440 [0.282, 0.627] |
| only A medians (16) | 0.744 [0.652, 0.833] | 0.756 [0.648, 0.855] | 0.686 [0.674, 0.698] | 0.707 [0.669, 0.770] | 0.786 [0.672, 0.978] | 0.702 [0.663, 0.763] | 0.812 [0.803, 0.821] | 0.803 [0.796, 0.811] | 0.817 [0.728, 0.898] | 0.638 [0.548, 0.749] | 0.345 [0.204, 0.517] |
| only A IQRs (16) | 0.804 [0.709, 0.881] | 0.801 [0.689, 0.897] | 0.815 [0.806, 0.823] | 0.738 [0.693, 0.805] | 0.726 [0.638, 0.933] | 0.739 [0.697, 0.804] | 0.750 [0.740, 0.759] | 0.642 [0.630, 0.654] | 0.736 [0.624, 0.862] | 0.637 [0.526, 0.763] | 0.453 [0.353, 0.563] |
| only B register (8) | 0.767 [0.680, 0.850] | 0.778 [0.674, 0.867] | 0.714 [0.701, 0.729] | 0.730 [0.692, 0.793] | 0.684 [0.637, 0.837] | 0.734 [0.694, 0.797] | 0.731 [0.721, 0.741] | 0.846 [0.836, 0.856] | 0.772 [0.692, 0.859] | 0.666 [0.589, 0.752] | 0.616 [0.538, 0.695] |
| only C tuning (7) | 0.751 [0.661, 0.821] | 0.744 [0.635, 0.831] | 0.787 [0.778, 0.796] | 0.729 [0.701, 0.777] | 0.627 [0.448, 0.741] | 0.737 [0.704, 0.795] | 0.700 [0.691, 0.709] | 0.523 [0.512, 0.534] | 0.751 [0.672, 0.838] | 0.672 [0.598, 0.758] | 0.568 [0.508, 0.636] |
| all minus A medians (31) | 0.932 [0.899, 0.962] | 0.933 [0.891, 0.969] | 0.931 [0.925, 0.937] | 0.907 [0.888, 0.925] | 0.898 [0.874, 0.948] | 0.908 [0.888, 0.927] | 0.834 [0.828, 0.841] | 0.796 [0.787, 0.805] | 0.804 [0.732, 0.880] | 0.733 [0.650, 0.825] | 0.565 [0.478, 0.660] |
| all minus A IQRs (31) | 0.938 [0.914, 0.962] | 0.939 [0.911, 0.967] | 0.930 [0.924, 0.937] | 0.890 [0.864, 0.916] | 0.826 [0.770, 0.925] | 0.895 [0.866, 0.921] | 0.868 [0.863, 0.873] | 0.854 [0.847, 0.863] | 0.923 [0.855, 0.969] | 0.821 [0.719, 0.907] | 0.457 [0.295, 0.643] |
| all minus B (39) | 0.937 [0.890, 0.973] | 0.934 [0.879, 0.979] | 0.953 [0.949, 0.958] | 0.906 [0.879, 0.931] | 0.942 [0.915, 0.976] | 0.903 [0.877, 0.929] | 0.845 [0.838, 0.851] | 0.769 [0.760, 0.779] | 0.911 [0.831, 0.963] | 0.765 [0.644, 0.862] | 0.398 [0.233, 0.584] |
| all minus C (40) | 0.882 [0.828, 0.929] | 0.877 [0.817, 0.934] | 0.909 [0.903, 0.916] | 0.867 [0.837, 0.908] | 0.834 [0.734, 0.954] | 0.870 [0.841, 0.908] | 0.851 [0.844, 0.858] | 0.794 [0.783, 0.805] | 0.917 [0.847, 0.966] | 0.788 [0.674, 0.887] | 0.384 [0.229, 0.571] |
| only 4 decay/sustain medians | 0.663 [0.581, 0.756] | 0.670 [0.574, 0.775] | 0.626 [0.615, 0.639] | 0.655 [0.622, 0.708] | 0.634 [0.575, 0.797] | 0.657 [0.620, 0.712] | 0.699 [0.687, 0.710] | 0.800 [0.793, 0.807] | 0.740 [0.632, 0.830] | 0.600 [0.510, 0.684] | 0.290 [0.163, 0.433] |
| 12 other A medians | 0.636 [0.491, 0.772] | 0.664 [0.502, 0.810] | 0.498 [0.487, 0.510] | 0.478 [0.387, 0.630] | 0.562 [0.423, 0.798] | 0.472 [0.380, 0.625] | 0.621 [0.610, 0.633] | 0.550 [0.540, 0.561] | 0.708 [0.630, 0.799] | 0.555 [0.496, 0.628] | 0.477 [0.371, 0.601] |

**T2: AUC [95% CI]**

| feature set | P1 | P1a | P1b | P2 | P2a | P2d | P3 | P4 | I1a | I1b | libnotes_mf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 47 | 0.955 [0.925, 0.980] | 0.956 [0.921, 0.986] | 0.953 [0.948, 0.958] | 0.887 [0.845, 0.917] | 0.918 [0.869, 0.948] | 0.885 [0.843, 0.915] | 0.865 [0.857, 0.873] | 0.891 [0.884, 0.898] | 0.934 [0.870, 0.976] | 0.807 [0.696, 0.900] | 0.530 [0.370, 0.708] |
| only A medians (16) | 0.757 [0.631, 0.857] | 0.759 [0.625, 0.876] | 0.748 [0.736, 0.760] | 0.676 [0.621, 0.709] | 0.800 [0.762, 0.901] | 0.666 [0.607, 0.702] | 0.685 [0.674, 0.697] | 0.889 [0.884, 0.893] | 0.834 [0.748, 0.907] | 0.684 [0.598, 0.774] | 0.445 [0.299, 0.613] |
| only A IQRs (16) | 0.812 [0.712, 0.893] | 0.811 [0.691, 0.911] | 0.813 [0.803, 0.822] | 0.710 [0.679, 0.740] | 0.819 [0.749, 0.923] | 0.702 [0.673, 0.728] | 0.724 [0.712, 0.736] | 0.776 [0.765, 0.787] | 0.744 [0.633, 0.866] | 0.653 [0.540, 0.782] | 0.497 [0.385, 0.631] |
| only B register (8) | 0.738 [0.629, 0.837] | 0.733 [0.607, 0.852] | 0.762 [0.750, 0.774] | 0.712 [0.643, 0.757] | 0.766 [0.661, 0.804] | 0.708 [0.638, 0.752] | 0.712 [0.702, 0.723] | 0.898 [0.891, 0.906] | 0.774 [0.692, 0.858] | 0.682 [0.607, 0.764] | 0.641 [0.560, 0.723] |
| only C tuning (7) | 0.731 [0.644, 0.799] | 0.717 [0.622, 0.805] | 0.802 [0.793, 0.810] | 0.791 [0.758, 0.817] | 0.786 [0.656, 0.865] | 0.791 [0.761, 0.817] | 0.694 [0.683, 0.704] | 0.672 [0.661, 0.683] | 0.750 [0.669, 0.838] | 0.692 [0.621, 0.772] | 0.583 [0.522, 0.661] |
| all minus A medians (31) | 0.941 [0.897, 0.975] | 0.940 [0.887, 0.981] | 0.950 [0.944, 0.955] | 0.868 [0.821, 0.902] | 0.936 [0.888, 0.969] | 0.863 [0.816, 0.897] | 0.828 [0.821, 0.836] | 0.894 [0.888, 0.901] | 0.825 [0.753, 0.901] | 0.740 [0.659, 0.832] | 0.602 [0.508, 0.702] |
| all minus A IQRs (31) | 0.937 [0.898, 0.969] | 0.941 [0.898, 0.977] | 0.917 [0.910, 0.923] | 0.863 [0.828, 0.890] | 0.834 [0.760, 0.901] | 0.865 [0.829, 0.896] | 0.840 [0.832, 0.848] | 0.912 [0.906, 0.918] | 0.940 [0.879, 0.979] | 0.829 [0.725, 0.911] | 0.548 [0.388, 0.733] |
| all minus B (39) | 0.942 [0.889, 0.982] | 0.939 [0.878, 0.988] | 0.956 [0.951, 0.961] | 0.847 [0.788, 0.893] | 0.912 [0.853, 0.969] | 0.842 [0.784, 0.888] | 0.839 [0.831, 0.848] | 0.839 [0.831, 0.846] | 0.926 [0.856, 0.971] | 0.789 [0.668, 0.886] | 0.507 [0.336, 0.689] |
| all minus C (40) | 0.890 [0.828, 0.941] | 0.885 [0.817, 0.950] | 0.916 [0.909, 0.922] | 0.803 [0.764, 0.829] | 0.868 [0.809, 0.913] | 0.798 [0.759, 0.823] | 0.772 [0.762, 0.782] | 0.868 [0.860, 0.877] | 0.929 [0.863, 0.974] | 0.802 [0.689, 0.897] | 0.493 [0.338, 0.664] |
| only 4 decay/sustain medians | 0.672 [0.591, 0.759] | 0.674 [0.582, 0.777] | 0.662 [0.652, 0.673] | 0.661 [0.612, 0.689] | 0.675 [0.623, 0.820] | 0.660 [0.601, 0.694] | 0.651 [0.639, 0.664] | 0.890 [0.886, 0.894] | 0.762 [0.652, 0.845] | 0.652 [0.550, 0.727] | 0.319 [0.158, 0.497] |
| 12 other A medians | 0.684 [0.526, 0.818] | 0.683 [0.513, 0.837] | 0.689 [0.676, 0.702] | 0.519 [0.478, 0.560] | 0.706 [0.654, 0.804] | 0.505 [0.460, 0.548] | 0.590 [0.578, 0.601] | 0.815 [0.808, 0.823] | 0.718 [0.648, 0.798] | 0.545 [0.494, 0.597] | 0.545 [0.469, 0.635] |

**Detection rate (P > 0.5) of the synthetic files, T1 / T2**

| feature set | P1 | P1a | P1b | P2 | P2a | P2d | P3 | P4 | I1a | I1b | libnotes_mf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 47 | 57.5% / 37.5% | 56.0% / 42.0% | 65.0% / 15.0% | 34.4% / 3.1% | 22.2% / 0.0% | 35.3% / 3.4% | 42.9% / 0.0% | 20.0% / 6.7% | 88.5% / 88.3% | 61.7% / 50.0% | 12.9% / 13.4% |
| only A medians (16) | 22.5% / 18.3% | 25.0% / 22.0% | 10.0% / 0.0% | 11.7% / 0.0% | 33.3% / 0.0% | 10.1% / 0.0% | 28.6% / 4.8% | 30.0% / 43.3% | 76.9% / 75.3% | 33.3% / 33.3% | 9.8% / 12.9% |
| only A IQRs (16) | 32.5% / 19.2% | 35.0% / 23.0% | 20.0% / 0.0% | 14.1% / 0.8% | 33.3% / 11.1% | 12.6% / 0.0% | 19.0% / 4.8% | 6.7% / 3.3% | 90.9% / 89.2% | 85.8% / 83.3% | 43.8% / 42.9% |
| only B register (8) | 27.5% / 15.8% | 33.0% / 19.0% | 0.0% / 0.0% | 14.1% / 2.3% | 0.0% / 0.0% | 15.1% / 2.5% | 19.0% / 4.8% | 26.7% / 13.3% | 87.1% / 84.4% | 72.5% / 73.3% | 67.4% / 65.2% |
| only C tuning (7) | 23.3% / 15.8% | 21.0% / 13.0% | 35.0% / 30.0% | 18.0% / 4.7% | 11.1% / 0.0% | 18.5% / 5.0% | 23.8% / 4.8% | 10.0% / 6.7% | 86.5% / 83.6% | 85.0% / 89.2% | 62.1% / 59.8% |
| all minus A medians (31) | 55.8% / 30.8% | 60.0% / 36.0% | 35.0% / 5.0% | 33.6% / 3.9% | 22.2% / 0.0% | 34.5% / 4.2% | 38.1% / 4.8% | 23.3% / 6.7% | 83.6% / 85.2% | 73.3% / 65.0% | 39.3% / 37.5% |
| all minus A IQRs (31) | 50.0% / 34.2% | 52.0% / 37.0% | 40.0% / 20.0% | 27.3% / 3.9% | 22.2% / 0.0% | 27.7% / 4.2% | 57.1% / 4.8% | 30.0% / 10.0% | 90.0% / 89.6% | 65.8% / 55.0% | 13.8% / 14.3% |
| all minus B (39) | 60.8% / 43.3% | 60.0% / 51.0% | 65.0% / 5.0% | 28.1% / 0.8% | 33.3% / 11.1% | 27.7% / 0.0% | 33.3% / 0.0% | 13.3% / 3.3% | 88.3% / 88.2% | 55.8% / 53.3% | 10.7% / 13.4% |
| all minus C (40) | 45.0% / 27.5% | 44.0% / 33.0% | 50.0% / 0.0% | 18.8% / 0.0% | 33.3% / 0.0% | 17.6% / 0.0% | 42.9% / 4.8% | 10.0% / 0.0% | 89.6% / 89.6% | 53.3% / 46.7% | 12.5% / 12.9% |
| only 4 decay/sustain medians | 22.5% / 19.2% | 25.0% / 21.0% | 10.0% / 10.0% | 17.2% / 14.8% | 11.1% / 22.2% | 17.6% / 14.3% | 19.0% / 0.0% | 56.7% / 76.7% | 68.9% / 66.0% | 35.8% / 44.2% | 10.3% / 8.9% |
| 12 other A medians | 19.2% / 11.7% | 23.0% / 14.0% | 0.0% / 0.0% | 2.3% / 0.8% | 11.1% / 0.0% | 1.7% / 0.8% | 14.3% / 4.8% | 16.7% / 13.3% | 71.5% / 68.5% | 47.5% / 37.5% | 36.6% / 40.6% |

Real false alarms under T2 (real performances / CCMusic notes): all 47 0.3% / 16.0%; only A medians (16) 1.9% / 22.6%; only A IQRs (16) 1.1% / 39.7%; only B register (8) 2.3% / 41.2%; only C tuning (7) 1.2% / 40.0%; all minus A medians (31) 0.3% / 30.4%; all minus A IQRs (31) 0.5% / 14.5%; all minus B (39) 0.3% / 18.3%; all minus C (40) 0.5% / 16.3%; only 4 decay/sustain medians 6.1% / 29.2%; 12 other A medians 2.6% / 37.3%.

### 7.2 Attribution for P1 and P2 (XGBoost all 47, T2)

Exact TreeSHAP values from XGBoost (`pred_contribs`, log-odds units) of the very fold models that produce the section-2 scores (max |score difference| to section 2 = 0.0e+00); out-of-fold rows use their fold model, rows excluded from training under T2 (demos, M1) the mean over the 5 fold models. "Separation" = mean SHAP over the synthetic files minus mean SHAP over the real performances (positive = the feature pushes this synthetic kind towards "synthetic" relative to real performances). "gain" = total-gain share, mean over the 5 fold models (model-wide, not comparison-specific).

**P1** (real_performance vs algo_performance; 2576 real vs 120 synthetic files). Separation summed by group (log-odds): A_med +1.93, A_iqr +1.19, B +1.33, C +1.80; total +6.25.

| rank | feature | group | separation | share of positive separation | mean |SHAP| (both classes) | gain share |
|---|---|---|---|---|---|---|
| 1 | bass_fund_strength | B | +0.683 | 11% | 1.253 | 9.2% |
| 2 | frac_split | C | +0.538 | 8% | 0.510 | 2.4% |
| 3 | global_scatter | C | +0.445 | 7% | 0.866 | 17.8% |
| 4 | log_t60_med | A_med | +0.404 | 6% | 0.837 | 5.9% |
| 5 | hf_decay_excess_iqr | A_iqr | +0.390 | 6% | 0.598 | 2.5% |
| 6 | detune_final | C | +0.382 | 6% | 0.453 | 3.6% |
| 7 | spectral_slope_med | A_med | +0.348 | 5% | 0.330 | 3.1% |
| 8 | centroid_h_med | A_med | +0.303 | 5% | 0.358 | 1.5% |
| 9 | log_inharmB_med | A_med | +0.287 | 5% | 0.464 | 1.9% |
| 10 | unison_index | C | +0.270 | 4% | 0.339 | 2.1% |

**P1a** (... vs SoundFont performance renders (sf2_perf); 2576 real vs 100 synthetic files). Separation summed by group (log-odds): A_med +2.17, A_iqr +1.21, B +1.40, C +1.68; total +6.45.

| rank | feature | group | separation | share of positive separation | mean |SHAP| (both classes) | gain share |
|---|---|---|---|---|---|---|
| 1 | bass_fund_strength | B | +0.699 | 11% | 1.258 | 9.2% |
| 2 | frac_split | C | +0.538 | 8% | 0.511 | 2.4% |
| 3 | centroid_h_med | A_med | +0.459 | 7% | 0.355 | 1.5% |
| 4 | spectral_slope_med | A_med | +0.420 | 6% | 0.331 | 3.1% |
| 5 | global_scatter | C | +0.372 | 6% | 0.869 | 17.8% |
| 6 | log_t60_med | A_med | +0.370 | 6% | 0.835 | 5.9% |
| 7 | detune_final | C | +0.359 | 5% | 0.454 | 3.6% |
| 8 | hf_decay_excess_iqr | A_iqr | +0.335 | 5% | 0.601 | 2.5% |
| 9 | unison_index | C | +0.262 | 4% | 0.339 | 2.1% |
| 10 | log_inharmB_med | A_med | +0.228 | 3% | 0.466 | 1.9% |

**P1b** (... vs Otey performance renders (otey_perf); 2576 real vs 20 synthetic files). Separation summed by group (log-odds): A_med +0.77, A_iqr +1.07, B +0.98, C +2.41; total +5.23.

| rank | feature | group | separation | share of positive separation | mean |SHAP| (both classes) | gain share |
|---|---|---|---|---|---|---|
| 1 | global_scatter | C | +0.809 | 12% | 0.874 | 17.8% |
| 2 | hf_decay_excess_iqr | A_iqr | +0.668 | 10% | 0.604 | 2.5% |
| 3 | bass_fund_strength | B | +0.605 | 9% | 1.271 | 9.2% |
| 4 | log_inharmB_med | A_med | +0.584 | 8% | 0.466 | 1.9% |
| 5 | log_t60_med | A_med | +0.573 | 8% | 0.833 | 5.9% |
| 6 | frac_split | C | +0.541 | 8% | 0.512 | 2.4% |
| 7 | detune_final | C | +0.498 | 7% | 0.457 | 3.6% |
| 8 | unison_index | C | +0.305 | 4% | 0.342 | 2.1% |
| 9 | log_t60_iqr | A_iqr | +0.278 | 4% | 0.309 | 0.8% |
| 10 | infl_density_iqr | A_iqr | +0.237 | 3% | 0.231 | 1.7% |

**P2** (real_performance vs sampled digital pianos (old demos + digital_v2); 2576 real vs 128 synthetic files). Separation summed by group (log-odds): A_med +0.34, A_iqr +0.26, B +1.24, C +1.51; total +3.35.

| rank | feature | group | separation | share of positive separation | mean |SHAP| (both classes) | gain share |
|---|---|---|---|---|---|---|
| 1 | bass_fund_strength | B | +0.825 | 21% | 1.253 | 9.2% |
| 2 | frac_split | C | +0.564 | 14% | 0.506 | 2.4% |
| 3 | hf_decay_excess_iqr | A_iqr | +0.439 | 11% | 0.597 | 2.5% |
| 4 | detune_final | C | +0.418 | 11% | 0.447 | 3.6% |
| 5 | bass_h2_over_h1 | B | +0.304 | 8% | 0.387 | 1.7% |
| 6 | bright_decay_med | A_med | +0.267 | 7% | 0.299 | 2.2% |
| 7 | unison_index | C | +0.237 | 6% | 0.335 | 2.1% |
| 8 | global_scatter | C | +0.178 | 5% | 0.871 | 17.8% |
| 9 | hf_decay_excess_med | A_med | +0.129 | 3% | 0.138 | 0.8% |
| 10 | sustain_db_1s_med | A_med | +0.108 | 3% | 0.213 | 3.5% |

**P2d** (... digital_v2 only; 2576 real vs 119 synthetic files). Separation summed by group (log-odds): A_med +0.33, A_iqr +0.21, B +1.27, C +1.52; total +3.33.

| rank | feature | group | separation | share of positive separation | mean |SHAP| (both classes) | gain share |
|---|---|---|---|---|---|---|
| 1 | bass_fund_strength | B | +0.841 | 21% | 1.255 | 9.2% |
| 2 | frac_split | C | +0.574 | 15% | 0.506 | 2.4% |
| 3 | hf_decay_excess_iqr | A_iqr | +0.436 | 11% | 0.597 | 2.5% |
| 4 | detune_final | C | +0.419 | 11% | 0.448 | 3.6% |
| 5 | bass_h2_over_h1 | B | +0.311 | 8% | 0.388 | 1.7% |
| 6 | bright_decay_med | A_med | +0.266 | 7% | 0.299 | 2.2% |
| 7 | unison_index | C | +0.240 | 6% | 0.336 | 2.1% |
| 8 | global_scatter | C | +0.174 | 4% | 0.871 | 17.8% |
| 9 | hf_decay_excess_med | A_med | +0.128 | 3% | 0.138 | 0.8% |
| 10 | sustain_db_1s_med | A_med | +0.113 | 3% | 0.213 | 3.5% |

**P3** (real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland); 2576 real vs 21 synthetic files). Separation summed by group (log-odds): A_med +0.69, A_iqr +0.25, B +1.09, C +0.99; total +3.02.

| rank | feature | group | separation | share of positive separation | mean |SHAP| (both classes) | gain share |
|---|---|---|---|---|---|---|
| 1 | bass_fund_strength | B | +0.556 | 16% | 1.272 | 9.2% |
| 2 | frac_split | C | +0.415 | 12% | 0.513 | 2.4% |
| 3 | bass_h2_over_h1 | B | +0.318 | 9% | 0.396 | 1.7% |
| 4 | hf_decay_excess_iqr | A_iqr | +0.303 | 9% | 0.606 | 2.5% |
| 5 | bright_decay_med | A_med | +0.290 | 8% | 0.301 | 2.2% |
| 6 | detune_final | C | +0.244 | 7% | 0.457 | 3.6% |
| 7 | unison_index | C | +0.158 | 4% | 0.341 | 2.1% |
| 8 | bass_body_no_fund | B | +0.157 | 4% | 0.321 | 2.2% |
| 9 | sustain_db_1s_med | A_med | +0.157 | 4% | 0.213 | 3.5% |
| 10 | global_scatter | C | +0.137 | 4% | 0.876 | 17.8% |

Model-wide total-gain top 10 (T2, all 47): global_scatter (C) 17.8%, bright_decay_iqr (A_iqr) 17.2%, bass_fund_strength (B) 9.2%, log_t60_med (A_med) 5.9%, detune_final (C) 3.6%, sustain_db_1s_med (A_med) 3.5%, upper_body_no_fund (B) 3.2%, spectral_slope_med (A_med) 3.1%, hf_decay_excess_iqr (A_iqr) 2.5%, frac_split (C) 2.4%.

### 7.3 "Too-perfect tuning" test: Group C (tuning) features

All seven are "more detuned = larger": detune_final (integral detuning 0-100, 0 = perfect), unison_index (unison spread 0-100), global_index (note-to-note inconsistency 0-100, 0 below 12 cents scatter), med_spread_cents (median resolved unison split), frac_split (share of notes with a resolved split), global_scatter (residual pitch scatter, cents), ref_offset_abs (|reference - A440|, cents). The hypothesis predicts digital pianos BELOW real performances (P(other > real) < 0.5).

Median [Q1, Q3] per file (n files: real perf: manifest 432, real perf: youtube_extra 2144, real perf: all 2576, sampled demos: old 9, sampled demos: digital_v2 119, sampled demos: digital_v2, line 12, sampled demos: digital_v2, mic 2, sampled demos: digital_v2, unknown 105, modelled digital 21, digital other (GP-3) 1, SC-55 module 2, SoundFont perf renders 100, Otey perf renders 20, Korg M1 30, CCMusic isolated (ref.) 668):

| feature | real perf: manifest | real perf: youtube_extra | real perf: all | sampled demos: old | sampled demos: digital_v2 | sampled demos: digital_v2, line | sampled demos: digital_v2, mic | sampled demos: digital_v2, unknown | modelled digital | digital other (GP-3) | SC-55 module | SoundFont perf renders | Otey perf renders | Korg M1 | CCMusic isolated (ref.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| detune_final | 10.4 [6.68, 16] | 11.7 [7.76, 17.4] | 11.6 [7.59, 17.2] | 6.66 [4.95, 7.86] | 5.67 [4.01, 7.19] | 4.01 [3.7, 6.94] | 3.09 [2.51, 3.68] | 5.75 [4.08, 7.21] | 7.19 [4.94, 9.56] | 1.66 [1.66, 1.66] | 4.36 [2.66, 6.05] | 6.35 [3.19, 9.4] | 5.73 [2.8, 7.42] | 11.1 [4.86, 17.6] | 8.97 [0, 57] |
| unison_index | 13.8 [8.95, 20.5] | 15.1 [10.1, 21.5] | 14.9 [9.86, 21.3] | 9.11 [7, 10.9] | 7.93 [5.59, 9.93] | 5.64 [5.13, 9.71] | 4.35 [3.51, 5.18] | 8.06 [5.65, 9.93] | 9.69 [5.81, 12.5] | 2.28 [2.28, 2.28] | 6.01 [3.63, 8.39] | 8.68 [4.45, 12.9] | 8.08 [3.96, 10.5] | 13.2 [6.58, 22.3] | 9.32 [0, 63.5] |
| global_index | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 42.9] |
| med_spread_cents | 18.8 [14.5, 24.4] | 17.6 [13.9, 22.7] | 17.9 [14, 23] | 16.2 [13.9, 18.1] | 17.1 [14.3, 24] | 18.8 [14.4, 27.4] | 24.3 [20.1, 28.5] | 16.9 [14.3, 23.5] | 17.6 [12.6, 24.7] | 14.7 [14.7, 14.7] | 12.2 [11.1, 13.3] | 16.8 [12.7, 22.4] | 20.4 [14.8, 22.8] | 11.4 [8.6, 15.3] | 4.6 [0, 39.2] |
| frac_split | 0.204 [0.138, 0.279] | 0.225 [0.152, 0.318] | 0.221 [0.148, 0.311] | 0.152 [0.126, 0.169] | 0.103 [0.0726, 0.158] | 0.0991 [0.0682, 0.128] | 0.0519 [0.0403, 0.0636] | 0.107 [0.0735, 0.161] | 0.138 [0.0981, 0.169] | 0.0806 [0.0806, 0.0806] | 0.104 [0.0671, 0.141] | 0.124 [0.0659, 0.201] | 0.09 [0.0425, 0.141] | 0.277 [0.144, 0.444] | 0.333 [0, 1] |
| global_scatter | 8.73 [7.03, 11.5] | 9.07 [7.04, 11.6] | 9.02 [7.03, 11.6] | 8.05 [6.75, 8.7] | 7 [5.85, 8.4] | 6.22 [5.5, 7.56] | 6.11 [5.98, 6.23] | 7.08 [6.12, 8.59] | 7.55 [5.87, 8.28] | 11 [11, 11] | 5.25 [4.02, 6.48] | 7.42 [5.55, 9.18] | 5.86 [3.71, 8] | 6.24 [4.68, 7.96] | 10.9 [5.11, 19.3] |
| ref_offset_abs | 0.447 [0.197, 0.778] | 0.429 [0.204, 0.743] | 0.434 [0.202, 0.753] | 0.306 [0.257, 0.613] | 0.385 [0.165, 0.605] | 0.545 [0.255, 0.608] | 0.206 [0.161, 0.251] | 0.385 [0.165, 0.607] | 0.405 [0.26, 0.759] | 1.48 [1.48, 1.48] | 0.215 [0.2, 0.231] | 0.428 [0.204, 0.651] | 0.298 [0.22, 0.468] | 0.484 [0.271, 0.632] | 12.7 [6.55, 23.7] |

Mann-Whitney against ALL real performances (2576 files): cell = P(other > real) [0.5 = same distribution; < 0.5 = the other kind is lower] and two-sided p. File-level test: demos cluster in 5 brands, so the p-values overstate the independent evidence. Last column: real manifest vs real youtube_extra (within-real heterogeneity).

| feature | vs sampled demos: old | vs sampled demos: digital_v2 | vs sampled demos: digital_v2, line | vs sampled demos: digital_v2, mic | vs sampled demos: digital_v2, unknown | vs modelled digital | vs SoundFont perf renders | vs Otey perf renders | vs Korg M1 | manifest vs youtube_extra |
|---|---|---|---|---|---|---|---|---|---|---|
| detune_final | 0.21 (p=0.003) | 0.17 (p=1.9e-33) | 0.13 (p=1.1e-05) | 0.04 (p=0.026) | 0.18 (p=1.3e-28) | 0.30 (p=0.0012) | 0.24 (p=5.4e-19) | 0.17 (p=5.2e-07) | 0.46 (p=0.44) | 0.45 (p=0.0025) |
| unison_index | 0.24 (p=0.0067) | 0.19 (p=4.9e-31) | 0.15 (p=2.5e-05) | 0.05 (p=0.028) | 0.19 (p=1.4e-26) | 0.29 (p=0.00081) | 0.26 (p=2.7e-16) | 0.20 (p=3.1e-06) | 0.46 (p=0.43) | 0.46 (p=0.0042) |
| global_index | 0.39 (p=0.11) | 0.41 (p=2.2e-06) | 0.39 (p=0.068) | 0.39 (p=0.46) | 0.41 (p=1.5e-05) | 0.46 (p=0.39) | 0.42 (p=6.4e-05) | 0.39 (p=0.019) | 0.42 (p=0.049) | 0.50 (p=0.8) |
| med_spread_cents | 0.40 (p=0.29) | 0.51 (p=0.84) | 0.55 (p=0.54) | 0.66 (p=0.42) | 0.50 (p=0.93) | 0.50 (p=0.96) | 0.45 (p=0.069) | 0.53 (p=0.6) | 0.26 (p=8.2e-06) | 0.54 (p=0.0072) |
| frac_split | 0.24 (p=0.0075) | 0.17 (p=9.9e-35) | 0.14 (p=1.5e-05) | 0.03 (p=0.021) | 0.17 (p=5.7e-30) | 0.25 (p=0.00011) | 0.25 (p=9.8e-18) | 0.14 (p=3.1e-08) | 0.56 (p=0.26) | 0.44 (p=3.4e-05) |
| global_scatter | 0.35 (p=0.13) | 0.28 (p=1.8e-15) | 0.21 (p=0.00043) | 0.13 (p=0.071) | 0.30 (p=1.5e-12) | 0.36 (p=0.023) | 0.32 (p=6.7e-10) | 0.19 (p=2.1e-06) | 0.25 (p=2.9e-06) | 0.49 (p=0.45) |
| ref_offset_abs | 0.44 (p=0.55) | 0.45 (p=0.087) | 0.49 (p=0.88) | 0.25 (p=0.22) | 0.45 (p=0.11) | 0.50 (p=0.97) | 0.48 (p=0.54) | 0.41 (p=0.17) | 0.51 (p=0.83) | 0.51 (p=0.5) |

### 7.4 "Dry channel" test: decay / sustain features

Group-A medians are register-detrended residuals (0 = the corpus trend); log_t60 = log10 T60 of the note window, sustain_db_1s = level 1 s after onset re peak, hf_decay_excess = HF minus LF energy change over the note, bright_decay = change of centroid/f0 early -> late. In performances the analysis window ends at the next onset (<= 2.5 s), so these measure the early decay as recorded, including the room. A dry (direct / line) signal predicts faster, cleaner decays than room-miked real pianos.

Median [Q1, Q3] per file (n files: real perf: manifest 432, real perf: youtube_extra 2144, real perf: all 2576, sampled demos: old 9, sampled demos: digital_v2 119, sampled demos: digital_v2, line 12, sampled demos: digital_v2, mic 2, sampled demos: digital_v2, unknown 105, modelled digital 21, digital other (GP-3) 1, SC-55 module 2, SoundFont perf renders 100, Otey perf renders 20, Korg M1 30, CCMusic isolated (ref.) 668):

| feature | real perf: manifest | real perf: youtube_extra | real perf: all | sampled demos: old | sampled demos: digital_v2 | sampled demos: digital_v2, line | sampled demos: digital_v2, mic | sampled demos: digital_v2, unknown | modelled digital | digital other (GP-3) | SC-55 module | SoundFont perf renders | Otey perf renders | Korg M1 | CCMusic isolated (ref.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| log_t60_med | 0.00638 [-0.102, 0.0961] | -0.00199 [-0.0969, 0.107] | -0.000762 [-0.0971, 0.106] | 0.0811 [0.0216, 0.192] | 0.0839 [0.0351, 0.152] | 0.123 [0.0874, 0.171] | 0.0795 [0.0662, 0.0928] | 0.0742 [0.0247, 0.145] | 0.16 [-0.0148, 0.213] | 0.17 [0.17, 0.17] | 0.183 [0.151, 0.216] | -0.0227 [-0.11, 0.0479] | -0.0604 [-0.133, 0.0216] | 0.162 [0.0951, 0.243] | 0.293 [0.0814, 0.453] |
| sustain_db_1s_med | 0.921 [-1.86, 3.32] | 0.797 [-2.33, 3.06] | 0.816 [-2.23, 3.09] | 2.44 [0.273, 2.69] | 2.3 [0.509, 3.51] | 1.25 [0.781, 1.76] | 7.06 [6.93, 7.18] | 2.34 [0.447, 3.47] | 2.97 [0.191, 4.04] | -2.46 [-2.46, -2.46] | 1.88 [1.03, 2.72] | 2.03 [-1.6, 4.52] | 1.08 [-0.329, 2.25] | 0.66 [-3.36, 1.79] | -5.01 [-10.4, -0.512] |
| hf_decay_excess_med | -0.0103 [-0.486, 0.547] | 0.0228 [-0.547, 0.578] | 0.0176 [-0.535, 0.576] | -0.629 [-1, -0.462] | -0.53 [-0.925, -0.156] | -0.666 [-1.33, -0.167] | -0.603 [-0.996, -0.21] | -0.517 [-0.886, -0.157] | -0.325 [-0.669, -0.0282] | -0.68 [-0.68, -0.68] | -0.34 [-0.614, -0.0654] | -0.141 [-0.997, 0.276] | -0.263 [-0.553, 0.221] | -1.7 [-2.83, 0.0241] | -1.27 [-5.8, 1.9] |
| bright_decay_med | -0.0256 [-0.251, 0.185] | -0.00253 [-0.204, 0.229] | -0.00633 [-0.214, 0.223] | -0.243 [-0.447, -0.182] | -0.268 [-0.488, -0.0562] | -0.341 [-0.473, -0.274] | -0.132 [-0.222, -0.0426] | -0.248 [-0.49, -0.0544] | -0.312 [-0.429, -0.181] | -0.258 [-0.258, -0.258] | -0.0554 [-0.147, 0.0365] | -0.31 [-0.809, -0.0163] | -0.378 [-0.687, -0.043] | -1.62 [-2.36, -0.938] | 0.184 [-0.926, 1.7] |
| sustain_db_1s_iqr | 3.46 [1.57, 5.27] | 3.62 [1.69, 5.62] | 3.6 [1.66, 5.55] | 4.43 [3.69, 5.28] | 3.9 [2.56, 5.53] | 5.08 [3.35, 5.78] | 4.91 [3.91, 5.91] | 3.81 [2.46, 5.52] | 3.24 [2.26, 4.74] | 6.13 [6.13, 6.13] | 2.62 [2.1, 3.13] | 2.9 [0.0262, 5.3] | 3.63 [1.17, 9.37] | 5.4 [4.37, 6.83] | 0.257 [0, 4.14] |
| log_t60_iqr | 0.431 [0.367, 0.532] | 0.437 [0.357, 0.529] | 0.436 [0.358, 0.529] | 0.506 [0.469, 0.571] | 0.421 [0.358, 0.507] | 0.421 [0.394, 0.473] | 0.488 [0.438, 0.538] | 0.421 [0.351, 0.507] | 0.434 [0.391, 0.538] | 0.38 [0.38, 0.38] | 0.488 [0.421, 0.555] | 0.492 [0.359, 0.596] | 0.534 [0.417, 0.593] | 0.4 [0.34, 0.425] | 0.0409 [0, 0.177] |

Mann-Whitney against ALL real performances (2576 files): cell = P(other > real) [0.5 = same distribution; < 0.5 = the other kind is lower] and two-sided p. File-level test: demos cluster in 5 brands, so the p-values overstate the independent evidence. Last column: real manifest vs real youtube_extra (within-real heterogeneity).

| feature | vs sampled demos: old | vs sampled demos: digital_v2 | vs sampled demos: digital_v2, line | vs sampled demos: digital_v2, mic | vs sampled demos: digital_v2, unknown | vs modelled digital | vs SoundFont perf renders | vs Otey perf renders | vs Korg M1 | manifest vs youtube_extra |
|---|---|---|---|---|---|---|---|---|---|---|
| log_t60_med | 0.65 (p=0.12) | 0.66 (p=1.2e-09) | 0.77 (p=0.0012) | 0.70 (p=0.34) | 0.65 (p=1.3e-07) | 0.68 (p=0.0037) | 0.44 (p=0.027) | 0.38 (p=0.069) | 0.71 (p=5.9e-05) | 0.50 (p=0.97) |
| sustain_db_1s_med | 0.55 (p=0.64) | 0.62 (p=2e-05) | 0.56 (p=0.48) | 0.98 (p=0.018) | 0.62 (p=6.7e-05) | 0.67 (p=0.007) | 0.57 (p=0.025) | 0.50 (p=0.98) | 0.41 (p=0.083) | 0.52 (p=0.2) |
| hf_decay_excess_med | 0.29 (p=0.028) | 0.29 (p=7.8e-15) | 0.27 (p=0.0059) | 0.33 (p=0.39) | 0.29 (p=3.7e-13) | 0.34 (p=0.013) | 0.41 (p=0.0019) | 0.40 (p=0.11) | 0.25 (p=2.9e-06) | 0.49 (p=0.71) |
| bright_decay_med | 0.27 (p=0.016) | 0.28 (p=7.2e-16) | 0.21 (p=0.00046) | 0.37 (p=0.51) | 0.29 (p=1.9e-13) | 0.24 (p=3.9e-05) | 0.30 (p=7.9e-12) | 0.29 (p=0.001) | 0.11 (p=3e-13) | 0.47 (p=0.072) |
| sustain_db_1s_iqr | 0.58 (p=0.42) | 0.55 (p=0.063) | 0.62 (p=0.17) | 0.63 (p=0.53) | 0.54 (p=0.15) | 0.49 (p=0.92) | 0.44 (p=0.088) | 0.54 (p=0.57) | 0.69 (p=0.00039) | 0.48 (p=0.25) |
| log_t60_iqr | 0.66 (p=0.094) | 0.47 (p=0.33) | 0.46 (p=0.59) | 0.61 (p=0.58) | 0.47 (p=0.35) | 0.52 (p=0.77) | 0.56 (p=0.03) | 0.63 (p=0.045) | 0.38 (p=0.019) | 0.51 (p=0.6) |

**Per file, old digital demos** (P = XGB all 47 T2 score):

| file | kind | group | P | detune_final | unison_index | global_index | frac_split | global_scatter | log_t60_med | sustain_db_1s_med | hf_decay_excess_med | bright_decay_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KORG_B2_-_Digital_Piano_Introduction_D | sampled_digital | korg | 0.005 | 6.657 | 9.109 | 0.000 | 0.181 | 5.305 | 0.246 | 3.154 | -0.555 | -0.243 |
| Korg_D1_Digital_Piano_-_Demo_with_Fran | sampled_digital | korg | 0.011 | 4.952 | 7.003 | 0.000 | 0.126 | 8.054 | -0.230 | -6.829 | -1.614 | 0.210 |
| KORG_Liano_Digital_Piano_-_Top_Feature | sampled_digital | korg | 0.284 | 4.119 | 5.844 | 0.000 | 0.140 | 10.078 | 0.042 | 2.687 | 0.515 | -0.447 |
| Korg_SP-280_Digital_Piano_-_Acoustic_a | sampled_digital | korg | 0.141 | 6.858 | 9.567 | 0.000 | 0.152 | 7.795 | -0.003 | -0.656 | -0.462 | -0.244 |
| Kurzweil_MP10_F_Grand_Piano_demo_Tripl | sampled_digital | kurzweil | 0.150 | 7.864 | 10.916 | 0.000 | 0.169 | 6.748 | 0.090 | 2.738 | 0.280 | -1.732 |
| PRIVIA_PX-S3000_SOUND_DEMO_v3fp315GBqA | sampled_digital | casio | 0.004 | 9.591 | 13.585 | 0.000 | 0.190 | 8.703 | 0.022 | 2.596 | -0.629 | 0.021 |
| Roland_GP-3_Ultra-Compact_Grand_Piano_ | digital_other | roland | 0.164 | 1.659 | 2.277 | 0.000 | 0.081 | 10.969 | 0.170 | -2.457 | -0.680 | -0.258 |
| Roland_GP-6_Baby_Grand_Piano_Performan | modelled_digital | roland | 0.164 | 3.427 | 4.704 | 0.000 | 0.095 | 7.989 | 0.222 | 4.099 | 0.147 | -0.312 |
| Roland_GP9M_white_grand_piano_with_mov | modelled_digital | roland | 0.002 | 19.986 | 19.833 | 20.427 | 0.169 | 14.970 | 0.171 | -3.417 | -0.736 | -0.091 |
| sc55_gs.wav | sound_module | roland_sc55 | 0.718 | 7.743 | 10.776 | 0.000 | 0.177 | 7.710 | 0.249 | 0.184 | 0.209 | -0.239 |
| sc55_short.wav | sound_module | roland_sc55 | 0.852 | 0.972 | 1.244 | 0.000 | 0.030 | 2.786 | 0.118 | 3.569 | -0.888 | 0.128 |
| Yamaha_Clavinova_CLP-695_Digital_Grand | sampled_digital | yamaha | 0.067 | 6.111 | 8.611 | 0.000 | 0.114 | 10.165 | 0.192 | 2.436 | -1.007 | -0.610 |
| Yamaha_Clavinova_CLP-795_Digital_Grand | sampled_digital | yamaha | 0.006 | 11.961 | 16.194 | 0.000 | 0.167 | 8.592 | 0.081 | 0.916 | -0.749 | -0.195 |
| Yamaha_Clavinova_CLP-865GP_Digital_Pia | sampled_digital | yamaha | 0.192 | 0.902 | 1.199 | 0.000 | 0.043 | 3.749 | 0.276 | 0.273 | -1.002 | -0.182 |
Real performances, median: detune_final 11.6, unison_index 14.9, global_index 0, frac_split 0.221, global_scatter 9.02, log_t60_med -0.000762, sustain_db_1s_med 0.816, hf_decay_excess_med 0.0176, bright_decay_med -0.00633.

**digital_v2 by brand** (medians; P = XGB all 47 T2 score):

| brand | kind | files | models | median P | bass_fund_strength | detune_final | unison_index | global_index | frac_split | global_scatter | log_t60_med | sustain_db_1s_med | hf_decay_excess_med | bright_decay_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| casio | sampled_digital | 16 | 6 | 0.006 | 0.0511 | 5.76 | 8.09 | 0 | 0.106 | 6.59 | 0.0367 | 2.58 | -0.737 | -0.0517 |
| dexibell | sampled_digital | 12 | 5 | 0.016 | 0.1 | 5.66 | 7.93 | 0 | 0.0899 | 6.75 | 0.0666 | 1.98 | -0.448 | -0.234 |
| kawai | sampled_digital | 31 | 11 | 0.024 | 0.0972 | 5.75 | 8.06 | 0 | 0.108 | 7.22 | 0.0973 | 2.26 | -0.665 | -0.33 |
| korg | sampled_digital | 14 | 5 | 0.031 | 0.101 | 5.24 | 7.28 | 0 | 0.0958 | 7.12 | 0.105 | 2.47 | -0.464 | -0.445 |
| kurzweil | sampled_digital | 8 | 3 | 0.017 | 0.0598 | 4.33 | 6.08 | 0 | 0.0891 | 6.97 | 0.056 | 1.26 | -0.565 | -0.131 |
| nord | sampled_digital | 10 | 5 | 0.023 | 0.157 | 8.33 | 11.6 | 0 | 0.166 | 7.07 | 0.0257 | 2.62 | -0.153 | -0.25 |
| roland | modelled_digital | 19 | 7 | 0.022 | 0.0683 | 7.19 | 9.69 | 0 | 0.138 | 7.05 | 0.158 | 2.97 | -0.325 | -0.318 |
| yamaha | sampled_digital | 28 | 10 | 0.102 | 0.104 | 4.94 | 6.85 | 0 | 0.0909 | 7.02 | 0.0992 | 2.65 | -0.358 | -0.23 |
| real performances |  | 2576 |  | 0.001 | 0.0268 | 11.6 | 14.9 | 0 | 0.221 | 9.02 | -0.000762 | 0.816 | 0.0176 | -0.00633 |

### 7.5 "Too-perfect" tuning or simply "in tune"? Real performances split by quality rating

`rating_cal` = the calibrated 0-99 instrument-quality rating stored with every real file in `features_cal.csv` (real performances: median 26.2; bin edges at the 25/50/75/90th percentiles = 15.5, 26.2, 47.7, 60.7). Raters hear tuning, so a negative rating-tuning correlation is expected; the question is whether well-rated real pianos reach the tuning values of the digital sources. Median per group; last column = Spearman rho of the feature with rating_cal over all real performances. bass_fund_strength / bass_h2_over_h1 (Group B: share of the fundamental in bass notes, log h2/h1) are included as recording-chain indicators.

| feature | real: rating < q25 (n=640) | real: q25-q50 (n=618) | real: q50-q75 (n=656) | real: q75-q90 (n=396) | real: top 10% rated (n=266) | sampled demos (old) (n=9) | sampled demos (digital_v2) (n=119) | SoundFont perf (n=100) | Otey perf (n=20) | Korg M1 (n=30) | CCMusic (studio, isolated) (n=668) | rho(rating) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| detune_final | 21.4 | 10.7 | 9.9 | 9.82 | 6.82 | 6.66 | 5.67 | 6.35 | 5.73 | 11.1 | 8.97 | -0.60 |
| unison_index | 24.7 | 13.7 | 13.2 | 13 | 9.09 | 9.11 | 7.93 | 8.68 | 8.08 | 13.2 | 9.32 | -0.52 |
| frac_split | 0.345 | 0.205 | 0.196 | 0.2 | 0.146 | 0.152 | 0.103 | 0.124 | 0.09 | 0.277 | 0.333 | -0.48 |
| global_scatter | 12.1 | 8.65 | 8.56 | 8.3 | 7.75 | 8.05 | 7 | 7.42 | 5.86 | 6.24 | 10.9 | -0.37 |
| bass_fund_strength | 0.0257 | 0.0276 | 0.0264 | 0.0286 | 0.0267 | 0.0605 | 0.0855 | 0.0827 | 0.0783 | 0.122 | 0.112 | +0.03 |
| bass_h2_over_h1 | 0.601 | 0.577 | 0.573 | 0.554 | 0.466 | 0.329 | 0.0419 | 0.177 | 0.555 | 0.235 | 0.184 | -0.06 |
| hf_decay_excess_iqr | 5.95 | 5.55 | 5.55 | 5.29 | 5.06 | 4.28 | 4.17 | 4.56 | 3.62 | 6.24 | 2.17 | -0.17 |
| bright_decay_med | 0.0304 | 0.0105 | -0.00676 | -0.0279 | -0.0684 | -0.243 | -0.268 | -0.31 | -0.378 | -1.62 | 0.184 | -0.10 |

AUC [95% CI] (T2) with the REAL class restricted by rating; synthetic class unchanged. If the separation were "too-perfect tuning", it should collapse against the well-tuned top-rated pianos for the tuning-only model, and so should the all-47 model if tuning carried it.

| id | feature set | vs all real perf (n=2576) | vs top 10% rated (n=266) | vs rating < q25 (n=640) |
|---|---|---|---|---|
| P1 | all 47 | 0.955 [0.925, 0.980] | 0.922 [0.869, 0.965] | 0.972 [0.951, 0.990] |
| P1 | only C tuning (7) | 0.731 [0.644, 0.799] | 0.583 [0.480, 0.670] | 0.901 [0.837, 0.945] |
| P1 | only B register (8) | 0.738 [0.629, 0.837] | 0.737 [0.622, 0.838] | 0.710 [0.595, 0.813] |
| P1 | only A IQRs (16) | 0.812 [0.712, 0.893] | 0.777 [0.669, 0.873] | 0.796 [0.693, 0.881] |
| P1 | only A medians (16) | 0.757 [0.631, 0.857] | 0.754 [0.620, 0.863] | 0.698 [0.577, 0.802] |
| P1 | all minus C (40) | 0.890 [0.828, 0.941] | 0.874 [0.796, 0.934] | 0.870 [0.800, 0.928] |
| P1 | all minus B (39) | 0.942 [0.889, 0.982] | 0.899 [0.812, 0.968] | 0.964 [0.924, 0.991] |
| P1a | all 47 | 0.956 [0.921, 0.986] | 0.924 [0.864, 0.974] | 0.972 [0.945, 0.992] |
| P1a | only C tuning (7) | 0.717 [0.622, 0.805] | 0.565 [0.462, 0.675] | 0.892 [0.827, 0.946] |
| P1a | only B register (8) | 0.733 [0.607, 0.852] | 0.733 [0.606, 0.853] | 0.705 [0.574, 0.824] |
| P1a | only A IQRs (16) | 0.811 [0.691, 0.911] | 0.778 [0.649, 0.886] | 0.798 [0.668, 0.903] |
| P1a | only A medians (16) | 0.759 [0.625, 0.876] | 0.758 [0.608, 0.884] | 0.705 [0.563, 0.828] |
| P1a | all minus C (40) | 0.885 [0.817, 0.950] | 0.868 [0.783, 0.942] | 0.866 [0.784, 0.935] |
| P1a | all minus B (39) | 0.939 [0.878, 0.988] | 0.896 [0.795, 0.975] | 0.962 [0.916, 0.995] |
| P1b | all 47 | 0.953 [0.948, 0.958] | 0.913 [0.891, 0.932] | 0.974 [0.966, 0.980] |
| P1b | only C tuning (7) | 0.802 [0.793, 0.810] | 0.671 [0.644, 0.698] | 0.947 [0.938, 0.955] |
| P1b | only B register (8) | 0.762 [0.750, 0.774] | 0.756 [0.717, 0.792] | 0.733 [0.708, 0.758] |
| P1b | only A IQRs (16) | 0.813 [0.803, 0.822] | 0.773 [0.742, 0.803] | 0.787 [0.766, 0.807] |
| P1b | only A medians (16) | 0.748 [0.736, 0.760] | 0.736 [0.700, 0.771] | 0.661 [0.633, 0.689] |
| P1b | all minus C (40) | 0.916 [0.909, 0.922] | 0.905 [0.884, 0.923] | 0.889 [0.873, 0.905] |
| P1b | all minus B (39) | 0.956 [0.951, 0.961] | 0.911 [0.890, 0.930] | 0.978 [0.970, 0.984] |
| P2 | all 47 | 0.887 [0.845, 0.917] | 0.809 [0.737, 0.862] | 0.928 [0.896, 0.950] |
| P2 | only C tuning (7) | 0.791 [0.758, 0.817] | 0.637 [0.584, 0.685] | 0.951 [0.931, 0.968] |
| P2 | only B register (8) | 0.712 [0.643, 0.757] | 0.705 [0.628, 0.763] | 0.682 [0.614, 0.733] |
| P2 | only A IQRs (16) | 0.710 [0.679, 0.740] | 0.660 [0.611, 0.702] | 0.690 [0.652, 0.725] |
| P2 | only A medians (16) | 0.676 [0.621, 0.709] | 0.653 [0.585, 0.703] | 0.595 [0.540, 0.635] |
| P2 | all minus C (40) | 0.803 [0.764, 0.829] | 0.774 [0.718, 0.818] | 0.772 [0.730, 0.804] |
| P2 | all minus B (39) | 0.847 [0.788, 0.893] | 0.738 [0.654, 0.808] | 0.902 [0.858, 0.936] |
| P2d | all 47 | 0.885 [0.843, 0.915] | 0.806 [0.735, 0.860] | 0.926 [0.893, 0.949] |
| P2d | only C tuning (7) | 0.791 [0.761, 0.817] | 0.639 [0.589, 0.685] | 0.950 [0.931, 0.968] |
| P2d | only B register (8) | 0.708 [0.638, 0.752] | 0.701 [0.624, 0.760] | 0.678 [0.610, 0.729] |
| P2d | only A IQRs (16) | 0.702 [0.673, 0.728] | 0.651 [0.604, 0.692] | 0.682 [0.647, 0.715] |
| P2d | only A medians (16) | 0.666 [0.607, 0.702] | 0.643 [0.571, 0.694] | 0.586 [0.528, 0.627] |
| P2d | all minus C (40) | 0.798 [0.759, 0.823] | 0.769 [0.714, 0.812] | 0.768 [0.727, 0.800] |
| P2d | all minus B (39) | 0.842 [0.784, 0.888] | 0.731 [0.650, 0.800] | 0.898 [0.853, 0.933] |
| P3 | all 47 | 0.865 [0.857, 0.873] | 0.787 [0.758, 0.815] | 0.906 [0.894, 0.918] |
| P3 | only C tuning (7) | 0.694 [0.683, 0.704] | 0.539 [0.505, 0.572] | 0.867 [0.856, 0.877] |
| P3 | only B register (8) | 0.712 [0.702, 0.723] | 0.712 [0.680, 0.744] | 0.682 [0.659, 0.703] |
| P3 | only A IQRs (16) | 0.724 [0.712, 0.736] | 0.674 [0.637, 0.712] | 0.698 [0.672, 0.722] |
| P3 | only A medians (16) | 0.685 [0.674, 0.697] | 0.666 [0.630, 0.700] | 0.611 [0.585, 0.637] |
| P3 | all minus C (40) | 0.772 [0.762, 0.782] | 0.739 [0.707, 0.772] | 0.745 [0.722, 0.766] |
| P3 | all minus B (39) | 0.839 [0.831, 0.848] | 0.729 [0.699, 0.758] | 0.895 [0.882, 0.908] |

## 8. Level-invariance test of the 47 features

python Stattia/revision/analysis/level_test.py      (also called by run_all.py; per-file results cached)  Stratified random sample (numpy default_rng(0)): 50 CCMusic notes, 50 real performances (25 manifest + 25 YouTube, <= 150 s), 40 SoundFont single notes (4 per SoundFont), 30 Otey single notes, 20 SoundFont performance renders (2 per SoundFont) and 10 Otey performance renders.  Every file is re-extracted with the unchanged pipeline (quality.pipeline.note_raw_descriptors + aggregate_file) after transforming the loaded mono signal in memory (detune.segment.load_mono is wrapped, so segmentation, per-note descriptors and the detune analysis all see the transformed audio):     orig   : as stored (float64 from the file)     x0.5   : x * 0.5   (float, no requantisation)     x0.1   : x * 0.1   (float, no requantisation)     pcm16  : peak-normalised to -1 dBFS and rounded to 16-bit (the real corpus' storage convention) Scores: the unchanged XGBoost fold models (T2, protocol v2 folds); every file is scored by the fold model that never saw it.

Sample: real: CCMusic note 50, real: performance 50, synth: Otey note 30, synth: SoundFont note 40, synth: performance render 30. Extraction errors per variant: {'orig': 0, 'x0.5': 0, 'x0.1': 0, 'pcm16': 0}. Re-extraction of the unscaled files reproduces the stored corpus rows to 5.0e-07 and their out-of-fold scores to 6.1e-03.

Native peak level of the sampled files (median dBFS): real: CCMusic note -0.1, real: performance -1.0, synth: Otey note -13.9, synth: SoundFont note -0.9, synth: performance render -0.9. (All 120 Otey notes: median -15.9 dBFS, range -32 to -6; real files are stored at -1 dBFS PCM-16.)

**Segmentation changes** (files whose number of detected onsets / number of analysed note segments / single-vs-piece mode differs from the unscaled file):

| variant | real: CCMusic note | real: performance | synth: Otey note | synth: SoundFont note | synth: performance render |
|---|---|---|---|---|---|
| x0.5 | 11 / 9 / 3 of 50 | 43 / 35 / 0 of 50 | 6 / 2 / 0 of 30 | 23 / 1 / 0 of 40 | 23 / 16 / 0 of 30 |
| x0.1 | 17 / 14 / 9 of 50 | 48 / 38 / 0 of 50 | 6 / 6 / 0 of 30 | 34 / 4 / 7 of 40 | 26 / 23 / 0 of 30 |
| pcm16 | 9 / 5 / 1 of 50 | 11 / 8 / 0 of 50 | 16 / 5 / 9 of 30 | 2 / 0 / 0 of 40 | 6 / 5 / 0 of 30 |

**Per-note descriptors on a fixed window** (segmentation held fixed; isolated-note files): max |change| of the 21 per-note descriptors over the files -- x0.5: 2.10e-09 (0 NaN flips), x0.1: 1.89e-08 (0 NaN flips), pcm16: 5.37e+01 (5 NaN flips). Descriptors changed by requantisation (number of files): centroid_var 119, hf_decay_excess 119, centroid_h 118, spec_irregularity 118, even_odd_db 114, bright_decay 112, sustain_db_1s 105, spectral_slope 104, body_no_fund 85, log_t60 69, h2_over_h1 51, log_inharmB 31, tristim2 23, high_partial_frac 19, fund_strength 13, partial_persist 4, h2_over_h1(NaN) 3, even_odd_db(NaN) 2 (of 120 isolated-note files).

**Per feature**: median over files of |change| / inter-file IQR of the feature in the corpus (std where the IQR is 0), and its 90th percentile; full list in `out/level_test_features.csv`. Summary over the 47 features: x0.5: median 0.000, max 0.071, features with median > 0.1: 0, x0.1: median 0.037, max 0.229, features with median > 0.1: 7, pcm16: median 0.000, max 0.037, features with median > 0.1: 0.

| feature (sorted by x0.1) | x0.5: median / p90 | x0.1: median / p90 | pcm16: median / p90 |
|---|---|---|---|
| ref_offset_abs | 0.071 / 0.63 (3 NaN flips) | 0.229 / 6.85 (7 NaN flips) | 0.000 / 0.57 (2 NaN flips) |
| centroid_var_med | 0.002 / 0.58 | 0.155 / 1.99 | 0.037 / 1.50 |
| deepbass_nonharm_ratio | 0.003 / 0.51 (4 NaN flips) | 0.151 / 0.87 (15 NaN flips) | 0.000 / 0.16 (3 NaN flips) |
| deepbass_phantom_idx | 0.000 / 0.69 (5 NaN flips) | 0.129 / 1.24 (24 NaN flips) | 0.000 / 0.05 (2 NaN flips) |
| global_scatter | 0.030 / 0.21 (3 NaN flips) | 0.128 / 0.91 (7 NaN flips) | 0.000 / 0.20 (2 NaN flips) |
| hf_decay_excess_med | 0.000 / 0.35 | 0.103 / 1.52 | 0.000 / 0.59 |
| attack_time_med | 0.001 / 0.32 | 0.103 / 0.66 | 0.000 / 0.12 |
| bright_decay_med | 0.000 / 0.41 (1 NaN flips) | 0.094 / 1.25 | 0.020 / 1.17 |
| log_inharmB_med | 0.000 / 0.33 | 0.093 / 3.44 | 0.000 / 0.15 |
| bass_body_no_fund | 0.002 / 0.17 | 0.088 / 0.60 (7 NaN flips) | 0.000 / 0.07 (2 NaN flips) |
| tristim2_med | 0.000 / 0.17 | 0.071 / 0.66 | 0.000 / 0.12 |
| even_odd_db_med | 0.000 / 0.18 | 0.066 / 0.50 | 0.000 / 0.14 |
| bass_h2_over_h1 | 0.003 / 0.09 | 0.062 / 0.31 (7 NaN flips) | 0.000 / 0.03 (2 NaN flips) |
| centroid_var_iqr | 0.000 / 0.20 | 0.062 / 0.56 | 0.000 / 0.27 |
| sustain_db_1s_med | 0.000 / 0.32 (1 NaN flips) | 0.061 / 0.52 (7 NaN flips) | 0.000 / 0.14 |
| upper_h2_over_h1 | 0.003 / 0.15 (2 NaN flips) | 0.055 / 0.30 (8 NaN flips) | 0.000 / 0.10 (4 NaN flips) |
| high_partial_frac_med | 0.000 / 0.16 | 0.046 / 1.39 | 0.000 / 0.14 |
| centroid_h_med | 0.000 / 0.10 | 0.046 / 0.68 | 0.003 / 0.27 |
| bass_fund_strength | 0.000 / 0.13 | 0.044 / 0.41 (7 NaN flips) | 0.000 / 0.03 (2 NaN flips) |
| log_t60_med | 0.000 / 0.13 | 0.043 / 0.43 | 0.000 / 0.14 |

By group (median of the per-feature medians): A medians x0.5 0.000, x0.1 0.054, pcm16 0.000; A IQRs x0.5 0.000, x0.1 0.030, pcm16 0.000; B x0.5 0.002, x0.1 0.058, pcm16 0.000; C x0.5 0.000, x0.1 0.028, pcm16 0.000.

**P(synthetic) under the unchanged T2 fold models**: median |dP|, decision flips at 0.5, and mean dP (flips) per stratum:

| features | variant | median |dP| | flips | real: CCMusic note: mean dP (flips) | real: performance: mean dP (flips) | synth: Otey note: mean dP (flips) | synth: SoundFont note: mean dP (flips) | synth: performance render: mean dP (flips) |
|---|---|---|---|---|---|---|---|---|
| all47 | x0.5 | 0.000 | 7/200 | -0.001 (3) | +0.001 (0) | -0.006 (1) | +0.015 (1) | -0.027 (2) |
| all47 | x0.1 | 0.001 | 11/200 | +0.036 (6) | +0.005 (0) | +0.036 (3) | +0.020 (1) | +0.007 (1) |
| all47 | pcm16 | 0.000 | 6/200 | -0.004 (0) | +0.000 (0) | +0.051 (4) | -0.009 (1) | +0.011 (1) |
| med16 | x0.5 | 0.000 | 5/200 | +0.002 (2) | +0.008 (0) | -0.000 (1) | +0.012 (1) | +0.004 (1) |
| med16 | x0.1 | 0.007 | 15/200 | +0.023 (7) | +0.006 (0) | +0.043 (3) | +0.032 (1) | +0.033 (4) |
| med16 | pcm16 | 0.000 | 10/200 | -0.010 (2) | -0.001 (0) | -0.014 (5) | -0.009 (2) | -0.009 (1) |

**Isolated-regime (and performance) AUCs inside the sample**, unchanged T2 fold models. "both v" = both classes transformed; "real orig / synth v" = only the synthetic files transformed (a level mismatch between the classes; pcm16 brings the synthetic files to the real corpus' convention).

| features | comparison | orig | both x0.5 | both x0.1 | both pcm16 | real orig / synth x0.5 | real orig / synth x0.1 | real orig / synth pcm16 |
|---|---|---|---|---|---|---|---|---|
| all47 | CCMusic vs SoundFont notes | 0.971 | 0.968 | 0.947 | 0.971 | 0.973 | 0.972 | 0.969 |
| all47 | CCMusic vs Otey notes | 0.863 | 0.863 | 0.832 | 0.881 | 0.862 | 0.876 | 0.882 |
| all47 | real performances vs performance renders | 0.953 | 0.950 | 0.946 | 0.954 | 0.943 | 0.951 | 0.955 |
| med16 | CCMusic vs SoundFont notes | 0.889 | 0.894 | 0.880 | 0.881 | 0.894 | 0.901 | 0.876 |
| med16 | CCMusic vs Otey notes | 0.713 | 0.707 | 0.721 | 0.707 | 0.709 | 0.739 | 0.692 |
| med16 | real performances vs performance renders | 0.818 | 0.796 | 0.787 | 0.824 | 0.817 | 0.815 | 0.819 |

**Why the features change with level (code references).** The per-note descriptors are computed on a
peak-normalised note (`calib/features.py` l.303-304 `x_n = x_t / peak`) with relative thresholds (`_trim_onset`
l.37-50, -50/-60 dB re the note's own peak; envelope dB re peak l.100-101; T60 fit on 0..-40 dB re peak
l.113-117), so they are exactly invariant to a pure gain on a FIXED window (fixed-window test above). The level
dependence enters before that, in the segmentation, which works on un-normalised magnitudes through `log1p`:

* `detune/segment.py` l.73 `m = np.log1p(np.abs(rfft(seg)))` -- the spectral-flux onset detector. log1p(g*|X|) is
  ~linear for |X| << 1 and ~log for |X| >> 1, so a gain changes the relative flux of quiet and loud partials;
  after `flux / flux.max()` the fixed thresholds `height = max(0.08, local + 0.06)` and `prominence=0.04`
  (l.86-88) then admit a different set of onsets.  This decides single-note vs performance mode
  (`n_strikes`, l.354-358) and which note windows exist (l.360-398).
* `detune/segment.py` l.171 `logmag = np.log1p(mag)` and `_harmonic_salience` (l.98-126, mean of log1p peaks over
  6 harmonics) -- the pitch salience; the relative salience of candidates and the chord-tone cut
  `s >= 0.55 * top_sal` (l.297) change with level, so the number and pitch of segments per note changes
  (and with it every IQR feature and the register-trend residuals).
* `quality/pipeline.py` l.139 (`_partial_stats`) `present = prof > -55.0` and `calib/features.py` l.229 `np.max(e) < -55.0`:
  absolute dB thresholds on the un-normalised FFT magnitude of the peak-normalised note (|X| ~ nfft/4 x amplitude,
  so the threshold sits ~110 dB below the note peak).  Invariant to a pure gain, but after 16-bit requantisation
  the quantisation floor (fixed re full scale) lands above it for quiet notes / late frames -- one reason the
  pcm16 variant differs from the float gains.
* The detune analysis (`detune/analyze.py`, `unison.py`, `tuning.py`) uses relative thresholds (0.12 x max,
  0.03 x max band energy) but runs on the same level-dependent segments.

## 9. Format-specific detectors

Each recording format gets its own detector, trained only on that format, so that every comparison in a row of the paper's Table 7 comes from one training population. XGBoost (train_realness.py parameters, n_jobs=1), protocol-v2 fixed folds, stratified grouped bootstrap (B = 2000, seed 0).

### 9.1 Performance-only detector

Training population (T2): 2576 real performances vs 120 algorithmic performance renders (11 engines: 10 SoundFonts + Otey; each render shares the CV group of its engine). Every other performance-format source is scored zero-shot (mean of the 5 fold models): sampled_digital_performance 128, modelled_digital 21, m1 30, sound_module 2, digital_other 1. T1: those sources are also trained on, labelled synthetic (out-of-fold, brand-grouped folds). Cells: AUC [95% CI, digital pianos grouped by brand]; det = share of the synthetic files with P > 0.5; P = mean P(synthetic) of the synthetic files; FA = real performances with P > 0.5.

| id | real performances vs | files/groups | all47 T2 | med16 T2 | all47 T1 | med16 T1 |
|---|---|---|---|---|---|---|
| P1 | algo_performance | 120/11 | 0.949 [0.904, 0.982]; det 45.0%; P 0.47 | 0.762 [0.655, 0.860]; det 16.7%; P 0.20 | 0.949 [0.920, 0.972]; det 58.3%; P 0.59 | 0.731 [0.637, 0.825]; det 28.3%; P 0.31 |
| P1a | ... vs SoundFont performance renders (sf2_perf) | 100/10 | 0.947 [0.896, 0.987]; det 47.0%; P 0.49 | 0.763 [0.643, 0.881]; det 20.0%; P 0.23 | 0.946 [0.914, 0.975]; det 57.0%; P 0.58 | 0.740 [0.633, 0.843]; det 32.0%; P 0.33 |
| P1b | ... vs Otey performance renders (otey_perf) | 20/1 | 0.963 [0.959, 0.967]; det 35.0%; P 0.33 | 0.755 [0.745, 0.766]; det 0.0%; P 0.07 | 0.964 [0.959, 0.968]; det 65.0%; P 0.64 | 0.687 [0.674, 0.700]; det 10.0%; P 0.19 |
| P2 | sampled digital pianos (old demos + digital_v2) | 128/7 | 0.889 [0.849, 0.917]; det 5.5%; P 0.11 | 0.623 [0.601, 0.654]; det 0.0%; P 0.03 | 0.927 [0.910, 0.944]; det 45.3%; P 0.42 | 0.720 [0.673, 0.798]; det 14.1%; P 0.22 |
| P2a | ... old demos only (SC-55 module excluded) | 9/4 | 0.917 [0.844, 0.965]; det 0.0%; P 0.16 | 0.671 [0.579, 0.799]; det 0.0%; P 0.07 | 0.918 [0.880, 0.966]; det 44.4%; P 0.34 | 0.726 [0.615, 0.957]; det 33.3%; P 0.35 |
| P2d | ... digital_v2 only | 119/7 | 0.887 [0.847, 0.915]; det 5.9%; P 0.10 | 0.619 [0.601, 0.646]; det 0.0%; P 0.02 | 0.928 [0.911, 0.945]; det 45.4%; P 0.42 | 0.719 [0.674, 0.798]; det 12.6%; P 0.21 |
| P2d-line | ... digital_v2, line capture | 12/6 | 0.945 [0.898, 0.975]; det 8.3%; P 0.12 | 0.654 [0.549, 0.744]; det 0.0%; P 0.02 | 0.956 [0.906, 0.984]; det 58.3%; P 0.54 | 0.745 [0.702, 0.846]; det 16.7%; P 0.22 |
| P2d-mic | ... digital_v2, microphone capture | 2/2 | 0.980 [0.965, 0.992]; det 0.0%; P 0.14 | 0.836 [0.818, 0.855]; det 0.0%; P 0.03 | 0.980 [0.962, 0.996]; det 100.0%; P 0.71 | 0.735 [0.583, 0.883]; det 0.0%; P 0.17 |
| P2d-unk | ... digital_v2, capture unknown | 105/7 | 0.879 [0.837, 0.908]; det 5.7%; P 0.10 | 0.611 [0.581, 0.645]; det 0.0%; P 0.02 | 0.924 [0.905, 0.944]; det 42.9%; P 0.41 | 0.716 [0.668, 0.795]; det 12.4%; P 0.21 |
| P2d-lt | ... digital_v2, long_tails flag | 10/6 | 0.857 [0.813, 0.916]; det 0.0%; P 0.08 | 0.503 [0.273, 0.624]; det 0.0%; P 0.02 | 0.937 [0.923, 0.966]; det 40.0%; P 0.38 | 0.744 [0.680, 0.827]; det 0.0%; P 0.16 |
| P2d-nolt | ... digital_v2, no long_tails flag | 109/7 | 0.890 [0.847, 0.922]; det 6.4%; P 0.10 | 0.630 [0.612, 0.655]; det 0.0%; P 0.02 | 0.927 [0.908, 0.946]; det 45.9%; P 0.43 | 0.717 [0.668, 0.799]; det 13.8%; P 0.22 |
| P3 | modelled digital pianos (old GP-6/9M + digital_v2 Roland) | 21/1 | 0.856 [0.849, 0.863]; det 4.8%; P 0.09 | 0.630 [0.617, 0.643]; det 0.0%; P 0.02 | 0.841 [0.836, 0.846]; det 57.1%; P 0.50 | 0.778 [0.769, 0.787]; det 33.3%; P 0.34 |
| P3d | ... digital_v2 modelled only | 19/1 | 0.860 [0.853, 0.868]; det 5.3%; P 0.09 | 0.638 [0.626, 0.651]; det 0.0%; P 0.02 | 0.847 [0.842, 0.852]; det 57.9%; P 0.50 | 0.789 [0.780, 0.798]; det 36.8%; P 0.37 |
| P4 | Korg M1 | 30/1 | 0.902 [0.895, 0.909]; det 20.0%; P 0.21 | 0.844 [0.836, 0.853]; det 13.3%; P 0.18 | 0.809 [0.801, 0.819]; det 30.0%; P 0.28 | 0.659 [0.647, 0.671]; det 16.7%; P 0.20 |
| P5 | digital_other (Roland GP-3, hybrid / unclear) | 1/1 | 0.942 [0.933, 0.951]; det 0.0%; P 0.02 | 0.851 [0.838, 0.864]; det 0.0%; P 0.03 | 0.927 [0.917, 0.937]; det 0.0%; P 0.16 | 0.893 [0.881, 0.905]; det 0.0%; P 0.34 |
| P6 | Roland SC-55 sound module (not a digital piano) | 2/1 | 0.997 [0.996, 0.999]; det 50.0%; P 0.68 | 0.794 [0.782, 0.807]; det 0.0%; P 0.04 | 0.994 [0.992, 0.996]; det 100.0%; P 0.90 | 0.904 [0.895, 0.914]; det 50.0%; P 0.41 |

Real-performance false alarms (P > 0.5): all47 T2 0.4%, all47 T1 3.4%, med16 T2 1.5%, med16 T1 6.7%.

### 9.2 Feature-group ablation, performance-only detector

| features | T2: vs sampled digital (P2) | T2: vs algorithmic perf (P1) | T2: vs modelled (P3) | T1: P2 | T1: P1 | T1: P3 |
|---|---|---|---|---|---|---|
| all 47 | 0.889 [0.849, 0.917]; det 5.5%; P 0.11 | 0.949 [0.904, 0.982]; det 45.0%; P 0.47 | 0.856 [0.849, 0.863]; det 4.8%; P 0.09 | 0.927 [0.910, 0.944]; det 45.3%; P 0.42 | 0.949 [0.920, 0.972]; det 58.3%; P 0.59 | 0.841 [0.836, 0.846]; det 57.1%; P 0.50 |
| only A medians (16) | 0.623 [0.601, 0.654]; det 0.0%; P 0.03 | 0.762 [0.655, 0.860]; det 16.7%; P 0.20 | 0.630 [0.617, 0.643]; det 0.0%; P 0.02 | 0.720 [0.673, 0.798]; det 14.1%; P 0.22 | 0.731 [0.637, 0.825]; det 28.3%; P 0.31 | 0.778 [0.769, 0.787]; det 33.3%; P 0.34 |
| only A IQRs (16) | 0.721 [0.676, 0.757]; det 3.9%; P 0.07 | 0.829 [0.738, 0.902]; det 25.8%; P 0.30 | 0.760 [0.750, 0.771]; det 0.0%; P 0.07 | 0.747 [0.709, 0.811]; det 21.1%; P 0.26 | 0.804 [0.705, 0.882]; det 44.2%; P 0.44 | 0.772 [0.763, 0.781]; det 14.3%; P 0.29 |
| only B (8) | 0.720 [0.676, 0.758]; det 3.9%; P 0.09 | 0.725 [0.609, 0.833]; det 19.2%; P 0.22 | 0.713 [0.703, 0.724]; det 9.5%; P 0.12 | 0.727 [0.680, 0.805]; det 24.2%; P 0.30 | 0.753 [0.662, 0.837]; det 35.0%; P 0.37 | 0.746 [0.736, 0.756]; det 33.3%; P 0.36 |
| only C (7) | 0.798 [0.776, 0.817]; det 12.5%; P 0.24 | 0.732 [0.642, 0.802]; det 24.2%; P 0.28 | 0.718 [0.707, 0.729]; det 9.5%; P 0.18 | 0.718 [0.685, 0.775]; det 32.0%; P 0.36 | 0.751 [0.658, 0.823]; det 41.7%; P 0.43 | 0.704 [0.694, 0.714]; det 33.3%; P 0.39 |

### 9.3 Isolated-only detector (I-only)

Training population (T2): 668 CCMusic notes (6 brand groups) vs 882 single-note renders (10 SoundFonts + Otey); library notes (999 files, 4 libraries) scored zero-shot (mean of the 5 fold models). T1: all 999 library note files added to training, labelled synthetic; they are scored out-of-fold (each library is one group). Cells as 9.1; FA = CCMusic notes with P > 0.5.

| CCMusic notes vs | files/groups | all47 T2 | med16 T2 | all47 T1 | med16 T1 |
|---|---|---|---|---|---|
| SoundFont single notes | 762/10 | 0.929 [0.856, 0.975]; det 87.0%; P 0.84 | 0.849 [0.744, 0.931]; det 75.6%; P 0.72 | 0.833 [0.736, 0.932]; det 90.8%; P 0.85 | 0.717 [0.590, 0.861]; det 81.6%; P 0.74 |
| Otey single notes | 120/1 | 0.779 [0.640, 0.891]; det 38.3%; P 0.41 | 0.606 [0.478, 0.726]; det 24.2%; P 0.29 | 0.575 [0.433, 0.737]; det 41.7%; P 0.45 | 0.437 [0.307, 0.625]; det 34.2%; P 0.40 |
| algorithmic (both) | 882/11 | 0.908 [0.820, 0.970]; det 80.4%; P 0.78 | 0.816 [0.695, 0.919]; det 68.6%; P 0.66 | 0.798 [0.668, 0.917]; det 84.1%; P 0.80 | 0.679 [0.525, 0.832]; det 75.2%; P 0.70 |
| library notes, 4 libraries, mf layer | 224/4 | 0.551 [0.384, 0.723]; det 11.2%; P 0.16 | 0.476 [0.316, 0.666]; det 12.1%; P 0.18 | 0.435 [0.259, 0.640]; det 25.9%; P 0.31 | 0.351 [0.201, 0.551]; det 27.2%; P 0.31 |
| library notes, 4 libraries, all layers | 999/4 | 0.600 [0.403, 0.763]; det 13.1%; P 0.19 | 0.560 [0.360, 0.715]; det 22.7%; P 0.27 | 0.487 [0.313, 0.667]; det 34.7%; P 0.38 | 0.424 [0.252, 0.604]; det 37.0%; P 0.39 |
| Salamander, mf layer | 30/1 | 0.665 [0.505, 0.802]; det 16.7%; P 0.20 | 0.609 [0.482, 0.732]; det 26.7%; P 0.30 | 0.520 [0.389, 0.680]; det 36.7%; P 0.42 | 0.397 [0.288, 0.555]; det 33.3%; P 0.37 |
| Salamander, all layers | 480/1 | 0.675 [0.516, 0.811]; det 14.4%; P 0.21 | 0.644 [0.519, 0.762]; det 32.3%; P 0.35 | 0.538 [0.403, 0.698]; det 41.5%; P 0.43 | 0.457 [0.336, 0.629]; det 41.9%; P 0.43 |
| Iowa, mf layer | 86/1 | 0.431 [0.293, 0.567]; det 0.0%; P 0.04 | 0.331 [0.230, 0.439]; det 0.0%; P 0.05 | 0.302 [0.169, 0.488]; det 3.5%; P 0.13 | 0.256 [0.144, 0.439]; det 11.6%; P 0.19 |
| Iowa, all layers | 261/1 | 0.426 [0.298, 0.553]; det 3.4%; P 0.06 | 0.360 [0.255, 0.469]; det 1.9%; P 0.07 | 0.342 [0.218, 0.513]; det 13.4%; P 0.20 | 0.304 [0.188, 0.485]; det 17.2%; P 0.24 |
| K18, mf layer | 30/1 | 0.642 [0.510, 0.760]; det 20.0%; P 0.28 | 0.613 [0.491, 0.728]; det 26.7%; P 0.31 | 0.571 [0.448, 0.713]; det 50.0%; P 0.49 | 0.528 [0.404, 0.697]; det 56.7%; P 0.53 |
| K18, all layers | 180/1 | 0.649 [0.511, 0.769]; det 21.7%; P 0.27 | 0.638 [0.515, 0.755]; det 31.1%; P 0.35 | 0.557 [0.434, 0.704]; det 46.7%; P 0.47 | 0.534 [0.416, 0.697]; det 55.6%; P 0.53 |
| FreePats, mf layer | 78/1 | 0.604 [0.467, 0.725]; det 17.9%; P 0.23 | 0.532 [0.410, 0.650]; det 14.1%; P 0.21 | 0.497 [0.372, 0.651]; det 37.2%; P 0.40 | 0.370 [0.248, 0.548]; det 30.8%; P 0.33 |
| FreePats, all layers | 78/1 | 0.604 [0.467, 0.725]; det 17.9%; P 0.23 | 0.532 [0.410, 0.650]; det 14.1%; P 0.21 | 0.497 [0.372, 0.651]; det 37.2%; P 0.40 | 0.370 [0.248, 0.548]; det 30.8%; P 0.33 |

CCMusic false alarms (P > 0.5): all47 T2 15.4%, all47 T1 38.2%, med16 T2 22.6%, med16 T1 47.0%.

### 9.4 Corpus-level detector with vs without the dry performance renders (T2)

Same corpus, folds and parameters as section 2; the only change is whether the 120 SoundFont / Otey performance renders are in the data (training and evaluation). P2-P6 sources are never trained on (T2).

| id | all47: without -> with renders | med16: without -> with renders |
|---|---|---|
| P2 | 0.652 [0.608, 0.681] (det 0.0%) -> 0.887 [0.845, 0.917] (det 3.1%) | 0.647 [0.588, 0.685] (det 0.0%) -> 0.676 [0.621, 0.709] (det 0.0%) |
| P2a | 0.699 [0.631, 0.738] (det 0.0%) -> 0.918 [0.869, 0.948] (det 0.0%) | 0.791 [0.704, 0.909] (det 0.0%) -> 0.800 [0.762, 0.901] (det 0.0%) |
| P2d | 0.649 [0.599, 0.680] (det 0.0%) -> 0.885 [0.843, 0.915] (det 3.4%) | 0.636 [0.574, 0.674] (det 0.0%) -> 0.666 [0.607, 0.702] (det 0.0%) |
| P3 | 0.724 [0.712, 0.737] (det 0.0%) -> 0.865 [0.857, 0.873] (det 0.0%) | 0.663 [0.651, 0.676] (det 0.0%) -> 0.685 [0.674, 0.697] (det 4.8%) |
| P4 | 0.855 [0.845, 0.864] (det 0.0%) -> 0.891 [0.884, 0.898] (det 6.7%) | 0.929 [0.924, 0.934] (det 10.0%) -> 0.889 [0.884, 0.893] (det 43.3%) |
| P6 | 0.734 [0.724, 0.743] (det 0.0%) -> 1.000 [0.999, 1.000] (det 100.0%) | 0.714 [0.702, 0.727] (det 0.0%) -> 0.861 [0.849, 0.873] (det 0.0%) |
| I1a | 0.922 [0.854, 0.970] (det 87.1%) -> 0.934 [0.870, 0.976] (det 88.3%) | 0.820 [0.736, 0.894] (det 71.1%) -> 0.834 [0.748, 0.907] (det 75.3%) |
| I1b | 0.804 [0.678, 0.902] (det 53.3%) -> 0.807 [0.696, 0.900] (det 50.0%) | 0.678 [0.599, 0.765] (det 36.7%) -> 0.684 [0.598, 0.774] (det 33.3%) |

### 9.5 Instrument-level tests (one value per instrument)

Median P(synthetic) of each instrument's notes under the I-only detector (9.3). Cells: P(an instrument of the second set has a higher median P than one of the first set) and exact two-sided Mann-Whitney p (n = 8 vs 4 gives a minimum p of 0.004; 6 vs 4: 0.0095; 8 vs 10: 4.6e-5).

| detector | 8 pianos vs 4 libraries (all) | 8 pianos vs 4 libraries (mf) | 6 CCMusic groups vs 4 libraries | 8 pianos vs 10 SoundFonts | 4 libraries vs 10 SoundFonts | Otey (1 engine) median P |
|---|---|---|---|---|---|---|
| all47 T2 | 0.69 (p=0.368) | 0.69 (p=0.368) | 0.71 (p=0.352) | 1.00 (p=4.57e-05) | 1.00 (p=0.002) | 0.274 |
| all47 T1 | 0.59 (p=0.683) | 0.59 (p=0.683) | 0.50 (p=1) | 0.90 (p=0.00306) | 1.00 (p=0.002) | 0.419 |
| med16 T2 | 0.66 (p=0.461) | 0.59 (p=0.683) | 0.62 (p=0.61) | 0.97 (p=0.000183) | 1.00 (p=0.002) | 0.162 |
| med16 T1 | 0.41 (p=0.683) | 0.41 (p=0.683) | 0.29 (p=0.352) | 0.84 (p=0.0155) | 1.00 (p=0.002) | 0.352 |

Per-instrument medians (all47 T2): pianos Hsinghai 0.020, Kawai 0.006, Kawai-G 0.177, PearlRiver 0.550, Steinway 0.001, Steinway-T 0.004, Yamaha 0.002, YoungChang 0.010; libraries FreePats 0.061, Iowa 0.007, K18 0.119, Salamander 0.110; SoundFonts casioca110pianofm 0.971, casiopx860concertgrandpiano 0.955, cp801644100 0.985, ctk230soundfont 0.993, ctk533piano100casio 0.999, jv1080nicepianom 0.968, u20piano 0.909, yamahapsr400500 0.975, yamahapsrf50 0.846, yamahas90esnaturalgrand 0.938.

Phaseogram drift irregularity (`figures/drift_notes_all.csv`, samples per cycle; one median per instrument). Cells for the tests: P(second set > first set) and exact two-sided Mann-Whitney p.

| note selection | 8 pianos: median of instrument medians (n) | 4 libraries | 10 SoundFonts | Otey | pianos vs libraries | pianos vs SoundFonts | libraries vs SoundFonts |
|---|---|---|---|---|---|---|---|
| MIDI 48-76, pitch check passed (figure-analysis selection) | 1.42 (8) | 0.63 (4) | 0.30 (10) | 0.054 | 0.12 (p=0.0485) | 0.00 (p=4.57e-05) | 0.12 (p=0.036) |
| all analysable notes | 4.17 (8) | 1.14 (4) | 0.30 (10) | 0.120 | 0.16 (p=0.0727) | 0.00 (p=4.57e-05) | 0.10 (p=0.024) |

### 9.6 Zero IQR values and undefined Group B / C values, by source

| source | files | zero IQR values / all IQR values | files with >= 1 zero IQR | files with all 16 IQR = 0 | undefined B values / all | files with >= 1 undefined B | undefined C values / all | files with >= 1 undefined C |
|---|---|---|---|---|---|---|---|---|
| real performances | 2576 | 3012 / 41216 | 1929 | 0 | 1375 / 20608 | 749 | 9 / 18032 | 3 |
|   manifest | 432 | 532 / 6912 | 338 | 0 | 250 / 3456 | 138 | 0 / 3024 | 0 |
|   youtube_extra | 2144 | 2480 / 34304 | 1591 | 0 | 1125 / 17152 | 611 | 9 / 15008 | 3 |
| SoundFont performance renders | 100 | 125 / 1600 | 81 | 0 | 16 / 800 | 10 | 0 / 700 | 0 |
| Otey performance renders | 20 | 20 / 320 | 12 | 0 | 6 / 160 | 4 | 0 / 140 | 0 |
| sampled digital pianos | 128 | 129 / 2048 | 82 | 0 | 8 / 1024 | 5 | 0 / 896 | 0 |
| modelled digital pianos | 21 | 21 / 336 | 14 | 0 | 5 / 168 | 3 | 0 / 147 | 0 |
| Korg M1 | 30 | 30 / 480 | 21 | 0 | 5 / 240 | 3 | 0 / 210 | 0 |
| CCMusic notes (isolated, ref.) | 668 | 3583 / 10688 | 502 | 182 | 2399 / 5344 | 584 | 815 / 4676 | 241 |
| SoundFont single notes (ref.) | 762 | 9517 / 12192 | 715 | 555 | 3643 / 6096 | 753 | 2481 / 5334 | 627 |
| Otey single notes (ref.) | 120 | 1459 / 1920 | 107 | 85 | 591 / 960 | 120 | 370 / 840 | 110 |

Real performances, undefined values per feature: Group B bass_fund_strength 27, bass_h2_over_h1 27, bass_body_no_fund 27, upper_fund_strength 39, upper_h2_over_h1 39, upper_body_no_fund 40, deepbass_nonharm_ratio 458, deepbass_phantom_idx 718; Group C detune_final 0, unison_index 1, global_index 2, med_spread_cents 1, frac_split 1, global_scatter 2, ref_offset_abs 2. Zero IQR values per feature: log_inharmB_iqr 1, bright_decay_iqr 3, n_partials_above_iqr 874, sustain_db_1s_iqr 313, partial_persist_iqr 1821.
