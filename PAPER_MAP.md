# Paper map

Every number of the paper with the script that computes it and the file it writes. Detection and listening
analyses and `figures/fig_scores.py` run from `data/` alone. Both Table 6 quality versions need audio
(`SPD_AUDIO`, see README).
Results go to `results/<folder>/`.

## Sec. 3.1, Table 1: data

| Paper | Source |
|---|---|
| Counts, groups, durations of every source | `protocol/corpus.py` (`build_corpus_v3`), `supplementary/S1/` |
| Collection, cleaning and rendering of the sources | `corpus/` (see S1 for the description of each step) |

## Sec. 3.6 and 4.1: detection accuracy and the recording channel (Table 5, Fig. 3)

| Paper | Number(s) | Script (`experiments/01_detection/`) | Result (`results/01_detection/`) |
|---|---|---|---|
| Table 5, single notes, all cells and thresholds | P_th 0.831 / 0.977 / 0.941; SoundFont 0.955 [0.90, 0.99] / 78% ...; real pianos 0.458 / 2%, 0.471 / 1%, 0.512 / 2% | `table5_notes.py` | `table5_notes.md` |
| Table 5, performances, feature detector (SoundFont, Otey, digital pianos, all trained, SFZ play; P_th 0.320) | 0.945 [0.91, 0.97] / 67% ... 0.733 [0.60, 0.85] / 10% | `performances_features.py` (setting B) | `q2_final.md` |
| Table 5, performances, feature detector: Korg M1 split, Pianoteq | 0.950 / 64%, 0.674 / 0%; 0.922 / 69% | `pianoteq_features.py` | `table3_provenance_perf.md` |
| Table 5, performances, CNN and fusion (all rows but Pianoteq and the real brands) | e.g. fusion 0.948 [0.92, 0.98] / 74% | `performances_cnn_fusion.py` | `perf_cnn_fusion_table.md` |
| Table 5, Pianoteq, CNN and fusion | 0.744 / 0%, 0.958 / 77% | `detection_details.py` | `detection_details.md` |
| Table 5, real performances (18 brands) | 0.557 / 4%, 0.541 / 7%, 0.552 / 5% | `real_brands.py` | `real_perf_brands.md` |
| Table 5, P_th of the CNN and the fusion on performances | 0.962; 0.4996 (printed 0.501) | `detection_details.py` | `detection_details.md` |
| Sec. 3.6: false alarms at a fixed P = 0.5 | 3 % to 36 % | `detection_details.py` | `detection_details.md` |
| Sec. 4.1: SoundFont and Otey performance renders together, fusion | 0.952 [0.92, 0.98] | `detection_details.py` | `detection_details.md` |
| Sec. 4.1: Otey notes detected between 3 % and 7 % false alarms | 39-58 % (47 to 69 of 120) | `detection_details.py` | `detection_details.md` |
| Fig. 3b: real performances below P = 0.0001 | half (54 %) | `detection_details.py` | `detection_details.md` |
| Sec. 4.1: Pianoteq scored by each of the five fold models | CNN 0.54-0.95, fusion 0.88-0.96 | `detection_details.py` | `detection_details.md` |
| Sec. 4.1: ten group-to-fold assignments, feature detector | no AUC moved more than 0.04 | `fold_assignments.py` | `xgb_seed_sweep.md` |
| Sec. 4.1: without the 16 IQR features, feature detector | 0.937 -> 0.928; Otey 85 % -> 55 % | `without_iqr.py` | `q2_no_iqr.md` |
| Sec. 4.1: without the 16 IQR features, fusion | 0.930 -> 0.910; Otey 85 % -> 60 % | `detection_details.py` | `detection_details.md` |
| Sec. 4.1: unseen rooms with noise, fusion | SoundFont 0.952 -> 0.975, Otey 0.870 -> 0.816 | `detection_details.py` | `detection_details.md` |
| Sec. 4.1: Korg M1 split, CNN and fusion | 0.938 / 77 %; 0.446 / 0 % | `performances_cnn_fusion.py` | `perf_cnn_fusion_table.md` |
| Sec. 4.1: SFZ play per library | medians 0.41 (Salamander), below 0.03 for the other three | `experiments/05_provenance_and_quality/table6_provenance.py` | `results/05_provenance_and_quality/` |
| Fig. 3 | | `figures/fig_scores.py fus` | `results/figures/Fig_scores_fus.{png,pdf}` |

In the feature column of the single-note block, the source rows come from the out-of-fold scores of the feature
detector (`data/predictions/notes_features_oof.csv`, produced by `training/feature_detectors.py`); the row "real
pianos" comes from the feature detector trained in the same run as the CNN (`notes_oof.csv`, column `xgb27_p`).
`notes_table.py` gives all three detectors of that one run.

## Sec. 4.2: an algorithmic / sample-based split

| Paper | Number(s) | Script (`experiments/02_sampled_vs_algorithmic/`) |
|---|---|---|
| Libraries against all real notes (fusion, mf layer) | Iowa 0.48, Salamander 0.58, K18 0.58, FreePats 0.59 | `libraries_and_brands.py` |
| Real brand groups, each against the other five (fusion) | Steinway 0.26 ... Pearl River 0.71 | `libraries_and_brands.py` |
| SoundFont notes that play an unaltered sample, feature detector | 0.961 against 0.955 | `soundfont_unaltered_samples.py` |
| Digital-piano cues and the 16 medians | unison index 8.0 vs 14.9 ...; AUC 0.56 -> 0.84, 0.57 -> 0.47 | `digital_cues.py` |
| Well-tuned real performances above the fusion threshold | 11.7 % vs 0.8 % (fifths of the integral detuning score) | `tuned_false_alarms.py` |

## Sec. 4.3: the sample-based limit and a recording-format shortcut

| Paper | Number(s) | Script (`experiments/03_sampled_limit_and_format/`) |
|---|---|---|
| Library render against the algorithmic render of the same excerpt | 89 % of 480 pairs (77 % ... 97 %); medians 0.004 / 0.977 | `sfz_vs_algorithmic_pairs.py` |
| Format shortcut | 0.9998; 0.996 against 0.943 | `format_shortcut.py` |
| eGeMAPS baseline | 0.948 against 0.943; 0.71 against 0.47 (mf layer) | `egemaps_baseline.py` (features: `corpus/egemaps_extract.py`) |

## Sec. 3.7 and 4.4: listening test

| Paper | Script (`experiments/04_listening/`) |
|---|---|
| Stimuli, levels; listeners: 65 % [58-72] / 37 % [32-42], listener AUC 0.67, 27 of 32, Wilcoxon, best 0.96, panel AUC 0.98 | `listening_test.py` (also `listening_check.py`) |
| Detector on the 30 clips: 4/5, 4/5, 1/20 at P > 0.941, AUC 0.965; pairs ranked by the panel and by the detector | `listening_test.py` (clip scores: `data/predictions/listening_clips.csv`) |

## Sec. 3.8 and 4.5: provenance and quality (Table 6)

| Paper | Script (`experiments/05_provenance_and_quality/`) |
|---|---|
| Table 6, P(synthetic): 0.03 / 0.94 / 0.41 / 0.99 / 0.97 | `table6_provenance.py` |
| Table 6, overall quality: 95 / 95 / 90 / 58 / 3 | `table6_quality.py` (needs the audio) |
| Quality regressor under grouped cross-validation, rho of about 0.54 | `quality_regressor_cv.py` |
| Tuning axis (entropy-minimizing target, score 100 e^(-d/12)) | `quality_axes/tuning/` |

## Produced from the audio (`training/`)

| Shipped table | Script |
|---|---|
| `data/features/*_iso1.csv`, `table3_perf_flux.csv` | `training/note_features.py` |
| `data/corpus/*.csv`, `data/performances/*_features.csv` | `training/corpus_features.py`, `synth_features.py`, `perf_features.py`, `sfz_perf_features.py` |
| `data/predictions/notes_features_oof.csv` (feature detector, isolated notes) | `training/feature_detectors.py` (runs from `data/`) |
| `data/predictions/notes_oof.csv` (CNN, fusion, rooms) | `training/notes_phaseograms.py`, `notes_room_features.py`, `notes_cnn.py`, `notes_cnn_predictions.py` |
| `data/predictions/perf/*.npz`, `perf_meta*.csv`, `perf_fusion_without_iqr_oof.csv` | `training/perf_phaseograms.py`, `sfz_phaseograms.py`, `perf_cnn.py` (`--drop-iqr` for the variant) |
| `data/predictions/pianoteq_*.csv` | `training/pianoteq_perf_cnn.py` |
| Fig. 2 | `figures/drift_regularity.py`, then `figures/fig_phaseogram.py` |

`data/corpus/samplelib_features.csv` supplies 32 library-take rows whose group sizes determine the fixed
fold assignment. The isolated-note detector scores individual library notes; sampled performances use
the 480 SFZ renders.

Performance CNN training uses `python training/perf_cnn.py`; the sampled digital pianos are in training
and SFZ play is only scored. The two models are `cnn_perf` and `fusion_perf`; `--drop-iqr` reproduces the
reported fusion ablation. Isolated-note training uses `cnn` and `fus27`. Its default listening splits
exclude clip pitches from each known engine's training notes, as described in Sec. 3.7.
