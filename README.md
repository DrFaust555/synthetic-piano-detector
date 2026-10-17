# Synthetic-Piano Detector

This repository accompanies the revised manuscript

> **A Semi-Deterministic Detector of Synthetic Pianos: What a Single Note Reveals**
> Viktor Rovinskyi and Olga Yevchuk
> *Journal of the Audio Engineering Society* (JAES), under revision.

## Supplementary material of the revised paper

| | Content | Location |
|---|---|---|
| S1 | Sources and processing, video identifiers, excluded real files, SoundFont instruments | `supplementary/S1_*.md`, `supplementary/S1_*.csv`, `supplementary/soundfont_instruments.csv` |
| S2 | CNN training details | `supplementary/S2_cnn.md` |
| S3 | Listening tests: procedure, stimuli, anonymized answers | `supplementary/S3/`; clips, answer keys, test pages and scorers in `listening/` (`test1/`, `pilot/`, `test2/`) |
| S4 | All result tables with intervals and per-source values | `supplementary/S4_*.md`, `supplementary/S4_*.csv` |

Revised evaluation protocol (scripts, fold assignments, out-of-fold scores, result tables):

| Folder | Content |
|---|---|
| `revision/analysis/` | protocol v3 (`final_v3.py`, `run_v3.py`, `v3_*.py`) on the v2 corpus code (`pv2_lib.py`, `protocol_v2.py`, ...); results in `out_v3/` (`FINAL_NUMBERS.md`, `folds.csv`, `fold_seed_sweep.csv`, `controls_scores.csv`, `checksums.txt`, ...) |
| `revision/analysis/v4/` | enlarged recorded class (13 instruments incl. MAPS Disklavier) |
| `revision/analysis/out/` | protocol v2 tables referenced by S4 (`oof_scores.csv`, `folds.csv`, `comparisons.csv`) |
| `revision/cnn_iso/` | phaseogram CNN and fusion on isolated notes |
| `revision/egemaps/` | eGeMAPS baseline |
| `figures/` | Figs. 1 and 2 of the revised paper and the scripts that draw them |

## Reproducing the revised protocol

The scripts were run from a project root in which this repository's `revision/` was `Stattia/revision/`
and the listening kits were `Stattia/listening_kit/` (`listening/test1`), `Stattia/listening_kit_v2/`
(`listening/pilot`) and `Stattia/listening_kit_v3/` (`listening/test2`); paths of the form
`Stattia/revision/X` in the scripts and supplementary files refer to `revision/X` here. To rerun,
recreate that layout around the top-level code of this repository (`quality/`, `detune/`, `calib/`, `*.py`)
and place the audio under `audio/` as listed in S1. Then:

```
pip install -r requirements.txt xgboost==3.2.0 pandas
python Stattia/revision/analysis/final_v3.py    # = revision/analysis/final_v3.py: Tables and Sec. 5 numbers -> out_v3/
python Stattia/revision/analysis/v4/run_v4.py   # = revision/analysis/v4/run_v4.py: enlarged recorded class
python Stattia/revision/cnn_iso/run_all.py      # = revision/cnn_iso/run_all.py: CNN / fusion on isolated notes
python Stattia/revision/egemaps/run_egemaps.py  # = revision/egemaps/run_egemaps.py: eGeMAPS baseline (needs opensmile)
```

Environment of the reported runs: Python 3.10, numpy 1.26.4, pandas 1.5.2, xgboost 3.2.0 (n_jobs=1,
random_state=0), torch 2.0.1 (CPU). The repository holds no YouTube audio; recordings must be obtained
from the sources listed in S1. `revision/**/checksums.txt` give the md5 of the output tables
(in `out_v3/c5_loop_index.csv` the local path prefix was removed for publication and its checksum updated).
`revision/audit/real_class_exclusions.csv` is the screen of the real class read by `v3_lib.py` (file, video id, decision,
category, group id; the title and evidence columns are omitted for publication); `out_v3/excluded_real_v3.csv` lists the 105 removed files.

## Note on the released models

The trained models released here in July 2026 (`quality/cache/realness_xgb.json`, `quality/cache/realness_net.pt`)
are later versions than the models of the original submission. The results of the revised paper are fully
defined by the scripts, fold assignments and out-of-fold scores in `revision/`; they do not use those files.

## Original submission (July 2026 release): paper section → code

Section numbers below refer to the originally submitted manuscript.

| Paper section | Code |
|---|---|
| §3.1 47 summary descriptors | `quality/extract_features.py`, `calib/features.py`, `quality/spectral.py`, `quality/waveform.py`, `quality/attack.py`, `quality/modulation.py` |
| §3.1 Group C tuning descriptors | `detune/` (segmentation `segment.py`, unison analysis `unison.py`, global tuning `tuning.py`, entry point `analyze.py`) |
| §3.1 pitch detrending curves | `quality/cache/register_trends*.json` |
| §3.2 phaseogram branch + fusion | `quality/phasefeat.py`, `quality/realness_net.py`, `quality/train_realness.py`, `quality/critic.py` |
| §3.3 anti-leakage evaluation (grouped CV, OOF) | `quality/critic_eval.py`, `quality/grouping.py` |
| §3.4 channel invariance | `quality/channel_robustness.py`, `quality/channel_aug.py`, `quality/effects.py` |
| §3.5 synthetic categories / corpus prep | `quality/render_soundfonts.py`, `quality/build_negatives.py`, cleaning: `quality/clean_synth_yt.py` |
| §3.5 Otey physical model (released) | `piano.py`, `note_engine.py`, `hammer.py`, `string_model.py`, `soundboard*.py`, `params.py`, `calib/` |
| §4 results reproduction | `quality/critic_eval.py` over `data/features/*.csv` |
| §4.3 eGeMAPS external baseline | `quality/opensmile_baseline.py` |
| §4.4 listening test | `listening/test1/` (form, instructions, answer key, clips), anonymised answers in `supplementary/S3/S3_test1_answers.csv`, `quality/build_listening_kit.py`, `quality/score_listening.py` |
| §5 quality axes (timbre / tuning / liveness) | `quality/axes.py`, `quality/timbre_quality.py`, `quality/aversiveness.py`, `quality/extract_liveness.py`, `detune/` |

## Released data (base tier)

`data/features/*.csv` — the extracted per-file feature vectors used by the
paper's models. These are numeric descriptors that cannot reconstruct audio;
every modelling result in the paper reproduces from these numbers alone.

Trained artifacts in `quality/cache/`: `realness_xgb.json` (summary-feature
detector), `realness_net.pt` (phaseogram CNN), `register_trends*.json`
(pitch-detrend curves), calibration JSONs.

## Audio (not redistributed)

The audio corpus is not redistributed. The citable core is reproducible from
public sources: the CCMusic piano sound-quality dataset, SoundFont renders
(`quality/render_soundfonts.py`), the Otey physical model (this repository),
and four freely downloadable sample libraries (Salamander Grand, University
of Iowa piano, K18 upright, FreePats old piano — see the paper's references
for URLs and licenses).

## Setup

```
pip install numpy scipy soundfile librosa scikit-learn xgboost torch
pip install opensmile   # only for the eGeMAPS baseline
```

Smoke test (renders a note with the physical model and extracts features):

```
python -c "from piano import Piano; a = Piano(48000).render_note(60, 75, 1.0, release=0.8); print(a.shape)"
```

## License

MIT (see `LICENSE`). The trained artifacts and feature CSVs are released under
the same terms. Third-party sample libraries and datasets referenced by the
paper keep their own licenses.
