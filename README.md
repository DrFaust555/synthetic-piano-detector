# Synthetic-Piano Detector

Code and data for the paper

> V. Rovinskyi and O. Yevchuk, "A Semi-Deterministic Detector of Synthetic Pianos,"
> submitted to the *Journal of the Audio Engineering Society*.

The detector tells recordings of acoustic pianos from synthetic ones. It has two branches: 27 summary features for
an isolated note or 47 for a performance (timbre, decay, tuning and bass behavior), and a convolutional network that reads the pitch-synchronous
phaseogram of each note. The two branches are fused into one probability, P(synthetic). The paper evaluates it on
isolated notes and on performances, with cross-validation grouped by instrument.

## Layout

The repository is organized by what is computed.

| Folder | Content |
|---|---|
| `detector/` | One audio file to features: note segmentation (`segment.py`), the 47 summary features (`pipeline.py`, `calib/`, `detune/`, `spectral.py`, `waveform.py`), the phaseogram (`phaseogram.py`) and the phaseogram CNN (`phaseogram_cnn.py`) |
| `protocol/` | Evaluation protocol: corpus assembly and cleaning (`corpus_base.py`, `corpus.py`), instrument groups and folds, XGBoost, AUC, grouped bootstrap (`common.py`), the isolated-note universe of the CNN (`cnn.py`) |
| `corpus/` | How the data were collected, cleaned and rendered (needs the audio and the original sources) |
| `training/` | Scripts that produce the shipped feature tables and out-of-fold predictions (need the audio) |
| `experiments/` | One folder per question of the paper. Every script reads `data/` and writes `results/<folder>/` |
| `quality_axes/` | The quality tool of Sec. 3.8 |
| `figures/` | Figs. 2 and 3 |
| `data/` | Feature tables, metadata tables and out-of-fold predictions |
| `supplementary/S1/` | Supplementary material S1: sources, processing, Otey parameters, digital-piano sound engines, YouTube identifiers |

`PAPER_MAP.md` lists every number of the paper with the script and the result file that produce it.

## Data

The audio is not redistributed. `data/` holds what the analyses need:

* `data/corpus/`, `data/features/`, `data/performances/`: per-file feature tables (numeric descriptors that cannot
  reconstruct the audio) and the manifests of the performance sources;
* `data/protocol/`: the real-class manifest, the screen of the real class, and the instrument groups and folds;
* `data/predictions/`: out-of-fold scores of the three detectors (feature detector, phaseogram CNN, fusion);
* `data/listening/`: answer key and anonymized answers of the listening test.

The detection and listening analyses in `experiments/` run from these tables alone.
Scripts in `training/`, `corpus/`, `table6_quality.py` and some figures need
the audio: set the environment variable `SPD_AUDIO` to the folder that holds it. File names in the tables are
relative to that folder (for example `audio/ccmusic_pianos/...`, `audio/pianoteq/...`).

## Running

Python 3.10. Install the packages of `requirements.txt`, then run any script from the repository root, e.g.

```
python experiments/01_detection/table5_notes.py
```

The outputs are written to `results/`. The feature detector (XGBoost, `n_jobs=1`, fixed seeds) is deterministic;
the shipped CNN and fusion scores are those of the paper.

The isolated-note method is fixed: one window from the strike, the nominal pitch refined on the spectrum,
and the 27-feature model input. Performances use onset segmentation, up to 15 phaseograms and 47 features.
Performance model files are named `cnn_perf_f*.npz` and `fusion_perf_f*.npz`.
The IQR ablation, room test, eGeMAPS baseline and recording-format controls implement the comparisons reported in the paper.
The two Table 6 quality calculations are documented in `PAPER_MAP.md`.

To check the analyses, run `python verification/run_scripts.py` (add `--audio-root PATH --audio`
when all audio is available). Full network retraining and source acquisition are separate from these checks.
`SPD_RESULTS` can point to an independent output folder, for example for the small training smoke test
`python verification/smoke_training.py`.

## License

MIT (see `LICENSE`) for the code, the feature tables and the predictions. Third-party datasets and sample
libraries keep their own licenses (see S1 and the references of the paper).
