# Synthetic-Piano Detector

Code and released data for the paper:

> **A Semi-Deterministic Detector of Synthetic Pianos**
> Viktor Rovinskyi and Olga Yevchuk
> Submitted to the *Journal of the Audio Engineering Society* (JAES).

The paper asks when a recording of a piano can be told apart from a synthetic
one, and answers it with a two-branch detector: 47 channel-robust summary
descriptors of timbre/sustain/tuning, and a phaseogram CNN that reads the
within-cycle waveform shape. Detection succeeds wherever the synthesis is
generative (algorithmic), and hits a structural limit on sample-based playback.

## Layout — paper section → code

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
| §4.4 listening test | `listening_kit/` (form, answer key, anonymised responses), `quality/build_listening_kit.py`, `quality/score_listening.py` |
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
