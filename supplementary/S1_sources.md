# Supplementary material S1: sources and processing

This document describes every source of Table 1: where it comes from, how the files were obtained and processed, how the real class was screened, and how files were grouped for cross-validation. It also covers the processing of the controls (Sec. 4.2) and the enlarged recorded class (Table 5). All counts were recomputed from the corpus table of the final run (`Stattia/revision/analysis/out_v3/folds.csv`) and from the audio files. Paths are relative to the repository root. "Not recorded" means that the fact cannot be established from the code or data.

Machine-readable tables in this folder:

| File | Rows | Content |
|---|---|---|
| `S1_real_performances.csv` | 2471 | real performances used: file, hand-selected/collected, video id, CV group, duration, format |
| `S1_excluded_real.csv` | 105 | real performances removed by the screen: video id, category, reason |
| `S1_digital_pianos.csv` | 149 | digital-piano demonstrations: video id, brand, model, sampled/modelled, stated sound engine, capture |
| `S1_libraries.csv` | 4 | the four sample libraries: version, licence, format, notes used |
| `soundfont_instruments.csv` | 10 | the ten SoundFonts: original hardware, sample and loop facts, render statistics |
| `S1_maestro_excerpts.csv` | 120 | MAESTRO excerpts behind the 100 SoundFont and 20 Otey performances |

## S1.1 Overview

| Source (Table 1) | Class | Format | CV groups | Files | Median duration | Section |
|---|---|---|---|---|---|---|
| CCMusic, 8 pianos | real | isolated | 6 | 668 | 4.3 s | S1.2 |
| YouTube, acoustic pianos | real | performance | 2275 | 2471 (426 hand-selected + 2045 collected) | 58.5 / 89.0 s (hand-selected / collected) | S1.3 |
| SoundFont instruments (10) | algorithmic | isolated / perf. | 10 | 678 / 100 | 1.8 / 60 s | S1.4, S1.6 |
| Otey physical model | algorithmic | isolated / perf. | 1 | 120 / 20 | 3.0 / 60 s | S1.5, S1.6 |
| Libraries: Salamander, Iowa, K18, FreePats | sample-based | isolated | 4 | 999 notes (224 mf) | 7.1–30.9 s (per-library medians) | S1.7 |
| Digital pianos, sampled (53 models, 7 brands) | sample-based | performance | 7 | 128 | 80 s | S1.8 |
| Digital pianos, modelled (Roland, 9 models) | separate | performance | 1 | 21 | 98 s | S1.8 |
| Korg M1 | separate | performance | 1 | 30 | 27 s | S1.9 |

The real-performance durations in Table 1 (59–89 s) are the medians of the 426 hand-selected and 2045 collected files after the screen (58.5 and 89.0 s).

**Common analysis convention.** Every file is read with `soundfile` and reduced to mono by the mean of its channels, at its native sample rate (`detune/segment.load_mono`). The analysis does not resample. All 47 features come from one code path, `quality.pipeline.note_raw_descriptors` + `aggregate_file`, with the fixed pitch-detrending curves `quality/cache/register_trends_cal.json`. A file enters the corpus only if at least 30 of the 47 features are defined.

## S1.2 Real isolated notes: CCMusic

- **Origin.** CCMusic Piano Sound Quality dataset, Hugging Face `ccmusic-database/pianos`, configuration `8_class`, all three splits (train 531, validation 68, test 69). Licence CC BY-NC-ND 4.0. References: Zhou et al. (2024), CSMT 2023; Zhou et al. (2025), TISMIR 8(1).
- **Instruments.** Eight pianos: Hsinghai, Kawai, Kawai-G, Pearl River, Steinway, Steinway-T, Yamaha and Young Chang. Seven pianos have 88 notes each; Steinway has 52. In total 7 × 88 + 52 = 668 files, one struck note per file.
- **Processing.** The audio was decoded from the dataset's `.arrow` files and written as WAV. The export script is not recorded. Measured format: 668 files, 44.1 kHz, mono, PCM 16 bit. Duration 1.1–17.9 s, median 4.3 s. Recording room, microphones and year: not recorded (see the dataset papers).
- **Groups.** One CV group per brand, so Kawai/Kawai-G and Steinway/Steinway-T share a group: 6 groups (`quality/grouping.py`). The enlarged recorded class (S1.12) uses one group per piano (8 groups).

## S1.3 Real performances: YouTube recordings of acoustic pianos

All files are YouTube audio. The audio is not redistributed; `S1_real_performances.csv` gives the video id of each file used. Several files can come from one video, either as separate segments or as several pianos in one video.

**Hand-selected subset (432 before the screen, 426 after).** The authors selected these recordings by hand. The list, URLs and annotations are in `audio/manifest.csv` (rows with `source = acoustic`); manifest rows marked digital, model or electroacoustic are not part of the real class.
- Download: yt-dlp, best audio, extracted to WAV at 44.1 kHz (`audio/download.ps1`).
- Processing: ffmpeg `silenceremove` at −45 dB at both ends, then gain to a peak of −1 dBFS, 44.1 kHz PCM 16 bit (`audio/process.ps1`). Some videos were cut by hand into segments or parts a/b/c by timestamp (`audio/process_seg.ps1` and the `build*.ps1` scripts; the manifest `note` column).
- Measured format after the screen: 406 files at 44.1 kHz stereo, 4 at 44.1 kHz mono, 16 at 32 kHz mono, all PCM 16 bit. The processing of the 32-kHz files is not recorded. Duration 4.8–482 s, median 58.5 s.

**Collected subset (2144 before the screen, 2045 after).** Collected automatically by YouTube search.
- `quality/collect_queue.py` searched each brand in the English Wikipedia list of piano manufacturers (folder `audio/brands_extra`). It used the queries "[brand] upright piano", "grand piano", "acoustic piano", "piano demo", "piano sound", "upright", "grand" and "piano".
- `quality/collect_yamaha.py` and the same code for other makers searched by model (folders `yamaha_extra`, `seiler_extra`, `hoffmann_extra`, `perzina_extra`, `samick_extra`).
- A candidate was kept only if:
  - the brand is named in the title;
  - the title has no comparison keyword and no other brand;
  - the video lasts 8–1200 s;
  - its id is not already in the corpus;
  - a PANNs CNN14 audio tagger (Kong et al. 2020) reports a piano probability ≥ 0.08 that exceeds every other instrument (`quality/pianoness.py`).
- Download: yt-dlp format 251/140/best audio, WAV at 44.1 kHz.
- Speech removal (`quality/despeech.py`): PANNs on 1.0-s windows with a 0.5-s hop. A window is dropped if speech ≥ 0.35 or piano < speech, with a 2.0-s guard on both sides of every speech window. Kept spans closer than 0.4 s are merged, spans shorter than 0.6 s are dropped, and the rest are joined with 10-ms fades. The result is peak-normalized to −1 dBFS and written as 44.1 kHz mono PCM 16 bit. The kept fraction of each file is in the CSV; none of the files used fell back to the unprocessed original. The timestamps of the kept spans are not stored.
- Measured format: 2045 files, 44.1 kHz mono PCM 16 bit, 3.5–858 s, median 89.0 s.

**Screen of the real class (105 files removed).** All 2576 files (432 hand-selected + 2144 collected) were screened. Method and counts: `Stattia/revision/audit/REAL_CLASS_SCREEN.md`; full table: `Stattia/revision/audit/real_class_exclusions.csv`.
- All titles were read. YouTube metadata (title, channel, description, tags, upload date) was fetched for 1079 files: all hand-selected videos and the high-risk part of the collected ones. For the other 1497 collected files the decision rests on the title.
- Duplicates were found with three checks: md5 of every WAV, the same video id, and a landmark audio fingerprint verified by waveform alignment. A pair counts as a duplicate when at least 50 % of both files, or at least 80 % of the shorter one, carry matching audio. The copy kept is the hand-selected one, otherwise the longer or containing one, otherwise the earliest upload.
- A file was excluded only with a quoted statement that the sound is digital, a virtual instrument, a MIDI rendering, another instrument or an ensemble, or when it was a duplicate. Player and reproducing pianos count as acoustic and were kept. Files marked "review" (189) were kept in the primary corpus.

| Category | Files removed |
|---|---|
| duplicate (second copy of shared audio) | 35 |
| virtual instrument (Pianoteq, Ravenscroft 275, IK SampleTank, Garritan) | 20 |
| not solo piano (voice, clarinet, flute, cello, violin, horn, saxophone, drums, orchestra) | 18 |
| digital piano | 13 |
| not a piano (organ, harmonium, reed organ) | 12 |
| Synthesia / MIDI visualisation | 5 |
| electric piano / electric celesta | 2 |
| **total** (6 hand-selected, 99 collected) | **105** |

`S1_excluded_real.csv` gives the video id, category and a short reason for each file. For duplicates it also gives the kept copy. Reasons are paraphrased, and names of performers and uploaders are omitted.

## S1.4 SoundFont instruments, isolated notes

- **Instruments.** Ten SoundFont 2 files reproducing keyboards, synthesizers and digital pianos of 1978–2017. The table `soundfont_instruments.csv` gives for each file:
  - the original hardware;
  - the name, credit and date embedded in the SF2 INFO chunk;
  - the preset rendered;
  - the number of samples reached by the render grid, their sample rates and loop mode;
  - the share of renders whose sample is transposed or looped.

  The columns `Samples`, `Looped in SF2` and `Median sample length (s)` describe the whole SF2 file. The hardware years are approximate (general knowledge). The file `YAMAHA_PSR400-500.sf2` contains, according to its embedded comment, the voices of a Yamaha PSS-51. Download URLs and licences: not recorded, except that `CP80 16_44100.sf2` states "CC".
- **Rendering** (`quality/render_soundfonts.py`):
  - Engine: TinySoundFont through the Python package `tinysoundfont` (version 0.3.7 confirmed bit-identical to the stored renders), 44.1 kHz.
  - Preset: bank 0, program 0, one synthesizer instance per SoundFont, notes rendered one after another.
  - Grid: MIDI notes 21–96 in steps of three semitones (A0–C7, 26 notes) at velocities 64, 100 and 127.
  - Each note is held 1.4 s, then 0.4 s is rendered after note-off, giving a 1.8-s file.
  - The stereo output is averaged to mono, peak-normalized to 0.9 and written as 32-bit float WAV.
  - Outputs with peak below 10⁻³ (notes outside the SoundFont's key range) were skipped. This leaves 762 renders (CTK-533: 63, U-20: 75, others: 78). The renders are dry: no reverb, no channel simulation.
- **Velocity copies.** In SoundFonts with a single velocity layer, peak normalization makes the three velocities of a note identical. Renders whose 47-feature vectors are identical (5 decimals) within one instrument were reduced to the first in file order. This removed 84 renders and leaves **678** (`Stattia/revision/analysis/v3_lib.py`, `build_corpus_v3`, step A3; per-instrument counts in `soundfont_instruments.csv`).
- **Groups.** One CV group per SoundFont (10). The performances of a SoundFont (S1.6) share its group.

## S1.5 Otey physical model, isolated notes

- **Model.** C. Otey (2007), "A Physical Model of the Piano for Sound Synthesis", Stanford Physics 210 coursework. The model has digital-waveguide strings, a hysteretic hammer, a bridge junction and a feedback-delay-network soundboard.
  - The rendering executable is the project's command-line build of this C++ code, with the tunable parameters added in this project.
  - The version used is the one of the original submission. It is rebuilt from the sources of repository commit `98cefecf` with MSVC 2022, and reproduces all 120 stored renders bit-exactly (`Stattia/revision/data_v2/scripts/build_otey_v0.py`).
  - Licence of the original 2007 code: not recorded.
- **Parameters** (`otey/best_knobs_B2.json` via `otey/render_full.py`):
  - knobs: hp 0.1245, B_scale 1.5086, K_scale 1.4616, alpha_scale 1.5923, c1 0.05, c3 1.6396, mass_scale 1.8293, c1b 38.83, c3b 60.07;
  - p_off 0, gain 100, mix 1;
  - hammer velocity v₀ = 0.8 + 4.5·vel/127 m/s.
- **Grid.** 30 pitches (B, C, E and G in octaves 0–8: B0–B7, C1–C8, E1–E7, G1–G7) × velocities 25, 50, 75, 127 (v₀ = 1.69, 2.57, 3.46, 5.30 m/s) = **120** files.
- **Rendering.** Each render lasts 3.0 s with the key held and no damper, pedal or note-off. Output: 44.1 kHz mono, 32-bit float, not normalized. Folder `otey/out_full`. One CV group.

## S1.6 Algorithmic performances: SoundFont (100) and Otey (20)

**MIDI source.** MAESTRO v3.0.0, MIDI files only (Hawthorne et al. 2019), licence CC BY-NC-SA 4.0. The 1276 MIDI files were used; no MAESTRO audio. The excerpt plan is deterministic and described in `Stattia/revision/data_v2/scripts/perf_common.py` and `make_excerpt_plan.py`:
- Only performances of at least 120 s were used, visited in a seeded random order (master seed 20261001). At most one excerpt was taken per composition (canonical title), so no piece is shared between instruments.
- For slot k, the start time s is drawn from U(10 s, duration − 70 s) with seed 20261001 + 1000 + k. The excerpt starts at the first note onset ≥ s and lasts exactly 60.0 s. A draw is rejected if the window holds fewer than 60 onsets (up to 5 draws per performance).
- Slots 0–99 go to the ten SoundFonts, 10 each, by a seeded permutation. Slots 100–119 go to Otey.
- Result: 120 excerpts from 120 different MIDI files and compositions, MAESTRO years 2004–2018, splits train 86 / validation 17 / test 17. File, composer, title, split and excerpt times: `S1_maestro_excerpts.csv`.
- Sustain pedal: CC64 ≥ 64 counts as down. CC66, CC67 and half-pedalling are ignored. Notes with an onset before the window are not rendered. The pedal state at the window start is taken from the last CC64 event before it.

**SoundFont performances** (`render_sf2_performances.py`):
- The synthesizer, preset and output format are those of S1.4: TinySoundFont 0.3.7, 44.1 kHz, bank 0 / program 0, mono mean, peak 0.9, float WAV.
- All MIDI channels are routed to channel 0. Program changes and other controllers are ignored.
- TinySoundFont ignores CC64, so the pedal is emulated in the event layer. While the pedal is down, a key release sends no note-off; the note-off is sent at pedal-up unless the key is down again.
- The renders are dry. 10 excerpts per SoundFont, 60.0 s each.

**Otey performances** (`render_otey_performances.py`):
- The model build, knobs and velocity map are those of S1.5. Each performance is the sum of per-note renders placed at their onsets. This sum is exact because the soundboard network is linear and note-independent, and the model has no coupling between notes. Sympathetic "ghost strings" are not used.
- Each note is damped by the model's damper (T60 = 0.18 s) at its effective release: key-up, or the next pedal-up if the pedal is down. The note is rendered until release + 0.5 s or the excerpt end.
- Output: 60.0 s, 44.1 kHz mono float, not normalized (gain 100 as in S1.5). 20 files, all in the Otey CV group.

## S1.7 Sample libraries (sample-based, isolated)

The libraries were read from their original distributed sample files with `soundfile`, without a sampler and without the SFZ patches. Channels were averaged and the native rate kept. Details: `S1_libraries.csv`.

| Library | Instrument | Licence | Format (measured) | Notes used | mf layer | mf notes |
|---|---|---|---|---|---|---|
| Salamander Grand Piano V3 (A. Holm) | Yamaha C5 grand | CC BY 3.0 | Ogg Vorbis, 48 kHz stereo | 480 (30 pitches in minor thirds × 16 layers) | v8 of 16 | 30 |
| University of Iowa MIS Piano | Steinway & Sons Model B, recorded Nov. 2001 | free for any use | AIFF PCM 16 bit, 44.1 kHz stereo | 261 (88 pitches × pp/mf/ff; A0, B♭0 ff only) | mf | 86 |
| K18 Upright Piano (R. Fiasco) | upright, maker not stated | CC BY 4.0 | WAV PCM 24 bit, 48 kHz stereo | 180 (30 pitches × pp, p, mf, f, ff, sf) | mf | 30 |
| FreePats Honky-tonk Piano "Piano FB", 2020-04-01 | Francis Bacon player piano | CC0 1.0 | WAV PCM 16 bit, 44.1 kHz stereo | 78 (one layer) | single layer | 78 |
| **total** | | | | **999** | | **224** |

- Release, pedal and harmonic samples were not used. Only files matching the note-name patterns of `LIB_SPECS` in `Stattia/revision/analysis/protocol_v2.py` were taken.
- All 999 note files gave valid features.
- Median note-file durations: FreePats 7.1 s, Salamander 14.8 s, K18 19.8 s, Iowa 30.9 s. The Iowa files include silence around each note.
- Each library is one CV group. Library notes are never used in training. Each library is assigned to one fold and scored by that fold's model.

## S1.8 Digital pianos (sampled: sample-based; modelled: separate)

All are YouTube demonstrations of a digital piano's acoustic-piano sound. They come from two collections: 11 demos of the original submission and a collection made for this revision. Per file: `S1_digital_pianos.csv`.

**Revision collection** (`Stattia/revision/data_v2/scripts/dpv2_*.py`, manifest `Stattia/revision/data_v2/digital_v2_manifest.csv`):
1. YouTube search per model with yt-dlp. Titles are filtered: the model must be named; comparisons, other brands, software instruments, non-acoustic patches and tutorials are rejected; ids already in the corpus are rejected.
2. Download: yt-dlp format 251/140/best audio, extracted to WAV without resampling (typically 48 kHz stereo).
3. PANNs CNN14 on 1-s windows:
   - a window is kept if piano > 0.10, speech < 0.10, other sounds < 0.35 and piano > other, with a ±2-s guard around rejected windows;
   - an acoustic-patch rule: Piano above Electric piano, and organ, synthesizer, strings and similar tags below 0.2;
   - for videos with chapters, only acoustic-piano or repertoire chapters are used.
4. One excerpt of 60–180 s that maximizes kept − 2 × rejected windows is cut sample-exactly from the download.
   - In multi-sound demos the excerpt is restricted to the first 300 s, where the main acoustic patch is presented (flag `patch_inferred_first300s`).
   - The excerpt start and end in the video are in the CSV.
5. The excerpt is cleaned with `quality/clean_synth_yt.py`: the same window rule is applied, kept windows are joined with 6-ms fades, and the output is 44.1 kHz mono 32-bit float.

**Original-submission demos** (`audio/SynthPianoSamples/yt`): the same cleaner was applied to the whole downloaded video. Excerpt boundaries and capture: not recorded.

**Sound-generation method.** For each model it was taken from the manufacturer's description, quoted with its source URL in the CSV (`engine_name_as_stated`, `engine_source_url`). "sample" means sampled, "modelling" means modelled. Model years marked "general knowledge" are approximate.

**Capture.** The capture column reads `line` when the video description states direct or line recording, `mic` when it states microphones, and `unknown` otherwise.
- Sampled demos: 12 line, 2 mic, 105 unknown, 9 not recorded (original-submission demos).
- Modelled demos: 1 line, 18 unknown, 2 not recorded.

The flags `long_tails` and `onboard_reverb_stated` mark recordings with long reverberant tails or with reverb stated in the description.

**Excluded during collection.** Candidate videos rejected during acquisition (`excluded = 1` in the manifest) and one demo to which the uploader had added external convolution reverb were not used.

**Counts.**
- Sampled: 128 files.
  - Brands: Yamaha 31, Kawai 31, Korg 18, Casio 17, Dexibell 12, Nord 10, Kurzweil 9.
  - Models: 53. The Korg B2 occurs in both collections.
  - Median duration 80 s.
- Modelled: 21 files from Roland, 9 models: FP-90, FP-90X, HP704, LX706, LX708, V-Piano, V-Piano Grand, GP-6, GP-9M. Median duration 98 s.

**Groups.** One CV group per brand (7 sampled groups; the modelled Roland files form one group). These files are never used in training.

## S1.9 Korg M1

- **Source.** 30 piano programs of a Korg M1 (1988, ROM-PCM synthesizer), all cut from one YouTube demonstration (video id `E5_QAlCf-GQ`).
- **Cutting.** Program boundaries were found from the program-name display (scene changes). The start and end times (0–1015.9 s in total) are in `audio/m1_patches/_index.csv`. The cutting script is not recorded.
- **Programs.** K01-01 M1HousePiano, K01-71 Res.Piano, M01-01 Piano 16', M01-21 MagicPiano, M01-41 Piano 8', M02-01 Piano2 16', M03-02 SimplPiano, M03-32 PianoWith?, M03-73 ToyPiano, M04-33 PianoTalk, M04-39 PhasePiano, M05-01 Soft Grand, M07-00 Piano2 Low, M07-01 Piano2High, M07-02 Piano1 Low, M07-03 Pianos 1, M07-04 Piano1High, M07-05 Pianos 2, M08-38 Stickpiano, M09-03 Toy-Piano, M10-21 Piano Soft, M10-22 Piano, M10-23 PianoBrite, M10-29 Obligatory, M14-11 DreamPiano, M14-12 Piatoron, M15-17 Vox Piano, M17-40 70's Piano, M18-16 AutumnPf, M18-49 Piano.
- **Format** (measured): 44.1 kHz stereo PCM 16 bit, 15.1–76.8 s, median 26.8 s.
- **Groups.** One CV group, never used in training.

## S1.10 Cross-validation groups and folds

**Groups** are instruments:
- CCMusic: brand (6).
- SoundFont: one per SoundFont file; the isolated notes and the performances share it (10).
- Otey: 1.
- Libraries: 4.
- Digital pianos: one per brand.
- Korg M1: 1.

**Real performances** (`Stattia/revision/analysis/v3_lib.py`, `_uf_groups`). A group is a connected component of three relations:
1. the original group key (`quality/grouping.py`, `group_key`):
   - hand-selected files: brand + model token parsed from the file name, so the same model across pieces is one group;
   - collected files: one group per file;
2. the YouTube video id, so all files cut from one video share a group;
3. the screen's `group_id`: `VID_<id>` for multi-piano videos and `DUPnn` for kept files sharing audio.

The result is **2275 groups**: 232 in the hand-selected and 2043 in the collected subset, 2211 of them with a single file. The largest component (63 files) joins comparison videos of many pianos. The channel or uploader was not used for grouping: files of one dealer in different videos are separate groups. Group ids in the CSV (`RP0001`…) are anonymous labels for these components.

**Folds** (`Stattia/revision/analysis/pv2_lib.py`, `assign_folds`). The greedy GroupKFold rule is applied with five folds:
- groups are sorted by name, permuted with seed s, and assigned largest first to the currently lightest fold;
- groups of the original corpus are assigned first, then groups that occur only in the revision data;
- seeds 0–9 give the ten fold assignments over which medians are reported.

Sample-based and separate sources are assigned to folds by the same rule, never trained on, and scored by the model of their own fold.

## S1.11 Controls (isolated notes; Table 6)

Code: `Stattia/revision/analysis/v3_controls.py`, `v3_lib.py` (`cut`, `c3`), `v3_render.py`, `v3_sf2.py`. All isolated sources (CCMusic, SoundFont, Otey, library notes) are transformed in the same way. Features are re-extracted after every transform.

**(i) 1.2 s.**
- The onset is the first 1-ms frame within 50 dB of the loudest frame. The cut keeps 10 ms before the onset and 1.2 s after it, followed by a 0.1-s raised-cosine fade.
- The fade starts at 1.2 s, before the SoundFont note-off at 1.4 s.

**(ii) 4 s.**
- SoundFont notes were re-rendered with the key held 5.0 s (then 0.4 s release), with the same code otherwise. The same code at 1.4 s reproduces the stored renders.
- Otey notes were re-rendered for 5.0 s with the same build, knobs and velocity map. The same call at 3.0 s reproduces the stored files.
- Every note is cut 4.0 s after onset with a 0.1-s fade.
- Real and library notes are kept only if they still sound at 4.0 s (20-ms frame level ≥ −60 dB re the loudest frame): CCMusic 380 of 668; libraries 874 of 999 (Iowa 261/261, K18 120/180, Salamander 443/480, FreePats 50/78).
- Synthetic notes that are already silent at 4.0 s are kept: SoundFont 304 of 678, Otey 43 of 120.

**(iii) Room.**
- Each note is convolved with one measured room impulse response from the MIT Acoustical Reverberation Scene Statistics Survey (Traer and McDermott 2016). The archive `Audio.zip` from mcdermottlab.mit.edu/Reverb/IR_Survey.html holds 270 WAV files at 32 kHz (the survey page lists 271).
- The response is drawn uniformly per file, resampled to the note's rate and normalized to unit peak. The convolution is truncated to the note's length and rescaled to the original peak.
- White Gaussian noise is added with RMS 40 dB below the RMS of the first 1.0 s after onset of the reverberant note.
- The response choice and the noise come from `numpy.random.default_rng(crc32(relative path))`, so every file has its own fixed seed.

**(iv) Combinations.** (iii) is applied to the full note, then (i) or (ii).

**SoundFont split.** For every SoundFont note, the SF2 preset data (`v3_sf2.py`) gives:
- the sample played, its root key and transposition;
- the loop mode and the time at which the loop starts.

A note counts as "not transposed, not looped" if |transposition| < 0.5 semitone and no loop is reached within the analysed window (1.8, 1.3 or 4.1 s). Per-instrument shares: `soundfont_instruments.csv`.

**Loop index.** Computed on the 5-s re-renders (window 0.5–4.0 s after onset). Per-instrument shares above 0.9999: `soundfont_instruments.csv`.

## S1.12 Enlarged recorded class (13 instruments; Table 5)

Code: `Stattia/revision/analysis/v4/run_v4.py`, `v4_lib.py`, `get_maps_isol.py`. The recorded class (label 0) contains 13 instruments, one CV group each:

| Instrument | Notes, all layers | Notes, mf variant |
|---|---|---|
| CCMusic, 8 pianos | 668 | 668 |
| Salamander | 480 | 30 |
| Iowa | 261 | 86 |
| K18 | 180 | 30 |
| FreePats | 78 | 78 |
| MAPS Yamaha Disklavier (ENSTDkCl 120 + ENSTDkAm 132) | 252 | 77 (dynamic M) |

**MAPS.** "MAPS – MIDI Aligned Piano Sounds", v0.4 (Emiya, Bertin, Badeau), Zenodo doi:10.5281/zenodo.18160555, licence CC BY-NC-SA 2.0 FR.
- Only the two subsets recorded from a real Yamaha Disklavier upright were used: ENSTDkCl (close microphones) and ENSTDkAm (ambient microphones). Of these, only the isolated normal notes (`ISOL/NO`) with the sustain pedal off (`S0`), all dynamics P/M/F, were taken.
- The files were fetched by HTTP range requests from the two zip archives. The software-piano subsets were not downloaded.
- Measured format: 44.1 kHz stereo PCM 16 bit, median 4.1 s.
- Both rooms are the same instrument and form one group.

**Synthetic class.** The 678 SoundFont notes and 120 Otey notes of S1.4–S1.5.

**Evaluation.** Grouped CV over 24 groups (8 CCMusic pianos, 4 libraries, the Disklavier, 10 SoundFonts, Otey) with the fold rule of S1.10, seeds 0–9.

**Room augmentation ("+ room").**
- Every note (both classes) has two channel copies. Each copy uses a random MIT survey response and white noise at an SNR drawn from U(30, 50) dB, relative to the RMS of the first second after onset. The seed is crc32("v4aug<k>:" + path).
- In training, each note is replaced with probability 0.5 by one of its copies (RNG seed 1000 + fold seed).
- "+ room in testing": every test note is replaced by one random copy.
- The features of a copy could not be extracted for 2 notes (first copy) and 3 notes (second copy); the unaltered note was used.

## S1.13 Facts that could not be established

- Download URLs and licences of nine of the ten SoundFont files. CP80 states only "CC", without a version.
- The CCMusic export script; recording room, microphones and year of CCMusic.
- The processing of the 16 hand-selected YouTube files stored at 32 kHz mono.
- Timestamps of the spans kept by speech removal in the collected YouTube files.
- Excerpt boundaries and capture (line or microphone) of the 11 original-submission digital-piano demos.
- The cutting script of the Korg M1 programs.
- The licence of Otey's original 2007 code.
- The versions of the Iowa and K18 libraries; the microphones of the Iowa recordings.
- The exact years of the original SoundFont hardware (approximate values from general knowledge in `soundfont_instruments.csv`).
- Capture of 105 of the 128 sampled and 18 of the 21 modelled demos: the descriptions do not state it.
