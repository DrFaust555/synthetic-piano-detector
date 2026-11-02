# Supplementary material S1: sources, processing, Otey model parameters, digital-piano sound engines, YouTube identifiers

This supplement describes where every source of the paper comes from and how its files were obtained and processed. It gives the parameter values of the Otey model, the manufacturers' descriptions of the digital-piano sound engines, and the YouTube identifiers of all recordings. Paths are relative to the repository root. "Not recorded" means that the fact cannot be established from the code or the data.

Tables in this folder:

| File | Rows | Content |
|---|---|---|
| `S1_real_performances.csv` | 2471 | real performances used: file, subset, collection route, brand as parsed from the name, YouTube id, duration, format, kept fraction after speech removal |
| `S1_excluded_real.csv` | 105 | real performances removed by the screen: YouTube id, category, reason, kept copy for duplicates |
| `S1_digital_pianos.csv` | 149 | digital-piano demonstrations: YouTube id, brand, model, stated sound-generation method, engine name and the manufacturer's description (quoted, with source URL), capture, excerpt times |
| `S1_m1_pianoteq.csv` | 43 | Korg M1 programs (30) and Pianoteq segments (13): YouTube id, start and end in the video, row of the paper |
| `S1_soundfonts.csv` | 10 | the ten SoundFonts: original hardware, embedded name, credit and date, preset, samples, render counts |
| `S1_libraries.csv` | 4 | the four sample libraries: instrument, version, license, URL, format, notes used |
| `S1_maestro_excerpts.csv` | 120 | MAESTRO excerpts and the five renders of each (SoundFont or Otey, and four SFZ libraries) |


## S1.1 Overview

| Source | Class | Format | Files | Section |
|---|---|---|---|---|
| CCMusic, 8 pianos | real | isolated notes | 668 | S1.2 |
| YouTube, acoustic pianos | real | performances | 2471 (426 hand-selected + 2045 collected) | S1.3 |
| SoundFont instruments (10) | algorithmic | isolated notes / performances | 679 / 100 | S1.4, S1.7 |
| Otey physical model | algorithmic | isolated notes / performances | 120 / 20 | S1.5, S1.7 |
| Sample libraries: Salamander, Iowa, K18, FreePats | sample-based | isolated notes | 999 (224 in the mf layer) | S1.6 |
| SFZ sampled play (the same four libraries) | sample-based | performances | 480 | S1.7 |
| Digital pianos, sample-based (7 brands, 53 models) | non-acoustic | performances | 128 | S1.8 |
| Digital pianos, modeled (Roland, 9 models) | non-acoustic | performances | 21 | S1.8 |
| Korg M1 (22 piano presets + 8 non-piano patches) | non-acoustic | performances | 30 | S1.9 |
| Pianoteq 9 | non-acoustic | performances | 13 | S1.10 |

**Files as analyzed.** Every file is read with `soundfile` and reduced to mono by the mean of its channels, at its native sample rate (`detector/segment.load_mono`). No resampling is done before the analysis.

## S1.2 Real isolated notes: CCMusic

- **Origin.** CCMusic Piano Sound Quality dataset, Hugging Face `ccmusic-database/pianos`, configuration `8_class`, all three splits (train 531, validation 68, test 69). License CC BY-NC-ND 4.0. References: Zhou et al. (2024), CSMT 2023; Zhou et al. (2025), TISMIR 8(1).
- **Instruments.** Eight pianos: Hsinghai, Kawai, Kawai-G, Pearl River, Steinway, Steinway-T, Yamaha and Young Chang. Seven pianos have 88 notes each and Steinway has 52, giving 7 × 88 + 52 = 668 files with one struck note per file.
- **Processing.** The audio was decoded from the dataset's `.arrow` files and written as WAV; the export script is not recorded. Measured format: 44.1 kHz, mono, PCM 16 bit, 1.1–17.9 s, median 4.3 s. The Pearl River notes last 1.1–2.2 s (median 1.6 s). Recording room, microphones and year are not recorded here (see the dataset papers).

## S1.3 Real performances: YouTube recordings of acoustic pianos

All files are YouTube audio. The audio is not redistributed; `S1_real_performances.csv` gives the video id of each file. Several files can come from one video, either as separate segments or as several pianos in one video.

**Hand-selected subset (432 before the screen, 426 after).** The authors selected these recordings by hand. The list, URLs and annotations are in `data/protocol/manifest.csv` (rows with `source = acoustic`).
- Download: yt-dlp, best audio, extracted to WAV at 44.1 kHz (`corpus/hand_selected/download.ps1`).
- Processing: ffmpeg `silenceremove` at −45 dB at both ends, then gain to a peak of −1 dBFS, 44.1 kHz PCM 16 bit (`corpus/hand_selected/trim_normalize.ps1`). Some videos were cut by hand into segments by timestamp (`corpus/hand_selected/trim_normalize.ps1` and `concat_segments.ps1`; column `note` of the manifest).
- Measured format: 406 files at 44.1 kHz stereo, 4 at 44.1 kHz mono and 16 at 32 kHz mono, all PCM 16 bit; 4.8–482 s, median 58.5 s. The processing of the 32-kHz files is not recorded.

**Collected subset (2144 before the screen, 2045 after).** Collected automatically by YouTube search.
- `corpus/collect_brands.py` searched for each brand in the English Wikipedia list of piano manufacturers (folder `audio/brands_extra`) with the queries "[brand] upright piano", "grand piano", "acoustic piano", "piano demo", "piano sound", "upright", "grand" and "piano". `corpus/collect_models.py` and the same code for other makers searched by model (folders `yamaha_extra`, `seiler_extra`, `hoffmann_extra`, `perzina_extra`, `samick_extra`).
- A candidate was kept only if the brand is named in the title, the title has no comparison keyword and no other brand, the video lasts 8–1200 s, its id is not already in the corpus, and a PANNs CNN14 audio tagger (Kong et al. 2020) gives a piano probability of at least 0.08 that exceeds every other instrument (`corpus/pianoness.py`).
- Download: yt-dlp format 251/140/best audio, WAV at 44.1 kHz.
- Speech removal (`corpus/despeech.py`): PANNs on 1.0-s windows with a 0.5-s hop. A window is dropped if speech ≥ 0.35 or piano < speech, with a 2.0-s guard on both sides of every speech window. Kept spans closer than 0.4 s are merged, spans shorter than 0.6 s are dropped, and the rest are joined with 10-ms fades. The result is peak-normalized to −1 dBFS and written as 44.1 kHz mono PCM 16 bit. The kept fraction of each file is in the table. The timestamps of the kept spans are not stored.
- Measured format: 44.1 kHz mono PCM 16 bit, 3.5–858 s, median 89.0 s.

**Screen of the real class (105 files removed).** All 2576 files were screened (method: `corpus/real_class_screen.md`).
- All titles were read. YouTube metadata (title, channel, description, tags, upload date) was fetched for 1079 files: all hand-selected videos and the high-risk part of the collected ones. For the other 1497 collected files the decision rests on the title.
- Duplicates were found with three checks: the md5 of every WAV, the same video id, and a landmark audio fingerprint verified by waveform alignment. A pair counts as a duplicate when at least 50% of both files, or at least 80% of the shorter one, carry matching audio. The copy kept is the hand-selected one, otherwise the longer or containing one, otherwise the earliest upload.
- A file was excluded only on a quoted statement that the sound is a digital piano, a virtual instrument, a MIDI rendering, another instrument or an ensemble, or when it was a duplicate. Player and reproducing pianos count as acoustic and were kept.

| Category | Files removed |
|---|---|
| duplicate (second copy of shared audio) | 35 |
| virtual instrument (Pianoteq, Ravenscroft 275, IK SampleTank, Garritan) | 20 |
| not solo piano (voice, clarinet, flute, cello, violin, horn, saxophone, drums, orchestra) | 18 |
| digital piano | 13 |
| not a piano (organ, harmonium, reed organ) | 12 |
| Synthesia / MIDI visualization | 5 |
| electric piano / electric celesta | 2 |
| **total** (6 hand-selected, 99 collected) | **105** |

`S1_excluded_real.csv` gives the video id, category and a short paraphrased reason for each removed file; names of performers and uploaders are omitted.

**Brand.** The column `brand_from_name` is the brand parsed from the source folder or file name (`experiments/01_detection/real_brands.py`). It is used only for the per-brand results; for hand-selected files without a brand in the name it holds the second token of the name.

**Reference recording of Table 6.** The real Bösendorfer 280VC is the hand-selected file `42_bosendorfer_280vc_97.wav` (video `UrT9d-Lx3Kg`, 28.8 s), processed as above. Two further recordings of the same instrument are in the real class: `45_bosendorfer_280vc_97_b.wav` (`6V9zreejkFU`) and `51_bosendorfer_280vc_97_c.wav` (`9kPCnFx5mV0`).

## S1.4 SoundFont instruments, isolated notes

- **Instruments.** Ten SoundFont 2 files that reproduce keyboards, synthesizers and digital pianos of 1978–2017. `S1_soundfonts.csv` gives for each file the original hardware, the name, credit and date embedded in the SF2 INFO chunk, the preset rendered, the number of samples, their sample rates and loop mode, and the share of renders whose sample is transposed. The hardware years are approximate. According to its embedded comment, the file `YAMAHA_PSR400-500.sf2` contains the voices of a Yamaha PSS-51. Download URLs and licenses are not recorded, except that `CP80 16_44100.sf2` states "CC".
- **Rendering** (`corpus/render_soundfont_notes.py`):
  - engine: TinySoundFont through the Python package `tinysoundfont` 0.3.7 (bit-identical to the stored renders), 44.1 kHz;
  - preset: bank 0, program 0, one synthesizer instance per SoundFont, notes rendered one after another;
  - grid: MIDI notes 21–96 in steps of three semitones (A0–C7, 26 notes) at velocities 64, 100 and 127;
  - each note is held 1.4 s and rendered for a further 0.4 s after note-off, giving a 1.8-s file;
  - the stereo output is averaged to mono, peak-normalized to 0.9 and written as 32-bit float WAV;
  - outputs with a peak below 10⁻³ (notes outside the key range of the SoundFont) were skipped, leaving 762 renders. The renders are dry: no reverb and no channel simulation.
- **Velocity copies.** In SoundFonts with a single velocity layer, peak normalization makes the three velocities of a note identical. Renders whose feature vectors are identical within one instrument were reduced to the first, which removed 83 renders and leaves **679** (`protocol/corpus.py`, `build_corpus_v3`; per-instrument counts in `S1_soundfonts.csv`).

## S1.5 Otey physical model: parameter values

**Model.** The model has digital-waveguide strings (one, two or three per note), a hysteretic felt hammer, a bridge junction and a feedback-delay-network soundboard.

**Parameter values.** The values were found by a Nelder–Mead search that matched the render of the note B2 to a sampled piano note B2, with a loss of log-mel distance, partial-amplitude distance and pitch error. In the formulas, x = ln(f/27.5)/ln(4192/27.5) for a note of fundamental frequency f.

| Parameter | Original value (Otey 2007) | Value used |
|---|---|---|
| hammer strike position, fraction of the string length | 1/7 | 0.12448 |
| multiplier on the string inharmonicity B = π³E r_core⁴ / (4L²T) | 1 | 1.50856 |
| multiplier on the hammer stiffness K = 40 / 0.0007^p | 1 | 1.46157 |
| multiplier on the hammer hysteresis α = 10⁻⁵·x | 1 | 1.59226 |
| multiplier on the hammer mass m = 0.06 − 0.058·x^0.1 kg | 1 | 1.82930 |
| offset added to the hammer exponent p = 2 + x | 0 | 0 |
| string loss filter, coefficient c1 | 0.25 | 0.05 |
| string loss filter, coefficient c3 | 5.85 | 1.63963 |
| soundboard network decay, coefficient c1 | 20 | 38.82625 |
| soundboard network decay, coefficient c3 | 20 | 60.06590 |
| soundboard wet/dry mix | 1 | 1 |
| output gain | 100 | 100 |

During the search c1 was limited to [0.05, 1]; the search returned the lower limit. The sample rate is 44.1 kHz. The hammer velocity is v₀ = 0.8 + 4.5·vel/127 m/s for MIDI velocity vel.

**Isolated notes.** 30 pitches (B, C, E and G in octaves 0–8: B0–B7, C1–C8, E1–E7, G1–G7) × velocities 25, 50, 75 and 127 (v₀ = 1.69, 2.57, 3.46 and 5.30 m/s) = **120** files. Each render lasts 3.0 s with the key held and no damper, pedal or note-off. Output: 44.1 kHz mono, 32-bit float, not normalized (folder `otey/out_full`).

## S1.6 Sample libraries, isolated notes

The libraries were read from their original sample files with `soundfile`, without a sampler. Details are in `S1_libraries.csv`.

| Library | Instrument | License | Format (measured) | Notes used | mf layer | mf notes |
|---|---|---|---|---|---|---|
| Salamander Grand Piano V3 (A. Holm) | Yamaha C5 grand | CC BY 3.0 | Ogg Vorbis, 48 kHz stereo | 480 (30 pitches in minor thirds × 16 layers) | v8 of 16 | 30 |
| University of Iowa MIS Piano | Steinway & Sons Model B, recorded Nov. 2001 | free for any use | AIFF PCM 16 bit, 44.1 kHz stereo | 261 (88 pitches × pp/mf/ff; A0 and B♭0 ff only) | mf | 86 |
| K18 Upright Piano (R. Fiasco) | upright, maker not stated | CC BY 4.0 | WAV PCM 24 bit, 48 kHz stereo | 180 (30 pitches × pp, p, mf, f, ff, sf) | mf | 30 |
| FreePats Honky-tonk Piano "Piano FB", 2020-04-01 | Francis Bacon player piano | CC0 1.0 | WAV PCM 16 bit, 44.1 kHz stereo | 78 (one layer) | single layer | 78 |
| **total** | | | | **999** | | **224** |

- Release, pedal and harmonic samples were not used. Only files matching the note-name patterns of `LIB_SPECS` in `protocol/corpus_base.py` were taken.
- Median note-file durations: FreePats 7.1 s, Salamander 14.8 s, K18 19.8 s, Iowa 30.9 s. The Iowa files include silence around each note.

## S1.7 Performances rendered from MIDI: SoundFont (100), Otey (20) and SFZ sampled play (480)

**MIDI source.** MAESTRO v3.0.0, MIDI files only (Hawthorne et al. 2019), license CC BY-NC-SA 4.0. No MAESTRO audio was used. The excerpt plan is deterministic (`corpus/perf_common.py`, `corpus/performance_excerpt_plan.py`; plan `data/performances/perf_excerpt_plan.csv`):
- Only performances of at least 120 s were used, visited in a seeded random order (master seed 20261001). At most one excerpt was taken per composition, so no piece is shared between instruments.
- For slot k, the start time s is drawn from U(10 s, duration − 70 s) with seed 20261001 + 1000 + k. The excerpt starts at the first note onset ≥ s and lasts exactly 60.0 s. A draw is rejected if the window holds fewer than 60 onsets (up to 5 draws per performance).
- Slots 0–99 go to the ten SoundFonts, 10 each, by a seeded permutation; slots 100–119 go to Otey.
- Result: 120 excerpts from 120 different MIDI files and compositions, MAESTRO years 2004–2018. File, composer, title, split and excerpt times are in `S1_maestro_excerpts.csv`.
- Sustain pedal: CC64 ≥ 64 counts as down; CC66, CC67 and half-pedaling are ignored. Notes with an onset before the window are not rendered. The pedal state at the window start is taken from the last CC64 event before it.
- The same pedal handling is used by all three renderers. While the pedal is down, a key release sends no note-off; the note-off is sent at pedal-up unless the key is down again. No CC64 event is sent to the synthesizer.

**SoundFont performances** (`corpus/render_soundfont_performances.py`). Synthesizer, preset and output format as in S1.4: TinySoundFont 0.3.7, 44.1 kHz, bank 0 / program 0, mono mean, peak 0.9, 32-bit float WAV. All MIDI channels are routed to channel 0; program changes and other controllers are ignored. Dry renders, 10 excerpts per SoundFont, 60.0 s each.

**Otey performances.** Parameters and velocity map as in S1.5. Each performance is the sum of single-note renders placed at their onsets. The sum is exact because the model has no coupling between notes and its soundboard network is linear. Each note is damped at its effective release, with a decay of 60 dB in 0.18 s, that is, at key-up or at the next pedal-up if the pedal is down. Each note is rendered until 0.5 s after its release or until the end of the excerpt. Output: 60.0 s, 44.1 kHz mono, 32-bit float, not normalized. 20 files.

**SFZ sampled play** (`corpus/render_sfz_performances.py`). The same 120 excerpts were rendered through each of the four libraries of S1.6, giving 4 × 120 = 480 files. The files of excerpt slot k are `sfz_performances_v2/<library>/<library>_slot<k>.wav` (listed in `S1_maestro_excerpts.csv`).
- Engine: sfizz 1.2.3 (official win64 release), driven through its C API. Settings: freewheeling mode, sample quality 10 (the highest), block 1024, polyphony 256, 44.1 kHz. Note events are sent with sample-accurate offsets and the MIDI velocity of MAESTRO.
- Patches: the SFZ files distributed with Salamander (`SalamanderGrandPianoV3.sfz`), K18 (`K18-Upright-Piano.sfz`) and FreePats (`PianoFB 20200401.sfz`), unmodified. Their release samples sound at note-off. Their pedal-noise regions stay silent because no CC64 is sent.
- Iowa has no SFZ patch, so one was built without modifying any sample (`corpus/build_iowa_sfz.py`):
  - one region per recorded file at its own key, with no transposition;
  - velocity layers pp 1–42, mf 43–85, ff 86–127;
  - the sample start is set just before the attack;
  - the level follows a per-key curve through the key's own layer peaks, so that the level is continuous across layer boundaries;
  - the release time is 0.5 s.
- Output: the stereo render averaged to mono, exactly 60.0 s, peak-normalized to 0.9, 32-bit float WAV. The renders are dry: no room, no reverb and no codec.

## S1.8 Digital pianos

All files are YouTube demonstrations of the acoustic-piano sound of a digital piano. They come from 11 demos collected for the original submission and from a collection made for this revision. Per-file details are in `S1_digital_pianos.csv`.

**Revision collection** (`corpus/digital_*.py`, manifest `data/performances/digital_v2_manifest.csv`):
1. YouTube search per model with yt-dlp. Titles are filtered: the model must be named; comparisons, other brands, software instruments, non-acoustic patches and tutorials are rejected; ids already in the corpus are rejected.
2. Download with yt-dlp (format 251/140/best audio), extracted to WAV without resampling (typically 48 kHz stereo).
3. PANNs CNN14 on 1-s windows. A window is kept if piano > 0.10, speech < 0.10, other sounds < 0.35 and piano > other, with a ±2-s guard around rejected windows. The patch must be an acoustic piano: Piano above Electric piano, and organ, synthesizer, strings and similar tags below 0.2. In videos with chapters, only acoustic-piano or repertoire chapters are used.
4. One excerpt of 60–180 s that maximizes kept − 2 × rejected windows is cut sample-exactly from the download. In demos of several sounds the excerpt is restricted to the first 300 s, where the main acoustic patch is presented (flag `patch_inferred_first300s`). The start and end of the excerpt in the video are in the table.
5. The excerpt is cleaned with `corpus/clean_demos.py`: the same window rule is applied, kept windows are joined with 6-ms fades, and the output is 44.1 kHz mono 32-bit float.

**Original-submission demos** (`audio/SynthPianoSamples/yt`). The same cleaner was applied to the whole downloaded video. Excerpt boundaries and capture are not recorded.

**Sound engines.** For each model, the sound-generation method and the engine name were taken from the manufacturer's own description: product page (127 files), press release (19) or manual (3). The table quotes the description (`manufacturer_description_quoted`) with its URL (`description_source_url`). Where the manufacturer states modeling added on top of samples (for example resonance modeling), this is noted in `note_on_added_modeling`; such models remain sample-based. The class used in the paper follows the stated method: `sample` → sample-based, `modeling` → modeled. Model years marked "general knowledge" are approximate.

**Capture.** The column `capture` reads `line` when the video description states direct or line recording, `mic` when it states microphones, and `unknown` otherwise. Sample-based demos: 12 line, 2 mic, 105 unknown, 9 not recorded (original-submission demos). Modeled demos: 1 line, 18 unknown, 2 not recorded. The flags `long_tails` and `onboard_reverb_stated` mark recordings with long reverberant tails or with reverb stated in the description.

**Excluded during collection.** Candidate videos rejected during acquisition (`excluded = 1` in the manifest) were not used. One demo was also dropped because the uploader had added external convolution reverb.

**Counts.**
- Sample-based: 128 files of 53 models from 7 brands: Yamaha 31, Kawai 31, Korg 18, Casio 17, Dexibell 12, Nord 10 and Kurzweil 9. Median duration 80 s.
- Modeled: 21 files of 9 Roland models (FP-90, FP-90X, GP-6, GP-9M, HP704, LX706, LX708, V-Piano and V-Piano Grand). Median duration 98 s.

## S1.9 Korg M1

- **Source.** 30 piano programs of a Korg M1 (1988, ROM-PCM synthesizer), all from one YouTube demonstration (video `E5_QAlCf-GQ`).
- **Cutting.** Program boundaries were taken from the program-name display. The start and end times in the video (0–1015.9 s in total) are in `S1_m1_pianoteq.csv`. The stored files are 1.7–5.5 s (median 2.2 s) shorter than these spans, so leading and trailing parts were trimmed. The files are peak-normalized to about −1 dBFS. The cutting script is not recorded.
- **Two rows in the paper.** 22 programs imitate a piano ("piano presets"). The other 8 imitate no piano ("non-piano patches"): K01-71 Res.Piano, M01-21 MagicPiano, M03-73 ToyPiano, M04-33 PianoTalk, M04-39 PhasePiano, M08-38 Stickpiano, M09-03 Toy-Piano and M15-17 Vox Piano. The assignment of each program is in the column `row_in_paper`.
- **Format** (measured): 44.1 kHz stereo PCM 16 bit, 15.1–76.8 s, median 26.8 s.

## S1.10 Pianoteq 9

- **Source.** One YouTube demonstration of Pianoteq 9 (Modartt), video `sVbC5Kfr1_s`, titled "Modartt Pianoteq 9 is Amazing! – Virtual Piano Demo (No Talking)". It was published by a user, not by Modartt. The presets played and the recording path are not recorded.
- **Manufacturer's description** (https://www.modartt.com/pianoteq, accessed Oct. 30, 2026): "Using physical modelling, it captures the wonderful tone of the instruments, while offering limitless customization." and "Lightweight — no need for samples! Instant load, notes are generated in real time as you play."
- **Processing.** The downloaded track (44.1 kHz stereo PCM 16 bit, 323.6 s) was averaged to mono and scaled to a peak of −1 dBFS (`audio/pianoteq/_raw/`). It was cut into 13 consecutive segments separated by 0.2-s gaps. Each segment was peak-normalized to about −1 dBFS and stored as 44.1 kHz mono PCM 16 bit, 13.4–53.2 s, median 22.8 s. The cutting script is not recorded. The segment boundaries in the video were measured by locating each stored segment in the full track; each segment matches the track exactly up to a gain factor. The boundaries are in `S1_m1_pianoteq.csv`.
