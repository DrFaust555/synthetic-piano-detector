# S3. Listening tests: stimuli, procedure and anonymized answers

Supplementary material to "A Semi-Deterministic Detector of Synthetic Pianos: What a Single Note Reveals" (Sec. 4.3, Sec. 5.6, Table 8).

Three tests are documented: **Test 1**, a **pilot** and **Test 2**. Files in this folder:

| File | Content |
|---|---|
| `S3_test1_stimuli.csv`, `S3_pilot_stimuli.csv`, `S3_test2_stimuli.csv` | one row per clip: class, source instrument, note, MIDI, velocity layer, source file name, measured processing facts, substitutions |
| `S3_test1_answers.csv`, `S3_pilot_answers.csv`, `S3_test2_answers.csv` | one row per listener: code, self-reported experience, submission date, presentation order, rating of every clip (1-4), free-text answer translated into English, code of the same person in the other test |

Listeners are identified only by codes (T1-01…T1-32, P-01…P-18, T2-01…T2-20). Test 1 codes follow the order of the anonymized results file; pilot and Test 2 codes follow submission time. Names and nicknames typed into the form were discarded. Experience is exactly the option the listener selected. Free-text answers were translated into English; none contained personal details.

## 1. Procedure common to all tests

- **Platform.** A static web page (one HTML file plus the WAV clips, no server-side code, no account). Answers were never sent automatically: at the end the page built a one-row CSV that the listener downloaded, copied or shared and sent to the experimenter by messenger (or e-mail).
- **Listening.** Own device; headphones and a quiet room recommended; free replay; no time limit; no training items and no feedback.
- **Task.** One question per clip with a four-point scale (Sec. 2). An answer of 3 or 4 counts as **"judged synthetic"**.
- **Audio.** 44.1 kHz, mono, 16-bit PCM WAV. Onset = first sample above 2% of the file peak; each clip starts 10 ms before the onset.

## 2. Form text (Ukrainian original and English translation)

### 2.1 Test 1

| Ukrainian (original) | English |
|---|---|
| Реальне піаніно чи синтетичне? | Real piano or synthetic? |
| Послухай кожен кліп (одна нота, ~3 с) і виріши: **жива акустична піаніно** чи **синтетичний/цифровий** звук. Тут **немає** 50/50 — суди кожен окремо. Найкраще в навушниках. | Listen to each clip (one note, ~3 s) and decide: a **live acoustic piano** or a **synthetic/digital** sound. It is **not** 50/50 — judge each one separately. Headphones are best. |
| Твій нік: (ім'я або нік) | Your nickname: (name or nickname) |
| Рівень: не музикант / аматор / музикант / настройщик/майстер | Level: not a musician / amateur / musician / tuner/technician |
| clip_01 … clip_30 (one player and one drop-down list per clip) | clip_01 … clip_30 |
| 1 — точно реальне; 2 — скоріше реальне; 3 — скоріше синтетичне; 4 — точно синтетичне | 1 — definitely real; 2 — probably real; 3 — probably synthetic; 4 — definitely synthetic |
| Завершити | Finish |

Accompanying instructions for listeners (translated): "Open the link … Headphones and quiet are best. For each of the **30 clips** (one note, ~3 s) listen and choose 1 definitely real acoustic / 2 probably real / 3 probably synthetic / 4 definitely synthetic. Press «Finish» and send me the result. **Important:** it is **not** 50/50 — judge each clip separately, trust your first impression, do not consult anyone."

### 2.2 Pilot and Test 2

The two pages are identical except for the duration and time given in the second paragraph and the volume instruction added in Test 2.

| Ukrainian (original) | English |
|---|---|
| Справжнє фортепіано чи синтетичне? | Real piano or synthetic? |
| Дякуємо, що допомагаєте! Тест займає приблизно 5 хвилин. *(Test 2: 8 хвилин)* | Thank you for helping! The test takes about 5 minutes. *(Test 2: 8 minutes)* |
| Нижче **24 кліпи**. Кожен кліп — **одна нота** фортепіано. Усі кліпи **короткі й однакової довжини** (приблизно 1,3 с): кожен наприкінці плавно затихає, тож за тривалістю звуку нічого не визначиш. *(Test 2: Усі кліпи **однакової довжини** (приблизно 4 с): …)* | Below are **24 clips**. Each clip is **one piano note**. All clips are **short and of equal length** (about 1.3 s): each fades out smoothly at the end, so the duration of the sound tells you nothing. *(Test 2: All clips are **of equal length** (about 4 s): …)* |
| Послухайте кожен кліп і виберіть: це **справжнє фортепіано** (запис акустичного інструмента) чи **синтетичне** звучання (створене електронним інструментом або комп'ютерною програмою). | Listen to each clip and choose: is it a **real piano** (a recording of an acoustic instrument) or a **synthetic** sound (made by an electronic instrument or a computer program)? |
| Кожен кліп можна слухати скільки завгодно разів. Справжніх і синтетичних кліпів не обов'язково порівну — оцінюйте кожен кліп окремо, довіряйте першому враженню й ні з ким не радьтеся. | You may play each clip as many times as you like. Real and synthetic clips are not necessarily equal in number — rate each clip separately, trust your first impression and do not consult anyone. |
| **Найкраще слухати в навушниках**, у тиші. *(Test 2 adds:* Встановіть зручну гучність на першому кліпі й далі її не змінюйте.*)* | **Headphones are best**, in a quiet place. *(Test 2 adds:* Set a comfortable volume on the first clip and do not change it afterwards.*)* |
| Ваше ім'я або нік | Your name or nickname |
| Ваш музичний досвід: Без музичної освіти / Аматор / Музична школа / Професійний музикант | Your musical experience: No musical training / Amateur / Music school / Professional musician |
| Кліп *i* з 24 — Справжнє фортепіано чи синтетичне? | Clip *i* of 24 — Real piano or synthetic? |
| 1 — точно справжнє; 2 — скоріше справжнє; 3 — скоріше синтетичне; 4 — точно синтетичне | 1 — definitely real; 2 — probably real; 3 — probably synthetic; 4 — definitely synthetic |
| На що ви звертали увагу, коли вирішували? (необов'язкове текстове поле) | What did you attend to when deciding? (optional text field) |
| Завершити | Finish |

Clips were shown as numbered cards ("Clip 1 of 24" …) with no clip identifier. Starting one clip paused any other. Every clip had to be rated before the answers could be exported (also in Test 1).

## 3. Design of each test

| | Test 1 | Pilot | Test 2 |
|---|---|---|---|
| Dates (2026) | 28 June – 2 July | 3 – 5 October | 5 – 16 October |
| Listeners | 32 | 18 | 20 |
| Self-reported experience | 17 not a musician, 14 amateur, 1 musician | 9 music school, 7 no musical training, 2 amateur | 11 music school, 7 no musical training, 1 amateur, 1 professional musician |
| Recruitment | open invitation | open invitation | listeners with musical or studio experience invited individually and asked to listen attentively; collection stopped at 20 listeners, a target set during collection |
| Clips | 30: 20 recorded (Iowa 7, K18 7, Salamander 6), 10 algorithmic (SoundFont Casio CA-110 5, Otey 5) | 24: 10 recorded (CCMusic 5 pianos, libraries 5), 14 algorithmic (8 SoundFonts, Otey 6) | 24: same design as the pilot; 17 notes identical, 7 substituted (Sec. 6) |
| Register | C3–E5 | C3–E5 (MIDI 48–76, 8 clips per third of the range) | C3–E5 |
| Clip length | 3.000 s | 1.300 s (1.2 s + 100-ms raised-cosine fade-out) | 4.150 s (4.0 s + 150-ms raised-cosine fade-out) |
| Fades | 10-ms linear fade-in and fade-out | 10-ms linear fade-in | 10-ms linear fade-in |
| Pre-roll before onset | 10 ms, clamped at file start | 10 ms, missing part padded with silence | 10 ms, missing part padded with silence |
| Level | RMS −20 dBFS; peak capped at −0.26 dBFS, which lowered the RMS of 4 clips to −20.3…−21.5 dBFS | RMS −20 dBFS (peak cap never reached) | peak −3 dBFS, no loudness equalization |
| Velocity | library mf (Salamander v8/16), SoundFont v100, Otey v127 | library mf / single layer, SoundFont v100 (key held 1.4 s), Otey v75 | as pilot; SoundFont and Otey re-rendered with the key held 5 s |
| Order | fixed, clip_01…clip_30 on one page | random per listener (recorded in the answer) | random per listener (recorded in the answer) |
| Replay | free | free | free |
| Free-text question | none | "What did you attend to when deciding?" | same |
| Judgments | 960 | 432 | 480 |

Nine listeners took part in both the pilot and Test 2 (column `same_listener_in_pilot` / `same_listener_in_test2`). Pairs were matched by the name entered in both tests (case- and space-insensitive); four pairs with differently written names were confirmed by the experimenter, and one identical name belonging to two different people was not paired. Overlap of Test 1 with the later tests was not assessed.

**Known shortcomings of Test 1** (Sec. 4.3): the SoundFont clips fall silent at 1.8 s (t60 1.4–1.75 s; digital silence afterwards; `last_nonzero_sample_s`); the 10-ms fade-in overlapped the attack of all 10 algorithmic clips and of 5 of the 7 K18 clips, whose files start less than 10 ms before the onset (`fade_in_overlaps_attack`); one SoundFont only; fixed order. **Pilot:** the 1.3-s window removed the decay; results were at chance (Sec. 5).

## 4. Analysis

- **Judged synthetic** = answer 3 or 4. Rates per class are pooled over listeners and clips.
- **95% intervals**: two-way bootstrap, resampling listeners and, independently within each class, clips (Test 1 classes: Iowa, K18, Salamander, SoundFont, Otey; pilot and Test 2: CCMusic, library, SoundFont, Otey); B = 2000 (500 for the mean listener AUC), seed 0, 2.5/97.5 percentiles. The listener order inside the resampling is that of the paper's computation, so the intervals reproduce Table 8 exactly.
- **Listener AUC**: Mann–Whitney AUC of one listener's 1–4 ratings, algorithmic (positive) vs recorded clips, ties = 1/2. **Panel AUC**: the same on the mean rating of each clip over all listeners.
- **Accuracy**: share of a listener's answers with 1–2 for recorded and 3–4 for algorithmic clips.
- **Tests over listeners**: one-sided Wilcoxon signed-rank tests of per-listener values minus 0.5 (zero differences split), e.g. listener AUC > 0.5 and per-listener library judged-synthetic rate < 0.5.
- **Detector**: XGBoost on the 47 features, trained on isolated notes without the clip's own instrument (library notes never in training); flagged = P(synthetic) > 0.5. Each clip WAV was scored as heard by the listeners (Test 1 and Test 2); the pilot clips were not scored.

## 5. Results

### 5.1 Test 1

| Class | Clips | Answers 3-4 | Judged synthetic [95% CI] | Mean rating | Detector flagged |
|---|---|---|---|---|---|
| Library notes (all 20) | 20 | 238/640 | 37% [30, 44] | 2.21 | 10 |
| Iowa | 7 | 96/224 | 43% [27, 58] | 2.27 | 3 |
| K18 | 7 | 78/224 | 35% [24, 45] | 2.18 | 4 |
| Salamander | 6 | 64/192 | 33% [22, 45] | 2.18 | 3 |
| SoundFont | 5 | 107/160 | 67% [51, 81] | 2.86 | 2 |
| Otey | 5 | 101/160 | 63% [48, 78] | 2.75 | 3 |
| Algorithmic (all 10) | 10 | 208/320 | 65% [55, 75] | 2.81 | 5 |

Mean listener AUC 0.67 [0.60, 0.73] (median 0.69; 27 of 32 listeners above 0.5, 5 below; Wilcoxon AUC > 0.5: p = 7.5e-07). Panel AUC 0.98 [0.80, 1.00]; detector AUC 0.575. Mean accuracy (1-2 = recorded, 3-4 = algorithmic) 63.5%.

Library notes judged synthetic below one half: Wilcoxon p = 2e-05. Salamander vs Iowa + K18 (paired over listeners, judged-synthetic rate): p = 0.21.

Per clip:

| Clip | Class / subclass | Source instrument | Note | Judged synthetic | Mean rating | Detector P |
|---|---|---|---|---|---|---|
| clip_01 | recorded / library | Salamander Grand V3 (Yamaha C5) | D#3 | 28% (9/32) | 2.22 | 0.10 |
| clip_02 | recorded / library | Iowa MIS Piano (Steinway) | D3 | 62% (20/32) | 2.66 | 0.11 |
| clip_03 | recorded / library | K18 upright | D#3 | 44% (14/32) | 2.38 | 0.34 |
| clip_04 | recorded / library | Salamander Grand V3 (Yamaha C5) | C4 | 50% (16/32) | 2.34 | 0.96 |
| clip_05 | recorded / library | Salamander Grand V3 (Yamaha C5) | D#4 | 25% (8/32) | 1.97 | 0.80 |
| clip_06 | recorded / library | Iowa MIS Piano (Steinway) | B4 | 9% (3/32) | 1.59 | 0.83 |
| clip_07 | algorithmic / physical_model | Otey waveguide model | B4 | 84% (27/32) | 3.12 | 0.98 |
| clip_08 | recorded / library | Salamander Grand V3 (Yamaha C5) | A3 | 38% (12/32) | 2.38 | 0.00 |
| clip_09 | recorded / library | Iowa MIS Piano (Steinway) | F4 | 50% (16/32) | 2.41 | 0.52 |
| clip_10 | algorithmic / sample_bank | SoundFont Casio CA-110 "Piano (FM)" | C4 | 69% (22/32) | 3.03 | 1.00 |
| clip_11 | recorded / library | Salamander Grand V3 (Yamaha C5) | F#4 | 28% (9/32) | 2.25 | 0.02 |
| clip_12 | recorded / library | Iowa MIS Piano (Steinway) | B3 | 44% (14/32) | 2.44 | 0.56 |
| clip_13 | algorithmic / physical_model | Otey waveguide model | C5 | 56% (18/32) | 2.75 | 0.82 |
| clip_14 | recorded / library | Iowa MIS Piano (Steinway) | G3 | 47% (15/32) | 2.38 | 0.01 |
| clip_15 | recorded / library | K18 upright | C5 | 38% (12/32) | 2.12 | 0.99 |
| clip_16 | recorded / library | K18 upright | C3 | 34% (11/32) | 2.22 | 0.00 |
| clip_17 | recorded / library | K18 upright | F#3 | 38% (12/32) | 2.19 | 0.63 |
| clip_18 | recorded / library | K18 upright | D#4 | 38% (12/32) | 2.22 | 0.20 |
| clip_19 | algorithmic / physical_model | Otey waveguide model | E4 | 69% (22/32) | 2.75 | 0.78 |
| clip_20 | recorded / library | Salamander Grand V3 (Yamaha C5) | D#5 | 31% (10/32) | 1.91 | 1.00 |
| clip_21 | algorithmic / sample_bank | SoundFont Casio CA-110 "Piano (FM)" | C3 | 78% (25/32) | 3.09 | 1.00 |
| clip_22 | recorded / library | K18 upright | A3 | 31% (10/32) | 2.09 | 0.99 |
| clip_23 | recorded / library | K18 upright | D#5 | 22% (7/32) | 2.03 | 0.61 |
| clip_24 | algorithmic / sample_bank | SoundFont Casio CA-110 "Piano (FM)" | A4 | 69% (22/32) | 2.91 | 0.02 |
| clip_25 | algorithmic / physical_model | Otey waveguide model | G4 | 59% (19/32) | 2.69 | 0.06 |
| clip_26 | algorithmic / sample_bank | SoundFont Casio CA-110 "Piano (FM)" | A3 | 53% (17/32) | 2.56 | 0.00 |
| clip_27 | recorded / library | Iowa MIS Piano (Steinway) | D5 | 31% (10/32) | 1.94 | 0.01 |
| clip_28 | algorithmic / physical_model | Otey waveguide model | E3 | 47% (15/32) | 2.44 | 0.15 |
| clip_29 | recorded / library | Iowa MIS Piano (Steinway) | F3 | 56% (18/32) | 2.50 | 0.02 |
| clip_30 | algorithmic / sample_bank | SoundFont Casio CA-110 "Piano (FM)" | C5 | 66% (21/32) | 2.72 | 0.32 |

Per listener:

| Listener | Experience (self-reported) | Accuracy | AUC |
|---|---|---|---|
| T1-01 | musician | 63% | 0.75 |
| T1-02 | not a musician | 77% | 0.80 |
| T1-03 | amateur | 80% | 0.84 |
| T1-04 | amateur | 83% | 0.82 |
| T1-05 | not a musician | 70% | 0.81 |
| T1-06 | amateur | 70% | 0.74 |
| T1-07 | amateur | 70% | 0.82 |
| T1-08 | not a musician | 70% | 0.69 |
| T1-09 | amateur | 77% | 0.70 |
| T1-10 | amateur | 63% | 0.65 |
| T1-11 | not a musician | 63% | 0.80 |
| T1-12 | amateur | 83% | 0.96 |
| T1-13 | amateur | 77% | 0.77 |
| T1-14 | not a musician | 67% | 0.76 |
| T1-15 | not a musician | 60% | 0.65 |
| T1-16 | amateur | 73% | 0.73 |
| T1-17 | not a musician | 80% | 0.69 |
| T1-18 | not a musician | 63% | 0.67 |
| T1-19 | amateur | 57% | 0.74 |
| T1-20 | not a musician | 63% | 0.68 |
| T1-21 | not a musician | 63% | 0.55 |
| T1-22 | not a musician | 53% | 0.55 |
| T1-23 | amateur | 63% | 0.71 |
| T1-24 | not a musician | 43% | 0.45 |
| T1-25 | not a musician | 50% | 0.45 |
| T1-26 | amateur | 63% | 0.60 |
| T1-27 | not a musician | 57% | 0.69 |
| T1-28 | not a musician | 57% | 0.48 |
| T1-29 | not a musician | 40% | 0.52 |
| T1-30 | amateur | 53% | 0.51 |
| T1-31 | amateur | 40% | 0.34 |
| T1-32 | not a musician | 40% | 0.35 |

### 5.2 Pilot (1.3-s clips)

| Class | Clips | Answers 3-4 | Judged synthetic [95% CI] | Mean rating |
|---|---|---|---|---|
| Library notes | 5 | 24/90 | 27% [13, 44] | 1.97 |
| Real notes (CCMusic) | 5 | 44/90 | 49% [32, 66] | 2.37 |
| Recorded (all 10) | 10 | 68/180 | 38% [27, 49] | 2.17 |
| SoundFont | 8 | 54/144 | 38% [18, 58] | 2.17 |
| Otey | 6 | 62/108 | 57% [43, 72] | 2.58 |
| Algorithmic (all 14) | 14 | 116/252 | 46% [34, 60] | 2.35 |

Mean listener AUC 0.55 [0.44, 0.65] (median 0.54; 11 of 18 listeners above 0.5, 7 below; Wilcoxon AUC > 0.5: p = 0.11). Panel AUC 0.60 [0.37, 0.83]. Mean accuracy (1-2 = recorded, 3-4 = algorithmic) 52.8%.

Per clip (`same_stimulus_in_test2` in `S3_pilot_stimuli.csv` links each pilot clip to its Test 2 counterpart):

| Clip | Class / subclass | Source instrument | Note | Judged synthetic | Mean rating | Detector P |
|---|---|---|---|---|---|---|
| clip_01 | algorithmic / physical_model | Otey waveguide model | G3 | 50% (9/18) | 2.44 | — |
| clip_02 | recorded / library | Salamander Grand V3 (Yamaha C5) | C3 | 28% (5/18) | 1.89 | — |
| clip_03 | algorithmic / physical_model | Otey waveguide model | C3 | 50% (9/18) | 2.33 | — |
| clip_04 | recorded / library | K18 Upright | F#4 | 22% (4/18) | 1.89 | — |
| clip_05 | algorithmic / physical_model | Otey waveguide model | E5 | 67% (12/18) | 3.06 | — |
| clip_06 | recorded / dataset | CCMusic Hsinghai | C3 | 50% (9/18) | 2.33 | — |
| clip_07 | recorded / library | Iowa MIS Piano (Steinway B) | A#4 | 11% (2/18) | 1.61 | — |
| clip_08 | recorded / library | FreePats Piano FB | A4 | 28% (5/18) | 2.06 | — |
| clip_09 | recorded / dataset | CCMusic PearlRiver | A#3 | 44% (8/18) | 2.33 | — |
| clip_10 | algorithmic / sample_bank | CTK_230_SoundFont (SoundFont) | F#3 | 44% (8/18) | 2.28 | — |
| clip_11 | algorithmic / sample_bank | CTK_533_Piano1_00_Casio (SoundFont) | A4 | 17% (3/18) | 1.61 | — |
| clip_12 | algorithmic / sample_bank | Yamaha_PSRF50 (SoundFont) | C3 | 22% (4/18) | 1.89 | — |
| clip_13 | recorded / dataset | CCMusic Steinway-T | D3 | 61% (11/18) | 2.67 | — |
| clip_14 | recorded / library | K18 Upright | C4 | 44% (8/18) | 2.39 | — |
| clip_15 | algorithmic / sample_bank | CP80_16_44100 (SoundFont) | A3 | 78% (14/18) | 3.11 | — |
| clip_16 | algorithmic / sample_bank | U20PIANO (SoundFont) | F#4 | 33% (6/18) | 2.06 | — |
| clip_17 | algorithmic / sample_bank | Yamaha_S90_ES_Natural_Grand (SoundFont) | D#5 | 33% (6/18) | 1.89 | — |
| clip_18 | recorded / dataset | CCMusic Yamaha | A4 | 33% (6/18) | 1.89 | — |
| clip_19 | algorithmic / sample_bank | Casio_CA_110_Piano_FM (SoundFont) | D#3 | 67% (12/18) | 3.00 | — |
| clip_20 | algorithmic / physical_model | Otey waveguide model | G4 | 61% (11/18) | 2.39 | — |
| clip_21 | algorithmic / physical_model | Otey waveguide model | C4 | 67% (12/18) | 2.78 | — |
| clip_22 | algorithmic / sample_bank | YAMAHA_PSR400_500 (SoundFont) | C5 | 6% (1/18) | 1.50 | — |
| clip_23 | algorithmic / physical_model | Otey waveguide model | B3 | 50% (9/18) | 2.50 | — |
| clip_24 | recorded / dataset | CCMusic Kawai | F#4 | 56% (10/18) | 2.61 | — |

Per listener:

| Listener | Experience (self-reported) | Accuracy | AUC | Same listener in Test 2 |
|---|---|---|---|---|
| P-01 | music school | 54% | 0.57 | — |
| P-02 | amateur | 46% | 0.42 | — |
| P-03 | no musical training | 50% | 0.41 | T2-03 |
| P-04 | music school | 42% | 0.31 | T2-02 |
| P-05 | no musical training | 50% | 0.47 | — |
| P-06 | no musical training | 50% | 0.60 | T2-20 |
| P-07 | amateur | 62% | 0.64 | — |
| P-08 | no musical training | 42% | 0.40 | — |
| P-09 | music school | 62% | 0.70 | — |
| P-10 | no musical training | 54% | 0.59 | — |
| P-11 | no musical training | 67% | 0.76 | T2-13 |
| P-12 | music school | 75% | 0.81 | T2-15 |
| P-13 | music school | 46% | 0.46 | — |
| P-14 | music school | 50% | 0.64 | T2-10 |
| P-15 | music school | 46% | 0.51 | T2-14 |
| P-16 | music school | 46% | 0.52 | — |
| P-17 | music school | 42% | 0.37 | T2-05 |
| P-18 | no musical training | 67% | 0.66 | T2-18 |

Free-text answers (translated): P-01: "Additional incidental sounds that do not occur in synthetic sound."; P-02: "In some cases there is background noise; it was distracting."; P-03: "The beginning of the sound when the key is pressed and the further structure of the sound while the key is held."; P-04: "The sound of the hammer strike, the clarity of the sound and the loudness."; P-07: "Everything."; P-13: "Overtones: their number and colour."; P-14: "Echo, background noise."; P-17: "The sound is very short, so it is hard to decide.". Other listeners left the field empty.

### 5.3 Test 2 (4-s clips)

| Class | Clips | Answers 3-4 | Judged synthetic [95% CI] | Mean rating | Detector flagged |
|---|---|---|---|---|---|
| Library notes | 5 | 24/100 | 24% [10, 40] | 1.89 | 2 |
| Real notes (CCMusic) | 5 | 41/100 | 41% [22, 61] | 2.16 | 2 |
| Recorded (all 10) | 10 | 65/200 | 32% [19, 47] | 2.02 | 4 |
| SoundFont | 8 | 91/160 | 57% [42, 72] | 2.63 | 6 |
| Otey | 6 | 71/120 | 59% [42, 75] | 2.63 | 0 |
| Algorithmic (all 14) | 14 | 162/280 | 58% [46, 70] | 2.63 | 6 |

Mean listener AUC 0.64 [0.55, 0.74] (median 0.62; 17 of 20 listeners above 0.5, 3 below; Wilcoxon AUC > 0.5: p = 0.0002). Panel AUC 0.91 [0.65, 1.00]; detector AUC 0.593. Mean accuracy (1-2 = recorded, 3-4 = algorithmic) 61.9%.

Library notes judged synthetic below one half: Wilcoxon p = 0.0004. Listeners who took both the pilot and Test 2 (n = 9): mean AUC 0.56 in the pilot, 0.62 in Test 2 (Wilcoxon two-sided p = 0.43).

Per clip:

| Clip | Class / subclass | Source instrument | Note | Judged synthetic | Mean rating | Detector P |
|---|---|---|---|---|---|---|
| c3_01 | recorded / dataset | CCMusic Yamaha | C3 | 40% (8/20) | 2.10 | 0.96 |
| c3_02 | algorithmic / physical_model | Otey waveguide model (v0 build) | E5 | 60% (12/20) | 2.60 | 0.40 |
| c3_03 | recorded / library | K18 Upright | C4 | 15% (3/20) | 1.90 | 0.51 |
| c3_04 | recorded / dataset | CCMusic Yamaha | A4 | 35% (7/20) | 2.00 | 0.02 |
| c3_05 | algorithmic / sample_bank | Casio_PX_860_Concert_Grand_Piano (SoundFont) | F#4 | 35% (7/20) | 2.15 | 0.98 |
| c3_06 | algorithmic / sample_bank | Yamaha_PSRF50 (SoundFont) | C3 | 45% (9/20) | 2.40 | 0.13 |
| c3_07 | recorded / dataset | CCMusic Kawai-G | D3 | 45% (9/20) | 2.20 | 1.00 |
| c3_08 | algorithmic / physical_model | Otey waveguide model (v0 build) | B3 | 60% (12/20) | 2.50 | 0.32 |
| c3_09 | algorithmic / physical_model | Otey waveguide model (v0 build) | C3 | 50% (10/20) | 2.50 | 0.22 |
| c3_10 | algorithmic / sample_bank | CP80_16_44100 (SoundFont) | A3 | 95% (19/20) | 3.60 | 0.97 |
| c3_11 | algorithmic / sample_bank | JV1080_Nice_Piano_m (SoundFont) | D#5 | 35% (7/20) | 2.15 | 0.85 |
| c3_12 | algorithmic / physical_model | Otey waveguide model (v0 build) | G4 | 45% (9/20) | 2.50 | 0.03 |
| c3_13 | algorithmic / sample_bank | YAMAHA_PSR400_500 (SoundFont) | C5 | 60% (12/20) | 2.70 | 0.01 |
| c3_14 | recorded / library | Salamander Grand V3 (Yamaha C5) | C3 | 15% (3/20) | 1.60 | 0.04 |
| c3_15 | algorithmic / sample_bank | Casio_CA_110_Piano_FM (SoundFont) | C3 | 65% (13/20) | 2.85 | 0.85 |
| c3_16 | recorded / library | Iowa MIS Piano (Steinway B) | A#4 | 25% (5/20) | 1.85 | 0.01 |
| c3_17 | algorithmic / sample_bank | CTK_230_SoundFont (SoundFont) | F#3 | 60% (12/20) | 2.55 | 0.57 |
| c3_18 | algorithmic / physical_model | Otey waveguide model (v0 build) | G3 | 70% (14/20) | 2.85 | 0.07 |
| c3_19 | algorithmic / sample_bank | CTK_533_Piano1_00_Casio (SoundFont) | A4 | 60% (12/20) | 2.65 | 0.82 |
| c3_20 | algorithmic / physical_model | Otey waveguide model (v0 build) | C4 | 70% (14/20) | 2.85 | 0.03 |
| c3_21 | recorded / dataset | CCMusic YoungChang | A#3 | 65% (13/20) | 2.65 | 0.00 |
| c3_22 | recorded / dataset | CCMusic Kawai-G | F#4 | 20% (4/20) | 1.85 | 0.40 |
| c3_23 | recorded / library | K18 Upright | F#4 | 25% (5/20) | 1.85 | 0.00 |
| c3_24 | recorded / library | FreePats Piano FB | A4 | 40% (8/20) | 2.25 | 0.85 |

Per listener:

| Listener | Experience (self-reported) | Accuracy | AUC | Same listener in pilot |
|---|---|---|---|---|
| T2-01 | no musical training | 46% | 0.51 | — |
| T2-02 | music school | 58% | 0.58 | P-04 |
| T2-03 | no musical training | 42% | 0.49 | P-03 |
| T2-04 | amateur | 58% | 0.54 | — |
| T2-05 | music school | 54% | 0.62 | P-17 |
| T2-06 | professional musician | 83% | 0.81 | — |
| T2-07 | music school | 58% | 0.64 | — |
| T2-08 | no musical training | 62% | 0.62 | — |
| T2-09 | music school | 79% | 0.88 | — |
| T2-10 | music school | 42% | 0.42 | P-14 |
| T2-11 | music school | 62% | 0.71 | — |
| T2-12 | music school | 46% | 0.51 | — |
| T2-13 | no musical training | 58% | 0.57 | P-11 |
| T2-14 | music school | 67% | 0.69 | P-15 |
| T2-15 | music school | 79% | 0.85 | P-12 |
| T2-16 | music school | 83% | 0.85 | — |
| T2-17 | music school | 46% | 0.40 | — |
| T2-18 | no musical training | 62% | 0.60 | P-18 |
| T2-19 | no musical training | 79% | 0.86 | — |
| T2-20 | no musical training | 71% | 0.72 | P-06 |

Free-text answers (translated): T2-02: "Clarity of the sound (ringing vs dull); and sometimes the duration of the sound gives it away. :)"; T2-05: "The sound in the recording is too drawn out."; T2-07: "Presence of unnatural beats in the sound."; T2-09: "Overtones and my own intuition."; T2-10: "Echo."; T2-12: "The decay; some files are slightly cut off, so it is hard to compare. Also, an artificial modulation is noticeable in some recordings."; T2-16: "I paid attention to the softness of the note, its clarity and dynamics."; T2-17: "The endings of the sounds."; T2-20: "Unnatural background incidental sounds and noises typical of synthesis; poor timbre. Richness of timbre and subtle natural imperfections, beats, typical of natural sound.". Other listeners left the field empty.

### 5.4 Check against Table 8 of the paper

All values in Table 8 are reproduced: Test 1 library 37% [30, 44], SoundFont 67% [51, 81], Otey 63% [48, 78], mean listener AUC 0.67 [0.60, 0.73], panel 0.98, detector 0.58 (0.575), flagged 10 (3/4/3), 2, 3; Test 2 library 24% [10, 40], CCMusic 41% [22, 61], SoundFont 57% [42, 72], Otey 59% [42, 75], mean listener AUC 0.64 [0.55, 0.74], panel 0.91, detector 0.59 (0.593), flagged 2, 2, 6, 0.

## 6. Test 2 stimuli: substitutions and a known limitation

**Substitutions relative to the pilot** (column `substitution_vs_pilot`). SoundFont and Otey notes were re-rendered with the key held 5 s. Three SoundFont notes whose sample ends inside the 4.15-s clip although the key is held were replaced (U20PIANO F#4 → Casio PX-860 F#4; Yamaha S90 ES D#5 → JV1080_Nice_Piano_m D#5; Casio CA-110 D#3 → Casio CA-110 C3). Two CCMusic notes with audible background noise (A-weighted noise floor above −60 dB re peak) were replaced by the cleanest CCMusic note of the same pitch (Hsinghai C3 → Yamaha C3; Steinway-T D3 → Kawai-G D3), and two CCMusic pianos with no note lasting 4 s were replaced (PearlRiver A#3 → YoungChang A#3; Kawai F#4 → Kawai-G F#4). The 10 recorded clips therefore come from CCMusic Yamaha (2), Kawai-G (2) and YoungChang (1) and from the K18 (2), Salamander, Iowa and FreePats libraries. No denoising or filtering was applied; all clips have a noise floor below −60 dB(A) re peak (`noise_floor_dba_re_peak`).

**Limitation: the fade-out shortens recorded notes that are still sounding.** The clip ends with a 150-ms fade-out from 4.0 s. At 3.9 s all 10 recorded notes were still sounding (level −24.8 to −55.1 dB re peak), against 9 of 14 algorithmic notes (−37.2 to −55.2 dB re peak); the other 5 algorithmic notes (c3_02 Otey waveguide model (v0 build) E5, c3_06 Yamaha_PSRF50 C3, c3_10 CP80_16_44100 A3, c3_11 JV1080_Nice_Piano_m D#5, c3_15 Casio_CA_110_Piano_FM C3) had decayed below −60 dB re peak before 3.9 s and end naturally. A recorded note is thus more often cut audibly than an algorithmic one, which could bias answers either way; one listener remarked that "some files are slightly cut off" (T2-12). Levels at 3.9 s for every clip are in `S3_test2_stimuli.csv` (`level_at_3p9s_dbfs`, `level_at_3p9s_db_re_peak`).

## 7. Column reference

- `presentation_order`: clip numbers in the order the listener saw them (pilot `clip_NN`, Test 2 `c3_NN`, numbers only, separated by "-"); Test 1: the same fixed order for all.
- `experience_value` / `experience_label`: the option value stored by the form and its English label.
- `date`: submission date (UTC). Test 1 answers carry no date (test period 28 June – 2 July 2026).
- `source_file`: base name of the source audio file (SoundFont and Otey: name of the rendered note).
- Test 1 stimuli: `sounding_t60_s` = end of the last 10-ms frame within 60 dB of the clip's peak frame; `last_nonzero_sample_s` = time of the last non-zero sample; `onset_after_file_start_ms` = onset position in the source file.
- Pilot stimuli: `silence_padded_preroll_ms` = digital silence added before the file start to complete the 10-ms pre-roll; `level_before_fadeout_db_re_peak` = level at the start of the fade-out relative to the clip peak.
- Test 2 stimuli: `gain_db` = gain applied for peak normalization; `still_sounding_when_fadeout_starts` = level at 3.9 s above −60 dB re peak.
