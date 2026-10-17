# RESULTS v4 -- isolated regime with an enlarged recorded-note class

Generated 2026-10-02 11:41 by one command, from the project root:

```
python Stattia/revision/analysis/v4/run_v4.py
```

Environment: Python 3.10.7, numpy 1.26.4, pandas 1.5.2, xgboost 3.2.0 (n_jobs=1 per model, random_state=0; protocol hyper-parameters of pv2_lib.make_model). Runtime 265 s (feature extraction cached in `v4/cache/`). Scripts: `v4/run_v4.py`, `v4/v4_lib.py`, `v4/get_maps_isol.py`. Existing protocol files and outputs are read only.

## 1. Data

**Recorded-note class (label 0)**, one CV group per instrument:

| instrument | collection | notes (all layers) | notes (mf variant) | CV group |
|---|---|---|---|---|
| Hsinghai | CCMusic | 88 | 88 | rec:cc:Hsinghai |
| Kawai | CCMusic | 88 | 88 | rec:cc:Kawai |
| Kawai-G | CCMusic | 88 | 88 | rec:cc:Kawai-G |
| PearlRiver | CCMusic | 88 | 88 | rec:cc:PearlRiver |
| Steinway | CCMusic | 52 | 52 | rec:cc:Steinway |
| Steinway-T | CCMusic | 88 | 88 | rec:cc:Steinway-T |
| Yamaha | CCMusic | 88 | 88 | rec:cc:Yamaha |
| YoungChang | CCMusic | 88 | 88 | rec:cc:YoungChang |
| Salamander | free library | 480 | 30 | rec:lib:Salamander |
| Iowa | free library | 261 | 86 | rec:lib:Iowa |
| K18 | free library | 180 | 30 | rec:lib:K18 |
| FreePats | free library | 78 | 78 | rec:lib:FreePats |
| MAPS ENSTDkCl | MAPS | 120 | 34 | rec:maps:disklavier |
| MAPS ENSTDkAm | MAPS | 132 | 43 | rec:maps:disklavier |

**Synthetic (label 1)**: 678 SoundFont notes of 10 instruments (velocity copies removed, as v3; one group each) + 120 Otey notes (one group). Features of CCMusic / SoundFont / Otey = the stored v3 features (quality/cache), libraries = the v2 extraction cache (out/cache/library_notes.csv, read only), MAPS = the same 47-feature pipeline (v4/cache/maps_orig.csv). Check: the v3 seed-0 CCMusic-vs-SoundFont AUC (0.937) is reproduced exactly from these inputs with the v3 folds.

**MAPS** (Emiya, Bertin & Badeau, MAPS -- MIDI Aligned Piano Sounds v0.4, Zenodo doi:10.5281/zenodo.18160555; licence Creative Commons BY-NC-SA 2.0 FR, non-commercial research use with attribution): only the two REAL Disklavier subsets ENSTDkCl (close mics) and ENSTDkAm (ambient mics), subfolder ISOL/NO (normal isolated notes), sustain pedal off (S0; the readme reports a pedal re-press bug in S1 isolated notes), all dynamics P/M/F ("mf variant" = M). The software-piano subsets (AkPn*, Sptk*, Stbg*) were not downloaded. Fetched by HTTP range requests from the two zip archives (about 90 MB per room instead of 2.6-2.7 GB) into `audio/_maps_isol/` with licence.txt and readme.txt. The two rooms are the same physical piano, so they form ONE CV group; false alarms are reported per room. Two variants of the recorded class: **all** = all velocity layers of the libraries and all MAPS dynamics; **mf** = mf layer of each library (Salamander v8, FreePats single layer), MAPS M only; CCMusic unchanged in both.

**Room augmentation**: two channel copies per note (all 2717 notes, both classes), each = the note convolved with a random measured room IR of the MIT IR Survey (270 IRs, 270 distinct drawn) + white noise at SNR ~ U(30, 50) dB re the RMS of the first second after onset (drawn SNR median 39.9 dB); deterministic per note (RNG = crc32 of "v4aug<k>:" + path). Features re-extracted for every copy (cached, `v4/cache/iso_aug.csv`; extraction failures / <30 features: [2, 3] -- such rows fall back to the unaltered note). No subsampling was needed. Training: each training note is, with probability 0.5, replaced by one of its two copies (Bernoulli and copy drawn per note and seed, RNG 1000+seed). Tests: **clean** = unaltered test notes; **channel** = every test note through one random copy of the same channel model. The no-augmentation model is also scored on the channel test (mismatch control).

**Evaluation**: grouped CV over the 24 instrument groups (8 CCMusic pianos, 4 libraries, the Disklavier, 10 SoundFonts, Otey) with the v2 greedy GroupKFold rule (5 folds; seeds 0-9 permute the tie order), every note scored by the model of its own fold (all notes are trainable, so this is the plain out-of-fold score); plus leave-one-group-out (LOGO, one fold per group). AUC = recorded notes vs the synthetic source; cells = median [min-max] over the 10 fold assignments; FA = share of recorded notes with P(synthetic) > 0.5. v3 reference = FINAL_NUMBERS.md (real class = CCMusic only, 6 brand groups, libraries never trained).

## 2a. Compact comparison, all47

| recorded vs | v3 (CCMusic only) | v4 all layers | v4 mf | v4+aug, clean test | v4+aug, channel test | v4 no aug, channel test (mismatch) | LOGO: v4 / v4+aug (clean) |
|---|---|---|---|---|---|---|---|
| SoundFont | 0.930 [0.919-0.937] | 0.942 [0.923-0.955] | 0.930 [0.909-0.949] | 0.954 [0.942-0.961] | 0.941 [0.937-0.954] | 0.891 [0.871-0.905] | 0.940 / 0.946 |
| Otey | 0.813 [0.783-0.831] | 0.826 [0.783-0.860] | 0.824 [0.781-0.861] | 0.855 [0.827-0.880] | 0.778 [0.744-0.822] | 0.740 [0.590-0.797] | 0.807 / 0.846 |
| algorithmic | 0.913 [0.900-0.920] | 0.925 [0.902-0.940] | 0.913 [0.890-0.935] | 0.937 [0.927-0.945] | 0.919 [0.909-0.931] | 0.867 [0.831-0.882] | 0.920 / 0.931 |
| FA, all recorded notes | 12.6% [10.3-13.0] (CCMusic) | 6.7% [6.1-7.7] | 8.7% [7.2-10.5] | 8.3% [7.4-9.9] | 8.8% [7.8-9.6] | 2.6% [1.8-3.8] | 7.7% / 8.3% |

Same models, real side restricted to the CCMusic notes (directly comparable with v3):

| CCMusic vs | v3 | v4 all layers | v4 mf | v4+aug, clean | v4+aug, channel |
|---|---|---|---|---|---|
| SoundFont | 0.930 [0.919-0.937] | 0.947 [0.929-0.959] | 0.934 [0.911-0.948] | 0.955 [0.949-0.965] | 0.934 [0.929-0.949] |
| Otey | 0.813 [0.783-0.831] | 0.856 [0.808-0.880] | 0.836 [0.786-0.861] | 0.874 [0.845-0.893] | 0.779 [0.750-0.826] |
| algorithmic | 0.913 [0.900-0.920] | 0.935 [0.911-0.947] | 0.920 [0.892-0.935] | 0.943 [0.937-0.951] | 0.914 [0.902-0.926] |

Seed 0 with grouped-bootstrap 95% CI (recorded groups and synthetic groups resampled separately; "real-side only" = one synthetic group); det = share of the synthetic notes with P > 0.5:

| recorded vs | v4 all layers | v4 mf | v4+aug, clean | v4+aug, channel |
|---|---|---|---|---|
| SoundFont | 0.937 [0.897, 0.971]; det 72.9% | 0.928 [0.859, 0.974]; det 76.7% | 0.946 [0.903, 0.978]; det 83.6% | 0.937 [0.894, 0.969]; det 81.7% |
| Otey | 0.844 (real-side only [0.792, 0.904]); det 27.5% | 0.810 (real-side only [0.721, 0.879]); det 25.8% | 0.865 (real-side only [0.820, 0.916]); det 40.8% | 0.777 (real-side only [0.721, 0.835]); det 34.2% |
| algorithmic | 0.923 [0.877, 0.966]; det 66.0% | 0.911 [0.841, 0.967]; det 69.0% | 0.934 [0.891, 0.974]; det 77.2% | 0.913 [0.854, 0.964]; det 74.6% |

## 2b. Compact comparison, med16

| recorded vs | v3 (CCMusic only) | v4 all layers | v4 mf | v4+aug, clean test | v4+aug, channel test | v4 no aug, channel test (mismatch) | LOGO: v4 / v4+aug (clean) |
|---|---|---|---|---|---|---|---|
| SoundFont | 0.862 [0.849-0.877] | 0.862 [0.843-0.885] | 0.871 [0.846-0.898] | 0.865 [0.848-0.889] | 0.890 [0.885-0.913] | 0.788 [0.767-0.817] | 0.859 / 0.844 |
| Otey | 0.631 [0.580-0.685] | 0.703 [0.665-0.756] | 0.673 [0.625-0.713] | 0.615 [0.554-0.660] | 0.599 [0.545-0.662] | 0.621 [0.463-0.699] | 0.691 / 0.637 |
| algorithmic | 0.826 [0.815-0.847] | 0.838 [0.821-0.866] | 0.841 [0.815-0.870] | 0.827 [0.807-0.853] | 0.846 [0.835-0.873] | 0.761 [0.725-0.792] | 0.834 / 0.813 |
| FA, all recorded notes | 18.0% [16.0-20.8] (CCMusic) | 10.4% [9.6-12.7] | 13.0% [11.0-14.7] | 15.9% [14.1-19.7] | 12.9% [10.8-14.6] | 1.8% [1.2-2.2] | 11.8% / 17.8% |

Same models, real side restricted to the CCMusic notes (directly comparable with v3):

| CCMusic vs | v3 | v4 all layers | v4 mf | v4+aug, clean | v4+aug, channel |
|---|---|---|---|---|---|
| SoundFont | 0.862 [0.849-0.877] | 0.877 [0.860-0.896] | 0.867 [0.838-0.889] | 0.872 [0.838-0.896] | 0.878 [0.867-0.902] |
| Otey | 0.631 [0.580-0.685] | 0.741 [0.681-0.778] | 0.654 [0.609-0.693] | 0.635 [0.538-0.681] | 0.599 [0.554-0.664] |
| algorithmic | 0.826 [0.815-0.847] | 0.857 [0.838-0.877] | 0.834 [0.803-0.860] | 0.837 [0.792-0.862] | 0.835 [0.820-0.865] |

Seed 0 with grouped-bootstrap 95% CI (recorded groups and synthetic groups resampled separately; "real-side only" = one synthetic group); det = share of the synthetic notes with P > 0.5:

| recorded vs | v4 all layers | v4 mf | v4+aug, clean | v4+aug, channel |
|---|---|---|---|---|
| SoundFont | 0.862 [0.775, 0.938]; det 60.9% | 0.846 [0.728, 0.934]; det 67.1% | 0.860 [0.790, 0.924]; det 70.8% | 0.892 [0.838, 0.932]; det 75.4% |
| Otey | 0.708 (real-side only [0.613, 0.819]); det 10.8% | 0.641 (real-side only [0.551, 0.726]); det 18.3% | 0.607 (real-side only [0.519, 0.725]); det 15.0% | 0.604 (real-side only [0.545, 0.661]); det 24.2% |
| algorithmic | 0.838 [0.748, 0.927]; det 53.4% | 0.815 [0.700, 0.920]; det 59.8% | 0.822 [0.726, 0.916]; det 62.4% | 0.849 [0.755, 0.924]; det 67.7% |

## 3. Per recorded instrument: false-alarm rate (P > 0.5), median [min-max] over seeds 0-9

| recorded instrument | all47 | all47, mf variant | med16 | all47 +aug, clean | all47 +aug, channel | all47 no aug, channel | all47 LOGO |
|---|---|---|---|---|---|---|---|
| Hsinghai | 1.1% [1.1-2.3] | 1.1% [0.0-3.4] | 5.7% [1.1-20.5] | 1.1% [0.0-5.7] | 6.2% [2.3-9.1] | 0.0% [0.0-0.0] | 1.1% |
| Kawai | 4.5% [1.1-10.2] | 6.2% [2.3-8.0] | 13.1% [4.5-17.0] | 4.5% [2.3-9.1] | 5.1% [2.3-11.4] | 1.1% [0.0-5.7] | 5.7% |
| Kawai-G | 1.1% [0.0-6.8] | 4.5% [2.3-8.0] | 1.1% [0.0-3.4] | 2.8% [0.0-8.0] | 4.5% [0.0-6.8] | 3.4% [1.1-5.7] | 3.4% |
| PearlRiver | 26.7% [21.6-38.6] | 34.1% [29.5-47.7] | 35.2% [27.3-47.7] | 34.7% [31.8-44.3] | 48.9% [40.9-58.0] | 21.6% [15.9-35.2] | 33.0% |
| Steinway | 0.0% [0.0-0.0] | 0.0% [0.0-0.0] | 1.9% [0.0-3.8] | 1.9% [0.0-5.8] | 1.9% [0.0-3.8] | 0.0% [0.0-3.8] | 0.0% |
| Steinway-T | 5.7% [3.4-8.0] | 6.8% [4.5-8.0] | 5.1% [2.3-14.8] | 5.1% [0.0-9.1] | 5.7% [3.4-11.4] | 1.1% [0.0-4.5] | 6.8% |
| Yamaha | 0.0% [0.0-2.3] | 1.1% [0.0-3.4] | 0.0% [0.0-2.3] | 0.0% [0.0-2.3] | 0.6% [0.0-3.4] | 0.0% [0.0-0.0] | 0.0% |
| YoungChang | 4.0% [1.1-8.0] | 8.5% [3.4-11.4] | 5.1% [0.0-9.1] | 2.3% [0.0-4.5] | 2.3% [0.0-3.4] | 1.1% [0.0-2.3] | 4.5% |
| Salamander | 8.5% [8.5-8.5] | 5.0% [0.0-13.3] | 18.1% [18.1-18.1] | 11.5% [9.8-12.3] | 8.9% [7.5-10.0] | 1.4% [1.0-2.1] | 9.2% |
| Iowa | 1.5% [0.8-2.3] | 0.0% [0.0-1.2] | 0.8% [0.4-1.9] | 2.1% [0.8-3.4] | 0.8% [0.0-1.5] | 0.4% [0.0-1.1] | 1.5% |
| K18 | 12.2% [10.6-15.0] | 15.0% [10.0-23.3] | 13.6% [11.7-15.6] | 10.8% [8.3-18.3] | 8.1% [5.6-14.4] | 1.9% [0.0-4.4] | 13.3% |
| FreePats | 10.3% [5.1-12.8] | 16.0% [7.7-33.3] | 13.5% [7.7-16.7] | 18.6% [11.5-25.6] | 16.0% [12.8-21.8] | 4.5% [2.6-7.7] | 10.3% |
| MAPS ENSTDkCl | 5.4% [4.2-9.2] | 7.4% [5.9-14.7] | 10.4% [5.8-20.0] | 6.7% [3.3-11.7] | 10.8% [7.5-13.3] | 2.5% [0.8-8.3] | 6.7% |
| MAPS ENSTDkAm | 6.1% [3.0-12.1] | 9.3% [4.7-16.3] | 4.5% [3.0-6.8] | 8.7% [5.3-11.4] | 10.6% [6.1-12.1] | 1.9% [0.0-9.8] | 8.3% |

Per recorded instrument: mean P(synthetic) and AUC of that instrument alone vs all algorithmic notes (all47, all layers, no augmentation / with augmentation, clean test):

| recorded instrument | mean P | AUC vs algorithmic | mean P, +aug | AUC vs algorithmic, +aug |
|---|---|---|---|---|
| Hsinghai | 0.031 [0.014-0.043] | 0.962 [0.951-0.980] | 0.047 [0.030-0.069] | 0.972 [0.960-0.980] |
| Kawai | 0.060 [0.035-0.089] | 0.936 [0.911-0.966] | 0.092 [0.048-0.129] | 0.948 [0.923-0.969] |
| Kawai-G | 0.031 [0.018-0.077] | 0.967 [0.910-0.976] | 0.048 [0.029-0.122] | 0.971 [0.924-0.981] |
| PearlRiver | 0.284 [0.253-0.395] | 0.781 [0.694-0.816] | 0.355 [0.308-0.441] | 0.798 [0.726-0.839] |
| Steinway | 0.007 [0.002-0.015] | 0.987 [0.979-0.996] | 0.047 [0.033-0.065] | 0.971 [0.965-0.982] |
| Steinway-T | 0.061 [0.041-0.087] | 0.944 [0.924-0.970] | 0.069 [0.021-0.103] | 0.964 [0.940-0.987] |
| Yamaha | 0.013 [0.005-0.027] | 0.982 [0.971-0.991] | 0.022 [0.013-0.043] | 0.984 [0.975-0.992] |
| YoungChang | 0.049 [0.025-0.087] | 0.953 [0.925-0.976] | 0.053 [0.034-0.084] | 0.971 [0.953-0.982] |
| Salamander | 0.110 [0.110-0.110] | 0.898 [0.872-0.913] | 0.148 [0.123-0.160] | 0.916 [0.900-0.934] |
| Iowa | 0.022 [0.014-0.033] | 0.978 [0.960-0.982] | 0.035 [0.025-0.062] | 0.979 [0.960-0.982] |
| K18 | 0.143 [0.126-0.161] | 0.884 [0.862-0.911] | 0.141 [0.111-0.195] | 0.924 [0.893-0.947] |
| FreePats | 0.132 [0.077-0.150] | 0.888 [0.858-0.939] | 0.205 [0.147-0.261] | 0.897 [0.849-0.921] |
| MAPS ENSTDkCl | 0.068 [0.053-0.115] | 0.938 [0.906-0.959] | 0.104 [0.065-0.128] | 0.944 [0.924-0.963] |
| MAPS ENSTDkAm | 0.076 [0.058-0.131] | 0.930 [0.893-0.951] | 0.119 [0.095-0.140] | 0.933 [0.917-0.945] |

## 4. Are recorded collections distinguishable among themselves?

**(a) Through the detector**: AUC that the detector scores a collection's notes as MORE synthetic than CCMusic notes (both labelled recorded; 0.5 = the detector treats them alike). Median [min-max] over seeds:

| collection vs CCMusic | all47 | med16 | all47 +aug, clean | all47 +aug, channel |
|---|---|---|---|---|
| Salamander | 0.700 [0.673-0.723] | 0.682 [0.658-0.708] | 0.649 [0.619-0.692] | 0.566 [0.517-0.593] |
| Iowa | 0.410 [0.336-0.423] | 0.308 [0.257-0.334] | 0.407 [0.344-0.459] | 0.402 [0.353-0.458] |
| K18 | 0.616 [0.593-0.660] | 0.598 [0.538-0.629] | 0.566 [0.501-0.639] | 0.579 [0.536-0.601] |
| FreePats | 0.602 [0.559-0.653] | 0.597 [0.524-0.638] | 0.630 [0.569-0.682] | 0.581 [0.529-0.630] |
| MAPS Disklavier (both rooms) | 0.508 [0.408-0.583] | 0.482 [0.408-0.551] | 0.522 [0.465-0.551] | 0.524 [0.479-0.572] |

**(b) Collection-signature classifier** (nobody labelled synthetic): XGBoost, same hyper-parameters, trained to tell a target collection (1) from the CCMusic notes (0); 5-fold GroupKFold with CCMusic grouped by piano (unseen pianos in each test fold) and the target grouped by nominal MIDI pitch (unseen pitches); pooled out-of-fold AUC. Reference rows: each CCMusic piano against the other seven (same collection and channel, different piano), each SoundFont and Otey against CCMusic (a synthetic source through the same test), and the two MAPS rooms against each other (same piano, different room). "channel" = every note of both classes replaced by its channel copy 0 (random IR + noise).

| target (1) vs CCMusic (0) | n target / n reference | all47 clean | all47 channel | med16 clean | med16 channel |
|---|---|---|---|---|---|
| Salamander | 480 / 668 | 0.946 | 0.879 | 0.906 | 0.796 |
| Iowa | 261 / 668 | 0.892 | 0.787 | 0.879 | 0.783 |
| K18 | 180 / 668 | 0.712 | 0.695 | 0.742 | 0.703 |
| FreePats | 78 / 668 | 0.774 | 0.775 | 0.721 | 0.643 |
| MAPS (both rooms) | 252 / 668 | 0.843 | 0.824 | 0.832 | 0.826 |
| MAPS ENSTDkCl | 120 / 668 | 0.779 | 0.701 | 0.769 | 0.711 |
| MAPS ENSTDkAm | 132 / 668 | 0.825 | 0.808 | 0.802 | 0.793 |
| CCMusic Hsinghai (vs the other 7) | 88 / 580 | 0.884 | 0.807 | 0.869 | 0.772 |
| CCMusic Kawai (vs the other 7) | 88 / 580 | 0.806 | 0.797 | 0.794 | 0.779 |
| CCMusic Kawai-G (vs the other 7) | 88 / 580 | 0.973 | 0.855 | 0.962 | 0.867 |
| CCMusic PearlRiver (vs the other 7) | 88 / 580 | 0.860 | 0.929 | 0.833 | 0.907 |
| CCMusic Steinway (vs the other 7) | 52 / 616 | 0.932 | 0.819 | 0.942 | 0.729 |
| CCMusic Steinway-T (vs the other 7) | 88 / 580 | 0.867 | 0.770 | 0.855 | 0.672 |
| CCMusic Yamaha (vs the other 7) | 88 / 580 | 0.876 | 0.772 | 0.850 | 0.730 |
| CCMusic YoungChang (vs the other 7) | 88 / 580 | 0.769 | 0.737 | 0.791 | 0.697 |
| SoundFont CP80_16_44100 | 53 / 668 | 0.978 | 0.950 | 0.935 | 0.831 |
| SoundFont CTK_230_SoundFont | 78 / 668 | 0.999 | 0.964 | 0.992 | 0.911 |
| SoundFont CTK_533_Piano1_00_Casio | 39 / 668 | 0.961 | 0.871 | 0.885 | 0.867 |
| SoundFont Casio_CA_110_Piano_FM | 77 / 668 | 0.995 | 0.991 | 0.992 | 0.971 |
| SoundFont Casio_PX_860_Concert_Grand_Piano | 78 / 668 | 0.895 | 0.845 | 0.835 | 0.741 |
| SoundFont JV1080_Nice_Piano_m | 78 / 668 | 0.968 | 0.912 | 0.953 | 0.875 |
| SoundFont U20PIANO | 44 / 668 | 0.940 | 0.891 | 0.910 | 0.835 |
| SoundFont YAMAHA_PSR400_500 | 75 / 668 | 0.977 | 0.936 | 0.928 | 0.867 |
| SoundFont Yamaha_PSRF50 | 78 / 668 | 0.984 | 0.957 | 0.964 | 0.893 |
| SoundFont Yamaha_S90_ES_Natural_Grand | 78 / 668 | 0.927 | 0.877 | 0.867 | 0.828 |
| Otey | 120 / 668 | 0.976 | 0.898 | 0.940 | 0.778 |
| MAPS ENSTDkAm vs ENSTDkCl (same piano, two rooms) | 132 / 120 | 0.678 | 0.593 | 0.692 | 0.592 |

Medians by kind: free libraries: all47 0.833 (channel 0.781), med16 0.810; MAPS (both rooms): all47 0.843 (channel 0.824), med16 0.832; CCMusic piano vs other 7: all47 0.871 (channel 0.802), med16 0.853; SoundFonts: all47 0.972 (channel 0.924), med16 0.931; Otey: all47 0.976 (channel 0.898), med16 0.940.

**Reading notes.** (i) A recorded group that fills a fold alone is trained on exactly the same data in every fold assignment, so its row can show zero spread over seeds (Salamander, 480 notes, all-layers variant). (ii) Section 4a uses the detector trained with every collection labelled recorded, so values far from 0.5 mean the detector still ranks one recorded collection as more (or less) synthetic-looking than CCMusic. (iii) Section 4b is a different question -- can ANY classifier tell a collection apart from CCMusic? -- and its scale is set by the reference rows: a different CCMusic piano through the same channel, and the same Disklavier in a different room, are themselves separable well above 0.5, so a library's AUC must be read against those rows, not against 0.5.

## 5. Files

`v4_runs.csv` (every CV run x test: all metrics), `v4_collection_signature.csv`, `v4_scores_seed0_all.csv` (per-note P, seed 0, all-layers variant), `key_numbers_v4.json`, `cache/` (maps_orig.csv, iso_aug.csv).
