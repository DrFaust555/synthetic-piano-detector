# Screen of the REAL performance class

Scope: all 2576 files of the real performance class of the original corpus
(432 curated manifest files + 2144 automatically collected YouTube files in `audio\brands_extra`, `yamaha_extra`, `seiler_extra`, `hoffmann_extra`, `perzina_extra`, `samick_extra`).
Output: `data/protocol/real_class_exclusions.csv` (one row per file). The screen was done by hand (titles, YouTube metadata fetched with yt-dlp, md5 and an audio fingerprint); there is no script for it. The protocol (protocol/corpus.py) drops the rows with decision `exclude` and groups the files that share a `group_id`.

## Columns of the CSV

| column | content |
|---|---|
| file, video_id | as in the corpus index / manifest |
| title | YouTube title (fetched), else the index title, else the manifest description |
| decision | `exclude` / `keep` / `review` |
| reason | why (category wording + short explanation) |
| evidence | quoted text: `T:` title, `D:` description, `TAGS:`, `CH:` channel, `M:` manifest; for duplicates the hash / fingerprint numbers |
| category | acoustic, acoustic_presumed, player_reproducing, digital_piano, virtual_instrument, silent_hybrid, synthesia_midi, electric_piano, not_piano, not_solo_piano, unverified_instrument, duplicate |
| group_id | `DUPnn` = files sharing audio (keep one / keep all grouped); `VID_<id>` = several files cut from one video (keep, but put in one CV group) |

## Method

1. **Titles**: all 2144 YouTube titles and all 432 manifest rows (description + note) were read by hand. A search-artefact list of "brand" words (Alfred, Becker, Bell, Chase, Clementi, Lindeman, Parsons, Ravenscroft, Strauss, Winter, Stuart, Jacob, Gabler, …; 43 words, 524 files) was handled file by file; other files were kept when the title names the acoustic maker/model, a shop demo, restoration, player piano, etc.
2. **YouTube metadata** (title, channel, description, tags, upload date; yt-dlp, metadata only, 10–14 s between requests after one rate-limit episode): obtained for 1079 of the 2576 files (1048 with metadata, 17 unavailable/private, 14 sign-in only) — all manifest videos and the high-risk part of the extras (titles without an acoustic indicator, covers, tutorials, silent/player/digital keywords, all search-artefact brands up to `pleyel`). For the remaining 1497 extras files (shop demos named after an acoustic model) the decision rests on the title; their evidence says "description not checked".
3. **Duplicates**: md5 of every WAV (byte-identical copies) + same video id + a landmark audio fingerprint over all 2576 files (peak-pair hashes, offset-consistent matching), verified by waveform alignment and per-file coverage. A pair is a duplicate when ≥ 50 % of both files, or ≥ 80 % of the shorter one, carry matching audio; the copy kept is the manifest one, else the containing one, else the earliest upload. Overlaps < 20 s (channel intro jingles: Alamo Music Center, Popplers, Sherwood Phoenix, Merriam, Bonners, Cunningham show) were ignored.

Decision rules: `exclude` only with a quoted source statement (digital / VI / organ / Synthesia / other instrument) or a duplicate; player and reproducing pianos (Spirio, Disklavier, PianoDisc, QRS, Ampico, Welte, rolls) are acoustic → keep; silent/hybrid pianos → review (no clip states that the digital voice is played); amateur / home / lesson / cover / self-published clips with no instrument named → review (`unverified_instrument`); live concerts, recitals, competitions, commercial and historical recordings with no instrument named → keep (`acoustic_presumed`).

## Counts

| category | exclude | keep | review | total |
|---|---|---|---|---|
| acoustic | 0 | 2112 | 2 | 2114 |
| acoustic_presumed | 0 | 36 | 2 | 38 |
| player_reproducing | 0 | 134 | 10 | 144 |
| digital_piano | 13 | 0 | 0 | 13 |
| virtual_instrument | 20 | 0 | 5 | 25 |
| synthesia_midi | 5 | 0 | 11 | 16 |
| electric_piano | 2 | 0 | 1 | 3 |
| silent_hybrid | 0 | 0 | 22 | 22 |
| not_piano (organ, harmonium, lecture, stock music, …) | 12 | 0 | 9 | 21 |
| not_solo_piano (voice, clarinet, horn, drums, orchestra …) | 18 | 0 | 7 | 25 |
| unverified_instrument | 0 | 0 | 120 | 120 |
| duplicate (second copy) | 35 | 0 | 0 | 35 |
| **total** | **105** | **2282** | **189** | **2576** |

By subset: manifest 6 exclude / 5 review / 421 keep; YouTube extras 99 / 184 / 1861.

## Main findings

- **The curated manifest set contains Pianoteq.** `147_petrof_p135_k1_88`, `148_petrof_p127_next_85`, `153_petrof_p125_m1_85` are PETROF COLOURS cover videos whose descriptions say: "In this video is used physically modeled sound of PETROF piano from the sound library of the award-winning virtual instrument PIANOTEQ by Modartt." The coloured uprights are only shown. The two other files from the same series (`154_…_b`, `155_…_b`, same producer) do not state the sound source → review. The B1 list of the code audit missed these because it searched titles only.
- **Manifest duplicates**: `347_glinka_t183_83` = `346_glinka_m185_84` (same 139-s audio uploaded by the maker for two models, r = 0.98); `329_kimball_console_55_c` = `328_kimball_console_58_b`; `219_yamaha_su7_90` is contained in `221_yamaha_su7_90_c`. Two extras files are re-uploads of manifest videos (`hailun_fHREFIzMDW8` ↔ 174, `sei_se138_-z1NIz7veX8` ↔ 215, `steingraeber_BWw-ErU460Q` ↔ 233).
- All 19 "clear" B1 files of the code audit are confirmed and excluded, plus the 2 probable Synthesia videos (both confirmed by title/description). New non-acoustic finds in the extras: Clavinova CLP-545 (`clementi_Tc3xizExH3o`), Garritan CFX rendered from MIDI (`clementi_HRDZajVJKDs`), Garritan/VSL tags (`becker_e7nYslQOxvI`), Becker BAP-62 digital review (`becker_l0r_rsH3jwI`), pianola on a Yamaha electronic piano (`aeolian_gfo6z74qbd8`), keyboard-to-computer recording, Roland-owning teacher demo, Yamaha CP88 "SU7" voice, PianoArc (2), Ravenscroft VI demos from MIDI / VI Labs (2), Rousseau / pianthesia / MuseScore MIDI tutorials, Schiedmayer Elektrolesta, 12 organ/harmonium clips (Estey, Gabler organ, Spaethe harmonium, Melville Clark Orpheus reed organ), 18 clear ensemble recordings (voice, clarinet, cello, sax, horn, flute, violin, drums, orchestra).
- **Byte-identical copies**: the 4 pairs of the audit (DSYTzNCbHac, KabAlEs9qcc, Fazioli Ltgy5uN_OqA/W-AaaICEO_w, Currier hTFHWsXA6Ic/UNpfRZqFe8w). The fingerprint found 31 further shared-audio pairs (re-uploads, re-cuts, the same demo posted for two models: Schiller 50H/52H, Jesse French console / Starck grand, Glinka M-185/T-183). In total 35 second copies are excluded; one more group (DUP13, Cunningham Piano Show episodes sharing ~30 s) is kept but must be grouped.
- **Multi-piano videos** (keep, group by `group_id`): VID_k7XMMZiVaV0 (12 files, 4 folds), VID_wdRekUGLQvM (11, 4), VID_T2GYYV8JSqM (9, 3; Vienna Synchron Stage, real pianos), VID_pf5xtwmctQA (6, 4), VID_ApGXujZVWf4 (5, 3), VID_9BiEslDFfcs (4, 2), VID_TRQd8LoC_14 (4, 2), VID_O0CE6CFW8WY (3, 2), VID_gJE7cd_eGac (3, 3), VID_fihRDTx17I0, VID_hsy1D9ZJBqQ, VID_-1MTYsE6T6M, VID_c0eUdh2GPig, VID_ctnO5SNf8Lw (2 files, 2 folds each), plus 11 same-video before/after-tuning or split pairs that are already in one fold. Single files that contain several pianos: `sei_se138_KabAlEs9qcc` (5 pianos), `stuart_j41dSytVR6w` (two pianos).
- The silent-piano question is open for 22 files; one silent-titled extra (`yam_b3_erZ-9r4VA7w`) is the same audio as the non-silent demo `yam_b3_2lk2uajZ0og` (acoustic) and is excluded as a duplicate.

## Review cases (189)

- **silent_hybrid (22)**: manifest 220, 222, 278; albert_uePiO_Do8NI, brinsmead_LZvFSvfy5a8, julius_TbeDV-8w2Ng, kawai_Cyc-aRQyhpw, kemble_PYQCt5KzP-w, schimmel_MIstzXeb7hw, zimmermann_tb3w7zMyWTA, sam_js121_ODp75TGb7LM, sei_gs118_tL6FD7n7eLQ, sei_se122_9V7qZijAZSg, yam_b2_mfbGn25_5Ro, yam_b3_Dkk3z9U-pZU, yam_b3_DnPZyAVYjPo, yam_b3_oA9aMN6EET4, yam_p114_4xKqGLFcv4c, yam_p114_7Hefz7QUltc, yam_su7_KzP1DMeTYYg, yam_yus1_HNa_DvlqBzY, yam_yus1_S0dHIYNFrek.
- **virtual_instrument suspected (5)**: manifest 154, 155 (PETROF COLOURS); ravenscroft_src7LVM2ito ("Room Mics"); haddorf_mK7vRc4Xugg, haddorf_ws9UlkHnRE0 (ambient album, tag "synthesizer").
- **synthesia/MIDI suspected (11)**: the 10 HotMelody "LINDEMANN" karaoke/tutorial covers (MIDI files sold), winter_s_PHTarArkM (sheet-music scrolling).
- **player with possible backing track / rendering (10)**: edelweiss_INo41Y9cfiA, edelweiss_q4GO5HuGt3c, george_CIxBSKkXt8s, george_yMvhHcH46SY, lauter_W_OIYGDbGtE, melville_m26dDftZBas (phonograph), schaaf_0JBUrDFrnXY, starr_BnzWa33FjWo, starr_HilqrevjdPg, starr_Q1sLfGJW9Ro.
- **electric (1)**: lesage_K3CbZUkufgU ("Acoustic Electric Piano").
- **not piano / not solo, unconfirmed (16 + 4)**: aeolian_TDm8_ohatuY, beale_q799wuhbEq8, chase_T0UlGqwC73U, estey_Gq2IFi1FARU, julius_RWyentrq1-E, parsons_tc_4j8alBhA, strauss__b8fLId6t2c, voigt_JnIYdM2CQ60, winter_6El8B8hJ4Sg, carl_WWRPeMpgh4w, gabler_tWSGbwFKR4Y, haddorf_89mKlNyi6_4, jacob_EV2JaN21JT8, steger_o_fZ-tY9z5k, strauss_2RwSMQSlSEY, voigt_sfVZEu0qCwg; acoustic but possibly with singing/others: falcone_19v3XfwoCG0, sterling_dGJtgVbnCyU, starr_QN2pVueFuBQ, starr_wi3Zsci7Tkk.
- **unverified_instrument (120)**: amateur/home/lesson/cover/self-published clips with no instrument named (e.g. 10 Alfred method-book clips, 11 Baus clips incl. 7 "BAUSZANSKY" sonatas, Amosdoll, Andrew Currier, ingopiano, Rousseau-like covers, Strauss karaoke accompaniments, Gary Becker, Borgato "Electro"-tagged Gnossiennes, "Universal Audio SP1 … Fazioli F308", "Perfect Wurlitzer Piano For Beginners"). The full list is the `review` rows with this category in the CSV.

A listen (or a look at the video) resolves almost all review rows; for a conservative corpus drop `exclude` + `review` (294 files, 11 %).

## Limitations

- Descriptions of 1497 extras files (titles naming an acoustic model, mostly dealer demos) were not fetched because YouTube rate-limited the session; these rows say "description not checked". The PETROF case shows that a sound-source note can hide in the description of an acoustically titled video, so a complete description pass (≈ 4 h at the polite rate) would be the next step.
- No listening was done. The excerpt actually used (`kept_frac`) may avoid the problematic part of a mixed video.
- The fingerprint finds shared recordings, not the same piano recorded twice; files of one dealer/piano in different videos remain separate groups.
