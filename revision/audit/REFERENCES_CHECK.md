# References check: manuscript_v3_short (assembled list, 35 entries)

Checked: 2026-10-04, independent checker. List checked: the assembled manuscript (`manuscript_v3_short_build.md`) (section "References", numbers after renumbering). New entries come from `build\build_v3_short.py` NEW_REFS (old numbers 28–39; 31 Lundberg and 35 Pollard–Jansson are in DROPPED and do not appear). The other entries come from `manuscript_v1_submitted.md`. Nothing in the manuscript or the build scripts was changed.

Sources used: the Crossref REST API (all 26 DOIs, `api.crossref.org/works/<doi>`), PubMed eSummary (Traer 2016, Rauhala 2007), arXiv abstract pages (1810.12247, 2305.07499, 2310.04722), OpenAlex (Lee & Horner 1999, Bensa et al. 2000), the University of Copenhagen research portal (Bensa et al. 2000), the publisher landing page for the engrXiv preprint, the GitHub API and the TinySoundFont source, Internet Archive item metadata (`archive.org/metadata/<id>`), the Iowa EMS and FreePats pages, the KVR Audio forum thread (Salamander V3), the Hugging Face card `ccmusic-database/pianos`, and the MIT IR Survey page. Every DOI in the list resolves, and Crossref returns the cited title, authors, volume, issue and pages, except where a row below says otherwise.

## Короткий підсумок українською

Усі 35 джерел існують, і всі DOI відкриваються. Помилки, які треба виправити:
1. **[28] TinySoundFont — автор указаний неправильно.** Правильно Bernhard **Schelling**, а не «Schellekens». Ліцензія MIT, перший випуск 2017 року, поточна версія v0.9.
2. **[16] Zhou et al. — неправильний рік.** Статтю опубліковано у 2024 році (онлайн 08.11.2024; друкований том LNEE 1268 вийшов у 2025). 2023 — це рік конференції CSMT. Крім того, курсив у записі стоїть не на тому місці.
3. **[23] Salamander — неправильний рік.** V3 випущено 2011 року (KVR, 03.10.2011; дата позиції в архіві 12.08.2011). 2015 — це лише рік завантаження в Internet Archive.
4. **[20] Morocutti et al.** — бракує сторінок і DOI: EUSIPCO 2023, pp. 176–180, https://doi.org/10.23919/EUSIPCO58844.2023.10289983.
5. **[19] ASVspoof 2021** — бракує сторінок: pp. 47–54.
6. **[9] Rauhala et al.** — у 2007 році «JASA Express Letters» ще не був окремим журналом. Правильна назва видання: *JASA* 121(5):EL184–EL189.
7. **Скорочені назви, які треба подати повністю:** [3] Chaigne & Askenfelt, [12] Timbre Toolbox, [13] openSMILE, [14] McAdams et al.
8. **[5] Geirhos et al.** — бракує номера випуску: 2(11).
9. **[26] FreePats** — на сторінці набір називається «Honky-tonk Piano» (банк «Old Piano FB», версія 2020-04-01). Записав його Piotr Barcz, ліцензія CC0 1.0. Назву в записі варто узгодити зі сторінкою.
10. **[29] MAESTRO** — бракує URL. Варто додати OpenReview або arXiv.
11. **[22] Otey** — сторінка доступна (лише по http). Повне ім'я автора — Clayton Otey, дата — 12.12.2007. Для стабільності варто додати знімок з archive.org.

Невідповідності між твердженнями і джерелами:
- **«The generic eGeMAPS set [13]»** — набір eGeMAPS визначено у статті Eyben et al. 2016 (IEEE TAC). Стаття про openSMILE 2010 року його ще не описує. Варто додати Eyben 2016 або написати «computed with openSMILE [13]».
- **[20] не підтримує «augmentation with room responses».** Morocutti et al. згортають сигнал з імпульсними характеристиками *пристроїв* (мікрофонів), а не кімнат.
- **[11] Bensa et al. 2000** — твердження «mainly by the attack» сильніше, ніж у поданій версії («errors concentrated in the attack»). Повний текст статті ICMC перевірити не вдалося (quod.lib.umich.edu блокує автоматичний доступ). Формулювання варто звірити з самою статтею.
- **«270 measured rooms [34]»** — в опитуванні MIT 271 IR. У завантаженому архіві, за `analysis\v3_controls.py`, 270 файлів. Розбіжність незначна, але точніше написати «270 measured room impulse responses».

Номер e-library AES для [10] (12114) перевірити не вдалося, бо aes.org повертає 403 на автоматичні запити. Бібліографічні дані підтверджено: JAES 47(3):101–111, березень 1999.

## Per-reference results

Status: **V** = verified as listed; **C** = correction needed (corrected entry given); **P** = a problem that needs a decision by the authors; **U** = could not be fully verified online.

| # | Status | Finding |
|---|---|---|
| 1 | V | Bank, Chabassier. IEEE SPM 36(1):103–114, Jan. 2019. DOI resolves. |
| 2 | V | Renault, Mignot, Roebel. JAES 71(9):552–565, 2023. DOI resolves. |
| 3 | C | Correct data, but the title is truncated. Full title: "Numerical simulations of piano strings. I. A physical model for a struck string using finite difference methods." |
| 4 | V | Bank, Zambon, Fontana. IEEE TASLP 18(4):809–821, May 2010. |
| 5 | C | Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann (7 authors, so "et al." is fine). Issue 11 is missing: 2(11):665–673. |
| 6 | V | Daniel Bordovský, sole author, engrXiv preprint v1 posted 2026-06-25, DOI 10.31224/7432 (CC BY 4.0). The title matches exactly. The content (a ResNet-18 detector of AI-generated piano music that relies on loudness and an ~18 kHz cutoff shortcut) supports the claims "learn the channel" and "shown for piano". |
| 7 | V | Fletcher, Blackham, Stratton. JASA 34(6):749–761, 1962. |
| 8 | V | Conklin. JASA 100(3):1286–1298, 1996. |
| 9 | C | Data are correct (121(5):EL184–EL189, May 2007), but the venue is *JASA* (Express Letters section). The separate journal *JASA Express Letters* starts in 2021. |
| 10 | U | Kathy Lee and Andrew Horner, JAES 47(3):101–111, March 1999 (OpenAlex/HKUST record; the abstract matches). The AES e-library ID 12114 could not be opened (aes.org returns 403 to automated requests). Check the link once in a browser. |
| 11 | U/P | Bensa, Jensen, Kronland-Martinet, Ystad. Proc. ICMC 2000 (Berlin), pp. 58–61 (University of Copenhagen record). The handle `hdl.handle.net/2027/spo.bbp2372.2000.124` redirects to the cited quod URL, so the item ID is right, but the full text could not be read. See the claim check below. |
| 12 | C | Correct data. Full title: "The Timbre Toolbox: Extracting audio descriptors from musical signals." |
| 13 | C | Correct data. Full title: "openSMILE: The Munich Versatile and Fast Open-Source Audio Feature Extractor." Proc. 18th ACM Int. Conf. Multimedia (MM '10), pp. 1459–1462. |
| 14 | C | Correct data. Full title: "Perceptual scaling of synthesized musical timbres: Common dimensions, specificities, and latent subject classes." |
| 15 | V | Cooper, Huang, Toda, Yamagishi. ICASSP 2022, pp. 8442–8446. |
| 16 | C | Authors, title, LNEE vol. 1268 and pp. 3–17 are correct. Year: published online 2024-11-08, print volume 2025, so cite as 2024 (CSMT 2023 is the conference). The italics cover the whole venue string; the list style italicizes only the venue. |
| 17 | V | Wang et al. (40 authors), CSL 64:101114, Nov. 2020. |
| 18 | V | Jung et al. (8 authors), ICASSP 2022, pp. 6367–6371. |
| 19 | C | Yamagishi et al. (11 authors). Pages are missing: pp. 47–54. |
| 20 | C | Morocutti, Schmid, Koutini, Widmer. EUSIPCO 2023, pp. 176–180, DOI 10.23919/EUSIPCO58844.2023.10289983 (arXiv:2305.07499). Pages and DOI are missing. See also the claim check below. |
| 21 | V | Zhou, Xu, Liu, Wang, Yu, Li, Han. TISMIR 8(1):22–38, 2025. The dataset card confirms the "8_class" subset (7 pianos + Yamaha upright, 668 files) used in Table 1. |
| 22 | V | The page exists (HTTP 200 on http; https does not answer). It reads "A Physical Model of the Piano for Sound Synthesis", C. Otey, December 12, 2007, "Submitted as coursework for Physics 210, Stanford University, Fall 2007". It is also listed on the course index `large.stanford.edu/courses/2007/ph210/` (Clayton Otey). Wayback snapshots exist (2015, 2024). |
| 23 | C | Alexander Holm, Yamaha C5, 16 velocity layers, CC BY 3.0: all verified. The year is wrong. The IA item date is 2011-08-12, V3 was announced on KVR on 2011-10-03, and the IA upload was 2015-02-18. Use 2011. |
| 24 | V | Steinway & Sons model B, recorded 5 and 27 Nov. 2001, Iowa EMS (project director Lawrence Fritts). The page gives no licence ("freely available"); the entry states none, which is correct. |
| 25 | V | Rudi Fiasco, 2016-09-06, CC BY 4.0, 6 velocity layers, SFZ + 212 WAV, upright: all verified on the IA metadata. |
| 26 | C | The page title is "Honky-tonk Piano" (bank "Old Piano FB", version 2020-04-01), recorded by Piotr Barcz from a Francis Bacon player piano, CC0 1.0, SFZ/SF2 in FLAC/WAV. Align the entry title with the page. |
| 27 | V | Kong, Cao, Iqbal, Wang, Wang, Plumbley. IEEE/ACM TASLP 28:2880–2894, 2020. |
| 28 | C | **Wrong author.** The author is Bernhard Schelling (GitHub `schellingb`; LICENSE "Copyright (C) 2017-2023 Bernhard Schelling"; MIT; based on SFZero by Steve Folta). Repository created 2017; header says v0.9. |
| 29 | C | Hawthorne, Stasyuk, Roberts, Simon, Huang, Dieleman, Elsen, Engel, Eck (9 authors, so "et al." is fine). ICLR 2019 (oral). The entry lacks a URL; add OpenReview and arXiv. |
| 30 | V | Conklin. JASA 105(1):536–545, Jan. 1999. The abstract is about phantom partials. |
| 31 | V | Bank, Sujbert. JASA 117(4):2268–2278, Apr. 2005. |
| 32 | V | Weinreich. JASA 62(6):1474–1484, Dec. 1977. |
| 33 | V | Chen, Guestrin. KDD '16, pp. 785–794. |
| 34 | V | Traer, McDermott. PNAS 113(48):E7856–E7865, 2016 (PubMed). The MIT IR Survey page lists 271 IRs. |
| 35 | V | Emiya, Badeau, David. IEEE TASLP 18(6):1643–1654, Aug. 2010. The MAPS database (with Yamaha Disklavier recordings) is introduced in this paper. |

## Corrected entries (same style as the list)

Only entries that change are given.

3. Chaigne and Askenfelt (1994). "Numerical simulations of piano strings. I. A physical model for a struck string using finite difference methods." *JASA* 95(2):1112–1118. https://doi.org/10.1121/1.408459

5. Geirhos et al. (2020). "Shortcut Learning in Deep Neural Networks." *Nature Machine Intelligence* 2(11):665–673. https://doi.org/10.1038/s42256-020-00257-z

9. Rauhala, Lehtonen, and Välimäki (2007). "Fast automatic inharmonicity estimation algorithm." *JASA* 121(5):EL184–EL189. https://doi.org/10.1121/1.2719043

10. Lee and Horner (1999). "Modeling Piano Tones with Group Synthesis." *JAES* 47(3):101–111. https://www.aes.org/e-lib/online/browse.cfm?elib=12114 *(data verified; check the e-lib ID once in a browser)*

11. Bensa, Jensen, Kronland-Martinet, and Ystad (2000). "Perceptual and Analytical Analysis of the Effect of the Hammer Impact on the Piano Tones." *Proc. ICMC 2000*, pp. 58–61. https://hdl.handle.net/2027/spo.bbp2372.2000.124 *(the handle is optional; the quod URL is valid)*

12. Peeters, Giordano, Susini, Misdariis, and McAdams (2011). "The Timbre Toolbox: Extracting Audio Descriptors from Musical Signals." *JASA* 130(5):2902–2916. https://doi.org/10.1121/1.3642604

13. Eyben, Wöllmer, and Schuller (2010). "openSMILE: The Munich Versatile and Fast Open-Source Audio Feature Extractor." *Proc. ACM MM 2010*, pp. 1459–1462. https://doi.org/10.1145/1873951.1874246

14. McAdams, Winsberg, Donnadieu, De Soete, and Krimphoff (1995). "Perceptual scaling of synthesized musical timbres: Common dimensions, specificities, and latent subject classes." *Psychological Research* 58(3):177–192. https://doi.org/10.1007/BF00419633

16. Zhou, Wu, Ji, Li, and Li (2024). "A Holistic Evaluation of Piano Sound Quality." *Proc. CSMT 2023*, LNEE vol. 1268, pp. 3–17 (Springer). https://doi.org/10.1007/978-981-97-7962-8_1 (arXiv:2310.04722)

19. Yamagishi et al. (2021). "ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection." *Proc. ASVspoof 2021 Workshop*, pp. 47–54. https://doi.org/10.21437/ASVSPOOF.2021-8

20. Morocutti, Schmid, Koutini, and Widmer (2023). "Device-Robust Acoustic Scene Classification via Impulse Response Augmentation." *Proc. EUSIPCO 2023*, pp. 176–180. https://doi.org/10.23919/EUSIPCO58844.2023.10289983 (arXiv:2305.07499)

22. Otey (2007). "A Physical Model of the Piano for Sound Synthesis." Coursework for Physics 210, Stanford University, Dec. 12, 2007. http://large.stanford.edu/courses/2007/ph210/otey2/ (archived: https://web.archive.org/web/20240414231232/http://large.stanford.edu/courses/2007/ph210/otey2/)

23. Holm (2011). "Salamander Grand Piano V3." Yamaha C5 grand, 16-velocity isolated-note sample library (CC BY 3.0), Internet Archive. https://archive.org/details/SalamanderGrandPianoV3

26. Barcz (2020). "Honky-tonk Piano (Old Piano FB), version 2020-04-01." Francis Bacon player piano, isolated-note WAV/SFZ library (CC0 1.0), FreePats project. https://freepats.zenvoid.org/Piano/honky-tonk-piano.html

28. Schelling (2017). "TinySoundFont." SoundFont2 synthesizer library (v0.9, MIT licence), GitHub. https://github.com/schellingb/TinySoundFont (accessed 2026 Oct.).

29. Hawthorne et al. (2019). "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." *Proc. ICLR 2019*. https://openreview.net/forum?id=r1lYRjC9F7 (arXiv:1810.12247)

Optional, if the eGeMAPS sentence keeps its citation (see below), a new entry:

Eyben et al. (2016). "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing." *IEEE Trans. Affective Computing* 7(2):190–202. https://doi.org/10.1109/TAFFC.2015.2457417

## Do the citations support the claims? (obvious mismatches only)

All citations in `manuscript_v3_short.md` were checked against the renumbered text. The following need attention. The others match their sources: [1–4], [5, 6] (shortcut learning; the piano shortcut study), [7–9] (inharmonicity and scale design), [10] additive/group synthesis, [12–14] timbre descriptors, [15] MOS generalization, [16, 21] CCMusic piano data and CNN quality predictor, [17–19] anti-spoofing including replay, [22] Otey, [27] PANNs sound-event classifier, [28] TinySoundFont, [29] MAESTRO, [30, 31] phantom partials, [32] unison coupling and splitting, [33] XGBoost, [34] measured room IRs, [35] MAPS Disklavier.

1. **Sec. 5 (build line 166): "The generic eGeMAPS set [13]".** eGeMAPS was defined in Eyben et al. 2016 (IEEE TAC 7(2):190–202), six years after the 2010 openSMILE paper. Either add the 2016 entry or write "the eGeMAPS set, computed with openSMILE [13]".
2. **Sec. 2: "Common remedies are device-mismatch benchmarks, spectral correction and augmentation with room responses [20]".** Morocutti et al. augment with *device* (microphone) impulse responses and compare with Freq-MixStyle. Their work supports "device-mismatch" and "impulse-response augmentation", not *room* responses. Suggested wording: "augmentation with device impulse responses [20]". Alternatively, cite [34] next to "room responses".
3. **Sec. 2: "Listeners distinguish such notes from real ones mainly by the attack [11]".** The source is a perceptual and analytical study of hammer-impact effects. The submitted version made a weaker claim ("errors concentrated in the attack"). The full text could not be accessed online, so "mainly" should be confirmed against the paper or softened.
4. **Sec. 4.2: "270 measured rooms [34]".** The MIT IR Survey reports 271 IRs. The downloaded archive holds 270 WAV files (`analysis\v3_controls.py`). The number is defensible, but "270 measured room impulse responses" is more accurate than "rooms".
5. **Minor, optional.** The stiff-string formula is cited to Fletcher et al. 1962 [7]. The 1962 paper uses B, but the standard derivation is Fletcher 1964, "Normal Vibration Frequencies of a Stiff Piano String", JASA 36(1):203–209, https://doi.org/10.1121/1.1918933. No change is required.

## Not checked / out of scope

- The aes.org e-library ID 12114 for [10] (blocked to automated access).
- The full text of [11].
- The version of MAESTRO used (v1/v2/v3) is not stated in the paper or the reference. Consider naming it in the text or in S1.
