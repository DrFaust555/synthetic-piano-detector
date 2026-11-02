"""Quality of the Table 6 sources computed with the quality tool of Sec. 3.8 (quality_axes/report.py
without the provenance call): timbre = gradient-boosted regressor on the 40 features of Groups A and B
(quality_axes/timbre_model.py), isotonic + affine calibration; overall = timbre - artifact guardrail (aversiveness,
phaser index) - liveness dock.  The regressor is fitted twice at start-up from data/corpus/features_cal.csv: without
any training row of the sources in the table (Boesendorfer recordings, Yamaha CFX recordings, the Korg M1 patch
recordings), so every source is scored out of sample, and on all rows ("full").  Performances as in
table6_provenance.py (Boesendorfer, Yamaha CFX, Pianoteq, a random draw of the Salamander SFZ play, Otey, Korg M1
piano presets).  Per-file rows + per-source medians.

    SPD_AUDIO=<audio root> python experiments/05_provenance_and_quality/table6_quality.py [--sfz-n 30] [--workers 6]
        -> results/05_provenance_and_quality/table6_quality.{csv,md}
"""
import os, sys, re, glob, json, time, argparse
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('05_provenance_and_quality')
ROOT = AUDIO_ROOT
NON_PIANO = re.compile(r'toy|vox|talk|phase|magic|res|stick', re.I)
EXCL = re.compile(r'bosendorfer|yamaha_cfx|korg_m1|^m1_patches/', re.I)    # not yamaha_m1 / petrof_p125_m1: real pianos
_G = {}


def _fit(excl):
    import xgboost as xgb
    import csv
    import quality_axes.timbre_model as E
    rows = [r for r in csv.DictReader(open(E.FEATURES_CAL, encoding='utf-8')) if not (excl and EXCL.search(r['file']))]
    col = lambda c: np.array([float(r[c]) if r.get(c, '') not in ('', 'nan') else np.nan for r in rows])
    M = np.column_stack([col(c) for c in E.TIMBRE_COLS])
    y = np.array([float(r['rating_cal']) for r in rows])
    w = np.array([E.ANCHOR_W if int(r.get('is_anchor', 0) or 0) else 1.0 for r in rows])
    model = xgb.XGBRegressor(**E.XGB).fit(M, y, sample_weight=w)
    return model, E._calibration(model, M, y), len(rows)


def _init():
    from quality_axes.score_files import TRENDS
    _G['excl'], _G['full'] = _fit(True), _fit(False)
    _G['trends'] = json.load(open(TRENDS, encoding='utf-8'))


def one(job):
    src, path = job
    if not _G:
        _init()
    import quality_axes.report as D
    from detector.pipeline import note_raw_descriptors
    from quality_axes.aversiveness import aversiveness_features
    from quality_axes.phaser import phaser_index
    from detector.segment import load_mono
    out = {'source': src, 'file': os.path.relpath(path, ROOT).replace(os.sep, '/')}
    t0 = time.time()
    try:
        _m, notes, tuning = note_raw_descriptors(path)
        timbre = D._timbre_axis(notes, tuning, _G['trends'], *_G['excl'][:2])
        timbre_full = D._timbre_axis(notes, tuning, _G['trends'], *_G['full'][:2])
        live = D._liveness_axis(path)
        x, sr = load_mono(path)
        aver = float(aversiveness_features(x, sr)['aversiveness'])
        ph = float(phaser_index(x, sr))
        guard = max(0.0, (aver - 35.0) / 5.0) + (3.0 if ph > 3.5 else 0.0)        # as deep_report.full_axes
        dock = max(0.0, (30.0 - live) / 30.0 * 10.0) if np.isfinite(live) else 0.0
        out.update(timbre=timbre, timbre_full=timbre_full, liveness=live, aver=aver, phaser=ph, guardrail=guard,
                   live_dock=dock, overall=float(np.clip(timbre - guard - dock, 0.0, 99.0)),
                   overall_full=float(np.clip(timbre_full - guard - dock, 0.0, 99.0)), notes=len(notes))
    except Exception as e:                                       # pragma: no cover
        out['err'] = repr(e)[:150]
    out['sec'] = round(time.time() - t0, 1)
    return out


def main(sfz_n, workers):
    import protocol.corpus as V
    from protocol.corpus import PERF_KINDS
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    d = df[df.kind.isin(PERF_KINDS)]
    jobs = [('Bosendorfer 280VC', os.path.join(ROOT, 'audio', '42_bosendorfer_280vc_97.wav'))]
    for name, pat in [('Bosendorfer 280VC, all recordings', '(?i)bosendorfer_280vc'), ('Yamaha CFX', '(?i)yamaha_cfx')]:
        sel = d.loc[(d.kind == 'real_performance') & d.file.str.contains(pat), 'file']   # excluded by EXCL
        jobs += [(name, V.audio_path(f)) for f in sel]
    jobs += [('Pianoteq', p) for p in sorted(glob.glob(os.path.join(ROOT, 'audio', 'pianoteq', 'pianoteq_modartt9_*.wav')))]
    sal = sorted(glob.glob(os.path.join(ROOT, 'audio', 'sfz_performances_v2', 'Salamander', '*.wav')))
    rng = np.random.default_rng(0)
    jobs += [('Salamander SFZ play', p) for p in (sorted(rng.choice(sal, sfz_n, replace=False)) if sfz_n < len(sal) else sal)]
    jobs += [('Otey', V.audio_path(f)) for f in d.loc[d.subkind == 'otey_perf', 'file']]
    m1 = d.loc[(d.kind == 'm1') & ~d.file.str.replace('m1_patches/m1_', '', regex=False).str.contains(NON_PIANO), 'file']
    jobs += [('Korg M1 piano presets', V.audio_path(f)) for f in m1]
    jobs += [('single file: ' + k, os.path.join(ROOT, p)) for k, p in [
        ('Bosendorfer 280VC', 'audio/42_bosendorfer_280vc_97.wav'), ('Pianoteq', 'audio/pianoteq/pianoteq_modartt9_01.wav'),
        ('Salamander C3 v25 note', 'modelSamples/Salamander_Yam/C3_v25_p0.wav'), ('Otey C4 v50 note', 'otey/out_full/C4_v50.wav'),
        ('Korg M1 Toy-Piano', 'audio/m1_patches/m1_M09-03_Toy-Piano.wav')]]
    print(f'{len(jobs)} files', flush=True)
    from multiprocessing import Pool
    with Pool(workers, initializer=_init) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            recs.append(r)
            if i % 10 == 0:
                print(f'{i + 1}/{len(jobs)} {r["source"]} {r.get("overall", r.get("err"))} {r["sec"]} s', flush=True)
    _init()
    t = pd.DataFrame(recs)
    t.to_csv(os.path.join(OUT, 'table6_quality.csv'), index=False)
    write_md(t)


def write_md(t):
    md = ['# Table 6 sources: quality with the current tool\n\n'
          f'Timbre regressor: "out of sample" = trained without the sources ({_G["excl"][2]} rows); '
          f'"full" = all {_G["full"][2]} rows.\n\n'
          '| Source | n | overall, out of sample: median [IQR] | overall, full regressor | liveness | guardrail | live dock |\n'
          '|---|---|---|---|---|---|---|\n']
    for s, g in t.groupby('source', sort=False):
        ok = g.dropna(subset=['overall'])
        q = lambda c: f'{ok[c].median():.0f}' if len(ok) else 'n/a'
        md.append(f'| {s} | {len(ok)} | {q("overall")} [{ok.overall.quantile(.25):.0f}-{ok.overall.quantile(.75):.0f}] | '
                  f'{q("overall_full")} | {q("liveness")} | {q("guardrail")} | {q("live_dock")} |\n')
    open(os.path.join(OUT, 'table6_quality.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sfz-n', type=int, default=30)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    main(a.sfz_n, a.workers)
