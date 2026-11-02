"""What the performance detector uses when the digital pianos are in training (Sec. 4.2).

(A) Feature-group ablation on the performance corpus + the SFZ sampled play (XGBoost, own-fold scoring): condition
    "renders only" (digital pianos never trained) vs "+ digital in training" (each digital piano scored by the model
    of its fold, which did not see it); the SFZ play is never trained.  Groups: A medians, A IQRs, B (register-
    localised fundamental structure), C (tuning), and the decay descriptors (6 x median + IQR).
(B) Single features: direction-free univariate AUC real vs sampled digital pianos, real vs SFZ play, SFZ vs digital,
    and the median of each class.
(C) Level between notes (needs the audio, --with-audio): per file, the 5th percentile of the 50-ms frame RMS
    relative to the 99th (dB) and the share of digital silence (|x| < 1e-6), by class.

Paper numbers (Sec. 4.2):
  - medians in (B), real / sampled digital / SFZ: unison_index 14.9 / 8.0 / 11.7; bass_h2_over_h1 0.578 / 0.0612 /
    0.405, a log10 power ratio, i.e. 5.8 / 0.6 / 4.1 dB (x 10); bright_decay_med -0.005 / -0.26 / -0.006;
  - (A), row "A medians only (16)": sampled digital 0.562 -> 0.844 (AUC 0.56 to 0.84); SFZ 0.570 -> 0.470.

    python experiments/02_sampled_vs_algorithmic/digital_cues.py [--with-audio] [--workers 4]
        -> results/02_sampled_vs_algorithmic/digital_cues.md
"""
import os, sys, time, glob, argparse
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('02_sampled_vs_algorithmic')
ROOT = AUDIO_ROOT
import protocol.common as L  # noqa: E402
import protocol.corpus as V  # noqa: E402
import protocol.corpus_base as P  # noqa: E402
from protocol.corpus import PERF_KINDS   # noqa: E402
from detector.pipeline import DESCRIPTORS, BASS_FEATURE_COLS, TUNING   # noqa: E402


def floor_stats(path):
    from detector.segment import load_mono
    try:
        x, sr = load_mono(path)
    except Exception:
        return path, np.nan, np.nan
    n = int(0.05 * sr)
    fr = x[:len(x) // n * n].reshape(-1, n)
    r = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    return path, float(20 * np.log10(np.percentile(r, 5) / np.percentile(r, 99))), float(np.mean(np.abs(x) < 1e-6))


def main(workers, with_audio=False):
    from multiprocessing import Pool
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    d = df[df.kind.isin(PERF_KINDS)].reset_index(drop=True)
    sfz, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
    sfz = sfz.reset_index(drop=True)
    sfz['grp'] = 'syn:sfz:' + sfz['instrument']
    fmap = L.assign_folds(df.loc[df.base, 'grp'], pd.concat([df.loc[~df.base, 'grp'], sfz['grp']]))
    sfz['fold'] = sfz.grp.map(fmap).astype(int)
    sfz['y'], sfz['kind'], sfz['subkind'] = 1, 'sfz_performance', 'sfz:' + sfz.instrument
    a = pd.concat([d, sfz[['file', 'y', 'kind', 'subkind', 'grp', 'fold'] + list(L.FEATURE_COLS)]], ignore_index=True)
    k, sk = a.kind.values, a.subkind.values
    rp, is_sfz = k == 'real_performance', k == 'sfz_performance'
    dig = np.isin(k, ['sampled_digital_performance', 'modelled_digital', 'm1', 'digital_other', 'sound_module'])
    y, grp = a.y.values, a.grp.values
    A_med = [f'{x}_med' for x in DESCRIPTORS]
    A_iqr = [f'{x}_iqr' for x in DESCRIPTORS]
    Bc, Cc = list(BASS_FEATURE_COLS), list(TUNING)
    dec = [f'{x}_{s}' for x in ('log_t60', 'sustain_db_1s', 'bright_decay', 'hf_decay_excess', 'partial_persist',
                                'centroid_var') for s in ('med', 'iqr')]
    F = list(L.FEATURE_COLS)
    groups = {'all 47': F, 'A medians only (16)': A_med, 'A IQRs only (16)': A_iqr, 'B only (8)': Bc, 'C only (7)': Cc,
              'decay only (12)': dec, 'without decay (35)': [c for c in F if c not in dec],
              'without B': [c for c in F if c not in Bc], 'without C': [c for c in F if c not in Cc]}
    tests = [('sampled digital', k == 'sampled_digital_performance'), ('Roland', k == 'modelled_digital'),
             ('SFZ (never trained)', is_sfz), ('SoundFont', sk == 'sf2_perf')]
    md = ['# What the performance detector uses when YouTube digital pianos are in training\n\n',
          '## (A) Feature-group ablation (AUC vs real performances, own-fold)\n\n'
          '| features | ' + ' | '.join(f'{n}: renders only → + digital in training' for n, _ in tests) + ' |\n'
          + '|---' * (len(tests) + 1) + '|\n']
    for gname, cols in groups.items():
        cells = []
        res = {}
        for cond, tr in (('A', ~is_sfz & ~dig), ('C', ~is_sfz)):
            b = a.copy()
            b['trainable'] = tr
            res[cond] = V.run_cv3(b, cols, 'T1')['own']
        for _, m in tests:
            aa = [V.cmp3(y, res[c], grp, rp, m)['auc'] for c in ('A', 'C')]
            cells.append(f'{aa[0]:.3f} → {aa[1]:.3f}')
        md.append(f'| {gname} | ' + ' | '.join(cells) + ' |\n')
        print(f'  {gname} {time.time() - t0:.0f} s', flush=True)
    # (B) univariate AUCs
    sd = k == 'sampled_digital_performance'

    def uauc(c, m0, m1):
        v0, v1 = a.loc[m0, c].dropna(), a.loc[m1, c].dropna()
        if len(v0) < 5 or len(v1) < 5:
            return np.nan
        u = roc_auc_score(np.r_[np.zeros(len(v0)), np.ones(len(v1))], np.r_[v0, v1])
        return max(u, 1 - u)
    rows = [{'feature': c, 'real_vs_digital': uauc(c, rp, sd), 'real_vs_sfz': uauc(c, rp, is_sfz),
             'sfz_vs_digital': uauc(c, is_sfz, sd),
             'med_real': a.loc[rp, c].median(), 'med_digital': a.loc[sd, c].median(), 'med_sfz': a.loc[is_sfz, c].median()}
            for c in F]
    u = pd.DataFrame(rows).sort_values('real_vs_digital', ascending=False)
    md.append('\n## (B) Single features: separation (AUC, direction-free) and medians, top 15 by real vs digital\n\n'
              '| feature | real vs digital | real vs SFZ | SFZ vs digital | median real | median digital | median SFZ |\n'
              '|---|---|---|---|---|---|---|\n')
    for r in u.head(15).itertuples():
        md.append(f'| {r.feature} | {r.real_vs_digital:.2f} | {r.real_vs_sfz:.2f} | {r.sfz_vs_digital:.2f} | '
                  f'{r.med_real:.3g} | {r.med_digital:.3g} | {r.med_sfz:.3g} |\n')
    open(os.path.join(OUT, 'digital_cues.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md), flush=True)
    if not with_audio:
        return
    # (C) level between notes
    paths = [V.audio_path(f) for f in d.file] + [os.path.join(ROOT, f) for f in
                                                  sorted(os.path.relpath(p, ROOT).replace(os.sep, '/') for p in
                                                         glob.glob(os.path.join(ROOT, 'audio', 'sfz_performances_v2', '*', '*.wav')))]
    cls = list(np.where(d.kind == 'algo_performance', d.subkind, d.kind)) + ['sfz (local)'] * (len(paths) - len(d))
    fl = []
    with Pool(workers, maxtasksperchild=50) as pool:
        for n, r in enumerate(pool.imap_unordered(floor_stats, paths, chunksize=4), 1):
            fl.append(r)
            if n % 200 == 0:
                print(f'  floor {n}/{len(paths)} {time.time() - t0:.0f} s', flush=True)
    order = {p: i for i, p in enumerate(paths)}
    fl.sort(key=lambda r: order[r[0]])
    f = pd.DataFrame(fl, columns=['path', 'floor_db', 'dig_silence'])
    f['cls'] = cls
    md.append('\n## (C) Level between notes (per file; median [IQR] over files)\n\n| class | files | 5th / 99th percentile of '
              '50-ms RMS, dB | share of digital silence |\n|---|---|---|---|\n')
    q = lambda s: f'{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]'
    for c, g in f.groupby('cls'):
        md.append(f'| {c} | {g.floor_db.notna().sum()} | {q(g.floor_db)} | {g.dig_silence.median():.1%} |\n')
    md.append(f'\n{time.time() - t0:.0f} s.\n')
    open(os.path.join(OUT, 'digital_cues.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md[-(len(f.cls.unique()) + 3):]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--with-audio', action='store_true', help='also (C), read from the audio (SPD_AUDIO)')
    a = ap.parse_args()
    main(a.workers, a.with_audio)
