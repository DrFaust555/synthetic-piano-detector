"""The recording format as a shortcut (Sec. 4.3): one detector for isolated notes AND performances vs the two
separate detectors, and how well the features tell the format apart.

XGBoost (protocol make_model), the corpus and folds of the paper, T2 (digital pianos, the Korg M1 and the library
notes never trained, scored by the model of their own fold).  Isolated rows: velocity copies removed.
Detectors (each scored own-fold on every row of both formats):
  separate        isolated-only model on the isolated rows / performance-only model on the performance rows;
  joint           one model on both formats, unweighted;
  joint, balanced one model, sample weights so that each (format x class) cell carries 1/4 of the weight;
  transfer        trained on one format only, scored on the other.
Feature sets: 27 (the isolated set, a subset of the 47) and 47.

Paper numbers (Sec. 4.3):
  - the features tell an isolated note from a performance at AUC 0.9998 (27 features, grouped 5-fold);
  - one detector trained on the whole corpus (both formats, 47 features) separates real performances from
    algorithmic single notes at AUC 0.996, against 0.943 for the single-note detector (27 features) on CCMusic vs
    algorithmic single notes (both printed at the end).

    python experiments/03_sampled_limit_and_format/format_shortcut.py
        -> results/03_sampled_limit_and_format/format_shortcut.md
"""
import os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('03_sampled_limit_and_format')
import protocol.common as L  # noqa: E402
import protocol.corpus as V  # noqa: E402


def fit_pred(X, y, w, tr, Xs):
    import warnings
    m = L.make_model('xgb', 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X[tr], y[tr], sample_weight=w[tr])
    return [m.predict_proba(A)[:, 1] for A in Xs]


def library_notes():
    """Isolated notes of the four sample libraries with their 47 features (data/features/library_notes_iso1.csv);
    notes with fewer than 30 defined features or an extraction error are left out."""
    import protocol.corpus_base as P
    libs = P.library_note_files()
    lf = L.extract_cached(list(libs['path']), 'library_notes', isolated=True)
    libs = pd.concat([libs.reset_index(drop=True), lf], axis=1)
    libs['grp'] = 'syn:lib:' + libs['library']
    libs['valid'] = (libs[list(L.FEATURE_COLS)].notna().sum(axis=1) >= 30) & (libs['error'].fillna('') == '')
    return libs[libs['valid']].reset_index(drop=True)


def main():
    import protocol.corpus_base as P
    from protocol.corpus import PERF_KINDS
    t0 = time.time()
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    iso = df[df.kind.isin(['real_isolated', 'algo_single']) & ~df.sf_dup].reset_index(drop=True)
    perf = df[df.kind.isin(PERF_KINDS)].reset_index(drop=True)
    lib = library_notes()
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in lib.grp.unique()}
    lib['fold'] = lib.grp.map(lib_fold).astype(int)
    lib['y'] = 1
    a = pd.concat([iso.assign(fmt='iso'), perf.assign(fmt='perf')], ignore_index=True)
    trainable = ~a.kind.isin(L.SB_KINDS).values
    y, fold, fmt = a.y.values, a.fold.values, a.fmt.values
    cell = pd.Series(list(zip(fmt, y)))
    nc = cell[trainable].value_counts()
    w_bal = np.array([1.0 / (4 * nc[c]) for c in cell]) * trainable.sum()
    spw_all = (y[trainable] == 0).sum() / (y[trainable] == 1).sum()
    w_unw = np.where(y == 1, spw_all, 1.0)                 # = scale_pos_weight with one global ratio
    FS = {'27': list(L.ISO27), '47': list(L.FEATURE_COLS)}
    scores = {}
    for fs, cols in FS.items():
        X = a[cols].values.astype(float)
        Xl = lib[cols].values.astype(float)
        for name, pool, w in (('separate', None, None), ('joint', np.ones(len(a), bool), w_unw),
                              ('joint, balanced', np.ones(len(a), bool), w_bal),
                              ('transfer', None, None)):
            p, pl = np.full(len(a), np.nan), np.full(len(lib), np.nan)
            for k in range(L.N_FOLDS):
                te, tel = fold == k, lib.fold.values == k
                if name in ('separate', 'transfer'):
                    for f_tr in ('iso', 'perf'):
                        tr = trainable & (fold != k) & (fmt == f_tr)
                        ww = np.where(y == 1, (y[tr] == 0).sum() / (y[tr] == 1).sum(), 1.0)
                        pa, pb = fit_pred(X, y, ww, tr, [X, Xl])
                        tgt = (fmt == f_tr) if name == 'separate' else (fmt != f_tr)
                        p[te & tgt] = pa[te & tgt]
                        if (name == 'separate') == (f_tr == 'iso'):
                            pl[tel] = pb[tel]
                else:
                    tr = trainable & (fold != k) & pool
                    pa, pb = fit_pred(X, y, w, tr, [X, Xl])
                    p[te], pl[tel] = pa[te], pb[tel]
            scores[(fs, name)] = (p, pl)
        print(f'  {fs} done {time.time() - t0:.0f} s', flush=True)
    # metrics
    comps = {c[1]: c for c in P._comparisons(perf)}
    n_iso = len(iso)
    grp = a.grp.values
    rows_def = []
    cc = (a.kind == 'real_isolated').values
    for lab, sm in (('iso: SoundFont notes', (a.subkind == 'soundfont_single').values),
                    ('iso: Otey notes', (a.subkind == 'otey_single').values)):
        rows_def.append((lab, lambda p, pl, sm=sm: V.cmp3(y, p, grp, cc, sm)))
    mf = lib.mf_layer.values.astype(bool)

    def lib_cmp(p, pl):
        yy = np.r_[np.zeros(cc.sum()), np.ones(mf.sum())]
        return V.cmp3(yy, np.r_[p[cc], pl[mf]], np.r_[grp[cc], lib.grp.values[mf]], yy == 0, yy == 1)
    rows_def.append(('iso: library notes, mf', lib_cmp))
    for cid in ('P1', 'P2', 'P3'):
        _, _, label, rm, sm = comps[cid]
        rmf, smf = np.r_[np.zeros(n_iso, bool), rm], np.r_[np.zeros(n_iso, bool), sm]
        rows_def.append((f'perf: {cid} {label.replace("real_performance vs ", "")[:40]}',
                         lambda p, pl, rmf=rmf, smf=smf: V.cmp3(y, p, grp, rmf, smf)))
    names = [(fs, n) for fs in FS for n in ('separate', 'joint', 'joint, balanced', 'transfer')]
    md = ['# One detector for both formats vs two separate detectors\n\n',
          f'v3 corpus, own-fold, T2. Training cells (format x class): ' +
          ', '.join(f'{f} {"synthetic" if c else "real"} {n}' for (f, c), n in nc.sort_index().items()) +
          '. Cell: AUC [95% CI]; "transfer" = trained on the other format only.\n\n',
          '| comparison | ' + ' | '.join(f'{n} ({fs})' for fs, n in names) + ' |\n' + '|---' * (len(names) + 1) + '|\n']
    for lab, fn in rows_def:
        md.append(f'| {lab} | ' + ' | '.join(V.fmt3(fn(*scores[k]), det=False) for k in names) + ' |\n')
    for lab, m in (('false alarms, CCMusic notes', cc), ('false alarms, real performances', (a.kind == 'real_performance').values)):
        md.append(f'| {lab} | ' + ' | '.join(f'{np.mean(scores[k][0][m] > 0.5):.1%}' for k in names) + ' |\n')
    # can the features tell the format?
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    for fs, cols in FS.items():
        X = a[cols].values.astype(float)
        f01 = (fmt == 'perf').astype(int)
        oof = np.zeros(len(a))
        for tr, te in GroupKFold(5).split(X, groups=grp):
            oof[te] = fit_pred(X, f01, np.ones(len(a)), np.isin(np.arange(len(a)), tr), [X[te]])[0]
        md.append(f'\nFormat (note vs performance) from the {fs} features: AUC {roc_auc_score(f01, oof):.6f}; misordered pairs {(1-roc_auc_score(f01, oof))*(f01.sum()*(1-f01).sum()):.0f} of {f01.sum()*(1-f01).sum()}; n_perf {f01.sum()} n_iso {(1-f01).sum()}.')
    # one detector for the whole corpus (both formats in training, 47 features, T2): real performances vs
    # algorithmic single notes; and the single-note detector (27 features): CCMusic vs algorithmic notes
    cdf = {c[1]: c for c in P._comparisons(df)}
    r = V.cmp3(df['y'].values, V.run_cv3(df, V.FS3['all47'], 'T2')['own'], df['grp'].values, cdf['C1'][3],
               cdf['C1'][4])
    md.append(f'\n\nFormats mixed (one detector for the whole corpus, 47 features): real performances vs '
              f'algorithmic single notes {V.fmt3(r)}.')
    r = V.cmp3(iso['y'].values, V.run_cv3(iso, V.FS3_ISO['all47'], 'T2')['own'], iso['grp'].values,
               (iso.kind == 'real_isolated').values, (iso.kind == 'algo_single').values)
    md.append(f'\nSingle-note detector (27 features): CCMusic notes vs algorithmic single notes {V.fmt3(r)}.')
    md.append('\n')
    open(os.path.join(OUT, 'format_shortcut.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
