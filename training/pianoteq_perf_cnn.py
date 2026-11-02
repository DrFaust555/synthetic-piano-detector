"""Pianoteq (13 segments of one demo, never trained) through the performance network and fusion trained by
    python training/perf_cnn.py --models cnn_perf,fusion_perf
(weights in results/cnn/jobs_perf, or the folder in SPD_PERF_JOBS): phaseograms as training/sfz_phaseograms.py
(select_diverse_notes, 15 notes), 47 features from data/features/table3_perf_flux.csv.  Scored by the model of the
fold the protocol's rule gives Pianoteq (lightest fold after the SFZ groups); the other four fold models are
reported as a check.  AUC / detected at 5 % false alarms against the out-of-fold scores of the real performances.
Needs the audio.
    python training/pianoteq_perf_cnn.py   -> results/training/pianoteq_cnn_fusion.{md,csv}
(shipped as data/predictions/pianoteq_cnn_fusion.csv)
"""
import os, sys, glob, pickle
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
ROOT = AUDIO_ROOT
JOBS = os.environ.get('SPD_PERF_JOBS', os.path.join(RESULTS, 'cnn', 'jobs_perf'))
from training.sfz_phaseograms import one as phaseo_one


def main():
    import torch
    import training.cnn_fast as FT
    import detector.phaseogram_cnn as rn
    import protocol.common as L
    import protocol.corpus as V
    import protocol.corpus_base as P
    files = sorted(os.path.relpath(p, ROOT).replace(os.sep, '/') for p in
                   glob.glob(os.path.join(ROOT, 'audio', 'pianoteq', 'pianoteq_modartt9_*.wav')))
    res = [phaseo_one(f) for f in files]
    assert not any(r[5] for r in res), [r[5] for r in res]
    ph, mask = np.stack([r[1] for r in res]).astype(np.float32), np.stack([r[2] for r in res])
    pt = L.extract_cached([os.path.join(ROOT, f) for f in files], 'table3_perf_flux', workers=1)
    X = pt[list(L.FEATURE_COLS)].values.astype(np.float32)
    # fold of Pianoteq: the paper's rule
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    sfz, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
    allg = pd.concat([df.loc[~df.base, 'grp'], 'syn:sfz:' + sfz['instrument']])
    fmap = L.assign_folds(df.loc[df.base, 'grp'], allg)
    load = pd.concat([df.loc[df.base, 'grp'], allg]).map(fmap).value_counts().reindex(range(L.N_FOLDS), fill_value=0)
    fpt = int(load.idxmin())
    # real performances: out-of-fold scores of the same runs
    mp = pd.read_csv(os.path.join(RESULTS, 'cnn', 'cache_perf', 'meta_perf.csv'))
    d = df.set_index('file').loc[mp.file].reset_index()
    rp = (d.kind == 'real_performance').values
    data = FT.Data(ph, mask, X, np.ones(len(files), int), device='cpu')
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    md = [f'# Pianoteq through the performance network / fusion\n\n{len(files)} files, notes per file '
          f'{int(np.median([r[3] for r in res]))}; Pianoteq fold {fpt}. AUC vs {rp.sum()} real performances '
          '(real-side-only interval) / detected at 5 % FA.\n\n| model | fold model | AUC | detected | median P |\n|---|---|---|---|---|\n']
    saved, by_fold = {}, {}
    for m in ('cnn_perf', 'fusion_perf'):
        oof = np.full(len(mp), np.nan)
        for k in range(5):
            oof[mp.fold.values == k] = np.load(os.path.join(JOBS, f'{m}_f{k}.npz'))['p'][:len(mp)][mp.fold.values == k]
        sr = lg(oof[rp])
        t5 = np.quantile(sr, 0.95)
        for k in [fpt] + [j for j in range(5) if j != fpt]:
            model = rn.FileModel(X.shape[1], use_analytic=m.startswith('fus'))
            model.load_state_dict(torch.load(os.path.join(JOBS, f'{m}_f{k}.pt'), map_location='cpu'))
            model.eval()
            imp, sc = pickle.load(open(os.path.join(JOBS, f'{m}_f{k}.si.pkl'), 'rb'))
            model._prep = lambda A, imp=imp, sc=sc: torch.as_tensor(sc.transform(imp.transform(A)).astype(np.float32))
            p = FT.predict(model, data, np.arange(len(files)))
            s = lg(p)
            y = np.r_[np.zeros(len(sr)), np.ones(len(s))]
            auc = L.auc_fast(y, np.r_[sr, s])
            lo, hi = L.grouped_boot_ci(y, np.r_[sr, s], np.r_[d.grp.values[rp], ['syn:pianoteq'] * len(s)])
            tag = ' (paper)' if k == fpt else ''
            if k == fpt:
                saved[m] = p
                by_fold[m] = p
            by_fold[f'{m}_f{k}'] = p
            md.append(f'| {m} | {k}{tag} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {np.mean(s > t5):.0%} | {np.median(p):.3f} |\n')
    pd.DataFrame({'file': files, 'fold': fpt, **{f'p_{m}': v for m, v in saved.items()}}).to_csv(
        os.path.join(OUT, 'pianoteq_cnn_fusion.csv'), index=False)
    out = os.path.join(OUT, 'pianoteq_cnn_fusion.md')
    pd.DataFrame({'file': files, 'fold': fpt, **{f'p_{m}': v for m, v in by_fold.items()}}).to_csv(
        os.path.join(os.path.dirname(out), 'pianoteq_by_fold_model.csv'), index=False)
    open(out, 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
