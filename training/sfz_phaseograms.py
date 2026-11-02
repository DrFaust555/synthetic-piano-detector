"""Phaseograms of the SFZ sample-library performance renders (audio/sfz_performances_v2, 480 files) for the
performance CNN: segmentation detector.segment.build_segments as for every performance, 15 notes by
detector.pipeline.select_diverse_notes.  Each library is a new group, assigned to a fold with the protocol's rule;
the 47 features are those of data/performances/sfz_perf_features.csv.  Needs the audio.

Outputs (results/cnn/cache_perf/): meta_sfz.csv (same columns as meta_perf.csv; trainable = False; shipped as
data/predictions/perf_meta_sfz.csv), feat47_sfz.npy, phaseograms_sfz.npy (N, 15, 48, 256, float16), phaseogram_mask_sfz.npy.
    python training/sfz_phaseograms.py [--workers 8]
"""
import os, sys, glob, time, argparse
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
ROOT = AUDIO_ROOT
OUT = os.path.join(RESULTS, 'cnn', 'cache_perf')
K = 15


def one(rel):
    import detector.segment as S
    from detector.phaseogram import compute_phaseogram
    from detector.pipeline import select_diverse_notes
    ph, m = np.zeros((K, 48, 256), np.float16), np.zeros(K, np.float32)
    try:
        x, sr = S.load_mono(os.path.join(ROOT, rel))
        segs, _ = S.build_segments(x, sr)
        segs = [s for s in segs if 24.0 < s.f0 < 4500.0]
        idx = select_diverse_notes(segs)[:K]
        for j, i in enumerate(idx):
            ph[j] = compute_phaseogram(segs[i].x, sr, segs[i].f0).astype(np.float16)
            m[j] = 1.0
        return rel, ph, m, len(idx), float(np.mean([segs[i].midi <= 48 for i in idx])) if idx else np.nan, ''
    except Exception as e:
        return rel, ph, m, 0, np.nan, repr(e)[:200]


def main(workers):
    from multiprocessing import Pool
    import protocol.corpus as V
    import protocol.common as L
    import protocol.corpus_base as P
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.path.relpath(p, ROOT).replace(os.sep, '/') for p in
                   glob.glob(os.path.join(ROOT, 'audio', 'sfz_performances_v2', '*', '*.wav')))
    assert len(files) == 480, len(files)
    feat, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
    feat['base'] = [os.path.basename(f) for f in feat.file]
    feat = feat.drop_duplicates('base').set_index('base')
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    lib = [os.path.basename(os.path.dirname(f)) for f in files]
    grp = pd.Series(['syn:sfz:' + l for l in lib])
    fmap = L.assign_folds(df.loc[df.base, 'grp'], pd.concat([df.loc[~df.base, 'grp'], grp]))
    with Pool(workers, maxtasksperchild=100) as pool:
        res = {r[0]: r for r in pool.imap_unordered(one, files, chunksize=4)}
    ph = np.stack([res[f][1] for f in files])
    mk = np.stack([res[f][2] for f in files])
    np.save(os.path.join(OUT, 'phaseograms_sfz.npy'), ph)
    np.save(os.path.join(OUT, 'phaseogram_mask_sfz.npy'), mk)
    X = feat.reindex([os.path.basename(f) for f in files])[list(L.FEATURE_COLS)].values.astype(np.float32)
    np.save(os.path.join(OUT, 'feat47_sfz.npy'), X)
    meta = pd.DataFrame({'file': files, 'rel': files, 'y': 1, 'kind': 'sfz_performance',
                         'subkind': ['sfz:' + l for l in lib], 'grp': grp.values, 'fold': grp.map(fmap).astype(int).values,
                         'trainable': False, 'n_phaseograms': [res[f][3] for f in files], 'bass_fraction': [res[f][4] for f in files],
                         'error': [res[f][5] for f in files]})
    meta.to_csv(os.path.join(OUT, 'meta_sfz.csv'), index=False)
    print(f'{len(files)} files, errors {int((meta.error != "").sum())}, features missing {int(np.isnan(X).all(1).sum())}; '
          f'notes per file median {meta.n_phaseograms.median():.0f}; folds {meta.groupby("subkind").fold.first().to_dict()}; '
          f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args().workers)
