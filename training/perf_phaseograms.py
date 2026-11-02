"""Phaseograms of the PERFORMANCES (real performances, SoundFont / Otey performance renders, digital pianos, M1,
SC-55, GP-3) for the file-level CNN (needs the audio).  Segmentation detector.segment.build_segments (performance
mode); 15 notes per file chosen by detector.pipeline.select_diverse_notes (octave bins from the lowest, loudest
first) -- the note selection of the paper's performance network.

Outputs (results/cnn/cache_perf/): meta_perf.csv (file, rel, y, kind, subkind, grp, fold, trainable = not excluded
from training under T2, n_phaseograms, bass_fraction, error; shipped as data/predictions/perf_meta.csv), feat47.npy (N, 47; the
corpus features), phaseograms.npy (N, 15, 48, 256, float16) and phaseogram_mask.npy (N, 15).

    python training/perf_phaseograms.py [--workers 8]
"""
import os, sys, time, argparse
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
    """15 phaseograms of one performance (select_diverse_notes over its segments) + mask and note counts."""
    import detector.segment as S
    from detector.phaseogram import compute_phaseogram, C, P
    from detector.pipeline import select_diverse_notes
    ph, m = np.zeros((K, C, P), np.float16), np.zeros(K, np.float32)
    info = {'file': rel, 'n_phaseograms': 0, 'bass_fraction': np.nan, 'error': ''}
    try:
        x, sr = S.load_mono(os.path.join(ROOT, rel))
        segs, _ = S.build_segments(x, sr)
        segs = [s for s in segs if 24.0 < s.f0 < 4500.0]
        idx = select_diverse_notes(segs)[:K]
        for j, i in enumerate(idx):
            ph[j] = compute_phaseogram(segs[i].x, sr, segs[i].f0).astype(np.float16)
            m[j] = 1.0
        info['n_phaseograms'] = len(idx)
        info['bass_fraction'] = float(np.mean([segs[i].midi <= 48 for i in idx])) if idx else np.nan
    except Exception as e:
        info['error'] = repr(e)[:200]
    return info, (ph, m)


def main(workers):
    from multiprocessing import Pool
    import protocol.corpus as V
    import protocol.common as L
    from protocol.corpus import PERF_KINDS
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    d = df[df.kind.isin(PERF_KINDS)].reset_index(drop=True)
    d['rel'] = [os.path.relpath(V.audio_path(f), ROOT).replace(os.sep, '/') for f in d.file]
    d['trainable'] = ~d.kind.isin(L.SB_KINDS)
    N = len(d)
    ph = np.lib.format.open_memmap(os.path.join(OUT, 'phaseograms.npy'), 'w+', np.float16, (N, K, 48, 256))
    mk = np.zeros((N, K), np.float32)
    pos = {r: i for i, r in enumerate(d.rel)}
    infos = []
    with Pool(workers, maxtasksperchild=200) as pool:
        for n, (info, out) in enumerate(pool.imap_unordered(one, d.rel.tolist(), chunksize=4), 1):
            i = pos[info['file']]
            ph[i], mk[i] = out
            infos.append(info)
            if n % 250 == 0:
                print(f'  {n}/{N} {time.time() - t0:.0f} s', flush=True)
    ph.flush()
    np.save(os.path.join(OUT, 'phaseogram_mask.npy'), mk)
    np.save(os.path.join(OUT, 'feat47.npy'), d[list(L.FEATURE_COLS)].values.astype(np.float32))
    inf = pd.DataFrame(infos).set_index('file').loc[d.rel].reset_index()
    meta = pd.concat([d[['file', 'rel', 'y', 'kind', 'subkind', 'grp', 'fold', 'trainable']], inf.drop(columns=['file'])], axis=1)
    meta.to_csv(os.path.join(OUT, 'meta_perf.csv'), index=False)
    print(f'{N} files, errors {int((meta.error.fillna("") != "").sum())}; notes per file median '
          f'{meta.n_phaseograms.median():.0f}; bass share median {meta.bass_fraction.median():.0%}; {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args().workers)
