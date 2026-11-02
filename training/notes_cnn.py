"""Train the paper's isolated-note CNN and fusion (CNN + 27 features).

Requires notes_phaseograms.py and notes_room_features.py caches. Each training
note is replaced with probability 0.5 by one room-and-noise copy, shared by both
branches. Five grouped folds score corpus and library notes. Listening clips
use the full corpus for libraries, and known-engine models with all training
notes at the clip pitches excluded for Otey and Casio.

CNN_NOTE_MODELS (cnn,fus27) and CNN_NOTE_SPLITS restrict resumable jobs.
    python training/notes_cnn.py [--workers 12] [--threads 2] [--epochs 40] [--device cpu|cuda]
"""
from __future__ import annotations
import os
import sys
import time
import pickle
import argparse

_THREADS = os.environ.get('CNN_ISO_THREADS', '2')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = _THREADS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
import protocol.cnn as CL  # noqa: E402

CACHE_ISO = os.path.join(RESULTS, 'cnn', 'cache_iso')
# set through the environment so that spawned workers see them (main() sets them from --device / --jobs-dir)
JOBS = os.environ.get('CNN_JOBS_DIR', os.path.join(RESULTS, 'cnn', 'jobs_notes'))
DEVICE = os.environ.get('CNN_DEVICE', 'cpu')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')      # deterministic cuBLAS on the GPU path
EPOCHS = 40
P_AUG = 0.5
CNN_MODELS = {'cnn': False, 'fus27': True}
ALL_SPLITS = [f'f{k}' for k in range(5)] + ['recorded_clips', 'known_Otey', 'known_Casio']
SPLITS = os.environ.get('CNN_NOTE_SPLITS', ','.join(ALL_SPLITS)).split(',')
if not set(SPLITS) <= set(ALL_SPLITS):
    raise ValueError('CNN_NOTE_SPLITS contains an unknown split')


def job_list():
    keep = os.environ.get('CNN_NOTE_MODELS', ','.join(CNN_MODELS)).split(',')
    if not set(keep) <= set(CNN_MODELS):
        raise ValueError('CNN_NOTE_MODELS must contain cnn and/or fus27')
    return [(m, s) for s in SPLITS for m in CNN_MODELS if m in keep]


def job_path(job):
    return os.path.join(JOBS, f'{job[0]}_T2_{job[1]}')


def basis(meta):
    """(fold of every row, corpus mask, usable mask) from the fold map of the protocol."""
    f3 = pd.read_csv(os.path.join(DATA, 'protocol', 'folds.csv'))
    iso3 = f3[f3.kind.isin(['real_isolated', 'algo_single']) & ~f3.sf_dup.astype(bool)]
    fold_iso = dict(zip(iso3['file'], iso3['fold']))
    fold_lib = f3[f3.kind == 'sampled_library'].groupby('grp')['fold'].first().to_dict()
    src = meta['source'].values
    corpus_src = np.isin(src, ['ccmusic', 'soundfont', 'otey'])
    fold3 = np.full(len(meta), -1)
    in3 = np.zeros(len(meta), bool)
    for i, (f, s, g) in enumerate(zip(meta['file'], src, meta['grp'])):
        if corpus_src[i] and f in fold_iso:
            fold3[i], in3[i] = int(fold_iso[f]), True
        elif s == 'library':
            fold3[i] = int(fold_lib[g])
    ok = meta['ok'].values.astype(bool)
    corpus = corpus_src & in3 & ok
    usable = ok & (~corpus_src | in3)
    return fold3, corpus, usable


def train_mask(meta, fold3, corpus, split):
    grp = meta['grp'].values
    if split.startswith('f'):
        return corpus & (fold3 != int(split[1:]))
    if split.startswith('known_'):
        g, clip_sub = {'known_Otey': (CL.OTEY_G, 'Otey'), 'known_Casio': (CL.CASIO_G, 'SoundFont (Casio CA-110)')}[split]
        midi = meta['nominal_midi'].values
        clip_midis = meta.loc[(meta['source'] == 'clip') & (meta['sub'] == clip_sub), 'nominal_midi'].values
        return corpus & ~((grp == g) & np.isin(midi, clip_midis))
    if split == 'recorded_clips':
        return corpus
    raise ValueError(split)


class _AugRec(dict):
    """training record: with probability P_AUG the phaseogram AND the features come from one room copy.  The copy
    is drawn at the first 'phase' read of a gather and kept until 'y' is read (detector.phaseogram_cnn._gather reads
    phase, phase, analytic, y), so both branches see the same signal."""
    def __init__(self, base, phases, feats, rng):
        super().__init__(base)
        self._ph, self._an, self._rng, self._k = phases, feats, rng, None

    def __getitem__(self, key):
        if key == 'phase':
            if self._k is None:
                self._k = int(self._rng.integers(len(self._ph))) if self._rng.random() < P_AUG else -1
            return super().__getitem__('phase') if self._k < 0 else self._ph[self._k]
        if key == 'analytic':
            return super().__getitem__('analytic') if self._k is None or self._k < 0 else self._an[self._k]
        if key == 'y':
            self._k = None
        return super().__getitem__(key)


_D = {}


def _init(threads):
    import torch
    import protocol.common as L
    torch.set_num_threads(int(threads))
    meta = pd.read_csv(os.path.join(CACHE_ISO, 'meta_iso.csv'))
    u = CL.universe().set_index('file').loc[meta['file']].reset_index()
    assert (u['path'].values == meta['path'].values).all(), 'universe changed since the phaseogram cache was built'
    idx27 = [CL.FEATURE_COLS.index(c) for c in L.ISO27]
    X = CL.features(u).astype(np.float32)
    Xtr = np.load(os.path.join(CACHE_ISO, 'feat_room_train.npy'))
    Xte = np.load(os.path.join(CACHE_ISO, 'feat_room_test.npy'))
    _D['meta'] = meta
    _D['basis'] = basis(meta)
    _D['F'] = (X[..., idx27], Xtr[..., idx27], Xte[..., idx27])
    _D['ph'] = (np.load(os.path.join(CACHE_ISO, 'phaseo_iso.npy'), mmap_mode='r'),
                np.load(os.path.join(CACHE_ISO, 'phaseo_iso_room_train.npy'), mmap_mode='r'),
                np.load(os.path.join(CACHE_ISO, 'phaseo_iso_room_test.npy'), mmap_mode='r'))
    _D['device'] = DEVICE
    if DEVICE != 'cpu':                     # fast path: everything resident on the GPU (training/cnn_fast.py)
        import training.cnn_fast as FT
        torch.backends.cudnn.deterministic = True      # same pooling and training loss as the CPU path
        torch.backends.cudnn.benchmark = False
        ph_c, ph_tr, ph_te = (np.asarray(a, np.float32) for a in _D['ph'])
        N = len(meta)
        mask = np.ones((N, 1), np.float32)
        Xc, Xa, Xt = _D['F']
        _D['fast'] = FT.Data(ph_c[:, None], mask, Xc, meta['y'].values, ph_aug=ph_tr[:, :, None], X_aug=Xa,
                             ph_test=ph_te[:, None], X_test=Xt, device=DEVICE)


def run_job(args):
    job, epochs = args
    out = job_path(job)
    if os.path.exists(out + '.npz'):
        return job, 'cached', 0.0
    import torch
    import protocol.common as L
    import detector.phaseogram_cnn as rn
    m, split = job
    meta = _D['meta']
    fold3, corpus, usable = _D['basis']
    rows = np.flatnonzero(usable)
    y = meta['y'].values.astype(int)
    trm = train_mask(meta, fold3, corpus, split)
    t0 = time.time()
    torch.manual_seed(rn.SEED)
    np.random.seed(rn.SEED)
    use_an = CNN_MODELS[m]
    if _D['device'] != 'cpu':
        import training.cnn_fast as FT
        data = _D['fast']
        tr_rows = np.flatnonzero(trm)
        model, si = FT.train(data, tr_rows, use_an, epochs, p_aug=P_AUG, seed=rn.SEED)
        p, p_room = FT.predict(model, data, rows), FT.predict(model, data, rows, test=True)
    else:
        Xc, Xtr, Xte = _D['F']
        ph_c, ph_tr, ph_te = _D['ph']
        mk = lambda i, img, a: {'phase': np.array(img[None], dtype=np.float32), 'analytic': a, 'y': int(y[i])}
        recs = [mk(i, ph_c[i], Xc[i]) for i in rows]
        recs_room = [mk(i, ph_te[i], Xte[i]) for i in rows]
        pos = {r: j for j, r in enumerate(rows)}
        tr = [pos[r] for r in np.flatnonzero(trm)]
        rng = np.random.default_rng(rn.SEED)
        recs_tr = list(recs)
        for j in tr:
            i = rows[j]
            recs_tr[j] = _AugRec(recs[j], [np.array(ph_tr[i, k][None], dtype=np.float32) for k in range(ph_tr.shape[1])],
                                 [Xtr[i, k] for k in range(Xtr.shape[1])], rng)
        rn.DEVICE = 'cpu'
        model, si = rn._train_fold(recs_tr, tr, use_an, epochs)
        p = rn._predict(model, recs, list(range(len(recs))), si)
        p_room = rn._predict(model, recs_room, list(range(len(recs_room))), si)
        tr_rows = rows[tr]
    dt = time.time() - t0
    os.makedirs(JOBS, exist_ok=True)
    torch.save(model.state_dict(), out + '.pt')
    with open(out + '.si.pkl', 'wb') as fh:
        pickle.dump(si, fh)
    np.savez(out + '.tmp.npz', p=np.asarray(p, np.float64), p_room=np.asarray(p_room, np.float64), rows=rows,
             train_rows=tr_rows, seconds=dt, epochs=epochs, torch_version=torch.__version__)
    os.replace(out + '.tmp.npz', out + '.npz')
    return job, 'trained', dt


def main(workers=12, threads=2, epochs=EPOCHS, device='cpu', jobs_dir=None):
    global JOBS, DEVICE
    os.environ['CNN_DEVICE'] = DEVICE = device
    if jobs_dir:
        os.environ['CNN_JOBS_DIR'] = JOBS = jobs_dir
    os.makedirs(CL.LOGS, exist_ok=True)
    jobs = job_list()
    todo = [j for j in jobs if not os.path.exists(job_path(j) + '.npz')]
    logf = os.path.join(CL.LOGS, 'train_notes.log')

    def L_(s):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {s}'
        print(line, flush=True)
        with open(logf, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    L_(f'{len(jobs)} jobs, {len(todo)} to train (epochs={epochs}, workers={workers}, threads={threads})')
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(threads,)) as pool:
            for k, (job, st, dt) in enumerate(pool.imap_unordered(run_job, [(j, epochs) for j in todo])):
                L_(f'[{k + 1}/{len(todo)}] {"/".join(job)} {st} {dt / 60:.1f} min (elapsed {(time.time() - t0) / 3600:.2f} h)')
    else:
        _init(threads)
        for k, j in enumerate(todo):
            job, st, dt = run_job((j, epochs))
            L_(f'[{k + 1}/{len(todo)}] {"/".join(job)} {st} {dt / 60:.1f} min')
    L_(f'all done in {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=int(_THREADS))
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--device', default='cpu', help='cpu (detector.phaseogram_cnn._train_fold) or cuda (training/cnn_fast.py)')
    ap.add_argument('--jobs-dir', default=None)
    a = ap.parse_args()
    main(a.workers, a.threads, a.epochs, a.device, a.jobs_dir)
