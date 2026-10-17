"""Train the paper's phaseogram CNN / fusion models on ISOLATED NOTES ONLY (format-matched), resumable.

Model, augmentation, consistency loss, optimiser and schedule are the paper's own code, called unchanged:
quality.realness_net._train_fold (FileModel/PhaseCNN, _view augmentation, BCE(pos_weight)+1.0*MSE consistency,
Adam lr 1e-3 wd 1e-4, batch 24 files, 40 epochs, torch.Generator seed 0) and quality.realness_net._predict.
Each fit starts from torch.manual_seed(0) / np.random.seed(0), so the result of a job does not depend on the
order in which jobs run (the paper's run() seeded once and trained fold models sequentially).

Models:  cnn   = CNN only (use_analytic=False; head on [mean; std] of the note embeddings)
         fus47 = fusion, CNN embedding + all 47 features (median-imputed + standardised on the training fold)
         fus16 = fusion, CNN embedding + the 16 Group-A medians. _train_fold sizes the head with
                 len(realness_net.FEATURE_COLS); for this model that module attribute is set to the 16 names
                 for the duration of the call (attribute assignment in the worker process; no file is edited).
Pooling: one file = one note, so FileModel's own single-note branch gives std = 0 for every file of every class.

Data variants (phaseograms from build_phaseograms.py; 'con' is a reserved file name on Windows, hence 'constr'):
  constr: f0 search constrained to +-1 semitone of the nominal pitch, all registers.
  det48 : the detector's unconstrained f0, only files with nominal MIDI 48-76 (all classes).

Conditions / splits (fixed folds of analysis/out/folds.csv):
  T2 f<k> : train = CCMusic + SoundFont + Otey notes outside fold k.
  T1 f<k> : as T2 + the libraries' mf-equivalent-layer notes outside fold k, labelled synthetic.
  LOGO    : T2 population, full / without Otey / without Casio CA-110 (listening-clip scoring, RESULTS_v2 s.4).
Every job writes P(synthetic) for every usable file of its variant to jobs/<variant>/<model>_<cond>_<split>.npz.

    python Stattia/revision/cnn_iso/train_cnn_iso.py [--workers 6] [--threads 2] [--scope all|priority]
"""
from __future__ import annotations

import os
import sys
import time
import argparse

_THREADS = os.environ.get('CNN_ISO_THREADS', '2')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = _THREADS

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

import cnn_iso_lib as CL               # noqa: E402

MODELS = {'cnn': (False, 'all47'), 'fus47': (True, 'all47'), 'fus16': (True, 'med16')}
EPOCHS = 40
SPLITS_CV = [f'f{k}' for k in range(5)]
SPLITS_LOGO = ['logo_full', 'logo_noOtey', 'logo_noCasio']


def job_list(scope='all'):
    """Priority order: constr T2 (cnn, fus47) -> constr T1 (cnn, fus47) -> constr fus16 -> constr LOGO -> det48."""
    J = []
    for cond in ('T2', 'T1'):
        for m in ('cnn', 'fus47'):
            J += [('constr', m, cond, s) for s in SPLITS_CV]
    if scope == 'priority':
        return J[:10]
    for cond in ('T2', 'T1'):
        J += [('constr', 'fus16', cond, s) for s in SPLITS_CV]
    for m in ('cnn', 'fus47', 'fus16'):
        J += [('constr', m, 'T2', s) for s in SPLITS_LOGO]
    for cond in ('T2', 'T1'):
        for m in ('cnn', 'fus47', 'fus16'):
            J += [('det48', m, cond, s) for s in SPLITS_CV]
    return J


def job_path(job, jobs_dir=CL.JOBS):
    v, m, c, s = job
    return os.path.join(jobs_dir, v, f'{m}_{c}_{s}.npz')


def usable(meta: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == 'constr':
        return meta['con_ok'].values.astype(bool)
    if variant == 'det48':
        return meta['det_ok'].values.astype(bool) & meta['in_48_76'].values.astype(bool)
    raise ValueError(variant)


def train_mask(meta: pd.DataFrame, ok: np.ndarray, cond: str, split: str) -> np.ndarray:
    src, fold, grp = meta['source'].values, meta['fold'].values, meta['grp'].values
    corpus = np.isin(src, ['ccmusic', 'soundfont', 'otey']) & ok
    if split.startswith('f'):
        k = int(split[1:])
        tr = corpus & (fold != k)
        if cond == 'T1':
            tr |= (src == 'library') & meta['mf_layer'].values.astype(bool) & ok & (fold != k)
        return tr
    assert cond == 'T2'
    if split == 'logo_full':
        return corpus
    if split == 'logo_noOtey':
        return corpus & (grp != CL.OTEY_G)
    if split == 'logo_noCasio':
        return corpus & (grp != CL.CASIO_G)
    raise ValueError(split)


# ----------------------------------------------------------------------------- worker
_D = {}


def _init(threads):
    import torch
    torch.set_num_threads(int(threads))
    meta = pd.read_csv(os.path.join(CL.CACHE, 'phaseo_meta.csv'))
    u = CL.universe()
    assert (u['file'].values == meta['file'].values).all(), 'universe changed since the phaseogram cache was built'
    X47 = CL.features(u)
    i16 = [CL.FEATURE_COLS.index(c) for c in CL.MED16]
    _D['meta'] = meta
    _D['X'] = {'all47': X47, 'med16': X47[:, i16].copy()}
    _D['img'] = {'constr': np.load(os.path.join(CL.CACHE, 'phaseo_con.npy'), mmap_mode='r'),
                 'det48': np.load(os.path.join(CL.CACHE, 'phaseo_det.npy'), mmap_mode='r')}


def run_job(args):
    job, epochs, jobs_dir = args
    out = job_path(job, jobs_dir)
    if os.path.exists(out):
        return job, 'cached', 0.0
    import torch
    import quality.realness_net as rn
    v, m, cond, split = job
    meta = _D['meta']
    ok = usable(meta, v)
    rows = np.flatnonzero(ok)
    use_an, fs = MODELS[m]
    Xa = _D['X'][fs]
    img = _D['img'][v]
    y = meta['y'].values.astype(int)
    recs = [{'phase': np.array(img[i][None], dtype=np.float32), 'analytic': Xa[i], 'y': int(y[i])} for i in rows]
    pos = {r: j for j, r in enumerate(rows)}
    trm = train_mask(meta, ok, cond, split)
    tr = [pos[r] for r in np.flatnonzero(trm)]
    t0 = time.time()
    torch.manual_seed(rn.SEED)
    np.random.seed(rn.SEED)
    saved = rn.FEATURE_COLS
    try:
        rn.FEATURE_COLS = list(CL.FEATURE_COLS) if fs == 'all47' else list(CL.MED16)
        model, si = rn._train_fold(recs, tr, use_an, epochs)
    finally:
        rn.FEATURE_COLS = saved
    p = rn._predict(model, recs, list(range(len(recs))), si)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.npz'
    np.savez(tmp, p=p.astype(np.float64), rows=rows, train_rows=rows[tr], n_train=len(tr),
             n_train_real=int((y[rows[tr]] == 0).sum()), n_train_syn=int((y[rows[tr]] == 1).sum()),
             seconds=dt, epochs=epochs, threads=torch.get_num_threads(), torch_version=torch.__version__)
    os.replace(tmp, out)
    return job, 'trained', dt


def main(workers=6, threads=2, scope='all', epochs=EPOCHS, jobs_dir=CL.JOBS, only=None, log=print):
    os.makedirs(CL.LOGS, exist_ok=True)
    jobs = job_list(scope)
    if only:
        jobs = [j for j in jobs if '/'.join(j).startswith(only)]
    todo = [j for j in jobs if not os.path.exists(job_path(j, jobs_dir))]
    logf = os.path.join(CL.LOGS, 'train.log')

    def L_(s):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {s}'
        log(line)
        with open(logf, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    L_(f'{len(jobs)} jobs in scope "{scope}", {len(todo)} to train (epochs={epochs}, workers={workers}, '
       f'threads/worker={threads}, out={jobs_dir})')
    if not todo:
        return
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(threads,)) as pool:
            for k, (job, st, dt) in enumerate(pool.imap_unordered(run_job, [(j, epochs, jobs_dir) for j in todo])):
                L_(f'[{k + 1}/{len(todo)}] {"/".join(job)} {st} {dt / 60:.1f} min (elapsed {(time.time() - t0) / 3600:.2f} h)')
    else:
        _init(threads)
        for k, j in enumerate(todo):
            job, st, dt = run_job((j, epochs, jobs_dir))
            L_(f'[{k + 1}/{len(todo)}] {"/".join(job)} {st} {dt / 60:.1f} min')
    L_(f'all done in {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--threads', type=int, default=int(_THREADS))
    ap.add_argument('--scope', default='all', choices=['all', 'priority'])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--jobs-dir', default=CL.JOBS)
    ap.add_argument('--only', default=None, help='prefix filter, e.g. constr/cnn/T2')
    a = ap.parse_args()
    main(a.workers, a.threads, a.scope, a.epochs, a.jobs_dir, a.only)
