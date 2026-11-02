"""Train the paper's performance CNN and fusion on up to 15 phaseograms per file.

All non-acoustic corpus sources enter training. SFZ play is appended for scoring
only. Five grouped folds; 47 summary features in the fusion. --drop-iqr repeats
the ablation reported in Sec. 4.1. Requires the caches from perf_phaseograms.py
and sfz_phaseograms.py.

    python training/perf_cnn.py [--epochs 40] [--device cpu|cuda] [--models cnn_perf,fusion_perf]
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
ROOT = AUDIO_ROOT
CACHE = os.environ.get('CNN_PERF_CACHE', os.path.join(RESULTS, 'cnn', 'cache_perf'))
MODELS = ['cnn_perf', 'fusion_perf']


def main(epochs, jobs_dir, device, models=None, drop_iqr=False):
    if models is not None and not set(models) <= set(MODELS):
        raise ValueError('Supported performance models: cnn_perf, fusion_perf')
    import torch
    import training.cnn_fast as FT
    import detector.phaseogram_cnn as rn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(jobs_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(CACHE, 'meta_perf.csv'))
    X = np.load(os.path.join(CACHE, 'feat47.npy'))
    n_perf = len(meta)
    ms = pd.read_csv(os.path.join(CACHE, 'meta_sfz.csv'))
    meta = pd.concat([meta, ms], ignore_index=True)
    X = np.concatenate([X, np.load(os.path.join(CACHE, 'feat47_sfz.npy'))])
    if drop_iqr:
        X = np.delete(X, np.arange(16, 32), axis=1)
    y = meta.y.values.astype(int)
    fold = meta.fold.values.astype(int)
    trainable = np.ones(len(meta), bool)
    trainable[n_perf:] = False
    run_models = models or MODELS
    log = lambda s: print(f'{time.strftime("%H:%M:%S")} {s}', flush=True)
    data = {}
    for m in run_models:
        for k in range(5):
            out = os.path.join(jobs_dir, f'{m}_f{k}.npz')
            if os.path.exists(out):
                continue
            t0 = time.time()
            tr = np.flatnonzero(trainable & (fold != k))
            sel = 'perf'
            if sel not in data:
                data.clear()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                ph = np.load(os.path.join(CACHE, 'phaseograms.npy')).astype(np.float32)
                mask = np.load(os.path.join(CACHE, 'phaseogram_mask.npy'))
                ph = np.concatenate([ph, np.load(os.path.join(CACHE, 'phaseograms_sfz.npy')).astype(np.float32)])
                mask = np.concatenate([mask, np.load(os.path.join(CACHE, 'phaseogram_mask_sfz.npy'))])
                data[sel] = FT.Data(ph, mask, X, y, device=device)
                del ph
                log(f'loaded {sel}')
            model, si = FT.train(data[sel], tr, m.startswith('fus'), epochs, p_aug=0.0, seed=rn.SEED)
            p = FT.predict(model, data[sel], np.arange(len(meta)))
            import pickle
            torch.save(model.state_dict(), out[:-4] + '.pt')          # weights (detector.phaseogram_cnn.FileModel)
            with open(out[:-4] + '.si.pkl', 'wb') as fh:              # (imputer, scaler) of the 47 features
                pickle.dump(si, fh)
            np.savez(out, p=np.asarray(p, np.float64), train_rows=tr, seconds=time.time() - t0, epochs=epochs)
            log(f'{m} fold {k}: {time.time() - t0:.0f} s')
    cols = {}
    for m in MODELS:
        p = np.full(len(meta), np.nan)
        for k in range(5):
            f = os.path.join(jobs_dir, f'{m}_f{k}.npz')
            if os.path.exists(f):
                p[fold == k] = np.load(f)['p'][fold == k]
        cols['p_' + m] = p
    pd.concat([meta[['file', 'rel', 'fold']], pd.DataFrame(cols)], axis=1).to_csv(os.path.join(jobs_dir, 'oof_perf.csv'), index=False)
    log('all done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--jobs-dir', default=os.path.join(RESULTS, 'cnn', 'jobs_perf'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--models', default=None)
    ap.add_argument('--drop-iqr', action='store_true', help='remove the 16 IQR features from the feature input')
    a = ap.parse_args()
    main(a.epochs, a.jobs_dir, a.device, a.models.split(',') if a.models else None, a.drop_iqr)
