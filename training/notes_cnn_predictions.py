"""Out-of-fold predictions of the isolated-note networks (training/notes_cnn.py) and of XGBoost on the same
notes, folds and features, on clean notes and in unseen rooms -> results/training/notes_oof.csv (shipped as
data/predictions/notes_oof.csv) and results/training/notes_cnn.md.

XGBoost baselines (protocol settings, protocol.common.make_model; the 27 isolated features, NaN kept):
  xgb27      trained on clean notes;
Scoring as in the network fits: corpus and library notes by the model of their fold (libraries never trained),
clips by the known-engine protocol of Sec. 3.7.  AUC vs CCMusic; FA = CCMusic P > 0.5; det = share of a source with P > 0.5.

    python training/notes_cnn_predictions.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
import training.notes_cnn as T  # noqa: E402

REC_CLIPS = ['Iowa', 'K18', 'Salamander']


def assemble(meta, fold3, per_split):
    """per_split: {split: full-length array (NaN where not scored)} -> one out-of-sample vector."""
    src, sub = meta['source'].values, meta['sub'].astype(str).values
    p = np.full(len(meta), np.nan)
    for s, q in per_split.items():
        if s.startswith('f'):
            sel = np.isin(src, ['ccmusic', 'soundfont', 'otey', 'library']) & (fold3 == int(s[1:]))
        else:
            sel = (src == 'clip') & {'recorded_clips': np.isin(sub, REC_CLIPS), 'known_Otey': sub == 'Otey',
                                     'known_Casio': sub == 'SoundFont (Casio CA-110)'}[s]
        p[sel] = q[sel]
    return p


def xgb_splits(meta, fold3, corpus, usable, Xc, Xte):
    import protocol.common as L
    y = meta['y'].values.astype(int)
    out = {'p': {}, 'p_room': {}}
    for s in T.SPLITS:
        tr = np.flatnonzero(T.train_mask(meta, fold3, corpus, s))
        A, b = Xc[tr], y[tr]
        if len(set(b)) < 2:                     # only in a --limit smoke cache
            continue
        m = L.make_model('xgb', (b == 0).sum() / (b == 1).sum())
        m.fit(A, b)
        for key, X in (('p', Xc), ('p_room', Xte)):
            q = np.full(len(meta), np.nan)
            q[usable] = m.predict_proba(X[usable])[:, 1]
            out[key][s] = q
    return out


def auc(neg, pos):
    neg, pos = neg[np.isfinite(neg)], pos[np.isfinite(pos)]
    return roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]) if len(neg) and len(pos) else np.nan


def share(x):
    x = x[np.isfinite(x)]
    return float(np.mean(x > 0.5)) if len(x) else np.nan


def main():
    t0 = time.time()
    T._init(2)
    meta = T._D['meta']
    fold3, corpus, usable = T._D['basis']
    src, sub = meta['source'].values, meta['sub'].astype(str).values
    cc = corpus & (src == 'ccmusic')
    sf, ot = corpus & (src == 'soundfont'), corpus & (src == 'otey')
    mf = usable & (src == 'library') & meta['mf_layer'].values.astype(bool)
    clip = src == 'clip'
    preds = {}
    for m in list(T.CNN_MODELS):
        for key in ('p', 'p_room'):
            ps = {}
            for s in T.SPLITS:
                f = T.job_path((m, s)) + '.npz'
                if os.path.exists(f):
                    d = np.load(f)
                    q = np.full(len(meta), np.nan)
                    q[d['rows']] = d[key]
                    ps[s] = q
            preds[(m, key)] = assemble(meta, fold3, ps)
    Xc, Xtr, Xte = T._D['F']
    r = xgb_splits(meta, fold3, corpus, usable, Xc.astype(float), Xte.astype(float))
    for key in ('p', 'p_room'):
        preds[('xgb27', key)] = assemble(meta, fold3, r[key])
    rows, tab = [], meta[['file', 'source', 'sub', 'y']].copy()
    tab['fold3'], tab['corpus'], tab['usable'] = fold3, corpus, usable
    label = {'xgb27': 'XGBoost, 27 features', 'cnn': 'CNN only', 'fus27': 'fusion CNN + 27'}
    for m in label:
        for key, cond in (('p', 'clean'), ('p_room', 'unseen rooms')):
            p = preds[(m, key)]
            tab[f'{m}_{key}'] = p
            rows.append([label[m], cond, auc(p[cc], p[sf]), auc(p[cc], p[ot]), auc(p[cc], p[sf | ot]), auc(p[cc], p[mf]),
                         share(p[cc]), share(p[sf]), share(p[ot]),
                         auc(p[clip & np.isin(sub, REC_CLIPS)], p[clip & ~np.isin(sub, REC_CLIPS)])])
    tab.to_csv(os.path.join(OUT, 'notes_oof.csv'), index=False)
    key = pd.read_csv(os.path.join(DATA, 'listening', 'answer_key.csv'))
    clip_rows = tab.loc[clip].set_index('file')
    clip_files = ['Stattia/listening_kit/web/' + c + '.wav' for c in key['clip']]
    scores = clip_rows.loc[clip_files, 'fus27_p'].values
    if not np.isfinite(scores).all():
        raise RuntimeError('Missing listening models: train recorded_clips, known_Otey and known_Casio')
    algorithmic = ~np.isin(key['source'].map(T.CL.CLASS_OF), REC_CLIPS)
    listening = pd.DataFrame(dict(clip=key['clip'], file=clip_files, source=key['source'],
                                  note=key['note'], algorithmic=algorithmic, p_fusion=scores))
    listening['training'] = ['known engine, clip pitches excluded' if a else 'full isolated-note corpus'
                             for a in algorithmic]
    listening.to_csv(os.path.join(OUT, 'listening_clips.csv'), index=False)
    done = sum(os.path.exists(T.job_path(j) + '.npz') for j in T.job_list())
    md = ['# Isolated notes: CNN / fusion vs XGBoost (signal-domain augmentation)\n\n',
          f'Generated {time.strftime("%Y-%m-%d %H:%M")} by `training/notes_cnn_predictions.py`; {done} of '
          f'{len(T.job_list())} neural fits present. Population: isolated corpus ({int(cc.sum())} CCMusic, {int(sf.sum())} '
          f'SoundFont, {int(ot.sum())} Otey), protocol folds; {int(mf.sum())} library mf notes never trained. "unseen rooms" = '
          'the same notes through measured rooms never used in training + noise (SNR 30-50 dB). '
          'the CNN and fusion saw room copies in training; xgb27 did not.\n\n',
          '| detector | test | SoundFont | Otey | algorithmic | library mf | FA CCMusic | det SoundFont | det Otey | clips |\n',
          '|---|---|---|---|---|---|---|---|---|---|\n']
    for r in rows:
        md.append(f'| {r[0]} | {r[1]} | ' + ' | '.join(f'{v:.3f}' for v in r[2:6]) + ' | ' +
                  ' | '.join(f'{100 * v:.1f}%' for v in r[6:9]) + f' | {r[9]:.3f} |\n')
    with open(os.path.join(OUT, 'notes_cnn.md'), 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    print(''.join(md), f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
