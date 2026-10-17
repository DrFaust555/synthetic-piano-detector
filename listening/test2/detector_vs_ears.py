"""Test 3: the isolated-note detector on the same 24 clips the listeners heard, clip by clip.

    python listening/test2/detector_vs_ears.py

Detector = XGBoost on the 47 features, trained on isolated notes (CCMusic real, SoundFont, Otey) with the
clip's OWN instrument group left out of training (leave-own-group-out, as for the test-1 clips in
final_v3.py item (d)); library notes are never in training.  Two training conditions:
  orig : features of the original notes (the main detector of the paper);
  c1b  : features of the notes cut to 4.0 s (control C1b) -- the same length as the 4.15-s clips.
Clip features are extracted from the clip WAVs with the same transform (orig / cut to 4.0 s).
Listener side: share of answers 3-4 and mean rating per clip, from responses/ via score_v3.py's reader.
Writes detector_vs_ears.csv and detector_vs_ears.md next to this script.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
KIT = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(KIT, '..', '..', 'revision', 'analysis')
sys.path.insert(0, os.path.abspath(ANA))
sys.path.insert(0, KIT)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

import pv2_lib as L           # noqa: E402
import v3_lib as V            # noqa: E402
import v3_controls as C       # noqa: E402
import followup_v2 as F       # noqa: E402
from pv2_lib import FEATURE_COLS, ROOT   # noqa: E402

CC_GROUP = {'CCMusic Yamaha': 'cc:yamaha', 'CCMusic Kawai-G': 'cc:kawai', 'CCMusic YoungChang': 'cc:young chang'}


def own_group(row):
    if row['subclass'] == 'physical_model':
        return 'syn:otey'
    if row['subclass'] == 'sample_bank':
        return 'syn:sf2:' + L.norm_token(row['instrument'].split(' (')[0])
    if row['subclass'] == 'dataset':
        return CC_GROUP[row['instrument']]
    return None                                   # library: never in training


def listeners():
    key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    rows = pd.concat([pd.read_csv(os.path.join(KIT, 'responses', f), dtype=str)
                      for f in sorted(os.listdir(os.path.join(KIT, 'responses'))) if f.endswith('.csv')],
                     ignore_index=True)
    R = rows[list(key['clip'])].astype(int).values
    return rows['listener'].tolist(), R


def main():
    key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    key['grp'] = key.apply(own_group, axis=1)
    df, _ = V.build_corpus_v3(log=lambda *a, **k: None)
    libv = F._libv(df)
    src = C.iso_sources(df, libv)
    src = src[src.src != 'library'].reset_index(drop=True)

    clip_paths = [os.path.join(KIT, 'web', c + '.wav') for c in key['clip']]
    out = key[['clip', 'class', 'subclass', 'instrument', 'note']].copy()
    for cond, fs in (('orig', 'all47'), ('c1b', 'all47'), ('orig', 'med16'), ('c1b', 'med16')):
        tag = cond if fs == 'all47' else cond + '_med16'
        if cond == 'orig':
            tr = C.orig_features(df, src, libv)
            keep = np.ones(len(src), bool)
        else:
            tr = C.features(src, 'c1b', log=print)
            lev = tr['level_4s_db'].values
            keep = (src.src != 'CCMusic').values | (np.isfinite(lev) & (lev >= -60.0))
        d = src.copy()
        d[FEATURE_COLS] = tr[FEATURE_COLS].values
        d = d[keep & (d[FEATURE_COLS].notna().sum(axis=1) >= 30).values].reset_index(drop=True)
        jobs = [(p, cond, os.path.relpath(p, ROOT).replace('\\', '/')) for p in clip_paths]
        X = V.extract_cached3(jobs, 'test3_clips', log=print)
        p = np.full(len(key), np.nan)
        for g in key['grp'].fillna('none').unique():
            m = (key['grp'].fillna('none') == g).values
            t = d[d['grp'] != g]
            spw = float((t.y == 0).sum()) / max(1, int((t.y == 1).sum()))
            _, (pp,) = L.fit_predict('xgb', t[V.FS3[fs]].values.astype(float), t['y'].values,
                                     [X[V.FS3[fs]].values.astype(float)], spw=spw)
            p[m] = pp[m]
        out[f'P_{tag}'] = p
        out[f'n_notes_{cond}'] = X['n_notes'].values
        if fs == 'all47':
            nn = d.groupby('src')['grp'].size()
            print(cond, 'training rows per source:', nn.to_dict())

    names, R = listeners()
    out['ears_synth_share'] = (R >= 3).mean(axis=0)
    out['ears_mean'] = R.mean(axis=0)
    for nm in names:
        out[f'r:{nm}'] = R[names.index(nm)]
    out.to_csv(os.path.join(KIT, 'detector_vs_ears.csv'), index=False)

    y = (out['class'] != 'recorded').astype(int).values
    md = ['# Test 3: detector vs listeners, clip by clip\n',
          f'{len(names)} listeners. Detector: XGBoost, 47 features, clip\'s own instrument left out of training; '
          'flagged = P(synthetic) > 0.5. orig = main detector; c1b = trained on notes cut to 4 s.\n',
          '| clip | class | instrument | note | notes found | P orig | P 4 s | P orig med16 | P 4 s med16 | listeners synth. | mean rating |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    o = out.sort_values(['subclass', 'clip'])
    for _, r in o.iterrows():
        md.append(f"| {r['clip']} | {r['subclass']} | {r['instrument'].split(' (')[0][:32]} | {r['note']} | "
                  f"{r['n_notes_orig']:.0f} | {r['P_orig']:.2f} | {r['P_c1b']:.2f} | {r['P_orig_med16']:.2f} | "
                  f"{r['P_c1b_med16']:.2f} | {100 * r['ears_synth_share']:.0f}% | {r['ears_mean']:.2f} |")
    md.append('\n| subclass | clips | flagged orig | flagged 4 s | flagged orig med16 | flagged 4 s med16 | '
              'listeners synth. |\n|---|---|---|---|---|---|---|')
    for sc, g in out.groupby('subclass'):
        md.append(f"| {sc} | {len(g)} | {(g.P_orig > .5).sum()} | {(g.P_c1b > .5).sum()} | "
                  f"{(g.P_orig_med16 > .5).sum()} | {(g.P_c1b_med16 > .5).sum()} | {100 * g.ears_synth_share.mean():.0f}% |")
    md.append('\nAUC recorded vs algorithmic over the 24 clips: '
              f"detector orig {L.auc_fast(y, out.P_orig.values):.3f}, detector 4 s {L.auc_fast(y, out.P_c1b.values):.3f}, "
              f"med16 orig {L.auc_fast(y, out.P_orig_med16.values):.3f}, med16 4 s {L.auc_fast(y, out.P_c1b_med16.values):.3f}, "
              f"panel (mean rating) {L.auc_fast(y, out.ears_mean.values):.3f}.")
    alg = y == 1
    for nm in ('P_orig', 'P_c1b', 'P_orig_med16', 'P_c1b_med16'):
        from scipy.stats import spearmanr
        rho = spearmanr(out.loc[alg, nm], out.loc[alg, 'ears_mean'])
        md.append(f'Spearman, algorithmic clips, {nm} vs mean rating: rho = {rho.correlation:+.2f} (p = {rho.pvalue:.2f}).')
    txt = '\n'.join(md) + '\n'
    open(os.path.join(KIT, 'detector_vs_ears.md'), 'w', encoding='utf-8').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
