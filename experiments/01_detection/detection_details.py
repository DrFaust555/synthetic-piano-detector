"""Numbers of Secs. 3.6 and 4.1 that are not cells of Table 5.

    python experiments/01_detection/detection_details.py      -> results/01_detection/detection_details.md

  * thresholds at 5 % false alarms of the three performance detectors (subheadings of Table 5);
  * false alarms of every detector at a fixed P = 0.5 (Sec. 3.6: "from 3% to 36%");
  * fusion, isolated notes: share of the Otey notes detected at 3-7 % false alarms (Sec. 4.1);
  * fusion, isolated notes in rooms never used in training, with added noise (Sec. 4.1: 0.952 -> 0.975, 0.870 -> 0.816);
  * fusion, performances: SoundFont and Otey renders together (AUC 0.952 [0.92, 0.98]);
  * fusion without the 16 IQR features (Sec. 4.1: AUC 0.930 -> 0.910, Otey 85 % -> 60 %);
  * Pianoteq scored by each of the five fold models (Sec. 4.1: CNN 0.54-0.95, fusion 0.88-0.96);
  * share of real performances below P = 0.0001 under the fusion (Fig. 3b: "half").
Inputs: data/predictions/ (out-of-fold scores) and the feature tables (the performance feature detector is retrained
here, as in performances_features.py, setting B).
"""
import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402
import protocol.common as L                    # noqa: E402
import protocol.corpus as V                    # noqa: E402
import protocol.corpus_base as P               # noqa: E402
from protocol.corpus import PERF_KINDS         # noqa: E402

OUT = results_dir('01_detection')
PRED = os.path.join(DATA, 'predictions')


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def perf_network(model, n_meta, fold):
    p = np.full(n_meta, np.nan)
    for k in range(5):
        p[fold == k] = np.load(os.path.join(PRED, 'perf', f'{model}_f{k}.npz'))['p'][fold == k]
    return p


def main():
    md = ['# Detection: further numbers of Secs. 3.6 and 4.1\n\n']
    # ------------------------------------------------------------------ isolated notes
    o = pd.read_csv(os.path.join(PRED, 'notes_oof.csv'))
    cc = ((o.source == 'ccmusic') & o.corpus).values
    sf = ((o.source == 'soundfont') & o.corpus).values
    ot = ((o.source == 'otey') & o.corpus).values
    f = pd.read_csv(os.path.join(PRED, 'notes_features_oof.csv'))
    fa05 = {'notes, feature detector': np.mean(f.loc[f.src == 'CCMusic', 'P'] > 0.5),
            'notes, phaseogram CNN': np.mean(o.cnn_p[cc] > 0.5), 'notes, fusion': np.mean(o.fus27_p[cc] > 0.5)}
    p = o.fus27_p.values
    md.append('## Fusion, isolated notes: Otey notes detected at other false-alarm rates\n\n'
              '| false alarms | threshold | Otey notes detected |\n|---|---|---|\n')
    for q in (3, 4, 5, 6, 7):
        t = np.quantile(p[cc], 1 - q / 100)
        md.append(f'| {q} % | {t:.3f} | {int(np.sum(p[ot] > t))}/{ot.sum()} = {np.mean(p[ot] > t):.1%} |\n')
    md.append('\n## Isolated notes in unseen rooms with noise (threshold: 5 % false alarms on the real notes in the same rooms)\n\n'
              '| detector | SoundFont, clean | SoundFont, rooms | Otey, clean | Otey, rooms |\n|---|---|---|---|---|\n')
    for name, c in (('fusion', 'fus27_p'), ('feature detector of the same run', 'xgb27_p'), ('phaseogram CNN', 'cnn_p')):
        cells = []
        for m in (sf, ot):
            for suf in ('', '_room'):
                q = o[c + suf].values
                t5 = np.quantile(q[cc], 0.95)
                y = np.r_[np.zeros(cc.sum()), np.ones(m.sum())]
                cells.append(f'{L.auc_fast(y, np.r_[q[cc], q[m]]):.3f} / {np.mean(q[m] > t5):.0%}')
        md.append(f'| {name} | ' + ' | '.join([cells[0], cells[1], cells[2], cells[3]]) + ' |\n')

    # ------------------------------------------------------------------ performances
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    mp = pd.read_csv(os.path.join(PRED, 'perf_meta.csv'))
    ms = pd.read_csv(os.path.join(PRED, 'perf_meta_sfz.csv'))
    meta = pd.concat([mp, ms], ignore_index=True)
    fold = meta.fold.values.astype(int)
    d = df.set_index('file').loc[mp.file].reset_index()
    grp = np.r_[d.grp.values, ms.grp.values]
    kind = np.r_[d.kind.values, ms.kind.values]
    sub = np.r_[d.subkind.values, ms.subkind.values]
    y = meta.y.values.astype(int)
    rp = kind == 'real_performance'
    other = np.isin(kind, ['sound_module', 'digital_other'])          # SC-55, GP-3: not in the paper
    trained = (y == 1) & (kind != 'sfz_performance') & ~other
    alg = (sub == 'sf2_perf') | (sub == 'otey_perf')
    otey = sub == 'otey_perf'
    thr = {}
    md.append('\n## Performances, fusion and CNN (setting B)\n\n| quantity | CNN | fusion |\n|---|---|---|\n')
    cells = {k: [] for k in ('th', 'alg', 'fa05', 'below')}
    for m in ('cnn_perf', 'fusion_perf'):
        p = perf_network(m, len(meta), fold)
        s = logit(p)
        t5 = np.quantile(s[rp], 0.95)
        thr[m] = np.quantile(p[rp], 0.95)
        cells['th'].append(f'{thr[m]:.4f}')
        cells['alg'].append(f'{V.fmt3(V.cmp3(y, s, grp, rp, alg), det=False)} / {np.mean(s[alg] > t5):.0%}')
        cells['fa05'].append(f'{np.mean(p[rp] > 0.5):.1%}')
        cells['below'].append(f'{np.mean(p[rp] < 1e-4):.1%}')
    md += [f'| threshold P_th (5 % false alarms) | {cells["th"][0]} | {cells["th"][1]} |\n',
           f'| SoundFont + Otey renders: AUC [95% CI] / detected | {cells["alg"][0]} | {cells["alg"][1]} |\n',
           f'| real performances above P = 0.5 | {cells["fa05"][0]} | {cells["fa05"][1]} |\n',
           f'| real performances below P = 0.0001 | {cells["below"][0]} | {cells["below"][1]} |\n']
    fa05['performances, phaseogram CNN'] = float(cells['fa05'][0][:-1]) / 100
    fa05['performances, fusion'] = float(cells['fa05'][1][:-1]) / 100

    # fusion without the 16 IQR features
    w = pd.read_csv(os.path.join(PRED, 'perf_fusion_without_iqr_oof.csv')).set_index('file')
    full = perf_network('fusion_perf', len(meta), fold)
    md.append('\n## Fusion with and without the 16 IQR features (performances, setting B)\n\n'
              '| fusion | all trained non-acoustic: AUC [95% CI] / detected | Otey renders detected |\n|---|---|---|\n')
    nm = len(mp)
    for name, p in (('all 47 features', full[:nm]), ('without the 16 IQRs', w.loc[mp.file, 'p_fusion_perf'].values)):
        s = logit(p)
        t5 = np.quantile(s[rp[:nm]], 0.95)
        r = V.cmp3(y[:nm], s, grp[:nm], rp[:nm], trained[:nm])
        md.append(f'| {name} | {V.fmt3(r, det=False)} / {np.mean(s[trained[:nm]] > t5):.0%} | '
                  f'{np.mean(s[otey[:nm]] > t5):.0%} |\n')

    # feature detector, setting B (as performances_features.py) -> threshold and false alarms at 0.5
    dd = df[df.kind.isin(PERF_KINDS) & ~df.kind.isin(['sound_module', 'digital_other'])].reset_index(drop=True)
    sfz, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
    sfz = sfz.reset_index(drop=True)
    sfz['grp'] = 'syn:sfz:' + sfz['instrument']
    fmap = L.assign_folds(df.loc[df.base, 'grp'], pd.concat([df.loc[~df.base, 'grp'], sfz['grp']]))
    r = V.run_cv3(dd.assign(trainable=True), list(L.FEATURE_COLS), 'T1', ext={'sfz': sfz},
                  ext_fold={'sfz': sfz.grp.map(fmap).astype(int).values})
    pf = r['own'][(dd.kind == 'real_performance').values]
    fa05['performances, feature detector'] = np.mean(pf > 0.5)
    md.append(f'\nFeature detector, performances: threshold P_th = {np.quantile(pf, 0.95):.3f}.\n')

    md.append('\n## False alarms on the real class at a fixed P = 0.5 (Sec. 3.6)\n\n| detector | false alarms |\n|---|---|\n')
    md += [f'| {k} | {v:.1%} |\n' for k, v in fa05.items()]

    # Pianoteq by fold model
    pt = pd.read_csv(os.path.join(PRED, 'pianoteq_by_fold_model.csv'))
    md.append('\n## Pianoteq scored by each fold model (AUC against all real performances / detected at 5 % false alarms)\n\n'
              '| model | ' + ' | '.join(f'fold {k}' for k in range(5)) + ' |\n|---' * 1 + '|---' * 5 + '|\n')
    for m in ('cnn_perf', 'fusion_perf'):
        p = perf_network(m, len(meta), fold)
        sr = logit(p[rp])
        t5 = np.quantile(sr, 0.95)
        cells = []
        for k in range(5):
            s = logit(pt[f'p_{m}_f{k}'].values)
            yy = np.r_[np.zeros(len(sr)), np.ones(len(s))]
            mark = ' (own fold)' if k == int(pt.fold.iloc[0]) else ''
            cells.append(f'{L.auc_fast(yy, np.r_[sr, s]):.3f} / {np.mean(s > t5):.0%}{mark}')
        md.append(f'| {"CNN" if m == "cnn_perf" else "fusion"} | ' + ' | '.join(cells) + ' |\n')
    open(os.path.join(OUT, 'detection_details.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
