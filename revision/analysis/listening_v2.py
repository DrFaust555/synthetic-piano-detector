"""Protocol v2 -- re-analysis of the 32-listener test (Stattia/listening_kit).

Classes as they really are: RECORDED notes = Iowa (7), K18 (7), Salamander (6) -- three sample
libraries of real recorded notes; ALGORITHMIC = SoundFont Casio CA-110 "Piano (FM)" renders (5) and
Otey physical model (5).  "Judged synthetic" = answer 3 or 4.

Statistics: pooled exact binomial (judgments treated as independent -- optimistic); per-listener
one-sided Wilcoxon signed-rank of (rate - 0.5) > 0 with zero_method='zsplit' (as quality/score_listening.py);
two-way bootstrap CIs (listeners and clips-within-class resampled, B = 2000, seed 0) that also carry
item variability; paired Salamander vs Iowa/K18 comparisons across listeners; listener-level AUC and
d' for recorded vs algorithmic; effective sounding duration of every clip (t40 / t60).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import binomtest, wilcoxon, norm, spearmanr

import pv2_lib as L
from pv2_lib import OUT, KIT

CLASS_OF = {'Iowa Steinway': 'Iowa', 'K18 upright': 'K18', 'Salamander (sampled)': 'Salamander',
            'soundfont': 'SoundFont (Casio CA-110)', 'Otey physical model': 'Otey'}
REC = ['Iowa', 'K18', 'Salamander']
ALG = ['SoundFont (Casio CA-110)', 'Otey']
GROUPS = {'Iowa': ['Iowa'], 'K18': ['K18'], 'Salamander': ['Salamander'],
          'SoundFont (Casio CA-110)': ['SoundFont (Casio CA-110)'], 'Otey': ['Otey'],
          'Iowa+K18 (paper\'s "real")': ['Iowa', 'K18'], 'RECORDED (all 3 libraries)': REC,
          'ALGORITHMIC (SoundFont + Otey)': ALG}


def load():
    key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    key['cls'] = key['source'].map(CLASS_OF)
    res = pd.read_csv(os.path.join(KIT, 'results_anon.csv'), dtype=str)
    clips = list(key['clip'])
    R = res[clips].astype(int).values                       # listeners x clips, 1..4
    return key, res, R


def wil(vals, alt):
    v = np.asarray(vals, float) - 0.5
    if np.all(v == 0):
        return np.nan
    return float(wilcoxon(v, alternative=alt, zero_method='zsplit').pvalue)


def two_way_boot(stat_fn, nL, cls_idx, n=L.N_BOOT, seed=L.SEED):
    """Resample listeners (rows) and, independently within each class, clips (columns)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        li = rng.integers(0, nL, nL)
        ci = {c: idx[rng.integers(0, len(idx), len(idx))] for c, idx in cls_idx.items()}
        out[b] = stat_fn(li, ci)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def auc_scores(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return L.auc_fast(y, np.r_[neg, pos])


def dprime(k_hit, n_hit, k_fa, n_fa):
    H = (k_hit + 0.5) / (n_hit + 1.0)
    F = (k_fa + 0.5) / (n_fa + 1.0)
    return float(norm.ppf(H) - norm.ppf(F))


def durations(key):
    rows = []
    for _, r in key.iterrows():
        y, sr = sf.read(os.path.join(KIT, 'web', r['clip'] + '.wav'))
        if y.ndim > 1:
            y = y.mean(axis=1)
        win, hop = int(0.010 * sr), int(0.005 * sr)
        n = 1 + (len(y) - win) // hop
        rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2)) for i in range(n)])
        db = 20 * np.log10(rms + 1e-12) - 20 * np.log10(rms.max() + 1e-12)
        dur = len(y) / sr
        out = {'clip': r['clip'], 'cls': r['cls'], 'note': r['note'], 'clip_len_s': dur}
        for X in (40, 60):
            above = np.where(db >= -X)[0]
            t = (above[-1] * hop + win) / sr
            out[f't{X}_s'] = t
            out[f't{X}_censored'] = bool(t >= dur - 0.03)      # still above threshold at the 10-ms fade-out
        nz = np.where(np.abs(y) > 0)[0]
        out['last_nonzero_s'] = (nz[-1] + 1) / sr
        rows.append(out)
    return pd.DataFrame(rows)


def main(log=print):
    import warnings
    warnings.filterwarnings('ignore', message='Exact p-value calculation')   # zsplit -> normal approximation
    key, res, R = load()
    nL, nC = R.shape
    J = (R >= 3).astype(float)                              # judged synthetic
    cls = key['cls'].values
    cls_idx = {c: np.where(cls == c)[0] for c in REC + ALG}
    md = ['\n## 6. Listening test re-analysed (classes as they really are)\n\n']
    md.append(f'{nL} listeners x {nC} clips = {nL * nC} judgments (`listening_kit/results_anon.csv`; self-reported '
              f'experience: {res["experience"].value_counts().to_dict()}). The paper\'s "real" class (14 clips) is Iowa (7) + '
              'K18 (7) sample-library notes; its "sample library" class is Salamander (6). All three are recorded acoustic notes '
              'from sample libraries. "Judged synthetic" = answer 3 or 4.\n\n')

    # ---------------- per class
    rows, csvrows, key_out = [], [], {}
    for name, members in GROUPS.items():
        cidx = np.where(np.isin(cls, members))[0]
        k = int(J[:, cidx].sum()); n = J[:, cidx].size
        rate = k / n
        p2 = binomtest(k, n, 0.5).pvalue
        p1 = binomtest(k, n, 0.5, alternative='greater').pvalue
        per = J[:, cidx].mean(axis=1)
        ci = two_way_boot(lambda li, ci_: J[np.ix_(li, np.concatenate([ci_[m] for m in members]))].mean(),
                          nL, cls_idx)
        w1, w2 = wil(per, 'greater'), wil(per, 'two-sided')
        mr = R[:, cidx].mean()
        rows.append([name, len(cidx), f'{k}/{n}', f'{100 * rate:.1f}%', f'[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]',
                     f'{p1:.2g}', f'{p2:.2g}', f'{int((per > .5).sum())}/{int((per == .5).sum())}/{int((per < .5).sum())}',
                     f'{w1:.2g}', f'{w2:.2g}', f'{mr:.2f}'])
        csvrows.append({'class': name, 'clips': len(cidx), 'k_synth': k, 'n': n, 'rate': rate, 'ci2_lo': ci[0], 'ci2_hi': ci[1],
                        'binom_p_greater': p1, 'binom_p_two': p2, 'listeners_gt': int((per > .5).sum()),
                        'listeners_eq': int((per == .5).sum()), 'listeners_lt': int((per < .5).sum()),
                        'wilcoxon_p_greater': w1, 'wilcoxon_p_two': w2, 'mean_rating': mr})
        key_out[f'cls:{name}'] = csvrows[-1]
    pd.DataFrame(csvrows).to_csv(os.path.join(OUT, 'listening_per_class.csv'), index=False)
    md.append('### 6.1 Judged-synthetic rate per class\n\n')
    md.append(L.md_table(['class', 'clips', 'judged synth.', 'rate', '95% CI (two-way boot.)', 'binom. p (>0.5)',
                          'binom. p (two-sided)', 'listeners >/=/< 0.5', 'Wilcoxon p (>0.5)', 'Wilcoxon p (two-sided)',
                          'mean rating (1-4)'], rows))
    md.append('\nThe pooled binomial treats the judgments as independent; the listener-level Wilcoxon accounts for clustering '
              'within listeners but not for the small number of clips per class; the two-way bootstrap CI accounts for both.\n\n')

    # ---------------- paired Salamander vs Iowa/K18
    md.append('### 6.2 Salamander vs Iowa / K18, paired across listeners\n\n')
    pr = []
    for a, bl in (('Salamander', ['Iowa', 'K18']), ('Salamander', ['Iowa']), ('Salamander', ['K18']), ('Iowa', ['K18'])):
        ia = cls_idx[a]; ib = np.concatenate([cls_idx[b] for b in bl])
        for metric, M in (('judged-synth rate', J), ('mean rating', R.astype(float))):
            d = M[:, ia].mean(axis=1) - M[:, ib].mean(axis=1)
            pw = float(wilcoxon(d, alternative='two-sided', zero_method='zsplit').pvalue) if np.any(d != 0) else np.nan
            npos, nneg = int((d > 0).sum()), int((d < 0).sum())
            ps = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else np.nan
            ci = two_way_boot(lambda li, ci_: M[np.ix_(li, ci_[a])].mean() -
                              M[np.ix_(li, np.concatenate([ci_[b] for b in bl]))].mean(), nL, cls_idx)
            lbl = f"{a} - {'+'.join(bl)}"
            pr.append([lbl, metric, f'{d.mean():+.3f}', f'[{ci[0]:+.3f}, {ci[1]:+.3f}]', f'{npos}/{int((d == 0).sum())}/{nneg}',
                       f'{pw:.3g}', f'{ps:.3g}'])
            key_out[f'paired:{lbl}:{metric}'] = {'mean_diff': float(d.mean()), 'ci': ci, 'wilcoxon_p': pw, 'sign_p': ps}
    md.append(L.md_table(['difference', 'metric', 'mean over listeners', '95% CI (two-way boot.)', 'listeners +/0/-',
                          'Wilcoxon p (two-sided)', 'sign test p'], pr))

    # ---------------- listener-level AUC and d'
    md.append("\n### 6.3 Listener-level discrimination: recorded vs algorithmic\n\n")
    rec_i = np.concatenate([cls_idx[c] for c in REC]); alg_i = np.concatenate([cls_idx[c] for c in ALG])
    ik_i = np.concatenate([cls_idx['Iowa'], cls_idx['K18']]); sal_i = cls_idx['Salamander']
    defs = [('recorded (20) vs algorithmic (10)', rec_i, alg_i),
            ('Iowa+K18 (14) vs algorithmic (10)  [paper\'s real vs synthetic-generated]', ik_i, alg_i),
            ('Salamander (6) vs algorithmic (10)', sal_i, alg_i),
            ('Iowa+K18 (14) vs Salamander (6)  [positive = Salamander]', ik_i, sal_i)]
    ar = []
    per_listener = pd.DataFrame({'listener': res['listener'], 'experience': res['experience']})
    for lbl, ni, pi in defs:
        aucs = np.array([auc_scores(R[l, ni], R[l, pi]) for l in range(nL)])
        dps = np.array([dprime(J[l, pi].sum(), len(pi), J[l, ni].sum(), len(ni)) for l in range(nL)])
        pooled = auc_scores(R[:, ni].ravel(), R[:, pi].ravel())
        panel = auc_scores(R[:, ni].mean(axis=0), R[:, pi].mean(axis=0))
        cls_neg = [c for c in REC + ALG if np.isin(cls_idx[c], ni).all()]
        cls_pos = [c for c in REC + ALG if np.isin(cls_idx[c], pi).all()]

        def pstat(li, ci_):
            nn = np.concatenate([ci_[c] for c in cls_neg]); pp = np.concatenate([ci_[c] for c in cls_pos])
            return auc_scores(R[np.ix_(li, nn)].mean(axis=0), R[np.ix_(li, pp)].mean(axis=0))

        def mstat(li, ci_):
            nn = np.concatenate([ci_[c] for c in cls_neg]); pp = np.concatenate([ci_[c] for c in cls_pos])
            return np.mean([auc_scores(R[l, nn], R[l, pp]) for l in li])
        ci_p = two_way_boot(pstat, nL, cls_idx)
        ci_m = two_way_boot(mstat, nL, cls_idx, n=500)
        w = wil(aucs, 'greater')
        ar.append([lbl, f'{aucs.mean():.3f} [{ci_m[0]:.3f}, {ci_m[1]:.3f}]', f'{np.median(aucs):.3f}',
                   f'{int((aucs > .5).sum())}/{int((aucs == .5).sum())}/{int((aucs < .5).sum())}', f'{w:.2g}',
                   f'{dps.mean():.2f} (median {np.median(dps):.2f})', f'{pooled:.3f}',
                   f'{panel:.3f} [{ci_p[0]:.3f}, {ci_p[1]:.3f}]'])
        key_out[f'auc:{lbl}'] = {'mean_listener_auc': float(aucs.mean()), 'ci': ci_m, 'median': float(np.median(aucs)),
                                 'n_gt': int((aucs > .5).sum()), 'wilcoxon_p': w, 'mean_dprime': float(dps.mean()),
                                 'pooled_auc': pooled, 'panel_auc': panel, 'panel_ci': ci_p}
        per_listener[f'auc: {lbl}'] = aucs
        per_listener[f'dprime: {lbl}'] = dps
    for c in REC + ALG:
        per_listener[f'rate_synth: {c}'] = J[:, cls_idx[c]].mean(axis=1)
    per_listener.to_csv(os.path.join(OUT, 'listening_per_listener.csv'), index=False)
    md.append(L.md_table(['comparison (positive = second class)', 'mean listener AUC [95% CI]', 'median', 'listeners >/=/< 0.5',
                          'Wilcoxon p (AUC>0.5)', "mean d'", 'pooled AUC (all judgments)', 'panel AUC (per-clip mean rating) [95% CI]'], ar))
    md.append("\nListener AUC uses the 1-4 rating as the score (ties = 1/2); d' uses the binary judgment with the log-linear "
              "correction (+0.5 / +1). CIs: two-way bootstrap over listeners and clips (B = 2000; 500 for the mean listener AUC).\n")
    l01 = res.index[res['experience'] == 'musician']
    if len(l01):
        i = l01[0]
        md.append(f"\nThe single self-reported musician ({res.loc[i, 'listener']}): judged synthetic "
                  + ', '.join(f"{c} {int(J[i, cls_idx[c]].sum())}/{len(cls_idx[c])}" for c in REC + ALG) + '.\n')

    # ---------------- durations
    du = durations(key)
    du['judged_synth_k'] = J.sum(axis=0).astype(int)
    du['judged_synth_rate'] = J.mean(axis=0)
    du['mean_rating'] = R.mean(axis=0)
    det = os.path.join(OUT, 'stimuli_detector_scores.csv')
    if os.path.exists(det):
        ds = pd.read_csv(det)[['clip', 'xgb_all47_clip_LOGO', 'xgb_med16_clip_LOGO', 'prehard_src_stored']]
        du = du.merge(ds, on='clip', how='left')
    du.to_csv(os.path.join(OUT, 'listening_per_clip.csv'), index=False, float_format='%.4f')
    md.append('\n### 6.4 Effective sounding duration of the clips (duration cue)\n\n')
    md.append('Frame RMS (10 ms window, 5 ms hop) of each 3.00-s clip; t40 / t60 = end of the last frame within 40 / 60 dB of '
              'the clip\'s peak frame, measured from the clip start (the onset is at about 10 ms). Values >= 2.97 s are censored '
              '(the note is still above the threshold when the 10-ms fade-out ends the clip).\n\n')
    sm = []
    for c in REC + ALG:
        g = du[du.cls == c]

        def fmt(col):
            cens = g[f'{col}_censored']
            vals = g.loc[~cens, f'{col}_s']
            s = f"{int(cens.sum())}/{len(g)} censored"
            if len(vals):
                s += f"; others {vals.min():.2f}-{vals.max():.2f} s"
            return s
        sm.append([c, len(g), fmt('t40'), fmt('t60'), f"{g['last_nonzero_s'].min():.2f}-{g['last_nonzero_s'].max():.2f} s",
                   f"{100 * g['judged_synth_rate'].mean():.0f}%"])
    md.append(L.md_table(['class', 'clips', 't40', 't60', 'last non-zero sample', 'judged synth.'], sm))
    rho, p = spearmanr(du['t40_s'], du['judged_synth_rate'])
    rr = du[du.cls.isin(REC)]
    rho_r, p_r = spearmanr(rr['t40_s'], rr['judged_synth_rate'])
    md.append(f'\nSpearman between t40 and the per-clip judged-synthetic rate: all 30 clips rho = {rho:.2f} (p = {p:.3g}); '
              f'recorded clips only rho = {rho_r:.2f} (p = {p_r:.3g}). Within the design, duration is confounded with class '
              '(only the SoundFont clips stop early), so these correlations cannot separate a duration cue from a timbre cue.\n\n')
    key_out['durations'] = du[['clip', 'cls', 't40_s', 't40_censored', 't60_s', 't60_censored', 'last_nonzero_s']].to_dict('records')

    md.append('Per-clip listener data (`out/listening_per_clip.csv`):\n\n')
    md.append(L.md_table(['clip', 'class', 'note', 't40 (s)', 't60 (s)', 'judged synth. (of 32)', 'mean rating',
                          'detector P, XGB all47 LOGO'],
                         [[r.clip, r.cls, r.note, f"{r.t40_s:.2f}{'+' if r.t40_censored else ''}",
                           f"{r.t60_s:.2f}{'+' if r.t60_censored else ''}", r.judged_synth_k, f'{r.mean_rating:.2f}',
                           f"{r.xgb_all47_clip_LOGO:.3f}" if 'xgb_all47_clip_LOGO' in du else 'n/a'] for r in du.itertuples()]))
    md.append('("+" = censored at the clip end.)\n')

    # ---------------- listeners vs detector
    if os.path.exists(det):
        ds = pd.read_csv(det)
        md.append('\n### 6.5 Listeners vs the out-of-sample detector, same 30 clips\n\n')
        cmp = []
        for c in REC + ALG:
            ci = cls_idx[c]
            maj = int((J[:, ci].mean(axis=0) > 0.5).sum())
            row = [c, len(ci), f'{100 * J[:, ci].mean():.0f}%', f'{maj}/{len(ci)}']
            for col in ('xgb_all47_clip_LOGO', 'xgb_med16_clip_LOGO', 'prehard_src_stored'):
                p = ds.set_index('clip').loc[key['clip'].values[ci], col].values
                row.append(f'{int((p > 0.5).sum())}/{len(ci)}')
            cmp.append(row)
        aucrow = ['AUC recorded vs algorithmic', '20 vs 10', '', '']
        pan = auc_scores(R[:, rec_i].mean(axis=0), R[:, alg_i].mean(axis=0))
        aucrow[2] = f'panel {pan:.3f}'
        aucrow[3] = f'mean listener {np.mean([auc_scores(R[l, rec_i], R[l, alg_i]) for l in range(nL)]):.3f}'
        for col in ('xgb_all47_clip_LOGO', 'xgb_med16_clip_LOGO', 'prehard_src_stored'):
            p = ds.set_index('clip').loc[key['clip'].values, col].values
            aucrow.append(f'{auc_scores(p[rec_i], p[alg_i]):.3f}')
        cmp.append(aucrow)
        md.append(L.md_table(['class', 'clips', 'listeners: judged synth.', 'clips judged synth. by >50% of listeners',
                              'detector XGB all47 (out of sample)', 'detector XGB med16 (out of sample)',
                              'OLD in-sample detector (paper)'], cmp))
        md.append('\n')
    return ''.join(md), key_out


if __name__ == '__main__':
    t, k = main()
    print(t)
