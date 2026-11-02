"""Sec. 4.4: listeners and the fusion detector on the 30 clips of the listening test.

Listeners (data/listening/answer_key.csv, results_anon.csv; 32 listeners x 30 clips, ratings 1-4):
  - "judged synthetic" = rating 3 or 4; the share is pooled over listeners and clips of a class, with a 95 % interval
    from a bootstrap over listeners (B = 10000, seed 0, 2.5 / 97.5 percentiles);
  - listener AUC = Mann-Whitney AUC of one listener's ratings, algorithmic (positive) vs recorded clips, ties 1/2;
    mean over listeners with the same bootstrap; one-sided Wilcoxon signed-rank test of (AUC - 0.5) > 0, zero
    differences split;
  - panel AUC = the same AUC on the rating of each clip averaged over all listeners; a clip is "called synthetic by
    the majority" when more than half of the listeners rated it 3 or 4.
Detector (data/predictions/listening_clips.csv): the fusion of the phaseogram CNN and the 27 isolated-note features,
applied to the clips exactly as the listeners heard them.  The recorded clips are scored by the fusion trained on the
whole isolated-note corpus (sample libraries are never trained; these values are the clip rows of
data/predictions/notes_oof.csv); each algorithmic clip by a fusion trained on the corpus without the notes of its own
engine (Otey or the Casio CA-110 SoundFont) at the pitches of the clips, all velocities -- new notes of a known engine.
Threshold = 95th percentile of the out-of-fold fusion scores of the real CCMusic notes (5 % false alarms on real notes),
from data/predictions/notes_oof.csv.

Stimuli (Sec. 3.7, data/listening/S3_test1_stimuli.csv): 20 recorded (Iowa 7, K18 7, Salamander 6) and 10 algorithmic
(SoundFont 5, Otey 5) clips, 3 s, 44.1 kHz, RMS -20 dBFS; four clips lowered to the peak limit, 0.3-1.5 dB quieter.

Paper numbers (Sec. 4.4): algorithmic 65 % [58-72] (SoundFont 67, Otey 63), recorded 37 % [32-42]; mean listener AUC
0.67 (0.61-0.71), 27 of 32 above 0.5, Wilcoxon p < 1e-6, best 0.96; 63 % of the answers on library clips "real";
panel AUC 0.98; majority: SoundFont 5/5, Otey 4/5, recorded 2/20; detector P > 0.941 flags SoundFont 4/5, Otey 4/5,
recorded 1/20, AUC 0.965; of the 200 recorded x algorithmic pairs the panel orders 196.5 correctly and the detector
193, a difference of 3.5 pairs ("fewer than four").

    python experiments/04_listening/listening_test.py   -> results/04_listening/listening_test.md
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402

OUT = results_dir('04_listening')
CLASS = {'Iowa Steinway': 'Iowa', 'K18 upright': 'K18', 'Salamander (sampled)': 'Salamander',
         'soundfont': 'SoundFont', 'Otey physical model': 'Otey'}
REC, ALG = ['Iowa', 'K18', 'Salamander'], ['SoundFont', 'Otey']
B, SEED = 10000, 0


def auc(neg, pos):
    """Mann-Whitney AUC, ties count 1/2."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def correct_pairs(neg, pos):
    d = np.asarray(pos, float)[:, None] - np.asarray(neg, float)[None, :]
    return float((d > 0).sum() + 0.5 * (d == 0).sum())


def main():
    key = pd.read_csv(os.path.join(DATA, 'listening', 'answer_key.csv'))
    res = pd.read_csv(os.path.join(DATA, 'listening', 'results_anon.csv'))
    clips = list(key['clip'])
    R = res[clips].values.astype(int)
    assert R.shape == (32, 30) and R.min() >= 1 and R.max() <= 4
    cls = key['source'].map(CLASS).values
    J = (R >= 3).astype(float)
    rec, alg = np.isin(cls, REC), np.isin(cls, ALG)
    nL = R.shape[0]
    boot = np.random.default_rng(SEED).integers(0, nL, (B, nL))      # one resampling of listeners for every statistic

    def ci(per_listener):
        v = np.asarray(per_listener)[boot].mean(axis=1)
        return np.percentile(v, [2.5, 97.5])

    st = pd.read_csv(os.path.join(DATA, 'listening', 'S3_test1_stimuli.csv')).set_index('clip').loc[clips]
    low = st[st['rms_dbfs'] < -20.005]
    md = ['# Listening test: listeners and the fusion detector on the 30 clips\n\n',
          '## Stimuli (data/listening/S3_test1_stimuli.csv)\n\n',
          ''.join(f'- {k}: {v} clips\n' for k, v in st.groupby('instrument', sort=False).size().items()),
          f'- duration {sorted(set(st.duration_s))} s, sample rate {sorted(set(st.sample_rate_hz))} Hz, '
          f'RMS -20 dBFS except {len(low)} clips lowered to a peak of {low.peak_dbfs.max():.2f} dBFS: '
          f'{", ".join(f"{c} {-20 - v:.2f} dB quieter" for c, v in low.rms_dbfs.items())}\n\n',
          '## Listeners\n\n',
          f'{nL} listeners x {len(clips)} clips = {R.size} answers. Judged synthetic = rating 3 or 4. 95 % intervals: '
          f'bootstrap over listeners (B = {B}, seed {SEED}).\n\n',
          '| class | clips | judged synthetic | 95 % interval | judged real | clips called synthetic by > 50 % of listeners |\n'
          '|---|---|---|---|---|---|\n']
    rows = [(c, cls == c) for c in REC + ALG] + [('recorded (3 libraries)', rec), ('algorithmic', alg)]
    for name, m in rows:
        lo, hi = ci(J[:, m].mean(axis=1))
        maj = int((J[:, m].mean(axis=0) > 0.5).sum())
        md.append(f'| {name} | {int(m.sum())} | {100 * J[:, m].mean():.1f} % | {100 * lo:.1f}-{100 * hi:.1f} % | '
                  f'{100 * (1 - J[:, m].mean()):.1f} % | {maj}/{int(m.sum())} |\n')
    lib_real = np.mean([1 - J[:, cls == c].mean() for c in REC])
    md.append(f'\nJudged real, mean over the three libraries: {100 * lib_real:.1f} % (pooled over the 20 recorded clips: '
              f'{100 * (1 - J[:, rec].mean()):.1f} %).\n')
    maj_rec = [f'{clips[i]} ({cls[i]})' for i in np.flatnonzero(rec & (J.mean(axis=0) > 0.5))]
    md.append(f'Recorded clips called synthetic by the majority: {", ".join(maj_rec)}.\n\n')

    la = np.array([auc(R[i, rec], R[i, alg]) for i in range(nL)])
    lo, hi = ci(la)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pw = float(wilcoxon(la - 0.5, alternative='greater', zero_method='zsplit').pvalue)
    mr = R.mean(axis=0)
    panel = auc(mr[rec], mr[alg])
    md.append('## Listener AUC (recorded vs algorithmic)\n\n'
              f'- mean listener AUC {la.mean():.4f} (95 % interval {lo:.3f}-{hi:.3f}); median {np.median(la):.4f}\n'
              f'- above 0.5: {int((la > .5).sum())} of {nL}; equal: {int((la == .5).sum())}; below: {int((la < .5).sum())}\n'
              f'- one-sided Wilcoxon signed-rank (AUC - 0.5 > 0, zero differences split): p = {pw:.2g}\n'
              f'- best listener {la.max():.4f}\n'
              f'- panel AUC (mean rating per clip over all listeners) {panel:.4f}\n\n')

    t = pd.read_csv(os.path.join(DATA, 'predictions', 'listening_clips.csv')).set_index('clip').loc[clips]
    p = t['p_fusion'].values
    o = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_oof.csv'))
    cc = ((o.source == 'ccmusic') & o.corpus).values
    t5 = float(np.quantile(o.loc[cc, 'fus27_p'], 0.95))
    oc = o[o.source == 'clip'].set_index('file')['fus27_p']
    same = np.allclose(oc.loc[t.loc[rec, 'file']].values, p[rec], rtol=0, atol=1e-12) if rec.any() else False
    md.append('## Fusion detector on the same clips\n\n'
              f'Threshold, 5 % false alarms on the real CCMusic notes (out-of-fold): P > {t5:.4f}. '
              f'Recorded-clip scores equal the clip rows of notes_oof.csv: {same}.\n\n'
              '| class | clips | flagged | P(synthetic) |\n|---|---|---|---|\n')
    for name, m in rows:
        md.append(f'| {name} | {int(m.sum())} | {int((p[m] > t5).sum())}/{int(m.sum())} | '
                  f'{", ".join(f"{v:.3f}" for v in p[m])} |\n')
    dauc = auc(p[rec], p[alg])
    cp_panel, cp_det = correct_pairs(mr[rec], mr[alg]), correct_pairs(p[rec], p[alg])
    dp = (mr[alg][:, None] - mr[rec][None, :])
    dd = (p[alg][:, None] - p[rec][None, :])
    disc = int((np.sign(dp) * np.sign(dd) < 0).sum())
    ties = int(((dp == 0) | (dd == 0)).sum())
    n_pairs = int(rec.sum() * alg.sum())
    md.append(f'\nDetector AUC on the 30 clips: {dauc:.4f}.\n\n'
              f'Of the {n_pairs} recorded x algorithmic pairs, the panel orders {cp_panel:.1f} correctly (ties 1/2) and '
              f'the detector {cp_det:.1f}: a difference of {cp_panel - cp_det:.1f} pairs. Pairs that the two order in '
              f'opposite directions: {disc} (pairs tied in either: {ties}).\n\n')

    md.append('## Per clip\n\n| clip | class | note | judged synthetic (of 32) | mean rating | fusion P | flagged |\n'
              '|---|---|---|---|---|---|---|\n')
    for i, c in enumerate(clips):
        md.append(f'| {c} | {cls[i]} | {key["note"].iloc[i]} | {int(J[:, i].sum())} | {mr[i]:.2f} | {p[i]:.3f} | '
                  f'{"yes" if p[i] > t5 else ""} |\n')
    open(os.path.join(OUT, 'listening_test.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
