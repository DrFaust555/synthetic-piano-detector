"""External baseline: openSMILE eGeMAPS features + the SAME grouped-CV real-vs-synth, on the
citable subset, to contextualize the channel-robust 47-feature critic (paper Sec. 4.1).

    python -m quality.opensmile_baseline

Standard eGeMAPSv02 functionals (88) are NOT engineered for channel invariance, so the
comparison shows how much our channel-robust design buys, and whether a generic descriptor
set reproduces the algorithmic-vs-sample-based split.
"""

from __future__ import annotations
import os
import glob
import random
import re
import numpy as np
import opensmile
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from detune.segment import load_mono

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED, TAKE_SIZE, N_TAKES = 0, 14, 8
LIBS = {
    'Salamander': 'audio/ref_bright_yamaha_salamander/**/*.ogg',
    'K18':        'audio/ref_dull_upright/**/*.wav',
    'FreePats':   'audio/ref_bad_old_piano/**/*.wav',
    'Iowa':       'audio/ref_top_grand_iowa_steinway/**/*.aif*',
}


def _note(name):
    m = re.search(r'([A-G]#?b?)(-?\d)', os.path.basename(name))
    return (m.group(1) + m.group(2)) if m else None


def _boot_ci(y, p, grp, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = np.unique(grp); g2 = {g: np.where(grp == g)[0] for g in gs}
    a = []
    for _ in range(n):
        idx = np.concatenate([g2[g] for g in rng.choice(gs, len(gs), replace=True)])
        if len(np.unique(y[idx])) == 2:
            a.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    rng = random.Random(SEED)
    X, y, grp, st = [], [], [], []

    def add_file(path, label, stype, group):
        try:
            v = smile.process_file(path).values[0]
            X.append(v); y.append(label); grp.append(group); st.append(stype)
        except Exception:
            pass

    # real = CCMusic (group by piano-name prefix); algorithmic = renders + Otey
    cc = glob.glob(os.path.join(ROOT, 'audio/ccmusic_pianos/wav/*.wav'))
    print(f"CCMusic {len(cc)}", flush=True)
    for i, f in enumerate(cc):
        add_file(f, 0, 'real', 'cc:' + os.path.basename(f).split('_')[0])
        if (i + 1) % 150 == 0:
            print(f"  cc {i+1}", flush=True)
    for d in sorted(glob.glob(os.path.join(ROOT, 'audio/SynthPianoSamples/rendered/*/'))):
        inst = os.path.basename(os.path.normpath(d))
        if inst == '_demos':
            continue
        for f in glob.glob(os.path.join(d, '*.wav')):
            add_file(f, 1, 'rendered', 'syn:rendered:' + inst)
    print(f"after renders: {len(X)}", flush=True)
    for f in glob.glob(os.path.join(ROOT, 'otey/out_full/*.wav')):
        add_file(f, 1, 'otey', 'syn:otey')

    # sample libraries: same take composition (concatenate the take's note-files)
    for lib, g in LIBS.items():
        files = sorted(glob.glob(os.path.join(ROOT, g), recursive=True))
        by = {}
        for f in files:
            t = _note(f)
            if t and t not in by:
                by[t] = f
        toks = list(by)
        for k in range(N_TAKES):
            sample = rng.sample(toks, min(TAKE_SIZE, len(toks)))
            sig = []
            for t in sample:
                try:
                    x, sr = load_mono(by[t]); sig.append(x)
                except Exception:
                    pass
            if not sig:
                continue
            try:
                v = smile.process_signal(np.concatenate(sig), sr).values[0]
                X.append(v); y.append(1); grp.append('syn:lib:' + lib); st.append('sampled_digital')
            except Exception:
                pass
    X = np.array(X); y = np.array(y); grp = np.array(grp); st = np.array(st)
    print(f"\nopenSMILE eGeMAPS units: {len(y)}  real={int((y==0).sum())} synth={int((y==1).sum())}", flush=True)

    spw = (y == 0).sum() / max(1, (y == 1).sum())
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                eval_metric='auc', tree_method='hist', n_jobs=4)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]

    ALGO = np.isin(st, ['rendered', 'otey']); SAMP = st == 'sampled_digital'
    print("\nopenSMILE eGeMAPS baseline (grouped CV, citable subset):")
    lo, hi = _boot_ci(y, oof, grp)
    print(f"  AUC real vs ALL synth     : {roc_auc_score(y, oof):.3f}  [{lo:.3f}, {hi:.3f}]")
    for nm, mk in [('ALGORITHMIC', ALGO), ('SAMPLE-BASED', SAMP)]:
        sel = (y == 0) | mk
        lo, hi = _boot_ci(y[sel], oof[sel], grp[sel])
        print(f"  AUC real vs {nm:12s}  : {roc_auc_score(y[sel], oof[sel]):.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\n  (compare: our channel-robust 47-feat citable-only = 0.874 all / 0.899 algo / 0.208 sampled)")


if __name__ == '__main__':
    main()
