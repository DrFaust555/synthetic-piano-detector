"""Train the real-vs-synthetic realness critic on the 47 channel-robust features.

    python -m quality.train_realness

REAL  (label 0) = features_cal.csv minus the m1 synth patches (acoustic pianos).
SYNTH (label 1) = synth_features.csv (soundfont renders + cleaned YouTube digitals +
                  Otey physical model) + the m1 patches from features_cal.

Anti-leakage: GroupKFold by instrument identity (no instrument in both train and test).
The score we care about is P(real)·100 = an "aliveness" gauge — near-real digitals
(Kurzweil, Roland GP) should land high (hard to catch = alive, like Pianoteq), dead
synths (SC-55, m1, cheap renders) low.  Reports out-of-fold AUC + per-synth-type aliveness.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from quality.pipeline import FEATURE_COLS
from quality.grouping import group_key

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
REAL = os.path.join(CACHE, 'features_cal.csv')
SYNTH = os.path.join(CACHE, 'synth_features.csv')
VERSIONS = os.path.join(CACHE, 'versions')
MAESTRO_FEATS = os.path.join(CACHE, 'features_maestro.csv')
NEG_FEATS = os.path.join(CACHE, 'features_negatives.csv')


def _load():
    rdf = pd.read_csv(REAL)
    is_m1 = rdf['file'].str.contains('m1_patches', na=False)
    real = rdf[~is_m1].copy()
    real['y'] = 0
    real['stype'] = 'real'
    real['grp'] = [group_key(f, b) for f, b in zip(real['file'], real.get('brand', ''))]

    sdf = pd.read_csv(SYNTH)
    sdf['y'] = 1
    sdf['stype'] = sdf['synth_type']
    sdf['grp'] = 'syn:' + sdf['synth_type'].astype(str) + ':' + sdf['instrument'].astype(str)

    m1 = rdf[is_m1].copy()
    m1['y'] = 1
    m1['stype'] = 'm1'
    m1['grp'] = 'syn:m1'

    keep = ['file', 'y', 'stype', 'grp'] + FEATURE_COLS
    parts = [real[keep], sdf[keep], m1[keep]]
    slp = os.path.join(CACHE, 'samplelib_features.csv')          # free sample libraries
    if os.path.exists(slp):
        ldf = pd.read_csv(slp)
        ldf['y'] = 1
        ldf['stype'] = 'sampled_digital'
        ldf['grp'] = 'syn:samplelib:' + ldf['instrument'].astype(str)
        parts.append(ldf[keep])
    df = pd.concat(parts, ignore_index=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    # drop rows that are all-NaN over features (extraction failures)
    df = df[df[FEATURE_COLS].notna().sum(axis=1) >= 30].reset_index(drop=True)
    return df


def _load_extra_positives():
    """MAESTRO-позитиви (features_maestro.csv, y=0). Група = РІК (один Disklavier/канал
    на рік конкурсу; per-clip групи пускали б той самий інструмент/канал по обидва боки
    фолда → оптимістичний CV). Ім'я кліпа = '<year>_<hash>.wav'."""
    if not os.path.exists(MAESTRO_FEATS):
        return None
    df = pd.read_csv(MAESTRO_FEATS)
    df['y'] = 0
    df['stype'] = 'maestro'
    base = df['file'].astype(str).str.replace('\\', '/').str.split('/').str[-1]
    df['grp'] = 'maestro:' + base.str.split('_').str[0]
    return df


def _load_extra_negatives():
    """Новорендерені негативи (features_negatives.csv, y=1). Всі otey_* — ОДНА група
    'syn:otey:Otey', як у базовому наборі: один синт-движок не має стояти по обидва
    боки фолда під різними іменами груп."""
    if not os.path.exists(NEG_FEATS):
        return None
    df = pd.read_csv(NEG_FEATS)
    df['y'] = 1
    df['stype'] = df['family'].astype(str)
    fam = df['family'].astype(str)
    df['grp'] = np.where(fam.str.startswith('otey'), 'syn:otey:Otey', 'neg:' + fam)
    return df


def _fit_cv_and_save(df, tag='v2'):
    """GroupKFold-по-інструменту CV + збереження ВЕРСІОНОВАНОГО артефакту (старий не чіпаємо)."""
    os.makedirs(VERSIONS, exist_ok=True)
    X = df[FEATURE_COLS].values
    y = df['y'].values
    grp = df['grp'].values
    spw = float(np.sum(y == 0)) / max(1, np.sum(y == 1))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=grp):
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                eval_metric='auc', tree_method='hist', n_jobs=4)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    print(f"\n[{tag}] Grouped CV-AUC (real vs synth): {auc:.3f}")
    fams = df[df.y == 1]['stype'].value_counts().to_dict()
    print(f"[{tag}] synth families: {fams}")
    df = df.copy(); df['alive'] = (1 - oof) * 100
    for st, g in df[df.y == 1].groupby('stype'):
        print(f"    {st:14s} n={len(g):4d}  alive={g['alive'].mean():5.1f}  "
              f"caught@50={(oof[g.index] > 0.5).mean()*100:4.0f}%")
    final = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              eval_metric='auc', tree_method='hist', n_jobs=4)
    final.fit(X, y)
    model_path = os.path.join(VERSIONS, f'realness_xgb_{tag}.json')
    final.save_model(model_path)
    import json
    json.dump({'features': FEATURE_COLS, 'cv_auc': round(auc, 4),
               'protocol': 'GroupKFold (5): real=instrument, maestro=year, otey*=one engine group',
               'positive_families': sorted(set(df[df.y == 0]['stype'])),
               'negative_families': sorted(set(df[df.y == 1]['stype'])),
               'n_real': int((y == 0).sum()), 'n_synth': int((y == 1).sum()),
               'note': 'P(synth) in [0,1]; MAESTRO positives ~70% channel-augmented'},
              open(os.path.join(VERSIONS, f'realness_xgb_{tag}.meta.json'), 'w'), indent=2)
    print(f"[{tag}] saved {os.path.relpath(model_path, HERE)} (+meta) — старий realness_xgb.json НЕ чіпано.")
    return auc


def train_v2():
    """Розширений набір: старі real+synth + MAESTRO-позитиви + банк негативів."""
    base = _load()
    keep = ['file', 'y', 'stype', 'grp'] + FEATURE_COLS
    parts = [base[keep]]
    for extra, name in ((_load_extra_positives(), 'MAESTRO'),
                        (_load_extra_negatives(), 'negatives')):
        if extra is None:
            print(f"[v2] {name}: таблиці нема — пропуск (спершу maestro_ingest --features / "
                  f"build_negatives --features)")
            continue
        extra[FEATURE_COLS] = extra[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
        parts.append(extra[keep])
        print(f"[v2] +{name}: {len(extra)} рядків")
    df = pd.concat(parts, ignore_index=True)
    df = df[df[FEATURE_COLS].notna().sum(axis=1) >= 30].reset_index(drop=True)
    print(f"[v2] REAL {int((df.y==0).sum())}  SYNTH {int((df.y==1).sum())}  "
          f"groups {len(set(df.grp))}")
    return _fit_cv_and_save(df, tag='v2')


def _grp_boot_ci(y, p, grp, n=2000, seed=0):
    """Grouped (instrument-clustered) bootstrap 95% CI for AUC: resample instruments
    with replacement, recompute AUC over their rows."""
    rng = np.random.default_rng(seed)
    groups = np.unique(grp)
    g2idx = {g: np.where(grp == g)[0] for g in groups}
    aucs = []
    for _ in range(n):
        idx = np.concatenate([g2idx[g] for g in rng.choice(groups, len(groups), replace=True)])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    if '--v2' in sys.argv:                       # розширений набір -> versions/realness_xgb_v2.json
        train_v2()
        return
    df = _load()
    if '--citable' in sys.argv:                  # fully-reproducible-data subset
        keep = ((df.y == 0) & df.file.str.startswith('ccmusic_pianos/')) | \
               (df.stype.isin(['rendered', 'otey', 'sampled_digital']))
        df = df[keep].reset_index(drop=True)
        print("[CITABLE-ONLY] real=CCMusic, synth=soundfont-renders+Otey+free-libraries")
    X = df[FEATURE_COLS].values
    y = df['y'].values
    grp = df['grp'].values
    print(f"REAL {np.sum(y==0)}  SYNTH {np.sum(y==1)}  (groups: {len(set(grp))})")
    by = df[df.y == 1]['stype'].value_counts().to_dict()
    print("synth breakdown:", by)

    spw = float(np.sum(y == 0)) / max(1, np.sum(y == 1))
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    aucs = []
    imp = np.zeros(len(FEATURE_COLS))
    for tr, te in gkf.split(X, y, groups=grp):
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric='auc', tree_method='hist', n_jobs=4)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
        imp += clf.feature_importances_
    auc = roc_auc_score(y, oof)
    print(f"\nGrouped CV-AUC (real vs synth): {auc:.3f}   (per-fold {np.mean(aucs):.3f}±{np.std(aucs):.3f})")

    # THE CEILING, quantified: AUC vs ALGORITHMIC synth (caught) vs SAMPLE-BASED (ceiling)
    st = df['stype'].values
    ALGO = np.isin(st, ['rendered', 'otey'])             # algorithmic / physical-model
    SAMP = np.isin(st, ['sampled_digital', 'yt', 'm1'])  # sample-based (recordings)
    print("  (with grouped/instrument-clustered bootstrap 95% CI)")
    lo, hi = _grp_boot_ci(y, oof, grp)
    print(f"  AUC real vs ALL synth {' ':21s}: {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
    for name, msk in [('vs ALGORITHMIC synth (soundfont+Otey)', ALGO),
                      ('vs SAMPLE-BASED (free libs+digitals+m1)', SAMP)]:
        sel = (y == 0) | msk
        lo, hi = _grp_boot_ci(y[sel], oof[sel], grp[sel])
        print(f"  AUC real {name:42s}: {roc_auc_score(y[sel], oof[sel]):.3f}  [{lo:.3f}, {hi:.3f}]")

    # aliveness = P(real)*100 ; per synth type (higher = more real-like = 'alive')
    df['alive'] = (1 - oof) * 100
    print("\nAliveness P(real)*100 by synth type (higher = harder to catch = more alive):")
    for st, g in df[df.y == 1].groupby('stype'):
        print(f"  {st:10s} n={len(g):4d}  alive={g['alive'].mean():5.1f}  "
              f"caught@50={(oof[g.index] > 0.5).mean()*100:4.0f}%")
    print(f"\n  real        n={np.sum(y==0):4d}  alive={df[df.y==0]['alive'].mean():5.1f}")

    # per-instrument aliveness for the YouTube digitals (the near-real hard cases)
    yt = df[(df.y == 1) & (df.stype == 'yt')]
    if len(yt):
        print("\n  YouTube digital pianos (per instrument):")
        for inst, g in yt.groupby(yt['file'].str.split('/').str[-1].str.split('_').str[0]):
            print(f"    {inst:12s} alive={g['alive'].mean():5.1f}  n={len(g)}")

    order = np.argsort(imp)[::-1]
    print("\nTop realness features:")
    for i in order[:12]:
        print(f"  {FEATURE_COLS[i]:26s} {imp[i]/5:.3f}")

    # ── robustness: median-only features (drop across-note IQR, which is degenerate
    #    for single-note render files → removes the single-note shortcut) ──
    med_cols = [c for c in FEATURE_COLS if not c.endswith('_iqr')]
    Xm = df[med_cols].values
    oofm = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(Xm, y, groups=grp):
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric='auc', tree_method='hist', n_jobs=4)
        clf.fit(Xm[tr], y[tr])
        oofm[te] = clf.predict_proba(Xm[te])[:, 1]
    print(f"\n[median-only, {len(med_cols)} feats] CV-AUC: {roc_auc_score(y, oofm):.3f}")
    dfm = df.copy(); dfm['alive'] = (1 - oofm) * 100
    for st, g in dfm[dfm.y == 1].groupby('stype'):
        print(f"    {st:10s} alive={g['alive'].mean():5.1f}  caught@50={(oofm[g.index] > 0.5).mean()*100:4.0f}%")

    # ── deploy: fit final model on ALL data, save as a reusable realness scorer ──
    if '--citable' not in sys.argv:              # don't overwrite the deployable model with a subset
        final = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric='auc', tree_method='hist', n_jobs=4)
        final.fit(X, y)
        import json
        final.save_model(os.path.join(CACHE, 'realness_xgb.json'))
        json.dump({'features': FEATURE_COLS, 'cv_auc': round(auc, 4),
                   'note': 'P(synth) in [0,1]; aliveness = (1-P)*100'},
                  open(os.path.join(CACHE, 'realness_xgb.meta.json'), 'w'), indent=2)
        print("\nSaved realness_xgb.json (final model on all data).")


if __name__ == '__main__':
    main()
