"""Fusion judge of the quality tool (Sec. 3.8): a small MLP over the two regressor signals plus a liveness dock.

  (1) learned fusion: the raw outputs of the overall regressor (D, 52 descriptors) and of the timbre + liveness
      regressor (B+L, 45 descriptors), out of fold, -> MLP (8 hidden units) -> rating_cal -> calibration.py;
  (2) explicit liveness dock: summary = quality - guardrail - K * relu(iq - 0.5), K = 18, where iq is the file-level
      P(synthetic) of the I/Q liveness critic (liveness.py, via objectives.iq_synth).  The critic enters only this
      penalty, not the fused prediction, so absent modulation is penalized once.
The MLP is stored as JSON (weights and biases), not as a pickle.

    python -m quality_axes.judge --save      # fit the MLP + calibration -> results/quality_models
    SPD_AUDIO=<audio root> python -m quality_axes.judge <wav>...    # summary score (+ decomposition)
"""
from __future__ import annotations
import os
import json
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

from quality_axes.regressor import _oof
from detector.pipeline import FEATURE_COLS
from quality_axes.timbre import _load_with_liveness, TIMBRE_L_COLS
from quality_axes.combiner import FULL_L_COLS
from quality_axes.calibration import fit_calibration, apply_calibration, save_calibration, load_calibration

from quality_axes.models import MODELS, TRENDS, save_path   # noqa: E402
JUDGE = os.path.join(MODELS, 'judge_mlp.json')
JUDGE_ISO = os.path.join(MODELS, 'judge_iso.json')

# IQ-док: сумарна −= K·relu(iq − IQ_MAX).  Помірний — мертва модуляція знижує, не анулює.
IQ_DOCK_MAX = 0.5
IQ_DOCK_K = 18.0


def _signals_oof():
    """OOF-сигнали двох XGBoost-моделей (без витоку) + рейтинг-ціль."""
    rows, y, anchor, w, groups, X = _load_with_liveness()
    d_raw = _oof(FULL_L_COLS, X, y, w, groups)      # сигнал D (усе)
    b_raw = _oof(TIMBRE_L_COLS, X, y, w, groups)    # сигнал B+L (тембр+живість)
    S = np.column_stack([d_raw, b_raw])
    return S, y, w, anchor


def _mlp():
    return MLPRegressor(hidden_layer_sizes=(8,), activation='relu', alpha=1e-2,
                        max_iter=2000, random_state=0)


def train_and_save():
    S, y, w, anchor = _signals_oof()
    mlp = _mlp().fit(S, y)                 # sample_weight не підтримується MLP → рівні ваги
    fused = mlp.predict(S)
    cal = fit_calibration(fused, y)
    # persist MLP (coefs/intercepts) as JSON so no pickle
    js = {'W': [c.tolist() for c in mlp.coefs_],
          'b': [c.tolist() for c in mlp.intercepts_]}
    json.dump(js, open(save_path('judge_mlp.json'), 'w'))
    save_calibration(cal, save_path('judge_iso.json'))
    print(f"saved judge MLP + calibration → {save_path('judge_mlp.json')}")
    print(f"fusion ρ(ALL)={spearmanr(fused, y).statistic:.3f}  "
          f"calibrated=(апрокс. D)  IQ-док K={IQ_DOCK_K} @>{IQ_DOCK_MAX}")
    return mlp, cal


def _mlp_forward(S: np.ndarray, js: dict) -> np.ndarray:
    a = S
    Ws, bs = js['W'], js['b']
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ np.array(W) + np.array(b)
        if i < len(Ws) - 1:
            a = np.maximum(a, 0.0)         # relu, крім вихідного шару
    return a.ravel()


# ---- inference --------------------------------------------------------------
def _raw_signals(path: str):
    """D_raw, B+L_raw для одного файлу (спільна екстракція)."""
    import xgboost as xgb
    from quality_axes.score_files import _feat
    from quality_axes.modulation import modulation_features
    from detector.segment import load_mono
    trends = json.load(open(TRENDS, encoding='utf-8'))
    xt, n = _feat(path, trends)
    wav, sr = load_mono(path)
    m = modulation_features(wav, sr)

    def raw(model_path):
        mdl = xgb.XGBRegressor(); mdl.load_model(model_path)
        cols = json.load(open(model_path + '.cols'))
        fidx = {c: FEATURE_COLS.index(c) for c in cols if c in FEATURE_COLS}
        vec = [xt[fidx[c]] if c in fidx else m.get(c, np.nan) for c in cols]
        return float(mdl.predict(np.array(vec)[None, :])[0])

    d = raw(os.path.join(MODELS, 'combiner_overall.json'))
    b = raw(os.path.join(MODELS, 'timbre_quality.json'))
    return d, b, n


def judge_from_signals(d_raw: float, b_raw: float, iq: float,
                       guardrail: float = 0.0) -> dict:
    """Сумарна оцінка з уже порахованих сигналів (D_raw, B+L_raw, IQ, G) — щоб єдиний
    звіт не робив подвійну екстракцію.  quality = калібр. fusion(D,B);
    summary = quality − guardrail(phaser/aversiveness) − IQ-док  (тож суддя ніколи
    не м'якший за D — лише додає IQ-док зверху)."""
    js = json.load(open(JUDGE))
    cal = load_calibration(JUDGE_ISO)
    fused = float(_mlp_forward(np.array([[d_raw, b_raw]]), js)[0])
    quality = float(apply_calibration(fused, cal))
    dock = IQ_DOCK_K * max(0.0, (iq - IQ_DOCK_MAX)) if np.isfinite(iq) else 0.0
    summary = float(np.clip(quality - guardrail - dock, 0.0, 99.0))
    return {'summary': summary, 'quality': quality, 'iq': iq, 'iq_dock': dock}


def judge_score(path: str) -> dict:
    from quality_axes.objectives import iq_synth
    from quality_axes.combiner import guardrail_penalty
    d_raw, b_raw, n = _raw_signals(path)
    out = judge_from_signals(d_raw, b_raw, iq_synth(path), guardrail_penalty(path, n))
    out.update({'d_raw': d_raw, 'b_raw': b_raw, 'n_notes': n, 'path': path})
    return out


def _fmt(r: dict) -> str:
    iqs = f"{r['iq']:.2f}" if np.isfinite(r['iq']) else "—"
    dock = f"  −{r['iq_dock']:.0f} (IQ-мертвість)" if r['iq_dock'] > 0.5 else ""
    return (f"  {os.path.basename(r['path'])}  (n={r['n_notes']})\n"
            f"    якість (fusion)   {r['quality']:5.1f}\n"
            f"    IQ p_synth        {iqs}\n"
            f"    СУМАРНА (суддя)   {r['summary']:5.1f}{dock}")


def main():
    if '--save' in sys.argv:
        train_and_save()
        return
    files = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not files:
        print("Вкажи WAV-файл(и) або --save."); return
    for f in files:
        try:
            print(_fmt(judge_score(f)))
        except Exception as e:  # noqa: BLE001
            print(f"  {os.path.basename(f)}: ERR {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
