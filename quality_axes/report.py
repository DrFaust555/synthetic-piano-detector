"""Quality report of one recording (Sec. 3.8): tuning, timbre, liveness and overall axes, plus per-file detail
cards (analytic regressor, aversiveness, modulation, phaser index).

  timbre    timbre_model.py regressor on the 40 features of Groups A and B, isotonic + affine calibration
  tuning    100 * exp(-d / 12 cents), d = median deviation of the notes from the instrument's entropy-minimizing
            target with one common offset removed (tuning/measure.py)
  liveness  100 * (1 - median P(synthetic) of the I/Q liveness critic over up to 12 notes) (liveness.py)
  overall   timbre - guardrail - live dock;  guardrail = max(0, (aversiveness - 35) / 5) + 3 if phaser index > 3.5;
            live dock = max(0, (30 - liveness) / 30 * 10)
Provenance is the detector of Sec. 3.2 (data/predictions or detector/); full_axes() takes it as an argument and
never uses it in a quality axis.

    SPD_AUDIO=<audio root> python -m quality_axes.report <wav> [<wav> ...] [--json out.json] [--cards]
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np

from quality_axes.score_files import _train, _feat, TRENDS, REFS, ROOT
from quality_axes.phaser import phaser_index
from quality_axes.objectives import phaser_penalty
from quality_axes.aversiveness import aversiveness_features
from quality_axes.modulation import modulation_features
from detector.segment import load_mono, build_segments

# --- full §5 decomposition deps (provenance / tuning / timbre / liveness) ------
from quality_axes.timbre_model import _train as _train_timbre, _calibration, TIMBRE_COLS  # noqa: E402
from detector.pipeline import note_raw_descriptors, aggregate_file                    # noqa: E402
from quality_axes.liveness import p_synth_iq                                             # noqa: E402
from quality_axes.tuning.measure import tuning_score as _tuning_score                         # noqa: E402


def score_one(path: str, model, trends) -> dict:
    """Full metric bundle for one WAV (realness + aversiveness + modulation)."""
    x, n = _feat(path, trends)
    p = float(model.predict(x[None, :])[0])
    wav, sr = load_mono(path)
    ph = float(phaser_index(wav, sr))
    gated = p - phaser_penalty(path, idx=ph)        # phaser red-flag penalty
    av = aversiveness_features(wav, sr)
    mod = modulation_features(wav, sr)
    return {
        'file': os.path.basename(path),
        'notes': int(n),
        'realness_xgb': round(p, 1),
        'realness_gated': round(gated, 1),
        'phaser': round(ph, 2),
        'aversiveness': round(float(av['aversiveness']), 0),
        'aver_components': {k: round(float(av[k]), 3) for k in
                            ('aver_rough', 'aver_sharp', 'aver_noise',
                             'aver_impulse', 'aver_loud')},
        'modulation': {k: round(float(mod[k]), 3) for k in
                       ('beat_mod', 'beat_flatness', 'fm_wander',
                        'slow_coherent', 'beat_to_slow')},
    }


def _verdict(r: dict) -> list[str]:
    """Human-readable interpretation lines for a metric bundle."""
    out = []
    g = r['realness_gated']
    if g >= 38:
        out.append("realness: висока (у діапазоні топ-реальних записів)")
    elif g >= 25:
        out.append("realness: помірна")
    else:
        out.append("realness: низька (ближче до синтетичних)")
    if r['phaser'] > 3.5:
        out.append(f"  ⚑ фленджер/фейзер артефакт (phaser {r['phaser']})")
    av = r['aversiveness']
    if av >= 50:
        out.append(f"aversiveness: {av:.0f}/100 — ПРОТИВНО (шорсткість/різкість/шум)")
    elif av >= 25:
        out.append(f"aversiveness: {av:.0f}/100 — помітна")
    else:
        out.append(f"aversiveness: {av:.0f}/100 — низька (приємно)")
    bf = r['modulation']['beat_flatness']
    if bf >= 0.30:
        out.append(f"modulation/«життя»: живе (beat_flatness {bf:.2f})")
    else:
        out.append(f"modulation/«життя»: мляве/«мертве» (beat_flatness {bf:.2f})")
    return out


def _print_card(r: dict) -> None:
    print("=" * 60)
    print(f"ГЛИБОКИЙ ЗВІТ ЯКОСТІ — {r['file']}  ({r['notes']} нот)")
    print("-" * 60)
    print(f"  realness (xgb)   {r['realness_xgb']:6.1f}    gated {r['realness_gated']:6.1f}")
    print(f"  aversiveness     {r['aversiveness']:6.0f} / 100")
    print(f"  modulation flat  {r['modulation']['beat_flatness']:6.2f}    "
          f"beat_mod {r['modulation']['beat_mod']:.2f}  fm_wander {r['modulation']['fm_wander']:.2f}")
    print(f"  phaser index     {r['phaser']:6.2f}")
    print("  --")
    for ln in _verdict(r):
        print("  " + ln)


# --------------------------------------------------------------- full §5 axes
def _timbre_axis(notes, tuning, trends, tmodel, calib) -> float:
    row = aggregate_file(notes, tuning, trends)
    x = np.array([row.get(c, np.nan) for c in TIMBRE_COLS])
    raw = float(tmodel.predict(x[None, :])[0])
    iso = float(np.interp(raw, calib['iso_x'], calib['iso_y']))
    return float(np.clip(calib['affine_a'] * iso + calib['affine_b'], 0.0, 99.0))


def _liveness_axis(path) -> float:
    x, sr = load_mono(path)
    try:
        segs, _ = build_segments(x, sr)
    except Exception:  # noqa: BLE001
        return float('nan')
    ps = []
    for s in segs[:12]:
        try:
            p = p_synth_iq(x[s.onset:], sr, s.f0, getattr(s, 'B', 2e-4))
            if np.isfinite(p):
                ps.append(p)
        except Exception:  # noqa: BLE001
            pass
    return float(100.0 * (1.0 - float(np.median(ps)))) if ps else float('nan')


def _tuning_axis(path) -> float:
    try:
        return float(_tuning_score(path)['score'])
    except Exception:  # noqa: BLE001
        return float('nan')


def full_axes(path, tmodel, calib, trends, p_synth=None) -> dict:
    """The paper's §5 decomposition: one PROVENANCE number (P synth) and four
    orthogonal QUALITY numbers (tuning / timbre / liveness / overall)."""
    prov = float('nan') if p_synth is None else float(p_synth)   # provenance: P(synthetic) of the detector
    _meta, notes, tuning = note_raw_descriptors(path)
    timbre = _timbre_axis(notes, tuning, trends, tmodel, calib)
    tuning_score = _tuning_axis(path)
    liveness = _liveness_axis(path)
    wav, sr = load_mono(path)
    aver = float(aversiveness_features(wav, sr)['aversiveness'])
    ph = float(phaser_index(wav, sr))
    # §5.2 overall: timbre docked by an artefact guardrail (aversiveness/phaser)
    # and by a liveness dock when the modulation is dead.
    guardrail = max(0.0, (aver - 35.0) / 5.0) + (3.0 if ph > 3.5 else 0.0)
    live_dock = max(0.0, (30.0 - liveness) / 30.0 * 10.0) if np.isfinite(liveness) else 0.0
    overall = float(np.clip(timbre - guardrail - live_dock, 0.0, 99.0))
    return dict(file=os.path.basename(path), p_synth=prov, tuning=tuning_score,
                timbre=timbre, liveness=liveness, overall=overall,
                guardrail=guardrail, live_dock=live_dock, aver=aver, phaser=ph,
                notes=len(notes))


def _print_axes_row(a: dict) -> None:
    def s(v):
        return f'{v:5.0f}' if np.isfinite(v) else '  -- '
    print(f"  {a['p_synth']:5.2f}    {s(a['tuning'])}  {s(a['timbre'])}  {s(a['liveness'])}  "
          f"{s(a['overall'])}   {a['file']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep quality report for a recording (full §5)")
    ap.add_argument('wavs', nargs='+', help="WAV file(s), absolute or repo-relative")
    ap.add_argument('--json', default=None, help="also write full metrics to this JSON file")
    ap.add_argument('--cards', action='store_true', help="also print the per-file detail cards")
    ap.add_argument('--no-refs', action='store_true', help="skip the reference anchors")
    args = ap.parse_args()

    trends = json.load(open(TRENDS, encoding='utf-8'))
    critic = _train()                       # analytic critic for the detail cards
    tmodel, M, y = _train_timbre()          # deployed timbre GBM (Group A+B)
    calib = _calibration(tmodel, M, y)

    print("\n=== §5 QUALITY DECOMPOSITION — provenance vs quality (orthogonal) ===")
    print("  P(syn)   TUNING TIMBRE  LIVE  OVERALL  file")
    print("  " + "-" * 60)

    out = []
    for a in args.wavs:
        f = a if os.path.isabs(a) else os.path.join(ROOT, a)
        if not os.path.exists(f):
            print(f"  НЕ ЗНАЙДЕНО: {f}")
            continue
        try:
            ax = full_axes(f, tmodel, calib, trends)
        except Exception as e:  # noqa: BLE001
            print(f"  ПОМИЛКА {os.path.basename(f)}: {e}")
            continue
        out.append(ax)
        _print_axes_row(ax)
        if args.cards:
            try:
                _print_card(score_one(f, critic, trends))
            except Exception as e:  # noqa: BLE001
                print(f"  card err: {e}")

    if not args.no_refs:
        print("\n  --- reference sources (Table 3 order: real/renderer high, PCM low) ---")
        for rel, lab in REFS:
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                continue
            try:
                ax = full_axes(p, tmodel, calib, trends)
                ax['file'] = lab
                _print_axes_row(ax)
            except Exception as e:  # noqa: BLE001
                print(f"  ERR  {lab}: {e}")

    print("\n  provenance = P(synth) (may FLAG a real-sounding sampler); quality axes are\n"
          "  orthogonal — a synthetic-flagged source can still be a fine instrument (§5).")

    if args.json:
        json.dump(out, open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f"\nJSON -> {args.json}")


if __name__ == '__main__':
    main()
