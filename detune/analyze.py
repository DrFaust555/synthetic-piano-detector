"""CLI: estimate the integral detuning of a piano in a recording.

    python -m detune.analyze --wav <file>

Input may be a full performance, a single note, or a triad/chord.  Output is a
single 0..100 detuning score plus two sub-components (unison + global tuning).
"""

from __future__ import annotations
import argparse
import numpy as np

from detune.segment import build_segments, freq_to_midi
from detune.unison import estimate_unison
from detune.tuning import global_consistency


# --- perceptual mapping: physical units → 0..100 index ---------------------
# The spectral estimator returns the resolved string offset in cents (register-
# independent).  A tight unison (intentional <~3-5 cent warmth) isn't resolved →
# 0; the index rises steeply once strings split measurably apart.
def _unison_index(detune_cents: float, baseline: float = 3.0, scale: float = 12.0) -> float:
    return 100.0 * (1.0 - np.exp(-max(0.0, detune_cents - baseline) / scale))


# Note residual f0 in polyphonic recordings carries an irreducible measurement
# floor (~10-12 cents even on a well-tuned piano), so the baseline sits there and
# only EXCESS scatter (genuine note-to-note tuning inconsistency) drives the index.
def _global_index(scatter_cents: float, baseline: float = 12.0, scale: float = 13.0) -> float:
    return 100.0 * (1.0 - np.exp(-max(0.0, scatter_cents - baseline) / scale))


# --- aggregation -----------------------------------------------------------
W_UNISON = 0.65
W_GLOBAL = 0.35
MIN_CONF = 0.25      # below this a per-note unison estimate is too unreliable to use


def analyze(path: str, n_partials: int = 10, max_notes: int = 400,
            debug: bool = False) -> dict:
    from detune.segment import load_mono
    x, sr = load_mono(path)
    segments, info = build_segments(x, sr, max_notes=max_notes)
    allow_u1 = info.get('mode') == 'single/chord'

    # Per-note unison spreads (only multi-string notes can have a unison beat).
    per_note = []          # (spread_cents, beat_hz, index, weight)
    global_notes = []      # (f0, weight) for ALL notes incl. monochords
    for seg in segments:
        w = seg.energy
        global_notes.append((seg.f0, w, seg.B))   # ALL notes (incl. monochords) feed global
        if seg.n_strings < 2:
            continue                       # monochord: no unison to measure
        ur = estimate_unison(seg.x, sr, seg.f0, seg.B, n_partials=n_partials,
                             allow_u1=allow_u1)
        if ur.method == 'none' or ur.confidence < MIN_CONF:
            continue
        wn = w * ur.confidence
        idx = _unison_index(ur.spread_cents)
        per_note.append((ur.spread_cents, ur.beat_hz, idx, wn))
        if debug:
            print(f"  note midi={seg.midi:3d} f0={seg.f0:7.2f}Hz B={seg.B:.2e} "
                  f"{ur.method} spread={ur.spread_cents:5.2f}c beat={ur.beat_hz:4.2f}Hz "
                  f"conf={ur.confidence:.2f} {ur.config}")

    # --- unison aggregate ---
    has_unison = len(per_note) > 0
    if has_unison:
        spreads = np.array([p[0] for p in per_note])
        beats = np.array([p[1] for p in per_note])
        idxs = np.array([p[2] for p in per_note])
        wns = np.array([p[3] for p in per_note])
        wns = wns / (wns.sum() + 1e-12)
        # energy*confidence-weighted MEAN of per-note indices — the integral
        # detuning across the performance.  (Mean, not RMS: RMS over-amplifies the
        # few residual outliers that survive the k-scaling filter.)
        unison_index = float(np.sum(wns * idxs))
        split = spreads > 0.5                         # notes with a resolved split
        frac_split = float(np.mean(split))
        if split.any():
            med_spread = float(np.median(spreads[split]))
            med_beat = float(np.median(beats[split]))
        else:
            med_spread = med_beat = 0.0
    else:
        unison_index = None
        med_spread = med_beat = frac_split = None

    # --- global aggregate ---
    gr = global_consistency(global_notes)
    global_index = _global_index(gr.scatter_cents) if gr.available else None

    # --- combine ---
    wu = W_UNISON if has_unison else 0.0
    wg = (W_GLOBAL * gr.diversity) if (gr.available and global_index is not None) else 0.0
    if wu + wg <= 0:
        final = None
    else:
        contrib = wu * (unison_index or 0.0) + wg * (global_index or 0.0)
        final = float(contrib / (wu + wg))

    return {
        'final': final,
        'unison_index': unison_index,
        'global_index': global_index,
        'med_spread_cents': med_spread,
        'med_beat_hz': med_beat,
        'frac_split': frac_split,
        'global': gr,
        'n_notes_unison': len(per_note),
        'n_notes_total': len(segments),
        'info': info,
    }


def _confidence_label(r: dict) -> str:
    """Reliability of the absolute score.  Short clips with few distinct pitches
    give noisy estimates (especially the global component), so they are flagged as
    lower confidence — absolute cross-recording comparison is only meaningful at
    high confidence."""
    if r['final'] is None:
        return '—'
    dur = r['info'].get('duration_s', 0.0)
    n_total = r['n_notes_total']
    distinct = r['global'].n_distinct
    if dur >= 40 and n_total >= 60 and distinct >= 24:
        return 'висока'
    if dur >= 10 and n_total >= 15:
        return 'середня'
    return 'низька'


def format_report(r: dict) -> str:
    lines = []
    info = r['info']
    if r['final'] is None:
        lines.append("Розстроєність: не визначається")
        lines.append("  унісон: — (немає багатострунних нот / биття)   "
                     "загальний стрій: — (недостатньо різних висот)")
        return "\n".join(lines)

    lines.append(f"Розстроєність: {r['final']:.1f} / 100   "
                 f"(0 = ідеальний стрій, 100 = сильно розстроєне)")

    if r['unison_index'] is not None:
        if r['frac_split'] and r['frac_split'] > 0:
            detail = (f"[{r['frac_split']*100:.0f}% нот розщеплено, типово "
                      f"{r['med_spread_cents']:.0f} цент / ~{r['med_beat_hz']:.1f} Гц биття]")
        else:
            detail = "[розщеплень не виявлено — унісони щільні]"
        lines.append(f"  унісон (струни в нотах):  {r['unison_index']:.1f}   {detail}")
    else:
        lines.append("  унісон (струни в нотах):  —      "
                     "(немає багатострунних нот для аналізу)")

    g = r['global']
    if r['global_index'] is not None:
        lines.append(f"  загальний стрій:          {r['global_index']:.1f}   "
                     f"[розкид залишків {g.scatter_cents:.1f} цента, "
                     f"{g.n_distinct} висот, reference {g.ref_offset_cents:+.0f}c vs 440]")
    else:
        lines.append("  загальний стрій:          —      "
                     "(одна/замало висот — reference не визначити)")

    trunc = "  (проаналізовано найгучніші ноти)" if info.get('truncated') else ""
    lines.append(f"  проаналізовано нот: {r['n_notes_unison']} (унісон) / "
                 f"{r['n_notes_total']} (всього)   "
                 f"впевненість: {_confidence_label(r)}{trunc}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Інтегральна розстроєність фортепіано у записі.")
    ap.add_argument('--wav', required=True, help="вхідний WAV (твір / нота / тризвук)")
    ap.add_argument('--max-partials', type=int, default=10,
                    help="скільки партіалів аналізувати на ноту")
    ap.add_argument('--max-notes', type=int, default=400,
                    help="ліміт нот для аналізу (беруться найгучніші)")
    ap.add_argument('--debug', action='store_true', help="друкувати per-note діагностику")
    args = ap.parse_args()

    r = analyze(args.wav, n_partials=args.max_partials,
                max_notes=args.max_notes, debug=args.debug)
    print(format_report(r))


if __name__ == '__main__':
    main()
