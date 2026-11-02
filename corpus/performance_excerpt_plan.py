"""Choose 120 MAESTRO MIDI excerpts (100 SoundFont + 20 Otey) -> perf_excerpt_plan.csv.

    python corpus/performance_excerpt_plan.py   -> results/corpus/perf_excerpt_plan.csv
                                                (shipped as data/performances/perf_excerpt_plan.csv)

Deterministic (see perf_common.py docstring for the rules). Reads only.
"""
from __future__ import annotations
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
from corpus.perf_common import (MAESTRO, MAESTRO_CSV, MAESTRO_CLIPS_MANIFEST, PLAN_CSV, DATA_V2,
                         MASTER_SEED, EXCERPT_S, MIN_PERF_S, MIN_NOTES, MAX_DRAWS,
                         N_SF2_SLOTS, N_OTEY_SLOTS, SF2_FILES, load_midi, window_notes,
                         pedal_down_at, pedal_events_in, frac_pedal_down)


def main():
    rows = list(csv.DictReader(open(MAESTRO_CSV, encoding='utf-8')))
    clip_perfs = set()
    if os.path.exists(MAESTRO_CLIPS_MANIFEST):
        for r in csv.DictReader(open(MAESTRO_CLIPS_MANIFEST, encoding='utf-8')):
            clip_perfs.add(os.path.splitext(r['source'])[0])
    rng = np.random.default_rng(MASTER_SEED)
    order = rng.permutation(len(rows))
    n_total = N_SF2_SLOTS + N_OTEY_SLOTS
    used_titles = set()
    plan = []
    n_skipped = 0
    for ri in order:
        if len(plan) >= n_total:
            break
        r = rows[int(ri)]
        dur = float(r['duration'])
        title_key = r['canonical_title'].strip().lower()
        if dur < MIN_PERF_S or title_key in used_titles:
            continue
        k = len(plan)
        seed = MASTER_SEED + 1000 + k
        prng = np.random.default_rng(seed)
        perf = load_midi(os.path.join(MAESTRO, r['midi_filename'].replace('/', os.sep)))
        ok = None
        for attempt in range(1, MAX_DRAWS + 1):
            s = float(prng.uniform(10.0, dur - 70.0))
            later = [n.onset for n in perf.notes if n.onset >= s]
            if not later:
                continue
            t0 = later[0]
            t1 = t0 + EXCERPT_S
            wn = window_notes(perf, t0, t1)
            if len(wn) >= MIN_NOTES and t1 <= perf.end:
                ok = (attempt, s, t0, t1, wn)
                break
        if ok is None:
            n_skipped += 1
            continue
        attempt, s, t0, t1, wn = ok
        used_titles.add(title_key)
        pitches = [n.pitch for n in wn]
        plan.append({
            'slot': k,
            'seed': seed,
            'attempt': attempt,
            'draw_start_s': round(s, 4),
            'excerpt_start_s': t0,            # full precision (repr): rounding could drop the first note
            'excerpt_end_s': t1,
            'excerpt_dur_s': EXCERPT_S,
            'maestro_midi_filename': r['midi_filename'],
            'canonical_composer': r['canonical_composer'],
            'canonical_title': r['canonical_title'],
            'year': r['year'],
            'maestro_split': r['split'],
            'perf_duration_s': round(dur, 3),
            'n_notes': len(wn),
            'min_pitch': min(pitches),
            'max_pitch': max(pitches),
            'median_velocity': float(np.median([n.vel for n in wn])),
            'pedal_down_at_start': int(pedal_down_at(perf, t0)),
            'n_cc64_events': len(pedal_events_in(perf, t0, t1)),
            'frac_time_pedal_down': round(frac_pedal_down(perf, t0, t1), 4),
            'perf_also_in_maestro_clips': int(os.path.splitext(r['audio_filename'])[0] in clip_perfs),
        })
        print(f"  slot {k:3d}  {r['canonical_composer'][:22]:22s} {r['canonical_title'][:40]:40s} "
              f"t0={t0:7.2f}s notes={len(wn):4d} ped={plan[-1]['frac_time_pedal_down']:.2f}",
              flush=True)
    if len(plan) < n_total:
        raise SystemExit(f"only {len(plan)} excerpts found")
    # assign: slots 0..99 -> SoundFonts (seeded permutation, 10 each); 100..119 -> Otey
    insts = sorted(SF2_FILES)
    arng = np.random.default_rng(MASTER_SEED + 7)
    perm = arng.permutation(N_SF2_SLOTS)
    for j, slot in enumerate(perm):
        plan[int(slot)]['engine'] = 'tinysoundfont'
        plan[int(slot)]['instrument'] = insts[j // 10]
    for k in range(N_SF2_SLOTS, n_total):
        plan[k]['engine'] = 'otey_v0'
        plan[k]['instrument'] = 'Otey'
    # file index within instrument (ordered by slot)
    cnt = {}
    for p in plan:
        cnt[p['instrument']] = cnt.get(p['instrument'], 0) + 1
        p['take'] = cnt[p['instrument']]
        p['file'] = (f"audio/sf2_performances_v2/{p['instrument']}/"
                     f"{p['instrument']}_perf{p['take']:02d}.wav")
    OUT_PLAN = os.path.join(results_dir('corpus'), 'perf_excerpt_plan.csv')
    cols = ['file', 'engine', 'instrument', 'take', 'slot', 'seed', 'attempt', 'draw_start_s',
            'excerpt_start_s', 'excerpt_end_s', 'excerpt_dur_s', 'maestro_midi_filename',
            'canonical_composer', 'canonical_title', 'year', 'maestro_split', 'perf_duration_s',
            'n_notes', 'min_pitch', 'max_pitch', 'median_velocity', 'pedal_down_at_start',
            'n_cc64_events', 'frac_time_pedal_down', 'perf_also_in_maestro_clips']
    with open(OUT_PLAN, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in plan:
            w.writerow({c: p[c] for c in cols})
    print(f"\nplan: {len(plan)} excerpts ({n_skipped} performances skipped: <{MIN_NOTES} notes "
          f"in {MAX_DRAWS} draws) -> {OUT_PLAN}")


if __name__ == '__main__':
    main()
