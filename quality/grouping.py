"""Anti-leakage CV group key = INSTRUMENT identity (not just brand).

Principle: recordings of the SAME physical instrument must never split across CV
folds, but distinct instruments (even of the same brand) may sit in different folds.
Brand-only grouping is too coarse (it lumps a Yamaha U1 upright with a CFX concert
grand); file-only grouping leaks (the same piano playing two pieces, e.g. 55a/55i, is
one instrument).  Finest honest unit per data source (see audio/RATINGS_README.md):

  * youtube  (brands_extra/…) — only brand + video id, NO model metadata → group by
    FILE (each video = a distinct recording/instrument session).
  * manifest (numbered flat names) — model token is in the filename → group by
    brand+model, so the same model across pieces (steinway_d274 liszt + chopin) merges
    while different models (ascot_112 vs ascot_118) stay separate.
  * ccmusic  (ccmusic_pianos/…) — the brand IS one specific piano (100+ notes) → group
    by brand.

On the 487-file proxy this lifts the honest Spearman from 0.49 (brand, over-strict) to
0.58 (model), below the leaky file-grouping 0.62.
"""

from __future__ import annotations
import re


def group_key(file: str, brand: str = '') -> str:
    b = (brand or '').lower().strip()
    if file.startswith('ccmusic_pianos/'):
        return 'cc:' + b
    if '/' in file:                                  # youtube / brands_extra
        return 'yt:' + file
    toks = file.rsplit('.', 1)[0].split('_')         # manifest: parse brand+model
    if toks and re.match(r'^\d+[a-z]?$', toks[0]):
        toks = toks[1:]                              # drop leading index (42, 55a)
    while toks and (toks[-1].isdigit() or toks[-1].lower() == 'old'):
        toks.pop()                                   # drop trailing rating / _OLD
    model = next((t for t in toks if any(c.isdigit() for c in t)),
                 toks[0] if toks else '')
    return f'mf:{b}:{model.lower()}'
