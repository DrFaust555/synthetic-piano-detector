"""Print description snippets around capture/effects keywords for accepted videos, for the
hand review recorded in data/performances/collection/manual_review.csv.

    python corpus/digital_description_hits.py
"""
from __future__ import annotations
import os, sys, csv, json, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
from corpus.digital_acquire import LINE_HINT, MIC_HINT, FX_HINT, DIRECT_FROM_DP

META_DIR = os.path.join(common.SCRIPTS, 'dpv2_meta')
KW = re.compile('|'.join(p.pattern for p in (LINE_HINT, MIC_HINT, FX_HINT, DIRECT_FROM_DP)) +
                r'|record', re.I)


def main():
    rows = list(csv.DictReader(open(os.path.join(common.SCRIPTS, 'dpv2_attempts.csv'), encoding='utf-8')))
    for r in rows:
        if r['status'] != 'accepted':
            continue
        mp = os.path.join(META_DIR, f"{r['video_id']}.json")
        if not os.path.exists(mp):
            continue
        d = json.load(open(mp, encoding='utf-8')).get('description') or ''
        hits = []
        for m in KW.finditer(d):
            a, b = max(0, m.start() - 70), min(len(d), m.end() + 70)
            hits.append(d[a:b].replace('\n', ' '))
        if hits:
            print(f"== {r['model_key']} {r['video_id']} | {r['title'][:60]}")
            for h in hits[:6]:
                print('   ...', h)


if __name__ == '__main__':
    main()
