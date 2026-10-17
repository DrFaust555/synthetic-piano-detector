"""Score the v2 control listening test (24 clips, recorded vs algorithmic).

    python listening/pilot/score_v2.py [folder]

Reads every *.csv and *.txt file in responses/ (or in the given folder): downloaded answer
files and/or the CSV text pasted from a messenger (several listeners per file are fine,
repeated header lines are skipped).  Joins the answers with answer_key.csv and reports:
  * per class and subclass: share of answers 3-4 ("judged synthetic") with an exact
    two-sided binomial p against 0.5 and a Clopper-Pearson 95 % CI, plus the 1/2/3/4 counts;
  * per listener: judged-synthetic rate per subclass, accuracy, and the recorded-vs-
    algorithmic AUC of the 1-4 ratings (Mann-Whitney, ties = 0.5);
  * per clip: judged-synthetic count and mean rating;
  * the free-text answers.
The report is printed and also saved as score_v2_report.txt next to the responses folder
(i.e. in listening/pilot/ for the default folder).
"""
from __future__ import annotations

import csv
import glob
import io
import os
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True

import numpy as np                          # noqa: E402
from scipy.stats import binomtest           # noqa: E402

KIT = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(KIT, 'answer_key.csv')
SUBS = [('recorded', 'dataset'), ('recorded', 'library'),
        ('algorithmic', 'sample_bank'), ('algorithmic', 'physical_model')]
SUB_UA = {'dataset': 'записані, датасет CCMusic', 'library': 'записані, семпл-бібліотеки',
          'sample_bank': 'алгоритмічні, семпл-банки SF2', 'physical_model': 'алгоритмічні, фізмодель Otey'}
EXP_UA = {'none': 'без муз. освіти', 'amateur': 'аматор', 'music_school': 'музична школа',
          'professional': 'професіонал'}

OUT = []


def say(s=''):
    OUT.append(s)
    print(s)


def load_key():
    with open(KEY, encoding='utf-8') as f:
        return OrderedDict((r['clip'], r) for r in csv.DictReader(f))


def read_responses(folder, clips):
    files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
    rows = []
    for fp in files:
        with open(fp, encoding='utf-8-sig', errors='replace') as f:
            text = f.read().replace('﻿', '')
        header = None
        for raw in csv.reader(io.StringIO(text)):
            rec = [c.strip() for c in raw]
            if not rec or not any(rec):
                continue
            if rec[0].lower() == 'listener':
                header = rec
                continue
            if header is None:
                say(f'  ! {os.path.basename(fp)}: line without a header skipped: {",".join(rec)[:60]}')
                continue
            if len(rec) > len(header) and header[-2:] == ['free_text', 'timestamp']:
                i = len(header) - 2                     # quotes lost in a messenger: commas
                rec = rec[:i] + [','.join(raw[i:-1]).strip()] + [rec[-1]]   # of the free text
            if len(rec) != len(header):
                say(f'  ! {os.path.basename(fp)}: row with {len(rec)} fields (header {len(header)}) skipped')
                continue
            r = dict(zip(header, rec))
            missing_cols = [c for c in clips if c not in r]
            if missing_cols:
                say(f'  ! {os.path.basename(fp)}: not a v2 answer row (no {missing_cols[0]}), skipped')
                continue
            r['_file'] = os.path.basename(fp)
            rows.append(r)
    return files, rows


def dedupe(rows):
    seen, uniq = set(), []
    for r in rows:                                   # same answers saved twice (file + text)
        k = (r['listener'], r.get('timestamp', ''), r.get('order', ''))
        if k not in seen:
            seen.add(k); uniq.append(r)
    by = OrderedDict()
    for r in sorted(uniq, key=lambda r: r.get('timestamp', '')):
        k = r['listener'].strip().lower()
        if k in by:
            say(f"  ! listener '{r['listener']}' answered more than once: keeping {r.get('timestamp')} "
                f"({r['_file']}), dropping {by[k].get('timestamp')} ({by[k]['_file']})")
        by[k] = r
    return list(by.values())


def binom_line(k, n):
    if n == 0:
        return 'n = 0'
    t = binomtest(k, n, 0.5)
    ci = t.proportion_ci(0.95, method='exact')
    return (f'{k:3d}/{n:<3d} = {100 * k / n:5.1f} %   95 % CI [{100 * ci.low:5.1f}, {100 * ci.high:5.1f}]   '
            f'p = {t.pvalue:.4g}')


def auc(pos, neg):
    """P(rating of an algorithmic clip > rating of a recorded clip), ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KIT, 'responses')
    key = load_key()
    clips = list(key)
    files, rows = read_responses(folder, clips)
    say(f'Folder: {folder}')
    say(f'Files read: {len(files)}   answer rows: {len(rows)}')
    rows = dedupe(rows)
    if not rows:
        say('No answers found. Put the returned CSV files (or pasted text saved as .txt) into responses/.')
        return
    for r in rows:                                    # sanity: order must be a permutation
        o = r.get('order', '').split('-')
        r['_order_ok'] = sorted(o) == sorted(c.replace('clip_', '') for c in clips)
        r['_ans'] = {c: int(r[c]) for c in clips if r.get(c, '').strip() in ('1', '2', '3', '4')}
    say(f'Listeners: {len(rows)}')
    bad = [r['listener'] for r in rows if not r['_order_ok']]
    if bad:
        say(f'  ! order field is not a permutation of the 24 clips for: {", ".join(bad)}')
    inc = [(r['listener'], len(clips) - len(r['_ans'])) for r in rows if len(r['_ans']) < len(clips)]
    for name, m in inc:
        say(f'  ! {name}: {m} clip(s) without a valid 1-4 answer (left out)')

    # ---------------------------------------------------------------- pooled per class / subclass
    say('\n=== Judged synthetic (answer 3 or 4), all listeners pooled ===')
    say('    (exact two-sided binomial test against 50 %; Clopper-Pearson 95 % CI)')
    say('    Pooling treats every listener x clip answer as independent; with few listeners')
    say('    read it together with the per-listener table below.')

    def answers(cls=None, sub=None):
        out = []
        for r in rows:
            for c, v in r['_ans'].items():
                if (cls is None or key[c]['class'] == cls) and (sub is None or key[c]['subclass'] == sub):
                    out.append(v)
        return out

    for cls in ('recorded', 'algorithmic'):
        a = answers(cls)
        say(f'  {cls:13s} {"":32s} {binom_line(sum(v >= 3 for v in a), len(a))}')
        for c2, sub in SUBS:
            if c2 != cls:
                continue
            a = answers(cls, sub)
            say(f'    {sub:15s} {SUB_UA[sub]:30s} {binom_line(sum(v >= 3 for v in a), len(a))}')
    say('\n  Answer counts 1 / 2 / 3 / 4  (1 = точно справжнє ... 4 = точно синтетичне)')
    for _, sub in SUBS:
        a = answers(sub=sub)
        say(f'    {sub:15s} ' + ' / '.join(str(sum(v == k for v in a)) for k in (1, 2, 3, 4)))
    corr = sum((v >= 3) == (key[c]['class'] == 'algorithmic') for r in rows for c, v in r['_ans'].items())
    tot = sum(len(r['_ans']) for r in rows)
    say(f'\n  Overall accuracy (recorded -> 1-2, algorithmic -> 3-4): {binom_line(corr, tot)}')

    # ---------------------------------------------------------------- per listener
    say('\n=== Per listener: % judged synthetic, accuracy, AUC (recorded vs algorithmic) ===')
    hdr = (f'  {"listener":18s} {"experience":15s} {"n":>3s} {"rec":>5s} {"alg":>5s} '
           + ' '.join(f'{s[:6]:>6s}' for _, s in SUBS) + f' {"acc":>5s} {"AUC":>5s}')
    say(hdr)
    aucs = []
    per_sub = {s: [] for _, s in SUBS}
    for r in rows:
        ans = r['_ans']

        def rate(pred):
            v = [x for c, x in ans.items() if pred(key[c])]
            return 100.0 * sum(x >= 3 for x in v) / len(v) if v else float('nan')
        rr = rate(lambda k: k['class'] == 'recorded')
        ra = rate(lambda k: k['class'] == 'algorithmic')
        subs = []
        for _, s in SUBS:
            x = rate(lambda k, s=s: k['subclass'] == s)
            subs.append(x); per_sub[s].append(x)
        acc = 100.0 * sum((x >= 3) == (key[c]['class'] == 'algorithmic') for c, x in ans.items()) / len(ans)
        a = auc([x for c, x in ans.items() if key[c]['class'] == 'algorithmic'],
                [x for c, x in ans.items() if key[c]['class'] == 'recorded'])
        aucs.append(a)
        exp = EXP_UA.get(r.get('experience', ''), r.get('experience', ''))
        say(f'  {r["listener"][:18]:18s} {exp[:15]:15s} {len(ans):3d} {rr:5.0f} {ra:5.0f} '
            + ' '.join(f'{x:6.0f}' for x in subs) + f' {acc:5.0f} {a:5.2f}')
    say('  (columns: rec/alg and the four subclasses = % of that listener\'s answers 3-4)')
    say('  mean of listeners: ' + '  '.join(f'{s} {np.nanmean(v):.0f} %' for s, v in per_sub.items()))
    av = np.array([x for x in aucs if not np.isnan(x)])
    above, below = int((av > 0.5).sum()), int((av < 0.5).sum())
    sign_p = binomtest(above, above + below, 0.5).pvalue if above + below else float('nan')
    say(f'  AUC: mean {av.mean():.3f}, median {np.median(av):.3f}, min {av.min():.3f}, max {av.max():.3f}; '
        f'listeners above 0.5: {above}/{len(av)} (below: {below}); exact sign test p = {sign_p:.3g}')

    # ---------------------------------------------------------------- per clip
    say('\n=== Per clip ===')
    say(f'  {"clip":8s} {"subclass":15s} {"instrument":44s} {"note":4s} {"synth":>7s} {"mean":>5s}')
    for c, k in key.items():
        v = [r['_ans'][c] for r in rows if c in r['_ans']]
        if not v:
            continue
        say(f'  {c:8s} {k["subclass"]:15s} {k["instrument"][:44]:44s} {k["note"]:4s} '
            f'{sum(x >= 3 for x in v):3d}/{len(v):<3d} {np.mean(v):5.2f}')

    # ---------------------------------------------------------------- free text
    say('\n=== «На що ви звертали увагу, коли вирішували?» ===')
    for r in rows:
        exp = EXP_UA.get(r.get('experience', ''), r.get('experience', ''))
        say(f'  - {r["listener"]} ({exp}, {r.get("timestamp", "")}): {r.get("free_text", "").strip() or "(порожньо)"}')

    rep = os.path.join(os.path.dirname(os.path.abspath(folder)), 'score_v2_report.txt')
    with open(rep, 'w', encoding='utf-8') as f:            # next to the responses folder
        f.write('\n'.join(OUT) + '\n')
    print(f'\n[saved {rep}]')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    main()
