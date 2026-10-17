"""Score listening test 3 (24 clips of 4.15 s, recorded vs algorithmic) and compare with test 2.

    python listening/test2/score_v3.py [folder]

Reads every *.csv and *.txt file in responses/ (or in the given folder): downloaded answer
files and/or the CSV text pasted from a messenger (several listeners per file are fine,
repeated header lines are skipped; rows of other tests are skipped).  Joins the answers with
answer_key.csv and reports, exactly as score_v2.py:
  * per class and subclass: share of answers 3-4 ("judged synthetic") with an exact two-sided
    binomial p against 0.5 and a Clopper-Pearson 95 % CI, plus the 1/2/3/4 counts;
  * per listener: judged-synthetic rate per subclass, accuracy, and the recorded-vs-algorithmic
    AUC of the 1-4 ratings (Mann-Whitney, ties = 0.5);
  * per clip: judged-synthetic count and mean rating;
  * the free-text answers;
plus the comparison with test 2 (listening/pilot: answer_key.csv + responses/):
  * pooled test-2 vs test-3 judged-synthetic rates per subclass and overall accuracy;
  * per listener who did BOTH tests (same listener name, case/space-insensitive): test-2 vs
    test-3 accuracy, AUC and judged-synthetic rates, with a paired summary (Wilcoxon / sign test);
  * per experience group: mean AUC in test 2 and test 3;
  * per stimulus (listeners of both tests): mean rating in test 2 vs test 3 (the key column
    test2_clip links the stimuli; substituted stimuli are marked).
The report is printed and saved as score_v3_report.txt next to the responses folder.
"""
from __future__ import annotations

import csv
import glob
import io
import os
import re
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True

import numpy as np                          # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

KIT = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(KIT, 'answer_key.csv')
V2 = os.path.join(os.path.dirname(KIT), 'pilot')
V2_KEY = os.path.join(V2, 'answer_key.csv')
V2_RESP = os.path.join(V2, 'responses')
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


def load_key(path):
    with open(path, encoding='utf-8') as f:
        return OrderedDict((r['clip'], r) for r in csv.DictReader(f))


def read_responses(folder, clips, quiet=False):
    files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
    rows = []
    warn = (lambda s: None) if quiet else say
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
                warn(f'  ! {os.path.basename(fp)}: line without a header skipped: {",".join(rec)[:60]}')
                continue
            if len(rec) > len(header) and header[-2:] == ['free_text', 'timestamp']:
                i = len(header) - 2                     # quotes lost in a messenger: commas
                rec = rec[:i] + [','.join(raw[i:-1]).strip()] + [rec[-1]]   # of the free text
            if len(rec) != len(header):
                warn(f'  ! {os.path.basename(fp)}: row with {len(rec)} fields (header {len(header)}) skipped')
                continue
            r = dict(zip(header, rec))
            missing_cols = [c for c in clips if c not in r]
            if missing_cols:
                warn(f'  ! {os.path.basename(fp)}: not an answer row of this test (no {missing_cols[0]}), skipped')
                continue
            r['_file'] = os.path.basename(fp)
            rows.append(r)
    return files, rows


def dedupe(rows, quiet=False):
    seen, uniq = set(), []
    for r in rows:                                   # same answers saved twice (file + text)
        k = (r['listener'], r.get('timestamp', ''), r.get('order', ''))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    by = OrderedDict()
    for r in sorted(uniq, key=lambda r: r.get('timestamp', '')):
        k = norm(r['listener'])
        if k in by and not quiet:
            say(f"  ! listener '{r['listener']}' answered more than once: keeping {r.get('timestamp')} "
                f"({r['_file']}), dropping {by[k].get('timestamp')} ({by[k]['_file']})")
        by[k] = r
    return list(by.values())


def norm(name):
    return re.sub(r'\s+', ' ', (name or '').strip().lower())


# Optional cross-test pairing of listener IDs (test-3 ID -> test-2 ID); empty in the public release.
ALIAS_T2 = {}
NOPAIR_T2 = set()


def t2name(name):
    k = norm(name)
    if k in NOPAIR_T2:
        return '#' + k
    return ALIAS_T2.get(k, k)


def prepare(rows, clips, prefix):
    for r in rows:
        o = r.get('order', '').split('-')
        r['_order_ok'] = sorted(o) == sorted(c.replace(prefix, '') for c in clips)
        r['_ans'] = {c: int(r[c]) for c in clips if r.get(c, '').strip() in ('1', '2', '3', '4')}
    return rows


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


def listener_stats(r, key):
    ans = r['_ans']

    def rate(pred):
        v = [x for c, x in ans.items() if pred(key[c])]
        return 100.0 * sum(x >= 3 for x in v) / len(v) if v else float('nan')
    st = dict(n=len(ans), rec=rate(lambda k: k['class'] == 'recorded'),
              alg=rate(lambda k: k['class'] == 'algorithmic'))
    for _, s in SUBS:
        st[s] = rate(lambda k, s=s: k['subclass'] == s)
    st['acc'] = 100.0 * sum((x >= 3) == (key[c]['class'] == 'algorithmic') for c, x in ans.items()) / max(1, len(ans))
    st['auc'] = auc([x for c, x in ans.items() if key[c]['class'] == 'algorithmic'],
                    [x for c, x in ans.items() if key[c]['class'] == 'recorded'])
    return st


def pooled(rows, key, cls=None, sub=None):
    out = []
    for r in rows:
        for c, v in r['_ans'].items():
            if (cls is None or key[c]['class'] == cls) and (sub is None or key[c]['subclass'] == sub):
                out.append(v)
    return out


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KIT, 'responses')
    key = load_key(KEY)
    clips = list(key)
    files, rows = read_responses(folder, clips)
    say(f'Folder: {folder}')
    say(f'Files read: {len(files)}   answer rows: {len(rows)}')
    rows = dedupe(rows)
    if not rows:
        say('No answers found. Put the returned CSV files (or pasted text saved as .txt) into responses/.')
        return
    prepare(rows, clips, 'c3_')
    say(f'Listeners: {len(rows)}')
    bad = [r['listener'] for r in rows if not r['_order_ok']]
    if bad:
        say(f'  ! order field is not a permutation of the 24 clips for: {", ".join(bad)}')
    for r in rows:
        if len(r['_ans']) < len(clips):
            say(f'  ! {r["listener"]}: {len(clips) - len(r["_ans"])} clip(s) without a valid 1-4 answer (left out)')

    # ---------------------------------------------------------------- pooled per class / subclass
    say('\n=== Judged synthetic (answer 3 or 4), all listeners pooled ===')
    say('    (exact two-sided binomial test against 50 %; Clopper-Pearson 95 % CI)')
    say('    Pooling treats every listener x clip answer as independent; with few listeners')
    say('    read it together with the per-listener table below.')
    for cls in ('recorded', 'algorithmic'):
        a = pooled(rows, key, cls)
        say(f'  {cls:13s} {"":32s} {binom_line(sum(v >= 3 for v in a), len(a))}')
        for c2, sub in SUBS:
            if c2 != cls:
                continue
            a = pooled(rows, key, cls, sub)
            say(f'    {sub:15s} {SUB_UA[sub]:30s} {binom_line(sum(v >= 3 for v in a), len(a))}')
    say('\n  Answer counts 1 / 2 / 3 / 4  (1 = точно справжнє ... 4 = точно синтетичне)')
    for _, sub in SUBS:
        a = pooled(rows, key, sub=sub)
        say(f'    {sub:15s} ' + ' / '.join(str(sum(v == k for v in a)) for k in (1, 2, 3, 4)))
    corr = sum((v >= 3) == (key[c]['class'] == 'algorithmic') for r in rows for c, v in r['_ans'].items())
    tot = sum(len(r['_ans']) for r in rows)
    say(f'\n  Overall accuracy (recorded -> 1-2, algorithmic -> 3-4): {binom_line(corr, tot)}')

    # ---------------------------------------------------------------- per listener
    say('\n=== Per listener: % judged synthetic, accuracy, AUC (recorded vs algorithmic) ===')
    say(f'  {"listener":18s} {"experience":15s} {"n":>3s} {"rec":>5s} {"alg":>5s} '
        + ' '.join(f'{s[:6]:>6s}' for _, s in SUBS) + f' {"acc":>5s} {"AUC":>5s}')
    per_sub = {s: [] for _, s in SUBS}
    aucs = []
    for r in rows:
        st = listener_stats(r, key)
        r['_st'] = st
        aucs.append(st['auc'])
        for _, s in SUBS:
            per_sub[s].append(st[s])
        exp = EXP_UA.get(r.get('experience', ''), r.get('experience', ''))
        say(f'  {r["listener"][:18]:18s} {exp[:15]:15s} {st["n"]:3d} {st["rec"]:5.0f} {st["alg"]:5.0f} '
            + ' '.join(f'{st[s]:6.0f}' for _, s in SUBS) + f' {st["acc"]:5.0f} {st["auc"]:5.2f}')
    say('  (columns: rec/alg and the four subclasses = % of that listener\'s answers 3-4)')
    say('  mean of listeners: ' + '  '.join(f'{s} {np.nanmean(v):.0f} %' for s, v in per_sub.items()))
    av = np.array([x for x in aucs if not np.isnan(x)])
    above, below = int((av > 0.5).sum()), int((av < 0.5).sum())
    sign_p = binomtest(above, above + below, 0.5).pvalue if above + below else float('nan')
    say(f'  AUC: mean {av.mean():.3f}, median {np.median(av):.3f}, min {av.min():.3f}, max {av.max():.3f}; '
        f'listeners above 0.5: {above}/{len(av)} (below: {below}); exact sign test p = {sign_p:.3g}')

    # ---------------------------------------------------------------- per clip
    say('\n=== Per clip ===')
    say(f'  {"clip":6s} {"test-2":7s} {"subclass":15s} {"instrument":40s} {"note":4s} {"synth":>7s} {"mean":>5s}  flags')
    for c, k in key.items():
        v = [r['_ans'][c] for r in rows if c in r['_ans']]
        if not v:
            continue
        say(f'  {c:6s} {k["test2_clip"]:7s} {k["subclass"]:15s} {k["instrument"][:40]:40s} {k["note"]:4s} '
            f'{sum(x >= 3 for x in v):3d}/{len(v):<3d} {np.mean(v):5.2f}  {k["flags"]}')

    # ---------------------------------------------------------------- free text
    say('\n=== «На що ви звертали увагу, коли вирішували?» ===')
    for r in rows:
        exp = EXP_UA.get(r.get('experience', ''), r.get('experience', ''))
        say(f'  - {r["listener"]} ({exp}, {r.get("timestamp", "")}): {r.get("free_text", "").strip() or "(порожньо)"}')

    # ---------------------------------------------------------------- comparison with test 2
    say('\n=== Comparison with test 2 (listening/pilot) ===')
    if not (os.path.exists(V2_KEY) and os.path.isdir(V2_RESP)):
        say('  test-2 key or responses folder not found: skipped')
    else:
        key2 = load_key(V2_KEY)
        clips2 = list(key2)
        _, rows2 = read_responses(V2_RESP, clips2, quiet=True)
        rows2 = prepare(dedupe(rows2, quiet=True), clips2, 'clip_')
        for r in rows2:
            r['_st'] = listener_stats(r, key2)
        say(f'  test-2 listeners: {len(rows2)}   test-3 listeners: {len(rows)}')
        say('\n  Pooled judged-synthetic rate (answers 3-4), all listeners of each test:')
        say(f'    {"":15s} {"test 2":>22s} {"test 3":>22s}')
        for cls, sub in [('recorded', None)] + [(c, s) for c, s in SUBS if c == 'recorded'] + \
                        [('algorithmic', None)] + [(c, s) for c, s in SUBS if c == 'algorithmic']:
            a2, a3 = pooled(rows2, key2, cls, sub), pooled(rows, key, cls, sub)
            f = lambda a: f'{sum(v >= 3 for v in a):3d}/{len(a):<3d} = {100 * sum(v >= 3 for v in a) / max(1, len(a)):5.1f} %'
            say(f'    {(sub or cls):15s} {f(a2):>22s} {f(a3):>22s}')
        acc = lambda rr, kk: (sum((v >= 3) == (kk[c]['class'] == 'algorithmic') for r in rr for c, v in r['_ans'].items()),
                              sum(len(r['_ans']) for r in rr))
        k2, n2 = acc(rows2, key2)
        k3, n3 = acc(rows, key)
        say(f'    {"accuracy":15s} {f"{k2}/{n2} = {100 * k2 / max(1, n2):.1f} %":>22s} {f"{k3}/{n3} = {100 * k3 / max(1, n3):.1f} %":>22s}')
        a2 = np.array([r['_st']['auc'] for r in rows2])
        say(f'    {"listener AUC":15s} {f"mean {np.nanmean(a2):.3f}":>22s} {f"mean {av.mean():.3f}":>22s}')

        say('\n  Mean listener AUC per experience group (test 2 | test 3):')
        for e in ('none', 'amateur', 'music_school', 'professional'):
            x2 = [r['_st']['auc'] for r in rows2 if r.get('experience') == e]
            x3 = [r['_st']['auc'] for r in rows if r.get('experience') == e]
            if x2 or x3:
                say(f'    {EXP_UA[e]:16s} test 2: {np.nanmean(x2) if x2 else float("nan"):.3f} (n={len(x2)})   '
                    f'test 3: {np.nanmean(x3) if x3 else float("nan"):.3f} (n={len(x3)})')

        by2 = {norm(r['listener']): r for r in rows2}
        pairs = [(by2[t2name(r['listener'])], r) for r in rows if t2name(r['listener']) in by2]
        say(f'\n  Listeners who did both tests (same name or confirmed alias): {len(pairs)}'
            + (f'  ({", ".join(r3["listener"] for _, r3 in pairs)})' if pairs else ''))
        nomatch = [r['listener'] for r in rows if t2name(r['listener']) not in by2]
        if nomatch:
            say(f'  test-3 listeners without a test-2 answer of the same name: {", ".join(nomatch)}')
        if pairs:
            say(f'    {"listener":18s} {"experience":15s} | {"acc2":>5s} {"acc3":>5s} {"d":>5s} | {"AUC2":>5s} {"AUC3":>5s} '
                f'{"d":>6s} | {"rec2":>4s} {"rec3":>4s} | {"alg2":>4s} {"alg3":>4s} | '
                + ' '.join(f'{s[:6] + "2":>7s} {s[:6] + "3":>7s}' for _, s in SUBS))
            for r2, r3 in pairs:
                s2, s3 = r2['_st'], r3['_st']
                exp = EXP_UA.get(r3.get('experience', ''), r3.get('experience', ''))
                say(f'    {r3["listener"][:18]:18s} {exp[:15]:15s} | {s2["acc"]:5.0f} {s3["acc"]:5.0f} '
                    f'{s3["acc"] - s2["acc"]:+5.0f} | {s2["auc"]:5.2f} {s3["auc"]:5.2f} {s3["auc"] - s2["auc"]:+6.2f} | '
                    f'{s2["rec"]:4.0f} {s3["rec"]:4.0f} | {s2["alg"]:4.0f} {s3["alg"]:4.0f} | '
                    + ' '.join(f'{s2[s]:7.0f} {s3[s]:7.0f}' for _, s in SUBS))
            d_auc = np.array([r3['_st']['auc'] - r2['_st']['auc'] for r2, r3 in pairs])
            d_acc = np.array([r3['_st']['acc'] - r2['_st']['acc'] for r2, r3 in pairs])
            up, dn = int((d_auc > 0).sum()), int((d_auc < 0).sum())
            sp = binomtest(up, up + dn, 0.5).pvalue if up + dn else float('nan')
            try:
                wp = wilcoxon(d_auc).pvalue if len(d_auc) >= 3 and np.any(d_auc != 0) else float('nan')
            except ValueError:
                wp = float('nan')
            say(f'    paired: mean AUC change {d_auc.mean():+.3f} (median {np.median(d_auc):+.3f}), '
                f'mean accuracy change {d_acc.mean():+.1f} points; AUC up for {up}, down for {dn}; '
                f'sign test p = {sp:.3g}; Wilcoxon p = {wp:.3g}')

            say('\n  Per stimulus, listeners of both tests: mean rating test 2 -> test 3 (1 = real ... 4 = synthetic)')
            say(f'    {"test-3":6s} {"test-2":7s} {"subclass":15s} {"test-3 stimulus":40s} {"note":4s} '
                f'{"mean2":>5s} {"mean3":>5s} {"d":>6s}  substitution')
            for c, k in key.items():
                c2 = k['test2_clip']
                v2 = [r2['_ans'][c2] for r2, r3 in pairs if c2 in r2['_ans'] and c in r3['_ans']]
                v3 = [r3['_ans'][c] for r2, r3 in pairs if c2 in r2['_ans'] and c in r3['_ans']]
                if not v2:
                    continue
                sub = '' if k['substituted'] == '0' else f'changed: {key2[c2]["instrument"][:28]} {key2[c2]["note"]}'
                say(f'    {c:6s} {c2:7s} {k["subclass"]:15s} {k["instrument"][:40]:40s} {k["note"]:4s} '
                    f'{np.mean(v2):5.2f} {np.mean(v3):5.2f} {np.mean(v3) - np.mean(v2):+6.2f}  {sub}')

    rep = os.path.join(os.path.dirname(os.path.abspath(folder)), 'score_v3_report.txt')
    with open(rep, 'w', encoding='utf-8') as f:            # next to the responses folder
        f.write('\n'.join(OUT) + '\n')
    print(f'\n[saved {rep}]')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    main()
