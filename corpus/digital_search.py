"""Stage 1: YouTube search per digital-piano model -> title-filtered candidate list.

    python corpus/digital_search.py [--per-search 15] [--max-meta 10] [--only KEY,KEY]

Search uses yt-dlp `ytsearchN:` (same mechanism as corpus/collect_models.yt_search, but
with --flat-playlist so channel names come back).  Title filtering follows the logic of
corpus/collect_brands.py / collect_models.py (model named in title, comparisons rejected,
other brands rejected, dedup against every known corpus ID) plus extra rules for this
task (software-instrument / non-acoustic-patch / tutorial titles rejected; "no talking"
and repertoire titles preferred).

Writes results/corpus_digital/digital_v2_candidates.csv (all candidates seen, with reject reason) and
caches full yt-dlp metadata (description, chapters, upload date) of the best candidates in
results/corpus_digital/dpv2_meta/<id>.json.
"""
from __future__ import annotations
import os, sys, re, csv, json, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common

META_DIR = os.path.join(common.SCRIPTS, 'dpv2_meta')

CMP_KW = ['vs ', 'vs.', 'versus', 'v.s', 'compar', 'порівн', 'сравн', 'shootout',
          'battle', 'which is best', 'which is better', 'head to head', 'blind test',
          'face off', 'faceoff', 'or ', 'x ']
BRANDS = ['yamaha', 'kawai', 'roland', 'casio', 'korg', 'nord', 'kurzweil', 'dexibell',
          'steinway', 'bosendorfer', 'bösendorfer', 'fazioli', 'bechstein', 'blüthner',
          'bluthner', 'alesis', 'williams', 'donner', 'artesia', 'medeli', 'thomann',
          'ringway', 'm-audio', 'studiologic', 'numa', 'viscount', 'physis', 'gewa',
          'orla', 'ketron', 'hammond', 'moog', 'arturia', 'novation', 'akai', 'boston',
          'essex', 'schimmel', 'petrof', 'samick', 'young chang', 'pearl river', 'hailun']
SOFTWARE = ['pianoteq', 'keyscape', 'garritan', 'vst', 'kontakt', 'plugin', 'plug-in',
            'ivory', 'synthogy', 'spitfire', 'native instruments', 'logic pro', 'garageband',
            'ableton', 'cubase', 'fl studio', 'daw', 'soundfont', 'virtual piano',
            'midi controller', 'via midi', 'midi keyboard']
TALKY = ['unbox', 'tutorial', 'lesson', 'how to', 'setup', 'set up', 'set-up', 'assembl',
         'repair', 'fix', 'problem', 'issue', 'broken', 'noise', 'click', 'buyer', 'buying guide',
         'should you buy', 'worth it', 'explained', 'walkthrough', 'walk through', 'q&a',
         'firmware', 'update', 'app ', 'bluetooth', 'unpack', 'delivery', 'install',
         'beginner', 'learn', 'teaching', 'exercise', 'scale', 'connect', 'recording tip',
         'how i record', 'mic placement', 'speaker test', 'headphone test', 'key action',
         'action test', 'keyboard action', 'touch test', 'velocity test']
NONPIANO = ['electric piano', 'e-piano', 'epiano', 'e.piano', 'rhodes', 'wurli', 'organ',
            'synth', 'strings', 'pad sound', 'harpsichord', 'vibraphone', 'drum', 'rhythm',
            'accompaniment', 'style', 'arranger', 'backing track', 'karaoke', 'band',
            'sing', 'vocal', 'choir', 'guitar', 'violin', 'cello', 'flute', 'sax', 'duet',
            'four hands', '4 hands', 'with orchestra', 'ensemble', 'layer', 'split',
            'registration', 'voices', 'tones demo', 'all sounds', 'all voices', 'all tones',
            'every sound', 'every voice', 'sounds demo']
GOOD = ['no talking', 'no talk', 'all playing', 'just playing', 'only playing',
        'pure sound', 'without talking', 'no commentary', 'no narration']
DEMO = ['sound demo', 'demo', 'sound test', 'sound sample', 'soundcheck', 'sound check',
        'audio demo', 'playing demo']
REPERTOIRE = ['chopin', 'debussy', 'bach', 'beethoven', 'mozart', 'liszt', 'satie',
              'schubert', 'rachmaninov', 'rachmaninoff', 'schumann', 'brahms', 'ravel',
              'grieg', 'tchaikovsky', 'scarlatti', 'haydn', 'mendelssohn', 'nocturne',
              'sonata', 'prelude', 'etude', 'étude', 'waltz', 'ballade', 'impromptu',
              'clair de lune', 'gymnop', 'moonlight', 'für elise', 'fur elise', 'einaudi',
              'yiruma', 'river flows', 'improvis', 'jazz', 'ragtime', 'joplin', 'gershwin',
              'canon', 'arabesque', 'mazurka', 'polonaise', 'scherzo', 'invention',
              'fugue', 'variations', 'romance', 'interstellar', 'ghibli', 'hisaishi']
PERF = ['performance', 'playing', 'plays', 'played on', 'played by', 'recital']
LINE_HINT = ['line out', 'line-out', 'lineout', 'direct out', 'directly recorded',
             'recorded directly', 'direct recording', 'audio interface', 'usb audio',
             'audio recorder', 'internal recorder', 'usb recording', 'no mic', 'no microphone',
             'without microphone', 'headphone out', 'phones out', 'di box', 'aux out']
MIC_HINT = ['microphone', ' mic ', 'mics', 'zoom h', 'room sound', 'recorded with my phone',
            'iphone', 'smartphone', 'built-in speakers', 'through the speakers',
            'speaker sound', 'binaural', 'rode ', 'shure', 'neumann', 'akg', 'tascam dr']


def has(t, kws):
    """keywords matched at a word START (so 'sing' does not fire inside 'using')."""
    return [k for k in kws
            if re.search(r'(?<![a-z0-9])' + re.escape(k.strip()) + (r'(?![a-z0-9])' if k.endswith(' ') else ''), t)]


def yt_search(query, n):
    cmd = common.ytdlp_cmd() + [f'ytsearch{n}:{query}', '--skip-download', '--no-warnings',
                                '--flat-playlist', '--print',
                                '%(id)s@@@%(title)s@@@%(duration)s@@@%(channel)s@@@%(channel_id)s']
    out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                         errors='replace', env=common.ytdlp_env())
    res = []
    for line in (out.stdout or '').splitlines():
        p = line.split('@@@')
        if len(p) >= 5:
            res.append({'id': p[0], 'title': p[1], 'duration': p[2], 'channel': p[3],
                        'channel_id': p[4]})
    return res


def fetch_meta(vid):
    os.makedirs(META_DIR, exist_ok=True)
    path = os.path.join(META_DIR, f'{vid}.json')
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    cmd = common.ytdlp_cmd() + ['-J', '--skip-download', '--no-warnings', '--no-playlist',
                                f'https://www.youtube.com/watch?v={vid}']
    out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                         errors='replace', env=common.ytdlp_env())
    try:
        j = json.loads(out.stdout)
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    keep = {k: j.get(k) for k in ('id', 'title', 'channel', 'channel_id', 'uploader',
                                  'upload_date', 'duration', 'description', 'chapters',
                                  'tags', 'categories', 'webpage_url', 'license',
                                  'availability', 'live_status', 'view_count')}
    keep['audio_formats'] = [{'format_id': f.get('format_id'), 'acodec': f.get('acodec'),
                              'asr': f.get('asr'), 'abr': f.get('abr'), 'ext': f.get('ext')}
                             for f in j.get('formats', []) if f.get('vcodec') == 'none']
    json.dump(keep, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return keep


def judge(m, c, known):
    """Return (reject_reason or '', score)."""
    t = ' ' + c['title'].lower().replace('_', ' ') + ' '
    if c['id'] in known:
        return 'known_id(existing corpus)', 0
    if not re.search(m['re'], t):
        return 'model_not_in_title', 0
    if m.get('not_re') and re.search(m['not_re'], t):
        return 'other_variant', 0
    if m.get('need') and m['need'] not in t:
        return 'brand_not_in_title', 0
    own = m['brand'].lower()
    if has(t, CMP_KW):
        return 'comparison:' + has(t, CMP_KW)[0].strip(), 0
    ob = [b for b in BRANDS if b in t and b != own and not (own == 'nord' and b == 'nord')]
    if ob:
        return 'other_brand:' + ob[0], 0
    if has(t, SOFTWARE):
        return 'software_instrument:' + has(t, SOFTWARE)[0], 0
    if has(t, TALKY):
        return 'talk/tutorial:' + has(t, TALKY)[0].strip(), 0
    if has(t, NONPIANO):
        return 'non_acoustic_patch_or_ensemble:' + has(t, NONPIANO)[0].strip(), 0
    try:
        d = float(c['duration'])
    except (TypeError, ValueError):
        d = 0
    if d and (d < 60 or d > 1800):
        return f'duration_{int(d)}s', 0
    if 'short' in t and d and d < 61:
        return 'short', 0
    s = 0
    s += 4 * bool(has(t, GOOD))
    s += 2 * bool(has(t, DEMO))
    s += 2 * bool(has(t, REPERTOIRE))
    s += 1 * bool(has(t, PERF))
    s -= 2 * bool(has(t, ['review', 'overview', 'features', 'first impression', 'feature',
                          'introduc', 'presentation', 'presents', 'test drive']))
    if d and 90 <= d <= 900:
        s += 1
    return '', s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-search', type=int, default=15)
    ap.add_argument('--max-meta', type=int, default=10)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    models = json.load(open(common.MODELS_JSON, encoding='utf-8'))
    if a.only:
        keep = set(a.only.split(','))
        models = [m for m in models if m['key'] in keep]
    known = set()
    with open(os.path.join(common.REV, 'known_video_ids.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            known.add(r['video_id'])
            if r['video_id'].startswith('_'):
                known.add(r['video_id'][1:])

    suffixes = [' sound demo no talking', ' all playing no talking', ' piano sound demo',
                ' piano performance', ' classical piano']
    cache_p = os.path.join(common.SCRIPTS, 'dpv2_search_cache.json')
    cache = json.load(open(cache_p, encoding='utf-8')) if os.path.exists(cache_p) else {}
    jobs = [(m, m['q'] + s) for m in models for s in suffixes]
    todo = [j for j in jobs if j[1] not in cache or not cache[j[1]]]
    with ThreadPoolExecutor(6) as ex:
        for q, res in zip([j[1] for j in todo], ex.map(lambda j: yt_search(j[1], a.per_search), todo)):
            cache[q] = res
    json.dump(cache, open(cache_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    results = [(m['key'], q, cache.get(q, [])) for m, q in jobs]

    by_key = {m['key']: m for m in models}
    seen_global = {}
    rows = []
    for key, q, res in results:
        m = by_key[key]
        for c in res:
            if c['id'] in seen_global:
                continue
            reason, score = judge(m, c, known)
            if not reason:
                seen_global[c['id']] = key
            rows.append({'model_key': key, 'video_id': c['id'], 'title': c['title'],
                         'duration': c['duration'], 'channel': c['channel'],
                         'channel_id': c['channel_id'], 'query': q, 'reject': reason,
                         'score': score})
    # dedupe rows by (model_key, video_id) keeping the first
    ded = {}
    for r in rows:
        ded.setdefault((r['model_key'], r['video_id']), r)
    rows = list(ded.values())
    # keep only the model a passing ID was first assigned to
    rows = [r for r in rows if r['reject'] or seen_global.get(r['video_id']) == r['model_key']]

    # metadata for the best candidates per model
    best = []
    for key in by_key:
        ok = sorted([r for r in rows if r['model_key'] == key and not r['reject']],
                    key=lambda r: -r['score'])
        best += [r['video_id'] for r in ok[:a.max_meta]]
    with ThreadPoolExecutor(6) as ex:
        metas = dict(zip(best, ex.map(fetch_meta, best)))
    for r in rows:
        mt = metas.get(r['video_id'])
        r['meta'] = 'yes' if mt else ''
        if mt:
            desc = (mt.get('description') or '').lower()
            r['upload_date'] = mt.get('upload_date') or ''
            r['n_chapters'] = len(mt.get('chapters') or [])
            r['desc_line_hint'] = ';'.join(has(' ' + desc + ' ', LINE_HINT))
            r['desc_mic_hint'] = ';'.join(has(' ' + desc + ' ', MIC_HINT))
            r['desc_software'] = ';'.join(has(desc, SOFTWARE[:9]))
    cols = ['model_key', 'video_id', 'title', 'duration', 'channel', 'channel_id', 'query',
            'reject', 'score', 'meta', 'upload_date', 'n_chapters', 'desc_line_hint',
            'desc_mic_hint', 'desc_software']
    with open(common.CANDIDATES, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    okc = Counter(r['model_key'] for r in rows if not r['reject'])
    for key in by_key:
        print(f'{key:22s} pass={okc.get(key, 0)}')
    print('total rows', len(rows), 'pass', sum(okc.values()), '->', common.CANDIDATES)


if __name__ == '__main__':
    main()
