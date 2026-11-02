"""Bulk collector over the full Wikipedia piano-manufacturer list.
Generic per-brand: search -> require brand named in title -> dedup -> is-piano gate
-> reject comparisons/other-brands -> despeech. Rating deferred. Resumable.

    python corpus/collect_brands.py [--target 5 --per-search 16]

Writes <SPD_AUDIO>/audio/brands_extra/ (index.csv, comparisons.csv).  The model-level sets
(yamaha_extra, samick_extra, hoffmann_extra, seiler_extra, perzina_extra) were collected with
corpus/collect_models.py (search query per model, the same gates).
"""
from __future__ import annotations
import sys, os, csv, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('corpus')
ROOT = AUDIO_ROOT
import corpus.collect_models as C
from corpus.pianoness import load_model, _labels, _ensure_assets, tag_file
from corpus.despeech import despeech_file

STAGE = os.path.join(C.AUDIO, 'brands_extra')
RAW = os.path.join(STAGE, '_raw')
INDEX = os.path.join(STAGE, 'index.csv')
COMPARISONS = os.path.join(STAGE, 'comparisons.csv')

BRANDS_RAW = [
 "Atlas","Apollo","August Förster","Baldwin","Bechstein","Becker","Beijing HsingHai",
 "Belarus","Blüthner","Borgato","Bösendorfer","Cavendish","Edelweiss","Estonia","Fazioli",
 "Ferd. Thürmer","Feurich","Fritz Dobbert","Gebr. Schulz","Hailun","Hazelton Brothers",
 "John Broadwood & Sons","Kawai","Maene","Mason & Hamlin","Narvesen","Parsons","Pearl River",
 "Petrof","Pfeiffer","Ravenscroft","Rönisch","Sauter","Schimmel","Steingraeber & Söhne",
 "Steinway & Sons","Stuart & Sons","Wilh. Steinberg","Young Chang","A. B. Chase",
 "Aeolian","McPhail","Adam Schaaf","Albert Fahr","Alfred Knight","Astin Weight",
 "Baus","Jacob Doll","Beale","Becker Brothers","Behr Brothers","Bell","Bogs & Voigt",
 "Boisselot","Brinkerhoff","Brinsmead","Brødrene Hals","Bush & Gerts","Cable",
 "Cable-Nelson","Carl Ecke","Challen","Chappell","Charles Stieff","Chickering",
 "Chute & Butler","Clementi","Collard & Collard","Cunningham","Currier","Danemann",
 "Decker Brothers","Ehrbar","Emerson","Érard","Ernst Kaps","Estey","Falcone","Gabler",
 "Gaveau","George Steck","Gerhard Heintzman","Grotrian-Steinweg","Gulbransen",
 "Haddorf","Haines Brothers","Hallet Davis","Hardman Peck","Heintzman","Hobart M. Cable",
 "Hornung & Møller","J. & C. Fischer","Cramer","Jesse French","Irmler","J. Strauss & Son",
 "Julius Bauer","Kemble","Kieselhorst","Kohler & Campbell","Krakauer","Kranich & Bach",
 "Kurtzmann","Lauter","Lesage","Lester","Lindeman","Marshall & Wendell","Mason & Risch",
 "Mathushek","Mehlin","Melville Clark","Monington & Weston","Nordheimer","P. A. Starck",
 "Pleyel","R. Görs & Kallmann","Rud. Ibach Sohn","Schiedmayer","Schiller","Sohmer",
 "Starr","Steger","Sterling","Story & Clark","Straube","Strich & Zeidler",
 "Vose & Sons","Ritmüller","Kimball","Weber","Welmar","Wendl & Lung","Whaley-Royce",
 "Wilhelm Spaethe","Winter & Co","Wm. Knabe","Wurlitzer","Zimmermann","Wieler",
]
SKIP_BRANDS = {'yamaha', 'samick', 'hoffmann', 'seiler', 'perzina'}

GENERIC = {'and', 'sons', 'son', 'brothers', 'bros', 'company', 'co', 'piano', 'pianos',
           'organ', 'cie', 'fils', 'und', 'the', 'of', 'inc', 'llc', 'gmbh', 'ag',
           'works', 'manufacturing', 'm', 'söhne', 'sohn', 'et'}


def norm_brand(name):
    base = name.lower().replace('&', ' ').replace('.', ' ').replace(',', ' ')
    words = [w for w in base.split() if w]
    sig = [w for w in words if w not in GENERIC and len(w) > 2]
    if not sig:
        sig = [w for w in words if len(w) > 1] or [name.lower()]
    qname = ' '.join(sig[:3])
    own = max(sig, key=len)
    return qname, own, sig


def all_known_video_ids():
    ids = set(C.existing_video_ids())
    for idx in glob.glob(os.path.join(C.AUDIO, '*_extra', 'index.csv')):
        try:
            with open(idx, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r.get('video_id'):
                        ids.add(r['video_id'])
        except OSError:
            pass
    return ids


def read_idx():
    if not os.path.exists(INDEX):
        return []
    with open(INDEX, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def append_idx(row):
    new = not os.path.exists(INDEX)
    with open(INDEX, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['file', 'brand', 'video_id', 'url', 'title', 'dur_s', 'kept_frac', 'speech_max', 'flagged'])
        w.writerow(row)


def log_cmp(brand, vid, title):
    new = not os.path.exists(COMPARISONS)
    with open(COMPARISONS, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['brand', 'video_id', 'url', 'title'])
        w.writerow([brand, vid, f'https://www.youtube.com/watch?v={vid}', title])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=5)
    ap.add_argument('--per-search', type=int, default=16)
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    _ensure_assets(); lab = _labels(); at = load_model()

    skip = all_known_video_ids()
    done = read_idx()
    for r in done:
        skip.add(r['video_id'])
    have_by_brand = {}
    for r in done:
        have_by_brand[r['brand']] = have_by_brand.get(r['brand'], 0) + 1

    for name in BRANDS_RAW:
        qname, own, sig = norm_brand(name)
        slug = own.replace(' ', '')
        if own in SKIP_BRANDS:
            continue
        have = have_by_brand.get(name, 0)
        need = max(0, a.target - have)
        if need == 0:
            continue
        other = [b for b in C.OTHER_BRANDS if b not in sig and b != own] + ['yamaha', 'samick']
        other = [b for b in other if b not in sig]
        print(f"\n=== {name}  (own={own}) need={need} ===", flush=True)
        got = 0
        for suffix in (' upright piano', ' grand piano', ' acoustic piano',
                       ' piano demo', ' piano sound', ' upright', ' grand', ' piano'):
            if got >= need:
                break
            try:
                cands = C.yt_search(qname + suffix, a.per_search)
            except Exception as e:
                print(f"  search ERR {e}", flush=True); continue
            for vid, title, dur in cands:
                if got >= need:
                    break
                if vid in skip:
                    continue
                skip.add(vid)
                tl = title.lower()
                if not any(s in tl for s in sig):
                    continue                                  # brand not named -> skip junk
                if any(k in ' ' + tl + ' ' for k in C.CMP_KW) or any(b in tl for b in other):
                    log_cmp(name, vid, title); continue
                try:
                    d = float(dur) if dur and dur != 'NA' else 0
                except ValueError:
                    d = 0
                if d and (d < 8 or d > 1200):
                    continue
                C.RAW = RAW
                raw = C.download(vid)
                if not raw:
                    continue
                try:
                    g = tag_file(raw, at, lab)
                except Exception as e:
                    print(f"  {vid} tag ERR {e}", flush=True); continue
                if not g['is_piano']:
                    try: os.remove(raw)
                    except OSError: pass
                    continue
                out = os.path.join(STAGE, f"{slug}_{vid}.wav")
                try:
                    rr = despeech_file(raw, out, at, lab)
                except Exception as e:
                    print(f"  {vid} despeech ERR {e}", flush=True); continue
                append_idx([os.path.basename(out), name, vid,
                            f"https://www.youtube.com/watch?v={vid}", title,
                            f"{rr['dur_out']:.1f}", f"{rr['kept_frac']:.2f}",
                            f"{rr['speech_max']:.2f}", '1' if rr['flagged'] else '0'])
                got += 1
                print(f"  + {vid} kept={rr['kept_frac']*100:.0f}% {title[:48]}", flush=True)
        print(f"  == {name}: +{got}", flush=True)
    print("\nQUEUE DONE", flush=True)


if __name__ == '__main__':
    main()
