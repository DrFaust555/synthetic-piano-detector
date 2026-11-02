"""Stage 2: download -> locate a solo acoustic-piano excerpt (60-180 s) -> raw excerpt WAV.

    python corpus/digital_acquire.py --plan data/performances/collection/acquisition_plan.csv [--target 6] [--max-per-channel 2]

plan: model_key,video_id in priority order (from digital_v2_candidates.csv; the two plans used are
data/performances/collection/acquisition_plan.csv and acquisition_plan2.csv).

Per video:
 1. yt-dlp, format '251/140/bestaudio', -x --audio-format wav (no resampling), i.e. the
    command of corpus/collect_models.download minus '-ar 44100'.  This reproduces the
    format of the existing 14 demos' raw files (audio/SynthPianoSamples/yt/raw: 48 kHz
    stereo PCM-16 from Opus).  Full download goes to scratch and is deleted afterwards.
 2. PANNs CNN14 on non-overlapping 1-s windows (same windowing and the same keep rule as
    corpus/clean_demos.py: piano>0.10, speech<0.10, other<0.35, piano>other, +-2 s
    guard) PLUS an acoustic-patch rule (Piano > Electric piano, organ/synth/strings/... <0.2)
    and, if the video has chapters, a chapter-title rule (acoustic piano / repertoire
    chapters only).
 3. Excerpt [start,end] with end-start in [60,180] s maximising good - 2*bad windows.
 4. The excerpt is cut sample-exactly from the downloaded WAV (no re-encode) to
    audio/digital_pianos_v2/raw/<ModelKey>_<videoid>.wav.
 5. Simple acoustic checks on the raw excerpt (noise floor, digital silence, stereo
    correlation, fastest energy decay) for the capture/effects columns.

Every attempt (accepted or not) is appended to results/corpus_digital/dpv2_attempts.csv.
"""
from __future__ import annotations
import os, sys, re, csv, json, glob, subprocess, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
common.setup()
import numpy as np
import soundfile as sf

RAW_OUT = os.path.join(common.AUDIO_OUT, 'raw')
DL = os.path.join(common.SCRATCH, 'dl')
ATTEMPTS = os.path.join(common.SCRIPTS, 'dpv2_attempts.csv')
META_DIR = os.path.join(common.SCRIPTS, 'dpv2_meta')
PROB_DIR = os.path.join(common.SCRIPTS, 'dpv2_probs')

# --- constants of corpus/clean_demos.py ---
import corpus.clean_demos as CS   # noqa: E402
WIN = CS.WIN
SR_PANNS = CS.SR_PANNS
MARGIN = int(round(CS.MARGIN_S / CS.WIN))

NONAC = ['Organ', 'Electronic organ', 'Hammond organ', 'Synthesizer', 'Harpsichord',
         'String section', 'Bowed string instrument', 'Orchestra', 'Violin, fiddle', 'Cello',
         'Vibraphone', 'Marimba, xylophone', 'Glockenspiel', 'Choir', 'Bass guitar',
         'Brass instrument', 'Wind instrument, woodwind instrument', 'Harp', 'Accordion',
         'Synthetic singing', 'Steelpan', 'Electronic music', 'Techno', 'Ambient music',
         'Theremin', 'Sampler']

CH_GOOD = re.compile(r'piano|grand|cfx|b[öo]sendorfer|imperial|sk-?ex|shigeru|concert|upright|'
                     r'steinway|bechstein|classic|ballad|mellow|bright|studio|chopin|debussy|'
                     r'bach|beethoven|mozart|liszt|satie|schubert|rachmanin|schumann|brahms|'
                     r'ravel|grieg|nocturne|sonata|prelude|etude|waltz|ballade|impromptu|'
                     r'clair de lune|gymnop|moonlight|elise|einaudi|improv|jazz|ragtime|'
                     r'joplin|gershwin|performance|demo|piece|song|playing', re.I)
CH_BAD = re.compile(r'electric|e\.?\s?-?piano|\bep\b|rhodes|wurl|organ|string|\bpad|synth|'
                    r'harpsi|\bclav|vibra|\bbass\b|choir|drum|rhythm|style|layer|split|dual|'
                    r'\bduo\b|intro|outro|feature|speaker|bluetooth|\bapp\b|recorder|'
                    r'metronome|touch|action|lesson|tutorial|summary|price|conclusion|talk|'
                    r'overview|design|cabinet|headphone|unbox|setup|connect|spec|'
                    r'final thoughts|verdict|review|comparison|\bvs\b|accompan|registration|'
                    r'voices|sounds|tones|organ|guitar|vocal|sing', re.I)

LINE_HINT = re.compile(r'line[\s\-]?out|direct(ly)?[\s\-]?(recorded|out|recording|from)|'
                       r'recorded directly|audio interface|usb audio|internal recorder|'
                       r'usb recording|no mic|without (a )?mic|headphone out|phones out|'
                       r'di box|aux out|recorded (straight|direct)|line[\s\-]?level|'
                       r'line input|line recording', re.I)
MIC_HINT = re.compile(r'microphone|\bmics?\b|zoom h\d|room sound|recorded (with|on) (my )?'
                      r'(phone|iphone)|iphone|smartphone|built[\s\-]?in speakers|'
                      r'through the speakers|from the speakers|binaural|\brode\b|shure|'
                      r'neumann|\bakg\b|tascam dr|recorded acoustically|camera mic', re.I)
FX_HINT = re.compile(r'reverb|\bhall\b|chorus|delay|ambien|effect|eq\b|equali[sz]|'
                     r'compress|post[\s\-]?process|mastered|mixing|mixed', re.I)
DIRECT_FROM_DP = re.compile(r'recorded (via|with|using) (the )?(built[\s\-]?in|internal|'
                            r'onboard) (audio )?recorder|usb (audio )?recorder|'
                            r'audio recorder (function|feature)', re.I)


def _labels():
    from panns_inference import labels
    return list(labels)


_AT = None


def panns():
    global _AT
    if _AT is None:
        from panns_inference import AudioTagging
        _AT = AudioTagging(checkpoint_path=None, device='cpu')
    return _AT


class RateLimited(Exception):
    pass


def download(vid):
    os.makedirs(DL, exist_ok=True)
    out = os.path.join(DL, f'{vid}.wav')
    if os.path.exists(out):
        return out
    for fmt in ('251/140/bestaudio', '140/bestaudio'):
        cmd = common.ytdlp_cmd() + ['--no-playlist', '--no-warnings', '--retries', '5',
                                    '--sleep-requests', '1.5', '--sleep-interval', '4',
                                    '--max-sleep-interval', '10',
                                    '-f', fmt, '-x', '--audio-format', 'wav',
                                    '--audio-quality', '0',
                                    '-o', os.path.join(DL, f'{vid}.%(ext)s'),
                                    f'https://www.youtube.com/watch?v={vid}']
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', env=common.ytdlp_env())
        if os.path.exists(out):
            return out
        last = (r.stderr or '')[-300:]
        if 'confirm you' in (r.stderr or '') or 'HTTP Error 429' in (r.stderr or ''):
            raise RateLimited(last)
    print('   download failed:', last.strip().splitlines()[-1:] if last else '')
    return None


def window_probs(path, vid):
    """PANNs clip-probabilities on non-overlapping 1-s windows of the whole file."""
    os.makedirs(PROB_DIR, exist_ok=True)
    cache = os.path.join(PROB_DIR, f'{vid}.npz')
    lab = _labels()
    if os.path.exists(cache):
        z = np.load(cache)
        return z['probs'], lab
    import librosa
    y, _ = librosa.load(path, sr=SR_PANNS, mono=True)
    w = int(WIN * SR_PANNS)
    n = len(y) // w
    wins = y[:n * w].reshape(n, w).astype(np.float32)
    out = []
    for i in range(0, n, 64):
        clip, _ = panns().inference(wins[i:i + 64])
        out.append(clip)
    probs = np.concatenate(out, 0) if out else np.zeros((0, len(lab)), np.float32)
    np.savez_compressed(cache, probs=probs.astype(np.float16))
    return probs, lab


def idx(lab, keys):
    return [i for i, l in enumerate(lab) if any(k.lower() in l.lower() for k in keys)]


def exact_idx(lab, keys):
    return [lab.index(k) for k in keys if k in lab]


def classify_windows(probs, lab, chapters, n):
    pi = idx(lab, CS.PIANO)
    spi = idx(lab, CS.SPEECH)
    oi = idx(lab, CS.OTHER)
    pn = probs[:, pi].max(1)
    speech = probs[:, spi].max(1)
    other = probs[:, oi].max(1)
    raw_keep = (pn > 0.10) & (speech < CS.SPEECH_THRESH) & (other < 0.35) & (pn > other)
    reject = ~raw_keep
    rej = reject.copy()
    for i in np.where(reject)[0]:
        rej[max(0, i - MARGIN):i + MARGIN + 1] = True
    keep = ~rej
    ac = probs[:, lab.index('Piano')]
    ep = probs[:, lab.index('Electric piano')]
    nonac = probs[:, exact_idx(lab, NONAC)].max(1)
    acoustic = (ac > ep) & (nonac < 0.20)
    ch_ok = np.ones(n, bool)
    ch_note = 'no_chapters'
    if chapters:
        mask = np.zeros(n, bool)
        good_titles = []
        for c in chapters:
            t = c.get('title') or ''
            if CH_GOOD.search(t) and not CH_BAD.search(t):
                a = int(np.ceil(c['start_time'])); b = int(np.floor(c['end_time']))
                mask[max(0, a):min(n, b)] = True
                good_titles.append(t)
        if mask.any():
            ch_ok = mask
            ch_note = 'chapters_used:' + ' | '.join(good_titles)[:200]
        else:
            ch_note = 'chapters_none_matched(PANNs only)'
    good = keep & acoustic & ch_ok
    stats = {'ch_ok': ch_ok, 'speech_frac': float(np.mean(speech >= CS.SPEECH_THRESH)),
             'ep_frac': float(np.mean(ep >= ac)), 'nonac_frac': float(np.mean(nonac >= 0.2))}
    return good, keep, acoustic, ch_note, stats


def best_excerpt(good, lmin=60, lmax=180):
    n = len(good)
    if n < lmin:
        return None
    g = np.concatenate([[0], np.cumsum(good.astype(int))])
    best = None
    for s in range(0, n - lmin + 1):
        if not good[s]:
            continue
        for L in range(lmin, min(lmax, n - s) + 1):
            e = s + L
            if not good[e - 1]:
                continue
            ng = g[e] - g[s]
            sc = ng - 2 * (L - ng)
            if best is None or sc > best[0]:
                best = (sc, s, e, ng)
    return best


def chapter_segments(good, ch_ok, lmax=180):
    """Acoustic-piano chapters are often short (30-60 s) and separated by talk or other
    patches.  Concatenate the chapter runs (each trimmed to its first..last good window)
    in time order up to lmax seconds.  Returns ([(s, e), ...], n_good)."""
    segs, tot, ng = [], 0, 0
    n = len(good)
    i = 0
    while i < n and tot < lmax:
        if not ch_ok[i]:
            i += 1; continue
        j = i
        while j < n and ch_ok[j]:
            j += 1
        gi = np.where(good[i:j])[0]
        if len(gi):
            a, b = i + gi[0], i + gi[-1] + 1
            b = min(b, a + (lmax - tot))
            segs.append((int(a), int(b)))
            tot += b - a
            ng += int(good[a:b].sum())
        i = j
    return segs, ng


def acoustic_checks(y, sr):
    """y: (n, ch) float.  Returns dict of simple capture/effects indicators."""
    mono = y.mean(1) if y.ndim > 1 else y
    hop = int(0.010 * sr); fr = int(0.050 * sr)
    nfr = max(1, (len(mono) - fr) // hop)
    idxs = np.arange(nfr) * hop
    rms = np.array([np.sqrt(np.mean(mono[i:i + fr] ** 2) + 1e-20) for i in idxs])
    db = 20 * np.log10(rms + 1e-12)
    p95 = float(np.percentile(db, 95)); p5 = float(np.percentile(db, 5)); p1 = float(np.percentile(db, 1))
    dsil = float(np.mean(db < -90))
    corr = ''
    side_db = ''
    if y.ndim > 1 and y.shape[1] == 2:
        L, R = y[:, 0], y[:, 1]
        if np.std(L) > 0 and np.std(R) > 0:
            corr = float(np.corrcoef(L, R)[0, 1])
        mid = 0.5 * (L + R); side = 0.5 * (L - R)
        side_db = float(10 * np.log10((np.mean(side ** 2) + 1e-20) / (np.mean(mid ** 2) + 1e-20)))
    # fastest energy decay: slope of 10-ms dB envelope over 150-ms spans (dB/s), 95th pct of
    # the negative slopes, only where the level is > 50 dB below... i.e. audible range
    sm = np.convolve(db, np.ones(5) / 5, mode='same')
    span = 15
    sl = (sm[span:] - sm[:-span]) / (span * 0.010)
    valid = (sm[:-span] > p95 - 45)
    dec = -sl[valid & (sl < 0)]
    fastest = float(np.percentile(dec, 95)) if len(dec) > 20 else float('nan')
    return {'peak_dbfs': float(20 * np.log10(np.max(np.abs(mono)) + 1e-12)),
            'p95_db': p95, 'floor_p1_rel_db': p1 - p95, 'floor_p5_rel_db': p5 - p95,
            'digital_silence_frac': dsil, 'lr_corr': corr, 'side_mid_db': side_db,
            'fastest_decay_dbps': fastest}


def log_attempt(row):
    cols = ['time', 'model_key', 'video_id', 'window', 'patch_evidence', 'status', 'reason',
            'title', 'channel',
            'duration', 'start_s', 'end_s', 'segments', 'excerpt_s', 'good_s', 'ch_note', 'speech_frac', 'ep_frac',
            'nonac_frac', 'raw_file', 'sr', 'channels', 'peak_dbfs', 'p95_db',
            'floor_p1_rel_db', 'floor_p5_rel_db', 'digital_silence_frac', 'lr_corr',
            'side_mid_db', 'fastest_decay_dbps', 'text_line', 'text_mic', 'text_fx',
            'text_onboard_recorder']
    new = not os.path.exists(ATTEMPTS)
    with open(ATTEMPTS, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def done_attempts():
    if not os.path.exists(ATTEMPTS):
        return []
    with open(ATTEMPTS, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def process(key, vid, meta, window='all', evidence=''):
    row = {'time': time.strftime('%Y-%m-%d %H:%M'), 'model_key': key, 'video_id': vid,
           'window': window, 'patch_evidence': evidence,
           'title': (meta or {}).get('title', ''), 'channel': (meta or {}).get('channel', ''),
           'duration': (meta or {}).get('duration', '')}
    text = ((meta or {}).get('title') or '') + '\n' + ((meta or {}).get('description') or '')
    row['text_line'] = ';'.join(sorted({m.group(0).lower() for m in LINE_HINT.finditer(text)}))
    row['text_mic'] = ';'.join(sorted({m.group(0).lower() for m in MIC_HINT.finditer(text)}))
    row['text_fx'] = ';'.join(sorted({m.group(0).lower() for m in FX_HINT.finditer(text)}))
    row['text_onboard_recorder'] = ';'.join(sorted({m.group(0).lower() for m in DIRECT_FROM_DP.finditer(text)}))
    if meta and (meta.get('live_status') not in (None, 'not_live', 'was_live')):
        row.update(status='excluded', reason='live/upcoming'); log_attempt(row); return False
    path = download(vid)
    if not path:
        row.update(status='excluded', reason='download_failed'); log_attempt(row); return False
    info = sf.info(path)
    row['sr'] = info.samplerate; row['channels'] = info.channels
    probs, lab = window_probs(path, vid)
    n = probs.shape[0]
    good, keep, acoustic, ch_note, st = classify_windows(probs, lab, (meta or {}).get('chapters'), n)
    if window == 'first300' and not ch_note.startswith('chapters_used'):
        # multi-sound demo without usable chapters: the main acoustic grand is normally
        # presented first, so the excerpt is restricted to the first 300 s
        good = good.copy(); good[300:] = False
    row.update(ch_note=ch_note, speech_frac=f"{st['speech_frac']:.2f}",
               ep_frac=f"{st['ep_frac']:.2f}", nonac_frac=f"{st['nonac_frac']:.2f}")
    be = best_excerpt(good)
    segs = None
    if be is not None and be[3] >= 45:
        segs, ng = [(be[1], be[2])], be[3]
    elif ch_note.startswith('chapters_used'):
        cs, cng = chapter_segments(good, st['ch_ok'])
        if cng >= 45 and sum(b - a for a, b in cs) >= 60:
            segs, ng = cs, cng
    if segs is None:
        row.update(status='excluded',
                   reason=f'insufficient_clean_acoustic_piano(good={int(good.sum())}s)')
        log_attempt(row)
        try: os.remove(path)
        except OSError: pass
        return False
    s, e = segs[0][0], segs[-1][1]
    parts = []
    for a, b in segs:
        yy, sr = sf.read(path, start=int(a * info.samplerate), stop=int(b * info.samplerate),
                         dtype='int16', always_2d=True)
        parts.append(yy)
    y = np.concatenate(parts, 0)
    row['segments'] = ';'.join(f'{a}-{b}' for a, b in segs)
    row['excerpt_s'] = sum(b - a for a, b in segs)
    os.makedirs(RAW_OUT, exist_ok=True)
    out = os.path.join(RAW_OUT, f'{key}_{vid}.wav')
    sf.write(out, y, sr, subtype='PCM_16')
    ac = acoustic_checks(y.astype(np.float64) / 32768.0, sr)
    row.update(status='accepted', reason='', start_s=s, end_s=e, good_s=ng,
               raw_file=os.path.relpath(out, common.ROOT).replace('\\', '/'))
    row.update({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in ac.items()})
    log_attempt(row)
    try: os.remove(path)
    except OSError: pass
    print(f'   + {key} {vid} [{row["segments"]}] good={ng}s {ch_note[:40]}', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--target', type=int, default=6)
    ap.add_argument('--max-per-channel', type=int, default=1)
    a = ap.parse_args()
    plan = list(csv.DictReader(open(a.plan, encoding='utf-8')))
    prev = done_attempts()
    tried = {(r['model_key'], r['video_id']) for r in prev}
    acc = {}
    chan = {}
    for r in prev:
        if r['status'] == 'accepted':
            acc[r['model_key']] = acc.get(r['model_key'], 0) + 1
            chan[(r['model_key'], r['channel'])] = chan.get((r['model_key'], r['channel']), 0) + 1
    tgt = {}
    for p in plan:
        tgt[p['model_key']] = int(p.get('target') or a.target)
    for p in plan:
        key, vid = p['model_key'], p['video_id']
        if (key, vid) in tried:
            continue
        if acc.get(key, 0) >= tgt[key]:
            continue
        mp = os.path.join(META_DIR, f'{vid}.json')
        if not os.path.exists(mp):
            import corpus.digital_search as dpv2_search
            dpv2_search.fetch_meta(vid)
        meta = json.load(open(mp, encoding='utf-8')) if os.path.exists(mp) else None
        ch = (meta or {}).get('channel', '')
        if ch and chan.get((key, ch), 0) >= a.max_per_channel:
            log_attempt({'time': time.strftime('%Y-%m-%d %H:%M'), 'model_key': key,
                         'video_id': vid, 'status': 'excluded',
                         'reason': f'channel_cap({a.max_per_channel} per model)',
                         'title': (meta or {}).get('title', ''), 'channel': ch})
            continue
        print(f'-- {key} {vid} {(meta or {}).get("title", "")[:60]}', flush=True)
        try:
            ok = process(key, vid, meta, p.get('window', 'all'), p.get('evidence', ''))
        except RateLimited as ex:
            print('RATE-LIMITED by YouTube; stopping (attempt not logged):', str(ex)[-160:], flush=True)
            break
        except Exception as ex:
            log_attempt({'time': time.strftime('%Y-%m-%d %H:%M'), 'model_key': key,
                         'video_id': vid, 'status': 'excluded', 'reason': f'error:{ex}'[:200],
                         'title': (meta or {}).get('title', ''), 'channel': ch})
            ok = False
        if ok:
            acc[key] = acc.get(key, 0) + 1
            chan[(key, ch)] = chan.get((key, ch), 0) + 1
    print('accepted per model:', acc)


if __name__ == '__main__':
    main()
