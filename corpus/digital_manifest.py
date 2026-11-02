"""Build digital_v2_manifest.csv (shipped as data/performances/digital_v2_manifest.csv) from the
acquisition log + YouTube metadata + manufacturer sound-engine table
(data/performances/collection/sound_engines.csv) + feature QC.

    python corpus/digital_manifest.py      -> results/corpus_digital/digital_v2_manifest.csv

One row per video that was downloaded/analysed (accepted AND excluded), so exclusions
are documented.  Title-level rejections (never downloaded) stay in
digital_v2_candidates.csv.

capture (line / mic / unknown), conservative:
  line    title/description explicitly says line-out / direct / onboard USB recorder,
          OR >= 2 % of 10-ms frames in the excerpt are digital silence (< -90 dBFS),
          which a microphone/room chain cannot produce (basis given in capture_basis).
  mic     title/description explicitly mentions microphones / phone / speakers
          (checked by hand, see capture_basis).
  unknown otherwise (the acoustic numbers are still listed).
effects_note: description keywords (reverb/hall/chorus/...) + fastest-decay check: the
  95th percentile of energy-decay slopes (dB/s over 150 ms).  A dry piano release or a
  short room gives fast decays (> ~150 dB/s); a long hall reverb caps the fastest decay.
"""
from __future__ import annotations
import os, sys, csv, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
import soundfile as sf

ATTEMPTS = os.path.join(common.SCRIPTS, 'dpv2_attempts.csv')
META_DIR = os.path.join(common.SCRIPTS, 'dpv2_meta')
ENGINES = os.path.join(common.COLLECTION, 'sound_engines.csv')
QC = os.path.join(common.SCRIPTS, 'dpv2_feature_qc.csv')
OVERRIDES = os.path.join(common.COLLECTION, 'manual_review.csv')   # hand-checked notes


def rows(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fnum(x, default=float('nan')):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    M = {m['key']: m for m in json.load(open(common.MODELS_JSON, encoding='utf-8'))}
    E = {r['model_key']: r for r in rows(ENGINES)}
    Q = {(r['model_key'], r['video_id']): r for r in rows(QC)}
    O = {(r['model_key'], r['video_id']): r for r in rows(OVERRIDES)}
    att = rows(ATTEMPTS)
    last = {}
    for r in att:
        last[(r['model_key'], r['video_id'])] = r      # latest attempt wins
    cols = ['file', 'video_id', 'url', 'title', 'channel', 'brand', 'model', 'model_key',
            'sound_method', 'engine_name', 'engine_source_url', 'model_year', 'model_year_source',
            'upload_date', 'start_s', 'end_s', 'segments_s', 'duration_s', 'clean_s', 'capture',
            'capture_basis', 'effects_note', 'excluded', 'exclusion_reason', 'flags',
            'excerpt_rule', 'patch_evidence', 'raw_file',
            'n_notes', 'note_mode', 'median_midi', 'raw_sr', 'raw_channels',
            'floor_p1_rel_db', 'digital_silence_frac', 'lr_corr', 'side_mid_db',
            'fastest_decay_dbps', 'speech_frac_video', 'chapter_note']
    out = []
    for (key, vid), r in sorted(last.items()):
        m = M.get(key, {})
        e = E.get(key, {})
        mt = {}
        mp = os.path.join(META_DIR, f'{vid}.json')
        if os.path.exists(mp):
            mt = json.load(open(mp, encoding='utf-8'))
        o = O.get((key, vid), {})
        q = Q.get((key, vid), {})
        clean = os.path.join(common.AUDIO_OUT, f'{key}_{vid}.wav')
        clean_s = ''
        if os.path.exists(clean):
            clean_s = round(sf.info(clean).duration, 1)
        excluded = r['status'] != 'accepted'
        reason = r.get('reason', '') if excluded else ''
        if not excluded and not clean_s:
            excluded, reason = True, 'cleaner_kept_<8s'
        if not excluded and q.get('status', 'ok') != 'ok':
            excluded, reason = True, q.get('status')
        if o.get('exclude'):
            excluded, reason = True, o['exclude']
        # capture
        # Calibration (2026-10-01, 30 real mic-recorded raws in audio/_raw + 14 old demo
        # raws): digital-silence frames and the noise floor do NOT separate line from mic
        # (edited/gated mic recordings also contain digital silence), so capture is set
        # ONLY from explicit text, hand-checked in data/performances/collection/manual_review.csv.
        cap, basis = 'unknown', 'no explicit statement in title/description'
        hints = [x for x in (r.get('text_line'), r.get('text_onboard_recorder'), r.get('text_mic')) if x]
        if hints:
            basis = 'unverified keyword hits: ' + ';'.join(hints)
        if o.get('capture'):
            cap, basis = o['capture'], o.get('capture_basis') or 'manual review of description'
        ds = fnum(r.get('digital_silence_frac'))
        if ds == ds:
            basis += f' | acoustic hint: digital-silence frames {100*ds:.1f}% (not decisive)'
        # effects
        fx = []
        if r.get('text_fx'):
            fx.append('description mentions: ' + r['text_fx'])
        fd = fnum(r.get('fastest_decay_dbps'))
        # reference: same check on 30 real mic-recorded raws (audio/_raw): median 45 dB/s,
        # 10-90 % 24-109 dB/s.
        if fd == fd:
            tag = ('long tails: heavy reverb or continuous pedal likely' if fd < 24 else
                   'fast releases: dry/short reverb' if fd > 109 else
                   'within real-recording range')
            fx.append(f'fastest decay {fd:.0f} dB/s ({tag}; real-corpus ref median 45, 10-90% 24-109)')
        if o.get('effects_note'):
            fx.append(o['effects_note'])
        flags = []
        en = (o.get('effects_note') or '').lower()
        if 'external' in en:
            flags.append('external_reverb_added')
        elif 'hall' in en or 'reverb' in en:
            flags.append('onboard_reverb_stated')
        if 'line+mic' in (o.get('capture_basis') or ''):
            flags.append('mixed_line_and_mic')
        if r.get('segments') and ';' in r.get('segments', ''):
            flags.append('concatenated_chapters')
        if fd == fd and fd < 24:
            flags.append('long_tails')
        if r.get('window') == 'first300' and not (r.get('ch_note') or '').startswith('chapters_used'):
            flags.append('patch_inferred_first300s')
        if r.get('window') == 'first300':
            rule = ('acoustic-piano chapters only' if (r.get('ch_note') or '').startswith('chapters_used')
                    else 'multi-sound demo: excerpt restricted to first 300 s (main acoustic patch)')
        else:
            rule = 'whole video searched'
        out.append({
            'file': os.path.relpath(clean, common.ROOT).replace('\\', '/') if clean_s else '',
            'video_id': vid, 'url': f'https://www.youtube.com/watch?v={vid}',
            'title': mt.get('title') or r.get('title', ''),
            'channel': mt.get('channel') or r.get('channel', ''),
            'brand': m.get('brand', ''), 'model': m.get('model', ''), 'model_key': key,
            'sound_method': e.get('sound_method') or 'UNKNOWN',
            'engine_name': e.get('engine_name_as_stated', ''),
            'engine_source_url': e.get('source_url', ''),
            'model_year': e.get('model_year') or 'UNKNOWN',
            'model_year_source': e.get('year_source', ''),
            'upload_date': mt.get('upload_date', ''),
            'start_s': r.get('start_s', ''), 'end_s': r.get('end_s', ''),
            'segments_s': r.get('segments', ''),
            'duration_s': r.get('excerpt_s', '') or ((int(r['end_s']) - int(r['start_s'])) if r.get('start_s') else ''),
            'flags': ';'.join(flags),
            'excerpt_rule': rule,
            'patch_evidence': r.get('patch_evidence', ''),
            'clean_s': clean_s, 'capture': cap, 'capture_basis': basis,
            'effects_note': ' | '.join(fx), 'excluded': int(excluded),
            'exclusion_reason': reason, 'raw_file': r.get('raw_file', ''),
            'n_notes': q.get('n_notes', ''), 'note_mode': q.get('mode', ''),
            'median_midi': q.get('median_midi', ''), 'raw_sr': r.get('sr', ''),
            'raw_channels': r.get('channels', ''),
            'floor_p1_rel_db': r.get('floor_p1_rel_db', ''),
            'digital_silence_frac': r.get('digital_silence_frac', ''),
            'lr_corr': r.get('lr_corr', ''), 'side_mid_db': r.get('side_mid_db', ''),
            'fastest_decay_dbps': r.get('fastest_decay_dbps', ''),
            'speech_frac_video': r.get('speech_frac', ''), 'chapter_note': r.get('ch_note', ''),
        })
    with open(common.MANIFEST, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in out:
            w.writerow(x)
    acc = [x for x in out if not x['excluded']]
    print(f'{len(out)} rows ({len(acc)} accepted) -> {common.MANIFEST}')


if __name__ == '__main__':
    main()
