"""Бракуючі групи ознак — швидкий екстрактор (без HPSS/yin).

    python -m quality.psycho2 [--limit N] [підрядок ...]

Додає: Zwicker sharpness, Bark-смуговий баланс, attack hardness (спектральна
твердість удару), mechanical artifacts. Кешує quality/cache/psycho2.csv.
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUDIO = os.path.join(ROOT, 'audio')
MANIFEST = os.path.join(ROOT, 'audio', 'manifest.csv')
OUT = os.path.join(HERE, 'cache', 'psycho2.csv')

SR = 22050
DUR = 35.0

# верхні межі 24 Bark-смуг (Гц)
BARK_HI = np.array([100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,
                    2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500.0])

COLS = ['sharpness', 'bark_tilt', 'bark_even', 'bark_lo', 'bark_mid', 'bark_hi',
        'attack_centroid_hz', 'attack_hf_ratio', 'attack_flux',
        'mech_hf_transient', 'mech_subbass', 'mech_interonset',
        'wiener_entropy', 'harm_entropy', 'temporal_entropy', 'noise_ratio',
        'spec_skew', 'spec_kurtosis', 'env_kurtosis', 'centroid_var',
        'rolloff99_hz', 'cutoff_hz', 'reverb_index']


def _safe(v):
    return float(v) if (v is not None and np.isfinite(v)) else np.nan


def bark_loudness(spec, freqs):
    """Питома гучність по 24 Bark-смугах: E(z)^0.23 (Zwicker)."""
    N = np.zeros(len(BARK_HI))
    lo = 0.0
    for z, hi in enumerate(BARK_HI):
        w = (freqs >= lo) & (freqs < hi)
        E = (spec[w] ** 2).sum()
        N[z] = E ** 0.23 if E > 0 else 0.0
        lo = hi
    return N


def zwicker_sharpness(N):
    z = np.arange(1, len(N) + 1)
    g = np.where(z < 16, 1.0, 0.066 * np.exp(0.171 * z))
    s = N.sum()
    return 0.11 * float((N * g * z).sum() / s) if s > 0 else np.nan


def extract(path):
    y, sr = librosa.load(path, sr=SR, mono=True, duration=DUR)
    if y.size < sr:
        return None
    y = y / (np.max(np.abs(y)) + 1e-9)
    n_fft, hop = 2048, 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spec = S.mean(axis=1)
    d = {}

    # --- Bark-баланс + Zwicker sharpness ---
    N = bark_loudness(spec, freqs)
    d['sharpness'] = zwicker_sharpness(N)
    tot = N.sum() + 1e-12
    Nf = N / tot
    # перцептивний нахил (питома гучність vs індекс Bark, dB/Bark)
    zz = np.arange(1, len(N) + 1)
    pos = N > 0
    if pos.sum() >= 4:
        d['bark_tilt'] = _safe(np.polyfit(zz[pos], 10 * np.log10(N[pos] + 1e-12), 1)[0])
    else:
        d['bark_tilt'] = np.nan
    # рівність «тіла» (розкид часток у смугах 4..18) — менше = рівніше
    d['bark_even'] = _safe(np.std(Nf[3:18]))
    d['bark_lo'] = _safe(Nf[:8].sum())     # ~до 1080 Гц
    d['bark_mid'] = _safe(Nf[8:15].sum())  # ~1080–2700 Гц
    d['bark_hi'] = _safe(Nf[15:].sum())    # >3150 Гц

    # --- ентропія / рівень шуму ---
    P = (S ** 2).mean(axis=1)                       # довготривалий power-спектр
    # ентропія Вінера (спектр. пласкість): 0=чистий тон, →1=шум  (= РІВЕНЬ ШУМУ)
    d['wiener_entropy'] = _safe(np.exp(np.mean(np.log(P + 1e-12))) / (P.mean() + 1e-12))
    # гармонічна ентропія (tonalness): ентропія верхніх спектр. піків
    loc = (P[1:-1] > P[:-2]) & (P[1:-1] > P[2:])
    pk = np.where(loc)[0] + 1
    if pk.size >= 4:
        mags = np.sort(P[pk])[-30:]; pm = mags / (mags.sum() + 1e-12)
        d['harm_entropy'] = _safe(-(pm * np.log(pm + 1e-12)).sum() / np.log(len(pm)))
        mask = np.zeros(len(P), bool)
        for o in (-1, 0, 1):
            mask[np.clip(pk + o, 0, len(P) - 1)] = True
        d['noise_ratio'] = _safe(P[~mask].sum() / (P.sum() + 1e-12))  # енергія МІЖ піками
    else:
        d['harm_entropy'] = d['noise_ratio'] = np.nan
    # часова ентропія обвідної (рівномірність енергії в часі)
    rmsv = np.sqrt((S ** 2).mean(axis=0)); rp = rmsv / (rmsv.sum() + 1e-12)
    d['temporal_entropy'] = _safe(-(rp * np.log(rp + 1e-12)).sum() / np.log(len(rp)))

    # --- статистичні моменти спектра й обвідної ---
    Pw = P / (P.sum() + 1e-12)                       # спектр як розподіл по частоті
    mu = float((freqs * Pw).sum())
    var = float((Pw * (freqs - mu) ** 2).sum()); sd = var ** 0.5 + 1e-9
    d['spec_skew'] = _safe(float((Pw * (freqs - mu) ** 3).sum()) / sd ** 3)
    d['spec_kurtosis'] = _safe(float((Pw * (freqs - mu) ** 4).sum()) / sd ** 4 - 3.0)
    # імпульсивність обвідної (kurtosis у часі) — стук/цокання дають важкі хвости
    e = rmsv - rmsv.mean(); es = rmsv.std() + 1e-9
    d['env_kurtosis'] = _safe(float((e ** 4).mean()) / es ** 4 - 3.0)
    # мінливість тембру в часі (std центроїда по кадрах)
    cen_t = (freqs[:, None] * S).sum(0) / (S.sum(0) + 1e-12)
    d['centroid_var'] = _safe(float(np.std(cen_t)))

    # --- ЯКІСТЬ ЗАПИСУ: смуга + реверб ---
    cum = np.cumsum(P); totP = cum[-1] + 1e-12
    d['rolloff99_hz'] = _safe(float(freqs[min(np.searchsorted(cum, 0.99 * totP), len(freqs) - 1)]))
    # зріз смуги ВІДНОСНО шумового порогу (ловить телефон/lossy: енергія вся в низах,
    # тож енергетичні метрики не годяться — шукаємо, де спектр падає до полички шуму)
    Pdb = 10 * np.log10(P + 1e-12)
    Ps = np.convolve(Pdb, np.ones(5) / 5, mode='same')
    floor = float(np.percentile(Ps, 15))
    above = np.where(Ps > floor + 8.0)[0]
    d['cutoff_hz'] = _safe(float(freqs[above.max()]) if above.size else np.nan)
    rms = librosa.feature.rms(S=S)[0]
    act = rms[rms > rms.max() * 0.05]                              # активні кадри
    d['reverb_index'] = (_safe(float(np.percentile(act, 25) / (np.percentile(act, 95) + 1e-12)))
                         if act.size > 8 else np.nan)             # вище => пауза заповнена (реверб)

    # --- attack hardness ---
    onset = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units='frames',
                                       backtrack=True)
    cen_t = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    hf = S[freqs > 2000].sum(axis=0) / (S.sum(axis=0) + 1e-12)
    flux = np.sqrt(((np.diff(S, axis=1).clip(min=0)) ** 2).sum(axis=0))
    ac, ah, af = [], [], []
    for o in onset:
        sl = slice(o, o + 4)               # ~перші 90 мс
        if cen_t[sl].size:
            ac.append(np.nanmax(cen_t[sl])); ah.append(np.nanmax(hf[sl]))
        if o < flux.size:
            af.append(flux[max(o - 1, 0):o + 2].max())
    d['attack_centroid_hz'] = _safe(np.median(ac)) if ac else np.nan
    d['attack_hf_ratio'] = _safe(np.median(ah)) if ah else np.nan
    d['attack_flux'] = _safe(np.median(af)) if af else np.nan

    # --- mechanical artifacts ---
    # ВЧ-транзієнти НЕ на онсетах (стук молоточків/клац)
    hfflux = np.sqrt(((np.diff(S[freqs > 5000], axis=1).clip(min=0)) ** 2).sum(axis=0))
    onset_set = set(int(o) for o in onset)
    nonon = np.array([hfflux[i] for i in range(hfflux.size)
                      if not any(abs(i - o) <= 2 for o in onset_set)])
    d['mech_hf_transient'] = _safe(np.percentile(nonon, 95)) if nonon.size > 8 else np.nan
    # суб-бас (<60 Гц) — гул/педаль/удари по корпусу
    d['mech_subbass'] = _safe(spec[freqs < 60].sum() / (spec.sum() + 1e-12))
    # щільність міжнотних транзієнтів (повна спектр. флюктуація поза онсетами)
    nf = np.array([flux[i] for i in range(flux.size)
                   if not any(abs(i - o) <= 2 for o in onset_set)])
    base = np.median(flux) + 1e-9
    d['mech_interonset'] = _safe(np.mean(nf > 2 * base)) if nf.size > 8 else np.nan

    return {k: _safe(d.get(k, np.nan)) for k in COLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('pats', nargs='*')
    args = ap.parse_args()
    man = list(csv.DictReader(open(MANIFEST, encoding='utf-8')))
    done = {}
    if os.path.exists(OUT):
        done = {r['file']: r for r in csv.DictReader(open(OUT, encoding='utf-8'))}
    todo = [r['file'] for r in man if r['source'] == 'acoustic' and 'm1_patches/' not in r['file']]
    if args.pats:
        todo = [f for f in todo if any(p in f for p in args.pats)]
    todo = [f for f in todo if f not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"До екстракції: {len(todo)} (в кеші: {len(done)})")
    write_header = not os.path.exists(OUT)
    fh = open(OUT, 'a', newline='', encoding='utf-8')
    w = csv.writer(fh)
    if write_header:
        w.writerow(['file'] + COLS)
    for i, f in enumerate(todo):
        p = os.path.join(AUDIO, f)
        if not os.path.exists(p):
            continue
        try:
            d = extract(p)
            if d is None:
                continue
            w.writerow([f] + [('' if not np.isfinite(d[c]) else f"{d[c]:.5g}") for c in COLS])
            fh.flush()
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(todo)}")
        except Exception as e:
            print(f"  [помилка] {f}: {e}")
    fh.close()
    print(f"Готово -> {OUT}")


if __name__ == '__main__':
    main()
