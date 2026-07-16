"""Детектор противності (aversiveness) — окрема гілка стека якості фортепіано.

Міряє, наскільки звук психоакустично *противний* — ті якості, що роблять неприємними
перфоратор по стіні, гарчання чи скрип — на абсолютній шкалі 0–100.  Спирається на
біологію слухової реакції тривоги/огиди:

  aver_rough   шорсткість у смузі амплітудної модуляції 30–150 Гц (Daniel & Weber 1997;
               Arnal et al. 2015, Current Biology — та сама «шорсткість» крику, що
               активує мигдалину) -> гарчання, буркіт, деренчання перфоратора
  aver_sharp   різкість за Zwicker (von Bismarck/Aures; концентрація енергії 2.5–5.5 кГц)
               -> скрипи, скрегіт, різкі ВЧ
  aver_noise   шумність/негармонічність: частка енергії МІЖ спектральними піками
               (обернена тональність, tone-to-noise ANSI S1.13; вимір неприємності з
               sensory pleasantness Zwicker) -> дриль/скрегіт/гарчання широкосмуго-шумні,
               піаніно тональне.  НАЙСИЛЬНІШИЙ сепаратор.
  aver_impulse повторювані різкі УДАРИ (камінь об камінь, гравій, стукіт) — модуляція
               обвідної 3–30 Гц, медіана по вікнах (одна атака ноти не рахується)
  aver_loud    гучність N5 за Zwicker — лише як діагностика (НЕ у зведенні)

Зведення (aver_raw) — рівне-нормована зважена сума цих вимірів неприємності, а НЕ
loudness-домінована Zwicker PA: сигнал нормований по піку, тож гучність тут була б
конфаундом, а не драйвером.  Маситься у 0–100 перцентильним калібруванням, заякореним
на реальних роялях (низько) проти референсних противних кліпів (високо).  Ваги підібрані
на ref-наборі (не на одному кліпі).  Див. cache/aver_calib.json.

    python -m quality.aversiveness <тека_або_glob> [ще...]   # таблиця по файлах
    python -m quality.aversiveness --calibrate               # перебудувати мапу 0–100
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
import librosa

from detune.segment import load_mono
from quality.psycho2 import bark_loudness, zwicker_sharpness, BARK_HI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CALIB = os.path.join(HERE, 'cache', 'aver_calib.json')

SR = 22050
N_FFT, HOP = 2048, 512
DUR = 12.0                         # аналізуємо перші DUR с (швидкість)

ROUGH_FMIN, ROUGH_FMAX = 100.0, 10000.0
N_ROUGH_BANDS = 10
MOD_LO, MOD_HI = 30.0, 150.0       # смуга «крику»/тривоги (Arnal)
REP_LO, REP_HI = 3.0, 30.0         # смуга «повторюваності»/ударів (нижче за roughness)

# ваги зведення (рівне-нормованого): noise — найсильніший сепаратор, далі sharpness,
# тоді roughness/impulse. Підібрані на ref-наборі, не на одному кліпі.
W_NOISE, W_SHARP, W_ROUGH, W_IMP = 100.0, 15.0, 8.0, 10.0
SHARP0 = 0.95                      # поріг різкості (медіана чистого піаніно)
ROUGH_CAP = 1.2                    # обрізаємо залишковий конфаунд високих нот

KEYS = ['aver_rough', 'aver_sharp', 'aver_noise', 'aver_impulse',
        'aver_loud', 'aver_raw', 'aversiveness']

_CALIB = None
_BARK_IDX_CACHE = {}


# --------------------------------------------------------------------------- #
#  компоненти
# --------------------------------------------------------------------------- #
def _roughness_30_150(x: np.ndarray, sr: int) -> float:
    """Шорсткість у смузі модуляції 30–150 Гц (Daniel–Weber/Arnal).
    Для ~10 півоктавних смуг: bandpass сигналу -> обвідна Гільберта -> bandpass обвідної
    до 30–150 Гц -> глибина модуляції (rms смугової обвідної / середня обвідна).
    Внески смуг зважуються ЕНЕРГІЄЮ смуги — інакше майже тихі смуги високих нот дають
    нестабільне rms/mean і фальшиво завищують шорсткість."""
    nyq = sr / 2.0
    edges = np.geomspace(ROUGH_FMIN, min(ROUGH_FMAX, nyq * 0.9), N_ROUGH_BANDS + 1)
    msos = butter(4, [MOD_LO / nyq, MOD_HI / nyq], btype='band', output='sos')
    num, den = 0.0, 0.0
    for b in range(N_ROUGH_BANDS):
        lo, hi = edges[b], edges[b + 1]
        try:
            bp = sosfiltfilt(butter(4, [lo / nyq, hi / nyq], btype='band', output='sos'), x)
            energy = float(np.mean(bp ** 2))
            env = np.abs(hilbert(bp))
            m = float(env.mean())
            if m <= 1e-9 or energy <= 0:
                continue
            depth = float(np.sqrt(np.mean(sosfiltfilt(msos, env) ** 2)) / m)
            num += energy * depth
            den += energy
        except Exception:
            continue
    return num / den if den else 0.0


def _sharpness(spec: np.ndarray, freqs: np.ndarray) -> float:
    """Різкість за Zwicker (acum) — reuse psycho2."""
    s = zwicker_sharpness(bark_loudness(spec, freqs))
    return float(s) if np.isfinite(s) else 0.0


def _noisiness(S: np.ndarray) -> float:
    """Негармонічність = частка перкусивно/шумової енергії (1 − harmonic ratio) через
    HPSS-розклад (гармоніка vs перкусія).  Найсильніший сепаратор противності:
      дриль/скрегіт/гарчання/скрип — перкусивно-шумові (~0.74–0.83),
      піаніно — гармонічне у БУДЬ-ЯКОМУ регістрі (~0.005–0.16).
    Регістр-робастна (на відміну від частки «між піками»): pp-ноти верхньої октави з
    кількома гармоніками лишаються гармонічними.  Обернена тональність — вимір
    неприємності з sensory pleasantness Zwicker (tone-to-noise, ANSI S1.13)."""
    try:
        H, Pn = librosa.decompose.hpss(S)
        eh = float((H ** 2).sum()); ep = float((Pn ** 2).sum())
        return float(ep / (eh + ep + 1e-12))
    except Exception:
        return 0.0


def _impulse(x: np.ndarray, sr: int, nwin: int = 6) -> float:
    """Повторювані різкі удари (камінь об камінь, гравій/пісок, перфоратор) — біологічно
    сигнал небезпеки.  Міра = модуляція широкосмугової обвідної у смузі 3–30 Гц, МЕДІАНА
    по вікнах: одна атака ноти дає сплеск лише в одному вікні -> медіана низька; повторювані
    удари -> високо в усіх вікнах.  Не плутає одиничну атаку з чергою ударів."""
    nyq = sr / 2.0
    sos = butter(4, [REP_LO / nyq, REP_HI / nyq], btype='band', output='sos')
    minlen = max(int(0.15 * sr), 64)
    L = len(x) // nwin
    if L < minlen:
        nwin, L = 1, len(x)
    vals = []
    for i in range(nwin):
        seg = x[i * L:(i + 1) * L] if nwin > 1 else x
        if seg.size < minlen:
            continue
        try:
            env = np.abs(hilbert(seg))
            m = float(env.mean())
            if m <= 1e-9:
                continue
            vals.append(float(np.sqrt(np.mean(sosfiltfilt(sos, env) ** 2)) / m))
        except Exception:
            continue
    return float(np.median(vals)) if vals else 0.0


def _bark_index(freqs: np.ndarray) -> np.ndarray:
    key = (len(freqs), float(freqs[-1]))
    if key not in _BARK_IDX_CACHE:
        idx = np.searchsorted(BARK_HI, freqs, side='right')
        _BARK_IDX_CACHE[key] = np.clip(idx, 0, len(BARK_HI) - 1)
    return _BARK_IDX_CACHE[key]


def _loudness_n5(S: np.ndarray, freqs: np.ndarray) -> float:
    """Гучність N5 за Zwicker: 95-й перцентиль покадрової суми питомої гучності."""
    bidx = _bark_index(freqs)
    Ez = np.zeros((len(BARK_HI), S.shape[1]))
    np.add.at(Ez, bidx, S ** 2)                       # E(z, t)
    N = np.where(Ez > 0, Ez ** 0.23, 0.0).sum(axis=0)  # покадрова гучність
    return float(np.percentile(N, 95)) if N.size else 0.0


def _combine(rough, sharp, noise, impulse) -> float:
    """Сире зведення противності — рівне-нормована зважена сума психоакустичних вимірів
    неприємності (кожен обґрунтований літературою), а НЕ loudness-домінована Zwicker PA:
    сигнал нормований по піку, тож гучність тут була б конфаундом, а не драйвером.
      noise   — шумність/негармонічність (обернена тональність; найсильніший сепаратор)
      sharp   — різкість Zwicker (von Bismarck/Aures), 2.5–5.5 кГц
      rough   — шорсткість 30–150 Гц (Daniel–Weber/Arnal)
      impulse — повторювана імпульсивність (повільний стукіт)"""
    return float(W_NOISE * noise
                 + W_SHARP * max(0.0, sharp - SHARP0)
                 + W_ROUGH * min(rough, ROUGH_CAP)
                 + W_IMP * impulse)


# --------------------------------------------------------------------------- #
#  калібрування 0–100
# --------------------------------------------------------------------------- #
def _load_calib() -> dict:
    global _CALIB
    if _CALIB is None:
        _CALIB = (json.load(open(CALIB, encoding='utf-8'))
                  if os.path.exists(CALIB) else {'lo': 0.0, 'hi': 1.0})
    return _CALIB


def _to_score(raw: float) -> float:
    c = _load_calib()
    return float(np.clip(100.0 * (raw - c['lo']) / (c['hi'] - c['lo'] + 1e-9), 0.0, 100.0))


# --------------------------------------------------------------------------- #
#  головний API
# --------------------------------------------------------------------------- #
def aversiveness_features(x: np.ndarray, sr: int) -> dict:
    """Компоненти противності + зведений бал 0–100.  Файлового рівня, БЕЗ ML."""
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        sr = SR
    x = x[:int(DUR * sr)]
    if x.size < sr // 2:
        return {k: 0.0 for k in KEYS}
    x = x / (np.max(np.abs(x)) + 1e-9)

    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    spec = S.mean(axis=1)

    rough = _roughness_30_150(x, sr)
    sharp = _sharpness(spec, freqs)
    noise = _noisiness(S)
    impulse = _impulse(x, sr)
    n5 = _loudness_n5(S, freqs)              # репортуємо як діагностику (не в raw)

    raw = _combine(rough, sharp, noise, impulse)
    return {
        'aver_rough': float(rough), 'aver_sharp': float(sharp),
        'aver_noise': float(noise), 'aver_impulse': float(impulse),
        'aver_loud': float(n5), 'aver_raw': float(raw),
        'aversiveness': _to_score(raw),
    }


# --------------------------------------------------------------------------- #
#  CLI / калібрування
# --------------------------------------------------------------------------- #
def _iter_files(args):
    files = []
    for a in args:
        p = a if os.path.isabs(a) else os.path.join(ROOT, a)
        if os.path.isdir(p):
            for ext in ('wav', 'flac', 'ogg', 'mp3', 'aiff', 'aif'):
                files += sorted(glob.glob(os.path.join(p, '**', f'*.{ext}'), recursive=True))
        else:
            files += sorted(glob.glob(p))
    return files


def _raws_in(rel: str, cap: int = 0) -> list:
    files = _iter_files([rel])
    if cap and len(files) > cap:            # рівномірна вибірка по діапазону
        files = files[::max(1, len(files) // cap)][:cap]
    raws = []
    for f in files:
        try:
            x, sr = load_mono(f)
            raws.append(aversiveness_features(x, sr)['aver_raw'])
        except Exception as e:
            print(f"  ! {os.path.basename(f)}: {e}", file=sys.stderr)
    return raws


def calibrate():
    """Заякорити 0–100: реальні роялі (низько) проти противних референсів (високо)."""
    low = _raws_in('audio/ref_top_grand_iowa_steinway', cap=48)
    mid = _raws_in('audio/ref_bad_old_piano', cap=24)
    high = _raws_in('audio/ref_aversive')
    if not low or not high:
        print("Бракує референсів: потрібні audio/ref_top_grand_iowa_steinway (+ref_bright) "
              "та audio/ref_aversive/.  Спершу наповни їх.")
        return
    lo = float(np.percentile(low, 25))      # 0  ≈ типовий гарний рояль
    hi = float(np.percentile(high, 75))     # 100 ≈ типовий противний звук
    json.dump({'lo': lo, 'hi': hi}, open(CALIB, 'w'))
    global _CALIB
    _CALIB = {'lo': lo, 'hi': hi}
    print(f"Записано {os.path.relpath(CALIB, ROOT)}:  lo={lo:.3f}  hi={hi:.3f}")
    for lab, raws in (('top/bright рояль', low), ('старе піаніно', mid), ('противні', high)):
        if raws:
            sc = [_to_score(r) for r in raws]
            print(f"  {lab:16s} n={len(raws):2d}  raw[{min(raws):7.2f}..{max(raws):7.2f}]  "
                  f"score[{min(sc):5.1f}..{max(sc):5.1f}]  median={np.median(sc):5.1f}")


def _selftest():
    """Біологічні критерії: білий шум противніший за рожевий; черга ударів — за тривкий тон."""
    sr, n = SR, int(6 * SR)
    rng = np.random.default_rng(0)
    norm = lambda y: (y / (np.abs(y).max() + 1e-9)).astype(np.float32)
    white = norm(rng.standard_normal(n))
    f = np.fft.rfftfreq(n); f[0] = f[1]
    pink = norm(np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(f), n=n))
    t = np.arange(n) / sr
    tone = norm(sum(np.sin(2 * np.pi * k * 220 * t) / k for k in range(1, 9)))
    impact = np.zeros(n, np.float32); d = int(0.05 * sr)
    for p in range(0, n - d, int(sr / 4)):              # 4 удари/с
        impact[p:p + d] += (np.exp(-np.arange(d) / (0.01 * sr))
                            * rng.standard_normal(d)).astype(np.float32)
    impact = norm(impact)
    sc = {k: aversiveness_features(v, sr)['aversiveness'] for k, v in
          [('білий шум', white), ('рожевий шум', pink),
           ('черга ударів', impact), ('тривкий тон', tone)]}
    for k, v in sc.items():
        print(f"  {k:14s} {v:6.1f}")
    ok1, ok2 = sc['білий шум'] > sc['рожевий шум'], sc['черга ударів'] > sc['тривкий тон']
    print(f"  білий > рожевий: {'OK' if ok1 else 'FAIL'}    "
          f"удари > тон: {'OK' if ok2 else 'FAIL'}")


def main():
    args = sys.argv[1:]
    if '--calibrate' in args:
        calibrate()
        return
    if '--selftest' in args:
        _selftest()
        return
    files = _iter_files(args)
    if not files:
        print("Вкажи теку або glob аудіо (або --calibrate).")
        return
    cols = KEYS
    hdr = {'aver_rough': 'rough', 'aver_sharp': 'sharp', 'aver_noise': 'noise',
           'aver_impulse': 'impulse', 'aver_loud': 'loud', 'aver_raw': 'raw',
           'aversiveness': 'SCORE'}
    print(f"{'file':34s}" + ''.join(f'{hdr[k]:>10s}' for k in cols))
    print('-' * (34 + 10 * len(cols)))
    for f in files:
        try:
            x, sr = load_mono(f)
            d = aversiveness_features(x, sr)
            print(f"{os.path.basename(f)[:34]:34s}" + ''.join(f'{d[k]:10.2f}' for k in cols))
        except Exception as e:
            print(f"{os.path.basename(f)[:34]:34s}  ERR {e}")


if __name__ == '__main__':
    main()
