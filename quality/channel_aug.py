"""Waveform-домен channel augmentation (Фаза 2) — навчає канальну робастність (головне
відоме обмеження: quality-neuro висновок «channel augmentation is the real precondition»).

Доповнює наявну ФАЗОГРАМНУ аугментацію в realness_net.py (не замінює її): та працює у
phaseogram-домені (harmonic-EQ + phase-roll + per-cycle gain), а ця — у ЧАСОВОМУ домені на
сирій хвилі перед будь-якою екстракцією, тож нею можна аугментувати і xgb/panns-фічі, і
позитиви MAESTRO (чистий студійний Disklavier → щоб «real» не став синонімом «чистий канал»).

`augment(x, sr, rng) -> x'` — випадкова КОМПОЗИЦІЯ незалежних компонент (кожна вмикається
з імовірністю ~0.5, детерміновано під seed):
  * синтетичний room-IR   (експоненційно затухаючий шум, RT60 0.2–1.2 c; без завантажень)
  * мікрофонний tilt-EQ    (±6 дБ/окт) + low/high-shelf
  * компресія динаміки
  * шумова підлога          (рожевий шум + гум 50 Гц)
  * легкий кліпінг          (soft-clip; вихід лишається в [-1,1])
  * OGG-roundtrip           (через soundfile; граційно пропускається, якщо кодека нема)

    python -m quality.channel_aug <wav> --out dir --n 5
"""
from __future__ import annotations
import os
import argparse
import numpy as np
from scipy.signal import fftconvolve

SR_DEFAULT = 22050


# --------------------------------------------------------------------------- components
def _room_ir(x, sr, rng):
    rt60 = rng.uniform(0.2, 1.2)
    n_ir = max(8, int(rt60 * sr))
    t = np.arange(n_ir) / sr
    tau = rt60 / 6.9078                                   # -60 dB за RT60
    tail = rng.standard_normal(n_ir) * np.exp(-t / max(tau, 1e-4))
    ir = np.zeros(n_ir, dtype=np.float64)
    ir[0] = 1.0                                           # прямий звук
    wet = rng.uniform(0.1, 0.4)
    ir += wet * (tail / (np.max(np.abs(tail)) + 1e-9))
    y = fftconvolve(x, ir)[:len(x)]
    return y.astype(np.float32)


def _tilt_eq(x, sr, rng):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f_ref = 1000.0
    with np.errstate(divide='ignore'):
        octaves = np.log2(np.maximum(f, 1e-6) / f_ref)
    tilt = rng.uniform(-6.0, 6.0)                         # дБ/окт
    gain_db = tilt * octaves
    # low-shelf + high-shelf
    lo_g, hi_g = rng.uniform(-6, 6), rng.uniform(-6, 6)
    gain_db += lo_g / (1.0 + (f / 200.0) ** 2)            # плавний низький шельф
    gain_db += hi_g * (f ** 2) / (f ** 2 + 4000.0 ** 2)   # плавний високий шельф
    gain = 10.0 ** (np.clip(gain_db, -18, 18) / 20.0)
    return np.fft.irfft(X * gain, n=n).astype(np.float32)


def _compress(x, sr, rng):
    from scipy.signal import lfilter
    thr = rng.uniform(0.1, 0.4)
    ratio = rng.uniform(2.0, 6.0)
    # згладжена обвідна (одно-полюсний слідкувач) e[i]=a·e[i-1]+(1-a)·|x[i]| —
    # векторизовано (Python-цикл по 264k семплів робив повний MAESTRO-прохід повільним)
    a = np.exp(-1.0 / (0.005 * sr))
    env = lfilter([1.0 - a], [1.0, -a], np.abs(x)).astype(np.float32)
    over = env > thr
    gain = np.ones_like(x)
    gain[over] = (thr + (env[over] - thr) / ratio) / (env[over] + 1e-9)
    makeup = rng.uniform(1.0, 1.0 / (thr + (1 - thr) / ratio + 1e-9))   # відновити рівень
    return (x * gain * makeup).astype(np.float32)


def _noise_floor(x, sr, rng):
    n = len(x)
    peak = np.max(np.abs(x)) + 1e-9
    # рожевий шум
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n=n)
    pink /= (np.max(np.abs(pink)) + 1e-9)
    lvl = 10.0 ** (rng.uniform(-55.0, -35.0) / 20.0) * peak
    y = x + lvl * pink.astype(np.float32)
    # гум 50 Гц (+ 2-а гармоніка)
    if rng.random() < 0.5:
        t = np.arange(n) / sr
        hum_lvl = 10.0 ** (rng.uniform(-60.0, -45.0) / 20.0) * peak
        y = y + (hum_lvl * (np.sin(2 * np.pi * 50 * t)
                            + 0.4 * np.sin(2 * np.pi * 100 * t))).astype(np.float32)
    return y.astype(np.float32)


def _soft_clip(x, sr, rng):
    peak = np.max(np.abs(x)) + 1e-9
    drive = rng.uniform(1.1, 2.5)
    y = np.tanh(drive * x / peak) / np.tanh(drive) * peak
    return y.astype(np.float32)


def _ogg_roundtrip(x, sr, rng):
    try:
        import soundfile as sf
        import io
        buf = io.BytesIO()
        pk = np.max(np.abs(x)) + 1e-9
        data = (x / pk * 0.98).astype(np.float32)
        # блоками, НЕ одним sf.write: vorbis-енкодер libsndfile тримає на стеку
        # буфер ~розміру виклику і на довгих кліпах (>~0.5M семплів) валить процес
        # stack overflow (0xC00000FD) — фатально, try/except не рятує
        with sf.SoundFile(buf, 'w', samplerate=sr, channels=1,
                          format='OGG', subtype='VORBIS') as f:
            step = 65536
            for i in range(0, len(data), step):
                f.write(data[i:i + step])
        buf.seek(0)
        y, _ = sf.read(buf, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        return (y[:len(x)] * pk).astype(np.float32) if len(y) >= len(x) \
            else np.pad(y, (0, len(x) - len(y))).astype(np.float32) * pk
    except Exception:
        return x.astype(np.float32)                       # граційно: кодека нема


_COMPONENTS = [_room_ir, _tilt_eq, _compress, _noise_floor, _soft_clip, _ogg_roundtrip]


# --------------------------------------------------------------------------- API
def augment(x: np.ndarray, sr: int = SR_DEFAULT, rng=None, p: float = 0.5) -> np.ndarray:
    """Одна аугментована копія x (та сама довжина/SR). Детермінована під даним rng/seed."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = x.copy()
    for comp in _COMPONENTS:
        if rng.random() < p:
            try:
                yn = comp(y, sr, rng)
                if np.all(np.isfinite(yn)) and len(yn) == len(y):
                    y = yn
            except Exception:
                pass
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    pk = np.max(np.abs(y))
    if pk > 1e-9:
        y = y / pk * 0.99                                 # peak-норм у [-0.99, 0.99]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- CLI
def main():
    import soundfile as sf
    from detune.segment import load_mono
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('wav')
    ap.add_argument('--out', default='.')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    x, sr = load_mono(args.wav)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.wav))[0]
    for i in range(args.n):
        y = augment(x, sr, np.random.default_rng(args.seed + i))
        p = os.path.join(args.out, f"{base}_aug{i}.wav")
        sf.write(p, y, sr)
        print(f"  -> {p}  (peak={np.max(np.abs(y)):.3f})")


if __name__ == '__main__':
    main()
