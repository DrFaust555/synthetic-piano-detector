"""I/Q liveness critic: P(synthetic) of one note from the modulation of its partials only.

Each partial is demodulated to its slow complex envelope z_k(t) (an analytic, I/Q, signal); statistics of the
envelopes (zero crossings of |z|, AM depth and skew, phase steps, AM/FM coupling, slow FM, the slope and peakiness of
the envelope spectrum, cross-partial coherence) feed a logistic regression trained to tell real isolated notes from
a physical model's renders of the same notes and velocities (grouped by note).  Timbre, level and spectrum are not
inputs, so a falling P(synthetic) means the modulation came alive.

The fitted critic is data/quality/models/iq_critic.json (standardization + logistic coefficients).  It was trained
on renders of the authors' physical model, which are not distributed, so it is shipped and not retrained here.
p_synth_iq() computes the feature version recorded in the model ("feat_version", 1 when absent).  Version 2 of the
features (_iq_features_v2) separates decay and noise floor from modulation and is kept for comparison.

    SPD_AUDIO=<audio root> python -m quality_axes.liveness <wav> [...]    # P(synthetic | I/Q) per file
"""
from __future__ import annotations
import os
import sys
import glob
import json
import numpy as np

from detector.segment import load_mono

from quality_axes.models import MODELS   # noqa: E402
CLF = os.path.join(MODELS, "iq_critic.json")
SR = 44100

# ---- I/Q-фічі однієї ноти (тільки модуляційний домен!) ----------------------
IQ_FEATS = (
    "nulls_rate",      # проходи |z| біля нуля, на секунду (медіана по парціалях)
    "nulls_iqr",       # розкид по парціалях (реал: різні струни живуть по-різному)
    "env_cv_med",      # глибина AM: std/mean обвідної
    "env_cv_iqr",
    "env_skew",        # асиметрія розподілу обвідної (інтерференція ≠ синус-LFO)
    "env_min_med",     # медіана нормованого мінімуму обвідної (чи гасне в нуль)
    "phstep_p95",      # 95-перцентиль |Δφ| (стрибки фази у провалах)
    "phstep_kurt",     # ексцес Δφ (реал: рідкі великі стрибки; LFO: гладко)
    "amfm_coup",       # кореляція |Δφ| з провалами обвідної
    "fm_med",          # повільний дрейф фази (пітч-вандер носійної)
    "fm_iqr",
    "am_spec_slope",   # нахил спектра обвідної 0.5-12 Гц (реал: 1/f-подібний)
    "am_spec_peaky",   # пік/медіана спектра обвідної (когерентний біт = один пік)
    "xcoh_med",        # крос-парціальна кореляція обвідних (Otey: lockstep)
    "xcoh_iqr",
)


def _iq_features_v1(x: np.ndarray, sr: int, f0: float, B: float = 2e-4,
                    n_part: int = 8, bw: float = 25.0,
                    t0: float = 0.25, t1: float = 2.2) -> dict | None:
    """v1 (модель iq_critic.json навчена на ній). Відомі вади:
    без детренду спаду, без контролю шуму, фіксований ФНЧ 25 Гц
    (бас), вікно 1.95 с, фіксоване B.  Лишено лише для старої моделі."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, dtype=np.float64)
    n1 = int(min(len(x), t1 * sr))
    if n1 < int((t0 + 0.5) * sr):
        return None
    nfft = 1 << 16
    w = x[:int(min(len(x), 1.0 * sr))]
    Xs = np.abs(np.fft.rfft(w * np.hanning(len(w)), nfft))
    frq = np.fft.rfftfreq(nfft, 1.0 / sr)
    sos = butter(4, bw / (sr / 2.0), btype="low", output="sos")
    t = np.arange(len(x)) / sr
    dec = 32
    fs_env = sr / dec
    per, envs = [], []
    k = 1
    while k <= n_part:
        fk = k * f0 * np.sqrt(1.0 + max(B, 0.0) * k * k)
        k += 1
        if fk > min(5000.0, 0.45 * sr):
            break
        m = (frq > fk * 0.97) & (frq < fk * 1.03)
        if m.any():
            fk = float(frq[m][int(np.argmax(Xs[m]))])
        z = (sosfiltfilt(sos, x * np.cos(2 * np.pi * fk * t)) +
             1j * sosfiltfilt(sos, -x * np.sin(2 * np.pi * fk * t)))[int(t0 * sr):n1:dec]
        a = np.abs(z)
        if len(a) < 64 or a.mean() < 1e-9:
            continue
        an = a / (np.median(a) + 1e-12)
        ph = np.unwrap(np.angle(z))
        dph = np.abs(np.diff(ph))
        dur = len(a) / fs_env
        nulls = float(np.sum((an[1:-1] < 0.15) & (an[1:-1] < an[:-2]) &
                             (an[1:-1] < an[2:]))) / dur
        lo = an[:-1] < np.percentile(an, 25)
        coup = float(np.corrcoef(dph, (an[:-1] < np.percentile(an, 25)).astype(float))[0, 1]) \
            if lo.any() and lo.sum() < len(lo) else 0.0
        # спектр обвідної 0.5-12 Гц
        e = an - an.mean()
        nfe = 1 << 12
        E = np.abs(np.fft.rfft(e * np.hanning(len(e)), nfe))
        fe = np.fft.rfftfreq(nfe, 1.0 / fs_env)
        band = (fe >= 0.5) & (fe <= 12.0)
        Eb = E[band] + 1e-12
        feb = fe[band]
        slope = float(np.polyfit(np.log(feb), np.log(Eb), 1)[0])
        peaky = float(20 * np.log10(Eb.max() / (np.median(Eb))))
        d4 = dph - dph.mean()
        kurt = float(np.mean(d4 ** 4) / (np.mean(d4 ** 2) ** 2 + 1e-18))
        per.append(dict(nulls=nulls, cv=float(a.std() / (a.mean() + 1e-12)),
                        skew=float(np.mean((an - an.mean()) ** 3) /
                                   (np.std(an) ** 3 + 1e-12)),
                        emin=float(an.min()), p95=float(np.percentile(dph, 95)),
                        kurt=kurt, coup=coup,
                        fm=float(np.std(np.diff(ph))), slope=slope, peaky=peaky))
        envs.append(e)
    if len(per) < 3:
        return None
    g = lambda key: np.array([p[key] for p in per], dtype=float)
    cohs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            n = min(len(envs[i]), len(envs[j]))
            c = np.corrcoef(envs[i][:n], envs[j][:n])[0, 1]
            if np.isfinite(c):
                cohs.append(abs(c))
    iqr = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))
    return dict(nulls_rate=float(np.median(g("nulls"))), nulls_iqr=iqr(g("nulls")),
                env_cv_med=float(np.median(g("cv"))), env_cv_iqr=iqr(g("cv")),
                env_skew=float(np.median(g("skew"))),
                env_min_med=float(np.median(g("emin"))),
                phstep_p95=float(np.median(g("p95"))),
                phstep_kurt=float(np.median(g("kurt"))),
                amfm_coup=float(np.median(g("coup"))),
                fm_med=float(np.median(g("fm"))), fm_iqr=iqr(g("fm")),
                am_spec_slope=float(np.median(g("slope"))),
                am_spec_peaky=float(np.median(g("peaky"))),
                xcoh_med=float(np.median(cohs)) if cohs else 0.0,
                xcoh_iqr=iqr(np.array(cohs)) if len(cohs) > 3 else 0.0)


# ---- feature version 2: decay and noise floor separated from the modulation ----
T_MAX = 6.0          # с: кінець вікна аналізу (окрема нота); раніше 2.2
SNR_DB = 20.0        # парціал аналізується, доки його згладжена обвідна > шум + 20 дБ
MIN_USE_S = 1.0      # мінімальна корисна довжина вікна парціала
MOD_FLOOR = 5e-3     # модуляція нижче 0.5 % (−46 дБ) вважається нульовою (безрозмірні ознаки не
                     # роздувають залишки фільтра / округлення до повних значень)
NULL_MASK = 0.3      # FM рахується без провалів обвідної (|z| < 0.3 тренду): стрибок фази
                     # ≈ π у нулі биття — властивість суми двох стабільних синусоїд, не дрейф


def _peak(mag, df, f, tol, ref_hw):
    """Параболічно уточнений локальний максимум |X| у f·(1±tol) (не вужче ±2 Гц); None, якщо піка
    немає або він менш ніж на 10 дБ над медіаною |X| у f ± ref_hw (проміжок між парціалами)."""
    hw = max(tol * f, 2.0)
    lo, hi = max(1, int((f - hw) / df)), min(len(mag) - 2, int((f + hw) / df) + 1)
    if hi <= lo + 2:
        return None
    i = lo + int(np.argmax(mag[lo:hi]))
    r0, r1 = max(0, int((f - ref_hw) / df)), min(len(mag), int((f + ref_hw) / df) + 1)
    if not (mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]) or mag[i] < 3.16 * np.median(mag[r0:r1]):
        return None
    a, b, c = np.log(mag[i - 1:i + 2] + 1e-20)
    den = a - 2 * b + c
    return (i + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)) * df


def _track_partials(x, sr, f0, B, n_part, f_max):
    """Частоти парціалів за спектром 0.03–1 с.  Перший шукається біля f0 (±3 %: годиться і f0
    ідеальної струни, і виміряна f1 — різниця ≤ 4 центи при B ≤ 5e-3); далі кожен передбачається від
    ВИМІРЯНОЇ f1 як k·f1·√((1+Bk²)/(1+B)) з B, переоціненим МНК за вже знайденими парціалами
    (стартове B — передане або 1e-4).  Повертає ([(k, f_k)], B)."""
    w = x[int(0.03 * sr):int(1.0 * sr)]
    nfft = 1 << 17
    mag = np.abs(np.fft.rfft(w * np.hanning(len(w)), nfft))
    df = sr / nfft
    f1 = _peak(mag, df, f0, 0.03, 0.45 * f0)
    if f1 is None:
        return [], B
    Bh = float(B) if B and B > 0 else 1e-4
    out = [(1, f1)]
    for k in range(2, n_part + 1):
        pred = k * f1 * np.sqrt((1 + Bh * k * k) / (1 + Bh))
        if pred > f_max:
            break
        fk = _peak(mag, df, pred, 0.03 if len(out) < 3 else 0.015, 0.45 * f1)
        if fk is None:
            continue
        out.append((k, fk))
        ks = np.array([q[0] for q in out[1:]], float)
        r = (np.array([q[1] for q in out[1:]]) / (ks * f1)) ** 2
        den = np.sum((r - ks ** 2) ** 2)
        if den > 0:
            Bh = float(np.clip(np.sum((1 - r) * (r - ks ** 2)) / den, 0.0, 0.05))
    return out, Bh


def _decay_trend(tt, a, n_iter=4):
    """Тренд спаду: поліном 2-го ступеня на log|z|, ітеративно переважений — точки глибоко під
    кривою (провали биття, > 6 дБ нижче) виключаються, тож нулі не тягнуть тренд донизу."""
    la = np.log(a + 1e-30)
    w = np.ones_like(la)
    for _ in range(n_iter):
        c = np.polyfit(tt, la, 2, w=w)
        w = np.where(la - np.polyval(c, tt) > -0.7, 1.0, 1e-3)
    return np.exp(np.polyval(c, tt))


def _mod_stats(z, tr, fs_env):
    """Статистики модуляції комплексної обвідної z відносно тренду спаду tr."""
    a = np.abs(z)
    am = a / tr
    am = am / am.mean()
    e = am - 1.0
    dph = np.diff(np.unwrap(np.angle(z)))
    dph = dph - np.median(dph)                      # залишковий зсув несучої
    keep = (am[:-1] > NULL_MASK) & (am[1:] > NULL_MASK)
    var_f = float(np.var(dph[keep])) if keep.sum() > 16 else float("nan")
    adph = np.abs(dph)
    d4 = dph - dph.mean()
    lo = am[:-1] < np.percentile(am, 25)
    coup = float(np.corrcoef(adph, lo.astype(float))[0, 1]) \
        if 0 < lo.sum() < len(lo) and adph.std() > 0 else 0.0
    return dict(am=am, e=e, var_e=float(np.var(e)), var_f=var_f,
                p95=float(np.percentile(adph, 95)),
                kurt=float(np.mean(d4 ** 4) / (np.mean(d4 ** 2) + MOD_FLOOR ** 2) ** 2), coup=coup)


def _iq_features_v2(x: np.ndarray, sr: int, f0: float, B: float | None = None,
                    n_part: int = 8, t0: float = 0.25, t1: float = T_MAX) -> dict | None:
    """I/Q-ознаки модуляції парціалів, v2.  Відмінності від v1 (за п'ятьма зауваженнями):

    1. спад: log-обвідна кожного парціала детрендується поліномом 2-го ступеня (як melmod);
       усі ознаки AM — від AM(t) = |z|/тренд; шум: рівень шуму в смузі парціала міряється
       демодуляцією посередині між сусідніми парціалами, вікно парціала обрізається, коли
       згладжена обвідна опускається до шум + SNR_DB;
    2. FM (fm_med/fm_iqr) — без ділянок провалів (NULL_MASK); стрибки фази в провалах
       лишаються окремими ознаками биття (phstep_*, amfm_coup);
    3. ФНЧ: смуга min(25, 0.25·f1) Гц — сусідній парціал (на відстані f1) пригнічено ≥ 90 дБ і в
       A0; биття швидші за смугу в найнижчому басі не видно;
    4. вікно до T_MAX = 6 с (биття 0.2 Гц — 1.2 циклу); спектр обвідної — від max(0.5, 2/T) Гц;
    5. частоти: виміряна f1 і B, переоцінене за самими парціалами (не фіксоване 2e-4).
    Безрозмірні ознаки (ексцес, асиметрія, нахил/пікуватість спектра обвідної, кореляції) мають
    підлогу MOD_FLOOR, тож без модуляції дають нейтральні значення, а не роздуті артефакти."""
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=np.float64)
    n1 = int(min(len(x), t1 * sr))
    if n1 < int((t0 + MIN_USE_S) * sr):
        return None
    parts, Bh = _track_partials(x, sr, f0, B, n_part, min(5000.0, 0.45 * sr))
    if len(parts) < 3:
        return None
    f1 = parts[0][1]
    bw = min(25.0, 0.25 * f1)
    sos = butter(4, bw / (sr / 2.0), btype="low", output="sos")
    dec = 32
    fs_env = sr / dec
    t = np.arange(n1) / sr
    xs = x[:n1]
    i0 = int(t0 * sr)

    def demod(f):
        return (sosfiltfilt(sos, xs * np.cos(2 * np.pi * f * t)) +
                1j * sosfiltfilt(sos, -xs * np.sin(2 * np.pi * f * t)))[i0::dec]

    per, envs, uses = [], [], []
    for k, fk in parts:
        z = demod(fk)
        a = np.abs(z)
        if a.mean() < 1e-12:
            continue
        zo = demod(fk + 0.5 * f1)                   # шум смуги: посередині між парціалами
        noise = float(np.median(np.abs(zo)))
        sm = median_filter(a, size=max(3, int(0.25 * fs_env)), mode="nearest")
        below = np.nonzero(sm < noise * 10 ** (SNR_DB / 20))[0]
        n_use = int(below[0]) if len(below) else len(a)
        if n_use < int(MIN_USE_S * fs_env):
            continue
        z, a, zo = z[:n_use], a[:n_use], zo[:n_use]
        tt = np.arange(n_use) / fs_env
        tr = _decay_trend(tt, a)
        dur = n_use / fs_env
        # нульова модель шуму: та сама обвідна без модуляції (тренд) + виміряний шум смуги;
        # її статистики віднімаються / йдуть у знаменник, тож шум не стає «модуляцією»
        S, Sn = _mod_stats(z, tr, fs_env), _mod_stats(tr + zo, tr, fs_env)
        vs = max(S["var_e"] - Sn["var_e"], 0.0)
        vf = max(S["var_f"] - Sn["var_f"], 0.0)
        sig_w = vs / (vs + Sn["var_e"] + MOD_FLOOR ** 2)    # частка справжньої AM у дисперсії
        ph_w = vf / (vf + Sn["var_f"] + MOD_FLOOR ** 2)
        e, am = S["e"], S["am"]
        nulls = float(np.sum((am[1:-1] < 0.15) & (am[1:-1] < am[:-2]) & (am[1:-1] < am[2:]))) / dur
        nfe = 1 << int(np.ceil(np.log2(max(len(e), 4096))))
        win = np.hanning(len(e))
        E = np.abs(np.fft.rfft(e * win, nfe))
        En = np.abs(np.fft.rfft(Sn["e"] * win, nfe))
        Ef = MOD_FLOOR * np.sqrt(np.sum(win ** 2))      # спектр «білої» модуляції рівня підлоги
        fe = np.fft.rfftfreq(nfe, 1.0 / fs_env)
        band = (fe >= max(0.5, 2.0 / dur)) & (fe <= min(12.0, 0.8 * bw))
        # підлога = 3× середня потужність шумового спектра (стала, не одна його реалізація):
        # шум без модуляції дає плаский Eb, справжні піки биття (> 20 дБ над шумом) не зачеплено
        Eb = np.sqrt(E[band] ** 2 + 3.0 * np.mean(En[band] ** 2) + Ef ** 2)
        slope = float(np.polyfit(np.log(fe[band]), np.log(Eb), 1)[0]) if band.sum() > 4 else 0.0
        peaky = float(20 * np.log10(Eb.max() / np.median(Eb))) if band.sum() > 4 else 0.0
        per.append(dict(nulls=nulls, cv=float(np.sqrt(vs)),
                        skew=float(np.mean(e ** 3) / (S["var_e"] + MOD_FLOOR ** 2) ** 1.5) * sig_w,
                        emin=float(am.min()),
                        p95=float(np.sqrt(max(S["p95"] ** 2 - Sn["p95"] ** 2, 0.0))),
                        kurt=S["kurt"] * ph_w, coup=max(S["coup"] - Sn["coup"], 0.0) * sig_w,
                        fm=float(np.sqrt(vf)) if np.isfinite(vf) else float("nan"),
                        slope=slope, peaky=peaky))
        envs.append(e)
        uses.append(dur)
    if len(per) < 3:
        return None
    g = lambda key: np.array([p[key] for p in per], dtype=float)
    cohs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            n = min(len(envs[i]), len(envs[j]))
            a_, b_ = envs[i][:n], envs[j][:n]
            c = np.mean(a_ * b_) / np.sqrt((np.mean(a_ ** 2) + MOD_FLOOR ** 2) *
                                           (np.mean(b_ ** 2) + MOD_FLOOR ** 2))
            cohs.append(abs(float(c)))
    iqr = lambda v: float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25))
    return dict(nulls_rate=float(np.median(g("nulls"))), nulls_iqr=iqr(g("nulls")),
                env_cv_med=float(np.median(g("cv"))), env_cv_iqr=iqr(g("cv")),
                env_skew=float(np.median(g("skew"))),
                env_min_med=float(np.median(g("emin"))),
                phstep_p95=float(np.median(g("p95"))),
                phstep_kurt=float(np.median(g("kurt"))),
                amfm_coup=float(np.median(g("coup"))),
                fm_med=float(np.nanmedian(g("fm"))), fm_iqr=iqr(g("fm")),
                am_spec_slope=float(np.median(g("slope"))),
                am_spec_peaky=float(np.median(g("peaky"))),
                xcoh_med=float(np.median(cohs)) if cohs else 0.0,
                xcoh_iqr=iqr(np.array(cohs)) if len(cohs) > 3 else 0.0,
                n_partials=len(per), use_s=float(np.median(uses)), bw_hz=bw, B_est=Bh)


def iq_note_features(x: np.ndarray, sr: int, f0: float, B: float | None = None,
                     version: int = 2, **kw) -> dict | None:
    """I/Q-ознаки однієї ноти (x починається з удару).  version=2 — виправлена (за замовч.),
    version=1 — та, на якій навчено iq_critic.json (p_synth_iq бере версію з моделі)."""
    if version == 1:
        return _iq_features_v1(x, sr, f0, 2e-4 if B is None else B, **kw)
    return _iq_features_v2(x, sr, f0, B, **kw)


# ---- скоринг ----------------------------------------------------------------
_M = None


def p_synth_iq(x: np.ndarray, sr: int, f0: float, B: float = 2e-4) -> float:
    """P(synth | IQ-патерни модуляції) для однієї ноти. 0 = живі патерни."""
    global _M
    if _M is None:
        _M = json.load(open(CLF, encoding="utf-8"))
    d = iq_note_features(x, sr, f0, B, version=int(_M.get("feat_version", 1)))
    if d is None:
        return float("nan")
    z = (np.array([d[k] for k in _M["feats"]]) - np.array(_M["mu"])) / np.array(_M["sd"])
    lg = float(np.dot(_M["coef"], z) + _M["intercept"])
    return float(1.0 / (1.0 + np.exp(-lg)))


def main():
    from detector.segment import build_segments
    for a in sys.argv[1:]:
        for f in sorted(glob.glob(a)) or [a]:
            try:
                x, sr = load_mono(f)
                segs, _ = build_segments(x, sr)
                ps = [p_synth_iq(x[s.onset:], sr, s.f0, s.B) for s in segs[:8]]
                ps = [p for p in ps if np.isfinite(p)]
                print(f"{os.path.basename(f):40s} P(synth|IQ) = "
                      f"{np.median(ps):.2f} (нот {len(ps)})")
            except Exception as e:
                print(f"{os.path.basename(f):40s} ERR {e}")


if __name__ == "__main__":
    main()
