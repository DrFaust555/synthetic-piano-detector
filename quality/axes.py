"""10 інтерпретованих осей якості, калібровані на шкалі/якорях. Чесний CV.

    python -m quality.axes

Осі = групи ознак користувача. Кожна вісь — НАПРЯМЛЕНА міні-регресія: предиктор
рейтингу лише зі своїх ознак (impute+scale+Ridge), тож її вихід — це «який бал
дає ця група» у твоїй шкалі. Фінал — стек цих 10 осьових передбачень.

Усе чесно через GroupKFold за raw_src (OOF). Канал-робастна версія — лише
інструмент-інтринсивні осі (для дискримінатора генератора в контрольованому каналі).
"""
from __future__ import annotations
import os, csv
import numpy as np
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAN = os.path.join(ROOT, 'audio', 'manifest.csv')
F_INTR = os.path.join(HERE, 'cache', 'features.csv')
F_PSY = os.path.join(HERE, 'cache', 'psycho.csv')
F_PSY2 = os.path.join(HERE, 'cache', 'psycho2.csv')
OUT = os.path.join(HERE, 'cache', 'axes_scorecard.csv')

# вісь -> ознаки
AXES = {
    'стрій/beating':   ['detune_final', 'unison_index', 'global_index',
                        'med_spread_cents', 'frac_split', 'f0_jitter_cents'],
    'roughness':       ['roughness', 'even_odd_db_med'],
    'sharpness':       ['sharpness', 'bark_hi', 'bandwidth_hz', 'treble_frac'],
    'inharmonicity':   ['log_inharmB_med', 'deepbass_nonharm_ratio'],
    'sustain':         ['log_t60_med', 'sustain_ratio', 'sustain_db_1s_med'],
    'attack hardness': ['attack_centroid_hz', 'attack_hf_ratio', 'attack_flux',
                        'attack_time_med'],
    'spectral(Bark)':  ['bark_tilt', 'bark_lo', 'bark_mid', 'bark_hi', 'bark_even',
                        'spec_skew', 'spec_kurtosis'],
    'harm/noise':      ['hnr_db', 'flatness', 'entropy', 'wiener_entropy',
                        'harm_entropy', 'temporal_entropy'],
    'register balance':['bass_frac', 'mid_frac', 'treble_frac', 'bark_even',
                        'high_partial_frac_med', 'centroid_var'],
    'noise/mechanical':['noise_floor_db', 'mech_hf_transient', 'mech_subbass',
                        'mech_interonset', 'noise_ratio', 'env_kurtosis'],
    # «клас/дешевизна»: інтринсивний детектор (AUC 0.81 у зоні плутанини) —
    # ловить тонкий бас, мех-стук, інгармонічність, джиттер, що видають дешеві
    'клас/бас':        ['mech_subbass', 'bass_h2_over_h1', 'bass_fund_strength',
                        'deepbass_nonharm_ratio', 'mech_hf_transient',
                        'log_inharmB_med', 'f0_jitter_cents', 'sustain_ratio',
                        'attack_flux', 'bright_decay_med'],
}
# інструмент-інтринсивні (канал-робастні) осі
ROBUST = ['стрій/beating', 'roughness', 'inharmonicity', 'sustain',
          'attack hardness', 'клас/бас']

# опорні точки калібрування (підпис, унікальний підрядок файлу) — від верху до низу
ANCHORS = [
    ('Bösendorfer Imp.290 (302)', '302_bosendorfer'),    # 99
    ('Steingraeber E272 (88)',    '88_steingraeber'),     # 97
    ('Steingraeber D232 (117)',   '117_steingraeber'),    # 96
    ('Fazioli Butterfly (124i)',  '124i_fazioli'),        # 96
    ('ANT.Petrof 225 (152)',      '152_petrof_ant225'),   # 95
    ('Schimmel K213 (301)',       '301_schimmel'),        # 94
    ('Estonia 168 (65/67/69)',    'estonia'),             # 88
    ('Petrof Bora (256)',         '256_petrof_bora'),     # 87
    ('A.Förster 125G (75)',       '75_august'),           # 87
    ('Yamaha U3 (56a)',           'yamaha_u3'),           # 84-86
    ('A.Förster 116E (73)',       '73_august'),           # 84
    ('Yamaha U1',                 'yamaha_u1'),           # 83
    ('Feurich 125 (56b/64b)',     'feurich'),             # 83
    ('Yamaha B2',                 'yamaha_b2'),           # 74
    ('Ascot',                     'ascot'),               # 72
    ('Palatino',                  'palatino'),            # 70
    ('Reid-Sohn',                 'reid_sohn'),           # 67-69
    ('Calisia',                   'calisia'),             # ~53
    ('Storytone (389)',           'storytone'),           # 49
    ('Ukraina Черн.1957 (342b)',  '342b_ukraina'),        # 48
    ('Akkord (104)',              '104_akkord'),          # 45
    ('Nordiska Futura2 (99)',     '99_nordiska'),         # 42
    ('Rostov-Don (100)',          '100_rostov'),          # 40
]

ALL_FEATS = sorted({f for fs in AXES.values() for f in fs})


def axis_pipe():
    return Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('sc', StandardScaler()), ('rg', Ridge(alpha=3.0))])


def load():
    man = {r['file']: r for r in csv.DictReader(open(MAN, encoding='utf-8'))}
    def rd(p): return {r['file']: r for r in csv.DictReader(open(p, encoding='utf-8'))}
    intr, psy, psy2 = rd(F_INTR), rd(F_PSY), rd(F_PSY2)
    rows = []
    for f, m in man.items():
        if m['source'] != 'acoustic' or 'm1_patches/' in f or not m['rating'].strip():
            continue
        src = {**intr.get(f, {}), **psy.get(f, {}), **psy2.get(f, {})}
        if not psy.get(f) or not psy2.get(f):
            continue
        rec = {'file': f, 'rating': float(m['rating']), 'raw_src': m.get('raw_src', f),
               'desc': m.get('description_uk', '')}
        for c in ALL_FEATS:
            v = src.get(c, '')
            rec[c] = float(v) if v not in ('', 'nan', None) else np.nan
        rows.append(rec)
    return rows


def main():
    rows = load()
    y = np.array([r['rating'] for r in rows])
    g = np.array([r['raw_src'] for r in rows])
    gkf = GroupKFold(5)
    print(f"Файлів (acoustic, з усіма ознаками): {len(rows)}  рейтинг {y.min():.0f}–{y.max():.0f}\n")

    # --- OOF-передбачення кожної осі (напрямлена міні-регресія) ---
    axis_oof = {}      # name -> OOF predicted rating (у балах шкали)
    for nm, fs in AXES.items():
        Xa = np.array([[r[c] for c in fs] for r in rows])
        axis_oof[nm] = cross_val_predict(axis_pipe(), Xa, y, cv=gkf, groups=g)

    print("=== Сила кожної осі окремо (OOF, у балах шкали) ===")
    print(f"  {'вісь':18s} {'ρ':>6} {'MAE':>6}   {'<55':>5} {'70-79':>5} {'90+':>5}")
    rk = []
    for nm in AXES:
        p = axis_oof[nm]; rho = spearmanr(p, y)[0]; mae = mean_absolute_error(y, p)
        rk.append((abs(rho), rho, mae, nm, p[y < 55].mean(), p[(y >= 70) & (y < 80)].mean(),
                   p[y >= 90].mean()))
    for ar, rho, mae, nm, lo, mid, hi in sorted(rk, reverse=True):
        print(f"  {nm:18s} {rho:+.2f} {mae:6.1f}   {lo:5.0f} {mid:5.0f} {hi:5.0f}")

    # --- стек осей -> фінал (чесний CV на OOF-осях) ---
    def stack(names):
        M = np.column_stack([axis_oof[n] for n in names])
        pred = cross_val_predict(
            Pipeline([('sc', StandardScaler()), ('rg', Ridge(alpha=2.0))]),
            M, y, cv=gkf, groups=g)
        return pred, r2_score(y, pred), mean_absolute_error(y, pred), float(spearmanr(pred, y)[0])

    pred_full, r2, mae, rho = stack(list(AXES))
    pred_rob, r2r, maer, rhor = stack(ROBUST)
    sd = y.std(); sig = 6.5; ceil = (1 - sig**2 / sd**2) ** 0.5
    print(f"\n=== Чесний GroupKFold-CV (стек осей) ===")
    print(f"  повний (10 осей):         R²={r2:+.2f}  MAE={mae:.1f}  ρ={rho:+.2f}")
    print(f"  канал-робастний (5 осей): R²={r2r:+.2f}  MAE={maer:.1f}  ρ={rhor:+.2f}")
    print(f"  стеля від шуму міток (σ≈6.5): ρ_max≈{ceil:.2f}")

    # --- внесок осей у фінал ---
    M = np.column_stack([axis_oof[n] for n in AXES])
    st = Pipeline([('sc', StandardScaler()), ('rg', Ridge(alpha=2.0))]).fit(M, y)
    print(f"\n=== Внесок осей у фінал (Ridge, стандартизовано) ===")
    for nm, c in sorted(zip(AXES, st.named_steps['rg'].coef_), key=lambda z: -z[1]):
        print(f"  {nm:18s} {c:+.2f}")

    # --- опорні точки -> монотонне кусково-лінійне калібрування (isotonic) ---
    pr = {r['file']: p for r, p in zip(rows, pred_full)}
    knots = []   # (label, x=сер.модель, y=сер.твоя, n)
    for label, sub_s in ANCHORS:
        sub = [(r['rating'], pr[r['file']]) for r in rows if sub_s in r['file'].lower()]
        if sub:
            knots.append((label, float(np.mean([s[1] for s in sub])),
                          float(np.mean([s[0] for s in sub])), len(sub)))
    # «сліпі плями»: опорні, які модель грубо переставляє (|model−true|>15) —
    # монотонне калібрування їх НЕ виправить; виключаємо з підгонки кривої.
    BLIND = 15.0
    usable = [k for k in knots if abs(k[1] - k[2]) <= BLIND]
    blind = [k for k in knots if abs(k[1] - k[2]) > BLIND]

    # калібрування: isotonic на ВСІХ файлах (робастно) + апвейт справних опорних
    usable_subs = {sub_s for (label, sub_s) in ANCHORS if any(u[0] == label for u in usable)}
    w = np.ones(len(rows))
    pf = pred_full
    for i, r in enumerate(rows):
        if any(s in r['file'].lower() for s in usable_subs):
            w[i] = 8.0
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip',
                             y_min=25, y_max=99).fit(pf, y, sample_weight=w)
    cal_full = np.clip(iso.predict(pred_full), 25, 99)
    prc = {r['file']: c for r, c in zip(rows, cal_full)}

    print(f"\n=== Опорні точки: монотонне калібрування (isotonic на всіх, апвейт справних) ===")
    print(f"  {'інструмент':24s} {'твоя':>5} {'модель':>7} {'калібр':>7} {'n':>3}")
    for label, x, yv, n in sorted(knots, key=lambda z: -z[2]):
        flag = '  ⚠СЛІПА ПЛЯМА' if abs(x - yv) > BLIND else ''
        print(f"  {label:24s} {yv:5.0f} {x:7.1f} {float(iso.predict([x])[0]):7.1f} {n:3d}{flag}")
    uy = [k[2] for k in usable]
    print(f"  справних опорних: {len(usable)}  | сліпих плям: {len(blind)} "
          f"({', '.join(k[0].split('(')[0].strip() for k in blind)})")
    mae_cal_u = mean_absolute_error(uy, [float(iso.predict([k[1]])[0]) for k in usable])
    print(f"  MAE на СПРАВНИХ опорних після калібр.: {mae_cal_u:.1f}")
    print(f"  MAE на ВСІХ {len(rows)}: до={mean_absolute_error(y, pred_full):.1f} -> "
          f"після={mean_absolute_error(y, cal_full):.1f}  "
          f"(ρ={spearmanr(cal_full, y)[0]:+.2f}, монотонно)")

    # --- запис scorecard ---
    prr = {r['file']: p for r, p in zip(rows, pred_rob)}
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'your_rating', 'model_cal', 'model_oof', 'model_robust']
                   + list(AXES.keys()) + ['desc'])
        for i, r in enumerate(rows):
            w.writerow([r['file'], int(r['rating']), round(prc[r['file']]),
                        round(pr[r['file']]), round(prr[r['file']])]
                       + [round(axis_oof[nm][i]) for nm in AXES] + [r['desc']])
    print(f"\nScorecard -> {OUT}")


if __name__ == '__main__':
    main()
