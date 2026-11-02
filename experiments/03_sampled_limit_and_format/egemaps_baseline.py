"""eGeMAPS baseline on the isolated-note population (Sec. 4.3): same notes, folds and T2 as the 27-feature single-note
detector; velocity copies of the SoundFont renders removed; library notes never trained, scored by the model of
their group's fold.  The 88 eGeMAPSv02 functionals are read from data/features/egemaps.csv (corpus/egemaps_extract.py:
native files -- loader 'file' for corpus notes, 'mono' for library notes -- and the listening-clip format, loader
'clip').  XGBoost with the protocol's settings.

Paper numbers (Sec. 4.3), column "native files" vs "our 27 features": algorithmic notes AUC 0.948 vs 0.943; library
notes, mf layer, 0.710 vs 0.470.

    python experiments/03_sampled_limit_and_format/egemaps_baseline.py
        -> results/03_sampled_limit_and_format/egemaps_baseline.md
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('03_sampled_limit_and_format')
ROOT = AUDIO_ROOT
import protocol.common as L
import protocol.corpus as V


def library_notes():
    """Isolated notes of the four sample libraries with their 47 features (data/features/library_notes_iso1.csv);
    notes with fewer than 30 defined features or an extraction error are left out."""
    import protocol.corpus_base as P
    libs = P.library_note_files()
    lf = L.extract_cached(list(libs['path']), 'library_notes', isolated=True)
    libs = pd.concat([libs.reset_index(drop=True), lf], axis=1)
    libs['grp'] = 'syn:lib:' + libs['library']
    libs['valid'] = (libs[list(L.FEATURE_COLS)].notna().sum(axis=1) >= 30) & (libs['error'].fillna('') == '')
    return libs[libs['valid']].reset_index(drop=True)


def main():
    e = pd.read_csv(os.path.join(DATA, 'features', 'egemaps.csv'), low_memory=False)
    FE = [c for c in e.columns[10:]]
    assert len(FE) == 88, len(FE)
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    iso = df[df.kind.isin(['real_isolated', 'algo_single']) & ~df.sf_dup].reset_index(drop=True)
    iso['rel'] = [os.path.relpath(V.audio_path(f), ROOT).replace(os.sep, '/') for f in iso.file]
    lib = library_notes()
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in lib.grp.unique()}
    lib['fold'] = lib.grp.map(lib_fold).astype(int)
    lib['rel'] = [os.path.relpath(p, ROOT).replace(os.sep, '/') for p in lib.path]
    nat = pd.concat([e[e.loader == 'file'], e[e.loader == 'mono']]).drop_duplicates('file').set_index('file')
    clp = e[e.loader == 'clip'].drop_duplicates('file').set_index('file')
    res = {}
    for fmt, src in (('native files', nat), ('clip format', clp)):
        X = src.reindex(iso.rel)[FE].values.astype(float)
        Xl = src.reindex(lib.rel)[FE].values.astype(float)
        miss = int(np.isnan(X).all(1).sum()), int(np.isnan(Xl).all(1).sum())
        b = iso.copy()
        for j, c in enumerate(FE):
            b[c] = X[:, j]
        lb = lib.copy()
        for j, c in enumerate(FE):
            lb[c] = Xl[:, j]
        r = V.run_cv3(b, FE, 'T2', ext={'lib': lb}, ext_fold={'lib': lb.fold.values})
        res[fmt] = (r['own'], r['ext_own']['lib'], miss)
    r27 = V.run_cv3(iso, V.FS3_ISO['all47'], 'T2', ext={'lib': lib}, ext_fold={'lib': lib.fold.values})
    res['our 27 features'] = (r27['own'], r27['ext_own']['lib'], (0, 0))
    y, grp = iso.y.values, iso.grp.values
    cc = (iso.kind == 'real_isolated').values
    rows = [('SoundFont', (iso.subkind == 'soundfont_single').values), ('Otey', (iso.subkind == 'otey_single').values),
            ('algorithmic (both)', (iso.kind == 'algo_single').values)]
    mf = lib.mf_layer.astype(bool).values
    md = ['# eGeMAPS on the protocol-v3 isolated-note population (T2, own-fold)\n\n',
          f'{int(cc.sum())} CCMusic notes, {int((~cc).sum())} algorithmic notes, {len(lib)} library notes (never trained). '
          'Cells: AUC [95% CI] vs CCMusic; det = share with P > 0.5. "clip format": lead-in, duration (3 s), level and '
          'sample rate made equal for all files.\n\n',
          '| CCMusic notes vs | ' + ' | '.join(res) + ' |\n' + '|---' * (len(res) + 1) + '|\n']
    for lab, m in rows:
        md.append(f'| {lab} | ' + ' | '.join(V.fmt3(V.cmp3(y, p, grp, cc, m)) for p, _, _ in res.values()) + ' |\n')
    for lab, sel in (('library notes, mf layer', mf), ('library notes, all layers', np.ones(len(lib), bool))):
        cells = []
        for p, pl, _ in res.values():
            yy = np.r_[np.zeros(cc.sum()), np.ones(sel.sum())]
            cells.append(V.fmt3(V.cmp3(yy, np.r_[p[cc], pl[sel]], np.r_[grp[cc], lib.grp.values[sel]], yy == 0, yy == 1)))
        md.append(f'| {lab} | ' + ' | '.join(cells) + ' |\n')
    md.append('| CCMusic false alarms (P > 0.5) | ' + ' | '.join(f'{np.mean(p[cc] > 0.5):.1%}' for p, _, _ in res.values()) + ' |\n')
    md.append('\nFiles without eGeMAPS features (corpus / library): '
              + ', '.join(f'{k} {v[2]}' for k, v in res.items()) + '.\n')
    open(os.path.join(OUT, 'egemaps_baseline.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
