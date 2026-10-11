"""Heart Disease (UCI, four sites): analyses of PREANALYSIS_AMENDMENT_OPEN_DATA.md, Part III (illustration only).
Usage: python3 heart.py [--zip PATH] [--out PATH] [--quick]"""
import os, sys, json, zipfile, argparse, itertools
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, 'lib'))
from analyses import cbd_strata, table, shown
from mimic_monitor import cf_triangle
from scipy.stats import chi2
COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
SITES = {'cleveland': 'processed.cleveland.data', 'hungarian': 'processed.hungarian.data', 'switzerland': 'processed.switzerland.data', 'va': 'processed.va.data'}

def load(zpath):
    z = zipfile.ZipFile(zpath); out = []
    for site, fn in SITES.items():
        name = [n for n in z.namelist() if n.endswith(fn)][0]
        d = pd.read_csv(z.open(name), header=None, names=COLS, na_values='?'); d['site'] = site; out.append(d)
    d = pd.concat(out, ignore_index=True)
    d['CP'] = d.cp.map({1: 0, 2: 0, 3: 1, 4: 2}); d['EX'] = d.exang; d['DX'] = (d.num > 0).astype(int)
    return d

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--zip', default=os.path.join(H, 'data', 'uci', 'heartdisease.zip'))
    ap.add_argument('--out', default=os.path.join(H, 'results', 'results_heart.json')); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); rng = np.random.default_rng(20261009); B = 50 if a.quick else 400; d = load(a.zip)
    res = dict(dataset='Heart Disease (UCI), four sites', plan='PREANALYSIS_AMENDMENT_OPEN_DATA.md (2026-10-10), Part III (illustration)',
               cohort={s: int((d.site == s).sum()) for s in SITES})
    D = {}; ps = []
    for s1, s2 in itertools.combinations(SITES, 2):
        e = d.dropna(subset=['CP', 'EX'])
        strata = {f'diagnosis {k}': (table(e[(e.site == s1) & (e.DX == k)].CP, e[(e.site == s1) & (e.DX == k)].EX, 3, 2),
                                     table(e[(e.site == s2) & (e.DX == k)].CP, e[(e.site == s2) & (e.DX == k)].EX, 3, 2)) for k in (0, 1)}
        r = cbd_strata(strata, (3, 2), rng, B=B); r['bonferroni_6'] = bool(r['p'] is not None and r['p'] < 0.05 / 6); D[f'{s1} vs {s2}'] = r
    res['D_contextuality_sites'] = D
    res['E_missing_share_by_site'] = {s: {c: round(float(d[d.site == s][c].isna().mean()), 3) for c in COLS if d[c].isna().any()} for s in SITES}
    e = d.dropna(subset=['age']).copy(); e['AG'] = np.select([e.age < 50, e.age < 60], [0, 1], 2); e['TH'] = e.thal.map({3: 0, 6: 1, 7: 2})
    Txy = table(e.AG, e.num.clip(upper=2), 3, 3); m = e.TH.notna()
    Txz = table(e[m].AG, e[m].TH, 3, 3); Tyz = table(e[m].num.clip(upper=2), e[m].TH, 3, 3)
    res['E_selection'] = (dict(cf_available_case=round(cf_triangle(Txy, Txz, Tyz), 4)) if min(Txy.min(), Txz.min(), Tyz.min()) >= 10
                          else 'not estimable: a table has a cell below 10')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)); print('wrote', a.out)
