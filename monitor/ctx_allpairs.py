"""Every GSS item pair (55): boundary-null p-value of the canonical CbD deficit in every year (site_separation.boundary_null_p,
200 simulations, design effect 1 and 2), combined over the years by Fisher's method, then Bonferroni over the 55 pairs.
Years with zero deficit get p = 1. Writes ctx_allpairs.json."""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_separation import wtable, YV, boundary_null_p
H = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(H, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]
S = json.load(open(os.path.join(H, 'site_separation.json'))); rng = np.random.default_rng(23)
pairs = sorted({(r['a'], r['b']) for r in S['L3']}); out = {}
for a, b in pairs:
    ps = {1.0: [], 2.0: []}
    for r in [r for r in S['L3'] if (r['a'], r['b']) == (a, b)]:
        if r['deficit'] <= 0:
            for k in ps: ps[k].append(1.0)
            continue
        sub = d[d.year == r['year']]; TX, nX, _ = wtable(sub, a, b); TY, nY, _ = wtable(sub, YV[a], YV[b])
        for k in ps: ps[k].append(boundary_null_p(TX, TY, nX, nY, r['deficit'], rng, B=200, deff=k)[0])
    res = {}
    for k, v in ps.items():
        X2 = -2 * np.log(v).sum(); res[f'deff{int(k)}'] = dict(fisher=round(float(X2), 2), df=2 * len(v), p=float(chi2.sf(X2, 2 * len(v))))
    res['years_flagged'] = sum(r['contextual'] for r in S['L3'] if (r['a'], r['b']) == (a, b)); out[f'{a}|{b}'] = res
    print(a, b, res, flush=True)
m = len(pairs)
out['_summary'] = {f'deff{k}': dict(min_p=min(v[f'deff{k}']['p'] for kk, v in out.items() if not kk.startswith('_')),
                                    argmin=min(((v[f'deff{k}']['p'], kk) for kk, v in out.items() if not kk.startswith('_')))[1],
                                    surviving=[kk for kk, v in out.items() if not kk.startswith('_') and v[f'deff{k}']['p'] < 0.05 / m]) for k in (1, 2)}
out['_summary']['bonferroni'] = 0.05 / m
print(out['_summary'])
json.dump(out, open(os.path.join(H, 'ctx_allpairs.json'), 'w'), indent=1)
