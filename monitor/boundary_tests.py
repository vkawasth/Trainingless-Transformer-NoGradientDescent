"""Boundary-null tests for the canonical-CbD flags of site_separation.json: every system-year with posterior > 0.95, and every year of
the crime-drugs system; Fisher combination over the years of crime-drugs. Writes boundary_tests.json."""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_separation import wtable, YV, boundary_null_p
H = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(H, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]
S = json.load(open(os.path.join(H, 'site_separation.json'))); rng = np.random.default_rng(17); out = {'flags': [], 'crime_drugs': []}
targets = [r for r in S['L3'] if r.get('posterior', 0) > 0.95]
cd = [r for r in S['L3'] if r['a'] == 'natcrime' and r['b'] == 'natdrug']
for lst, key in ((targets, 'flags'), (cd, 'crime_drugs')):
    for r in lst:
        sub = d[d.year == r['year']]; TX, nX, _ = wtable(sub, r['a'], r['b']); TY, nY, _ = wtable(sub, YV[r['a']], YV[r['b']])
        if r['deficit'] <= 0: out[key].append(dict(year=r['year'], a=r['a'], b=r['b'], deficit=0.0, p1=1.0, p2=1.0)); continue
        p1, lam = boundary_null_p(TX, TY, nX, nY, r['deficit'], rng, B=500, deff=1.0)
        p2, _ = boundary_null_p(TX, TY, nX, nY, r['deficit'], rng, B=500, deff=2.0)
        out[key].append(dict(year=r['year'], a=r['a'], b=r['b'], deficit=r['deficit'], p1=round(p1, 4), p2=round(p2, 4), lam=round(lam, 4)))
        print(key, r['year'], r['a'], r['b'], 'deficit', r['deficit'], 'p (deff 1)', round(p1, 4), 'p (deff 2)', round(p2, 4), flush=True)
for k in ('p1', 'p2'):
    ps = [x[k] for x in out['crime_drugs']]; X2 = -2 * np.log(ps).sum()
    out['crime_drugs_fisher_' + k] = dict(stat=round(float(X2), 2), df=2 * len(ps), p=float(chi2.sf(X2, 2 * len(ps))))
print('crime-drugs, Fisher over years:', out['crime_drugs_fisher_p1'], out['crime_drugs_fisher_p2'])
m = len(S['L3']); out['bonferroni'] = 0.05 / m
out['flags_surviving_bonferroni'] = [(x['year'], x['a'], x['b']) for x in out['flags'] if x['p2'] < 0.05 / m]
print('flags with p (deff 2) below 0.05 /', m, ':', out['flags_surviving_bonferroni'])
json.dump(out, open(os.path.join(H, 'boundary_tests.json'), 'w'), indent=1)
