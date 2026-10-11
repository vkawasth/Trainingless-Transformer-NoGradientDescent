"""Sampling uncertainty of the Contextuality-by-Default verdicts for the GSS wording systems, in the canonical form (all
dichotomizations, monitor.cbd_split). For each year, the two context tables (wording X, wording Y) get independent Dirichlet
posteriors with concentration n_eff * table + 1/2 (Kish effective size, Jeffreys prior); the reported number is the posterior
probability that the system is contextual (1000 draws, so resolution 0.001). This is a screen, not a test: it uses only the Kish
effective size and ignores clustering. Also with only the order-respecting cuts of the ordered answer scale (sensitivity analysis)."""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import cbd_split
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, 'data', 'gss')
d = pd.read_csv(os.path.join(G, 'gss_extract.csv'), low_memory=False)
PAIRS = {'welfare-cities': (('natfare', 'natcity'), ('natfarey', 'natcityy')), 'welfare-race': (('natfare', 'natrace'), ('natfarey', 'natracey')),
         'health-education': (('natheal', 'nateduc'), ('nathealy', 'nateducy'))}
def wtable(sub, a, b):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub['wtssps'].notna(); s = sub[m]; w = s['wtssps'].values
    T = np.zeros((3, 3)); np.add.at(T, (s[a].astype(int).values - 1, s[b].astype(int).values - 1), w)
    return T / T.sum(), w.sum() ** 2 / (w ** 2).sum()
r = json.load(open(os.path.join(H, 'gss_results.json'))); rng = np.random.default_rng(3); out = {}; B = 1000
for name, (X, Y) in PAIRS.items():
    res, resc = {}, {}
    for yr in sorted(int(y) for y in r[name]['cbd']):
        sub = d[d.year == yr]; TX, nX = wtable(sub, *X); TY, nY = wtable(sub, *Y)
        hits = hcut = 0
        for _ in range(B):
            dX, dY = rng.dirichlet(nX * TX.ravel() + 0.5).reshape(3, 3), rng.dirichlet(nY * TY.ravel() + 0.5).reshape(3, 3)
            hits += cbd_split(dX, dY)['contextual']; hcut += cbd_split(dX, dY, ordinal=True)['contextual']
        res[yr] = round(hits / B, 3); resc[yr] = round(hcut / B, 3)
    flagged = [int(y) for y, v in r[name]['cbd'].items() if v['split_contextual']]
    out[name] = dict(flagged=flagged, post_contextual=res, flagged_cuts=[int(y) for y, v in r[name]['cbd'].items() if v['cuts_contextual']], post_contextual_cuts=resc)
    print(name, "| point-estimate split flags:", flagged, "| posterior P(contextual) there:", {y: res[y] for y in flagged},
          "| max over years:", max(res.values()), "| years with P > 0.95:", [y for y, v in res.items() if v > 0.95],
          "| cuts only: flags", out[name]['flagged_cuts'], "max", max(resc.values()), "> 0.95:", [y for y, v in resc.items() if v > 0.95])
json.dump(out, open(os.path.join(H, 'cbd_boot.json'), 'w'), indent=1)
