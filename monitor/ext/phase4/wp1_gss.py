"""WP1b (PREANALYSIS_PHASE4.md): do tear margin, toric type, steepness or a previous chamber change forecast the next chamber change of
GSS item-pair systems beyond the margin shifts?  Usage: python3 wp1_gss.py synth | run  -> results/wp1_gss.json
Implementation detail fixed before the run (not in the plan): a system-year is used if at least 200 respondents answered both items."""
import os, sys, json, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import geom
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.abspath(os.path.join(H, '..', '..'))
sys.path.insert(0, M); from site_separation import ITEMS, YV, wtable
SEED = 20261010

def steps():
    d = pd.read_csv(os.path.join(M, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]; rows = []
    for w, cols in (('X', ITEMS), ('Y', [YV[i] for i in ITEMS])):
        for a, b in itertools.combinations(cols, 2):
            seq = []
            for yr in sorted(d.year.unique()):
                sub = d[d.year == yr]
                if a not in sub or b not in sub: continue
                T, neff, n = wtable(sub, a, b)
                if n >= 200: seq.append((int(yr), T.sum(1), T.sum(0)))
            for k in range(1, len(seq) - 1):
                (y0, f0, g0), (y1, f1, g1), (y2, f2, g2) = seq[k - 1], seq[k], seq[k + 1]
                st = geom.steepness(f1, g1)
                rows.append(dict(system=f'{w}|{a}|{b}', year=y1, next_year=y2,
                                 event=int((geom.signs(f1, g1) != geom.signs(f2, g2)).any()),
                                 tv_prev=0.25 * (np.abs(f1 - f0).sum() + np.abs(g1 - g0).sum()), min_share=float(min(f1.min(), g1.min())),
                                 tm=geom.tear_margin(f1, g1), v=geom.n_vertices(f1, g1), steep=np.log(st) if np.isfinite(st) else np.nan,
                                 prev_change=int((geom.signs(f0, g0) != geom.signs(f1, g1)).any())))
    return pd.DataFrame(rows)

BASE, GEO = ['tv_prev', 'min_share'], ['tm', 'steep', 'prev_change']
def X(F, geo, flex):
    Mx = F[BASE].values
    if flex: Mx = PolynomialFeatures(2, include_bias=False).fit_transform(Mx)
    if geo: Mx = np.hstack([Mx, F[GEO].values] + [np.stack([(F.v == v).astype(float) for v in (9, 12, 15, 18)], 1)])
    return Mx
def ll(p, y): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def compare(tr, te, rng, B=2000):
    P = {}; out = {}
    for name, flex, geo in (('B1', False, False), ('B2', True, False), ('S1', False, True), ('S2', True, True)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X(tr, geo, flex), tr.event.values)
        P[name] = ll(m.predict_proba(X(te, geo, flex))[:, 1], te.event.values); out[f'logloss_{name}'] = round(float(P[name].mean()), 5)
    sysids = te.system.values; uniq = np.unique(sysids); groups = [np.where(sysids == s)[0] for s in uniq]
    for a, b in (('B2', 'S2'), ('B1', 'S1')):
        d = []
        for _ in range(B):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]); d.append((P[a][idx] - P[b][idx]).mean())
        out[f'delta_{a}_{b}'] = round(float(np.mean(d)), 6); out[f'ci_{a}_{b}'] = [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)]
    out['geometry_forecasts_beyond_margins'] = bool(out['ci_B2_S2'][0] > 0)
    out.update(n_train=len(tr), n_test=len(te), event_rate_train=round(float(tr.event.mean()), 3), event_rate_test=round(float(te.event.mean()), 3), systems=len(uniq))
    return out

def run():
    F = steps(); F['steep'] = F.steep.fillna(F.steep.max())
    tr, te = F[F.next_year <= 2004], F[F.next_year >= 2006]
    res = dict(plan='PREANALYSIS_PHASE4.md WP1b', min_respondents=200, result=compare(tr, te, np.random.default_rng(SEED)),
               descriptive=dict(event_rate_by_prev_change={str(k): round(float(v), 3) for k, v in F.groupby('prev_change').event.mean().items()},
                                event_rate_by_tear_margin_quartile={str(k): round(float(v), 3) for k, v in F.groupby(pd.qcut(F.tm, 4, labels=False)).event.mean().items()},
                                vertex_counts={int(k): int(v) for k, v in F.v.value_counts().sort_index().items()}))
    json.dump(res, open(os.path.join(H, 'results', 'wp1_gss.json'), 'w'), indent=1); print(json.dumps(res, indent=1))

def synth():
    rng = np.random.default_rng(9); ok = True
    for kind in ('geometric', 'marginal'):
        n = 3000; F = pd.DataFrame(dict(system=rng.integers(0, 110, n).astype(str), tv_prev=rng.gamma(2, 0.01, n), min_share=rng.uniform(0.05, 0.3, n),
                                        tm=rng.gamma(1.5, 0.02, n), steep=rng.normal(3, 0.5, n), prev_change=rng.integers(0, 2, n), v=rng.choice([9, 12, 15, 18], n)))
        z = (1.0 - 60 * F.tm + 0.8 * F.prev_change) if kind == 'geometric' else (-1.0 + 40 * F.tv_prev)
        F['event'] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        r = compare(F.iloc[:1500], F.iloc[1500:], np.random.default_rng(1), B=500); ok &= r['geometry_forecasts_beyond_margins'] == (kind == 'geometric')
        print(kind, r['delta_B2_S2'], r['ci_B2_S2'])
    print('PASS' if ok else 'FAIL'); open(os.path.join(H, 'results', 'wp1_gss_synth.txt'), 'w').write('PASS' if ok else 'FAIL')
if __name__ == '__main__':
    {'synth': synth, 'run': run}[sys.argv[1]]()
