"""Toric and Floer-theoretic structure of couplings with margins taken from public data (Bridges, Secs. 5 and 8).

2 x 3 systems (a binary content A and a three-valued content B): the coupling polygon in the coordinates (a11, a12), its facet cells
and toric surface X_{f,g} (Prop. fano23), the wall signs (chamber), the nonzero torus branes (critical points of the Cho-Oh potential;
exact valuations from Newton polygons, Thm sweep23), the leximin coupling, the product (maximum-entropy) coupling, Guillemin's
critical point, the observed coupling, and the entropy-brane isotopy phi_t = (1-t) H + t log prod(p) of Prop. sheaf (its critical
point on the fibre for t in [0, 1]).
  * NHANES 2009-10 and 2011-12, adults 20+: diabetes x BMI class (exam weights).
  * GSS 1985-2024, wording Y: 'assistance to the poor' (too little / not) x 'assistance to big cities' (3 answers), and wording X:
    health (too little / not) x education (3 answers); every year, with the chamber decided on the bands or not.
3 x 3 systems: the number of vertices of the coupling polytope (9, 12, 15, 18; Fano iff 9, Prop. semifano33), every GSS item pair and
year, and around the decided tear of welfare x race (wording Y, 2014 -> 2016).
Writes toric_public.json."""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np, pandas as pd, pyreadr
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, '..', 'bridges', 'fukaya'))
from sweep23 import NU, lam, facets_of, self_int, surface, leximin as lex23, wallvals, WALLS
from newton23 import exact_positions
from nef33 import vertices as vert33
from monitor import bands, wall_signs, State
from scipy.optimize import minimize
D = 1000

def rat(x): return Fr(int(round(float(x) * D)), D)

def cells(x1, x2, f1, g):
    """The six cells at (a11, a12) = (x1, x2)."""
    return np.array([x1, x2, f1 - x1 - x2, g[0] - x1, g[1] - x2, g[2] - f1 + x1 + x2])

def polygon(f1, g):
    """Vertices of the coupling polygon, in order."""
    from scipy.spatial import ConvexHull
    pts = []
    A = np.array([[1, 0], [0, 1], [-1, -1], [-1, 0], [0, -1], [1, 1]], float); b = -np.array([0, 0, f1, g[0], g[1], g[2] - f1])
    for i, j in itertools.combinations(range(6), 2):
        M = A[[i, j]]
        if abs(np.linalg.det(M)) < 1e-12: continue
        x = np.linalg.solve(M, b[[i, j]])
        if (A @ x - b >= -1e-12).all(): pts.append(x)
    pts = np.unique(np.round(pts, 12), axis=0); hull = ConvexHull(pts); return pts[hull.vertices]

def argmax_on_fibre(fun, f1, g, x0):
    r = minimize(lambda x: -fun(cells(x[0], x[1], f1, g)), x0, method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    return r.x

def analyse23(f1f, gf, obs=None, ne=None):
    f1, g = rat(f1f), (rat(gf[0]), rat(gf[1]), 1 - rat(gf[0]) - rat(gf[1]))
    if any(v == 0 for v in wallvals(f1, g)): f1 += Fr(1, 10 * D)                 # nudge off a wall at the rounding scale
    fac = facets_of(f1, g); D2 = self_int(fac); kind = surface(fac, D2)
    v1, v2, vp, vq, sols = exact_positions(fac, f1, g)
    br = sorted(list(sols)[0]) if len(sols) == 1 else None
    ul, levels = lex23(fac, f1, g)
    f1x, gx = float(f1), [float(x) for x in g]
    prod = (f1x * gx[0], f1x * gx[1])
    x0 = np.array(prod)
    H_ = lambda c: -np.sum(c * np.log(np.clip(c, 1e-300, None)))
    guill = argmax_on_fibre(lambda c: -0.5 * sum(c[k] * np.log(max(c[k], 1e-300)) for k, cc in enumerate(NU) if cc in fac), f1x, gx, x0)
    iso = [argmax_on_fibre(lambda c, t=t: (1 - t) * H_(c) + t * np.sum(np.log(np.clip(c, 1e-300, None))), f1x, gx, x0).round(6).tolist() for t in np.linspace(0, 1, 11)]
    res = dict(f1=str(f1), g=[str(x) for x in g], facets=[f'{a+1}{b+1}' for a, b in fac], surface=kind, n_vertices=len(fac),
               walls=[1 if v > 0 else -1 for v in wallvals(f1, g)], polygon=polygon(f1x, gx).round(6).tolist(),
               branes=None if br is None else [[float(a), float(b)] for a, b in br],
               leximin=[float(ul[0]), float(ul[1])], leximin_levels=[float(x) for x in levels], product=list(prod), guillemin=guill.round(6).tolist(), isotopy=iso)
    if br is not None:
        from collections import Counter
        cnt = Counter((float(a), float(b)) for a, b in br)
        res['branes_at_leximin'] = cnt.get((float(ul[0]), float(ul[1])), 0); res['branes_at_product'] = sum(v for k, v in cnt.items() if np.allclose(k, prod, atol=1e-12))
    if obs is not None:
        res['observed'] = [float(obs[0, 0]), float(obs[0, 1])]
        if ne is not None:      # is the chamber (hence the toric surface) decided by the data?
            S = State(obs, ne); res['chamber_decided'] = bool((S.signs != 0).all())
    return res

def wt23(sub, a, b, binary_first=True):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub['wtssps'].notna(); s = sub[m]; w = s['wtssps'].values
    T = np.zeros((2, 3)); np.add.at(T, ((s[a].values != 1).astype(int), s[b].astype(int).values - 1), w)
    return T / T.sum(), w.sum() ** 2 / (w ** 2).sum()

if __name__ == '__main__':
    out = {}
    # NHANES
    df = list(pyreadr.read_r(os.path.join(H, 'data', 'nhanes', 'NHANESraw.rda')).values())[0]
    df = df[df.Age >= 20]; df['diab'] = df['Diabetes'].astype(str).map({'Yes': 0, 'No': 1})
    df['bmi3'] = pd.cut(df['BMI'], [0, 25, 30, 200], labels=[0, 1, 2], right=False).astype(float); df['w'] = df['WTMEC2YR']
    for cyc in sorted(df.SurveyYr.astype(str).unique()):
        s = df[(df.SurveyYr.astype(str) == cyc) & df.diab.notna() & df.bmi3.notna() & df.w.notna()]
        T = np.zeros((2, 3)); np.add.at(T, (s.diab.astype(int).values, s.bmi3.astype(int).values), s.w.values); T /= T.sum()
        ne = s.w.sum() ** 2 / (s.w ** 2).sum()
        out[f'NHANES {cyc}'] = r = analyse23(T[0].sum(), T.sum(0), T, ne)
        print('NHANES', cyc, r['surface'], r['facets'], 'branes', r['branes'], 'leximin', r['leximin'], 'product', [round(x, 4) for x in r['product']], 'observed', [round(x, 4) for x in r['observed']], flush=True)
    # GSS 2 x 3, every year
    d = pd.read_csv(os.path.join(H, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]
    for name, (a, b) in {'poor x cities (Y)': ('natfarey', 'natcityy'), 'health x education (X)': ('natheal', 'nateduc')}.items():
        series = {}
        for yr in sorted(d.year.unique()):
            T, ne = wt23(d[d.year == yr], a, b)
            series[int(yr)] = analyse23(T[0].sum(), T.sum(0), T, ne)
        out[name] = series
        kinds = [(y, r['surface'], r['chamber_decided'], r.get('branes_at_leximin'), r.get('branes_at_product')) for y, r in series.items()]
        print(name, kinds, flush=True)
    # GSS 3 x 3: vertex counts
    ITEMS = ['natfare', 'natcity', 'natrace', 'natcrime', 'natdrug', 'nateduc', 'natheal', 'natenvir', 'natspac', 'natarms', 'nataid']
    YV = {i: (i + 'y').replace('natspacy', 'natspacy') for i in ITEMS}
    YV.update({'natspac': 'natspacy', 'natenvir': 'natenviy', 'natheal': 'nathealy', 'natcity': 'natcityy', 'natcrime': 'natcrimy', 'natdrug': 'natdrugy',
               'nateduc': 'nateducy', 'natrace': 'natracey', 'natarms': 'natarmsy', 'nataid': 'nataidy', 'natfare': 'natfarey'})
    from collections import Counter
    vc = Counter(); series33 = {}
    for yr in sorted(d.year.unique()):
        sub = d[d.year == yr]
        for a, b in itertools.combinations(ITEMS, 2):
            for w, (aa, bb) in (('X', (a, b)), ('Y', (YV[a], YV[b]))):
                m = sub[aa].isin([1, 2, 3]) & sub[bb].isin([1, 2, 3]) & sub['wtssps'].notna(); s = sub[m]
                if len(s) < 100: continue
                T = np.zeros((3, 3)); np.add.at(T, (s[aa].astype(int).values - 1, s[bb].astype(int).values - 1), s['wtssps'].values); T /= T.sum()
                k = len(vert33(T.sum(1), T.sum(0))); vc[k] += 1; series33[f'{yr}|{a}|{b}|{w}'] = k
    out['vertices33'] = {str(k): v for k, v in sorted(vc.items())}; out['vertices33_series'] = series33
    print('3x3 vertex counts over all GSS item pairs, years and wordings:', dict(sorted(vc.items())))
    print('welfare x race, wording Y:', [(y, series33.get(f'{y}|natfare|natrace|Y')) for y in sorted(d.year.unique())])
    json.dump(out, open(os.path.join(H, 'toric_public.json'), 'w'), indent=1)
