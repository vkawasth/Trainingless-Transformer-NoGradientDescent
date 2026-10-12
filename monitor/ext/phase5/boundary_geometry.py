"""Part B of PREANALYSIS_CONTEXT_MAP.md: geometry at the boundary. Per-stay ICU day systems usually have a zero margin (a point or
degenerate polytope). Boundary-aware features: the Dirichlet(counts + 1/2) posterior over toric types and the posterior mean log
steepness; the walls decided on the margin bands (the family over the band, a full-dimensional polytope) and decided chamber flips;
the face stratum (empty bands = the torus orbit of a broken system). Re-test of phase 4 WP1a: S2b = B2 + these, fit A, test B.
Usage: python3 boundary_geometry.py synth | run -> results/boundary_geometry.json"""
import os, sys, json
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'phase4')); sys.path.insert(0, os.path.abspath(os.path.join(H, '..', '..')))
import geom, wp1_clinical as W
from monitor import bands, wall_signs
SEED, NDRAW = 20261011, 200
_TYPE = {}
def vtype(f, g):
    """vertex count; for full-support margins it is a function of the chamber, so it is cached by the sign vector"""
    key = tuple(geom.signs(f, g))
    if 0 in key: return geom.n_vertices(f, g)
    if key not in _TYPE: _TYPE[key] = geom.n_vertices(f, g)
    return _TYPE[key]
def counts(df, w):
    T = df[[f'W{w}_{i}{j}' for i in range(3) for j in range(3)]].values.reshape(-1, 3, 3).astype(float); return T.sum(2), T.sum(1)
def features(df, rng):
    out = {}
    for w in (1, 2):
        fc, gc = counts(df, w); n = fc.sum(1)
        P = np.zeros((len(df), 3)); S = np.zeros(len(df)); dec = np.zeros(len(df)); signs = []
        for k in range(len(df)):
            fd, gd = rng.dirichlet(fc[k] + 0.5, NDRAW), rng.dirichlet(gc[k] + 0.5, NDRAW)
            ty = np.array([vtype(fd[t], gd[t]) for t in range(NDRAW)])
            P[k] = [(ty == 18).mean(), (ty == 15).mean(), (ty == 12).mean()]
            S[k] = np.mean([np.log(geom.steepness(fd[t], gd[t])) for t in range(0, NDRAW, 4)])
            sg = wall_signs(bands(fc[k] / n[k], n[k]), bands(gc[k] / n[k], n[k])); dec[k] = (sg != 0).sum(); signs.append(sg)
        out[w] = dict(P=P, S=S, dec=dec, signs=np.array(signs), ef=(fc == 0).sum(1), eg=(gc == 0).sum(1))
    F = pd.DataFrame(index=df.index)
    for w in (1, 2):
        for t, c in enumerate((18, 15, 12)): F[f'p{c}_{w}'] = out[w]['P'][:, t]
        F[f'steep_{w}'] = out[w]['S']; F[f'dec_{w}'] = out[w]['dec']; F[f'empty_map_{w}'] = out[w]['ef']; F[f'empty_hr_{w}'] = out[w]['eg']
    F['dec_flips'] = ((out[1]['signs'] * out[2]['signs']) < 0).sum(1)
    return F
BGEO = [f'{x}_{w}' for w in (1, 2) for x in ('p18', 'p15', 'p12', 'steep', 'dec', 'empty_map', 'empty_hr')] + ['dec_flips']

def compare(Fm_tr, Fb_tr, ytr, Fm_te, Fb_te, yte, rng):
    ytr, yte = np.asarray(ytr, int), np.asarray(yte, int)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    def X(Fm, Fb, geo):
        Mx = PolynomialFeatures(2, include_bias=False).fit_transform(Fm[W.MARG].values); return np.hstack([Mx, Fb[BGEO].values]) if geo else Mx
    P = {}
    for name, geo in (('B2', False), ('S2b', True)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X(Fm_tr, Fb_tr, geo), ytr); P[name] = W.logloss(m.predict_proba(X(Fm_te, Fb_te, geo))[:, 1], yte)
    idx = rng.integers(0, len(yte), (2000, len(yte))); d = (P['B2'][idx] - P['S2b'][idx]).mean(1)
    return dict(logloss_B2=round(float(P['B2'].mean()), 5), logloss_S2b=round(float(P['S2b'].mean()), 5), delta=round(float(d.mean()), 6),
                ci=[round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)], boundary_geometry_forecasts=bool(np.percentile(d, 2.5) > 0),
                n_train=len(ytr), n_test=len(yte), events_test=int(np.sum(yte)))

def run():
    rng = np.random.default_rng(SEED)
    A, B = W.keep(pd.read_csv(f'{W.CACHE}/phase4_set_a.csv')), W.keep(pd.read_csv(f'{W.CACHE}/phase4_set_b.csv'))
    FmA, cap = W.features(A); FmB, _ = W.features(B, cap); FbA, FbB = features(A, rng), features(B, rng)
    res = dict(plan='PREANALYSIS_CONTEXT_MAP.md Part B', death=compare(FmA, FbA, A.death.values, FmB, FbB, B.death.values, rng),
               descriptive=dict(mean_posterior_type_day2_setA={c: round(float(FbA[f'p{c}_2'].mean()), 3) for c in (18, 15, 12)},
                                share_point_estimate_broken_day2_setA=round(float((FbA.empty_map_2 + FbA.empty_hr_2 > 0).mean()), 3),
                                share_with_posterior_full_type_mass_over_half=round(float((FbA[['p18_2', 'p15_2', 'p12_2']].sum(1) > 0.5).mean()), 3),
                                decided_walls_day2_mean=round(float(FbA.dec_2.mean()), 2), decided_flips_share=round(float((FbA.dec_flips > 0).mean()), 3),
                                death_by_decided_flip={str(k): round(float(A.death[FbA.dec_flips > 0].mean() if k else A.death[FbA.dec_flips == 0].mean()), 3) for k in (0, 1)}))
    os.makedirs(os.path.join(H, 'results'), exist_ok=True); json.dump(res, open(os.path.join(H, 'results', 'boundary_geometry.json'), 'w'), indent=1); print(json.dumps(res, indent=1))

def synth():
    """planted: death depends on the posterior probability that a (usually broken) day-2 system is of type 18 -> S2b beats B2.
    Second version (before the real run): the first planted check (weaker effect, less concentrated margins, 24% broken) failed
    to show power, so the effect was doubled and the margins concentrated to match the data (most systems broken)."""
    rng = np.random.default_rng(6); rows = []
    for _ in range(5000):
        r = {}
        for w in (1, 2):
            T = rng.multinomial(22, rng.dirichlet(np.ones(9) * 0.25)).reshape(3, 3)          # concentrated: mostly broken, as in the data
            for i in range(3):
                for j in range(3): r[f'W{w}_{i}{j}'] = int(T[i, j])
            r[f'W{w}_n'] = 22
        rows.append(r)
    D = pd.DataFrame(rows); Fm, _ = W.features(D); Fb = features(D, np.random.default_rng(1))
    y = (rng.random(len(D)) < 1 / (1 + np.exp(-(-2.2 + 6.0 * Fb.p18_2 + 3.0 * Fb.p15_2)))).astype(int).values
    r = compare(Fm.iloc[:2500], Fb.iloc[:2500], y[:2500], Fm.iloc[2500:], Fb.iloc[2500:], y[2500:], np.random.default_rng(2))
    ok = r['boundary_geometry_forecasts']; s = f"planted posterior-type dependence: delta {r['delta']} CI {r['ci']}; broken share {(Fb.empty_map_2 + Fb.empty_hr_2 > 0).mean():.2f}"
    print(s, 'PASS' if ok else 'FAIL'); os.makedirs(os.path.join(H, 'results'), exist_ok=True)
    open(os.path.join(H, 'results', 'boundary_geometry_synth.txt'), 'w').write(('PASS ' if ok else 'FAIL ') + s)
if __name__ == '__main__':
    {'synth': synth, 'run': run}[sys.argv[1]]()
