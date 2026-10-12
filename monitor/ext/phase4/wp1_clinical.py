"""WP1a (PREANALYSIS_PHASE4.md): do toric type, chamber transitions and entropy-brane steepness of per-stay MAP x HR systems forecast death
(and shock onset) beyond the margins?  Usage: python3 wp1_clinical.py synth | run   -> results/wp1_clinical.json"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import geom
H = os.path.dirname(os.path.abspath(__file__)); CACHE = os.environ.get('PHYSIONET_DIR', '/home/claude/phase2/data/challenge2012') + '/cache'
SEED = 20261010

def systems(df):
    T = {w: df[[f'W{w}_{i}{j}' for i in range(3) for j in range(3)]].values.reshape(-1, 3, 3).astype(float) for w in (1, 2)}
    return {w: (T[w].sum(2) / T[w].sum((1, 2))[:, None], T[w].sum(1) / T[w].sum((1, 2))[:, None]) for w in (1, 2)}

def features(df, cap=None):
    S = systems(df); rows = []
    for k in range(len(df)):
        f1, g1, f2, g2 = S[1][0][k], S[1][1][k], S[2][0][k], S[2][1][k]
        v1, v2 = geom.n_vertices(f1, g1), geom.n_vertices(f2, g2); st = geom.steepness(f2, g2)
        rows.append(dict(f1_0=f1[0], f1_1=f1[1], g1_0=g1[0], g1_1=g1[1], f2_0=f2[0], f2_1=f2[1], g2_0=g2[0], g2_1=g2[1],
                         v1=v1, v2=v2, dv=v2 - v1, flips=int((geom.signs(f1, g1) != geom.signs(f2, g2)).sum()), tm2=geom.tear_margin(f2, g2),
                         br1=geom.breaks(f1, g1), br2=geom.breaks(f2, g2), steep2=np.log(st) if np.isfinite(st) else np.nan,
                         v1_full=geom.n_vertices(f1, g1), steep1=np.log(geom.steepness(f1, g1)) if np.isfinite(geom.steepness(f1, g1)) else np.nan))
    F = pd.DataFrame(rows, index=df.index)
    for w in (1, 2):
        for a in ('f', 'g'):
            for c in (0, 1): F[f'd_{a}{c}'] = F[f'{a}2_{c}'] - F[f'{a}1_{c}']
    cap = cap if cap is not None else {c: float(np.nanmax(F[c])) for c in ('steep1', 'steep2')}
    for c in ('steep1', 'steep2'): F[c] = F[c].fillna(cap[c])
    return F, cap

MARG = ['f1_0', 'f1_1', 'g1_0', 'g1_1', 'f2_0', 'f2_1', 'g2_0', 'g2_1', 'd_f0', 'd_f1', 'd_g0', 'd_g1']
GEO = ['v1', 'v2', 'dv', 'flips', 'tm2', 'br1', 'br2', 'steep2']
MARG1 = ['f1_0', 'f1_1', 'g1_0', 'g1_1']; GEO1 = ['v1', 'br1', 'steep1']          # W1 only, for shock onset
def design(F, marg, geo, flexible):
    M = F[marg].values
    if flexible: M = PolynomialFeatures(2, include_bias=False).fit_transform(M)
    if geo:
        G = F[geo].values; vcols = [c for c in geo if c in ('v1', 'v2')]
        dums = [np.stack([(F[c].values == v).astype(float) for v in (6, 9, 12, 15, 18)], 1) for c in vcols]   # toric type as categories
        M = np.hstack([M, G] + dums)
    return M
def logloss(p, y): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def compare(Ftr, ytr, Fte, yte, marg, geo, rng):
    ytr, yte = np.asarray(ytr, int), np.asarray(yte, int); out = {}
    preds = {}
    for name, flex, g in (('B1', False, []), ('B2', True, []), ('S1', False, geo), ('S2', True, geo)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(design(Ftr, marg, g, flex), ytr)
        preds[name] = m.predict_proba(design(Fte, marg, g, flex))[:, 1]; out[f'logloss_{name}'] = round(float(logloss(preds[name], yte).mean()), 5)
    idx = rng.integers(0, len(yte), (2000, len(yte)))
    for a, b in (('B2', 'S2'), ('B1', 'S1')):
        d = (logloss(preds[a], yte)[idx] - logloss(preds[b], yte)[idx]).mean(1)
        out[f'delta_{a}_{b}'] = round(float(d.mean()), 6); out[f'ci_{a}_{b}'] = [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)]
    out['geometry_forecasts_beyond_margins'] = bool(out['ci_B2_S2'][0] > 0); out['n_train'], out['n_test'], out['events_test'] = len(ytr), len(yte), int(yte.sum())
    return out

def keep(df): return df[(df.W1_n >= 12) & (df.W2_n >= 12)].copy()
def run():
    A, B = keep(pd.read_csv(f'{CACHE}/phase4_set_a.csv')), keep(pd.read_csv(f'{CACHE}/phase4_set_b.csv'))
    FA, cap = features(A); FB, _ = features(B, cap); rng = np.random.default_rng(SEED)
    res = dict(plan='PREANALYSIS_PHASE4.md WP1a', primary_death=compare(FA, A.death.values, FB, B.death.values, MARG, GEO, rng))
    shockA, shockB = (A.filter(like='W1_0').sum(1) == 0), (B.filter(like='W1_0').sum(1) == 0)
    yA, yB = (A.filter(like='W2_0').sum(1) > 0).astype(int), (B.filter(like='W2_0').sum(1) > 0).astype(int)
    res['secondary_shock_onset'] = compare(FA[shockA], yA[shockA].values, FB[shockB], yB[shockB].values, MARG1, GEO1, rng)
    res['descriptive'] = dict(vertex_counts_W2_setA={int(k): int(v) for k, v in FA.v2.value_counts().sort_index().items()},
                              broken_share_W2_setA=round(float((FA.br2 > 0).mean()), 3), steep_cap=cap,
                              death_by_dv_setA={str(k): [int(v), round(float(A.death[FA.dv == k].mean()), 3)] for k, v in FA.dv.value_counts().sort_index().items() if v >= 10})
    os.makedirs(os.path.join(H, 'results'), exist_ok=True); json.dump(res, open(os.path.join(H, 'results', 'wp1_clinical.json'), 'w'), indent=1); print(json.dumps(res, indent=1))

def synth():
    """planted: (a) death depends on the W2 toric type and on chamber flips, not smoothly on the margins -> S2 beats B2;
               (b) death depends linearly on the margins only -> S2 does not beat B2"""
    rng = np.random.default_rng(5); ok = True; out = {}
    def make(n):
        rows = []
        for _ in range(n):
            r = {}
            for w in (1, 2):
                T = rng.multinomial(22, rng.dirichlet(np.ones(9) * 0.7)).reshape(3, 3)
                for i in range(3):
                    for j in range(3): r[f'W{w}_{i}{j}'] = int(T[i, j])
                r[f'W{w}_n'] = 22
            rows.append(r)
        return pd.DataFrame(rows)
    for kind in ('geometric', 'marginal'):
        tr, te = make(3500), make(3500); Ftr, cap = features(tr); Fte, _ = features(te, cap)
        def y(F):
            z = -2.0 + (1.6 * (F.v2 == 18) - 0.8 * (F.v2 <= 9) + 0.35 * F.flips if kind == 'geometric' else 2.5 * F.f2_0 - 1.5 * F.g2_1)
            return (rng.random(len(F)) < 1 / (1 + np.exp(-z))).astype(int)
        r = compare(Ftr, y(Ftr), Fte, y(Fte), MARG, GEO, np.random.default_rng(1)); out[kind] = r
        ok &= r['geometry_forecasts_beyond_margins'] == (kind == 'geometric')
        print(kind, {k: r[k] for k in ('logloss_B2', 'logloss_S2', 'delta_B2_S2', 'ci_B2_S2', 'geometry_forecasts_beyond_margins')})
    json.dump(out, open(os.path.join(H, 'results', 'wp1_clinical_synth.json'), 'w'), indent=1); print('PASS' if ok else 'FAIL'); return ok

if __name__ == '__main__':
    os.makedirs(os.path.join(H, 'results'), exist_ok=True)
    {'synth': synth, 'run': run}[sys.argv[1]]()
