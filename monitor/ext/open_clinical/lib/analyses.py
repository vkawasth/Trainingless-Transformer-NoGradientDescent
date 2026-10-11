"""Analyses A-E of the pre-analysis plan (PREANALYSIS_PLAN.md and PREANALYSIS_AMENDMENT_OPEN_DATA.md), written once for every
dataset. Each function takes arrays of small integer codes and returns aggregate results only."""
import itertools
import numpy as np, pandas as pd
from scipy.stats import chi2
from monitor import State, regime, cbd_split, churn, gluing_gap, bootstrap
from panel_models import fit as lc_fit, G2 as lc_G2, gap as lc_gap, DF as LC_DF
from panel_paths import perm_test, loo_bits
from mimic_monitor import cf_triangle, ipf3, boundary_null_p

def table(a, b, ka, kb, w=None):
    T = np.zeros((ka, kb)); np.add.at(T, (np.asarray(a, int), np.asarray(b, int)), 1 if w is None else w); return T

def shown(T, mincell=10):
    return {'proportions': (T / T.sum()).round(4).tolist(), 'n': int(round(T.sum()))} if T.min() >= mincell else {'suppressed': f'a cell below {mincell}', 'n': int(round(T.sum()))}

def toric_2x3(T):
    """Toric surface of a 2 x 3 coupling polygon from its margins (Bridges, Prop. 8.1), and whether the chamber is decided."""
    f, g = T.sum(1) / T.sum(), T.sum(0) / T.sum()
    fac = [(i, j) for i in range(2) for j in range(3) if f[i] + g[j] < 1]
    cols = [j for j in range(3) if (0, j) in fac and (1, j) in fac]
    name = {3: 'CP^2', 5: 'Bl_2 CP^2', 6: 'Bl_3 CP^2'}.get(len(fac), 'CP^1 x CP^1' if len(fac) == 4 and len(cols) == 2 else 'F_1')
    return dict(facets=[f'{i+1}{j+1}' for i, j in fac], surface=name, decided=bool((State(T + 1e-9, T.sum()).signs != 0).all()))

def geometry(ctx, kx, ky, pairs='all', mincell=10, binary_first=None):
    """ctx: {name: (x, y)}. Regimes between contexts (all pairs, or consecutive), mutual information, tables; optional toric type of
    the 2 x 3 system (x collapsed to binary_first(x))."""
    S, out = {}, {'tables': {}, 'steps': [], 'mutual_information_bits': {}}
    for k, (x, y) in ctx.items():
        T = table(x, y, kx, ky); out['tables'][k] = shown(T, mincell)
        if T.sum() >= 100: S[k] = State(T + 1e-9, T.sum()); out['mutual_information_bits'][k] = round(S[k].MI, 4)
    names = list(S); prs = list(itertools.combinations(names, 2)) if pairs == 'all' else list(zip(names, names[1:]))
    for a, b in prs:
        r = regime(S[a], S[b])
        out['steps'].append(dict(frm=a, to=b, kind=r['kind'], stress=round(r['stress'], 2), tear_margin=round(r['mT'], 4),
                                 walls_crossed=[[list(I), list(J)] for I, J in r['walls_crossed']]))
    if binary_first is not None:
        out['toric_2x3'] = {k: toric_2x3(table(binary_first(np.asarray(x)), y, 2, 3)) for k, (x, y) in ctx.items() if len(x) >= 100}
    return out

def composition(x1, x2, x3, rng, B=400, starts=30):
    x1, x2, x3 = (np.asarray(v, int) for v in (x1, x2, x3))
    gg = gluing_gap(x1, x2, x3); ci = bootstrap(gluing_gap, [x1, x2, x3], B=B, seed=1)
    cum = lambda a, b: np.cumsum(pd.crosstab(a, b, normalize='index').reindex(index=range(3), columns=range(3), fill_value=0).values, 1)
    K12c, K23c = cum(x1, x2), cum(x2, x3); step = lambda Kc, v: np.minimum((rng.random(len(v))[:, None] > Kc[v]).sum(1), 2)
    nulls = []
    for _ in range(B):
        a = rng.choice(x1, len(x1)); b = step(K12c, a); c = step(K23c, b); nulls.append(gluing_gap(a, b, c))
    N3 = np.zeros((3, 3, 3)); np.add.at(N3, (x1, x2, x3), 1); models = {}
    for m in ('stable', 'drifting', 'markov'):
        th, P = lc_fit(N3, m, starts=starts); g2 = lc_G2(N3, P)
        models[m] = dict(G2=round(g2, 2), df=LC_DF[m], p=float(chi2.sf(g2, LC_DF[m])), gap_implied=round(lc_gap(P), 4), min_class=round(float(min(th[0])), 3))
    K = lambda a, b: pd.crosstab(a, b, normalize='index').reindex(index=range(3), columns=range(3), fill_value=0).values
    return dict(n=len(x1), gap_bits=round(gg, 4), gap_ci=[round(v, 4) for v in ci], markov_null95=round(float(np.percentile(nulls, 95)), 4),
                p_markov=float((np.sum(np.array(nulls) >= gg) + 1) / (B + 1)), latent_models=models,
                tv_K13_vs_K12K23=(0.5 * np.abs(K(x1, x3) - K(x1, x2) @ K(x2, x3)).sum(1)).round(4).tolist(),
                churn_12={k: round(v, 4) for k, v in churn(x1, x2, [0, 1, 2]).items() if k in ('agree', 'max_agree', 'excess')},
                decision=dict(answers_not_glued=bool(gg > np.percentile(nulls, 95)), latent_markov_fits=bool(models['markov']['p'] > 0.05)))

def path_outcome(x1, x2, x3, y, rng, B=2000):
    x1, x2, x3, y = (np.asarray(v, int) for v in (x1, x2, x3, y)); K = int(y.max()) + 1
    path = x1 * 9 + x2 * 3 + x3; ms = np.array([''.join(map(str, sorted(t))) for t in zip(x1, x2, x3)])
    g1, p1 = perm_test(y, ms, x1, K, rng, B=B); g2, p2 = perm_test(y, path, ms, K, rng, B=B)
    base = loo_bits(y, np.zeros(len(y), int), K)
    return dict(n=len(y), outcome_rate=round(float(y.mean()), 4) if K == 2 else None, bits_gained_x1=round(base - loo_bits(y, x1, K), 4),
                bits_gained_multiset=round(base - loo_bits(y, ms, K), 4), bits_gained_path=round(base - loo_bits(y, path, K), 4),
                p_multiset_vs_x1=p1, p_order_vs_multiset=p2)

def cbd_strata(strata, shape, rng, B=400, mincell=10, min_n=0):
    """strata: {name: (TX, TY)} count tables of the two contexts. Canonical CbD, naive CF, boundary-null p, Fisher over strata."""
    res, ps = {}, []
    for k, (TX, TY) in strata.items():
        if min(TX.sum(), TY.sum()) <= max(min_n, 0) or TX.sum() == 0 or TY.sum() == 0: res[k] = 'too few'; continue
        sp = cbd_split(TX / TX.sum(), TY / TY.sum()); naive = 0.5 * float(np.abs(TX / TX.sum() - TY / TY.sum()).sum())
        p = boundary_null_p(TX / TX.sum(), TY / TY.sum(), TX.sum(), TY.sum(), sp['deficit'], rng, B) if sp['deficit'] > 0 else 1.0
        ps.append(p); res[k] = dict(naive_cf=round(naive, 4), deficit=round(sp['deficit'], 5), contextual_point=sp['contextual'], p_boundary=round(float(p), 4),
                                    nX=int(TX.sum()), nY=int(TY.sum()), X=shown(TX, mincell), Y=shown(TY, mincell))
    X2 = float(-2 * np.log(ps).sum()) if ps else 0.0; p = float(chi2.sf(X2, 2 * len(ps))) if ps else None
    return dict(strata=res, fisher=round(X2, 3), df=2 * len(ps), p=p, contextual=bool(p is not None and p < 0.05))

def selection(x, y, z, rng, B=400, extra_weights=None):
    """x, y observed for all; z observed where not nan. Contextual fraction of (x,y), (x,z), (y,z): available case, after IPW on the
    (x, y) cells, optionally after a second set of weights; sampling null from the IPF joint of the IPW tables."""
    x, y, z = np.asarray(x, int), np.asarray(y, int), np.asarray(z, float); m = ~np.isnan(z)
    Txy = table(x, y, 3, 3); Txz = table(x[m], z[m].astype(int), 3, 3); Tyz = table(y[m], z[m].astype(int), 3, 3)
    pm = pd.Series(m).groupby([x, y]).mean(); w = 1 / pm.loc[list(zip(x[m], y[m]))].values
    Wxz = table(x[m], z[m].astype(int), 3, 3, w); Wyz = table(y[m], z[m].astype(int), 3, 3, w)
    cf_av, cf_ipw = cf_triangle(Txy, Txz, Tyz), cf_triangle(Txy, Wxz, Wyz)
    P = ipf3(Txy, Wxz, Wyz); null = []; nxy, nm = int(Txy.sum()), int(m.sum())
    for _ in range(B):
        null.append(cf_triangle(rng.multinomial(nxy, P.sum(2).ravel()).reshape(3, 3) + 1e-12, rng.multinomial(nm, P.sum(1).ravel()).reshape(3, 3) + 1e-12,
                                rng.multinomial(nm, P.sum(0).ravel()).reshape(3, 3) + 1e-12))
    q95 = float(np.percentile(null, 95))
    out = dict(n_all=nxy, n_measured=nm, measured_share=round(nm / nxy, 4), cf_available_case=round(cf_av, 4), cf_after_ipw=round(cf_ipw, 4),
               null_median=round(float(np.median(null)), 4), null_95=round(q95, 4),
               p_available=float((np.sum(np.array(null) >= cf_av) + 1) / (B + 1)), p_ipw=float((np.sum(np.array(null) >= cf_ipw) + 1) / (B + 1)))
    if extra_weights is not None:
        w2 = np.asarray(extra_weights)[m]
        out['cf_after_secondary_weights'] = round(cf_triangle(Txy, table(x[m], z[m].astype(int), 3, 3, w2), table(y[m], z[m].astype(int), 3, 3, w2)), 4)
    out['decision'] = ('selection explains the incompatibility' if cf_av > q95 and cf_ipw <= q95 else
                       'a residual remains after weighting' if cf_av > q95 else 'no incompatibility beyond sampling')
    return out
