"""Site separation on the General Social Survey, 1985-2024 (ThreeLayers: 'each invariant is non-trivial on its own layer and zero on
the others'). The same survey gives, for the eleven national-spending items asked in two randomized wordings (X, Y):

  Layer 3 (outcomes, contexts):  two items as contents, the two wordings as contexts. Contextuality-by-Default in canonical form
         (all dichotomizations) and the 'naive' contextual fraction that a test assuming identical connections would report,
         CF_naive = 1 - max{lambda : lambda q <= T_X, lambda q <= T_Y for one law q} = TV(T_X, T_Y).
  Layer 2 (probability):  within one wording, the joint law of two items, its distance from the entropy brane's intersection point
         (KL to the maximum-entropy coupling = mutual information I = delta_t H), tear margin, toric type of the coupling polytope.
  Layer 1 (operations):   the bias site of ThreeLayers: the graph K_{2,k} joining the two wordings to the k items, with the connection
         b(w, i) = logit P(too little | wording w, item i). The connection is flat (a coboundary u_w + v_i) iff every loop sum
         Phi(i, j) = [b(X,i) - b(Y,i)] - [b(X,j) - b(Y,j)] vanishes; the class lives in H^1 = R^(k-1).
  Panels:  kernels K12, K23, K13 of the welfare answer, the composite K12 K23, and whether the ordered path of answers predicts the 2008
           vote and wave-3 party identification beyond the order-free summary of the same three answers.
Outputs site_separation.json."""
import os, sys, json, itertools
import numpy as np, pandas as pd, pyreadr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import cbd_split, cbd_rank2, tear_margin, State
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, 'data', 'gss')
ITEMS = ['natfare', 'natcity', 'natrace', 'natcrime', 'natdrug', 'nateduc', 'natheal', 'natenvir', 'natspac', 'natarms', 'nataid']
YV = {'natspac': 'natspacy', 'natenvir': 'natenviy', 'natheal': 'nathealy', 'natcity': 'natcityy', 'natcrime': 'natcrimy',
      'natdrug': 'natdrugy', 'nateduc': 'nateducy', 'natrace': 'natracey', 'natarms': 'natarmsy', 'nataid': 'nataidy', 'natfare': 'natfarey'}
LABEL = {'natfare': 'welfare / the poor', 'natcity': 'big cities', 'natrace': 'Blacks', 'natcrime': 'crime / law enforcement',
         'natdrug': 'drug addiction / drug rehabilitation', 'nateduc': 'education', 'natheal': 'health', 'natenvir': 'environment',
         'natspac': 'space exploration', 'natarms': 'military / national defense', 'nataid': 'foreign aid'}
SHORT = {'natfare': 'welfare', 'natcity': 'cities', 'natrace': 'race', 'natcrime': 'crime', 'natdrug': 'drugs', 'nateduc': 'education',
         'natheal': 'health', 'natenvir': 'environment', 'natspac': 'space', 'natarms': 'defense', 'nataid': 'foreign aid'}
logit = lambda p: np.log(p / (1 - p))

def mi_bits(T):
    T = T / T.sum(); f, g = T.sum(1), T.sum(0); m = T > 0
    return float((T[m] * np.log2(T[m] / np.outer(f, g)[m])).sum())

def kernel_basis(n, m):
    """Orthonormal basis of the kernel of the marginal map on n x m tables (the circulations)."""
    M = np.vstack([np.kron(np.eye(n), np.ones(m)), np.kron(np.ones(n), np.eye(m))])
    u, s, vt = np.linalg.svd(M); r = (s > 1e-10).sum(); return vt[r:].T

def fisher_steepness(f, g):
    """Eigenvalues of the Fisher form sum_c u_c^2 / p_c on the fibre at the maximum-entropy coupling p = f g^T (unit u), the Hessian of
    -H on the fibre at the point where the entropy brane meets the conormal brane (Bridges, Prop 2.1, Thm 3.1). The smallest is positive
    (transversality); the largest, the steepness of the intersection, grows without bound as a margin -> 0, where the brane disappears."""
    p = np.outer(f, g).ravel(); B = kernel_basis(len(f), len(g))
    ev = np.linalg.eigvalsh(B.T @ np.diag(1 / p) @ B); return float(ev.min()), float(ev.max())

def facet_cells(f, g):
    return [(i, j) for i in range(len(f)) for j in range(len(g)) if f[i] + g[j] < 1]

def leximin(f, g):
    """FOOO's iterated max-min over the facet cells of the coupling polytope (Bridges, leximin coupling), in cell coordinates."""
    from scipy.optimize import linprog
    n, m = len(f), len(g); N = n * m; idx = {(i, j): i * m + j for i in range(n) for j in range(m)}
    Aeq = np.vstack([np.kron(np.eye(n), np.ones(m)), np.kron(np.ones(n), np.eye(m))]); beq = np.r_[f, g]
    fac = facet_cells(f, g); fixed = {}
    for _ in range(N):
        free = [c for c in fac if c not in fixed]
        if not free: break
        A = [];  b = []
        for c in free: row = np.zeros(N + 1); row[idx[c]] = -1; row[-1] = 1; A.append(row); b.append(0)
        Ae = np.hstack([Aeq, np.zeros((Aeq.shape[0], 1))]); be = beq.copy()
        for c, v in fixed.items():
            row = np.zeros(N + 1); row[idx[c]] = 1; Ae = np.vstack([Ae, row]); be = np.r_[be, v]
        r = linprog(np.r_[np.zeros(N), -1], A_ub=np.array(A), b_ub=b, A_eq=Ae, b_eq=be, bounds=[(0, None)] * N + [(None, None)], method='highs')
        t = r.x[-1]; newly = []
        for c in free:
            Ab = np.array(A)[:, :N]; bb = -t * np.ones(len(free)) + 1e-12
            rr = linprog(-np.eye(N)[idx[c]], A_ub=-np.eye(N)[[idx[d] for d in free]], b_ub=bb, A_eq=Ae[:, :N], b_eq=be, bounds=[(0, None)] * N, method='highs')
            if -rr.fun <= t + 1e-9: newly.append(c)
        for c in newly: fixed[c] = t
        # is the point determined?
        ext = []
        for sgn in (1, -1):
            for k in range(N):
                rr = linprog(sgn * np.eye(N)[k], A_ub=-np.eye(N)[[idx[d] for d in fac if d not in fixed]] if len(fac) > len(fixed) else None,
                             b_ub=(-t * np.ones(len(fac) - len(fixed)) + 1e-12) if len(fac) > len(fixed) else None,
                             A_eq=np.vstack([Aeq] + [np.eye(N)[idx[c]] for c in fixed]), b_eq=np.r_[beq, list(fixed.values())], bounds=[(0, None)] * N, method='highs')
                ext.append(rr.x)
        ext = np.array(ext)
        if np.ptp(ext, axis=0).max() < 1e-9: return ext.mean(0).reshape(n, m), sorted(set(fixed.values()))
    return ext.mean(0).reshape(n, m), sorted(set(fixed.values()))

def vertex_count(f, g):
    sys.path.insert(0, os.path.join(H, '..', 'bridges', 'fukaya'))
    from nef33 import vertices
    return len(vertices(list(f), list(g)))

def wtable(sub, a, b):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub['wtssps'].notna(); s = sub[m]; w = s['wtssps'].values
    T = np.zeros((3, 3)); np.add.at(T, (s[a].astype(int).values - 1, s[b].astype(int).values - 1), w)
    return T / T.sum(), (w.sum() ** 2 / (w ** 2).sum() if len(w) else 0), len(w)

if __name__ == '__main__':
    d = pd.read_csv(os.path.join(G, 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]
    years = sorted(int(y) for y in d.year.unique()); rng = np.random.default_rng(11); out = {'years': years, 'items': ITEMS, 'labels': LABEL}
    # ------------------------------------------------------------------ Layer 1: the connection on K_{2,k} and its holonomy
    L1 = {}
    for yr in years:
        sub = d[d.year == yr]; res = {'b': {}, 'n': {}}
        mats = {}
        for w, cols in (('X', ITEMS), ('Y', [YV[i] for i in ITEMS])):
            s = sub[sub[cols].isin([1, 2, 3]).any(axis=1)]
            V = s[cols].values.astype(float); V[~np.isin(V, [1, 2, 3])] = np.nan; wt = s['wtssps'].fillna(0).values
            mats[w] = (V, wt)
        def bvec(V, wt):
            out_ = []
            for k in range(len(ITEMS)):
                ok = ~np.isnan(V[:, k]); p = np.average(V[ok, k] == 1, weights=wt[ok]); out_.append(logit(p))
            return np.array(out_)
        bX, bY = bvec(*mats['X']), bvec(*mats['Y']); dlt = bX - bY
        boots = []
        for _ in range(400):
            bb = []
            for w in ('X', 'Y'):
                V, wt = mats[w]; ix = rng.integers(0, len(V), len(V)); bb.append(bvec(V[ix], wt[ix]))
            boots.append(bb[0] - bb[1])
        boots = np.array(boots)
        C = np.eye(len(ITEMS))[1:] - np.eye(len(ITEMS))[0]          # loops through welfare: Phi(i, welfare) = d_i - d_welfare
        phi = C @ dlt; S = np.cov((boots @ C.T).T)
        Q = float(phi @ np.linalg.solve(S, phi))
        Cc = np.eye(len(ITEMS)) - 1 / len(ITEMS)                     # centred class (d modulo constants)
        L1[yr] = dict(bX=bX.round(4).tolist(), bY=bY.round(4).tolist(), d=dlt.round(4).tolist(), d_se=boots.std(0).round(4).tolist(),
                      d_cov=np.cov(boots.T).round(6).tolist(),
                      phi_welfare=phi.round(4).tolist(), phi_ci=np.percentile(boots @ C.T, [2.5, 97.5], axis=0).round(4).tolist(),
                      flat_Q=round(Q, 2), flat_df=len(ITEMS) - 1,
                      class_norm=round(float(np.linalg.norm(Cc @ dlt)), 4))
    from scipy.stats import chi2
    for yr in years: L1[yr]['flat_p'] = float(chi2.sf(L1[yr]['flat_Q'], L1[yr]['flat_df']))
    # pooled over years (independent samples): inverse-variance mean of d_i
    D = np.array([L1[y]['d'] for y in years]); SE = np.array([L1[y]['d_se'] for y in years]); Wt = 1 / SE ** 2
    dpool = (D * Wt).sum(0) / Wt.sum(0); sepool = np.sqrt(1 / Wt.sum(0))
    out['L1'] = {str(k): v for k, v in L1.items()}
    out['L1_pooled'] = dict(d=dpool.round(4).tolist(), se=sepool.round(4).tolist(),
                            years_flat_rejected=int(sum(L1[y]['flat_p'] < 0.05 / len(years) for y in years)))
    print("L1: wording effect on logit 'too little' (pooled):", {SHORT[i]: round(v, 2) for i, v in zip(ITEMS, dpool)},
          "| years whose connection is not flat (Bonferroni over years):", out['L1_pooled']['years_flat_rejected'], "of", len(years))
    # ------------------------------------------------------------------ Layer 3 and Layer 2 on every pair of items, every year
    L3 = []; L2 = []
    for yr in years:
        sub = d[d.year == yr]
        for a, b in itertools.combinations(ITEMS, 2):
            TX, nX, cX = wtable(sub, a, b); TY, nY, cY = wtable(sub, YV[a], YV[b])
            if cX < 100 or cY < 100: continue
            sp = cbd_split(TX, TY); spo = cbd_split(TX, TY, ordinal=True)
            bX = np.array([[TX[0, 0], TX[0, 1:].sum()], [TX[1:, 0].sum(), TX[1:, 1:].sum()]]); bY = np.array([[TY[0, 0], TY[0, 1:].sum()], [TY[1:, 0].sum(), TY[1:, 1:].sum()]])
            r2 = cbd_rank2(bX, bY)
            L3.append(dict(year=yr, a=a, b=b, naive_cf=round(0.5 * float(np.abs(TX - TY).sum()), 4), deficit=round(sp['deficit'], 5),
                           contextual=sp['contextual'], cuts_contextual=spo['contextual'], rank2=r2['contextual'], s=round(r2['s_odd'], 4), Delta=round(r2['Delta'], 4),
                           nX=round(nX, 1), nY=round(nY, 1)))
            for w, T, ne in (('X', TX, nX), ('Y', TY, nY)):
                f, g = T.sum(1), T.sum(0); mi = mi_bits(T)
                L2.append(dict(year=yr, a=a, b=b, w=w, mi=round(mi, 5), G=round(2 * ne * np.log(2) * mi, 2), mT=round(tear_margin(f, g), 4),
                               steep_min=round(fisher_steepness(f, g)[0], 2), steep=round(fisher_steepness(f, g)[1], 2), min_fg=round(float(np.outer(f, g).min()), 5), n_eff=round(ne, 1)))
    out['L3'] = L3; out['L2'] = L2
    nctx = sum(r['contextual'] for r in L3); ncut = sum(r['cuts_contextual'] for r in L3)
    print(f"L3: {len(L3)} system-years; naive CF median {np.median([r['naive_cf'] for r in L3]):.3f} (min {min(r['naive_cf'] for r in L3):.3f});"
          f" canonical CbD contextual at the point estimate in {nctx}, with order-respecting cuts in {ncut}, rank 2 in {sum(r['rank2'] for r in L3)}")
    # posterior screen for the canonical flags
    flagged = [r for r in L3 if r['contextual']]; post = []
    for r in flagged:
        sub = d[d.year == r['year']]; TX, nX, _ = wtable(sub, r['a'], r['b']); TY, nY, _ = wtable(sub, YV[r['a']], YV[r['b']])
        h = sum(cbd_split(rng.dirichlet(nX * TX.ravel() + 0.5).reshape(3, 3), rng.dirichlet(nY * TY.ravel() + 0.5).reshape(3, 3))['contextual'] for _ in range(400))
        r['posterior'] = round(h / 400, 4); post.append(r['posterior'])
    out['L3_screen'] = dict(n=len(L3), flagged=len(flagged), max_posterior=max(post) if post else 0, over95=sum(p > 0.95 for p in post),
                            threshold=1 - 0.05 / len(L3))
    print("   posterior screen: max", out['L3_screen']['max_posterior'], "| above 0.95:", out['L3_screen']['over95'], "| screen threshold", round(out['L3_screen']['threshold'], 6))
    mis = np.array([r['mi'] for r in L2]); Gs = np.array([r['G'] for r in L2])
    print(f"L2: {len(L2)} tables; MI median {np.median(mis):.4f} bits; independence rejected (G > chi2_4 at 0.05/{len(L2)}) in {(chi2.sf(Gs, 4) < 0.05 / len(L2)).sum()}")
    # ------------------------------------------------------------------ dissociations: the three invariants on the same systems
    dis = []
    for r in L3:
        yr = r['year']; i, j = ITEMS.index(r['a']), ITEMS.index(r['b'])
        dd = np.array(L1[yr]['d']); Cv = np.array(L1[yr]['d_cov'])
        phi = dd[i] - dd[j]; z = phi / np.sqrt(Cv[i, i] + Cv[j, j] - 2 * Cv[i, j])     # bootstrap variance of the loop, with the covariance
        mi = [x['mi'] for x in L2 if x['year'] == yr and x['a'] == r['a'] and x['b'] == r['b']]
        Gx = [x['G'] for x in L2 if x['year'] == yr and x['a'] == r['a'] and x['b'] == r['b']]
        dis.append(dict(year=yr, a=r['a'], b=r['b'], L3_ctx=r['contextual'], naive_cf=r['naive_cf'], phi=round(float(phi), 4), phi_z=round(float(z), 2),
                        mi_X=mi[0], mi_Y=mi[1], dep=bool(min(chi2.sf(Gx[0], 4), chi2.sf(Gx[1], 4)) < 0.05 / (2 * len(L3)))))
    out['dissociation'] = dis
    from scipy.stats import norm
    zc = float(norm.isf(0.025 / len(dis))); out['z_bonferroni'] = zc
    ok = [x for x in dis if not x['L3_ctx'] and abs(x['phi_z']) > zc and x['dep']]
    ok2 = [x for x in dis if abs(x['phi_z']) > zc and x['dep']]
    out['dissociation_counts'] = dict(L1_and_L2=len(ok2), L1_and_L2_not_flagged=len(ok), L1=sum(abs(x['phi_z']) > zc for x in dis), L2=sum(x['dep'] for x in dis))
    print(f"dissociation (Bonferroni over {len(dis)}: |z| > {zc:.2f}, G-test at 0.05/{2*len(dis)}):", out['dissociation_counts'])
    from scipy.stats import spearmanr
    rc = spearmanr([x['naive_cf'] for x in dis], [abs(x['phi']) for x in dis]).correlation
    rm = spearmanr([abs(x['phi']) for x in dis], [x['mi_X'] for x in dis]).correlation
    rn = spearmanr([x['naive_cf'] for x in dis], [x['mi_X'] for x in dis]).correlation
    out['rank_corr'] = dict(naiveCF_absPhi=round(float(rc), 3), absPhi_MI=round(float(rm), 3), naiveCF_MI=round(float(rn), 3))
    print("rank correlations across system-years:", out['rank_corr'])
    json.dump(out, open(os.path.join(H, 'site_separation.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

def boundary_null_p(TX, TY, nX, nY, obs, rng, B=1000, deff=1.0):
    """Parametric test of canonical contextuality against a noncontextual truth on the boundary: move the Y table toward the X table,
    T_Y(l) = (1 - l) T_Y + l T_X, to the first l at which the deficit vanishes (bisection), simulate both tables at that truth with the
    effective sample sizes divided by the design effect, and return the share of simulated deficits at least the observed one."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if cbd_split(TX, (1 - mid) * TY + mid * TX)['deficit'] > 1e-9: lo = mid
        else: hi = mid
    TYb = (1 - hi) * TY + hi * TX; mX, mY = int(round(nX / deff)), int(round(nY / deff)); ge = 0
    for _ in range(B):
        ge += cbd_split(rng.multinomial(mX, TX.ravel()).reshape(3, 3) + 1e-12, rng.multinomial(mY, TYb.ravel()).reshape(3, 3) + 1e-12)['deficit'] >= obs - 1e-12
    return (ge + 1) / (B + 1), hi
