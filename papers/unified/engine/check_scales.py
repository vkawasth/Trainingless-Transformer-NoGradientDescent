"""Checks for the scale theorem (thm:scales, section on one scale per removed dimension): at the positive object the scale valuations of
Q/W are h_j - h_1 with multiplicity r_j, for every shape and nondegenerate margins. New script; it modifies nothing.
Proof idea checked here: Q = sum_k w_k v_k v_k^T with w_k > 0, so by Cauchy-Binet e_k(Q) = sum over k-sets S of prod_{s in S} w_s det(V_S^T V_S),
a sum of nonnegative terms (no cancellation); its valuation is the least weight of an independent k-set, which the matroid greedy
algorithm attains by taking, level by level, as many new independent normals as the level adds to the span.
  SV1 IDENTITY  Cauchy-Binet: e_k(V W V^T) = sum_S prod w_S det(V_S^T V_S) (random V, W > 0)
  SV2 IDENTITY  matroid greedy: the least weight of an independent k-set equals the sum of the k smallest greedy weights (random vectors
                with repeated integer directions and tied weights; brute force)
  SV3 IDENTITY  on random nondegenerate margins of shapes 2x3, 3x3, 2x4 and 3x4, with the facet-leximin levels h_j and their sets S_j:
                the least weights of independent k-sets of facet normals, weighted by the cell values at the leximin point, are exactly
                h_j repeated r_j = rank(S_1..S_j) - rank(S_1..S_j-1) times
  SV4 IDENTITY  finite L: at the minimiser of F_L(u) = sum_k e^(-L x_k(u)) (the positive critical point), the eigenvalues mu of
                Hess F_L / (L^2 F_L) satisfy -log(mu) / L -> h_j - h_1 with multiplicity r_j (250-digit arithmetic; L = 60, 120, 240,
                linear extrapolation in 1/L; tolerance 0.01). First run crashed before any SV4 result: at 80 digits and L = 400 mpmath's
                LU solver declared the Newton Hessian singular (its entries span more than 80 orders); now 250 digits and L <= 240
"""
import os, sys, math, itertools
import numpy as np
import mpmath as mp
from scipy.optimize import linprog
from scipy.linalg import null_space
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261117)
def esym(M, k):
    n = len(M); return sum(np.linalg.det(M[np.ix_(I, I)]) for I in itertools.combinations(range(n), k))
worst = 0.0
for _ in range(200):
    d, m = int(rng.integers(2, 5)), int(rng.integers(3, 8)); V = rng.normal(size=(d, m)); w = rng.uniform(0.1, 3, size=m); Q = V @ np.diag(w) @ V.T
    for k in range(1, d + 1):
        cb = sum(np.prod(w[list(S)]) * np.linalg.det(V[:, S].T @ V[:, S]) for S in map(list, itertools.combinations(range(m), k)))
        worst = max(worst, abs(esym(Q, k) - cb) / max(1, abs(cb)))
check("SV1 IDENTITY Cauchy-Binet: e_k(V W V^T) is a sum of prod w_S det(V_S^T V_S) >= 0", worst < 1e-9, f"200 cases; max relative error {worst:.1e}")
def rank(vs): return int(np.linalg.matrix_rank(np.array(vs, float))) if len(vs) else 0
def greedy(vs, ws):
    order = sorted(range(len(vs)), key=lambda i: ws[i]); chosen = []; out = []
    for i in order:
        if rank([vs[j] for j in chosen] + [vs[i]]) > len(chosen): chosen.append(i); out.append(ws[i])
    return out
bad = 0
for _ in range(300):
    d = int(rng.integers(2, 4)); base = [rng.integers(-2, 3, size=d) for _ in range(int(rng.integers(2, 5)))]
    vs = [base[int(rng.integers(len(base)))] * int(rng.choice([1, 1, -1, 2])) for _ in range(int(rng.integers(3, 8)))]
    vs = [v for v in vs if np.any(v)]
    if not vs: continue
    ws = [float(rng.integers(0, 4)) for _ in vs]; g = greedy(vs, ws)
    for k in range(1, len(g) + 1):
        best = min((sum(ws[i] for i in S) for S in itertools.combinations(range(len(vs)), k) if rank([vs[i] for i in S]) == k), default=None)
        bad += best is None or abs(best - sum(g[:k])) > 1e-12
check("SV2 IDENTITY the least weight of an independent k-set is the sum of the k smallest greedy weights", bad == 0, f"300 configurations; mismatches {bad}")
from pce.couplinglp import margins_A, facets, leximin_levels
def predicted(levels, Vrows):
    out = []; prev = []
    for h, Sj in levels:
        r0 = rank(prev); prev = prev + [Vrows[k] for k in Sj]; out += [h] * (rank(prev) - r0)
    return out
shapes = [(2, 3), (3, 3), (2, 4), (3, 4)]; bad = 0; cases = 0; multi = 0; cases_store = []
for it in range(160):
    n, m = shapes[it % 4]; conc = float(rng.choice([0.5, 1, 3]))
    f = rng.dirichlet([conc] * n); g = rng.dirichlet([conc] * m)
    if min(f.min(), g.min()) < 1e-3: continue
    Fc = facets(f, g); Kb = null_space(margins_A(n, m)); x, levels = leximin_levels(f, g, Fc)
    V = {k: Kb[k] for k in Fc}; pred = predicted(levels, V); ws = [x[k] for k in Fc]; vs = [V[k] for k in Fc]
    got = greedy(vs, ws); cases += 1; multi += len(levels) > 1 and any(c > 1 for c in [pred.count(h) for h in set(pred)])
    bad += len(got) != len(pred) or any(abs(a - b) > 1e-7 for a, b in zip(sorted(got), sorted(pred)))
    if len(cases_store) < 40: cases_store.append((n, m, f, g, Fc, Kb, pred))
check("SV3 IDENTITY least-weight independent sets of facet normals at the leximin point give h_j with multiplicity r_j", bad == 0 and cases > 100,
      f"{cases} margins over 4 shapes; mismatches {bad}; cases with a multiplicity > 1: {multi}")
mp.mp.dps = 250
def scales_at(L, Fc, Kb, x0):
    """minimise F_L(u) = sum_k exp(-L x_k(u)), x(u) = x0 + Kb u restricted to facets; return -log(eigs of Hess/(L^2 F))/L sorted."""
    d = Kb.shape[1]; rows = [[mp.mpf(float(Kb[k, i])) for i in range(d)] for k in Fc]; c0 = [mp.mpf(float(x0[k])) for k in Fc]; u = [mp.mpf(0)] * d
    for _ in range(200):
        e = [mp.e ** (-L * (c0[a] + mp.fsum(rows[a][i] * u[i] for i in range(d)))) for a in range(len(Fc))]
        gvec = mp.matrix([-L * mp.fsum(e[a] * rows[a][i] for a in range(len(Fc))) for i in range(d)])
        H = mp.matrix(d, d)
        for i in range(d):
            for j in range(d): H[i, j] = L * L * mp.fsum(e[a] * rows[a][i] * rows[a][j] for a in range(len(Fc)))
        step = mp.lu_solve(H, gvec)
        u = [u[i] - step[i] for i in range(d)]
        if mp.norm(step) < mp.mpf(10) ** (-60): break
    Fv = mp.fsum(e); ev = mp.eigsy(H / (L * L * Fv))[0]
    return sorted(float(-mp.log(abs(ev[i])) / L) for i in range(d))
bad = 0; done = 0; worst = 0.0
for (n, m, f, g, Fc, Kb, pred) in cases_store[:24]:
    x0, _ = leximin_levels(f, g, Fc)                                           # base point: the leximin point itself (u = 0 there)
    h1 = min(pred); target = sorted(p - h1 for p in pred); Ls = [60, 120, 240]
    vals = np.array([scales_at(L, Fc, Kb, x0) for L in Ls])
    ext = [np.polyfit([1 / L for L in Ls], vals[:, i], 1)[1] for i in range(vals.shape[1])]
    err = max(abs(a - b) for a, b in zip(sorted(ext), target)); worst = max(worst, err); bad += err > 0.01; done += 1
check("SV4 IDENTITY finite-L eigenvalue valuations of Q/W at the positive critical point tend to h_j - h_1 with multiplicity r_j", bad == 0 and done >= 20,
      f"{done} margins; worst extrapolation error {worst:.4f}")
print(f"{sum(results)}/{len(results)} passed")
