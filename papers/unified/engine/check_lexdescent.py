"""Study 36: leximin descent under merging (sec:lexdescent). Pre-registered in study36_descent/PREREG_study36.md, frozen by SHA-256
(study36_descent/FROZEN_PREREG_study36.txt) before any code for it ran. New script; reuses helpers of check_four6.py (code before LZ1,
and exact_leximin from its LZ5 block) and wleximin from example_descent.py; modifies nothing.
  LW1 IDENTITY prop:symdescent (exact)        LW2 IDENTITY prop:firstbound        LW3 IDENTITY piecewise linearity in a chamber
  LW4 RECORD sensitivity                       LW5 HYPOTHESIS weighting helps       LW6 RECORD descent rates
Deviation (recorded before results, study36_descent/DEVIATION_study36.md): arm S certifies the fine leximin point exactly instead of solving
for it exactly.
First complete run: 6/6 (after the two recorded deviations; the earlier attempts produced no results).
Output: check_lexdescent_output.txt, study36_descent/lexdescent.json.
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from pce import credalledger as CL
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "check_four6.py")).read(); exec(src.split("# population ----")[0])
exec(src.split("# LZ5 Theorem 196 ------------------------------------------------------")[1].split("mp.mp.dps")[0])
exec(open(os.path.join(HERE, "example_descent.py")).read().split("def show")[0])
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
r, c = 4, 6; cells, A, N, D = shape(r, c)
MERGES = {"E": ([[0, 1], [2, 3]], [[0, 1], [2, 3], [4, 5]]), "U": ([[0], [1, 2, 3]], [[0, 1], [2, 3, 4, 5]])}
def merged_margins(f, g, RG, CG): return [sum(f[i] for i in R_) for R_ in RG], [sum(g[j] for j in C_) for C_ in CG]
def mergeX(X, RG, CG): return np.array([[X[np.ix_(R_, C_)].sum() for C_ in CG] for R_ in RG])
def ok_margins(f, g): return nondeg(f, g) and all(f[i] + g[j] < 1 for i in range(len(f)) for j in range(len(g)))
def ok_facets(f, g): return all(f[i] + g[j] < 1 for i in range(len(f)) for j in range(len(g)))
def symmetric(rng, groups, den=240):
    sizes = [len(G) for G in groups]
    while True:
        w = rng.dirichlet([2] * len(groups)); v = [max(1, int(round(den * w[k] / sizes[k]))) for k in range(len(groups))]
        rem = den - sum(s * x for s, x in zip(sizes, v)); k = int(np.argmax(sizes)); s = sizes[k]
        for kk in range(len(groups)):
            if rem % sizes[kk] == 0 and v[kk] + rem // sizes[kk] >= 1: v[kk] += rem // sizes[kk]; rem = 0; break
        if rem != 0: continue
        out = [None] * sum(sizes)
        for G, x in zip(groups, v):
            for i in G: out[i] = Fr(x, den)
        return out
def wlex_float(fv, gv, w):
    rr_, cc_ = len(fv), len(gv); cl = [(i, j) for i in range(rr_) for j in range(cc_)]; n = len(cl)
    Aq = np.array([[1.0 if x[0] == i else 0.0 for x in cl] for i in range(rr_)] + [[1.0 if x[1] == j else 0.0 for x in cl] for j in range(cc_ - 1)])
    b = list(fv) + list(gv[:-1]); fixed = {}; free = set(range(n)); levels = []
    while free:
        cvec = np.zeros(n + 1); cvec[-1] = -1; Aeq = np.hstack([Aq, np.zeros((len(Aq), 1))]); beq = list(b)
        for k, v in fixed.items(): row = np.zeros(n + 1); row[k] = 1; Aeq = np.vstack([Aeq, row]); beq.append(v)
        Aub = []
        for k in sorted(free): row = np.zeros(n + 1); row[k] = -w[k]; row[-1] = 1; Aub.append(row)
        t = linprog(cvec, A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * n + [(None, None)], method="highs").x[-1]
        newly = []
        for k in sorted(free):
            c2 = np.zeros(n); c2[k] = -1; Aub2 = []
            for kk in sorted(free): row = np.zeros(n); row[kk] = -w[kk]; Aub2.append(row)
            Aeq2 = Aq.copy(); beq2 = list(b)
            for kk, v in fixed.items(): row = np.zeros(n); row[kk] = 1; Aeq2 = np.vstack([Aeq2, row]); beq2.append(v)
            rr2 = linprog(c2, A_ub=np.array(Aub2), b_ub=-t * np.ones(len(Aub2)), A_eq=Aeq2, b_eq=beq2, bounds=[(0, None)] * n, method="highs")
            if -rr2.fun * w[k] <= t + 1e-10: newly.append(k)
        for k in newly: fixed[k] = t / w[k]; free.discard(k)
        levels.append((t, frozenset(cl[k] for k in newly)))
    Aeq2 = Aq.copy(); beq2 = list(b)
    for kk, v in fixed.items(): row = np.zeros(n); row[kk] = 1; Aeq2 = np.vstack([Aeq2, row]); beq2.append(v)
    x = linprog(np.zeros(n), A_eq=Aeq2, b_eq=beq2, bounds=[(0, None)] * n, method="highs").x
    return x.reshape(rr_, cc_), levels
def frank(rows):
    M = [list(rw) for rw in rows]; rk = 0; ncol = len(M[0]) if M else 0
    for col in range(ncol):
        p = next((i for i in range(rk, len(M)) if M[i][col] != 0), None)
        if p is None: continue
        M[rk], M[p] = M[p], M[rk]
        for i in range(len(M)):
            if i != rk and M[i][col] != 0:
                fct = M[i][col] / M[rk][col]; M[i] = [a - fct * b for a, b in zip(M[i], M[rk])]
        rk += 1
    return rk
def certify_lex(X, f, g):
    """exact certificate of lem:lexunique(ii) for a candidate table X (all cells facets): margins exact, and every level set carries a
    strictly positive relation modulo the earlier levels."""
    if any(sum(X[i]) != f[i] for i in range(r)) or any(sum(X[i][j] for i in range(r)) != g[j] for j in range(c)) or min(min(rw) for rw in X) < 0: return False
    vals = sorted({X[i][j] for i in range(r) for j in range(c)}); earlier = []
    for h in vals:
        S_ = [(i, j) for i in range(r) for j in range(c) if X[i][j] == h]; V = np.array([N[cl] for cl in S_], float).T
        E = np.array(earlier, float).T if earlier else np.zeros((D, 0))
        res = linprog(np.zeros(len(S_) + E.shape[1]), A_eq=np.hstack([V, -E]), b_eq=np.zeros(D), bounds=[(1, None)] * len(S_) + [(None, None)] * E.shape[1], method="highs")
        if res.status != 0: return False
        q = [Fr(v).limit_denominator(10 ** 6) for v in res.x[:len(S_)]]
        if min(q) <= 0: return False
        w = [sum(q[k] * N[S_[k]][d] for k in range(len(S_))) for d in range(D)]
        if earlier:
            rk = frank([[Fr(x) for x in e] for e in earlier]); rk2 = frank([[Fr(x) for x in e] for e in earlier] + [w])
            if rk2 != rk: return False
        elif any(x != 0 for x in w): return False
        earlier += [N[cl] for cl in S_]
    return True
def exact_from_float(fv, gv, w):
    """float weighted leximin, then exact level values from the margin equations; returns {cell: Fraction} or None."""
    rr_, cc_ = len(fv), len(gv); Xf, lv = wlex_float([float(x) for x in fv], [float(x) for x in gv], [float(x) for x in w])
    cl = [(i, j) for i in range(rr_) for j in range(cc_)]; lev = {}
    for k, (t, S_) in enumerate(lv):
        for x in S_: lev[x] = k
    J = len(lv); rows = []; rhs = []
    for i in range(rr_): rows.append([sum(Fr(1) / w[cl.index(x)] for x in cl if x[0] == i and lev[x] == k) for k in range(J)]); rhs.append(fv[i])
    for j in range(cc_): rows.append([sum(Fr(1) / w[cl.index(x)] for x in cl if x[1] == j and lev[x] == k) for k in range(J)]); rhs.append(gv[j])
    # exact least squares via normal equations (consistent system)
    AtA = [[sum(rows[q][a] * rows[q][b] for q in range(len(rows))) for b in range(J)] for a in range(J)]; Atb = [sum(rows[q][a] * rhs[q] for q in range(len(rows))) for a in range(J)]
    h = CL.gsolve(AtA, Atb) if J else []
    if h is None: return None
    y = {x: h[lev[x]] / w[cl.index(x)] for x in cl}
    if any(sum(y[x] for x in cl if x[0] == i) != fv[i] for i in range(rr_)) or any(sum(y[x] for x in cl if x[1] == j) != gv[j] for j in range(cc_)) or min(y.values()) < 0: return None
    return y
def certify_general(y, rr_, cc_, w):
    _, _, Nn, Dd = shape(rr_, cc_); cl = [(i, j) for i in range(rr_) for j in range(cc_)]
    vals = sorted({y[x] * w[cl.index(x)] for x in cl}); earlier = []
    for hv in vals:
        S_ = [x for x in cl if y[x] * w[cl.index(x)] == hv]; V = np.array([Nn[x] for x in S_], float).T
        E = np.array(earlier, float).T if earlier else np.zeros((Dd, 0))
        res = linprog(np.zeros(len(S_) + E.shape[1]), A_eq=np.hstack([V, -E]), b_eq=np.zeros(Dd), bounds=[(1, None)] * len(S_) + [(None, None)] * E.shape[1], method="highs")
        if res.status != 0: return False
        q = [Fr(v).limit_denominator(10 ** 6) for v in res.x[:len(S_)]]
        if min(q) <= 0: return False
        wv = [sum(q[k] * Nn[S_[k]][d] for k in range(len(S_))) for d in range(Dd)]
        if earlier:
            if frank([[Fr(x) for x in e] for e in earlier] + [wv]) != frank([[Fr(x) for x in e] for e in earlier]): return False
        elif any(x != 0 for x in wv): return False
        earlier += [Nn[x] for x in S_]
    return True
# LW1 ------------------------------------------------------------------
bad1 = 0; info1 = []; ex_store = {}
for mi, (key, (RG, CG)) in enumerate(MERGES.items()):
    rng = np.random.default_rng(20261136 + mi); done = 0; nb = 0; plain_ok = 0
    while done < 30:
        f, g = symmetric(rng, RG), symmetric(rng, CG)
        if not ok_facets(f, g): continue
        fc, gc = merged_margins(f, g, RG, CG); wts = [Fr(1, len(R_) * len(C_)) for R_ in RG for C_ in CG]
        xa = exact_from_float(fc, gc, wts); xp = exact_from_float(fc, gc, [Fr(1)] * (len(RG) * len(CG)))
        if xa is None or xp is None or not certify_general(xa, len(RG), len(CG), wts) or not certify_general(xp, len(RG), len(CG), [Fr(1)] * (len(RG) * len(CG))):
            bad1 += 1; done += 1; continue
        X = [[None] * c for _ in range(r)]
        for a, R_ in enumerate(RG):
            for b_, C_ in enumerate(CG):
                for i in R_:
                    for j in C_: X[i][j] = xa[(a, b_)] / (len(R_) * len(C_))
        certified = certify_lex(X, f, g); blockconst = certified
        M = [[sum(X[i][j] for i in R_ for j in C_) for C_ in CG] for R_ in RG]
        avg_ok = all(M[a][b_] == xa[(a, b_)] for a in range(len(RG)) for b_ in range(len(CG))); pl = all(M[a][b_] == xp[(a, b_)] for a in range(len(RG)) for b_ in range(len(CG)))
        bad1 += (not blockconst) or (not avg_ok) or (key == "E" and not pl); plain_ok += pl; done += 1
    info1.append(f"merge {key}: 30 tables, block-constant and weighted descent exact in all unless counted; plain coarse descent {plain_ok}/30")
check("LW1 IDENTITY prop:symdescent: symmetric margins give a block-constant fine leximin point that descends to weighted coarse leximin (and to plain coarse leximin for equal blocks)", bad1 == 0, f"failures {bad1}; " + "; ".join(info1))
# LW2 + LW5 + LW6 (random arm) -------------------------------------------
RG, CG = MERGES["U"]; rng = np.random.default_rng(20261136 + 2); dp = []; dw = []; bad2 = 0; n2 = 0
while n2 < 1000:
    f, g = margins(rng, r, 240), margins(rng, c, 240)
    if not (f and g and ok_margins(f, g)): continue
    fv = [float(x) for x in f]; gv = [float(x) for x in g]; Xf, lf = wlex_float(fv, gv, [1.0] * 24)
    fc, gc = merged_margins(fv, gv, RG, CG); wts = [1.0 / (len(R_) * len(C_)) for R_ in RG for C_ in CG]
    Xw, lw = wlex_float(fc, gc, wts); Xp, _ = wlex_float(fc, gc, [1.0] * 4); M = mergeX(Xf, RG, CG)
    dp.append(float(np.abs(M - Xp).max())); dw.append(float(np.abs(M - Xw).max())); bad2 += lf[0][0] > lw[0][0] + 1e-9; n2 += 1
dp, dw = np.array(dp), np.array(dw)
check("LW2 IDENTITY prop:firstbound: the fine first level never exceeds the weighted coarse first level", bad2 == 0, f"1000 tables; violations {bad2}")
# LW3 + LW4 (small-imbalance arm) -----------------------------------------
RG, CG = MERGES["E"]; rng = np.random.default_rng(20261136 + 3); n3 = 0; bad3 = 0; tested3 = 0; same_all = 0; sens = []; worst3 = 0.0
sig = lambda lv: [S_ for _, S_ in lv]
while n3 < 100:
    f, g = symmetric(rng, RG), symmetric(rng, CG)
    if not ok_facets(f, g): continue
    n3 += 1; fv0 = np.array([float(x) for x in f]); gv0 = np.array([float(x) for x in g])
    a = rng.uniform(-1, 1, size=len(RG) + len(CG)); a /= np.abs(a).max(); df = np.zeros(r); dg = np.zeros(c)
    for k, R_ in enumerate(RG): df[R_[0]] += a[k]; df[R_[1]] -= a[k]
    for k, C_ in enumerate(CG): dg[C_[0]] += a[len(RG) + k]; dg[C_[1]] -= a[len(RG) + k]
    X0, _ = wlex_float(fv0, gv0, [1.0] * 24); M0 = mergeX(X0, RG, CG)
    def at(e): X, lv = wlex_float(fv0 + e * df, gv0 + e * dg, [1.0] * 24); return mergeX(X, RG, CG), sig(lv)
    Mh, sh = at(1e-6); Dv = (Mh - M0) / 1e-6; sens.append(float(np.abs(Dv).max())); sigs = {}; Ms = {}
    for e in (0.0005, 0.001, 0.002, 0.004): Ms[e], sigs[e] = at(e)
    same_all += all(sigs[e] == sh for e in sigs)
    for e in (0.0005, 0.001, 0.002):
        if sigs[e] == sh and sigs[2 * e] == sh:
            tested3 += 1; err = float(np.abs(Ms[e] - M0 - e * Dv).max()); worst3 = max(worst3, err); bad3 += err > 1e-7
check("LW3 IDENTITY piecewise linearity: within an unchanged level signature the defect is eps times the directional derivative", bad3 == 0,
      f"{tested3} (table, eps) cases tested; violations {bad3}; worst error {worst3:.1e}")
sens = np.array(sens)
check("LW4 RECORD sensitivity the coarse view cannot see", True,
      f"signature unchanged up to eps = 0.004 in {same_all}/100; |q D x_f*[delta]| median {np.median(sens):.3f}, max {sens.max():.3f}, zero (<1e-6) in {(sens < 1e-6).sum()}")
check("LW5 HYPOTHESIS weighting helps under unequal blocks: median weighted defect < median plain defect", np.median(dw) < np.median(dp),
      f"median weighted {np.median(dw):.4f} vs plain {np.median(dp):.4f}")
check("LW6 RECORD descent rates and correction gain (merge U, 1000 random tables)", True,
      f"descent plain {(dp <= 1e-9).mean():.3f}, weighted {(dw <= 1e-9).mean():.3f}; max defect plain {dp.max():.4f}, weighted {dw.max():.4f}; "
      f"correction gain plain - weighted: median {np.median(dp - dw):.4f}, positive in {((dp - dw) > 1e-9).mean():.3f}, negative in {((dp - dw) < -1e-9).mean():.3f}")
json.dump(dict(defect_plain=dp.tolist(), defect_weighted=dw.tolist(), sensitivity=sens.tolist(), lw3_tested=tested3, lw3_worst=worst3, same_all=same_all),
          open(os.path.join(HERE, "study36_descent", "lexdescent.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
