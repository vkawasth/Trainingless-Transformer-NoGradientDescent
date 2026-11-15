"""The obstruction map (section sec:obstruction): where gluing fails, of which type, whether it matters for the decision, and since when.
New module; it modifies nothing. Tables are probability vectors on contexts of the Study 26 variables (outerrelay.ctxmat order).
  eta(ctx, n, t)                    l1 tolerance of a table: 3 x sum of cell standard errors (Study 26 rule)
  shared(ctxA, ctxB)                shared variables of two contexts
  pair_ok(ca, ta, na, cb, tb, nb)   do two tables agree on their overlap, within the summed tolerances of the restricted tables
  glues(blocks)                     is there a global law within tolerance of every table (LP)
  obstruction(blocks, g_observers)  type 'none' / 'defect' / 'cycle' / 'unresolved'; explanations (single removals that restore gluing);
                                    named source for a defect; decision relevance (does the region contain an observer of g)
  anova(gvals, ses, groups)         the register's inertia Q split into within-group and between-group parts, with chi-square tails
  twisted_cycle(P, ctxs)            three pairwise tables on a cycle of contexts with the honest single margins, two couplings maximal
                                    and one minimal, globally incompatible (a contextual decoy)
"""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import chi2
from . import outerrelay as O
NS = 32
def eta(ctx, n, t):
    t = np.asarray(t, float); return 3 * float(np.sum(np.sqrt(np.clip(t * (1 - t), 1e-12, None) / n)))
def shared(a, b): return tuple(v for v in a if v in b)
def marg(ctx, t, sub):
    st = list(itertools.product((0, 1), repeat=len(ctx))); out = {}
    for i, s in enumerate(st):
        k = tuple(s[ctx.index(v)] for v in sub); out[k] = out.get(k, 0.0) + t[i]
    return np.array([out.get(k, 0.0) for k in itertools.product((0, 1), repeat=len(sub))])
def pair_ok(ca, ta, na, cb, tb, nb):
    sh = shared(ca, cb)
    if not sh: return True
    ma, mb = marg(ca, ta, sh), marg(cb, tb, sh)
    return float(np.abs(ma - mb).sum()) <= eta(sh, na, ma) + eta(sh, nb, mb)
def glues(blocks):
    """blocks: list of (ctx, table, n)"""
    if not blocks: return True
    A_ub, b_ub = [], []; nsl = sum(2 ** len(c) for c, _, _ in blocks); nv = NS + nsl; k = 0
    for c, t, n in blocks:
        M = O.ctxmat(c); m = len(t)
        for i in range(m):
            r = np.zeros(nv); r[:NS] = M[i]; r[NS + k + i] = -1; A_ub.append(r); b_ub.append(t[i])
            r = np.zeros(nv); r[:NS] = -M[i]; r[NS + k + i] = -1; A_ub.append(r); b_ub.append(-t[i])
        r = np.zeros(nv); r[NS + k:NS + k + m] = 1; A_ub.append(r); b_ub.append(eta(c, n, t)); k += m
    res = linprog(np.zeros(nv), A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.r_[np.ones(NS), np.zeros(nsl)][None], b_eq=[1.0],
                  bounds=[(0, None)] * nv, method="highs")
    return res.status == 0
def obstruction(named_blocks, g_observers=()):
    """named_blocks: dict name -> (ctx, table, n). Returns dict(type, explanations, named, relevant, pair_failures)."""
    names = list(named_blocks); blocks = [named_blocks[k] for k in names]
    if glues(blocks): return dict(type="none", explanations=[], named=[], relevant=False, pair_failures=[])
    pf = [(a, b) for a, b in itertools.combinations(names, 2) if not pair_ok(*named_blocks[a], *named_blocks[b])]
    expl = [k for k in names if glues([named_blocks[j] for j in names if j != k])]
    if not expl: typ, named = "unresolved", []
    elif pf:
        # a defect fails some overlap; the culprit is the explanation common to every failing pair
        cand = [k for k in expl if all(k in p for p in pf)]
        typ, named = ("defect", cand) if len(cand) == 1 else ("unresolved", [])
    else:
        typ, named = ("cycle", [])                       # every overlap agrees, yet the whole does not glue: a cycle obstruction
    region = named if typ == "defect" else expl
    return dict(type=typ, explanations=expl, named=named, relevant=bool(set(region) & set(g_observers)), pair_failures=pf)
def anova(gvals, ses, groups):
    """gvals, ses: dict name -> value / standard error; groups: list of lists of names. Returns total Q, within parts, between part,
    with degrees of freedom and upper tail probabilities (independent under the Gaussian model)."""
    w = {k: 1 / ses[k] ** 2 for k in gvals}; W = sum(w.values()); xb = sum(w[k] * gvals[k] for k in gvals) / W
    Q = sum(w[k] * (gvals[k] - xb) ** 2 for k in gvals); out = dict(Q=Q, df=len(gvals) - 1, p=float(chi2.sf(Q, len(gvals) - 1)), within=[], between=None)
    btw = 0.0
    for gset in groups:
        wg = sum(w[k] for k in gset); mg = sum(w[k] * gvals[k] for k in gset) / wg
        qw = sum(w[k] * (gvals[k] - mg) ** 2 for k in gset); out["within"].append(dict(group=list(gset), Q=qw, df=len(gset) - 1,
                                                                                   p=(float(chi2.sf(qw, len(gset) - 1)) if len(gset) > 1 else None)))
        btw += wg * (mg - xb) ** 2
    out["between"] = dict(Q=btw, df=len(groups) - 1, p=float(chi2.sf(btw, len(groups) - 1)))
    return out
def pair_table(P, a, b, mode):
    """2x2 table on (a, b) with the honest single margins of P and the maximal (mode = +1) or minimal (-1) coupling"""
    pa = float(O.ev(lambda v: v[a] == 1) @ P); pb = float(O.ev(lambda v: v[b] == 1) @ P)
    p11 = min(pa, pb) if mode > 0 else max(0.0, pa + pb - 1)
    return np.array([1 - pa - pb + p11, pb - p11, pa - p11, p11])        # (a,b) = 00, 01, 10, 11
def twisted_cycle(P, ctxs):
    """ctxs: three 2-variable contexts forming a cycle; first two maximal couplings, the third minimal"""
    return [pair_table(P, *ctxs[0], +1), pair_table(P, *ctxs[1], +1), pair_table(P, *ctxs[2], -1)]
