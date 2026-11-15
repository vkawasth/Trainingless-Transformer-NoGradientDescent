"""The snapshot engine (section sec:engine): one operator representation for time-indexed families of overlapping contingency tables.
New module; it modifies nothing. Variables may have any number of levels. Every infeasible check returns a certificate.
Levels (engine levels, not the paper's tower):
  World(levels)                          the joint outcome space (all variables, given level counts) and its marginal matrices
  Table(id, ctx, counts, family, role)   one reported table; role 'source' (inner layer) or 'anchor' (outer layer)
 1 observation geometry
  geometry(world, ctxs, p, grad)         V = ker A on the tangent space, H = V^perp in the Fisher metric, dims, the share of a
                                         functional's Fisher gradient in V (invisible share), sigma_min^+ of A
 2 compatibility
  coherence(world, tables)               the Cech operator d0 (pairwise overlap marginals) and the weighted projection of the tables onto
                                         ker d0: coherent fit x, residual statistic Q (Pearson-weighted), per-table contributions; Q is
                                         approximately chi-square with df = rank d0 for sampling noise
  classify(world, tables, ...)           'none' / 'defect' (named, by refitting without the largest contributor) / 'cycle' (linearly
                                         coherent but no positive global law within tolerance; certificate: the contexts with non-zero
                                         dual multipliers in the minimum-slack LP) / 'unresolved' (linear incoherence no single refit removes)
  positivity(world, fits, tol)           min-slack LP for a global law within tolerance; returns slack per table (the certificate)
  contextual_fraction(world, fits, tol)  1 - max{lambda : lambda p <= table + tol on every cell, p a law} (Abramsky-Barbosa-Mansfield)
  gyo_core(ctxs)                         GYO reduction of the cover's hypergraph; an empty core (acyclic cover) admits no cycle obstruction
  bracket(world, fits, tol, num, den)    exact bounds of a ratio a.p / b.p over the laws within tolerance (Charnes-Cooper LP)
  veto(world, inner, outer)              Q(inner + outer) - Q(inner): the outer layer's added incoherence, with its df
 3 decision and protection
  voi_rank(world, ctxs, p, grad, cands)  candidate contexts ranked by how much of the invisible share they remove
  singularity(fit_hist)                  per-cell trend scan over snapshots: cells heading for the simplex boundary
"""
from __future__ import annotations
import itertools, math
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg
from scipy.optimize import linprog
from scipy.linalg import null_space
class World:
    def __init__(self, levels):
        self.vars = list(levels); self.levels = dict(levels)
        self.states = list(itertools.product(*[range(levels[v]) for v in self.vars])); self.NS = len(self.states)
        self._cache = {}
    def ncells(self, ctx): return int(np.prod([self.levels[v] for v in ctx]))
    def A(self, ctx):
        ctx = tuple(ctx)
        if ctx not in self._cache:
            idx = [self.vars.index(v) for v in ctx]; L = [self.levels[v] for v in ctx]; M = np.zeros((self.ncells(ctx), self.NS))
            for i, s in enumerate(self.states):
                c = 0
                for j, l in zip(idx, L): c = c * l + s[j]
                M[c, i] = 1
            self._cache[ctx] = M
        return self._cache[ctx]
    def marg(self, ctx, sub, t):
        """marginal of a table on ctx onto the sub-context sub (a matrix product with a small 0/1 matrix)."""
        return self.submat(ctx, sub) @ t
    def submat(self, ctx, sub):
        key = (tuple(ctx), tuple(sub))
        if key not in self._cache:
            L = [self.levels[v] for v in ctx]; cells = list(itertools.product(*[range(l) for l in L])); pos = [ctx.index(v) for v in sub]
            Ls = [self.levels[v] for v in sub]; M = np.zeros((int(np.prod(Ls)) if sub else 1, len(cells)))
            for i, c in enumerate(cells):
                k = 0
                for j, l in zip(pos, Ls): k = k * l + c[j]
                M[k, i] = 1
            self._cache[key] = M
        return self._cache[key]
class Table:
    def __init__(self, id, ctx, counts, family="", role="source"):
        self.id, self.ctx, self.counts, self.family, self.role = id, tuple(ctx), np.asarray(counts, float), family, role
        self.n = float(self.counts.sum()); self.t = self.counts / self.n
# ---------- level 1: observation geometry ----------
def geometry(world, ctxs, p, grad=None):
    NS = world.NS; one = np.ones((1, NS)); A = np.vstack([world.A(c) for c in ctxs]) if ctxs else np.zeros((0, NS))
    Vb = null_space(np.vstack([A, one])); G = np.diag(1 / p); Hb = null_space(np.vstack([Vb.T @ G, one])) if Vb.shape[1] else null_space(one)
    sv = np.linalg.svd(np.vstack([A, one]), compute_uv=False); smin = float(sv[sv > 1e-9 * sv.max()].min())
    out = dict(dimV=int(Vb.shape[1]), dimH=int(Hb.shape[1]), sigma_min_plus=smin, invisible_share=None)
    if grad is not None:
        fg = p * (grad - p @ grad); nf = math.sqrt(fg @ G @ fg)
        if Vb.shape[1] == 0 or nf == 0: out["invisible_share"] = 0.0
        else:
            pv = Vb @ np.linalg.solve(Vb.T @ G @ Vb, Vb.T @ G @ fg); out["invisible_share"] = float(math.sqrt(max(pv @ G @ pv, 0)) / nf)
    return out
# ---------- level 2: compatibility ----------
class Cover:
    """The structure of a set of tables (contexts only): the sparse Cech operator d0 over every overlapping pair, and its rank."""
    def __init__(self, world, ctxs):
        self.world, self.ctxs = world, [tuple(c) for c in ctxs]; self.sizes = [world.ncells(c) for c in self.ctxs]
        self.off = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int); rows, cols, vals = [], [], []; r = 0; self.pairs = []
        for a, b in itertools.combinations(range(len(self.ctxs)), 2):
            S = tuple(v for v in self.ctxs[a] if v in self.ctxs[b])
            if not S: continue
            Ma, Mb = world.submat(self.ctxs[a], S), world.submat(self.ctxs[b], S)
            for M, k, sg in ((Ma, a, 1.0), (Mb, b, -1.0)):
                ii, jj = np.nonzero(M); rows += list(ii + r); cols += list(jj + self.off[k]); vals += [sg] * len(ii)
            self.pairs.append((a, b, S, r, Ma.shape[0])); r += Ma.shape[0]
        self.d0 = sps.csr_matrix((vals, (rows, cols)), shape=(r, int(self.off[-1])))
        self.rank = None
    def df(self):
        """degrees of freedom of Q: the rank of d0 on the noise subspace (each table's cells sum to zero), not the rank of d0 itself,
        because every table has total mass 1 and the total-mass rows of d0 carry no noise (found by check EN3; see DEVNOTES_engine.md)."""
        if self.rank is None:
            D = self.d0.toarray(); Pi = np.eye(D.shape[1])
            for k in range(len(self.ctxs)):
                a, b = self.off[k], self.off[k + 1]; Pi[a:b, a:b] -= 1.0 / (b - a)
            M = D @ Pi; w = np.linalg.eigvalsh(M.T @ M); self.rank = int(np.sum(w > 1e-8 * max(w.max(), 1)))
        return self.rank
def coherence(cover, tables, floor=None):
    """Weighted projection of the stacked tables onto ker d0 (Pearson weights n / t). Returns x (list per table), Q, contributions."""
    t = np.concatenate([tb.t for tb in tables]); n = np.concatenate([np.full(len(tb.t), tb.n) for tb in tables])
    fl = floor if floor is not None else 0.5 / n
    winv = np.maximum(t, fl) / n; D = cover.d0
    if D.shape[0] == 0: x = t.copy()
    else:
        rhs = D @ t; op = sps.linalg.LinearOperator((D.shape[0],) * 2, matvec=lambda y: D @ (winv * (D.T @ y)))
        lam, _ = cg(op, rhs, rtol=1e-12, atol=1e-14, maxiter=5000); x = t - winv * (D.T @ lam)
    res = t - x; Qc = res ** 2 / winv
    xs = [x[cover.off[k]:cover.off[k + 1]] for k in range(len(tables))]
    contrib = np.array([Qc[cover.off[k]:cover.off[k + 1]].sum() for k in range(len(tables))])
    pair_res = [(a, b, float(np.abs(cover.d0[r:r + m] @ t).sum())) for a, b, S, r, m in cover.pairs]
    return dict(x=xs, Q=float(Qc.sum()), contrib=contrib, pair_res=pair_res)
def tol_of(tb, z=3.0): return z * np.sqrt(np.clip(tb.t * (1 - tb.t), 1e-12, None) / tb.n)
def positivity(world, ctxs, fits, tols):
    """min sum of slacks s_k such that some law p has |A_k p - fit_k| <= tol_k + s_k (cellwise). Returns total slack and slack per table."""
    NS = world.NS; K = len(ctxs); nv = NS + K; Aub, bub = [], []
    for k, (c, f, tl) in enumerate(zip(ctxs, fits, tols)):
        A = world.A(c); S = np.zeros((A.shape[0], K)); S[:, k] = -1
        Aub.append(np.hstack([A, S])); bub.append(f + tl); Aub.append(np.hstack([-A, S])); bub.append(-(f - tl))
    Aeq = np.hstack([np.ones((1, NS)), np.zeros((1, K))]); cvec = np.concatenate([np.zeros(NS), np.ones(K)])
    r = linprog(cvec, A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=Aeq, b_eq=[1], bounds=[(0, None)] * nv, method="highs")
    s = r.x[NS:]; y = np.abs(r.ineqlin.marginals); dual = []; o = 0
    for c in ctxs:
        m = world.ncells(c); dual.append(float(y[o:o + m].sum() + y[o + m:o + 2 * m].sum())); o += 2 * m
    return dict(total=float(s.sum()), slack=s, p=r.x[:NS], dual=np.array(dual))
def contextual_fraction(world, ctxs, fits, tols):
    NS = world.NS; Aub = np.vstack([world.A(c) for c in ctxs]); bub = np.concatenate([f + tl for f, tl in zip(fits, tols)])
    r = linprog(-np.ones(NS), A_ub=Aub, b_ub=bub, bounds=[(0, None)] * NS, method="highs")
    return float(max(0.0, 1 - min(1.0, -r.fun)))
def gyo_core(ctxs):
    """GYO reduction: repeatedly delete variables that lie in one context only, and contexts contained in another (or empty).
    The cover is acyclic exactly when nothing is left; only the core can carry a cycle (contextual) obstruction (Vorob'ev)."""
    E = [set(c) for c in ctxs]
    while True:
        changed = False
        for v in (set().union(*E) if E else set()):
            hs = [e for e in E if v in e]
            if len(hs) == 1: hs[0].discard(v); changed = True
        newE = []
        for i, e in enumerate(E):
            if not e or any((e < f) or (e == f and j < i) for j, f in enumerate(E) if j != i): changed = True; continue
            newE.append(e)
        E = newE
        if not changed: break
    return [tuple(sorted(e)) for e in E]
def bracket(world, ctxs, fits, tols, num, den, sense):
    """max (sense=-1) or min (sense=1) of num.p / den.p over laws with |A_k p - fit_k| <= tol_k (Charnes-Cooper)."""
    NS = world.NS; Aub, bub = [], []
    for c, f, tl in zip(ctxs, fits, tols):
        A = world.A(c)
        Aub.append(np.hstack([A, -(f + tl)[:, None]])); bub.append(np.zeros(A.shape[0]))
        Aub.append(np.hstack([-A, (f - tl)[:, None]])); bub.append(np.zeros(A.shape[0]))
    Aeq = np.vstack([np.concatenate([den, [0]]), np.concatenate([np.ones(NS), [-1]])]); beq = [1, 0]
    r = linprog(sense * np.concatenate([num, [0]]), A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=Aeq, b_eq=beq,
                bounds=[(0, None)] * (NS + 1), method="highs")
    return float(sense * r.fun) if r.status == 0 else float("nan")
def classify(world, cover, tables, q_thr, refit_cover=None, ztol=3.0):
    """the inner obstruction: type and certificate."""
    coh = coherence(cover, tables)
    if coh["Q"] > q_thr:
        k = int(np.argmax(coh["contrib"])); rest = [tb for j, tb in enumerate(tables) if j != k]
        cov2 = Cover(world, [tb.ctx for tb in rest]); q2 = coherence(cov2, rest)["Q"]
        # the refit's threshold scales with its df
        thr2 = q_thr * max(cov2.df(), 1) / max(cover.df(), 1)
        return dict(type="defect" if q2 <= thr2 else "unresolved", named=[tables[k].id] if q2 <= thr2 else [], Q=coh["Q"], coh=coh)
    fits, ctxs, tols = dedup(tables, coh, ztol)
    pos = positivity(world, ctxs, fits, tols)
    if pos["total"] > 1e-7:
        # linearly coherent, yet no positive global law within tolerance: a cycle (contextual) obstruction. Certificate: the contexts
        # carrying non-zero dual multipliers of the minimum-slack LP, a Farkas-type proof of infeasibility (it lies in the GYO core).
        # Earlier versions used the slack of one optimum (check EN4) and then single removals that restore gluing (pilot of Study 27i:
        # with other honest paths between the same variables no single removal restores gluing); see DEVNOTES_engine.md.
        cert = [c for c, d in zip(ctxs, pos["dual"]) if d > 1e-9]
        return dict(type="cycle", named=[], Q=coh["Q"], coh=coh, certificate=cert, slack=pos["total"])
    return dict(type="none", named=[], Q=coh["Q"], coh=coh, slack=0.0)
def dedup(tables, coh, ztol=3.0):
    """one fitted table per context (coherent fits of replicates agree), with a pooled tolerance."""
    by = {}
    for tb, x in zip(tables, coh["x"]):
        d = by.setdefault(tb.ctx, [np.zeros_like(x), 0.0]); d[0] += tb.n * x; d[1] += tb.n
    ctxs = list(by); fits = [np.clip(by[c][0] / by[c][1], 0, None) for c in ctxs]; fits = [f / f.sum() for f in fits]
    tols = [ztol * np.sqrt(np.clip(f * (1 - f), 1e-12, None) / by[c][1]) for c, f in zip(ctxs, fits)]
    return fits, ctxs, tols
def veto(world, inner, outer, cover_in=None, cover_all=None):
    ci = cover_in or Cover(world, [tb.ctx for tb in inner]); ca = cover_all or Cover(world, [tb.ctx for tb in inner + outer])
    qi = coherence(ci, inner)["Q"]; qa = coherence(ca, inner + outer)["Q"]
    return dict(dQ=qa - qi, df=ca.df() - ci.df(), Q_inner=qi, Q_all=qa)
# ---------- level 3: decision and protection ----------
def voi_rank(world, ctxs, p, grad, cands):
    base = geometry(world, ctxs, p, grad)["invisible_share"]
    rows = [(c, geometry(world, list(ctxs) + [c], p, grad)["invisible_share"]) for c in cands]
    return base, sorted(rows, key=lambda r: r[1])
def singularity(hist, low=0.02, drop=0.5, horizon=3):
    """hist: list over snapshots of a fitted table (one context). Flags cells below 'low' that fell by at least 'drop' (relative) over
    the last 'horizon' snapshots. Returns the first flagged snapshot or None, and the flagged cells."""
    for s in range(horizon, len(hist)):
        a, b = hist[s - horizon], hist[s]; cells = [i for i in range(len(b)) if b[i] < low and a[i] > 0 and b[i] <= (1 - drop) * a[i]]
        if cells: return s, cells
    return None, []
