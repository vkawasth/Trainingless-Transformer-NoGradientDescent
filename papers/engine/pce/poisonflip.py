"""How little poisoned data tilts a decision, in many dimensions (companion note, section 12.20). New module; it modifies nothing and
uses pce/margsec.py.

Data are pair tables on a cover (the 4-cycle in margsec's CHSH coordinates, or the n-cycle in margsec.ncycle coordinates). An edit is
"stealthy" when the edited tables still come from one joint law (consistent margins and no contextuality, i.e. they stay in D).
Its cost is the summed total variation over the context tables, 1/2 ||E delta||_1 (the 2|s| of section 12.14 for one table).
  entries_chsh(), entries_ncycle(n)   E, e0: the table entries of a data vector d are E d + e0
  circuits(E)                         the circuits (minimal-support edits) of the consistent-edit space V = range(E), exact
  lower_upper(...)                    the identified interval [L, U] of w.x over the fibre (any sub-cover)
  flip_some(d, ...), flip_all(d, ...) cheapest stealthy edit after which some rule / every rule puts w.x <= tau
  along(d, c, ...)                    first t at which d + t c flips (some rule, every rule, or a given section)
  tangent_count(V, F, C)              signed circuits that keep d inside D at the relative interior of the face V
  Stream, removal                     pooled counts on information states (batches, contexts); removal maps; trajectories
"""
from __future__ import annotations
import itertools, collections
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from . import margsec as G

TOL = 1e-9


# ------------------------------------------------------------------ table entries as an affine map of the data coordinates
def _entries_from(fn, dim):
    e0 = np.array([float(x) for x in fn([Fr(0)] * dim)])
    E = np.array([[float(x) for x in fn([Fr(1) if j == k else Fr(0) for j in range(dim)])] for k in range(dim)]).T - e0[:, None]
    return E, e0


def _tab_chsh(d): return [x for t in G.table(d) for r in t for x in r]


def _tab_ncycle(n):
    def f(d):
        out = []
        for i in range(n):
            a, b, t = d[n + i], d[n + (i + 1) % n], d[i]; out += [t, a - t, b - t, 1 - a - b + t]
        return out
    return f


def entries_chsh(): return _entries_from(_tab_chsh, 8)
def entries_ncycle(n): return _entries_from(_tab_ncycle(n), 2 * n)
def cost(E, delta): return 0.5 * float(np.abs(E @ np.asarray(delta, float)).sum())


# ------------------------------------------------------------------ circuits of the consistent-edit space
def circuits(E):
    """all circuits (up to sign) of V = range(E): data edits delta whose entry vector E delta has minimal support. Each is returned as an
    exact integer vector delta with gcd 1 and first nonzero entry positive, normalised separately for cost. Enumeration: a circuit's zero
    set Z has rank(E_Z) = dim - 1, so every circuit is the kernel of some dim - 1 independent rows of E."""
    m, dim = E.shape; Ei = np.rint(E).astype(int); seen = {}
    for rows in itertools.combinations(range(m), dim - 1):
        A = E[list(rows)]
        if np.linalg.matrix_rank(A) != dim - 1: continue
        v = np.linalg.svd(A)[2][-1]; v = v / v[np.argmax(np.abs(v) > 1e-9)]
        fr = [Fr(x).limit_denominator(60) for x in v]; den = int(np.lcm.reduce([f.denominator for f in fr])); iv = [int(f * den) for f in fr]
        g = int(np.gcd.reduce([abs(x) for x in iv if x])); iv = tuple(x // g for x in iv)
        if any(sum(Ei[r][k] * iv[k] for k in range(dim)) != 0 for r in rows): continue          # exact kernel check
        ent = [sum(Ei[r][k] * iv[k] for k in range(dim)) for r in range(m)]; sup = frozenset(r for r in range(m) if ent[r])
        seen.setdefault(sup, iv)
    # keep minimal supports only
    sups = sorted(seen, key=len)
    out = [(s, seen[s]) for s in sups if not any(t < s for t in sups)]
    return [(sorted(s), np.array(v, float)) for s, v in out]


def signed_unit(E, C):
    """both signs of each circuit, scaled to cost 1"""
    out = []
    for sup, v in C:
        v = v / cost(E, v); out += [v, -v]
    return out


# ------------------------------------------------------------------ fibre LPs (any cover given by its constraint rows)
def chsh_rows(ctx=(0, 1, 2, 3)):
    """rows of the 4-cycle data coordinates observed by the sub-cover ctx (indices into margsec.CTX): the 00 entry and both singles"""
    rows = set()
    for k in ctx:
        i, j = G.CTX[k]; rows |= {k, 4 + i, 6 + (j - 2)}
    return sorted(rows)


def lower_upper(d, w, M=None, rows=None):
    M = np.array(G.M if M is None else M, float); rows = list(range(M.shape[0])) if rows is None else list(rows)
    A = np.vstack([M[rows], np.ones(M.shape[1])]); b = np.r_[np.asarray(d, float)[rows], 1.0]; w = np.asarray(w, float)
    lo = linprog(w, A_eq=A, b_eq=b, bounds=[(0, None)] * M.shape[1], method="highs")
    hi = linprog(-w, A_eq=A, b_eq=b, bounds=[(0, None)] * M.shape[1], method="highs")
    return (lo.fun, -hi.fun) if lo.status == 0 and hi.status == 0 else None


def flip_some(d, w, tau, E, M=None):
    """min 1/2 ||E (M P - d)||_1 over joint laws P with w.P <= tau: the cheapest stealthy edit after which some rule decides w.x <= tau.
    The edited data M P lie in D by construction. Returns (cost, delta)."""
    M = np.array(G.M if M is None else M, float); n, m = M.shape[1], E.shape[0]; d = np.asarray(d, float); EM = E @ M; Ed = E @ d
    c = np.r_[np.zeros(n), 0.5 * np.ones(m)]
    Aub = np.vstack([np.c_[EM, -np.eye(m)], np.c_[-EM, -np.eye(m)], np.r_[np.asarray(w, float), np.zeros(m)][None]])
    bub = np.r_[Ed, -Ed, tau]
    for meth in ("highs-ds", "highs"):                       # dual simplex gives a vertex; interior point as a fallback
        r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=np.r_[np.ones(n), np.zeros(m)][None], b_eq=[1.0], bounds=[(0, None)] * (n + m), method=meth)
        if r.status == 0: return r.fun, M @ r.x[:n] - d
    return None, None


def dual_vertices(w, M=None):
    """vertices (y, y0) of {M^T y + y0 1 >= w}: U(d) = min over them of y.d + y0 (LP duality; the dual is pointed because the vertex images
    affinely span the data space). Exhaustive: every vertex is the solution of rank-many tight constraints."""
    M = np.array(G.M if M is None else M, float); K = np.c_[M.T, np.ones(M.shape[1])]; w = np.asarray(w, float); k = K.shape[1]; out = {}
    for rows in itertools.combinations(range(K.shape[0]), k):
        A = K[list(rows)]
        if abs(np.linalg.det(A)) < 1e-9: continue
        z = np.linalg.solve(A, w[list(rows)])
        if np.all(K @ z >= w - 1e-9): out[tuple(np.round(z, 9))] = z
    return list(out.values())


def flip_all(d, w, tau, E, DV, M=None):
    """cheapest stealthy edit after which every rule decides w.x <= tau (U(d') <= tau): {U <= tau} is the union over dual vertices of the
    half-spaces y.d' + y0 <= tau, so the cost is the minimum of one LP per vertex. Returns (cost, delta)."""
    M = np.array(G.M if M is None else M, float); n, m = M.shape[1], E.shape[0]; d = np.asarray(d, float); EM = E @ M; Ed = E @ d; best = (None, None)
    for z in DV:
        y, y0 = z[:-1], z[-1]
        Aub = np.vstack([np.c_[EM, -np.eye(m)], np.c_[-EM, -np.eye(m)], np.r_[y @ M, np.zeros(m)][None]]); bub = np.r_[Ed, -Ed, tau - y0]
        r = linprog(np.r_[np.zeros(n), 0.5 * np.ones(m)], A_ub=Aub, b_ub=bub, A_eq=np.r_[np.ones(n), np.zeros(m)][None], b_eq=[1.0],
                    bounds=[(0, None)] * (n + m), method="highs")
        if r.status == 0 and (best[0] is None or r.fun < best[0] - 1e-12): best = (r.fun, M @ r.x[:n] - d)
    return best


def along_some(d, c, w, tau, M=None):
    """first t >= 0 with L(d + t c) <= tau and d + t c in D (one LP in (t, P)); None if never"""
    M = np.array(G.M if M is None else M, float); n = M.shape[1]; d = np.asarray(d, float)
    A = np.c_[M, -np.asarray(c, float)[:, None]]; Aeq = np.vstack([A, np.r_[np.ones(n), 0.0][None]]); beq = np.r_[d, 1.0]
    r = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.r_[np.asarray(w, float), 0.0][None], b_ub=[tau], A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (n + 1), method="highs")
    return r.x[-1] if r.status == 0 else None


def tmax_in_D(d, c, M=None):
    M = np.array(G.M if M is None else M, float); n = M.shape[1]; d = np.asarray(d, float)
    Aeq = np.vstack([np.c_[M, -np.asarray(c, float)[:, None]], np.r_[np.ones(n), 0.0][None]])
    r = linprog(np.r_[np.zeros(n), -1.0], A_eq=Aeq, b_eq=np.r_[d, 1.0], bounds=[(0, None)] * n + [(0, 10.0)], method="highs")
    return r.x[-1] if r.status == 0 else 0.0


def along_all(d, c, w, tau, DV, M=None):
    """first t >= 0 with U(d + t c) <= tau, d + t c in D. U is concave along the line, so {U <= tau} on [0, tmax] is an interval
    (t*, tmax]; t* = min over dual vertices of the crossing of the affine function y.(d + t c) + y0 (if inside [0, tmax])."""
    tm = tmax_in_D(d, c, M); M_ = np.array(G.M if M is None else M, float); best = None
    for z in DV:
        y, y0 = z[:-1], z[-1]; a0 = y @ np.asarray(d, float) + y0; a1 = y @ np.asarray(c, float)
        if a0 <= tau: return 0.0
        if a1 < 0:
            t = (tau - a0) / a1
            if t <= tm + 1e-12 and (best is None or t < best): best = t
    return best


def along_rule(d, c, w, tau, rule, tmax, steps=60):
    """first t (bisection after a grid scan) at which w.rule(d + t c) <= tau, for a rule given as a function of the data"""
    f = lambda t: float(np.asarray(w, float) @ np.asarray(rule(np.asarray(d, float) + t * np.asarray(c, float)), float)) - tau
    grid = np.linspace(0, tmax, 41); prev = 0.0
    for t in grid[1:]:
        if f(t) <= 0:
            lo, hi = prev, t
            for _ in range(steps):
                mid = (lo + hi) / 2; lo, hi = (lo, mid) if f(mid) <= 0 else (mid, hi)
            return hi
        prev = t
    return None


# ------------------------------------------------------------------ placement: which circuits stay stealthy at a face
def tangent_count(V, F, SU):
    """number of signed circuits c with h.c >= 0 for every facet h of D containing the face with vertex set V (feasible directions at
    its relative interior)"""
    act = [np.array([float(x) for x in f[0]]) for f in F if set(V) <= set(f[2])]
    return sum(1 for c in SU if all(h @ c >= -1e-12 for h in act))


def closure(Sset, F):
    """the smallest face of D containing pi(conv S): the intersection of the facets that contain S (all 16 readings if none)"""
    out = set(range(16))
    for f in F:
        if set(Sset) <= set(f[2]): out &= set(f[2])
    return frozenset(out)


# ------------------------------------------------------------------ information states over time, and removal
class Stream:
    """batches of units; each unit is one joint reading (a0, a1, b0, b1) reported in the contexts observed at that time.
    counts[b][k] = 2 x 2 integer table of batch b in context k (None if k unobserved in batch b)"""
    def __init__(self): self.counts = []

    def add(self, readings, ctx):
        tabs = {}
        for k in ctx:
            i, j = G.CTX[k]; t = [[0, 0], [0, 0]]
            for r in readings: t[r[i]][r[j]] += 1
            tabs[k] = t
        self.counts.append(tabs); return len(self.counts) - 1


def pooled(stream, B, C):
    """the object of the information state (B, C): integer pooled tables for the contexts in C (only batches in B)"""
    out = {}
    for k in C:
        t = [[0, 0], [0, 0]]
        for b in B:
            if k in stream.counts[b]:
                for a in (0, 1):
                    for c in (0, 1): t[a][c] += stream.counts[b][k][a][c]
        out[k] = t
    return out


def remove(stream, obj, B, Bsmall, Csmall):
    """the removal map from the object of (B, C) to that of (Bsmall, Csmall), Bsmall <= B, Csmall <= C: subtract the removed batches'
    counts, then forget the removed contexts. It uses only obj and the removed batches (a map of objects, not a recomputation)."""
    out = {}
    for k in Csmall:
        t = [r[:] for r in obj[k]]
        for b in set(B) - set(Bsmall):
            if k in stream.counts[b]:
                for a in (0, 1):
                    for c in (0, 1): t[a][c] -= stream.counts[b][k][a][c]
        out[k] = t
    return out


def data_of(obj):
    """data coordinates (CHSH) from pooled tables, as floats, and the observed rows. Singles are read from the first observed context
    containing each variable (consistent when every unit is reported in all contexts of its batch)."""
    d = np.zeros(8); have = set()
    for k, t in obj.items():
        n = sum(map(sum, t))
        if n == 0: continue
        i, j = G.CTX[k]; d[k] = t[0][0] / n; d[4 + i] = (t[0][0] + t[0][1]) / n; d[6 + (j - 2)] = (t[0][0] + t[1][0]) / n; have.add(k)
    return d, chsh_rows(sorted(have))


def maxent_rows(d, rows, iters=4000, tol=1e-12):
    """maximum-entropy joint law matching the observed rows (IPF over the observed 00 entries and singles via their pair tables)"""
    ctx = [k for k in range(4) if k in rows]; T = {}
    for k in ctx:
        i, j = G.CTX[k]; a, b, t = d[4 + i], d[6 + (j - 2)], d[k]; T[k] = [[t, a - t], [b - t, 1 - a - b + t]]
    x = np.full(16, 1 / 16); idx = np.array(G.CELLS)
    for _ in range(iters):
        err = 0.0
        for k in ctx:
            i, j = G.CTX[k]
            for sa in (0, 1):
                for sb in (0, 1):
                    msk = (idx[:, i] == sa) & (idx[:, j] == sb); cur = x[msk].sum(); tg = max(T[k][sa][sb], 0.0); err = max(err, abs(cur - tg))
                    if cur > 0: x[msk] *= tg / cur
        if err < tol: break
    return x


# ------------------------------------------------------------------ one poisoned batch, how long its flip lasts, and attribution
def poison_mix(d, w, tau, DV, M=None):
    """the smallest mixing weight q such that some joint law R gives U((1 - q) d + q M R) <= tau (every rule flips): one LP per dual
    vertex in (q, R') with R' = q R >= 0, sum R' = q. Returns (q, R)."""
    M = np.array(G.M if M is None else M, float); n = M.shape[1]; d = np.asarray(d, float); best = (None, None)
    for z in DV:
        y, y0 = z[:-1], z[-1]
        # y.((1-q) d + M R') + y0 <= tau  <=>  -q (y.d) + (y M).R' <= tau - y0 - y.d
        Aub = np.r_[-(y @ d), y @ M][None]; bub = [tau - y0 - y @ d]
        r = linprog(np.r_[1.0, np.zeros(n)], A_ub=Aub, b_ub=bub, A_eq=np.r_[-1.0, np.ones(n)][None], b_eq=[0.0], bounds=[(0, 1)] + [(0, None)] * n, method="highs")
        if r.status == 0 and r.x[0] > 1e-12 and (best[0] is None or r.x[0] < best[0]): best = (r.x[0], r.x[1:] / r.x[0])
    return best


def units_from_law(R, k):
    """k units with reading counts as close as possible to k R (largest remainder)"""
    raw = np.asarray(R, float) * k; base = np.floor(raw).astype(int); rem = k - base.sum()
    for i in np.argsort(-(raw - base))[:rem]: base[i] += 1
    return [G.CELLS[i] for i in range(16) for _ in range(base[i])]


def first_order(f, d, dd, h=1e-6):
    """first-order estimate of f(d + dd) - f(d) by a central-difference gradient (an approximation, reported as such)"""
    d = np.asarray(d, float); g = np.array([(f(d + h * e) - f(d - h * e)) / (2 * h) for e in np.eye(len(d))]); return float(g @ dd)


def circuits_upto(E, s):
    """circuits of range(E) with support at most s (direct search over supports; for n-cycles too large for circuits())"""
    m, dim = E.shape; out = []
    for size in range(2, s + 1):
        for S in itertools.combinations(range(m), size):
            if any(set(t) <= set(S) for t, _ in out): continue
            Z = [r for r in range(m) if r not in S]; A = E[Z]
            if dim - np.linalg.matrix_rank(A) != 1: continue
            v = np.linalg.svd(A)[2][-1] if len(Z) else None
            if v is None: continue
            ent = E @ v
            if np.all(np.abs(ent[list(S)]) > 1e-9): out.append((list(S), v / np.abs(v).max()))
    return out


def conformal_count(E, delta, SU, d, w, tau, M=None):
    """number of circuits used by a vertex solution of the cheapest-flip LP restricted to circuits conformal to delta (an upper bound
    on the number needed for this optimum). Returns (count, cost)."""
    M = np.array(G.M if M is None else M, float); ent = E @ delta; n = M.shape[1]; d = np.asarray(d, float)
    cand = [c for c in SU if np.all((np.abs(E @ c) <= 1e-9) | ((np.abs(ent) > 1e-9) & (np.sign(E @ c) == np.sign(ent))))]
    if not cand: return None, None
    Cm = np.array(cand).T                                      # dim x K, each column has cost 1
    K = Cm.shape[1]
    Aeq = np.vstack([np.c_[-Cm, M], np.r_[np.zeros(K), np.ones(n)][None]]); beq = np.r_[d, 1.0]
    r = linprog(np.r_[np.ones(K), np.zeros(n)], A_ub=np.r_[np.zeros(K), np.asarray(w, float)][None], b_ub=[tau], A_eq=Aeq, b_eq=beq,
                bounds=[(0, None)] * (K + n), method="highs-ds")
    if r.status != 0: return None, None
    return int((r.x[:K] > 1e-9).sum()), r.fun


# ------------------------------------------------------------------ cell-level attribution: an observed change as a sum of circuits
def decompose(E, delta, C, tol=1e-9):
    """a conformal decomposition delta = sum_k lam_k c_k into signed circuits c_k (unit cost), lam_k > 0. Every consistent edit has one
    (Rockafellar 1969), and then sum_k lam_k = cost(delta) because nothing cancels. A vertex of the linear program below has at most
    rank-many nonzero weights; the objective prefers small supports, so the decomposition reads in the simplest edits available.
    Returns [(index into C, sign, lam)], or None if delta is not in range(E)."""
    ent = E @ np.asarray(delta, float); cand = []
    for i, (sup, v) in enumerate(C):
        for sg in (1, -1):
            e = sg * (E @ v) / cost(E, v)
            if np.all((np.abs(e) <= tol) | ((np.abs(ent) > tol) & (np.sign(e) == np.sign(ent)))): cand.append((i, sg, e, len(sup)))
    if not cand: return None
    A = np.array([c[2] for c in cand]).T
    r = linprog(np.array([c[3] for c in cand], float), A_eq=A, b_eq=ent, bounds=[(0, None)] * len(cand), method="highs-ds")
    if r.status != 0: return None
    return [(cand[j][0], cand[j][1], float(r.x[j])) for j in range(len(cand)) if r.x[j] > tol]


def circuit_type(E, v, ncontexts=4):
    """'swap' (4 cells in one table), 'margin shift' (2 + 2 cells in two tables), or 'support k'"""
    ent = E @ v; sup = [r for r in range(len(ent)) if abs(ent[r]) > 1e-9]; per = collections.Counter(r // 4 for r in sup)
    if len(sup) == 4: return "swap" if len(per) == 1 else "margin shift"
    return f"support {len(sup)}"


def coordination(E, delta, C, tol=1e-9):
    """the least share of cost(delta) that any conformal decomposition must put on circuits with more than 4 cells (a linear program;
    an invariant of delta, unlike the decomposition itself). 0 means the change is a conformal sum of 4-cell edits."""
    ent = E @ np.asarray(delta, float); cand = []
    for i, (sup, v) in enumerate(C):
        for sg in (1, -1):
            e = sg * (E @ v) / cost(E, v)
            if np.all((np.abs(e) <= tol) | ((np.abs(ent) > tol) & (np.sign(e) == np.sign(ent)))): cand.append((e, len(sup) > 4))
    A = np.array([c[0] for c in cand]).T
    r = linprog(np.array([1.0 if c[1] else 0.0 for c in cand]), A_eq=A, b_eq=ent, bounds=[(0, None)] * len(cand), method="highs")
    return r.fun / cost(E, delta) if r.status == 0 else None


# ------------------------------------------------------------------ the geometry of poisoning (section 12.21)
def dual_lower(w, M=None):
    """vertices (y, y0) with L(d) = max over them of y.d + y0 (the dual of the minimisation, by symmetry)"""
    return [-z for z in dual_vertices(-np.asarray(w, float), M)]


def lipschitz(DV, SU):
    """the Lipschitz constant of d -> min (or max) over DV of y.d + y0 in the cost: the largest |y.c| over unit-cost circuits, because the
    unit cost ball's extreme points are the unit circuits (Rockafellar)"""
    return max(abs(z[:-1] @ c) for z in DV for c in SU)


def ctx_radius(d, F, SU, which=8):
    """the least cost of an edit that reaches a facet of D with `which` vertices (8: a Clauser-Horne facet, the contextuality boundary):
    min over those facets of slack / (largest rate at which a unit circuit lowers the slack). Closed form; no optimisation over edits."""
    best = np.inf
    for c, c0, V in F:
        if len(V) != which: continue
        h = np.array([float(x) for x in c]); sl = h @ np.asarray(d, float) + float(c0); rate = max(-(h @ cc) for cc in SU)
        if rate > 1e-12: best = min(best, sl / rate)
    return best


def contextual_fraction(d, E):
    """1 - the largest mass of a sub-normalised joint law whose tables fit under the data's tables (Abramsky, Barbosa and Mansfield)"""
    M = np.array(G.M, float); ent = E @ np.asarray(d, float) + _e0_chsh()
    A = E @ M + np.outer(_e0_chsh(), np.ones(16))                         # entries of an unnormalised joint b: E (M b) + e0 (1.b)
    r = linprog(-np.ones(16), A_ub=A, b_ub=ent, bounds=[(0, None)] * 16, method="highs")
    return 1 + r.fun


def _e0_chsh():
    return np.array([float(x) for x in _tab_chsh([Fr(0)] * 8)])


def fibre_shadow(d, u, v, ndir=120, M=None):
    """the projection of the fibre over d onto the two never-observed quantities (u.x, v.x): its vertices, by support linear programs
    in ndir directions (ordered by angle). None if the fibre is empty."""
    M = np.array(G.M if M is None else M, float); n = M.shape[1]; A = np.vstack([M, np.ones(n)]); b = np.r_[np.asarray(d, float), 1.0]; pts = []
    for th in np.linspace(0, 2 * np.pi, ndir, endpoint=False):
        r = linprog(-(np.cos(th) * np.asarray(u, float) + np.sin(th) * np.asarray(v, float)), A_eq=A, b_eq=b, bounds=[(0, None)] * n, method="highs-ds")
        if r.status != 0: return None
        p = (round(float(u @ r.x), 10), round(float(v @ r.x), 10))
        if not pts or p != pts[-1]: pts.append(p)
    out = []
    for p in pts:
        if p not in out: out.append(p)
    return out


def vertex_rule(d, cvec, M=None):
    """a rule that picks a vertex: the joint law minimising a fixed linear objective over the fibre (it jumps across chamber walls)"""
    M = np.array(G.M if M is None else M, float); n = M.shape[1]
    r = linprog(np.asarray(cvec, float), A_eq=np.vstack([M, np.ones(n)]), b_eq=np.r_[np.asarray(d, float), 1.0], bounds=[(0, None)] * n, method="highs-ds")
    return r.x if r.status == 0 else None


def line_cross(d, direction, F, which=8):
    """first t > 0 at which d + t direction reaches a facet with `which` vertices (exact from the facet slacks); inf if never"""
    best = np.inf
    for c, c0, V in F:
        if len(V) != which: continue
        h = np.array([float(x) for x in c]); sl = h @ np.asarray(d, float) + float(c0); rt = h @ np.asarray(direction, float)
        if rt < -1e-15: best = min(best, max(sl, 0.0) / -rt)
    return best


def pr_box(signs):
    """the PR box with correlation signs (one per context of margsec.CTX, product -1): 00 entry 1/2 where correlated, 0 where not"""
    return np.array([0.5 if s_ == 1 else 0.0 for s_ in signs] + [0.5] * 4)
