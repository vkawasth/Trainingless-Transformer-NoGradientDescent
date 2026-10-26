"""Perturbing evidence and hypotheses: the connexion budget, the data ball, and the level-two fibre.

Level one (binary variables; windows of any arity; a variable may appear in any number of windows). Keep one copy of each variable per
window (Contextuality-by-Default). For every pair of copies of the same variable, a connexion with direct influence s = TV of the two copies'
marginals. The hypothesis "connexions maximally coupled" says copies disagree with probability exactly s; a budget kappa >= 0 per
connexion weakens it to "disagree with probability at most s + kappa".

  CopySystem(windows)                 copies, atoms, connexion pairs, window cell rows
  support(sys, kappa=None, r=None, data_radius=0.0, duals=False)
        largest mass lam of a sub-measure mu on the copies with every window marginal <= its table and every connexion's disagreement
        <= s + kappa. Give kappa (a vector, one per connexion) or a total budget r (free allocation). With data_radius = rho > 0 the
        tables may move: every window's table may be replaced by any table within TV rho of it, and s is recomputed from the moved
        tables. This is exact, as the maximum over orientations of one linear program each: |p - q| >= sigma (p - q) for sigma = +-1,
        with equality for the right sign.
        duals=True also returns the shadow prices: d lam / d kappa_q for each connexion, and d lam / d r.
  profile(sys, rs, which=None)        lam(r) for a total budget (which=None) or for one connexion's budget alone (which = index)
  agreement_polygon(sys, i, j, data_radius=0.0, n_dir=72)
        the projection onto (P(copies of connexion i agree), P(copies of connexion j agree)) of the full couplings of the copies (mass
        1, window tables exact, or within the data ball); i or j may be a list of connexions (summed agreement). The CbD point is
        (1 - s_i, 1 - s_j) (summed likewise); the system is noncontextual iff some coupling reaches it in every connexion at once
  min_contextuality(sys, data_radius) the smallest CbD degree among systems whose tables lie within TV rho of the observed ones (exact:
        one LP per orientation). It equals c at rho = 0

Level two (two latent types, three binary variables; the closed form of pce.ladder.level2_m3):
  track_fibre(P0, P1, n)              the fibre along the segment P(t) = (1 - t) P0 + t P1: rung, weights, susceptibilities (real and
                                      imaginary parts), Delta, Det, negativity; wall crossings where Delta, Det or a box coordinate changes sign
  wall_distance(P, n_dir, seed)       an upper bound on the TV distance from P to the nearest wall (rung change), by bisection along random
                                      directions inside the simplex
  ball_rungs(P, r, n, seed)           the share of tables in each rung, among tables sampled uniformly from the TV ball of radius r around P
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


class CopySystem:
    def __init__(self, windows):
        self.windows = windows; self.copies = []; self.cells = []; occ = {}
        for wi, w in enumerate(windows):
            T = np.asarray(w.table, float); assert all(d == 2 for d in T.shape), "binary variables only"
            idx = []
            for v in w.vars:
                self.copies.append((v, wi)); idx.append(len(self.copies) - 1); occ.setdefault(v, []).append(len(self.copies) - 1)
            self.cells.append((idx, T))
        n = len(self.copies); self.n = n
        self.atoms = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int8)
        self.pairs = [(v, i, j) for v, cs in occ.items() for i, j in itertools.combinations(cs, 2)]
        self.dis = np.array([(self.atoms[:, i] != self.atoms[:, j]).astype(float) for _, i, j in self.pairs]).reshape(len(self.pairs), -1)
        self.win_rows = []
        for idx, T in self.cells:
            rows = []
            for cell in itertools.product((0, 1), repeat=len(idx)):
                rows.append((np.all(self.atoms[:, idx] == np.array(cell), axis=1).astype(float), cell))
            self.win_rows.append(rows)

    def marg(self, c, tables=None):
        """P(copy c = 0), i.e. P(+1), from the observed (or given) tables"""
        v, wi = self.copies[c]; idx, T = self.cells[wi]; T = T if tables is None else tables[wi]
        k = idx.index(c); return float(np.asarray(T).sum(axis=tuple(a for a in range(T.ndim) if a != k))[0])

    def s(self):
        return np.array([abs(self.marg(i) - self.marg(j)) for _, i, j in self.pairs])

    def labels(self):
        return [f"{v}:{self.windows[self.copies[i][1]].vars}-{self.windows[self.copies[j][1]].vars}" for v, i, j in self.pairs]


def _lp(sys, kappa, r, rho, sigma, full_mass=False, objective=None):
    """variables: mu (na), kappa (np, if r is a budget), tables t'_w (cells per window, if rho > 0), with |t' - t| split as e+ - e-."""
    na = len(sys.atoms); npairs = len(sys.pairs); free_k = r is not None
    ncell = [len(rows) for rows in sys.win_rows]; nt = sum(ncell) if rho > 0 else 0
    nv = na + (npairs if free_k else 0) + 3 * nt          # t', e+, e-
    off_k = na; off_t = na + (npairs if free_k else 0)
    A, b, Ae, be = [], [], [], []
    def row(): return np.zeros(nv)
    # windows
    pos = 0; tcell = {}
    for w, rows in enumerate(sys.win_rows):
        T = sys.cells[w][1]
        for (ind, cell) in rows:
            R = row(); R[:na] = ind
            if rho > 0:
                k = off_t + pos; tcell[(w, cell)] = k
                if full_mass: R[k] = -1.0; Ae.append(R); be.append(0.0)
                else: R[k] = -1.0; A.append(R); b.append(0.0)
                E = row(); E[k] = 1.0; E[off_t + nt + pos] = -1.0; E[off_t + 2 * nt + pos] = 1.0; Ae.append(E); be.append(float(T[cell]))
                pos += 1
            else:
                if full_mass: Ae.append(R); be.append(float(T[cell]))
                else: A.append(R); b.append(float(T[cell]))
        if rho > 0:
            S = row()
            for (ind, cell) in rows: S[tcell[(w, cell)]] = 1.0
            Ae.append(S); be.append(1.0)
            D = row(); D[off_t + nt + pos - len(rows): off_t + nt + pos] = 0.5; D[off_t + 2 * nt + pos - len(rows): off_t + 2 * nt + pos] = 0.5
            A.append(D); b.append(rho)
    # connexions: mu(disagree) <= s + kappa  (s = sigma (p - q) of the moved tables when rho > 0)
    s0 = sys.s()
    for k, (v, i, j) in enumerate(sys.pairs):
        R = row(); R[:na] = sys.dis[k]; rhs = 0.0
        if free_k: R[off_k + k] = -1.0
        else: rhs += float(kappa[k]) if kappa is not None else 0.0
        if rho > 0:
            sg = sigma[k]
            for c, sgn in ((i, 1.0), (j, -1.0)):
                vv, wi = sys.copies[c]; idx, T = sys.cells[wi]; kk = idx.index(c)
                for (ind, cell) in sys.win_rows[wi]:
                    if cell[kk] == 0: R[tcell[(wi, cell)]] -= sg * sgn
        else:
            rhs += s0[k]
        A.append(R); b.append(rhs)
    if free_k:
        R = row(); R[off_k:off_k + npairs] = 1.0; A.append(R); b.append(float(r))
    if full_mass:
        R = row(); R[:na] = 1.0; Ae.append(R); be.append(1.0)
    c = np.zeros(nv)
    if objective is None: c[:na] = -1.0
    else: c[:na] = -objective
    res = linprog(c, A_ub=np.array(A) if A else None, b_ub=np.array(b) if b else None, A_eq=np.array(Ae) if Ae else None,
                  b_eq=np.array(be) if be else None, bounds=(0, None), method="highs")
    return res, (len(A), npairs, free_k)


def _orientations(sys, rho):
    return [None] if rho <= 0 else list(itertools.product((1.0, -1.0), repeat=len(sys.pairs)))


def support(sys, kappa=None, r=None, data_radius=0.0, duals=False):
    best = None
    for sg in _orientations(sys, data_radius):
        res, (nA, npairs, free_k) = _lp(sys, kappa, r, data_radius, sg)
        if res.status == 0 and (best is None or -res.fun > -best[0].fun + 1e-12): best = (res, nA, npairs, free_k)
    res, nA, npairs, free_k = best; lam = float(-res.fun)
    if not duals: return lam
    m = -res.ineqlin.marginals                              # d lam / d b
    nwin = nA - npairs - (1 if free_k else 0)
    out = {"lam": lam, "price_kappa": m[nwin:nwin + npairs].tolist()}
    if free_k: out["price_r"] = float(m[-1]); out["kappa"] = res.x[len(sys.atoms):len(sys.atoms) + npairs].tolist()
    return out


def profile(sys, rs, which=None, data_radius=0.0):
    out = []
    for r in rs:
        if which is None: out.append(support(sys, r=r, data_radius=data_radius))
        else:
            k = np.zeros(len(sys.pairs)); k[which] = r; out.append(support(sys, kappa=k, data_radius=data_radius))
    return np.array(out)


def agreement_polygon(sys, i, j, data_radius=0.0, n_dir=72):
    """support-function sampling of the projected set; returns the polygon's vertices (in order) or None if empty"""
    pts = []
    for th in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        gi = _agree(sys, i); gj = _agree(sys, j)
        obj = np.cos(th) * gi + np.sin(th) * gj
        best = _poly_lp(sys, obj, data_radius, None)              # all couplings with exact (or moved) tables; no connexion constraint
        if best is None: return None
        mu = best.x[:len(sys.atoms)]; pts.append((float(gi @ mu), float(gj @ mu)))
    return np.array(pts)


def _agree(sys, k):
    """agreement indicator over atoms: connexion k (an int) or the sum over several connexions (a list)"""
    ks = [k] if np.isscalar(k) else list(k)
    return sum(1 - sys.dis[q] for q in ks)


def _poly_lp(sys, obj, rho, sg):
    na = len(sys.atoms); ncell = sum(len(r) for r in sys.win_rows); nt = ncell if rho > 0 else 0; nv = na + 3 * nt
    Ae, be, A, b = [], [], [], []; pos = 0
    for w, rows in enumerate(sys.win_rows):
        T = sys.cells[w][1]; start = pos
        for (ind, cell) in rows:
            R = np.zeros(nv); R[:na] = ind
            if rho > 0:
                R[na + pos] = -1.0; Ae.append(R); be.append(0.0)
                E = np.zeros(nv); E[na + pos] = 1.0; E[na + nt + pos] = -1.0; E[na + 2 * nt + pos] = 1.0; Ae.append(E); be.append(float(T[cell])); pos += 1
            else: Ae.append(R); be.append(float(T[cell]))
        if rho > 0:
            D = np.zeros(nv); D[na + nt + start:na + nt + pos] = 0.5; D[na + 2 * nt + start:na + 2 * nt + pos] = 0.5; A.append(D); b.append(rho)
    R = np.zeros(nv); R[:na] = 1.0; Ae.append(R); be.append(1.0)
    c = np.zeros(nv); c[:na] = -obj
    res = linprog(c, A_ub=np.array(A) if A else None, b_ub=np.array(b) if b else None, A_eq=np.array(Ae), b_eq=np.array(be), bounds=(0, None), method="highs")
    return res if res.status == 0 else None


def min_contextuality(sys, data_radius):
    """min over moved tables t' (TV <= rho per window) of c(t') = sum_q (1 - s_q(t')) - max_mu sum_q P_mu(agree_q), full couplings mu."""
    best = np.inf; na = len(sys.atoms)
    for sg in _orientations(sys, data_radius) if data_radius > 0 else [None]:
        # objective: minimise sum_q (1 - sigma_q (p_q - q_q)) - sum_q agree_q(mu)  (linear in mu, t')
        if data_radius <= 0:
            res = _poly_lp(sys, (1 - sys.dis).sum(axis=0), 0.0, None)
            val = float(np.sum(1 - sys.s()) - (-res.fun)); best = min(best, val); continue
        ncell = sum(len(r) for r in sys.win_rows); nt = ncell; nv = na + 3 * nt
        obj_t = np.zeros(nv); pos = 0; tcell = {}
        for w, rows in enumerate(sys.win_rows):
            for (ind, cell) in rows: tcell[(w, cell)] = na + pos; pos += 1
        for k, (v, i, j) in enumerate(sys.pairs):
            for c_, sgn in ((i, 1.0), (j, -1.0)):
                vv, wi = sys.copies[c_]; idx, T = sys.cells[wi]; kk = idx.index(c_)
                for (ind, cell) in sys.win_rows[wi]:
                    if cell[kk] == 0: obj_t[tcell[(wi, cell)]] -= sg[k] * sgn          # - sigma (p - q)
        # constraint sigma (p - q) >= 0 is implied by choosing the best orientation; enforce it so s = sigma (p - q) is valid
        res = _poly_lp_obj(sys, obj_t, data_radius, sg)
        if res is not None: best = min(best, len(sys.pairs) + float(res.fun))
    return max(0.0, best)


def _poly_lp_obj(sys, obj_t, rho, sg):
    na = len(sys.atoms); ncell = sum(len(r) for r in sys.win_rows); nt = ncell; nv = na + 3 * nt
    Ae, be, A, b = [], [], [], []; pos = 0; tcell = {}
    for w, rows in enumerate(sys.win_rows):
        T = sys.cells[w][1]; start = pos
        for (ind, cell) in rows:
            tcell[(w, cell)] = na + pos
            R = np.zeros(nv); R[:na] = ind; R[na + pos] = -1.0; Ae.append(R); be.append(0.0)
            E = np.zeros(nv); E[na + pos] = 1.0; E[na + nt + pos] = -1.0; E[na + 2 * nt + pos] = 1.0; Ae.append(E); be.append(float(T[cell])); pos += 1
        D = np.zeros(nv); D[na + nt + start:na + nt + pos] = 0.5; D[na + 2 * nt + start:na + 2 * nt + pos] = 0.5; A.append(D); b.append(rho)
    for k, (v, i, j) in enumerate(sys.pairs):                 # sigma_q (p_q - q_q) >= 0
        R = np.zeros(nv)
        for c_, sgn in ((i, 1.0), (j, -1.0)):
            vv, wi = sys.copies[c_]; idx, T = sys.cells[wi]; kk = idx.index(c_)
            for (ind, cell) in sys.win_rows[wi]:
                if cell[kk] == 0: R[tcell[(wi, cell)]] -= sg[k] * sgn
        A.append(R); b.append(0.0)
    R = np.zeros(nv); R[:na] = 1.0; Ae.append(R); be.append(1.0)
    c = obj_t.copy(); c[:na] -= (1 - sys.dis).sum(axis=0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=np.array(Ae), b_eq=np.array(be), bounds=(0, None), method="highs")
    return res if res.status == 0 else None


# ---------------- level two ----------------
def _l2(P):
    from .ladder import level2_m3
    return level2_m3(np.asarray(P, float).reshape(2, 2, 2))


def _rung_code(r):
    return {"R>=0": 0, "R>=0 off box": 1, "R": 2, "C": 3, "none": 4}[r]


def track_fibre(P0, P1, n=401):
    P0 = np.asarray(P0, float).reshape(8); P1 = np.asarray(P1, float).reshape(8); rows = []
    for t in np.linspace(0, 1, n):
        P = (1 - t) * P0 + t * P1; o = _l2(P)
        if o["rung"] in ("R>=0", "R>=0 off box", "R"):
            w = np.array(o["w"], float); a = np.array(o["a"], float); wim = 0.0; aim = 0.0
        elif o["rung"] == "C":
            w = np.array([complex(x) for x in o["w"]]); wim = float(np.abs(w.imag).max()); w = w.real; a = np.full((2, 3), np.nan); aim = o["imag"]
        else:
            w = np.full(2, np.nan); a = np.full((2, 3), np.nan); wim = np.nan; aim = np.nan
        rows.append({"t": float(t), "rung": o["rung"], "code": _rung_code(o["rung"]), "Delta": o["Delta"], "Det": o["Det"], "w1": float(w[0]),
                     "w_imag": wim, "a": a.tolist(), "imag": aim, "neg2": o["negativity2"]})
    walls = [rows[k]["t"] for k in range(1, n) if rows[k]["code"] != rows[k - 1]["code"]]
    return rows, walls


def _rand_dir(rng, P, ):
    d = rng.normal(size=8); d -= d.mean(); return d / (0.5 * np.abs(d).sum())        # unit TV direction, sum 0


def wall_distance(P, n_dir=400, seed=0, tol=1e-6):
    P = np.asarray(P, float).reshape(8); rng = np.random.default_rng(seed); code0 = _rung_code(_l2(P)["rung"]); best = np.inf
    for _ in range(n_dir):
        d = _rand_dir(rng, P); neg = d < 0
        tmax = np.min(P[neg] / -d[neg]) if neg.any() else 1.0                       # stay in the simplex
        hi = min(tmax, best)
        if _rung_code(_l2(P + hi * d)["rung"]) == code0: continue
        lo = 0.0
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if _rung_code(_l2(P + mid * d)["rung"]) == code0: lo = mid
            else: hi = mid
        best = min(best, hi)
    return best


def ball_rungs(P, r, n=2000, seed=0):
    P = np.asarray(P, float).reshape(8); rng = np.random.default_rng(seed); counts = np.zeros(5); m = 0
    while m < n:
        d = _rand_dir(rng, P); rad = r * rng.uniform() ** (1 / 7)                      # uniform in the 7-dimensional TV ball (by radius law)
        Q = P + rad * d
        if np.any(Q < 0): continue
        counts[_rung_code(_l2(Q)["rung"])] += 1; m += 1
    return dict(zip(["R>=0", "R>=0 off box", "R", "C", "none"], (counts / n).tolist()))
