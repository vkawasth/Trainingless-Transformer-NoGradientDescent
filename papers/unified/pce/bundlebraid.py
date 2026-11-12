"""The bundle braid: the decision maker's sources as strands of a bundle (companion note, section 'What braids and knots carry'). New
module; it modifies nothing.

Strands. For each persistent source x (S1..S5, R1, R2 of pce/watchtoy) and each step t:
   s_x(t) = residual of all cells except x   (suspicion height: low = removing x explains the most; inspections are always kept)
   v_x(t) = midpoint of the least-incoherent range of B without x   (the decision value if x were distrusted)
   z_x(t) = s_x(t) + i v_x(t) + eps * exp(i (2 pi k_x / 7 + 0.3))     (k_x = index of x; the fixed ring is the bundle's cross-section)
Between steps the points move on straight segments. Without the ring, consistent snapshots put strands on top of one another (all s_x
are 0 and the v_x can coincide), so the braid would not be defined; the ring is a declared regularisation, and every result is checked
for stability in eps.
Braid. Project onto the direction exp(i phi) (phi = 0: the suspicion axis). Order strands by the projection, lowest first. A crossing of
the strands in positions k, k+1 at time tau is the letter +(k+1) if the strand coming from position k (rising) is in front, i.e. has
the larger coordinate along i exp(i phi), else -(k+1). The projection is generic if no two crossings share a time and no crossing is a
tangency; the module reports both, and the smallest distance between strands along all segments (no collisions).
Prime suspect: the strand in the lowest position. With phi = 0 it changes exactly at letters +-1 (bottom crossings).
Closure invariants (the closure of the whole history; conjugation, i.e. a different starting order, does not change them):
   components                     cycles of the permutation
   burau_det(word, n)             |det(I - Burau_r(word))| at t = -1 divided by |1 + t + ... + t^{n-1}|, n odd: the determinant
                                  |Delta(-1)| of the closure (reduced Burau formula for the Alexander polynomial). A value other than
                                  0 and 1 shows the closure is not an unknot or unlink; 1 does not show it is; 0 may be a split link.
   alexander(word, n)             the one-variable Alexander polynomial, up to units, by the same formula (sympy)
   braid_pd(word, n)              PD code of the closure, for the Jones polynomial via pce.pdknot (counterclockwise from incoming under)
"""
from __future__ import annotations
import itertools
import numpy as np
import sympy as sp
from . import watchtoy as WT, bridgetoy as BT, tamper as T, braidknot as BK

N = 5; BASE = WT.BASE; FCTX = {"F1": (2, 4), "F2": (1, 3), "F3": (2, 3), "F4": (2, 3)}


def extra_history(name, W):
    """two new histories on the watch-toy world: pump_retreat (S2, S3, R1, R2 move in lockstep to the false law by t = 5, hold, return
    to the truth by t = 13) and cascade (R2 lies from t = 3; F1, F2 planted from t = 6; F3, F4 duplicate R2 from t = 8; the honest S2
    is recruited and reports the false law on its context from t = 11); inspection of (X3, X5) from t = 15 as in pce/watchtoy"""
    P, gB, p1 = W["P"], W["gB"], W["p1"]; out = []
    for t in range(WT.STEPS + 1):
        rep = {s: BT.table(P, WT.CTXS[s]) for s in BASE}
        if name == "pump_retreat":
            lam = min(t / 5, 1) if t <= 8 else max(0, 1 - (t - 8) / 5); pl = (1 - lam) * P + lam * p1
            for s in ("S2", "S3", "R1", "R2"): rep[s] = BT.table(pl, WT.CTXS[s])
        if name == "cascade":
            if t >= 3: rep["R2"] = BT.pair_table(P, (2, 3), 0.35)
            if t >= 6: rep["F1"] = BT.table(p1, (2, 4)); rep["F2"] = BT.table(p1, (1, 3))
            if t >= 8: rep["F3"] = rep["F4"] = BT.pair_table(P, (2, 3), 0.35)
            if t >= 11: rep["S2"] = BT.table(p1, (1, 2))
        cells = [(s, WT.CTXS.get(s, FCTX.get(s)), rep[s]) for s in rep]
        if t >= 15: cells.append(("Q", (2, 4), BT.table(P, (2, 4))))
        out.append((cells, gB @ P))
    return out


def history(name, W):
    return WT.history(name, W) if name in WT.SCENARIOS else extra_history(name, W)


def _mid(cells, g):
    res = T.residual(N, cells); I = T.interval(N, cells, g, tol=res + 1e-9 if res > 1e-9 else 0.0); return 0.5 * (I[0] + I[1])


def explanations(cells, k=3):
    """minimal explanations of size <= k among the reports; inspections (Q) are trusted and never removed"""
    Q = [c for c in cells if c[0].startswith("Q")]; R = [c for c in cells if not c[0].startswith("Q")]; out = []
    for r in range(k + 1):
        for H in itertools.combinations(range(len(R)), r):
            if any(set(h) <= set(H) for h in out): continue
            if T.feasible(N, [c for i, c in enumerate(R) if i not in H] + Q): out.append(H)
    return [tuple(R[i][0] for i in H) for H in out]


def features(H, g, noise=0.0, seed=0):
    """per step: s, v (arrays over BASE), smallest explanation size (k+1 if none of size <= 3), number of minimal explanations, residual"""
    rng = np.random.default_rng(seed); S, V, rows = [], [], []
    for cells, tv in H:
        if noise:
            cells = [(c[0], c[1], _jitter(c[2], noise, rng)) if not c[0].startswith("Q") else c for c in cells]
        S.append([T.residual(N, [c for c in cells if c[0] != x]) for x in BASE])
        V.append([_mid([c for c in cells if c[0] != x], g) for x in BASE])
        ex = explanations(cells)
        rows.append(dict(min_expl=min((len(e) for e in ex), default=4), n_expl=len([e for e in ex if e]),
                         residual=T.residual(N, [c for c in cells if not c[0].startswith("Q")]), naive=_mid(cells, g), truth=tv,
                         expl=sorted(ex)))
    return np.array(S), np.array(V), rows


def _jitter(tab, eta, rng):
    keys = sorted(tab); v = np.array([tab[k] for k in keys]) * np.exp(eta * rng.standard_normal(len(keys))); v /= v.sum()
    return dict(zip(keys, v))


def points(S, V, eps=0.02):
    ring = eps * np.exp(1j * (2 * np.pi * np.arange(S.shape[1]) / S.shape[1] + 0.3))
    return S + 1j * V + ring[None, :]


def braid(Z, phi=0.0):
    """exact braid word of the piecewise-linear strands Z[t, x] projected on exp(i phi); returns word, crossing log, genericity, min distance"""
    R = Z * np.exp(-1j * phi); p, q = R.real, R.imag; n = Z.shape[1]
    order = list(np.argsort(p[0], kind="stable")); word, log = [], []; generic = True; dmin = np.inf
    for t in range(len(Z) - 1):
        for a, b in itertools.combinations(range(n), 2):                # smallest distance on the segment (quadratic in u)
            d0 = Z[t, a] - Z[t, b]; d1 = (Z[t + 1, a] - Z[t + 1, b]) - d0
            u = 0.0 if abs(d1) < 1e-15 else float(np.clip(-np.real(np.conj(d0) * d1) / abs(d1) ** 2, 0, 1)); dmin = min(dmin, abs(d0 + u * d1))
        ev = []
        for a, b in itertools.combinations(range(n), 2):
            f0 = p[t, a] - p[t, b]; f1 = p[t + 1, a] - p[t + 1, b]
            if f0 == 0 or f1 == 0:
                if f0 != f1: generic = False
                continue
            if f0 * f1 < 0:
                u = f0 / (f0 - f1); ev.append((u, a, b, (q[t, a] + u * (q[t + 1, a] - q[t, a])) - (q[t, b] + u * (q[t + 1, b] - q[t, b]))))
        ev.sort()
        for e1, e2 in zip(ev, ev[1:]):
            if abs(e1[0] - e2[0]) < 1e-9 and {e1[1], e1[2]} & {e2[1], e2[2]}: generic = False
        for u, a, b, dq in ev:
            ia, ib = order.index(a), order.index(b)
            if abs(ia - ib) != 1: generic = False; continue
            k = min(ia, ib); lower = order[k]                            # the strand at position k is rising past position k+1
            front = dq if lower == a else -dq
            if abs(front) < 1e-12: generic = False
            g = (k + 1) * (1 if front > 0 else -1); word.append(g); log.append((t + u, BASE[order[k]], BASE[order[k + 1]], g))
            order[k], order[k + 1] = order[k + 1], order[k]
    return word, log, generic, float(dmin), [BASE[i] for i in order]


def prime_suspects(S, eps_ring, tol=1e-9):
    """the lowest strand at each step when some residual is positive, else '-'"""
    out = []
    for s in S:
        out.append(BASE[int(np.argmin(s))] if s.max() > tol else "-")
    return out


def _burau(word, n, t):
    M = sp.eye(n - 1) if isinstance(t, sp.Basic) else np.eye(n - 1)
    for g in word:
        i = abs(g) - 1
        S = sp.eye(n - 1) if isinstance(t, sp.Basic) else np.eye(n - 1, dtype=complex)
        if n == 2: S[0, 0] = -t
        elif i == 0: S[0, 0] = -t; S[1, 0] = 1
        elif i == n - 2: S[i - 1, i] = t; S[i, i] = -t
        else: S[i - 1, i] = t; S[i, i] = -t; S[i + 1, i] = 1
        if g < 0: S = S.inv() if isinstance(t, sp.Basic) else np.linalg.inv(S)
        M = M * S if isinstance(t, sp.Basic) else M @ S
    return M


def burau_det(word, n):
    if n % 2 == 0: raise ValueError("use odd n (the normaliser vanishes at t = -1 for even n)")
    M = _burau(word, n, -1.0); return float(abs(np.linalg.det(np.eye(n - 1) - M)))


def alexander(word, n):
    t = sp.Symbol("t"); M = _burau(word, n, t)
    num = sp.factor(sp.cancel((sp.eye(n - 1) - M).det() / sum(t ** k for k in range(n))))
    return num


def braid_pd(word, n):
    """PD code of the closure, strands oriented upward. Positive sigma_i: the strand entering at position i (bottom left) passes over to
    position i+1 (top right); this is a positive crossing. Bottom-left a, bottom-right b, top-left c, top-right d; counterclockwise from the
    incoming under-strand: positive [b, d, c, a], negative [a, b, d, c]. Returns (pd, number of components with no crossing)."""
    lab = list(range(n)); nxt = n; pd = []
    for g in word:
        i = abs(g) - 1; a, b = lab[i], lab[i + 1]; c, d = nxt, nxt + 1; nxt += 2
        pd.append([b, d, c, a] if g > 0 else [a, b, d, c])
        lab[i], lab[i + 1] = c, d
    par = list(range(nxt))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(n): par[f(lab[i])] = f(i)
    pd = [[f(e) for e in X] for X in pd]
    moved = {abs(g) - 1 for g in word} | {abs(g) for g in word}
    free = sum(1 for cyc in BK.cycles(BK.permutation(word, n)) if not any(i in moved for i in cyc))
    return pd, free


def jones_closure(word, n):
    """Jones polynomial of the closure in t (V = (-A^3)^(-w) <K>, t = A^-4, unknot = 1), by the state sum counted by (A-power, loops)"""
    pd, free = braid_pd(word, n); labels = sorted({e for X in pd for e in X}); idx = {l: i for i, l in enumerate(labels)}
    counts = {}
    for state in itertools.product((0, 1), repeat=len(pd)):
        par = list(range(len(labels)))
        def f(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for (a, b, c, d), s in zip(pd, state):
            pr = ((a, b), (c, d)) if s == 0 else ((a, d), (b, c))
            for u, v in pr: par[f(idx[u])] = f(idx[v])
        loops = len({f(i) for i in range(len(labels))}) if labels else 0
        key = (len(pd) - 2 * sum(state), loops); counts[key] = counts.get(key, 0) + 1
    Asym = sp.Symbol("A"); dd = -Asym ** 2 - Asym ** -2
    br = sum(c * Asym ** k * dd ** (l - 1 + free) for (k, l), c in counts.items()) if pd else dd ** (free - 1)
    w = sum(1 if g > 0 else -1 for g in word)
    V = sp.expand((-Asym ** 3) ** (-w) * br)
    return sp.expand(V.subs(Asym, sp.Symbol("t") ** sp.Rational(-1, 4)))


def corevision(H, beta=1.0, strong=0.5):
    """co-revision network of the persistent sources, after the time-aware, node-normalised collaboration model of Iannucci et al.
    (arXiv:2512.19677): a source 'acts' at step t when its report differs from the previous step; for each action of u at t, the pair
    (u, v) gains exp(-beta * lag) / (n_t - 1), where lag is the delay to v's next action (>= 0) and n_t the number of sources acting at t.
    Returns the symmetric weights and the sources in pairs of weight > strong (the coordinating group)."""
    acts = {}; prev = None
    for t, (cells, _) in enumerate(H):
        cur = {c[0]: c[2] for c in cells if c[0] in BASE}
        if prev is not None:
            for s in BASE:
                if any(abs(cur[s][k] - prev[s][k]) > 1e-12 for k in cur[s]): acts.setdefault(s, []).append(t)
        prev = cur
    W = {}
    for a, b in itertools.combinations(BASE, 2):
        w = 0.0
        for u, v in ((a, b), (b, a)):
            for t in acts.get(u, []):
                lag = [x - t for x in acts.get(v, []) if x >= t]
                if lag:
                    n = sum(1 for s in BASE if t in acts.get(s, [])); w += np.exp(-beta * min(lag)) / max(n - 1, 1)
        W[(a, b)] = w / 2
    group = tuple(sorted({x for (a, b), w in W.items() if w > strong for x in (a, b)}, key=BASE.index))
    return W, group
