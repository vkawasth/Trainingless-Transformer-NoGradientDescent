"""Perturbing context and probability, and watching the outcome: how far a decision is from changing, and what changes on the way.

The chain is  context -> cover (active windows) -> fibre of gluings -> support / place layer -> merged law -> decision -> value.
A perturbation enters at the top (the context, or the belief about it) or in the middle (the window tables); this module follows it
down the chain.

  decision_radius(domains, windows, utilities, evidence, action, weights)
        rho* = the smallest rho such that some law q with TV(q_w, t_w) <= rho * weight_w for every window makes another action at least
        as good as `action` given the evidence (one LP per competitor). For every law whose marginals are within rho < rho* the action
        stays strictly best, so every gluing, every soft gluing and every merge rule inside that tube makes the same decision.
        rho* = 0 means the windows alone do not decide: the fibre of gluings already contains a law under which another action ties
        (the decision depends on the filler). With weights = the experts' probe errors, rho* > 1 certifies the decision against them.
  tempered(windows, beta)   t_w^beta normalised: beta -> infinity keeps only each window's highest-probability cells (the mode skeleton),
        beta -> 0 flattens to uniform. The tropical defect grows like beta * delta_inf, delta_inf = the defect of the skeleton valuations.
  skeleton_defect(domains, windows)   delta_inf: the tropical defect of v_w - min v_w (the valuations measured from each window's mode)
  path(domains, windows_of_s, s_grid, eps, utilities, evidences)   along a perturbation path: lam*, theta*, place layer, tropical defect,
        mode of the merged law, size of its support, and the decision and value for each evidence
  envelope(values_by_context)  for a belief pi over two contexts: V_a(pi) = (1 - pi) V_a(0) + pi V_a(1); the breakpoints of the upper
        envelope (where the best action changes) and the best action on each piece
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from .coherence import Window, _constraints
from .integrate import gluable_fraction, maxent_merge
from .places import tropical_defect, layer
from .cover import glue_soft, expected_utility


def _evidence_mask(names, atoms, evidence):
    return np.array([1.0 if not evidence or all(a[names.index(v)] == val for v, val in evidence.items()) else 0.0 for a in atoms])


def decision_radius(domains, windows, utilities, evidence, action, weights=None):
    """returns (rho*, {competitor: rho_b}, witness law at the binding competitor, atoms). Also the evidence radius under key '_evidence':
    the smallest rho at which the evidence can become impossible (there the conditional decision is undefined)."""
    A, t, atoms, index = _constraints(domains, windows)
    names = list(domains); n, m, k = A.shape[1], A.shape[0], len(windows)
    wts = np.ones(k) if weights is None else np.maximum(np.asarray(weights, float), 1e-12)
    inE = _evidence_mask(names, atoms, evidence)
    nv = n + 2 * m + 1
    Aeq = np.zeros((m + 1, nv)); Aeq[:m, :n] = A; Aeq[:m, n:n + m] = -np.eye(m); Aeq[:m, n + m:n + 2 * m] = np.eye(m)
    Aeq[m, :n] = 1.0; beq = np.concatenate([t, [1.0]])
    Atv = np.zeros((k, nv))
    for r, (wi, _) in enumerate(index): Atv[wi, n + r] = 0.5; Atv[wi, n + m + r] = 0.5
    for wi in range(k): Atv[wi, -1] = -wts[wi]
    c = np.zeros(nv); c[-1] = 1.0
    ua = np.array([utilities[action](dict(zip(names, a))) for a in atoms])
    out, wit = {}, {}
    for b, u in utilities.items():
        if b == action: continue
        ub = np.array([u(dict(zip(names, a))) for a in atoms])
        g = np.zeros(nv); g[:n] = -(ub - ua) * inE                          # sum_E q (u_b - u_a) >= 0
        r = linprog(c, A_ub=np.vstack([Atv, g]), b_ub=np.zeros(k + 1), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        out[b] = float(r.fun); wit[b] = r.x[:n]
    gE = np.zeros(nv); gE[:n] = inE                                         # sum_E q <= 0
    r = linprog(c, A_ub=np.vstack([Atv, gE]), b_ub=np.zeros(k + 1), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    out["_evidence"] = float(r.fun) if r.status == 0 else float("inf")
    bstar = min((b for b in out if b != "_evidence"), key=out.get)
    return min(out.values()), out, wit[bstar], atoms


def tempered(windows, beta):
    out = []
    for w in windows:
        T = np.asarray(w.table, float)
        if beta == 0: P = (T > 0).astype(float)
        else:
            with np.errstate(divide="ignore"): L = np.where(T > 0, beta * np.log(np.where(T > 0, T, 1.0)), -np.inf)
            P = np.exp(L - L.max())
        out.append(Window(w.vars, P / P.sum()))
    return out


def skeleton_defect(domains, windows):
    """delta_inf: the tropical defect of the valuations v_w - min v_w; it is the growth rate of the defect under tempering."""
    A, t, atoms, index = _constraints(domains, windows)
    with np.errstate(divide="ignore"): v = np.where(t > 0, -np.log(np.where(t > 0, t, 1.0)), np.inf)
    vmin = {}
    for r, (wi, _) in enumerate(index): vmin[wi] = min(vmin.get(wi, np.inf), v[r])
    vb = np.array([v[r] - vmin[wi] for r, (wi, _) in enumerate(index)])
    x = np.array([max(vb[r] for r in np.nonzero(A[:, k])[0]) for k in range(A.shape[1])])
    d = 0.0
    for r in range(A.shape[0]):
        if not np.isfinite(vb[r]): continue
        d = max(d, min(x[k] for k in np.nonzero(A[r])[0]) - vb[r])
    return float(d)


def _merged(domains, windows):
    q, atoms, H, lam = maxent_merge(domains, windows)
    return q, atoms, lam


def path(domains, windows_of_s, s_grid, eps, utilities, evidences):
    names = list(domains); rows = []
    for s in s_grid:
        W = windows_of_s(s)
        g = glue_soft(domains, W, eps); d, N = tropical_defect(domains, W)
        q, atoms, lam = _merged(domains, W)
        P = np.zeros([domains[v] for v in names])
        for a, v in zip(atoms, q): P[a] = v
        dec = []
        for ev in evidences:
            eu = expected_utility(P, names, utilities, ev); best = max(eu, key=eu.get); dec.append((best, eu[best]))
        rows.append({"s": float(s), "lam": g["lam"], "theta": g["theta"], "layer": layer(domains, W)[0], "defect": d, "logN": float(np.log(N)),
                     "mode": atoms[int(np.argmax(q))], "support": int((q > 1e-9).sum()), "decisions": dec})
    return rows


def envelope(values):
    """values: {action: (V at pi = 0, V at pi = 1)}. Returns the breakpoints in (0, 1) and the best action on each piece."""
    acts = list(values)
    cand = {0.0, 1.0}
    for a, b in itertools.combinations(acts, 2):
        (a0, a1), (b0, b1) = values[a], values[b]; den = (a1 - a0) - (b1 - b0)
        if abs(den) > 1e-14:
            p = (b0 - a0) / den
            if 0 < p < 1: cand.add(p)
    pts = sorted(cand)
    best = lambda p: max(acts, key=lambda a: (1 - p) * values[a][0] + p * values[a][1])
    pieces = [(pts[i], pts[i + 1], best(0.5 * (pts[i] + pts[i + 1]))) for i in range(len(pts) - 1)]
    merged = [list(pieces[0])]
    for lo, hi, a in pieces[1:]:
        if a == merged[-1][2]: merged[-1][1] = hi
        else: merged.append([lo, hi, a])
    return [p[1] for p in merged[:-1]], [tuple(p) for p in merged]


# ------------------------------------------------------------------------------------------- the channels through which outcomes move
def gap_interval(domains, windows, utilities, evidence, a, b, rho=0.0, weights=None):
    """[min, max] of E_q[u_a - u_b | evidence] over the laws q with TV(q_w, t_w) <= rho * weight_w (rho = 0: the fibre of gluings).
    Charnes-Cooper: y = s q with sum_E y = 1. Returns (lo, hi); (nan, nan) if the tube is empty."""
    A, t, atoms, index = _constraints(domains, windows)
    names = list(domains); n, m, k = A.shape[1], A.shape[0], len(windows)
    wts = np.ones(k) if weights is None else np.asarray(weights, float)
    inE = _evidence_mask(names, atoms, evidence)
    d = np.array([(utilities[a](dict(zip(names, x))) - utilities[b](dict(zip(names, x)))) * inE[i] for i, x in enumerate(atoms)])
    nv = n + 2 * m + 1                                                          # y, s+, s-, s
    Aeq = np.zeros((m + 2, nv)); Aeq[:m, :n] = A; Aeq[:m, n:n + m] = -np.eye(m); Aeq[:m, n + m:n + 2 * m] = np.eye(m); Aeq[:m, -1] = -t
    Aeq[m, :n] = 1.0; Aeq[m, -1] = -1.0; Aeq[m + 1, :n] = inE; beq = np.concatenate([np.zeros(m + 1), [1.0]])
    Aub = np.zeros((k, nv))
    for r, (wi, _) in enumerate(index): Aub[wi, n + r] = 0.5; Aub[wi, n + m + r] = 0.5
    for wi in range(k): Aub[wi, -1] = -rho * wts[wi]
    out = []
    for sgn in (1.0, -1.0):
        c = np.zeros(nv); c[:n] = sgn * d
        r = linprog(c, A_ub=Aub, b_ub=np.zeros(k), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        out.append(sgn * r.fun if r.status == 0 else float("nan"))
    return out[0], out[1]


def joint_radius(P, names, utilities, evidence, action):
    """the smallest TV change of the joint law P that makes another action at least as good given the evidence (filler held: the
    world law itself is perturbed). LP over q = P + d+ - d-."""
    p = np.asarray(P, float).ravel(); atoms = list(itertools.product(*[range(s) for s in np.shape(P)])); n = len(p)
    inE = _evidence_mask(names, atoms, evidence)
    ua = np.array([utilities[action](dict(zip(names, x))) for x in atoms]); best = np.inf
    for b, u in utilities.items():
        if b == action: continue
        g = (np.array([u(dict(zip(names, x))) for x in atoms]) - ua) * inE
        # variables q (n), d+ (n), d- (n): q - d+ + d- = p, sum q = 1, -g q <= 0; minimise 0.5 sum (d+ + d-)
        c = np.concatenate([np.zeros(n), 0.5 * np.ones(2 * n)])
        Aeq = np.vstack([np.hstack([np.eye(n), -np.eye(n), np.eye(n)]), np.concatenate([np.ones(n), np.zeros(2 * n)])])
        r = linprog(c, A_ub=np.concatenate([-g, np.zeros(2 * n)])[None, :], b_ub=[0.0], A_eq=Aeq, b_eq=np.concatenate([p, [1.0]]),
                    bounds=(0, None), method="highs")
        if r.status == 0: best = min(best, float(r.fun))
    return best


def _law(domains, windows):
    q, atoms, _, _ = maxent_merge(domains, windows); names = list(domains)
    P = np.zeros([domains[v] for v in names])
    for a, v in zip(atoms, q): P[a] = v
    return P


def merge_gap(domains, windows, utilities, evidence, a, b):
    P = _law(domains, windows); eu = expected_utility(P, list(domains), utilities, evidence); return eu[a] - eu[b]


def _push(windows, grads, rho, weights):
    """move mass rho * weight_w in each window from its lowest-gradient cells (as much as they hold) to its highest-gradient cell."""
    out = []
    for w, g, wt in zip(windows, grads, weights):
        T = np.asarray(w.table, float).ravel().copy(); gg = g.ravel(); budget = min(rho * wt, 1.0 - T[np.argmax(gg)])
        top = int(np.argmax(gg)); moved = 0.0
        for i in np.argsort(gg):
            if i == top or moved >= budget: continue
            dm = min(T[i], budget - moved); T[i] -= dm; moved += dm
        T[top] += moved; out.append(Window(w.vars, T.reshape(np.shape(w.table))))
    return out


def window_radius(domains, windows, utilities, evidence, action, weights=None, h=1e-5, rho_max=4.0):
    """for the maximum-entropy merge: (linearised radius, realised radius). The gradient of the gap g_ab = E[u_a - u_b | e] in the
    window cells gives the first-order radius |g_ab| / sum_w weight_w * osc(grad_w) (osc = max - min over the cells: the most a TV
    move of size weight_w can change a linear functional). The realised radius pushes mass along that steepest schedule and bisects
    on the sign of the true gap; it is an upper bound on the radius of the maximum-entropy decision."""
    k = len(windows); wts = np.ones(k) if weights is None else np.asarray(weights, float)
    best_lin, best_real = np.inf, np.inf
    for b in utilities:
        if b == action: continue
        g0 = merge_gap(domains, windows, utilities, evidence, action, b)
        grads = []
        for wi, w in enumerate(windows):
            G = np.zeros(np.size(w.table)); T = np.asarray(w.table, float).ravel()
            for i in range(len(T)):
                Tp = T.copy(); Tp[i] += h; Tp /= Tp.sum()
                W2 = list(windows); W2[wi] = Window(w.vars, Tp.reshape(np.shape(w.table)))
                G[i] = (merge_gap(domains, W2, utilities, evidence, action, b) - g0) / h
            grads.append(-G)                                               # push in the direction that lowers the gap
        rate = sum(wt * (g.max() - g.min()) for wt, g in zip(wts, grads))
        best_lin = min(best_lin, g0 / rate if rate > 0 else np.inf)
        if merge_gap(domains, _push(windows, grads, rho_max, wts), utilities, evidence, action, b) > 0: continue
        lo, hi = 0.0, rho_max
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if merge_gap(domains, _push(windows, grads, mid, wts), utilities, evidence, action, b) > 0: lo = mid
            else: hi = mid
        best_real = min(best_real, hi)
    return best_lin, best_real


def bridge_gain(domains, windows, truth_law, utilities, evidence, action, candidates):
    """the value of an overlap: for each candidate (a tuple of variable sets), add the true marginal on each set as a window and return
    the fibre gap interval against every competitor. The decision is pinned by the windows iff every interval excludes 0."""
    names = list(domains); out = {}
    for cand in candidates:
        W2 = list(windows)
        for S in cand:
            T = np.asarray(truth_law).sum(axis=tuple(i for i, v in enumerate(names) if v not in S))
            W2.append(Window(tuple(v for v in names if v in S), T))
        out[tuple(cand)] = {b: gap_interval(domains, W2, utilities, evidence, action, b) for b in utilities if b != action}
    return out


def pinned(intervals, tol=1e-9):
    return all(lo > tol for lo, hi in intervals.values())


def minimal_bridges(domains, windows, truth_law, utilities, evidence, action, pool, kmax=3):
    """the smallest sets of bridge windows (drawn from pool) whose true marginals pin the decision; [] if none up to kmax."""
    for k in range(0, kmax + 1):
        found = [c for c, iv in bridge_gain(domains, windows, truth_law, utilities, evidence, action,
                                            list(itertools.combinations(pool, k))).items() if pinned(iv)]
        if found: return found
    return []
