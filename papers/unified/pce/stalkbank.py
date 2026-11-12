"""The decision maker's stalks along a history: every signal computed from his own reports, inspections and impediment log, kept side
by side because we do not know in advance which will work (companion note, 'Watching for coordinated deformation'). New module.
Cells are (name, context, table); a name's prefix gives its source (S1..S5 cycle sources, R1, R2 redundant reporters of B's context,
F1, F2 planted reports, Q inspections, which are trusted).
  stalks(cells, history, g, tau, tol)   one dictionary of scalar signals at a time step:
     data:          within-context disagreement (max L1 between reports of one context), ladder counts (monitor / flag), overlap
                    disagreement (max L1 between marginals on shared variables)
     compatibility: possibilistic incoherence (no assignment consistent with every support), cycle-inequality violation on the five-cycle
                    (Araujo et al. form: max over odd sign patterns of sum s_i E_i minus 3), residual, number of minimal conflicts, smallest
                    explanation, Farkas support size
     hidden:        mean fibre width along the 21 hidden Walsh directions, forced zeros, smallest face distance (min over cells of the
                    largest value the cell can take), mean move capacity range
     decision:      robust range (k = 2) width, naive value, excess reach (largest over pairs of cells), correlation between each cell's
                    leverage (widening of the least-incoherent range without it) and its number of minimal conflicts
     history:       revisions, historical residual
     anchors:       residual of reports and inspections together (inspections trusted)
     observability: dimension of the hidden space invisible to the inspections, and the norm of the decision's component in it
  decisions(cells, g, tau)   naive, majority (redundant reports of a context replaced by their cellwise median), reputation (cells in the
                    most minimal conflicts dropped), Occam (a smallest explanation removed), robust (k = 2: accept / reject / investigate)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import tamper as T

N = 5; ATOMS = T.atoms(N)
WALSH_OBSERVED = [()] + [(i,) for i in range(5)] + [(i, (i + 1) % 5) for i in range(5)]


def walsh(S): return np.array([np.prod([x[i] for i in S]) if S else 1.0 for x in ATOMS])


HIDDEN = [S for k in range(2, 6) for S in itertools.combinations(range(5), k) if not (k == 2 and (S[1] - S[0]) % 5 in (1, 4))]
HIDDEN_DIRS = [walsh(S) for S in HIDDEN]                              # 21 directions spanning ker A (A = five pair contexts + 1)


def src(name): return name.split("@")[0].rstrip("0123456789") if name[0] in "SRFQ" and False else name.split("@")[0]


def _lp(c, cells, extra_ub=None):
    M, b = T.system(N, cells)
    r = linprog(c, A_eq=M, b_eq=b, bounds=[(0, None)] * 32, method="highs")
    return None if r.status else r


def fibre_range(cells, v):
    lo = _lp(v, cells); hi = _lp(-v, cells)
    return None if lo is None or hi is None else (lo.fun, -hi.fun)


def least_incoherent(cells):
    """the set of laws at minimal total L1 discrepancy, as cells with tolerance: returns tol"""
    return T.residual(N, cells)


def within_context(cells, tol):
    by = {}
    for c in cells: by.setdefault(tuple(c[1]), []).append(c)
    dmax, mon, flag = 0.0, 0, 0
    for ctx, cs in by.items():
        for a, b in itertools.combinations(cs, 2):
            d = sum(abs(a[2][k] - b[2][k]) for k in a[2]); dmax = max(dmax, d)
            if d > tol: flag += 1
            elif d > 1e-12: mon += 1
    return dmax, mon, flag


def overlap(cells):
    worst = 0.0
    for a, b in itertools.combinations(cells, 2):
        sh = sorted(set(a[1]) & set(b[1]))
        if not sh or tuple(a[1]) == tuple(b[1]): continue
        def m(c):
            out = {}
            for k, v in c[2].items():
                kk = tuple(k[list(c[1]).index(i)] for i in sh); out[kk] = out.get(kk, 0) + v
            return out
        ma, mb = m(a), m(b); worst = max(worst, sum(abs(ma[k] - mb.get(k, 0)) for k in ma))
    return worst


def possibilistic_incoherent(cells):
    ok = [all(c[2][tuple(x[i] for i in c[1])] > 1e-12 for c in cells) for x in ATOMS]
    for c in cells:                                                     # every possible local outcome must extend to an allowed atom
        for k, v in c[2].items():
            if v > 1e-12 and not any(ok[j] and tuple(ATOMS[j][i] for i in c[1]) == k for j in range(32)): return 1.0
    return 0.0


def cycle_violation(cells):
    E = {}
    for c in cells:
        a, b = c[1]
        if (b - a) % 5 in (1, 4):
            e = sum(v * k[0] * k[1] for k, v in c[2].items()); E.setdefault(tuple(sorted((a, b))), []).append(e)
    if len(E) < 5: return 0.0
    Es = [np.mean(E[tuple(sorted((i, (i + 1) % 5)))]) for i in range(5)]
    best = -np.inf
    for s in itertools.product((1, -1), repeat=5):
        if np.prod(s) == -1: best = max(best, sum(si * ei for si, ei in zip(s, Es)))
    return float(best - 3)


def hidden_geometry(cells):
    res = T.residual(N, cells)
    if res > 1e-9: return dict(hid_width=np.nan, forced0=np.nan, face_min=np.nan, cap_mean=np.nan)
    widths = []
    for d in HIDDEN_DIRS:
        r = fibre_range(cells, d / 32.0); widths.append(r[1] - r[0])
    ub = [-_lp(-np.eye(32)[j], cells).fun for j in range(32)]
    caps = []
    M, b = T.system(N, cells)
    for d in HIDDEN_DIRS[:7]:                                           # largest two-sided capacity of a move, over the fibre
        A_ub = np.hstack([-np.eye(32), np.abs(d)[:, None] / 32.0]); A_eq = np.hstack([M, np.zeros((M.shape[0], 1))])
        r = linprog(np.r_[np.zeros(32), -1.0], A_ub=A_ub, b_ub=np.zeros(32), A_eq=A_eq, b_eq=b, bounds=[(0, None)] * 33, method="highs")
        caps.append(-r.fun if r.status == 0 else 0.0)
    return dict(hid_width=float(np.mean(widths)), forced0=float(sum(u < 1e-9 for u in ub)), face_min=float(min(ub)), cap_mean=float(np.mean(caps)))


def tol_interval(cells, g):
    res = T.residual(N, cells); return T.interval(N, cells, g, tol=res + 1e-9 if res > 1e-9 else 0.0), res


def robust(cells, g, k=2):
    removable = [c for c in cells if not c[0].startswith("Q")]; Is = []
    for H in T.explanations(N, removable, k):
        rest = [c for c in cells if c[0] not in H]
        if T.feasible(N, rest): Is.append(T.interval(N, rest, g))
    return (min(i[0] for i in Is), max(i[1] for i in Is)) if Is else (0.0, 1.0)


def decisions(cells, g, tau, k=2):
    def nv(cs):
        I, _ = tol_interval(cs, g); return "accept" if 0.5 * (I[0] + I[1]) >= tau else "reject"
    out = dict(naive=nv(cells))
    by = {}
    for c in cells: by.setdefault(tuple(c[1]), []).append(c)
    maj = []
    for ctx, cs in by.items():
        if len(cs) == 1 or cs[0][0].startswith("Q"): maj += cs; continue
        med = {kk: float(np.median([c[2][kk] for c in cs])) for kk in cs[0][2]}; s = sum(med.values())
        maj.append((f"maj{ctx}", ctx, {kk: v / s for kk, v in med.items()}))
    out["majority"] = nv(maj)
    conf = T.conflicts(N, [c for c in cells if not c[0].startswith("Q")]) if T.residual(N, cells) > 1e-9 else []
    cnt = {c[0]: sum(c[0] in C for C in conf) for c in cells}
    top = max(cnt.values()) if cnt else 0
    out["reputation"] = nv([c for c in cells if top == 0 or cnt[c[0]] < top or c[0].startswith("Q")])
    ex = [e for e in T.explanations(N, [c for c in cells if not c[0].startswith("Q")], 3)]
    smallest = min(ex, key=len) if ex else ()
    out["occam"] = nv([c for c in cells if c[0] not in smallest])
    lo, hi = robust(cells, g, k); out["robust"] = "accept" if lo >= tau else ("reject" if hi < tau else "investigate")
    return out, conf, ex


def stalks(cells, hist_cells, g, tau, tol, k=2):
    reports = [c for c in cells if not c[0].startswith("Q")]; insp = [c for c in cells if c[0].startswith("Q")]
    s = {}
    s["within_max"], s["ladder_monitor"], s["ladder_flag"] = within_context(reports, tol)
    s["overlap_max"] = overlap(reports); s["possibilistic"] = possibilistic_incoherent(reports); s["cycle_violation"] = cycle_violation(reports)
    s["residual"] = T.residual(N, reports)
    dec, conf, ex = decisions(cells, g, tau, k)
    s["n_conflicts"] = float(len(conf)); s["min_expl"] = float(min((len(e) for e in ex), default=4))
    y, touched = (T.farkas(N, reports) if s["residual"] > 1e-9 else (None, [])); s["farkas_support"] = float(len(touched))
    s.update(hidden_geometry(reports))
    lo, hi = robust(cells, g, k); I, _ = tol_interval(cells, g); s["robust_width"] = hi - lo; s["naive_value"] = 0.5 * (I[0] + I[1])
    reach = 0.0
    for a, b in itertools.combinations(reports, 2):
        rest = [c for c in reports if c[0] not in (a[0], b[0])]; Ir = T.interval(N, rest, g) if T.feasible(N, rest) else None
        ia = T.interval(N, [c for c in reports if c[0] != a[0]], g) if T.feasible(N, [c for c in reports if c[0] != a[0]]) else None
        ib = T.interval(N, [c for c in reports if c[0] != b[0]], g) if T.feasible(N, [c for c in reports if c[0] != b[0]]) else None
        if Ir and ia and ib: reach = max(reach, (Ir[1] - Ir[0]) - max(ia[1] - ia[0], ib[1] - ib[0]))
    s["excess_reach"] = reach
    if conf:
        Ib, _ = tol_interval(cells, g); base = Ib[1] - Ib[0]; lev, mem = [], []
        for c in reports:                                               # leverage: widening of the least-incoherent range without the cell
            Ic, _ = tol_interval([d for d in cells if d[0] != c[0]], g); lev.append((Ic[1] - Ic[0]) - base); mem.append(sum(c[0] in C for C in conf))
        s["lev_conflict_corr"] = float(np.corrcoef(lev, mem)[0, 1]) if np.std(lev) > 1e-12 and np.std(mem) > 1e-12 else 0.0
    else: s["lev_conflict_corr"] = 0.0
    keys = {}
    for c in hist_cells: keys.setdefault((c[0].split("@")[0], tuple(c[1])), []).append(c)
    s["revisions"] = float(sum(len(v) - 1 for v in keys.values())); s["hist_residual"] = T.residual(N, [c for c in hist_cells if not c[0].startswith("Q")])
    s["anchor_residual"] = T.residual(N, reports + insp) if insp else 0.0
    A = np.vstack([T.marg(N, c[1])[0] for c in reports] + [np.ones((1, 32))])
    Aq = np.vstack([A] + [T.marg(N, c[1])[0] for c in insp]) if insp else A
    U, S, Vt = np.linalg.svd(Aq); rank = int((S > 1e-9).sum()); null = Vt[rank:]
    s["invisible_dim"] = float(null.shape[0]); s["decision_invisible"] = float(np.linalg.norm(null @ g)) if null.size else 0.0
    return s, dec
