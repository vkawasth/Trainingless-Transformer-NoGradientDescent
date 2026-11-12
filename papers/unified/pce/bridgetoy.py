"""The toy bridge of the companion note (sec. 'Beliefs under contamination'): five sources on a five-cycle of binary variables, two
decisions, an inspection, and six matched histories. Everything computed here is what the decision maker can compute from the data he
has received; the true law is used only to generate honest tables and, in the checks, to score decisions. New module.
  world()            true law P (Ising-type on the cycle, coupling 0.7, field 0.15 on X1), cells S1..S5 on (X1,X2),...,(X5,X1),
                     functionals gA = P(X1 = X2), gB = P(X3 = X4), threshold 1/2
  pair_table(P, ctx, a)   a table on a pair context with the true marginals and agreement probability a
  views(cells)       the explanation strands: for each source S_i, the midpoint and width of the gB interval of 'trust all but S_i'
                     (at the least-incoherent tolerance); inspections are trusted
  decide(cells, k)   naive decision (midpoint of the least-incoherent fibre vs 1/2) and the k-robust decision (accept / reject / investigate)
  histories()        H1 honest; H1b honest with S3 and the inspection swapped; H2 S3 withheld and its inspection refused; H3 S3
                     poisoned beyond its budget; H4 S3 poisoned within its budget, exposed by the inspection, then S2 recruited to restore
                     coherence; H5 S3 poisoned then corrected
  run(steps)         the DM's state after each step
  path_word(rows), loop_word(rows)  braid words of the strands (declared rules: ties broken by a fixed offset 1e-6 x source index;
                     at a crossing the strand with the wider interval passes over)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from . import tamper as T

N_VAR = 5; TAU = 0.5; LABELS = ["S1", "S2", "S3", "S4", "S5"]
CTX = {"S1": (0, 1), "S2": (1, 2), "S3": (2, 3), "S4": (3, 4), "S5": (4, 0)}
QCTX = (1, 3)


def world():
    A = T.atoms(N_VAR)
    w = np.array([np.exp(0.7 * sum(x[i] * x[(i + 1) % 5] for i in range(5)) + 0.15 * x[0]) for x in A]); P = w / w.sum()
    agree = lambda i, j: np.array([1.0 if x[i] == x[j] else 0.0 for x in A])
    return P, agree(0, 1), agree(2, 3)


def table(P, ctx):
    M, keys = T.marg(N_VAR, ctx); return dict(zip(keys, M @ P))


def honest_cells(P): return {k: (k, c, table(P, c)) for k, c in CTX.items()}


def pair_table(P, ctx, a):
    t = table(P, ctx); m1 = t[(1, 1)] + t[(1, -1)]; m2 = t[(1, 1)] + t[(-1, 1)]; x = (a + m1 + m2 - 1) / 2
    return {(1, 1): x, (-1, -1): a - x, (1, -1): m1 - x, (-1, 1): m2 - x}


def _tolerant_interval(cells, g):
    res = T.residual(N_VAR, cells); return T.interval(N_VAR, cells, g, tol=res + 1e-9 if res > 1e-9 else 0.0), res


def views(cells, gB):
    out = {}
    for s in LABELS:
        I, _ = _tolerant_interval([c for c in cells if c[0] != s], gB); out[s] = (0.5 * (I[0] + I[1]), I[1] - I[0])
    return out


def decide(cells, gB, k=1):
    I, res = _tolerant_interval(cells, gB); mid = 0.5 * (I[0] + I[1])
    removable = [c for c in cells if not c[0].startswith("Q")]
    Is = []
    for H in T.explanations(N_VAR, removable, k):
        rest = [c for c in cells if c[0] not in H]
        if T.feasible(N_VAR, rest): Is.append(T.interval(N_VAR, rest, gB))
    lo, hi = (min(i[0] for i in Is), max(i[1] for i in Is)) if Is else (0.0, 1.0)
    robust = "accept" if lo >= TAU else ("reject" if hi < TAU else "investigate")
    return dict(res=res, naive="accept" if mid >= TAU else "reject", mid=mid, robust=robust, rob=(lo, hi))


def recruit(P, name, ctx, others):
    M, b = T.system(N_VAR, others); r = linprog(np.zeros(2 ** N_VAR), A_eq=M, b_eq=b, bounds=[(0, None)] * 2 ** N_VAR, method="highs")
    return None if r.status else (name, ctx, table(r.x, ctx))


def histories():
    P, gA, gB = world(); h = honest_cells(P); Q = ("Q", QCTX, table(P, QCTX))
    S3_out = ("S3", CTX["S3"], pair_table(P, CTX["S3"], 0.10)); S3_in = ("S3", CTX["S3"], pair_table(P, CTX["S3"], 0.40))
    c2 = recruit(P, "S2", CTX["S2"], [h["S1"], h["S4"], h["S5"], S3_in, Q])
    base = [("a1", h["S1"]), ("a2", h["S2"]), ("a4", h["S4"]), ("a5", h["S5"])]
    H = {"H1": base + [("a3", h["S3"]), ("aQ", Q)],
         "H1b": base + [("aQ", Q), ("a3", h["S3"])],
         "H2": base + [("w3", None), ("rQ3", None)],
         "H3": base + [("a3", S3_out), ("aQ", Q)],
         "H4": base + [("a3", S3_in), ("aQ", Q), ("c2", c2)],
         "H5": base + [("a3", S3_out), ("aQ", Q), ("k3", h["S3"])]}
    return P, gA, gB, h, Q, S3_in, S3_out, c2, H


def run(steps, gB, k=1):
    cells = []; rows = []
    for lab, op in steps:
        if op is not None: cells = [c for c in cells if c[0] != op[0]] + [op]
        rows.append((lab, views(cells, gB), decide(cells, gB, k)))
    return cells, rows


def _order(row): return sorted(LABELS, key=lambda s: row[1][s][0] + 1e-6 * LABELS.index(s))


def path_word(rows):
    word, labs = [], []
    for t in range(1, len(rows)):
        sw = T.bubble_word(_order(rows[t - 1]), _order(rows[t]), lambda lo, hi, t=t: 1 if rows[t][1][lo][1] >= rows[t][1][hi][1] else -1)
        word += sw; labs += [rows[t][0]] * len(sw)
    return word, labs


def empty_row(gB): return ("0", views([], gB), None)


def replay(cells):
    key = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4, "Q": 5}
    return [("r" + c[0], c) for c in sorted(cells, key=lambda c: key[c[0]])]


def loop_word(steps, gB):
    """the history, then back along the canonical replay of its final cells: a closed loop from and to the empty state"""
    cells, rows = run(steps, gB); _, rr = run(replay(cells), gB); e = empty_row(gB)
    return path_word([e] + rows + list(reversed(rr[:-1])) + [e])[0]


# ---- the six components of the DM's state (pce/trajknot.py) ----
_CACHE = {}
CANDIDATES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]


def _robust_width(cells, gB, k=1):
    return decide(cells, gB, k)["rob"][1] - decide(cells, gB, k)["rob"][0]


def _sample_laws(cells, m, seed=0):
    rng = np.random.default_rng(seed); N = 2 ** N_VAR
    if not cells: return [rng.dirichlet(np.ones(N)) for _ in range(m)]
    res = T.residual(N_VAR, cells); out = []
    M, b = T.system(N_VAR, cells); M, b = M[1:], b[1:]; r = len(b)
    for _ in range(m):
        c = np.concatenate([rng.normal(size=N) * 1e-3, np.ones(2 * r)])
        Aeq = [np.concatenate([np.ones(N), np.zeros(2 * r)])]; beq = [1.0]
        for i in range(r):
            row = np.concatenate([M[i], np.zeros(2 * r)]); row[N + i] = -1; row[N + r + i] = 1; Aeq.append(row); beq.append(b[i])
        sol = linprog(c, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * (N + 2 * r), method="highs"); out.append(sol.x[:N])
    return out


def voi(cells, gB, m=4):
    key = tuple(sorted((c[0], tuple(round(v, 9) for v in c[2].values())) for c in cells))
    if key in _CACHE: return _CACHE[key]
    w0 = _robust_width(cells, gB); present = {tuple(c[1]) for c in cells}; best = 0.0
    laws = _sample_laws(cells, m)
    for ctx in CANDIDATES:
        if ctx in present: continue
        red = [w0 - _robust_width(cells + [("Qc", ctx, table(p, ctx))], gB) for p in laws]
        best = max(best, float(np.mean(red)))
    _CACHE[key] = best; return best


def components(steps, gB, imped_labels=("w3", "rQ3")):
    """the closed trajectory: the history from the empty state, then back along the canonical replay of its final cells"""
    first = {}; cells = []; imp = 0; pts = []
    def point(cells, imp):
        d = decide(cells, gB); drift = sum(sum(abs(c[2][k] - first[c[0]][k]) for k in c[2]) for c in cells)
        return [voi(cells, gB), d["res"], d["rob"][1] - d["rob"][0], d["mid"], drift, imp]
    pts.append(point([], 0))
    for lab, op in steps:
        if op is not None:
            first.setdefault(op[0], op[2]); cells = [c for c in cells if c[0] != op[0]] + [op]
        if lab in imped_labels: imp += 1
        pts.append(point(cells, imp))
    back = []; rc = []
    for lab, op in replay(cells):
        rc = rc + [op]; back.append(point(rc, 0))
    pts += list(reversed(back[:-1])) + [point([], 0)]
    return np.array(pts[:-1])                                           # closed: the last point equals the first


# ---- the typed, two-tier trace (rem: events) ----
def history_cells(steps):
    """every report kept as its own cell, named source@step, plus the impediment log"""
    cells, log = [], []
    for t, (lab, op) in enumerate(steps):
        if op is None: log.append((t, lab))
        else: cells.append((f"{op[0]}@{t}", op[1], op[2]))
    return cells, log


def typed_letters(steps):
    """order-invariant types, read off the final history (so equal for every reordering inside a trace class):
    'x' the report lies in some minimal conflict of the full history, 'o' otherwise; '+chg' if its source had already reported a
    different table on the same context; impediments keep their own letters"""
    cells, log = history_cells(steps); conf = T.conflicts(N_VAR, cells); inconf = {c for C in conf for c in C}
    seen = {}; out = []
    for t, (lab, op) in enumerate(steps):
        if op is None: out.append(lab); continue
        name = f"{op[0]}@{t}"; key = (op[0], tuple(op[1]))
        chg = key in seen and any(abs(seen[key][k] - op[2][k]) > 1e-9 for k in op[2]); seen[key] = op[2]
        out.append(f"{lab}:{'x' if name in inconf else 'o'}{'+chg' if chg else ''}")
    return out, conf


def operation_independence(steps):
    """the report key of each operation, (source, context); the state is S = (R, W), R a map from report keys to tables, W the log of
    withheld and refused reports. Inserting or replacing under different keys, or appending to W for a different key, commute exactly
    on every state; two operations on the same key are treated as dependent."""
    src = {}
    for lab, op in steps: src[lab] = (op[0], tuple(op[1])) if op is not None else {"w3": ("S3", CTX["S3"]), "rQ3": ("S3", CTX["S3"])}[lab]
    return src


def two_tier(steps, gB, k=1):
    """tier 1: the operations; tier 2: a letter dB:<new decision> after every step at which the robust decision on B changes.
    A decision letter and an adjacent operation are declared independent only if moving the operation across it (recomputing the
    decision on the state without that operation) leaves the decision unchanged: tested on the visited states."""
    cells = []; prev = None; word = []; states = []
    for lab, op in steps:
        before = list(cells)
        if op is not None: cells = [c for c in cells if c[0] != op[0]] + [op]
        word.append(lab); states.append((before, list(cells)))
        d = decide(cells, gB, k)["robust"]
        if d != prev: word.append(f"dB:{d}@{len(word)}"); prev = d
    return word


def two_tier_independence(steps, gB, word, k=1):
    src = operation_independence(steps); ops = [w for w in word if not w.startswith("dB")]; indep = set()
    for a in ops:
        for b in ops:
            if src[a] != src[b]: indep.add((a, b))
    # decision letters: test against each operation by removing that operation from the prefix and recomputing
    pos = {w: i for i, w in enumerate(word)}; stepmap = {lab: op for lab, op in steps}
    for dl in [w for w in word if w.startswith("dB")]:
        target = dl.split(":")[1].split("@")[0]; prefix = [w for w in word[:pos[dl]] if not w.startswith("dB")]
        for a in prefix:
            cells = []
            for lab in prefix:
                if lab == a: continue
                op = stepmap[lab]
                if op is not None: cells = [c for c in cells if c[0] != op[0]] + [op]
            if decide(cells, gB, k)["robust"] == target: indep.add((a, dl))
    return indep


def durable_budget(P, gB):
    """the coalition {S3}'s budget for B if every one of the five chord inspections were bought (honest): analyst-side, since it uses
    the inspections' true tables"""
    h = honest_cells(P); cells = [h[k] for k in ("S1", "S2", "S4", "S5")]
    chords = [("Q%d" % i, ch, table(P, ch)) for i, ch in enumerate([(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)])]
    return T.interval(N_VAR, cells + chords, gB), T.interval(N_VAR, cells + [c for c in chords if c[1] == (1, 3)], gB)


def payload(op):
    """the intrinsic content of a report on a pair context: its agreement probability, to two decimals"""
    t = op[2]; return round(sum(v for k, v in t.items() if k[0] == k[1]), 2)


def annotated(steps):
    """operations with payload (the letters) and, separately, the arrival-time annotation l_S(e) the DM could compute when each
    arrived: snapshot compatibility of the current reports, historical consistency of every report kept, and revision (same source,
    same context, different table). Annotations do not take part in commutation."""
    cur, hist, seen, out = [], [], {}, []
    for t, (lab, op) in enumerate(steps):
        if op is None: out.append((lab, None)); continue
        key = (op[0], tuple(op[1])); rev = key in seen and payload(op) != payload(seen[key]); seen[key] = op
        cur = [c for c in cur if (c[0], tuple(c[1])) != key] + [op]; hist = hist + [(f"{op[0]}@{t}", op[1], op[2])]
        ann = dict(snapshot=T.residual(N_VAR, cur) < 1e-9, history=T.residual(N_VAR, hist) < 1e-9, revision=rev)
        out.append((f"{lab}({payload(op)})", ann))
    return out
