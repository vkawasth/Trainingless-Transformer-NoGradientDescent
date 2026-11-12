"""Checks for 'Computability' (section sec:computability): certificates that can be verified in exact rational arithmetic.
Floating-point linear programmes decide; the same question is then decided exactly by a rational simplex (Phase I, Bland's rule, Python
fractions), which returns either an exact global law or, from its final duals, an exact Farkas certificate; both are verified with exact
arithmetic. (A first version rebuilt exact certificates from the floating-point solution's support or active set; at degenerate vertices,
common here because neighbouring tables share margins, it produced no certificate in 56 of 200 instances, and EX1 and EX2 failed. It was
replaced by the exact simplex; no verdict disagreed in either version.)
  EX1 IDENTITY  rational instances on three binary variables, cover {X,Y}, {Y,Z}, {X,Z}, every table locally compatible with its neighbours
                (same single margins); feasible ones receive an exact rational global law (a vertex rebuilt on the solver's support),
                infeasible ones an exact rational Farkas certificate (M^T y >= 0, d.y = -1, rebuilt on the solver's active set); the exact
                verdict agrees with the floating-point verdict in every instance (200 instances)
  EX2 IDENTITY  the four-variable cycle with perfect correlations (Study 27a): the local tables extend (exact law); adding the twisted pair of
                outer tables does not (exact Farkas certificate supported on the four tables of the cycle)
  EX3 IDENTITY  event equivalence is decided by one procedure (cancellation across independent events, then the Cartier-Foata normal form):
                the verdict does not depend on the order in which cancellations are made (300 words, 10 random cancellation orders each)
  EX4 IDENTITY  the relay clock is computable on traces and does not descend to A_G: the trace (normal form without cancellation) is decided
                by the same procedure, activation is constant on trace classes, and two words equal in A_G with different activation are
                exhibited (records Theorem thm:relayclock(a), (c) in computability terms)
"""
import os, sys, itertools
import numpy as np
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import linprog
from pce import outerrelay as O
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(1)

def cmat(ctx, names, states):
    idx = [names.index(v) for v in ctx]; A = np.zeros((2 ** len(ctx), len(states)), dtype=int)
    for i, s in enumerate(states): A[int("".join(str(s[j]) for j in idx), 2), i] = 1
    return A
from fractions import Fraction as Fr
def exact_simplex_phase1(Mi, d):
    """minimise sum of artificials for M x + a = d (d >= 0 after sign flips), x, a >= 0, in exact fractions with Bland's rule.
    Returns (value, x, y) with y the final duals (reduced cost of artificial j = 1 - y_j)."""
    m, n = Mi.shape; rows = []
    for i in range(m):
        sgn = -1 if d[i] < 0 else 1
        rows.append([Fr(sgn * int(v)) for v in Mi[i]] + [Fr(1 if k == i else 0) for k in range(m)] + [Fr(sgn) * d[i]])
    flips = [(-1 if d[i] < 0 else 1) for i in range(m)]
    basis = [n + i for i in range(m)]; cost = [Fr(0)] * n + [Fr(1)] * m
    while True:
        y = [cost[basis[i]] for i in range(m)]
        red = [cost[j] - sum(y[i] * rows[i][j] for i in range(m)) for j in range(n + m)]
        enter = next((j for j in range(n + m) if red[j] < 0), None)
        if enter is None: break
        cand = [(rows[i][-1] / rows[i][enter], basis[i], i) for i in range(m) if rows[i][enter] > 0]
        _, _, r = min(cand)
        piv = rows[r][enter]; rows[r] = [v / piv for v in rows[r]]
        for i in range(m):
            if i != r and rows[i][enter] != 0:
                f = rows[i][enter]; rows[i] = [vi - f * vr for vi, vr in zip(rows[i], rows[r])]
        basis[r] = enter
    x = [Fr(0)] * (n + m)
    for i in range(m): x[basis[i]] = rows[i][-1]
    yv = [Fr(1) - red[n + i] for i in range(m)]                     # duals of the (sign-flipped) equality rows
    yv = [yv[i] * flips[i] for i in range(m)]                       # back to the original rows
    return sum(x[n:]), x[:n], yv
def exact_decide(Mi, d):
    """('law', x) with M x = d, x >= 0 exactly, or ('farkas', y) with M^T y >= 0 and d.y = -1 exactly"""
    d = [Fr(int(sp.fraction(v)[0]), int(sp.fraction(v)[1])) if not isinstance(v, Fr) else v for v in d]
    val, x, yv = exact_simplex_phase1(Mi, d); m, n = Mi.shape
    if val == 0:
        ok = all(sum(int(Mi[i, j]) * x[j] for j in range(n)) == d[i] for i in range(m)) and all(v >= 0 for v in x)
        return ("law", x) if ok else (None, None)
    y = [-v for v in yv]; s = sum(d[i] * y[i] for i in range(m)); y = [v / (-s) for v in y]     # normalise d.y = -1
    ok = all(sum(int(Mi[i, j]) * y[i] for i in range(m)) >= 0 for j in range(n)) and sum(d[i] * y[i] for i in range(m)) == -1
    return ("farkas", y) if ok else (None, None)

# EX1
names3 = ("X", "Y", "Z"); S3 = list(itertools.product((0, 1), repeat=3)); cover = [("X", "Y"), ("Y", "Z"), ("X", "Z")]
Mi = np.vstack([cmat(c, names3, S3) for c in cover]); agree = True; nf = ni = 0
for _ in range(200):
    w = rng.integers(1, 20, 8); law = [sp.Rational(int(v), int(w.sum())) for v in w]
    tabs = [list(sp.Matrix(cmat(c, names3, S3).tolist()) * sp.Matrix(law)) for c in cover]
    xz = tabs[2]; tmin = -min(xz[0], xz[3]); tmax = min(xz[1], xz[2]); t = tmin + (tmax - tmin) * sp.Rational(int(rng.integers(0, 101)), 100)
    tabs[2] = [xz[0] - t, xz[1] + t, xz[2] + t, xz[3] - t]          # same X and Z margins as before: locally compatible
    d = [v for tb in tabs for v in tb]
    flt = linprog(np.zeros(8), A_eq=Mi.astype(float), b_eq=np.array([float(v) for v in d]), bounds=[(0, None)] * 8, method="highs").status == 0
    kind, cert = exact_decide(Mi, d); agree &= kind is not None and (kind == "law") == flt
    nf += kind == "law"; ni += kind == "farkas"
check("EX1 IDENTITY exact rational witnesses and certificates agree with the floating-point verdict", agree, f"200 locally compatible rational instances: {nf} exact global laws, {ni} exact Farkas certificates")

# EX2
names4 = ("W", "X", "Y", "Z"); S4 = list(itertools.product((0, 1), repeat=4)); h = sp.Rational(1, 2)
corr, anti = [h, 0, 0, h], [0, h, h, 0]
loc = [(("W", "X"), corr), (("Y", "Z"), corr)]; out = [(("X", "Y"), corr), (("W", "Z"), anti)]
k1, x1 = exact_decide(np.vstack([cmat(c, names4, S4) for c, _ in loc]), [v for _, t in loc for v in t])
blocks = loc + out; k2, y2 = exact_decide(np.vstack([cmat(c, names4, S4) for c, _ in blocks]), [v for _, t in blocks for v in t])
supp = [] if y2 is None else ["".join(c) for k, (c, _) in enumerate(blocks) if any(y2[4 * k + j] != 0 for j in range(4))]
check("EX2 IDENTITY the twisted cycle: local tables extend (exact law); the twisted outer pair yields an exact Farkas certificate on the cycle",
      k1 == "law" and k2 == "farkas" and len(supp) == 4, f"certificate supported on {supp}")

# EX3, EX4 on the six-source hexagon of Study 27a
E6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
def ev_word(n): return [(E6[int(rng.integers(len(E6)))], int(rng.choice([1, -1]))) for _ in range(n)]
def normal_form_random_order(w):
    w = list(w)
    while True:
        cands = [(i, j) for i in range(len(w)) for j in range(i + 1, len(w)) if w[j][0] == w[i][0] and w[j][1] == -w[i][1]
                 and all(not O.dependent(w[k], w[i]) for k in range(i + 1, j))]
        if not cands: break
        i, j = cands[int(rng.integers(len(cands)))]; del w[j]; del w[i]
    L = O.layers(w); return tuple(tuple(sorted(e for e, l in zip(w, L) if l == t)) for t in range(1, max(L, default=0) + 1))
ok = True
for _ in range(300):
    w = ev_word(int(rng.integers(4, 12)))
    if rng.random() < 0.5:
        pos = int(rng.integers(len(w) + 1)); e = E6[int(rng.integers(len(E6)))]; w = w[:pos] + [(e, 1), (e, -1)] + w[pos:]
    ref = O.reduce(w)[0]; ok &= all(normal_form_random_order(w) == ref for _ in range(10))
check("EX3 IDENTITY the equality procedure for A_G gives one verdict whatever the order of cancellations", ok, "300 words x 10 orders")
def trace_key(w):
    L = O.layers(w); return tuple(tuple(sorted(e for e, l in zip(w, L) if l == t)) for t in range(1, max(L, default=0) + 1))
ok = True; classes = {}
for _ in range(500):
    w = ev_word(int(rng.integers(3, 10)))
    for _ in range(3):
        i = int(rng.integers(len(w) - 1))
        if not O.dependent(w[i], w[i + 1]): w[i], w[i + 1] = w[i + 1], w[i]
    classes.setdefault(trace_key(w), set()).add(tuple(sorted(O.activation(w, 0).items())))
ok = all(len(v) == 1 for v in classes.values())
wa = [((0, 1), 1), ((0, 1), -1), ((1, 2), 1)]; wb = [((1, 2), 1)]
ok &= O.reduce(wa)[0] == O.reduce(wb)[0] and O.activation(wa, 0) != O.activation(wb, 0)
check("EX4 IDENTITY the clock is a computable function on traces and does not descend to A_G", ok,
      f"{len(classes)} trace classes, activation constant on each; e e^-1 f and f are equal in A_G with activation {O.activation(wa, 0)} vs {O.activation(wb, 0)}")
print(f"{sum(results)}/{len(results)} passed")
