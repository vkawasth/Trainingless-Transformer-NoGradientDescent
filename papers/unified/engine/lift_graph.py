"""EXPLORATORY: check prop:liftgraph on K4 (cycle rank 3) and the 'theta' graph (cycle rank 2), random no-signalling binary data."""
import itertools, numpy as np
from scipy.optimize import linprog
rng = np.random.default_rng(1125)
def run(V, E, T, trials=200):
    agree = []; nctx = 0
    for _ in range(trials):
        s = {v: rng.uniform(0.2, 0.8) for v in V}; D = {}
        for (a, b) in E:
            lo, hi = max(0, s[a] + s[b] - 1), min(s[a], s[b]); c = rng.uniform(lo, hi)
            D[(a, b)] = {(1, 1): c, (1, 0): s[a] - c, (0, 1): s[b] - c, (0, 0): 1 - s[a] - s[b] + c}
        # CF
        G = list(itertools.product((0, 1), repeat=len(V))); A = []; b = []
        for (a, bb), P in D.items():
            for (x, y), p in P.items(): A.append([1.0 if (g[V.index(a)] == x and g[V.index(bb)] == y) else 0.0 for g in G]); b.append(p)
        cf = 1 + linprog(-np.ones(len(G)), A_ub=A, b_ub=b, bounds=(0, None), method="highs").fun
        # lift: variables V plus copies for non-tree edges
        cut = [e for e in E if e not in T]; W = V + [f"{e[1]}'{k}" for k, e in enumerate(cut)]; L = list(itertools.product((0, 1), repeat=len(W))); A = []; b = []
        for (a, bb), P in D.items():
            other = bb if (a, bb) in T else f"{bb}'{cut.index((a, bb))}"
            for (x, y), p in P.items(): A.append([1.0 if (g[W.index(a)] == x and g[W.index(other)] == y) else 0.0 for g in L]); b.append(p)
        obj = [1.0 if any(g[W.index(f"{e[1]}'{k}")] != g[W.index(e[1])] for k, e in enumerate(cut)) else 0.0 for g in L]
        lf = linprog(obj, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
        agree.append(abs(lf - cf)); nctx += cf > 1e-9
    return max(agree), nctx
V = ["a", "b", "c", "d"]
K4 = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]; T4 = [("a", "b"), ("a", "c"), ("a", "d")]
TH = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")]; TT = [("a", "b"), ("b", "c"), ("c", "d")]
for nm, E, T in (("K4 (rank 3)", K4, T4), ("theta (rank 2)", TH, TT)):
    m, n = run(V, E, T); print(f"{nm}: max |lift defect - CF| = {m:.1e}; contextual cases {n}/200")
