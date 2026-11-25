"""EXPLORATORY (not a registered check): is the credal lower envelope (min over the transportation polytope) 2-monotone on cell sets?"""
import itertools, numpy as np
from scipy.optimize import linprog
def lower(f, g, S, r, c):
    A = np.zeros((r + c, r * c)); 
    for i in range(r):
        for j in range(c): A[i, i * c + j] = 1; A[r + j, i * c + j] = 1
    cobj = np.array([1.0 if k in S else 0.0 for k in range(r * c)]); res = linprog(cobj, A_eq=A, b_eq=np.r_[f, g], bounds=(0, None), method="highs")
    return res.fun
rng = np.random.default_rng(0)
for r, c in ((2, 2), (2, 3), (3, 3)):
    bad = tot = 0; worst = 0
    for trial in range(20 if r * c < 9 else 3):
        f = rng.dirichlet([2] * r); g = rng.dirichlet([2] * c); n = r * c
        subs = [frozenset(s) for k in range(n + 1) for s in itertools.combinations(range(n), k)]
        if n == 9: subs = [subs[i] for i in rng.choice(len(subs), 120, replace=False)] + [frozenset(), frozenset(range(n))]
        P = {S: lower(f, g, S, r, c) for S in subs}
        for S, T in itertools.combinations(subs, 2):
            if S | T in P and S & T in P:
                tot += 1; gap = P[S | T] + P[S & T] - P[S] - P[T]
                if gap < -1e-9: bad += 1; worst = min(worst, gap)
    print(f"{r}x{c}: pairs tested {tot}, 2-monotonicity violations {bad}, worst gap {worst:.3f}")
