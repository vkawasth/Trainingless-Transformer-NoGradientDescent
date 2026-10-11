"""Deviations, run after experiments.py (see FROZEN_RESULTS.txt). Same seed and the same random draws as experiments.py.
E1: order-free vertex-set comparison (the pre-registered comparator sorted rounded rows, which is order-sensitive).
E2b (exploratory): duals on margin equalities are defined only up to (u + c, v - c); the gauge-invariant check is whether the
binding floor-cell inequalities lie in the bottleneck row or column."""
import os, sys, json, numpy as np
from scipy.optimize import linprog
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import support as SP
rng = np.random.default_rng(20261012)
def rand_D(n, m):
    cells = [(x, y) for x in range(n) for y in range(m)]; k = rng.integers(1, len(cells) + 1); return [cells[i] for i in rng.choice(len(cells), k, replace=False)]
eq, worst, done = 0, 0.0, 0
while done < 500:
    n, m = int(rng.integers(2, 4)), int(rng.integers(2, 5)); f, g = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m)); D = rand_D(n, m)
    tau = np.zeros((n, m))
    for (x, y) in D: tau[x, y] = rng.uniform(0, 0.5 * min(f[x], g[y]) / max(n, m))
    fr, gr = SP.reduced_margins(f, g, tau, D)
    if min(fr.min(), gr.min()) <= 0: continue
    lower = tau * SP.mask(D, n, m); V1 = SP.vertices_bounded(f, g, lower); V2 = SP.vertices_bounded(fr, gr, np.zeros((n, m))) + lower.ravel()
    d = max(max(np.abs(V2 - p).max(1).min() for p in V1), max(np.abs(V1 - p).max(1).min() for p in V2)); worst = max(worst, d)
    eq += int(len(V1) == len(V2) and d < 1e-7); done += 1
k = 0
while k < 200:                                                    # advance the generator exactly as experiments.py does
    n, m = int(rng.integers(2, 4)), int(rng.integers(2, 5)); f, g = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m)); D = rand_D(n, m)
    tau = np.zeros((n, m))
    for (x, y) in D: tau[x, y] = rng.uniform(0, 0.6)
    fr, gr = SP.reduced_margins(f, g, tau, D)
    if min(fr.min(), gr.min()) >= 0: continue
    k += 1
ok, nb = 0, 0
for _ in range(1000):
    n, m = int(rng.integers(2, 5)), int(rng.integers(2, 5)); f, g = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m)); D = rand_D(n, m)
    tau, w = rng.uniform(0, .1, (n, m)), rng.uniform(.5, 2, (n, m)); h = SP.headroom(f, g, tau, w, D)
    if h['delta'] < 0 or len(h['ties']) != 1: continue
    N = n * m; A = np.hstack([SP._transport_eq(n, m), np.zeros((n + m, 1))]); Aub, bub = [], []
    for (x, y) in D: r = np.zeros(N + 1); r[x * m + y] = -1; r[-1] = w[x, y]; Aub.append(r); bub.append(-tau[x, y])
    c = np.zeros(N + 1); c[-1] = -1
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=A, b_eq=np.concatenate([f, g]), bounds=[(0, None)] * N + [(None, None)], method='highs')
    kind, idx = h['bottleneck']; act = [D[i] for i, l in enumerate(res.ineqlin.marginals) if abs(l) > 1e-9]
    nb += 1; ok += int(all((x == idx) if kind == 'row' else (y == idx) for x, y in act))
out = dict(E1_order_free=dict(equal_within_1e7=eq, of=500, worst_hausdorff=worst, holds=bool(eq == 500)),
           E2b_gauge_invariant_exploratory=dict(binding_floors_in_bottleneck=ok, of=nb, share=round(ok / nb, 4)))
json.dump(out, open(os.path.join(H, 'results', 'support_rechecks.json'), 'w'), indent=1); print(out)
