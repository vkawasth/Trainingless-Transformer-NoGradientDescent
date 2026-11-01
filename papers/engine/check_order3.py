"""Checks for order three at corners (companion "A tangent structure for injected data", Section "Higher order at corners").
  O31  quadratic maps (D^3 = 0): the second-order criterion decides the direct T^3 test as well (random quadratic self-maps of Delta_3)
  O32  a cubic map that passes at second order and fails at third: f_1 = X_1 (X_1 - X_2)^2 at the vertex (1,0,0). Its third derivative is
       non-negative on each ray (D^3(r,r,r) >= 0, as positivity forces) but negative on a mixed triple, D^3(r1,r1,r2) < 0
  O33  the order-3 criterion (second order, and D^3 >= 0 on generator triples with vanishing pairwise Hessian) agrees with the direct T^3
       test on random cubic maps and on the designed maps
"""
import itertools, os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import order3 as O3, tangent_decide as TD
from pce.tangent import HD
rng = np.random.default_rng(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
grid3 = O3.grid3; PTS3 = O3.PTS3
def rand_quad():
    forms = []
    for _ in range(2):
        t = rng.integers(3)
        forms.append(("sq", rng.integers(-2, 3, 3).astype(float)) if t == 0 else ("prod", tuple(rng.choice(3, 2, replace=False))) if t == 1 else ("sqi", int(rng.integers(3))))
    def ev(fm, X):
        k, p = fm
        if k == "sq": z = X[0] * p[0] + X[1] * p[1] + X[2] * p[2]; return z * z
        if k == "prod": return X[p[0]] * X[p[1]]
        return X[p] * X[p]
    mx = max(sum(ev(fm, HD.var(q, [])).c[0] for fm in forms) for q in grid3); a = 0.9 / mx if mx > 0 else 1.0
    def f(X):
        o = [ev(fm, X) * a for fm in forms]; s = X[0] + X[1] + X[2]; return [s * s - o[0] - o[1], o[0], o[1]]
    return f
# O31
agree = tot = fails = 0
for _ in range(20):
    f = rand_quad()
    for x in PTS3:
        ok2 = TD.second_order_ok(f, x)[0]; d = O3.direct3(f, x, rng, samples=150); tot += 1; agree += ok2 == d; fails += not d
check("O31 quadratic maps: the second-order criterion decides T^3 as well", agree == tot and fails > 0, f"{agree}/{tot} agree, {fails} fail")
# O32
x = np.array([1.0, 0, 0]); f = lambda X: (lambda s, g: [s * s * s - g, g, X[0] * 0.0])(X[0] + X[1] + X[2], 0.2 * X[1] * (X[1] - X[2]) * (X[1] - X[2]))
T = O3.D3(lambda X: f(X)[1], x, 3); r1, r2 = np.array([-1.0, 1, 0]), np.array([-1.0, 0, 1])
diag = [float(np.einsum("ijk,i,j,k", T, r, r, r)) for r in (r1, r2)]; mixed = float(np.einsum("ijk,i,j,k", T, r1, r1, r2))
ok2 = TD.second_order_ok(f, x)[0]; d3 = O3.direct3(f, x, rng, samples=600, tries=200000)
OUT["O32"] = dict(diag=diag, mixed=mixed)
check("O32 X1(X1-X2)^2 at a vertex: passes order two, fails order three; D^3 >= 0 on each ray, < 0 on a mixed triple",
      ok2 and not d3 and min(diag) >= 0 and mixed < 0, f"D3 on rays {diag}, mixed {mixed:.2f}")
# O33
agree = tot = 0
while tot < 70:
    f = O3.rand_cubic(rng)
    if f is None: continue
    for x in PTS3:
        tot += 1; agree += O3.crit3(f, x) == O3.direct3(f, x, rng, samples=150)
designs = [lambda X: (lambda s, g: [s * s * s - g, g, X[0] * 0.0])(X[0] + X[1] + X[2], 0.2 * e(X)) for e in
           (lambda X: X[1] * (X[1] - X[2]) * (X[1] - X[2]), lambda X: X[1] * X[1] * X[1] + X[2] * X[2] * X[2], lambda X: X[1] * X[2] * (X[1] + X[2]),
            lambda X: (X[1] - X[2]) * (X[1] - X[2]) * X[0], lambda X: X[1] * X[1] * X[2])]
dag = [O3.crit3(g, np.array([1.0, 0, 0])) == O3.direct3(g, np.array([1.0, 0, 0]), rng, samples=600, tries=200000) for g in designs]
check("O33 the order-3 criterion agrees with the direct T^3 test (random cubic maps and five designed maps)", agree == tot and all(dag), f"{agree}/{tot} random, {sum(dag)}/5 designed")
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "order3.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
