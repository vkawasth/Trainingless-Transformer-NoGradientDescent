import numpy as np, sympy as sp
def glue(a, mid, b): return a @ np.diag(1 / mid) @ b
def w(h, a):   # copy whiskering: product action plus copy correction on equal pairs
    return h.T @ a @ h + sum(a[y, y] * (np.diag(h[y]) - np.outer(h[y], h[y])) for y in range(a.shape[0]))
def pi(h, a): return h.T @ a @ h
f = np.array([0.5, 0.5]); S = np.array([[0, .5], [.5, 0]]); Dg = np.diag(f)
def defect(h, a, b):
    mid = h.T @ f
    return glue(w(h, a), mid, w(h, b)) - w(h, glue(a, f, b))
for e in (0.0, 0.1, 0.25, 0.5):
    h = np.array([[1 - e, e], [e, 1 - e]]); d = defect(h, S, S)
    print("eps", e, "defect entry t =", round(abs(d[0, 1]), 6))
# closed form in eps
e = sp.symbols('e'); h = sp.Matrix([[1 - e, e], [e, 1 - e]]); Ss = sp.Matrix([[0, sp.Rational(1, 2)], [sp.Rational(1, 2), 0]])
P = h.T * Ss * h; mid = sp.diag(sp.Rational(1, 2), sp.Rational(1, 2))
t = sp.factor(sp.simplify((P * mid.inv() * P)[0, 1]))
print("t(eps) =", t, " at 0.1:", t.subs(e, sp.Rational(1, 10)))
# defect over the band of 2-cells: alpha_s = [[s, 1/2-s],[1/2-s, s]]
h = np.array([[0.9, 0.1], [0.1, 0.9]]); best = 0; zeros = 0
grid = np.linspace(0, 0.5, 51)
vals = np.array([[abs(defect(h, np.array([[s, .5 - s], [.5 - s, s]]), np.array([[r, .5 - r], [.5 - r, r]]))[0, 1]) for r in grid] for s in grid])
print("max defect over the band:", vals.max(), "at", np.unravel_index(vals.argmax(), vals.shape), "zero set size:", (vals < 1e-12).sum())
# constant kernel: defect = (tr a tr b - tr(a.b)) E
q = np.array([0.3, 0.7]); hc = np.tile(q, (2, 1)); E = np.diag(q) - np.outer(q, q)
ok = True
for _ in range(100):
    s, r = np.random.rand(2) * 0.5
    a = np.array([[s, .5 - s], [.5 - s, s]]); b = np.array([[r, .5 - r], [.5 - r, r]])
    d = defect(hc, a, b); ok &= np.allclose(d, (np.trace(a) * np.trace(b) - np.trace(glue(a, f, b))) * E)
print("constant kernel formula:", ok)
# unitality vs healing: w_h maps the diagonal coupling of f to the diagonal of h^T f; pi_h heals it when h > 0
hpos = np.array([[0.8, 0.2], [0.3, 0.7]])
print("w_h(diag f) =", np.round(w(hpos, Dg), 4).tolist(), " pi_h(diag f) =", np.round(pi(hpos, Dg), 4).tolist())
