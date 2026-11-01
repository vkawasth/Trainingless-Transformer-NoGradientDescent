"""Order three at corners (companion "A tangent structure for injected data", Section "Higher order at corners"). New module.

  D3(fy, x, n)            the third derivative of one output cell at x (exact, from W_3 points)
  crit3(f, x)             the conjectured order-3 criterion: the second-order criterion, and D^3 f_y >= 0 on every triple of generators of
                          K_y (rays, and both signs of lineality directions) whose pairwise Hessian values all vanish
  direct3(f, x, rng)      the direct test: T^3 f of sampled points of T^3 Delta stays in T^3 Delta (lexicographic membership)
  rand_cubic(rng)         random cubic self-maps of Delta_3
"""
import itertools
import numpy as np
from . import tangent as TG, tangent_decide as TD
from .tangent import HD
grid3 = [np.array(q) for q in itertools.product(np.linspace(0, 1, 21), repeat=3) if abs(sum(q) - 1) < 1e-9]
PTS3 = [np.array(p, float) for p in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.3, .3, .4)]]
def rand_cubic(rng):
    forms = []
    for _ in range(2):
        t = rng.integers(4)
        if t == 0: forms.append(("cube", rng.integers(-2, 3, 3).astype(float)))
        elif t == 1: forms.append(("prod3", tuple(rng.choice(3, 3, replace=True))))
        elif t == 2: forms.append(("sqlin", (int(rng.integers(3)), rng.integers(-2, 3, 3).astype(float))))
        else: forms.append(("prod2s", tuple(rng.choice(3, 2, replace=False))))
    def ev(fm, X):
        k, p = fm; s = X[0] + X[1] + X[2]
        if k == "cube": z = X[0] * p[0] + X[1] * p[1] + X[2] * p[2]; return z * z * z
        if k == "prod3": return X[p[0]] * X[p[1]] * X[p[2]]
        if k == "sqlin": i, q = p; z = X[0] * q[0] + X[1] * q[1] + X[2] * q[2]; return X[i] * X[i] * (z * z) if False else X[i] * X[i] * s
        return X[p[0]] * X[p[1]] * s
    def evs(fm, q):
        v = ev(fm, HD.var(q, [])).c[0]; return v
    vals = [sum(abs(evs(fm, q)) for fm in forms) for q in grid3]; a = 0.9 / max(vals) if max(vals) > 0 else 1.0
    def f(X):
        o = [ev(fm, X) * a for fm in forms]; s = X[0] + X[1] + X[2]; return [s * s * s - o[0] - o[1], o[0], o[1]]
    # require image in the simplex on the grid
    for q in grid3:
        y = [v.c[0] for v in f(HD.var(q, []))]
        if min(y) < -1e-12: return None
    return f
def D3(fy, x, n):
    I = np.eye(n); T = np.zeros((n, n, n))
    for i, j, l in itertools.product(range(n), repeat=3): T[i, j, l] = fy(HD.var(x, [I[i], I[j], I[l]])).c[7]
    return T
def crit3(f, x, tol=1e-9):
    ok2, _ = TD.second_order_ok(f, x)
    if not ok2: return False
    n = len(x); vals = TD.outputs(f, HD.var(x, []))[0]
    for y in range(len(vals)):
        if vals[y] > tol: continue
        fy = lambda Xh, y=y: f(Xh)[y]; g, H = TD.jets_scalar(fy, x, n); L, R = TD.cone_K(x, g); T = D3(fy, x, n)
        G = R + L + [-l for l in L]
        for a, b, c in itertools.product(G, repeat=3):
            if abs(a @ H @ b) < tol and abs(a @ H @ c) < tol and abs(b @ H @ c) < tol:
                if np.einsum("ijk,i,j,k", T, a, b, c) < -tol: return False
    return True
def direct3(f, x, rng, samples=300, tries=30000):
    n = len(x); tested = 0
    for _ in range(tries):
        c = np.zeros((8, n)); c[0] = x
        for S in range(1, 8):
            z = rng.integers(-2, 3, size=n).astype(float); z[-1] -= z.sum(); c[S] = z
        if not TG.in_T(c, 3): continue
        tested += 1
        if not TG.in_T(TD.outputs(f, HD(c, 3)), 3): return False
        if tested >= samples: break
    return True
