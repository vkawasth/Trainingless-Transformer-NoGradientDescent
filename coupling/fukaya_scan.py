"""fukaya_scan.py -- all critical points of PO at large lambda by elimination (resultant + mp.polyroots), and their
tropical locations u_k = -log|z_k| / lambda, compared with the leximin and product couplings.
Runs the K4 point of check_fukaya_probe.py and three random points of the hexagon chamber."""
import random
from collections import Counter
from fractions import Fraction as Fr
import numpy as np, sympy as sp, mpmath as mp
from scipy.optimize import linprog
N = {"00": (1, 0), "01": (0, 1), "02": (-1, -1), "10": (-1, 0), "11": (0, -1), "12": (1, 1)}; C = list(N)
def k0(f1, g1, g2): return {"00": 0, "01": 0, "02": f1, "10": g1, "11": g2, "12": -(f1 + g1 + g2 - 1)}
def alpha(u, p): k = k0(*p); return {c: N[c][0] * u[0] + N[c][1] * u[1] + k[c] for c in C}
def prod_u(p): return (p[0] * p[1], p[0] * p[2])
def leximin(p):
    k = k0(*[float(q) for q in p]); fixed = {}; Aeq, beq = [], []
    while True:
        free = [c for c in C if c not in fixed]
        r = linprog([0, 0, -1], A_ub=[[-N[c][0], -N[c][1], 1] for c in free], b_ub=[k[c] for c in free],
                    A_eq=Aeq or None, b_eq=beq or None, bounds=[(None, None)] * 3, method="highs")
        for c, l in zip(free, -r.ineqlin.marginals):
            if l > 1e-9: fixed[c] = r.x[2]; Aeq.append([N[c][0], N[c][1], 0]); beq.append(r.x[2] - k[c])
        if np.linalg.matrix_rank(np.array([N[c] for c in fixed])) == 2: return tuple(r.x[:2])
x, y = sp.symbols("x y"); a = {c: sp.Symbol("a" + c) for c in C}; asyms = [a[c] for c in C]
P = sum(a[c] * x ** N[c][0] * y ** N[c][1] for c in C)
E1 = sp.expand(sp.cancel(x * sp.diff(P, x) * x * y)); E2 = sp.expand(sp.cancel(y * sp.diff(P, y) * x * y))
Rx = sp.Poly(sp.resultant(E1, E2, y), x)
Rc = [sp.lambdify(asyms, cf, "mpmath") for cf in Rx.all_coeffs()]
Py = sp.Poly(E2, y); Pyc = [sp.lambdify((x, *asyms), cf, "mpmath") for cf in Py.all_coeffs()]
e1 = sp.lambdify((x, y, *asyms), E1, "mpmath")
def crit(p, lam):
    mp.mp.dps = int(lam * 0.7 / 2.3) + 80
    kk = k0(*[mp.mpf(q.numerator) / q.denominator for q in p]); A = [mp.e ** (-lam * kk[c]) for c in C]
    coeffs = [mp.mpf(f(*A)) for f in Rc]
    while coeffs and coeffs[-1] == 0: coeffs.pop()
    xs = mp.polyroots(coeffs, maxsteps=5000, extraprec=3 * mp.mp.dps)
    out = []
    for xr in xs:
        cy = [f(xr, *A) for f in Pyc]
        while cy and cy[-1] == 0: cy.pop()
        if len(cy) == 3:
            d = mp.sqrt(cy[1] ** 2 - 4 * cy[0] * cy[2]); ys = [(-cy[1] + d) / (2 * cy[0]), (-cy[1] - d) / (2 * cy[0])]
        else: ys = [-cy[1] / cy[0]]
        out.append((xr, min(ys, key=lambda yy: abs(e1(xr, yy, *A)))))
    return out
def ispos(z): return all(abs(mp.im(v)) < mp.mpf(10) ** -20 * abs(v) and mp.re(v) > 0 for v in z)
def loc(z, lam): return (round(-float(mp.log(abs(z[0]))) / lam, 3), round(-float(mp.log(abs(z[1]))) / lam, 3))
if __name__ == "__main__":
    print("resultant degree in x:", Rx.degree(), " lowest power", min(m[0] for m in Rx.monoms()))
    R = random.Random(5); pts = [(Fr(52, 100), Fr(31, 100), Fr(35, 100))]
    def sg(p):
        f = (p[0], 1 - p[0]); g = (p[1], p[2], 1 - p[1] - p[2])
        return tuple((f[i] > g[j]) - (f[i] < g[j]) for i in range(2) for j in range(3))
    hexs = sg((Fr(1, 2), Fr(1, 3), Fr(1, 3)))
    while len(pts) < 4:
        p = tuple(Fr(R.randint(1, 99), 100) for _ in range(3))
        if p[1] + p[2] < 1 and sg(p) == hexs and len(set(alpha(prod_u(p), p).values())) == 6: pts.append(p)
    allok = True
    for p in pts:
        print("p =", [str(q) for q in p], " leximin", tuple(round(v, 4) for v in leximin(p)),
              " product", tuple(round(float(v), 4) for v in prod_u(p)))
        for lam in (500, 2000):
            cps = crit(p, lam)
            print(f"   lam={lam}: {len(cps)} critical points, locations {dict(Counter(loc(z, lam) for z in cps))}, "
                  f"positive at {[loc(z, lam) for z in cps if ispos(z)]}", flush=True)
        L = [np.array(loc(z, lam)) for z in cps]; cl = []
        for v in L:
            for c in cl:
                if np.max(np.abs(c[0] - v)) < 3e-3: c[1] += 1; break
            else: cl.append([v, 1])
        lx, pr = np.array(leximin(p)), np.array([float(v) for v in prod_u(p)])
        big = [c for c in cl if c[1] == 4]
        pos = [np.array(loc(z, lam)) for z in cps if ispos(z)]
        allok &= (len(cps) == 6 and sorted(c[1] for c in cl) == [1, 1, 4] and len(big) == 1 and np.max(np.abs(big[0][0] - lx)) < 3e-3
                  and len(pos) == 1 and np.max(np.abs(pos[0] - lx)) < 3e-3 and all(np.max(np.abs(c[0] - pr)) > 3e-3 for c in cl))
    print(("PASS" if allok else "FAIL") + " S1 at 4 hexagon-chamber points: 6 critical points over 3 couplings (4+1+1); the 4-fold one and "
          "the positive critical point lie over the leximin coupling; the product coupling is not among them")
