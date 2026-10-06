"""check_fukaya_probe.py -- (5) probing the Fukaya-type functor (Conj 9.5) on the 2x3 hexagon chamber (dP6).

Potential (FOOO, toric Fano):  PO(u; w) = sum_c exp(-lam alpha_c(u) + <n_c, w>),  n_c the cell normals,
alpha_c(u) the cell functionals at the coupling u = (alpha_00, alpha_01), T = e^{-lam}.  A critical point w of
PO(u; .) whose real part stays bounded as lam -> infinity (val = 0) makes the fibre L(u) Floer-nontrivial; the
imaginary part is the holonomy.  Candidate object map of the conjectured functor: (L(u), holonomy) -> coupling u.

  K1 translation covariance (exact identity): PO(u; w) = PO(0; w - lam u), so the critical points of PO(u; .) are those
     of PO(0; .) shifted by -lam u; the Floer location u_k = -Re(w_k)/lam does not depend on the base fibre
  K2 monotone point f = (1/2,1/2), g = (1/3,1/3,1/3): all six alpha_c equal 1/6 at the product coupling; W = x+y+1/x+1/y+
     xy+1/(xy) has 6 critical points, all |x| = |y| = 1, values 6, -2, -2, -2, -3, -3, all located at the product coupling
  K3 Hessian vs Fisher at the monotone point: Hess_log W at (1,1) is a positive multiple of the Fisher metric
     sum n n^T / alpha_c at the product coupling; at the other five critical points it is not a real multiple
  K4 off the monotone point (f1,g1,g2) = (0.52,0.31,0.35) in the hexagon chamber: tracking the 6 critical points to
     lam = 2560, they converge to three couplings: the leximin coupling (4 critical points) and two further couplings;
     the product coupling is not among them
  K5 at each limit coupling the minimal cells, with signs from the holonomy, balance exactly: sum_c sign_c n_c = 0
     (rational check); the leximin coupling has trivial holonomy among its critical points (the positive real one)
  K6 Hessian of PO at the positive critical point is not proportional to the Fisher metric at the product coupling:
     the weights are Boltzmann weights exp(-lam alpha_c), not 1/alpha_c (proportional only at the monotone point)
"""
import random
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.optimize import linprog
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

N = {"00": (1, 0), "01": (0, 1), "02": (-1, -1), "10": (-1, 0), "11": (0, -1), "12": (1, 1)}
C = list(N)
def k0(f1, g1, g2): return {"00": 0, "01": 0, "02": f1, "10": g1, "11": g2, "12": -(f1 + g1 + g2 - 1)}
def alpha(u, p): k = k0(*p); return {c: N[c][0] * u[0] + N[c][1] * u[1] + k[c] for c in C}
def prod_u(p): return (p[0] * p[1], p[0] * p[2])

# K1
lam, u1, u2, w1, w2 = sp.symbols("lam u1 u2 w1 w2"); ps = sp.symbols("f1 g1 g2")
PO = lambda u, w: sum(sp.exp(-lam * alpha(u, ps)[c] + N[c][0] * w[0] + N[c][1] * w[1]) for c in C)
check("K1 PO(u; w) = PO(0; w - lam u): Floer locations u_k = -Re(w_k)/lam are independent of the base fibre",
      sp.simplify(PO((u1, u2), (w1, w2)) - PO((0, 0), (w1 - lam * u1, w2 - lam * u2))) == 0)

# K2 monotone
x, y = sp.symbols("x y")
W = x + y + 1 / x + 1 / y + x * y + 1 / (x * y)
sols = sp.solve([sp.numer(sp.together(sp.diff(W, x) * x)), sp.numer(sp.together(sp.diff(W, y) * y))], [x, y], dict=True)
sols = [s for s in sols if s[x] != 0 and s[y] != 0]
vals = sorted([complex(sp.N(W.subs(s))).real for s in sols])
mono = (Fr(1, 2), Fr(1, 3), Fr(1, 3))
eq = set(alpha(prod_u(mono), mono).values())
check("K2 monotone dP6: product coupling has all alpha_c = 1/6; 6 critical points, |x| = |y| = 1, values 6,-2,-2,-2,-3,-3",
      eq == {Fr(1, 6)} and len(sols) == 6 and all(abs(abs(complex(sp.N(s[v]))) - 1) < 1e-12 for s in sols for v in (x, y))
      and np.allclose(vals, [-3, -3, -2, -2, -2, 6]), f"values {[round(v, 6) for v in vals]}")

# K3 Hessian vs Fisher
nn = [np.outer(N[c], N[c]) for c in C]
Fisher = sum(m / float(Fr(1, 6)) for m in nn)
def hess(z): return sum(complex(sp.N(z[x] ** N[c][0] * z[y] ** N[c][1])) * m for c, m in zip(C, nn))
def real_multiple(H, G):
    r = np.trace(H @ G.T) / np.trace(G @ G.T); return np.allclose(H, r * G, atol=1e-10) and abs(r.imag) < 1e-12 and r.real > 0, r
flags = []
for s in sols:
    ok, r = real_multiple(hess(s), Fisher); flags.append((ok, complex(sp.N(s[x])), complex(sp.N(s[y])), r))
good = [f for f in flags if f[0]]
check("K3 Hess_log W is a positive multiple of the Fisher metric at the product coupling exactly at the critical point (1,1)",
      len(good) == 1 and abs(good[0][1] - 1) < 1e-12 and abs(good[0][2] - 1) < 1e-12, f"ratio {good[0][3].real:.6f} = 1/6")

# K4 off the monotone point
p = (Fr(52, 100), Fr(31, 100), Fr(35, 100)); pf = tuple(mp.mpf(q.numerator) / q.denominator for q in p)
kk = k0(*pf)
def G(L):
    return lambda a, b: [sum(mp.e ** (-L * kk[c] + N[c][0] * a + N[c][1] * b) * N[c][i] for c in C) for i in (0, 1)]
mp.mp.dps = 40; L = mp.mpf(40); Rg = random.Random(0); roots = []
for _ in range(150):
    st = [mp.mpc(-Rg.uniform(0.05, 0.3) * L, Rg.uniform(-3.2, 3.2)) for _ in range(2)]
    try: r = mp.findroot(G(L), st, tol=mp.mpf(10) ** -25, maxsteps=100)
    except Exception: continue
    r = [r[0], r[1]]
    key = tuple(round(float(v), 5) for v in (r[0].real, r[1].real, mp.cos(r[0].imag), mp.sin(r[0].imag), mp.cos(r[1].imag), mp.sin(r[1].imag)))
    if key not in [q[0] for q in roots]: roots.append((key, r))
roots = [q[1] for q in roots]
Lt = mp.mpf(2560); tracked = []
for r in roots:
    prev, cur = L, r
    while prev < Lt:
        nxt = min(prev * mp.mpf("1.25"), Lt)
        z = mp.findroot(G(nxt), [mp.mpc(cur[i].real * nxt / prev, cur[i].imag) for i in (0, 1)], tol=mp.mpf(10) ** -20, maxsteps=100)
        cur, prev = [z[0], z[1]], nxt
    tracked.append(cur)
locs = [(-float(r[0].real / Lt), -float(r[1].real / Lt)) for r in tracked]
signs_ = [tuple(int(round(float(mp.cos(r[i].imag)))) if abs(float(mp.sin(r[i].imag))) < 1e-6 else 0 for i in (0, 1)) for r in tracked]
# leximin by iterated LP
def leximin(p):
    k = k0(*[float(q) for q in p]); fixed = {}; Aeq, beq = [], []
    while True:
        free = [c for c in C if c not in fixed]
        r = linprog([0, 0, -1], A_ub=[[-N[c][0], -N[c][1], 1] for c in free], b_ub=[k[c] for c in free],
                    A_eq=Aeq or None, b_eq=beq or None, bounds=[(None, None)] * 3, method="highs")
        for c, l in zip(free, -r.ineqlin.marginals):
            if l > 1e-9: fixed[c] = r.x[2]; Aeq.append([N[c][0], N[c][1], 0]); beq.append(r.x[2] - k[c])
        if np.linalg.matrix_rank(np.array([N[c] for c in fixed])) == 2: return tuple(r.x[:2])
ulex = leximin(p); uprod = tuple(float(v) for v in prod_u(p))
clusters = {}
for l, sg in zip(locs, signs_):
    key = (round(l[0], 4), round(l[1], 4)); clusters.setdefault(key, []).append(sg)
mult_lex = sum(len(v) for kq, v in clusters.items() if np.allclose(kq, ulex, atol=2e-4))
check("K4 the 6 critical points (tracked to lam = 2560) sit over 3 couplings: leximin (x4) and two others; the product "
      "coupling is not among them", len(roots) == 6 and len(clusters) == 3 and mult_lex == 4
      and all(np.hypot(kq[0] - uprod[0], kq[1] - uprod[1]) > 5e-3 for kq in clusters),
      f"leximin {tuple(round(v, 4) for v in ulex)}, product {tuple(round(v, 4) for v in uprod)}, clusters {list(clusters)}")

# K5 exact signed balancing at each limit coupling
ok = True; desc = []
for kq, sgs in clusters.items():
    u = (Fr(kq[0]).limit_denominator(400), Fr(kq[1]).limit_denominator(400)); a = alpha(u, p); m = min(a.values())
    act = [c for c in C if a[c] == m]
    real_sgs = [s for s in sgs if 0 not in s]
    bal = False
    for s in real_sgs:
        tot = [sum((s[0] ** N[c][0]) * (s[1] ** N[c][1]) * N[c][i] for c in act) for i in (0, 1)]
        bal |= tot == [0, 0]
    if np.allclose(kq, ulex, atol=2e-4):
        bal = bal and (1, 1) in real_sgs
        # leximin balancing is the LP dual: positive multipliers on the active cells sum the normals to 0
    ok &= bal; desc.append(f"{u}: min {m} on {act}, holonomy {real_sgs}")
check("K5 at each limit coupling the minimal cells balance with holonomy signs (sum sign_c n_c = 0, exact); the leximin "
      "coupling carries the trivial-holonomy (positive) critical point", ok, "; ".join(desc))

# K6 Hessian at the positive critical point vs Fisher at the product coupling (lam = 40)
mp.mp.dps = 40
pos = [r for r in roots if all(abs(mp.sin(r[i].imag)) < 1e-12 and mp.cos(r[i].imag) > 0 for i in (0, 1))][0]
H = np.array([[float(sum(mp.e ** (-40 * kk[c] + N[c][0] * pos[0].real + N[c][1] * pos[1].real) * N[c][i] * N[c][j] for c in C))
               for j in (0, 1)] for i in (0, 1)])
ap = alpha(prod_u(p), p); Fp = sum(np.outer(N[c], N[c]) / float(ap[c]) for c in C)
cosang = np.trace(H @ Fp) / np.linalg.norm(H) / np.linalg.norm(Fp)
check("K6 off the monotone point the Hessian of PO (Boltzmann weights e^{-lam alpha}) is not proportional to the Fisher "
      "metric (weights 1/alpha) at the product coupling", cosang < 1 - 1e-4, f"cosine between the two forms = {cosang:.6f}")
print(f"\n{sum(res)}/{len(res)} checks passed")
