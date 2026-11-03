"""Checks for D1, "soft evidence moves lambda" (module pce/datawalls.py; companion note kahler/conditioned_states_kahler.tex, section 6).
  DW1  the convention: multiplying term k of W_T (T = e^-N) by e^(-n rho_k) is the shift lam -> lam - (n/N) rho, and the positive critical point
       of the multiplied potential is that of the shifted one
  DW2  LP duality: max_u min_k l_k = -max {<q, lam> : q a law on the facets with mean exponent 0} (40 random positively spanning configurations);
       the support of the optimal face is K(u) at a relative-interior point of argmax min l, and dim argmax = n - dim sigma_0
  DW3  the limit law of the positive critical point lies in the optimal face (CP^2, F_1 at alpha = 1/5, 3/5, 20 random configurations)
  DW4  the walls of F_1: one circuit a = (2, -1, 2, -3); the outcome wall c(lam) = -<a, lam> = 0 is alpha / L = 1/3, inside the Kahler cone
       0 < alpha / L < 1; the lower diagonal is v1 v3 below it and v2 v4 above it (exact rationals); CP^2 has no circuit and one chamber
  DW5  actual selection against candidates across alpha = 1/10, ..., 9/10: candidate dimension 0 below the wall and 1 above; the valuation sits in
       the candidate set and equals (1/3, 1/3) or ((1 + alpha)/4, (1 - alpha)/2), which agree at 1/3; at alpha = 1/3 the positive law is
       (1/x, x^2, 1/x, 1/x^2) / sum with x^4 = x + 1 for every T, an interior point of the balance segment
  DW6  the discriminant of w^4 - T^(1/2) w - T^(1 - alpha) is -27 T^2 - 256 T^(3 - 3 alpha): never zero, and its two terms tie exactly at alpha = 1/3
  DW7  data paths from alpha = 1/5: evidence r = e1 reaches the wall at s* = 1/5 (n* = N/5) and u*(s) is affine on each side and continuous;
       r = e4 and r = (0, -1, 0, 1) never reach it; translations and the constant direction never change the leading cell (u* -> u* - s t,
       h* -> h* + s); the least evidence with rate range <= 1 is s = |c| / (||a||_1 / 2) = 1/10, at r = (1, 0, 1, 0); on CP^2 no direction crosses
"""
import os, sys, random
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import kahlerbridge as KB, datawalls as DW
random.seed(20261103); rng = np.random.default_rng(20261103); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rconf():
    while True:
        m = random.randint(3, 6); v = sorted({(random.randint(-2, 2), random.randint(-2, 2)) for _ in range(m)} - {(0, 0)})
        if len(v) < 3: continue
        V = np.array(v, float).T; from scipy.optimize import linprog
        r = linprog(np.zeros(len(v)), A_eq=V, b_eq=np.zeros(2), bounds=[(1, None)] * len(v), method="highs")   # 0 interior: positive spanning
        if r.status == 0 and np.linalg.matrix_rank(V) == 2: return v, list(rng.normal(0, 0.6, len(v)))

# DW1
ok = True
for _ in range(10):
    v, lam = rconf(); N = random.uniform(20, 60); n = random.uniform(1, 40); rho = rng.uniform(0, 1, len(v)); T = np.exp(-N)
    coef = np.exp(-np.array(lam) * np.log(T)) * np.exp(-n * rho); lam2 = DW.shift(lam, rho, n / N)
    ok &= np.allclose(np.log(coef), -lam2 * np.log(T), rtol=1e-12, atol=1e-9)
    b1, w1 = KB.positive_crit(v, lam2, T); ok &= np.allclose(np.array(v, float).T @ w1, 0, atol=1e-8)
check("DW1 soft evidence e^(-n rho) on T = e^-N is the shift lam - (n/N) rho", ok)

# DW2
ok = True
for _ in range(40):
    v, lam = rconf(); h1, _ = KB.maxmin(v, lam); h2, q, supp = DW.balance_lp(v, lam); ok &= abs(h1 - h2) < 1e-9
    V = np.array(v, float); P = []
    for d in rng.normal(size=(12, 2)):
        from scipy.optimize import linprog
        P.append(linprog(-d, A_ub=-V, b_ub=-(np.array(lam) + h2) + 1e-12, bounds=[(None, None)] * 2, method="highs").x)
    c = np.mean(P, 0); ok &= KB.ties(v, lam, c, tol=1e-7) == supp
    dsig = np.linalg.matrix_rank(V[list(supp)][1:] - V[supp[0]]); ok &= DW.argmax_dim(v, lam) == 2 - dsig
check("DW2 h* = -max over balanced laws of <q, lam>; optimal support = leading set at the centre of argmax; dim argmax = 2 - dim sigma_0", ok)

# DW3
ok = True; cases = [KB.CP2, KB.F1(0.2), KB.F1(0.6)] + [rconf() for _ in range(20)]
for v, lam in cases:
    h, q, supp = DW.balance_lp(v, lam); b, w = KB.positive_crit(v, lam, 1e-40)
    ok &= abs(-(w @ np.array(lam)) - h) < 2e-2 * max(1, abs(h)) and set(np.where(w > 1e-2)[0]) <= set(supp)
check("DW3 the limit law of the positive critical point lies in the optimal face of the balance polytope (23 cases)", ok)

# DW4
ok = True; a = DW.circuits(KB.F1(0.2)[0]); ok &= a == [[2, -1, 2, -3]] and DW.circuits(KB.CP2[0]) == []
for al in [Fr(k, 30) for k in range(1, 30)]:
    lam = [Fr(0), Fr(0), Fr(-1), -(1 - al)]; c = -sum(x * y for x, y in zip(a[0], lam)); ok &= c == 3 * al - 1
    hgt = [-x for x in lam]; d13 = (hgt[0] + hgt[2]) / 2; d24 = (3 * hgt[3] + hgt[1]) / 4      # heights over the crossing point (0, -1/2)
    ok &= (d13 < d24) == (al < Fr(1, 3)) and (d13 == d24) == (al == Fr(1, 3))
    Lq = -(lam[0] + lam[1] + lam[2]); aq = -lam[0] - lam[2] + lam[3]; ok &= Lq == 1 and aq == al
ok &= DW.balance_lp(*KB.F1(1 / 3))[2] == (0, 1, 2, 3)
for _ in range(200):
    lam = list(rng.normal(0, 1, 3)); ok &= DW.balance_lp(KB.CP2[0], lam)[2] == (0, 1, 2)
check("DW4 F_1: one circuit (2,-1,2,-3); outcome wall alpha/L = 1/3 inside the Kahler cone 0 < alpha/L < 1; lower diagonal flips there; CP^2 one chamber", ok)

# DW5
ok = True; rows = []
for al in [k / 10 for k in range(1, 10)]:
    v, lam = KB.F1(al); dim = DW.argmax_dim(v, lam); exp = (1 / 3, 1 / 3) if al < 1 / 3 else ((1 + al) / 4, (1 - al) / 2)
    val = KB.valuation(v, lam, [1e-30])[0]; ok &= dim == (0 if al < 1 / 3 else 1) and np.allclose(val, exp, atol=2e-3)
    ok &= KB.balanced(v, KB.ties(v, lam, exp)); rows.append((al, dim, tuple(np.round(val, 3))))
ok &= np.allclose((1 / 3, 1 / 3), ((1 + 1 / 3) / 4, (1 - 1 / 3) / 2))
mp.mp.dps = 40; x = mp.findroot(lambda t: t ** 4 - t - 1, 1.2); wx = np.array([1 / x, x ** 2, 1 / x, 1 / x ** 2], float); wx /= wx.sum()
for T in (1e-20, 1e-60):
    _, w = KB.positive_crit(*KB.F1(1 / 3), T); ok &= np.allclose(w, wx, atol=1e-6)
t = 2 * wx[3]; ok &= 0 < t < 1 and np.allclose(wx, (1 - t) * np.array([1 / 3, 1 / 3, 1 / 3, 0]) + t * np.array([0, 0.5, 0, 0.5]), atol=1e-9)
check("DW5 candidates vs actual valuation across alpha; at the wall the positive law is fixed by x^4 = x + 1, inside the balance segment", ok,
      f"{rows}; wall law {np.round(wx, 4).tolist()}, segment parameter t = {t:.4f}")

# DW6
import sympy as sp
w_, A_, B_ = sp.symbols("w A B", positive=True); D = sp.discriminant(w_ ** 4 - A_ * w_ - B_, w_)
ok = sp.expand(D - (-27 * A_ ** 4 - 256 * B_ ** 3)) == 0
ok &= all((Fr(2) == 3 - 3 * al) == (al == Fr(1, 3)) for al in [Fr(k, 30) for k in range(1, 30)])
check("DW6 discriminant = -27 A^4 - 256 B^3 = -27 T^2 - 256 T^(3 - 3 alpha) < 0: no collision; its terms tie exactly at alpha = 1/3", ok, f"{D}")

# DW7
ok = True; v, lam0 = KB.F1(0.2)
def u_pred(lam):
    l1, l2, l3, l4 = lam
    if DW.balance_lp(v, lam)[2] == (0, 1, 2): return np.linalg.solve(np.array([[1, -1], [2, 1]], float), np.array([l1 - l2, l1 - l3]))
    u2 = (l2 - l4) / 2; return np.array([(l1 - l3 - u2) / 2, u2])
s1 = DW.first_wall(v, lam0, [1, 0, 0, 0]); ok &= abs(s1 - 0.2) < 1e-6
for s in (0.0, 0.1, 0.14, 0.26, 0.4, 0.6):
    lam = DW.shift(lam0, [1, 0, 0, 0], s); ok &= np.allclose(KB.valuation(v, lam, [1e-30])[0], u_pred(lam), atol=3e-3)
ok &= np.allclose(u_pred(DW.shift(lam0, [1, 0, 0, 0], 0.2 - 1e-9)), u_pred(DW.shift(lam0, [1, 0, 0, 0], 0.2 + 1e-9)), atol=1e-6)
ok &= DW.first_wall(v, lam0, [0, 0, 0, 1]) is None and DW.first_wall(v, lam0, [0, -1, 0, 1]) is None
h0, _, K0 = DW.balance_lp(v, lam0); _, u0 = KB.maxmin(v, lam0)
for t in rng.normal(size=(5, 2)):
    r = np.array(v, float) @ t
    for s in (0.3, 1.0, 2.5):
        lam = DW.shift(lam0, r, s); h, _, K = DW.balance_lp(v, lam); _, u = KB.maxmin(v, lam); ok &= K == K0 and abs(h - h0) < 1e-9 and np.allclose(u, u0 - s * t, atol=1e-7)
for s in (0.5, 2.0):
    h, _, K = DW.balance_lp(v, DW.shift(lam0, np.ones(4), s)); ok &= K == K0 and abs(h - (h0 + s)) < 1e-9
smin, rbest, c, a = DW.min_evidence(v, lam0); ok &= abs(smin - 0.1) < 1e-12 and rbest.tolist() == [1, 0, 1, 0]
R = rng.uniform(0, 1, (4000, 4)); R = (R - R.min(1, keepdims=True)) / np.maximum(1, (R.max(1) - R.min(1)))[:, None]; ar = R @ a
ok &= np.all(-c / ar[ar > 0] >= smin - 1e-12) and abs(DW.first_wall(v, lam0, rbest) - 0.1) < 1e-6
for r in rng.normal(size=(20, 3)):
    ok &= DW.first_wall(KB.CP2[0], KB.CP2[1], r, smax=3.0, n=61) is None
check("DW7 data paths: r = e1 reaches the wall at s* = 1/5 with u* affine and continuous; e4 and (0,-1,0,1) never; translations and the "
      "constant never; least evidence 1/10 at r = (1,0,1,0); CP^2 never", ok, f"s*(e1) = {s1:.6f}; least s = {smin:.3f}")
print(f"{sum(results)}/{len(results)} passed")
