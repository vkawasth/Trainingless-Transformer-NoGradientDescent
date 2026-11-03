"""Checks for the companion note "Conditioned states as toric Kahler data" (module pce/kahlerbridge.py).
  KB1  Tier A: the moment map is the gradient of the Kahler potential phi and the Fisher information is its Hessian (finite differences),
       positive definite on 30 random supports
  KB2  the symplectic potential: psi(p) = sum q log(q / c) at the Birch law with mean p; grad psi = u(p) and Hess psi = Fisher^-1
  KB3  Duistermaat-Heckman: the integral of det Fisher over a box in log coordinates equals the area of its moment image (triangle, square, random)
  KB4  the bridge: W_T has one positive critical point (four starts agree), its law has mean 0 (Birch at target 0), its valuation tends to
       (1/3, 1/3) for CP^2 and for F_1 with alpha < 1/3, and to ((1 + alpha)/4, (1 - alpha)/2) for alpha > 1/3; the leading terms there balance
  KB5  candidates versus selection: on F_1 with alpha = 3/5 every point (u1, 1/5) with 1/5 < u1 < 3/5 has balanced leading terms, but only
       u1 = 2/5 is a critical valuation; with alpha = 1/5, and on CP^2, the barycentre is the only candidate (exact rational points)
  KB6  the sign split (F_1, all four critical points from the quartic w^4 - T^(1/2) w - T^(1-alpha), 60-digit arithmetic): for alpha = 1/5 three
       sit over (1/3, 1/3), one of them positive, and one over (alpha, 1 - 2 alpha) with y1 < 0, whose leading terms balance only with signed
       weights; for alpha = 3/5 all four sit over ((1 + alpha)/4, (1 - alpha)/2), exactly one positive
  KB5 also checks the segment's endpoints (1/5, 1/5) and (3/5, 1/5): their leading sets are {v1, v2, v4} and {v2, v3, v4}, with no positive balance
  KB7  the positive thimble: at y_T the Hessian of Re W(e^b) in (Re b, Im b) is diag(H, -H), so the real directions ascend and the imaginary ones
       descend; the law e^(-W / hbar) db has mean b_T and covariance hbar H^-1 up to O(eps), eps = hbar / W(y_T) = 1e-2, 1e-3, 1e-4 (CP^2, F_1 a = 3/5)
  KB8  the tropical limit of the thimble law in u = log y / log T, hbar = T^h (log T = -200, -800): for h < h* its mean and covariance tend to those
       of the uniform law on Q_h = {min_k l_k >= h} (CP^2 at h = 0, 1/5; F_1 a = 1/5 at h = 1/5; F_1 a = 3/5 at h = 0, 1/10); at h = h* the mass
       concentrates on argmax min l: the barycentre for CP^2 and F_1 a = 1/5, the segment for a = 3/5, where the u1-marginal tends to the uniform law
       on [1/5, 3/5]; for a = 3/5 and h = 3/10 (between h* = 1/5 and the next max-min level 2/5) it is uniform on [3/10, 1/2], and for
       h = 9/20 (above 2/5) it concentrates at u1 = 2/5 (log T = -40, -100); argmax min l is
       {u : 0 in conv v_K(u)}: it contains the candidates and the endpoints, not the signed fibre (1/5, 3/5) of a = 1/5, whose min l = 1/5 < h*
"""
import os, sys, random, itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import kahlerbridge as KB
random.seed(20261102); rng = np.random.default_rng(20261102); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rsupp():
    while True:
        S = sorted({(random.randint(-2, 3), random.randint(-2, 3)) for _ in range(random.randint(3, 6))})
        if len(S) >= 3 and np.linalg.matrix_rank(np.array(S[1:], float) - np.array(S[0], float)) == 2: return S, [random.uniform(0.3, 3) for _ in S]

# KB1
ok = True; h = 1e-5
for _ in range(30):
    S, c = rsupp(); u = rng.normal(0, 0.7, 2); E = np.eye(2)
    gr = np.array([(KB.phi(S, c, u + h * e) - KB.phi(S, c, u - h * e)) / (2 * h) for e in E]); ok &= np.allclose(gr, KB.moment(S, c, u), atol=1e-7)
    H = np.array([[(KB.phi(S, c, u + h * a + h * b) - KB.phi(S, c, u + h * a - h * b) - KB.phi(S, c, u - h * a + h * b) + KB.phi(S, c, u - h * a - h * b)) / (4 * h * h)
                   for b in E] for a in E]); I = KB.fisher(S, c, u); ok &= np.allclose(H, I, atol=1e-4) and np.all(np.linalg.eigvalsh(I) > 0)
check("KB1 moment map = grad phi, Fisher information = Hess phi, positive definite (30 random supports)", ok)

# KB2
ok = True
for _ in range(20):
    S, c = rsupp(); lam = rng.dirichlet(3 * np.ones(len(S))); p = lam @ np.array(S, float)
    psi, up, q = KB.legendre(S, c, p); ok &= abs(psi - float(np.sum(q * np.log(q / np.array(c))))) < 1e-8 and np.allclose(q @ np.array(S, float), p, atol=1e-8)
    E = np.eye(2); hh = 1e-4
    gpsi = np.array([(KB.legendre(S, c, p + hh * e)[0] - KB.legendre(S, c, p - hh * e)[0]) / (2 * hh) for e in E]); ok &= np.allclose(gpsi, up, atol=1e-4)
    Hp = np.array([(KB.legendre(S, c, p + hh * e)[1] - KB.legendre(S, c, p - hh * e)[1]) / (2 * hh) for e in E]).T
    Hi = np.linalg.inv(KB.fisher(S, c, up)); ok &= np.max(np.abs(Hp - Hi)) < 1e-3 * max(1, np.max(np.abs(Hi)))
check("KB2 symplectic potential psi = sum q log(q/c) (minus entropy relative to c); grad psi = u(p); Hess psi = Fisher^-1 (20 supports)", ok)

# KB3
def image_area(S, c, box, n=3000):
    (a0, a1), (b0, b1) = box; t = np.linspace(0, 1, n, endpoint=False)
    B = np.vstack([np.c_[a0 + (a1 - a0) * t, np.full(n, b0)], np.c_[np.full(n, a1), b0 + (b1 - b0) * t],
                   np.c_[a1 - (a1 - a0) * t, np.full(n, b1)], np.c_[np.full(n, a0), b1 - (b1 - b0) * t]])
    P = np.array([KB.moment(S, c, u) for u in B]); x, y = P[:, 0], P[:, 1]; return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
def jac_int(S, c, box, n=160):
    (a0, a1), (b0, b1) = box; us = np.linspace(a0, a1, n + 1); vs = np.linspace(b0, b1, n + 1)
    mu = (us[:-1] + us[1:]) / 2; mv = (vs[:-1] + vs[1:]) / 2; hh = (us[1] - us[0]) * (vs[1] - vs[0])
    return sum(np.linalg.det(KB.fisher(S, c, (a, b))) for a in mu for b in mv) * hh
ok = True; rows = []
for name, (S, c) in (("CP2", ([(0, 0), (1, 0), (0, 1)], [1, 1, 1])), ("square", ([(0, 0), (1, 0), (0, 1), (1, 1)], [1, 1, 1, 1])), ("random", rsupp())):
    box = ((-1.2, 0.8), (-0.6, 1.1)); A1 = image_area(S, c, box); A2 = jac_int(S, c, box); rows.append((name, round(A1, 5), round(A2, 5))); ok &= abs(A1 - A2) < 2e-4 * max(1, A1)
check("KB3 Duistermaat-Heckman: integral of det Fisher over a box = area of its moment image", ok, f"{rows}")

# KB4
ok = True; rows = []
cases = [("CP2", KB.CP2, (Fr(1, 3), Fr(1, 3)))] + [(f"F1 a={a}", KB.F1(float(a)), (Fr(1, 3), Fr(1, 3)) if a < Fr(1, 3) else ((1 + a) / 4, (1 - a) / 2))
                                                  for a in (Fr(1, 5), Fr(1, 4), Fr(2, 5), Fr(3, 5), Fr(4, 5))]
for name, (v, lam), us in cases:
    T = 1e-12; sols = []
    for _ in range(4):
        b0 = rng.normal(0, 3, 2); V = np.array(v, float); lw = -np.array(lam, float) * np.log(T)
        from scipy.optimize import minimize
        f = lambda b: KB._lse(lw + V @ b)
        gg = lambda b: (lambda z: (np.exp(z - z.max()) / np.exp(z - z.max()).sum()) @ V)(lw + V @ b)
        r = minimize(f, b0, jac=gg, method="BFGS", options=dict(gtol=1e-14, maxiter=20000))
        z = lw + V @ r.x; w = np.exp(z - z.max()); sols.append(w / w.sum())          # the critical law, which is unique (Birch)
    ok &= max(np.max(np.abs(s - sols[0])) for s in sols) < 1e-6
    b, q = KB.positive_crit(v, lam, T); ok &= np.allclose(q @ np.array(v, float), 0, atol=1e-9)
    val = KB.valuation(v, lam, [1e-20, 1e-30])[-1]; ok &= np.allclose(val, [float(x) for x in us], atol=2e-3)
    K = KB.ties(v, lam, [float(x) for x in us]); ok &= KB.balanced(v, K); rows.append((name, tuple(round(float(x), 4) for x in val), K))
check("KB4 one positive critical point, with mean-zero law (Birch at 0); valuations (1/3,1/3) or ((1+a)/4,(1-a)/2); leading terms balance there", ok, f"{rows}")

# KB5
def ties_exact(v, lam, u):
    l = [Fr(vk[0]) * u[0] + Fr(vk[1]) * u[1] - Fr(lk) for vk, lk in zip(v, lam)]; m = min(l); return tuple(k for k in range(len(v)) if l[k] == m)
ok = True
a = Fr(3, 5); v, lam = [(1, 0), (0, 1), (-1, -1), (0, -1)], [0, 0, -1, -(1 - a)]
seg = [(Fr(k, 100), (1 - a) / 2) for k in range(21, 60)]; ok &= all(KB.balanced(v, ties_exact(v, lam, u)) for u in seg)
off = [(Fr(random.randint(1, 99), 100), Fr(random.randint(1, 99), 100)) for _ in range(300)]
inside = [u for u in off if all(Fr(vk[0]) * u[0] + Fr(vk[1]) * u[1] > Fr(lk) for vk, lk in zip(v, lam))]
ok &= all((not KB.balanced(v, ties_exact(v, lam, u))) or u[1] == (1 - a) / 2 for u in inside)
a2 = Fr(1, 5); lam2 = [0, 0, -1, -(1 - a2)]
cand2 = [u for u in [(Fr(i, 60), Fr(j, 60)) for i in range(1, 60) for j in range(1, 60)]
         if all(Fr(vk[0]) * u[0] + Fr(vk[1]) * u[1] > Fr(lk) for vk, lk in zip(v, lam2)) and KB.balanced(v, ties_exact(v, lam2, u))]
vC, lC = KB.CP2; candC = [u for u in [(Fr(i, 60), Fr(j, 60)) for i in range(1, 60) for j in range(1, 60) if i + j < 60] if KB.balanced(vC, ties_exact(vC, lC, u))]
ok &= cand2 == [(Fr(1, 3), Fr(1, 3))] and candC == [(Fr(1, 3), Fr(1, 3))]
ends = [((1 - a) / 2, (1 - a) / 2), (a, (1 - a) / 2)]; ok &= [ties_exact(v, lam, u) for u in ends] == [(0, 1, 3), (1, 2, 3)]
ok &= not any(KB.balanced(v, ties_exact(v, lam, u)) for u in ends)
check("KB5 candidates vs selection: for alpha = 3/5 a whole segment of points balances at leading order, one is critical; for alpha = 1/5 and CP^2 "
      "only the barycentre", ok, f"segment of {len(seg)} balanced points; alpha = 1/5 candidates {cand2}; CP^2 candidates {candC}")

# KB6
mp.mp.dps = 80; ok = True; rows = []
for a in (Fr(1, 5), Fr(3, 5)):
    T = mp.mpf(10) ** -60; roots = mp.polyroots([1, 0, 0, -mp.sqrt(T), -T ** (1 - mp.mpf(a.numerator) / a.denominator)], maxsteps=400, extraprec=400)
    crit = []
    for w in roots:
        y2 = w * w; y1 = mp.sqrt(T) / w; u = (float(mp.log(abs(y1)) / mp.log(T)), float(mp.log(abs(y2)) / mp.log(T)))
        pos = abs(mp.im(y1)) < mp.mpf(10) ** -50 and abs(mp.im(y2)) < mp.mpf(10) ** -50 and mp.re(y1) > 0 and mp.re(y2) > 0
        neg1 = abs(mp.im(y1)) < mp.mpf(10) ** -50 and mp.re(y1) < 0
        crit.append((u, pos, neg1))
    if a < Fr(1, 3):
        at_c = [c for c in crit if np.allclose(c[0], (1 / 3, 1 / 3), atol=2e-2)]; other = [c for c in crit if np.allclose(c[0], (float(a), float(1 - 2 * a)), atol=2e-2)]
        ok &= len(at_c) == 3 and sum(c[1] for c in at_c) == 1 and len(other) == 1 and other[0][2]
        K = ties_exact([(1, 0), (0, 1), (-1, -1), (0, -1)], [0, 0, -1, -(1 - a)], (a, 1 - 2 * a))
        ok &= not KB.balanced([(1, 0), (0, 1), (-1, -1), (0, -1)], K) and KB.signed_balanced([(1, 0), (0, 1), (-1, -1), (0, -1)], K)
    else:
        tgt = (float((1 + a) / 4), float((1 - a) / 2)); ok &= all(np.allclose(c[0], tgt, atol=2e-2) for c in crit) and sum(c[1] for c in crit) == 1
    rows.append((str(a), [(tuple(round(x, 3) for x in c[0]), "positive" if c[1] else ("y1<0" if c[2] else "complex")) for c in crit]))
check("KB6 the sign split: the fibre over (alpha, 1 - 2 alpha) comes from a critical point with y1 < 0, whose leading terms balance only with "
      "signed weights; exactly one critical point is positive", ok, f"{rows}")

# KB7
ok = True; rows = []
for name, (v, lam) in (("CP2", KB.CP2), ("F1 a=3/5", KB.F1(0.6))):
    T = 1e-6; b0, _ = KB.positive_crit(v, lam, T); V = np.array(v, float); cw = -np.array(lam, float) * np.log(T)
    W = lambda z: np.sum(np.exp(cw) * np.exp(V @ z)); Wb = float(np.real(W(b0.astype(complex))))
    hh = 1e-4; E = [np.r_[1.0, 0, 0, 0], np.r_[0, 1.0, 0, 0], np.r_[0, 0, 1.0, 0], np.r_[0, 0, 0, 1.0]]
    z = lambda x: (b0 + x[:2]) + 1j * x[2:]; f = lambda x: float(np.real(W(z(x))))
    Hc = np.array([[(f(hh * a + hh * b) - f(hh * a - hh * b) - f(-hh * a + hh * b) + f(-hh * a - hh * b)) / (4 * hh * hh) for b in E] for a in E])
    H = (np.exp(cw + V @ b0)[:, None, None] * np.einsum("ki,kj->kij", V, V)).sum(0)
    ok &= np.allclose(Hc, np.block([[H, np.zeros((2, 2))], [np.zeros((2, 2)), -H]]), atol=1e-5 * np.abs(H).max() + 1e-9)
    errs = []
    for eps in (1e-2, 1e-3, 1e-4):
        _, mean, C, Cp = KB.laplace_moments(v, lam, T, eps * Wb); errs.append(np.abs(C - Cp).max() / np.abs(Cp).max())
        ok &= np.abs(mean - b0).max() < 0.05 * np.sqrt(np.abs(Cp).max())
    ok &= all(e < 10 * x for e, x in zip(errs, (1e-2, 1e-3, 1e-4))) and errs[0] > errs[1] > errs[2]; rows.append((name, [f"{e:.1e}" for e in errs]))
check("KB7 the positive orthant is the ascending thimble at y_T (Hessian diag(H, -H)); Laplace: covariance hbar H^-1 + O(eps)", ok, f"{rows}")

# KB8
def gmom(U1, U2, w):
    m = np.array([(w * U1).sum(), (w * U2).sum()]); d1, d2 = U1 - m[0], U2 - m[1]
    return m, np.array([[(w * d1 * d1).sum(), (w * d1 * d2).sum()], [(w * d1 * d2).sum(), (w * d2 * d2).sum()]])
ok = True; rows = []
for name, (v, lam), h in (("CP2", KB.CP2, 0.0), ("CP2", KB.CP2, 0.2), ("F1 a=1/5", KB.F1(0.2), 0.2), ("F1 a=3/5", KB.F1(0.6), 0.0), ("F1 a=3/5", KB.F1(0.6), 0.1)):
    hs, us = KB.maxmin(v, lam); me, Ce = KB.polygon_moments(v, lam, h, us); er = []
    for lt in (-200.0, -800.0):
        m, C = gmom(*KB.tropical_grid(v, lam, lt, h)); er.append(max(np.abs(m - me).max(), np.abs(C - Ce).max()))
    ok &= er[1] < 5e-4 and er[1] < er[0]; rows.append((name, h, f"{er[0]:.1e} -> {er[1]:.1e}"))
hstars = {}
for name, (v, lam) in (("CP2", KB.CP2), ("F1 a=1/5", KB.F1(0.2)), ("F1 a=3/5", KB.F1(0.6))):
    hs, us = KB.maxmin(v, lam); hstars[name] = round(hs, 12); U1, U2, w = KB.tropical_grid(v, lam, -800.0, hs)
    L = U1[..., None] * np.array(v, float)[:, 0] + U2[..., None] * np.array(v, float)[:, 1] - np.array(lam, float); g = hs - L.min(-1)
    ok &= w[g < 0.02].sum() > 0.99; m, C = gmom(U1, U2, w)
    if name == "F1 a=3/5":
        p1 = w.sum(1); a1 = U1[:, 0]; cdf = np.cumsum(p1); unif = np.clip((a1 - 0.2) / 0.4, 0, 1)
        ks = np.abs(cdf - unif).max(); ok &= ks < 0.01 and abs(m[0] - 0.4) < 1e-3 and abs(C[0, 0] - 0.16 / 12) < 5e-4 and C[1, 1] < 1e-5
        rows.append((name, "h*", f"u1-marginal KS distance to uniform[1/5,3/5] {ks:.1e}, var {C[0, 0]:.5f} (1/75 = 0.01333)"))
    else:
        ok &= np.allclose(m, (1 / 3, 1 / 3), atol=1e-3) and np.abs(C).max() < 1e-5; rows.append((name, "h*", "point mass at the barycentre"))
ok &= hstars == {"CP2": round(1 / 3, 12), "F1 a=1/5": round(1 / 3, 12), "F1 a=3/5": 0.2}
v, lam = KB.F1(0.6); U1, U2, w = KB.tropical_grid(v, lam, -200.0, 0.3, box=((0.1, 0.7), (0.15, 0.25))); m, C = gmom(U1, U2, w)
ok &= abs(m[0] - 0.4) < 1e-3 and abs(C[0, 0] - 0.04 / 12) < 3e-4 and w[(U1 > 0.27) & (U1 < 0.53)].sum() > 0.99; rows.append(("F1 a=3/5", 0.3, f"u1 var {C[0, 0]:.5f} (1/300)"))
vs = []
for lt in (-40.0, -100.0):
    U1, U2, w = KB.tropical_grid(v, lam, lt, 0.45, box=((0.2, 0.6), (0.12, 0.28))); m, C = gmom(U1, U2, w); vs.append(C[0, 0])
    ok &= abs(m[0] - 0.4) < 1e-3 and w[np.abs(U1 - 0.4) < 0.03].sum() > 0.999
ok &= vs[1] < vs[0] < 1e-4; rows.append(("F1 a=3/5", 0.45, f"u1 var {vs[0]:.1e} -> {vs[1]:.1e}"))
seg = [(k / 100, 0.2) for k in range(20, 61)]; ok &= all(KB.argmax_set(v, lam, u) for u in seg)
ok &= not KB.argmax_set(v, lam, (0.4, 0.21)) and not KB.argmax_set(v, lam, (0.65, 0.2))
v2, lam2 = KB.F1(0.2); lsig = np.array(v2, float) @ np.array([0.2, 0.6]) - np.array(lam2, float)
ok &= abs(lsig.min() - 0.2) < 1e-12 and not KB.argmax_set(v2, lam2, (0.2, 0.6)) and KB.argmax_set(v2, lam2, (1 / 3, 1 / 3))
check("KB8 tropical limit of the thimble law: uniform on Q_h for h < h*; at h* concentrated on argmax min l (uniform on the F_1 segment); "
      "at the next level the segment shrinks to [h, 4/5 - h], then to u*", ok, f"h* = {hstars}; {rows}")
print(f"{sum(results)}/{len(results)} passed")
