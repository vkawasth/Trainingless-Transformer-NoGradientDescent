"""Stage IV of the companion note: a path through the column-binding example (section 12.8). New module; imports datawalls, couplingfibre and
floermorph, and modifies none.

The path keeps f = (2/5, 3/5) and g2 = 3/10 and moves g3 = gamma, g1 = 7/10 - gamma, for gamma in [3/20, 11/40]. The facets stay 11, 12, 13, 22, 23
(the pentagon; no blow-up wall for gamma in (1/10, 3/10)), and only the exponent of cell 23 moves: with A = T^(2/5), B = T^(3/10), C = T^(gamma - 2/5),
    W = y1 + y2 + A / (y1 y2) + B / y2 + C y1 y2 .
The critical points are exactly y1 = (s^2 - B) / s, y2 = s, with
    (s^2 - B)^2 (1 + C s) = A s ,
a quintic in s. The column-binding example of section 12.7 is gamma = 1/5.

  margins(gam)              (f, g) as Fractions
  leximin(gam)              the facet-leximin levels, level sets and point, exactly
  quintic()                 sympy: the reduction, the quintic in (A, B, C) and its discriminant
  regimes()                 the exact tropical picture on each side of the walls gamma = 1/5 (level 2) and 4/15 (level 1): fibres, multiplicities, leading forms
  newton(gam)               the Newton polygon of the quintic in s at rational gamma (exact); root valuations with multiplicities, and the fibres (val y1, val y2)
  roots(gam, L)             the five critical points at T = e^-L (mpmath): y1, y2, W, Q/W, term values
  objects_along(L, gams)    tracked objects with real/complex type and sign pattern
  disc_zeros(L, lo, hi, n)  the zeros of the discriminant along the path (sign changes, then bisection)
  stokes(L, gams, theta)    Stokes alignments Im(e^(-i theta)(W_j - W_i)) = 0 along the path, with the pair; conjugate pairs at the degenerate phases excluded
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
from pce import datawalls as DW, couplingfibre as CF, floermorph as FM

F = (Fr(2, 5), Fr(3, 5)); G2 = Fr(3, 10); LO, HI = Fr(3, 20), Fr(11, 40); W2, W1 = Fr(1, 5), Fr(4, 15)
V = [(1, 0), (0, 1), (-1, -1), (0, -1), (1, 1)]; NAMES = ["11", "12", "13", "22", "23"]


def margins(gam):
    gam = Fr(gam); return F, (Fr(7, 10) - gam, G2, gam)


def lam(gam):
    return [Fr(0), Fr(0), -F[0], -G2, Fr(2, 5) - Fr(gam)]


def leximin(gam):
    import sympy as sp
    L = lam(gam); u, lev = DW.lex_maxmin(V, [float(x) for x in L]); us = sp.symbols("u1:3"); ts = sp.symbols(f"t1:{len(lev) + 1}"); eqs = []
    for j, (h, S) in enumerate(lev):
        for k in S: eqs.append(V[k][0] * us[0] + V[k][1] * us[1] - sp.Rational(L[k].numerator, L[k].denominator) - ts[j])
    sol = sp.solve(eqs, list(us) + list(ts), dict=True)[0]
    return {"point": [Fr(str(sol[x])) for x in us], "levels": [Fr(str(sol[t])) for t in ts], "sets": [tuple(NAMES[k] for k in S) for _, S in lev]}


def quintic():
    import sympy as sp
    A, B, C, s, y1, y2 = sp.symbols("A B C s y1 y2")
    W = y1 + y2 + A / (y1 * y2) + B / y2 + C * y1 * y2
    diff = sp.simplify(y1 * sp.diff(W, y1) - y2 * sp.diff(W, y2))                     # = y1 - y2 + B / y2
    e1 = sp.together((y1 * sp.diff(W, y1)).subs(y1, (s ** 2 - B) / s).subs(y2, s))
    q = sp.expand((s ** 2 - B) ** 2 * (1 + C * s) - A * s)
    return dict(A=A, B=B, C=C, s=s, W=W, diff=diff, e1=e1, q=q, disc=sp.factor(sp.discriminant(q, s)))


def regimes():
    """exact leading data (derived in the note): list of (interval, [(fibre as functions of gamma, multiplicity, leading y2, leading y1, label)])"""
    return {
        "below 1/5": [("(1/4 - g/2, 3/20)", 4, "+-T^(3/20)", "+-T^(1/4 - g/2)", "leximin: P, N, and two real (+,-), (-,+)"),
                      ("(g - 1/10, 2/5 - g)", 1, "-T^(2/5 - g)", "+T^(g - 1/10)", "R (+,-)")],
        "at 1/5": [("(3/20, 3/20)", 2, "", "", "P, N"), ("(2/15, 1/6)", 3, "", "", "R, C_omega, C_omegabar")],
        "between": [("((4/5 - g)/4, (4/5 - g)/4)", 4, "i^k T^((4/5 - g)/4)", "same as y2", "leximin: P, N, I+, I-"),
                    ("(1/10, 1/5)", 1, "T^(1/5)", "-T^(1/10)", "R (-,+)")],
        "above 4/15": [("(2/15, 2/15)", 3, "omega^k T^(2/15)", "same as y2", "leximin: P, C_omega, C_omegabar"),
                       ("(2/5 - g, 2/5 - g)", 1, "-T^(2/5 - g)", "same as y2", "N (-,-)"), ("(1/10, 1/5)", 1, "T^(1/5)", "-T^(1/10)", "R (-,+)")]}


def newton(gam):
    """exact: the quintic C s^5 + s^4 - 2BC s^3 - 2B s^2 + (B^2 C - A) s + B^2 with valuations in T; gamma != 1/5"""
    g = Fr(gam); assert g != W2
    pts = [(5, g - Fr(2, 5)), (4, Fr(0)), (3, g - Fr(1, 10)), (2, Fr(3, 10)), (1, min(g + Fr(1, 5), Fr(2, 5))), (0, Fr(3, 5))]
    hull = [pts[-1]]; rest = sorted(pts[:-1])
    while hull[-1][0] < 5:
        x0, y0 = hull[-1]; cand = [p for p in pts if p[0] > x0]; m = min(Fr(p[1] - y0, 1) / (p[0] - x0) for p in cand)
        nxt = max((p for p in cand if Fr(p[1] - y0) / (p[0] - x0) == m), key=lambda p: p[0]); hull.append(nxt)
    roots = []
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        v = -(y1 - y0) / (x1 - x0); roots.append((v, x1 - x0))
    fib = []
    for v, mult in roots:
        if 2 * v != Fr(3, 10): vy1 = min(2 * v, Fr(3, 10)) - v
        else: vy1 = (Fr(2, 5) + v - min(Fr(0), g - Fr(2, 5) + v)) / 2 - v          # s^2 = B (1 + o(1)): delta^2 (1 + C s) = A s
        fib.append(((vy1, v), mult))
    return hull, roots, fib


def roots(gam, L, dps=None):
    mp.mp.dps = dps or int(L / 2.3) + 60
    A, B, C = (mp.exp(-L * mp.mpf(e)) for e in (mp.mpf(2) / 5, mp.mpf(3) / 10, mp.mpf(gam) - mp.mpf(2) / 5))
    out = []
    for s in mp.polyroots([C, 1, -2 * B * C, -2 * B, B * B * C - A, B * B], maxsteps=4000, extraprec=6 * mp.mp.dps):
        y1 = (s * s - B) / s; y2 = s; c = [y1, y2, A / (y1 * y2), B / y2, C * y1 * y2]; W = mp.fsum(c)
        Q = mp.matrix(2, 2)
        for ck, vk in zip(c, V):
            for i in range(2):
                for j in range(2): Q[i, j] += ck * vk[i] * vk[j]
        out.append(dict(y=(y1, y2), W=W, Qn=Q / W, terms=c))
    return out


def kind(o, tol=None):
    tol = tol or mp.mpf(10) ** (-(mp.mp.dps // 3)); y = o["y"]
    if all(abs(mp.im(v)) <= tol * abs(v) for v in y): return "".join("+" if mp.re(v) > 0 else "-" for v in y)
    return "cx"


def disc_zeros(L, lo=0.12, hi=0.275, n=3000):
    mp.mp.dps = int(L / 2.3) + 60; d = quintic(); import sympy as sp
    Df = sp.lambdify((d["A"], d["B"], d["C"]), d["disc"], "mpmath")
    def val(g): return Df(mp.exp(-L * mp.mpf(2) / 5), mp.exp(-L * mp.mpf(3) / 10), mp.exp(-L * (g - mp.mpf(2) / 5)))
    gs = [mp.mpf(lo) + k * (mp.mpf(hi) - mp.mpf(lo)) / n for k in range(n + 1)]; sg = [mp.sign(val(g)) for g in gs]; out = []
    for k in range(n):
        if sg[k] != sg[k + 1]:
            a, b = gs[k], gs[k + 1]
            for _ in range(100):
                m = (a + b) / 2
                if mp.sign(val(m)) == sg[k]: a = m
                else: b = m
            out.append(a)
    return out


def track(L, gams):
    def solver(g):
        r = roots(g, L); return [o["y"] for o in r], [o["W"] for o in r]
    return FM.track(gams, solver)


def stokes(L, gams, theta, P=None, Wt=None):
    if P is None: P, Wt = track(L, gams)
    ev = FM.stokes_events(Wt, gams, theta); out = []
    for g, i, j in ev:
        k = int(np.searchsorted(gams, g)); k = min(max(k, 1), len(gams) - 1)
        conj = all(abs(P[k, i, t] - np.conj(P[k, j, t])) < 1e-6 * abs(P[k, i, t]) and abs(P[k, i, t].imag) > 1e-6 * abs(P[k, i, t]) for t in range(2))
        if conj and (abs(np.cos(theta)) < 1e-9 or abs(np.sin(theta)) < 1e-9): continue        # W_j = conj W_i: Im(e^-i theta (W_j - W_i)) = 2 cos(theta) Im W_j, or identically 0
        out.append((g, i, j, conj))
    return out
