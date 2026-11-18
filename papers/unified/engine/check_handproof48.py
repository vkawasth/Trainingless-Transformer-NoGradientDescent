"""Verification of the identities and constants used in the hand proof of the leading-braid theorem for the Study 45/46 wall (thm:leadbraid,
hand proof in the paper). The proof itself is written so that every step can be checked by hand; this script only re-derives the
identities symbolically and evaluates the explicit constants (with Arb balls, so the printed bounds are rigorous). No pre-registered checks:
a proof, not a study. Output: check_handproof48_output.txt.
"""
import os, math, itertools
import sympy as sp
from flint import arb, acb, ctx
ctx.prec = 128
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = open(os.path.join(HERE, "check_handproof48_output.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")
Y, Z, Y1, Y2, a, b, s = sp.symbols("Y Z Y1 Y2 a b s")

say("1. Column family is the row family after Y = a^(1/3) Z:")
G = Y1 * Y2 + 1 / (Y1 * Y2) + a * (1 / Y1 + 1 / Y2)
Gs = sp.simplify(G.subs({Y1: a ** sp.Rational(1, 3) * Y1, Y2: a ** sp.Rational(1, 3) * Y2}) / a ** sp.Rational(2, 3))
f = Y1 * Y2 + 1 / Y1 + 1 / Y2 + b / (Y1 * Y2)
say("   a^(-2/3) G(a^(1/3) Z) - f_b(Z) with b = a^(-4/3):", sp.simplify(Gs - f.subs(b, a ** sp.Rational(-4, 3))))
say("2. Critical points of f_b: Y1 Y2-derivatives")
d1, d2 = sp.expand(Y1 * sp.diff(f, Y1)), sp.expand(Y2 * sp.diff(f, Y2))
say("   Y1 d1 f - Y2 d2 f =", sp.simplify(d1 - d2), " (so Y1 = Y2 = Y); then", sp.factor(sp.simplify(d1.subs({Y1: Y, Y2: Y}) * Y ** 2)))
say("   value at Y1 = Y2 = Y using Y^4 = Y + b:", sp.simplify(sp.rem(sp.numer(sp.together(f.subs({Y1: Y, Y2: Y}) * Y ** 2)), Y ** 4 - Y - b, Y) / Y ** 2))
P = sp.Poly(sp.resultant(Y ** 4 - Y - b, s * Y - 2 * Y ** 3 - 1, Y), s)
say("   critical-value polynomial P(s, b) =", P.as_expr())
say("   disc_s P =", sp.factor(sp.discriminant(P.as_expr(), s)), "; disc_Y (Y^4 - Y - b) =", sp.factor(sp.discriminant(Y ** 4 - Y - b, Y)))
say("   Res_Y(Y^4 - Y - b, 4Y^3 - 1) =", sp.factor(sp.resultant(Y ** 4 - Y - b, 4 * Y ** 3 - 1, Y)), "; 4Y^3 - 1 = (3Y + 4b)/Y on Y^4 = Y + b:",
    sp.simplify(sp.rem(sp.expand((4 * Y ** 3 - 1) * Y - (3 * Y + 4 * b)), Y ** 4 - Y - b, Y)) == 0)
say("3. Hessian in log coordinates at Y1 = Y2 = Y: entries")
H11 = Y ** 2 + 1 / Y + b / Y ** 2; H12 = Y ** 2 + b / Y ** 2
say("   H11 - 2Y^2 and H12 - (2Y^2 - 1/Y) on Y^4 = Y + b:", [sp.simplify((sp.numer(sp.together((h - t) * Y ** 2)).subs(b, Y ** 4 - Y))) for h, t in ((H11, 2 * Y ** 2), (H12, 2 * Y ** 2 - 1 / Y))])

# ---------------------------------------------------------------- constants -----------------------------------------------------------------
b0, b1 = arb(-4) / 3, arb(4) / 3; b0, b1 = b0.exp(), b1.exp()
say(f"4. Window: b in [e^-4/3, e^4/3] = [{b0.str(6)}, {b1.str(6)}]; eps = e^(4/3 - 0.15 L) bounds both perturbation coefficients (row and column side).")
rlo, rhi = arb(1) / 4, arb(155) / 100
say(f"   |Y| > 1/4: |Y^4 - Y| <= (1/4)^4 + 1/4 = {(rlo ** 4 + rlo).str(6)} < b0;   |Y| < 1.55: |Y^4 - Y| >= 1.55^4 - 1.55 = {(rhi ** 4 - rhi).str(6)} > b1")
# |3Y + 4b| >= g(b) = (256 b^3 + 27) b / (3 * 1.55 + 4 b)^3 (product of the four factors is (256 b^3 + 27) b; the other three are <= 3*1.55 + 4b)
N = 8; gs = []
for k in range(N):
    lo = b0 + (b1 - b0) * k / N; hi = b0 + (b1 - b0) * (k + 1) / N
    g = (256 * lo ** 3 + 27) * lo / (3 * rhi + 4 * hi) ** 3; gs.append(g)
    say(f"   b in [{lo.str(4)}, {hi.str(4)}]:  |3Y + 4b| >= (256 b^3 + 27) b / (4.65 + 4b)^3 >= {g.str(6)}")
gmin = min(gs, key=lambda x: float(x.mid()))
M = rhi ** 2 / gmin + rhi
say(f"   ||H^-1||_inf <= |Y|^2/|3Y + 4b| + |Y| <= M = {M.str(6)}")
r = arb(1) / 100
K = 4 * (rhi ** 2 + 2 / rlo + b1 / rlo ** 2 + 1) * (2 * r).exp()
say(f"   Lipschitz constant of the Hessian on the ball of radius r = 0.01 (sup norm): K <= 4 (|Y|^2 + 2/|Y| + b/|Y|^2 + 2 eps|Y|) e^(2r) <= {K.str(6)}")
# separation of the model values: prod_{i<j} |s_i - s_j|^2 = (256 b^3 + 27)^3 / b^6 >= its minimum at b^3 = 27/128
bst = (arb(27) / 128) ** (arb(1) / 3); pmin = (256 * bst ** 3 + 27) ** 3 / bst ** 6
smax = 2 * rhi ** 2 + 1 / rlo
sep = (pmin / (2 * smax) ** 10).sqrt()
say(f"   prod |s_i - s_j|^2 >= {pmin.str(6)} (minimum at b = (27/128)^(1/3) = {bst.str(5)}); |s| <= 2|Y|^2 + 1/|Y| <= {smax.str(5)}; so every |s_i - s_j| >= {sep.str(5)}")
# Kantorovich: M' = M / (1 - 1.55 M eps), eta <= M' 1.55 eps, h = M' K eta <= 1/2;  value change delta <= 2 * 1.55 eps e^r + (1/2) Hmax (2 eta)^2
def ok(eps):
    Mp = M / (1 - rhi * M * eps); eta = Mp * rhi * eps; h = Mp * K * eta
    Hmax = 4 * (rhi ** 2 + 2 / rlo + b1 / rlo ** 2 + 1) * (2 * r).exp()
    delta = 2 * rhi * eps * r.exp() + Hmax * (2 * eta) ** 2 / 2
    return bool(h < arb(1) / 2) and bool(2 * eta < r) and bool(2 * delta < sep), h, eta, delta
lo, hi = 1e-12, 1e-2
for _ in range(80):
    mid = math.sqrt(lo * hi)
    if ok(arb(mid))[0]: lo = mid
    else: hi = mid
epsmax = lo; okk, h, eta, delta = ok(arb(epsmax))
Lhand = (4 / 3 - math.log(epsmax)) / 0.15
say(f"5. Kantorovich holds and the value change stays below half the separation for eps <= {epsmax:.3e} (h = {h.str(4)}, step bound {(2 * eta).str(4)}, value change {delta.str(4)}),")
say(f"   i.e. for e^(4/3 - 0.15 L) <= {epsmax:.3e}, i.e. L >= {Lhand:.1f}.")
# the braid of the contractible stratum, read on the e^{-0.7i} projection, from the endpoint configurations (straight line in C)
import numpy as np
def vals_row(bb): Ys = np.roots([1, 0, 0, -1, -bb]); return [2 * y * y + 1 / y for y in Ys]
def vals_col(aa): Ys = np.roots([1, 0, 0, -aa, -1]); return [2 * y * y + aa / y for y in Ys]
A = vals_row(math.exp(-4 / 3)); B = vals_col(math.exp(-1))
def canon(v):
    re = sorted([x for x in v if abs(x.imag) < 1e-12], key=lambda x: x.real); cx = [x for x in v if x.imag > 1e-12][0]
    return [re[0].real, re[1].real, cx]
ca, cb = canon(A), canon(B)
def pts(c): return [complex(c[0]), complex(c[1]), c[2], c[2].conjugate()]
PA, PB = pts(ca), pts(cb); rot = np.exp(-0.7j)
ev = []
for i, j in itertools.combinations(range(4), 2):
    da, db = (rot * (PA[i] - PA[j])).real, (rot * (PB[i] - PB[j])).real
    if da * db < 0:
        lam = da / (da - db); zi = PA[i] + lam * (PB[i] - PA[i]); zj = PA[j] + lam * (PB[j] - PA[j]); ev.append((lam, i, j, (rot * (zi - zj)).imag))
order = sorted(range(4), key=lambda k: (rot * PA[k]).real); word = []
for lam, i, j, im in sorted(ev):
    k = min(order.index(i), order.index(j)); p_, q_ = order[k], order[k + 1]; above = (rot * ((PA[p_] + lam * (PB[p_] - PA[p_])) - (PA[q_] + lam * (PB[q_] - PA[q_])))).imag > 0
    word.append(f"s{k + 1}{'' if above else '^-1'}"); order[k], order[k + 1] = q_, p_
say(f"6. Canonical braid of the stratum (x < y real, z, conj z) between the window ends, straight line, e^(-0.7i) reading: {' '.join(word)}")
say(f"   ends: tau = -2: x, y = {ca[0]:.4f}, {ca[1]:.4f}, z = {ca[2]:.4f};  tau = 2: x, y = {cb[0]:.4f}, {cb[1]:.4f}, z = {cb[2]:.4f}")
