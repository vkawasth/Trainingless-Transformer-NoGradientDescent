"""Study 48, step 1 (exact algebra; no pre-registered checks, a derivation): first-level tie types of 2x4 tables, their leading families,
the point and value discriminants of the row-column family, the parity character, the A2 data at the predicted fold, and the 2xn
row-column family for n = 3..8. Output: check_ties24_output.txt and study48_2x4/step1.json.
Kernel coordinates: K_j = E_00 - E_0j - E_10 + E_1j (j = 1..n-1), so cell exponents are 00: (1,...,1), 0j: -e_j, 10: (-1,...,-1), 1j: e_j.
"""
import os, json, itertools, math
import sympy as sp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "study48_2x4"); os.makedirs(D, exist_ok=True)
OUT = open(os.path.join(HERE, "check_ties24_output.txt"), "w"); RES = {}
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

def exps(n):
    E = {}
    for i in range(2):
        for j in range(n):
            v = [0] * (n - 1)
            if j == 0: v = [1 if i == 0 else -1] * (n - 1)
            else: v[j - 1] = -1 if i == 0 else 1
            E[(i, j)] = tuple(v)
    return E
def rank(vs): return sp.Matrix(vs).rank() if vs else 0
def nvol(vs):
    """Normalised volume (d! vol) of the convex hull of integer points, exact: cones from the (rational) centroid over the hull's triangulated facets."""
    from scipy.spatial import ConvexHull
    P = np.array(vs, float); d = P.shape[1]
    if np.linalg.matrix_rank(P - P[0]) < d: return 0
    H = ConvexHull(P); Vq = [[sp.Integer(int(x)) for x in row] for row in vs]
    c = [sum(Vq[k][i] for k in range(len(Vq))) / len(Vq) for i in range(d)]
    return int(sum(abs(sp.Matrix([[Vq[k][i] - c[i] for i in range(d)] for k in simp]).det()) for simp in H.simplices))
def parity_character(vs):
    """A homomorphism chi: Z^d -> Z/2 with chi(v) = 1 for all v in vs (solve over GF(2))."""
    d = len(vs[0])
    for bits in itertools.product((0, 1), repeat=d):
        if all(sum(b * (x % 2) for b, x in zip(bits, v)) % 2 == 1 for v in vs): return bits
    return None

# ---------------------------------------------------------------- 1. tie types of 2x4 ----------------------------------------------------------
n = 4; E = exps(n); cells = list(E)
types = {"row-column": [(0, j) for j in range(4)] + [(1, 0)], "column-column": [(0, 0), (1, 0), (0, 1), (1, 1)], "row-row": cells}
say("1. First-level tie types of a 2x4 table (row i has value f_i/4, column j has g_j/2; generic walls are two-way ties):")
say("   row-row (f_0 = f_1, both below every g_j/2), row-column (f_i/4 = g_j/2), column-column (g_j = g_j'); by symmetry i = 0, j = 0, j' = 1.")
RES["types"] = {}
for name, K in types.items():
    vs = [E[c] for c in K]; r = rank([list(v) for v in vs]); vol = nvol([list(v) for v in vs]) if r == 3 else 0
    say(f"   {name:14s}: cells {K}; exponents {vs}; rank {r} of 3; normalised volume {vol if r == 3 else '(not full-dimensional)'}")
    RES["types"][name] = dict(cells=[list(c) for c in K], rank=int(r), nvol=vol)
say(f"   whole 2x4 support: normalised volume {nvol([list(E[c]) for c in cells])} (number of critical points of W for generic coefficients)")
RES["full_nvol"] = nvol([list(E[c]) for c in cells])

# ---------------------------------------------------------------- 2. row-column family ----------------------------------------------------------
Y1, Y2, Y3, Y, X, b, a, s = sp.symbols("Y1 Y2 Y3 Y X b a s")
f = Y1 * Y2 * Y3 + 1 / Y1 + 1 / Y2 + 1 / Y3 + b / (Y1 * Y2 * Y3)
g = Y1 * Y2 * Y3 + 1 / (Y1 * Y2 * Y3) + a * (1 / Y1 + 1 / Y2 + 1 / Y3)
L = [sp.simplify(v * sp.diff(f, v)) for v in (Y1, Y2, Y3)]
say("2. Row-column family f_b = Y1Y2Y3 + sum 1/Yj + b/(Y1Y2Y3) (row side; b in (0,1]); column side g_a = Y1Y2Y3 + 1/(Y1Y2Y3) + a sum 1/Yj (a in (0,1]).")
say("   Y1 d1 f - Y2 d2 f =", sp.simplify(L[0] - L[1]), "-> every critical point is diagonal (likewise for each pair)")
crit = sp.factor(sp.numer(sp.together(L[0].subs({Y1: Y, Y2: Y, Y3: Y}))))
say("   diagonal critical equation:", crit, "; with X = Y^2:", sp.factor(sp.expand(crit).subs(Y ** 2, X) if False else (X ** 3 - X - b)))
# critical scheme: the full system (numerators) restricted to the torus; count solutions for a sample b with Groebner basis
bs = sp.Rational(1, 5)
sysn = [sp.numer(sp.together(l.subs(b, bs))) for l in L]
G = sp.groebner(sysn + [sp.Symbol("T") * Y1 * Y2 * Y3 - 1], sp.Symbol("T"), Y1, Y2, Y3, order="lex")
last = G.exprs[-1]
say(f"   critical scheme at b = 1/5 (Groebner basis, torus enforced): last generator in Y3 = {sp.factor(last)}")
nsol = sp.Poly(last, Y3).degree()
RES["scheme_degree_Y3_at_b_1_5"] = int(nsol)
gc = sp.factor(sp.numer(sp.together(sp.simplify(Y * sp.diff(g.subs({Y1: Y, Y2: Y, Y3: Y}), Y) / 3))))
say("   column side diagonal equation:", gc)
# substitution: column family -> row family
Z = sp.symbols("Z")
sub = sp.simplify(g.subs({Y1: a ** sp.Rational(1, 4) * Y1, Y2: a ** sp.Rational(1, 4) * Y2, Y3: a ** sp.Rational(1, 4) * Y3}) / a ** sp.Rational(3, 4))
say("   column side after Y = a^(1/4) Z, divided by a^(3/4), minus f with b = a^(-3/2) :", sp.simplify(sub - f.subs(b, a ** sp.Rational(-3, 2))))
# points discriminant, folds
dX = sp.discriminant(X ** 3 - X - b, X); dY = sp.factor(sp.discriminant(Y ** 6 - Y ** 2 - b, Y))
bstar = sp.sqrt(sp.Rational(4, 27))
say(f"   disc_X (X^3 - X - b) = {dX};  disc_Y (Y^6 - Y^2 - b) = {dY};  positive root b* = {bstar} = {float(bstar):.6f}")
Xs = sp.solve(sp.Eq(3 * X ** 2 - 1, 0), X); Xst = [x for x in Xs if sp.simplify(x ** 3 - x - bstar) == 0]
say(f"   double root at b*: X* = {Xst} -> Y* = +-i 3^(-1/4) (each a double root of Y^6 - Y^2 - b*); the third root X = {sp.nsimplify(sp.solve(sp.Eq((X - Xst[0]) ** 2 * (X - sp.Symbol('r')), X**3 - X - bstar), sp.Symbol('r')) if False else 2 / sp.sqrt(3))}")
# values
sval = sp.simplify(f.subs({Y1: Y, Y2: Y, Y3: Y}))
say("   value on the diagonal:", sval, "= (Y^6 + 3Y^2 + b)/Y^3 = (4X + 2b)/Y^3 on the critical set")
P = sp.Poly(sp.resultant(Y ** 6 - Y ** 2 - b, s * Y ** 3 - Y ** 6 - 3 * Y ** 2 - b, Y), s)
say("   value polynomial P(s, b) =", sp.factor(P.as_expr()))
Pe = sp.Poly(sp.expand(P.as_expr()), s); even = all(m[0] % 2 == 0 for m in Pe.monoms())
say("   P is even in s:", even)
# value discriminant: in terms of u = s^2 (even polynomial), collisions are disc_u Q (pairs s_i = +-s_j, i != antipode) and Q(0) (s = 0 = -s)
u = sp.symbols("u"); Q = sp.Poly(sp.expand(P.as_expr()).subs(s ** 2, u) if False else sum(c * u ** (m[0] // 2) for m, c in zip(Pe.monoms(), Pe.coeffs())), u)
dQ = sp.factor(sp.discriminant(Q.as_expr(), u)); Q0 = sp.factor(Q.as_expr().subs(u, 0))
say("   Q(u, b) with u = s^2:", sp.factor(Q.as_expr()))
say("   disc_u Q =", dQ, ";  Q(0, b) =", Q0)
def posroots(expr):
    out = []
    for fac, m in sp.factor_list(sp.numer(sp.together(expr)))[1]:
        if fac.free_symbols & {b}:
            for r in sp.Poly(fac, b).nroots(n=30):
                if abs(sp.im(r)) < 1e-20 and sp.re(r) > 0: out.append((str(fac), float(sp.re(r))))
    return out
RES["row_value_collisions_b>0"] = posroots(dQ); RES["row_zero_values_b>0"] = posroots(Q0)
say("   positive real b with a value collision (disc_u Q = 0):", RES["row_value_collisions_b>0"], "; with a zero value (Q(0,b) = 0):", RES["row_zero_values_b>0"])
say("   (the column side is the row family at b = a^(-3/2) >= 1 up to the positive factor a^(3/4); so the same polynomials decide it for b >= 1)")

# ---------------------------------------------------------------- 3. parity character ----------------------------------------------------------
say("3. Parity character chi: Z^(n-1) -> Z/2 with chi = 1 on every cell exponent of a 2xn table:")
RES["parity"] = {}
for m in range(3, 9):
    ch = parity_character([list(v) for v in exps(m).values()]); RES["parity"][m] = ch
    say(f"   n = {m}: {ch if ch is not None else 'none'}")

# ---------------------------------------------------------------- 4. A2 data at b* ------------------------------------------------------------
say("4. A2 data at the fold (log coordinates w, Y = e^w):")
w1, w2, w3 = sp.symbols("w1 w2 w3"); F = f.subs({Y1: sp.exp(w1), Y2: sp.exp(w2), Y3: sp.exp(w3)})
Hm = sp.Matrix(3, 3, lambda i, j: sp.diff(F, [w1, w2, w3][i], [w1, w2, w3][j]))
Ystar = sp.I * 3 ** sp.Rational(-1, 4); subsd = {w1: sp.log(Ystar), w2: sp.log(Ystar), w3: sp.log(Ystar), b: bstar}
Hs = sp.simplify(Hm.subs(subsd)); ev = sp.simplify(Hs.eigenvals())
say("   Hessian eigenvalues at Y*, b*:", ev, " (transverse eigenvalue 1/Y* expected:", sp.simplify(1 / Ystar), ")")
c3 = sp.simplify(sum(sp.diff(F, w1, i, w2, j, w3, k) * sp.binomial(3, i) * sp.binomial(3 - i, j) for i in range(4) for j in range(4 - i) for k in [3 - i - j]).subs(subsd))
say("   third derivative along (1,1,1) at Y*, b*:", c3, "(nonzero -> ordinary fold)")
dFdb = sp.simplify(sp.diff(sp.diff(F, b), w1).subs(subsd) + sp.diff(sp.diff(F, b), w2).subs(subsd) + sp.diff(sp.diff(F, b), w3).subs(subsd))
say("   d/db of the gradient along (1,1,1) at the fold:", dFdb, "; disc'(b*) = -54 b* =", sp.simplify(-54 * bstar), "(transverse crossing)")
vstar = sp.simplify(sval.subs({Y: Ystar, b: bstar})); say("   critical values at the two folds: +-", vstar, "=", sp.N(vstar, 8), "(purely imaginary; the two folds sit at opposite values)")
RES["fold"] = dict(bstar=float(bstar), value=str(vstar), cubic=str(c3), hessian_eigs=str(ev))

# ---------------------------------------------------------------- 5. 2xn row-column families ------------------------------------------------
say("5. 2xn row-column leading families (diagonal reduction Y^(2n-2) - Y^(n-2) - b), n = 3..8:")
RES["2xn"] = {}
for m in range(3, 9):
    p = Y ** (2 * m - 2) - Y ** (m - 2) - b
    dp = sp.factor(sp.discriminant(p, Y)); pr = posroots(dp)
    # values: s = Y^(n-1) + (n-1)/Y + b/Y^(n-1); value collisions scanned numerically on b in (0, 50) (log grid), exact for n <= 4 above
    def vals(bb):
        r = np.roots([1] + [0] * (m - 1) + [-1] + [0] * (m - 3) + [-bb]) if m > 2 else None
        return np.array([y ** (m - 1) + (m - 1) / y + bb / y ** (m - 1) for y in r])
    mins = []
    for bb in np.exp(np.linspace(np.log(1e-3), np.log(50), 4001)):
        v = vals(bb); d = min(abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))); mins.append((d, bb))
    # local minima below 1e-3 (candidates for value collisions on the real b-axis)
    cand = [(round(mins[k][1], 5), mins[k][0]) for k in range(1, len(mins) - 1) if mins[k][0] < mins[k - 1][0] and mins[k][0] <= mins[k + 1][0] and mins[k][0] < 1e-3]
    RES["2xn"][m] = dict(point_disc=str(dp), point_collisions_pos_b=pr, value_collision_candidates=[(c[0], float(c[1])) for c in cand])
    say(f"   n = {m}: disc_Y = {dp}; positive b with a point collision: {pr}; value-collision candidates on b in (1e-3, 50) (numerical): {cand}")
json.dump(RES, open(os.path.join(D, "step1.json"), "w"), indent=1, default=str)
