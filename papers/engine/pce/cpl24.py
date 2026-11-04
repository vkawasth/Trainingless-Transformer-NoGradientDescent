"""Part IV of the companion note: 2 x 4 coupling polytopes, three-dimensional fibres. New module; imports couplingfibre and datawalls and
modifies neither.

Coordinates u = (a11, a12, a13). The eight cells are the facet functions l_c(u) = <v_c, u> - lam_c (the coupling entries):
  a1j = u_j (j < 4),  a14 = f1 - sum u,  a2j = g_j - u_j (j < 4),  a24 = sum u - (f1 + g1 + g2 + g3 - 1),
so the mirror potential over the facet cells is
  W = sum_j (y_j + T^(g_j) / y_j) + T^(f1) / P + T^(g4 - f1) P,   P = y1 y2 y3   (terms of non-facet cells omitted).

Everything combinatorial is exact (Fractions):
  config(f, g)               normals V (8 x 3, integers), lam (Fractions), cell names (i, j)
  polytope(f, g)             exact vertices with their tight cells, the facet cells, smoothness (|det| = 1 at every vertex)
  edges(P)                   the edges (torus-invariant curves): pairs of vertices sharing two tight facets
  wall_relation(...)         v_k + v_l + a v_i + b v_j = 0 for the edge with tight facets {i, j} and end-facets k, l (smooth fan: a, b integers)
  intersections(f, g)        for every invariant curve C: -K.C = 2 + a + b and the Kahler degree [omega].C = -lam_k - lam_l - a lam_i - b lam_j
                             (which must equal the lattice length of the edge)
  fano_class(f, g)           'Fano' (all -K.C > 0), 'weak Fano' (all >= 0, some = 0), 'not nef'; by the toric Kleiman criterion
  identify(f, g)             the toric threefold for the Fano chambers, from the primitive relations
  chambers()                 every Kahler chamber of 2 x 4 margins: sign vectors of f1 - g(J), J a nonempty proper subset of the columns, realised
                             by linear programming; an interior rational point of each
  leximin_exact(f, g)        the leximin point and levels, exactly; the drop pattern (rank increments of the fixed normals)
  is_box(f, g)               the box criterion: some column with g_j > max(f1, f2)
  vertex_test(f, g)          the independent anticanonical vertex test (ample / nef) used as a cross-check of the curve test
  EX_A, EX_B, EX_C           the three examples of Part IV
  reduction(f, g)            the exact reduction of the critical equations to one resultant in c (sympy)
  critical_points_exact(...) all critical points from that reduction
Numerical (mpmath):
  potential(f, g)            the facet terms (coefficient exponent, normal) of W
  critical_points(f, g, L)   all critical points at T = e^-L by Newton from many starts, certified complete when their number reaches the
                             normalised volume of conv{v_c : c a facet} (Bernstein-Kushnirenko bound)
  positive_point(f, g, L)    the positive critical point (log coordinates), multiprecision
  clifford_spectrum(...)     eigenvalues of Q/W, Q = sum_c coef_c y^(v_c) v_c v_c^T (the Hessian of W in log coordinates)
"""
from __future__ import annotations
import itertools, math
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

NAMES = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
EX_A = ((Fr(2, 5), Fr(3, 5)), (Fr(1, 20), Fr(13, 20), Fr(1, 10), Fr(1, 5)))      # the cube: CP^1 x CP^1 x CP^1, levels 1/40, 1/20, 1/10
EX_B = ((Fr(3, 20), Fr(17, 20)), (Fr(1, 20), Fr(1, 5), Fr(1, 4), Fr(1, 2)))      # Bl_pt CP^3, levels 1/40, 1/24 (drops 1, 2)
EX_C = ((Fr(3, 10), Fr(7, 10)), (Fr(1, 10), Fr(3, 20), Fr(1, 4), Fr(1, 2)))      # weak Fano, 7 facets, levels 1/20, 3/40, 7/80 (+ 13/80)


def config(f, g):
    f1 = Fr(f[0]); g = [Fr(x) for x in g]
    V = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1), (-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 1, 1)]
    lam = [Fr(0), Fr(0), Fr(0), -f1, -g[0], -g[1], -g[2], f1 + g[0] + g[1] + g[2] - 1]
    return V, lam, NAMES


def _dot(v, u): return sum(Fr(a) * b for a, b in zip(v, u))


def _solve3(A, b):
    """exact 3 x 3 solve (Cramer); None if singular"""
    def det(M): return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det(A)
    if D == 0: return None, 0
    x = []
    for i in range(3):
        M = [list(r) for r in A]
        for r in range(3): M[r][i] = b[r]
        x.append(Fr(det(M)) / D)
    return x, D


def polytope(f, g):
    V, lam, _ = config(f, g); verts = []
    for S in itertools.combinations(range(8), 3):
        x, D = _solve3([V[k] for k in S], [lam[k] for k in S])
        if x is None: continue
        vals = [_dot(V[k], x) - lam[k] for k in range(8)]
        if min(vals) < 0: continue
        if not any(x == y for y, _ in verts): verts.append((x, tuple(k for k in range(8) if vals[k] == 0)))
    facets = []
    for k in range(8):                                      # a facet: its zero set contains 3 affinely independent vertices
        pts = [np.array([float(c) for c in x]) for x, t in verts if k in t]
        if len(pts) >= 3 and np.linalg.matrix_rank(np.array(pts[1:]) - pts[0]) == 2: facets.append(k)
    simple = all(len([k for k in t if k in facets]) == 3 for _, t in verts)
    smooth = simple and all(abs(_solve3([V[k] for k in t if k in facets], [0, 0, 0])[1]) == 1 for _, t in verts)
    return {"V": V, "lam": lam, "verts": [(x, tuple(k for k in t if k in facets)) for x, t in verts], "facets": facets, "simple": simple,
            "smooth": smooth}


def edges(P):
    out = []
    for (x, s), (y, t) in itertools.combinations(P["verts"], 2):
        common = sorted(set(s) & set(t))
        if len(common) == 2:
            k = [a for a in s if a not in common][0]; l = [a for a in t if a not in common][0]; out.append((common, k, l, x, y))
    return out


def wall_relation(V, i, j, k, l):
    """v_k + v_l + a v_i + b v_j = 0: least squares on integers, verified exactly"""
    A = np.array([V[i], V[j]], float).T; rhs = -np.array(V[k], float) - np.array(V[l], float)
    a, b = np.linalg.lstsq(A, rhs, rcond=None)[0]; a, b = Fr(round(a)), Fr(round(b))
    ok = all(V[k][r] + V[l][r] + a * V[i][r] + b * V[j][r] == 0 for r in range(3))
    return (a, b) if ok else (None, None)


def lattice_length(x, y):
    d = [b - a for a, b in zip(x, y)]; nz = [c for c in d if c != 0]
    den = 1
    for c in nz: den = den * c.denominator // math.gcd(den, c.denominator)
    ints = [int(c * den) for c in nz]; gg = 0
    for c in ints: gg = math.gcd(gg, abs(c))
    return Fr(gg, den)


def intersections(f, g):
    P = polytope(f, g); V, lam = P["V"], P["lam"]; out = []
    for (i, j), k, l, x, y in edges(P):
        a, b = wall_relation(V, i, j, k, l)
        out.append({"curve": (i, j), "ends": (k, l), "a": a, "b": b, "minusK": 2 + a + b,
                    "omega": -lam[k] - lam[l] - a * lam[i] - b * lam[j], "length": lattice_length(x, y)})
    return P, out


def fano_class(f, g):
    P, I = intersections(f, g); ks = [c["minusK"] for c in I]
    cls = "Fano" if min(ks) > 0 else ("weak Fano" if min(ks) == 0 else "not nef")
    return cls, P, I


def primitive_relations(P):
    """minimal sets of facets with empty common intersection, and v-sums"""
    F = P["facets"]; cones = [set(t) for _, t in P["verts"]]; V = P["V"]; prim = []
    for r in range(2, len(F) + 1):
        for S in itertools.combinations(F, r):
            if any(set(S) <= c for c in cones): continue
            if any(set(Q) <= set(S) for Q in prim): continue
            prim.append(S)
    return [(S, tuple(sum(V[k][c] for k in S) for c in range(3))) for S in prim]


def identify(f, g):
    cls, P, I = fano_class(f, g); F = P["facets"]; pr = primitive_relations(P)
    if cls != "Fano": return None, pr
    zero = [S for S, s in pr if s == (0, 0, 0)]; nf = len(F)
    if nf == 4: return "CP^3", pr
    if nf == 6 and len(zero) == 3 and all(len(S) == 2 for S in zero): return "CP^1 x CP^1 x CP^1", pr
    if nf == 5:
        sizes = sorted(len(S) for S, _ in pr); z = sorted(len(S) for S in zero)
        if z == [2, 3]: return "CP^2 x CP^1", pr
        if z == [2]: return "P_(CP^2)(O + O(1)) = Bl_pt CP^3", pr
        if z == [3]: return "P_(CP^1)(O + O + O(1)) = Bl_line CP^3", pr
    return "unidentified", pr


def minus_K_cubed(P):
    """(-K)^3 = 3! vol {m : <m, v_c> >= -1, c a facet} (exact for nef -K: the anticanonical polytope)"""
    F = P["facets"]; V = P["V"]; pts = []
    for S in itertools.combinations(F, 3):
        x, D = _solve3([V[k] for k in S], [-1, -1, -1])
        if x is None: continue
        if all(_dot(V[k], x) >= -1 for k in F): pts.append(tuple(x))
    pts = list(set(pts)); h = ConvexHull(np.array(pts, float)); return 6 * h.volume, pts


def normalized_volume(P):
    pts = np.array([P["V"][k] for k in P["facets"]], float); return 6 * ConvexHull(pts).volume


SUBSETS = [J for r in (1, 2, 3) for J in itertools.combinations(range(4), r)]


def chambers():
    """all realisable sign vectors of f1 - g(J) (J in SUBSETS); returns [(signs, (f, g) interior rational point)]"""
    out = []
    for signs in itertools.product((1, -1), repeat=len(SUBSETS)):
        # variables (f1, g1, g2, g3, g4, delta); maximise delta; s * (f1 - g(J)) >= delta; g_j >= delta; f1, 1 - f1 >= delta; sum g = 1
        A = []; b = []
        for s, J in zip(signs, SUBSETS):
            r = np.zeros(6); r[0] = -s
            for j in J: r[1 + j] = s
            r[5] = 1; A.append(r); b.append(0)
        for j in range(4): r = np.zeros(6); r[1 + j] = -1; r[5] = 1; A.append(r); b.append(0)
        r = np.zeros(6); r[0] = -1; r[5] = 1; A.append(r); b.append(0)
        r = np.zeros(6); r[0] = 1; r[5] = 1; A.append(r); b.append(1)
        res = linprog(-np.eye(6)[5], A_ub=np.array(A), b_ub=np.array(b), A_eq=np.array([[0, 1, 1, 1, 1, 0]]), b_eq=[1],
                      bounds=[(0, 1)] * 5 + [(None, 1)], method="highs")
        if res.status == 0 and -res.fun > 1e-9:
            x = res.x; f1 = Fr(x[0]).limit_denominator(2000); gg = [Fr(c).limit_denominator(2000) for c in x[1:4]]; gg.append(1 - sum(gg))
            ok = min(gg) > 0 and 0 < f1 < 1 and all(s * (f1 - sum(gg[j] for j in J)) > 0 for s, J in zip(signs, SUBSETS))
            if not ok: raise RuntimeError("rounding left the chamber")
            out.append((signs, ((f1, 1 - f1), tuple(gg))))
    return out


def leximin_exact(f, g):
    """levels via datawalls.lex_maxmin on the facet cells, then the point and level values solved exactly"""
    import sympy as sp
    from pce import datawalls as DW
    P = polytope(f, g); F = P["facets"]; V, lam = P["V"], P["lam"]
    u, lev = DW.lex_maxmin([V[k] for k in F], [float(lam[k]) for k in F])
    us = sp.symbols("u1:4"); ts = sp.symbols(f"t1:{len(lev) + 1}"); eqs = []
    for j, (h, S) in enumerate(lev):
        for i in S: k = F[i]; eqs.append(sum(V[k][r] * us[r] for r in range(3)) - sp.Rational(lam[k].numerator, lam[k].denominator) - ts[j])
    sol = sp.solve(eqs, list(us) + list(ts), dict=True)
    if len(sol) != 1 or len(sol[0]) != 3 + len(lev): raise RuntimeError("leximin system not uniquely solvable")
    s = sol[0]; point = [Fr(str(s[x])) for x in us]; levels = [Fr(str(s[t])) for t in ts]
    fixed = []; ranks = []
    for h, S in lev: fixed += [F[i] for i in S]; ranks.append(int(np.linalg.matrix_rank(np.array([V[k] for k in fixed], float))))
    drops = [b - a for a, b in zip([0] + ranks[:-1], ranks)]
    sets = [tuple(NAMES[F[i]] for i in S) for _, S in lev]
    vals = [_dot(V[k], point) - lam[k] for k in F]
    return {"point": point, "levels": levels, "sets": sets, "drops": drops, "facet_values": vals}


def is_box(f, g):
    return any(Fr(x) > max(Fr(f[0]), Fr(f[1])) for x in g)


def vertex_test(f, g):
    """at each vertex solve <m, v_k> = -1 on its three facets; ample iff <m, v_k> > -1 on all other facets, nef iff >= -1"""
    P = polytope(f, g); V = P["V"]; worst = None
    for x, t in P["verts"]:
        m, _ = _solve3([V[k] for k in t], [-1, -1, -1])
        for k in P["facets"]:
            if k in t: continue
            val = _dot(V[k], m) + 1; worst = val if worst is None else min(worst, val)
    return "Fano" if worst > 0 else ("weak Fano" if worst == 0 else "not nef")


# ---------------------------------------------------------------- numerical layer


def potential(f, g):
    P = polytope(f, g); return [(-P["lam"][k], P["V"][k]) for k in P["facets"]]     # coefficient T^(-lam), monomial y^v


def _W_parts(terms, L, y):
    out = []
    for e, v in terms:
        out.append(mp.exp(-L * mp.mpf(e.numerator) / e.denominator) * y[0] ** v[0] * y[1] ** v[1] * y[2] ** v[2])
    return out


def _grad_hess(terms, L, y):
    parts = _W_parts(terms, L, y); g = [0, 0, 0]; H = mp.matrix(3, 3); W = sum(parts)
    for p, (e, v) in zip(parts, terms):
        for a in range(3):
            g[a] += p * v[a]
            for b in range(3): H[a, b] += p * v[a] * v[b]
    return W, mp.matrix(g), H


def newton(terms, L, y0, iters=200):
    """Newton in log coordinates z = log y on y dW/dy = 0; returns y or None"""
    z = mp.matrix([mp.log(c) for c in y0])
    for _ in range(iters):
        y = [mp.exp(z[i]) for i in range(3)]; W, gr, H = _grad_hess(terms, L, y)
        try: dz = mp.lu_solve(H, -gr)
        except ZeroDivisionError: return None
        z = z + dz
        if mp.norm(dz) < mp.mpf(10) ** (-mp.mp.dps + 10): break
    y = [mp.exp(z[i]) for i in range(3)]; W, gr, H = _grad_hess(terms, L, y)
    scale = sum(abs(p) for p in _W_parts(terms, L, y))
    return y if mp.norm(gr) < scale * mp.mpf(10) ** (-mp.mp.dps // 2) else None


def critical_points(f, g, L, dps=60, tries=4000, seed=0):
    mp.mp.dps = dps; terms = potential(f, g); P = polytope(f, g); bound = int(round(normalized_volume(P)))
    rng = np.random.default_rng(seed); found = []
    us = [np.array([float(c) for c in x]) for x, _ in P["verts"]]; cen = np.mean(us, axis=0)
    for it in range(tries):
        w = rng.dirichlet(np.ones(len(us))); u = 0.5 * cen + 0.5 * (w @ np.array(us))     # a valuation point inside the polytope
        ph = rng.uniform(0, 2 * np.pi, 3); mag = rng.uniform(0.3, 3, 3)
        y0 = [mp.mpc(mag[i] * math.cos(ph[i]), mag[i] * math.sin(ph[i])) * mp.exp(-L * u[i]) for i in range(3)]
        y = newton(terms, L, y0)
        if y is None: continue
        if all(max(abs(mp.log(y[i]) - mp.log(q[i])) if abs(mp.im(mp.log(y[i]) - mp.log(q[i]))) < 3 else abs(y[i] / q[i] - 1) for i in range(3))
               > mp.mpf(10) ** (-dps // 3) for q in found):
            found.append(y)
        if len(found) == bound: break
    return found, bound


def positive_point(f, g, L, dps=None):
    """the positive critical point = the minimiser of the strictly convex W(e^z) over real z; damped Newton from the exact leximin
    valuation point, multiprecision"""
    mp.mp.dps = dps or int(L / 2.3) + 60; terms = potential(f, g); u = leximin_exact(f, g)["point"]
    z = mp.matrix([-L * mp.mpf(c.numerator) / c.denominator for c in u])
    def Wz(z): return sum(_W_parts(terms, L, [mp.exp(z[i]) for i in range(3)]))
    for _ in range(400):
        W, gr, H = _grad_hess(terms, L, [mp.exp(z[i]) for i in range(3)]); dz = mp.lu_solve(H, -gr); s_ = mp.mpf(1)
        while Wz(z + s_ * dz) > W + mp.mpf("1e-4") * s_ * (gr.T * dz)[0] and s_ > mp.mpf(10) ** -30: s_ /= 2
        z = z + s_ * dz
        if mp.norm(s_ * dz) < mp.mpf(10) ** (-mp.mp.dps + 15): break
    y = [mp.exp(z[i]) for i in range(3)]; W, gr, H = _grad_hess(terms, L, y)
    if mp.norm(gr) > abs(W) * mp.mpf(10) ** (-mp.mp.dps // 2): raise RuntimeError("positive point not converged")
    return y


def clifford_spectrum(f, g, L, y):
    terms = potential(f, g); W, gr, H = _grad_hess(terms, L, y); M = H / W
    ev = mp.eig(M)[0]; return sorted(ev, key=lambda z: -abs(z)), W


def reduction(f, g):
    """the exact reduction of the critical equations. Write t_j = T^(g_j), s = T^(f1), r = T^(g4 - f1), P = y1 y2 y3 and
    c = [14] s / P - [24] r P. Then y_j d_j W = 0 reads a_j(y_j) = c with a_j(y) = [1j] y - [2j] t_j / y ([ij] = 1 if cell ij is a facet).
    Returns (D, polynomials) with T = tau^D: E_j = [1j] y_j^2 - c y_j - [2j] t_j (times y_j) and F = c P - [14] s + [24] r P^2, and the
    resultant R(c) eliminating y1, y2, y3 (sympy, symbolic in tau)"""
    import sympy as sp
    P_ = polytope(f, g); F = set(P_["facets"]); f1 = Fr(f[0]); g = [Fr(x) for x in g]
    exps = [f1, g4m := g[3] - f1] + g[:3]
    D = 1
    for e in exps: D = D * e.denominator // math.gcd(D, e.denominator)
    tau, c = sp.symbols("tau c"); ys = sp.symbols("y1:4")
    def tp(e): return tau ** int(e * D)
    has = lambda i, j: NAMES.index((i, j)) in F
    E = []
    for j in range(3):
        if not has(0, j) and not has(1, j): E.append(None); continue
        E.append((1 if has(0, j) else 0) * ys[j] ** 2 - c * ys[j] - (tp(g[j]) if has(1, j) else 0))
    P = ys[0] * ys[1] * ys[2]
    Fp = sp.expand(c * P - (tp(f1) if has(0, 3) else 0) + (tp(g[3] - f1) * P ** 2 if has(1, 3) else 0))
    if any(e is None for e in E): return D, E, Fp, None
    R = Fp
    for j in range(3): R = sp.resultant(R, E[j], ys[j])
    R = sp.factor(R); return D, E, Fp, R


def critical_points_exact(f, g, L, dps=None):
    """all critical points at T = e^-L from the reduction: roots c of the resultant, then y_j from E_j, the combination fixed by F = 0,
    polished by Newton; returns the list and the Bernstein-Kushnirenko bound"""
    import sympy as sp
    mp.mp.dps = dps or int(L / 2.3) + 60; P_ = polytope(f, g); bound = int(round(normalized_volume(P_)))
    D, E, Fp, R = reduction(f, g); tau = sp.Symbol("tau"); c = sp.Symbol("c"); ys = sp.symbols("y1:4"); tv = mp.exp(-L / D)
    terms = potential(f, g); out = []
    if R is None:                                    # a column with no facet: its equation is c = 0
        cands = [mp.mpf(0)]
    else:
        num = sp.Poly(sp.numer(sp.together(R)), c); coeffs = [sp.lambdify(tau, co, "mpmath")(tv) for co in num.all_coeffs()]
        while coeffs and coeffs[-1] == 0: coeffs.pop()
        cands = [z for z in mp.polyroots(coeffs, maxsteps=2000, extraprec=4 * mp.mp.dps) if abs(z) > 0]
    for cv in cands:
        opts = []
        for j in range(3):
            if E[j] is None: opts.append([None]); continue
            pj = sp.Poly(E[j].subs(c, sp.Symbol("cc")), ys[j]); co = [sp.lambdify((tau, sp.Symbol("cc")), a, "mpmath")(tv, cv) for a in pj.all_coeffs()]
            while co and co[0] == 0: co.pop(0)
            opts.append(mp.polyroots(co, maxsteps=400, extraprec=2 * mp.mp.dps) if len(co) > 1 else [])
        Ff = sp.lambdify((tau, c) + ys, Fp, "mpmath")
        for combo in itertools.product(*opts):
            if any(v is None for v in combo):              # free variable fixed by F: P^2 = s / r
                known = [v for v in combo if v is not None]; Pq = mp.sqrt(mp.exp(-L * mp.mpf(Fr(f[0]).numerator) / Fr(f[0]).denominator) /
                                                                      mp.exp(-L * mp.mpf((Fr(g[3]) - Fr(f[0])).numerator) / (Fr(g[3]) - Fr(f[0])).denominator))
                for Pv in (Pq, -Pq):
                    y = list(combo); i = y.index(None); y[i] = Pv / mp.fprod(known); yy = newton(terms, L, y)
                    if yy is not None: out.append(yy)
                continue
            if abs(Ff(tv, cv, *combo)) < mp.mpf(10) ** (-mp.mp.dps // 3) * (1 + abs(cv * mp.fprod(combo))):
                yy = newton(terms, L, list(combo))
                if yy is not None: out.append(yy)
    uniq = []
    for y in out:
        if all(max(abs(y[i] / q[i] - 1) for i in range(3)) > mp.mpf(10) ** (-mp.mp.dps // 3) for q in uniq): uniq.append(y)
    return uniq, bound


def certificate():
    """the machine-readable certificate of Theorem 41: one record per chamber (all numbers as exact fraction strings)"""
    out = []
    for signs, (f, g) in chambers():
        cls, P, I = fano_class(f, g); name, _ = identify(f, g); V = P["V"]
        out.append({"signs": [[list(J), s] for s, J in zip(signs, SUBSETS)], "f": [str(x) for x in f], "g": [str(x) for x in g],
                    "facets": [list(NAMES[k]) for k in P["facets"]], "rays": [list(V[k]) for k in P["facets"]],
                    "cones": [{"vertex": [str(c) for c in x], "facets": [list(NAMES[k]) for k in t],
                               "det": int(_solve3([V[k] for k in t], [0, 0, 0])[1])} for x, t in P["verts"]],
                    "curves": [{"on": [list(NAMES[i]) for i in c["curve"]], "ends": [list(NAMES[k]) for k in c["ends"]], "a": int(c["a"]), "b": int(c["b"]),
                                "minusK": int(c["minusK"]), "kahler_degree": str(c["omega"]), "lattice_length": str(c["length"])} for c in I],
                    "type": name or f"weak Fano, {len(P['facets'])} facets", "class": cls, "minusK_cubed": int(round(minus_K_cubed(P)[0]))})
    return out
