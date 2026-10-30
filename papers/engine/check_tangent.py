"""check_tangent.py -- the tangent structure for injected data (companion paper "A tangent structure for injected data").
Writes lab/tangent.json for the figures and worked examples.

Points of T^k M are elements of the Weil algebra W_k = R[e_1..e_k]/(e_i^2); every structure map is a substitution of generators.

  TG1  the Weil arithmetic is exact: first and second coefficients of T and T^2 of the lab's maps (moments, rhat, the decision gap,
       the tetrads) equal finite differences; T^2 f is symmetric (naturality of c is Schwarz)
  TG2  the lexicographic lemma: membership in T^k of the half-line from the definition equals the lexicographic test, exhaustively over
       {-1, 0, 1}^(2^k) for k = 1, 2, 3 (3, 9 and 6561 points)
  TG3  the equational axioms of a tangent structure as identities of substitutions: c c = 1, c l = l, l_T l = T(l) l, Yang-Baxter,
       c_T T(c) l_T = T(l) c; l and c are additive bundle morphisms (T(p) l = 0 p, p_T c = T(p), c T(0) = 0_T, additivity)
  TG4  on Delta_3 and Delta_4, at every face type (interior, open edge, vertex; and the open 2-face of Delta_4): p, 0, +, l, c and their
       whiskerings send T^k Delta into the right T^j Delta (k <= 3), and + is a commutative monoid with no negatives off the interior
  TG5  morphisms: stochastic matrices (affine maps of simplices: FinStoch) and the product of laws preserve T^k Delta (k <= 3); a smooth
       self-map of Delta_3 that is not affine breaks second order at a vertex: T^2 f leaves T^2 Delta
  TG6  universality of the vertical lift: on Delta_3 and Delta_4 the fibre of T(p) over the zero section equals the image of the vertical
       lift exactly at interior points; at every boundary point it is strictly larger, and it equals {(u, w): u in C_x, w in C_{x,u}}
  TG6b the same on a polyhedron that is not a simplex (the cube cut by x1 + x2 + x3 <= 2): universal exactly at interior points
  TG7  the explicit counterexample on the edge of Delta_3, with the two-parameter family that realises it
  TG8  the interior category: the decision pipeline P -> (mu, M, m_T) -> rhat -> U_T(rhat) - U_P is a morphism off the coincidence
       strata; T F (P, Q - P) = sum_x Q(x) IF(x) (Theorem E(2)) for F = rhat and the gap; |T rhat| blows up approaching a coincidence
  TG9  stealth directions are vertical: ker T(K) for the pair-marginal map K is the line of chi_ABC at interior P, and a ray at the two
       ends of the stealth range, where T of the fibre is the same ray
  TG10 walls: the value function V = max(U_T, U_P) has directional derivative DV(P; .) that is additive inside a chamber and fails
       additivity on a wall by exactly |l(xi)| (the relevance functional of Theorem L(4)); along chi the defect is |l(chi)| = 1
  TG11 the tetrad scheme with the Weil tangent structure: the vertical lift is universal at the generic, one-coincidence and
       two-coincidence truths (fibre = T_2 V, dimension 2 dim ker J)
  TG12 completing (u, v) in T_2 V to T^2 V: always possible at the generic and one-coincidence truths; on the two-coincidence stratum
       obstructed by [H(u, v)] in coker J, of norm |q(u, v)|/sqrt 2 for the polarised quadric q; its diagonal is twice the
       obstruction of Theorem G(4); on the cone side (p_T, T p) : T^2 Delta -> T_2 Delta is always onto
  TG13 2-jets are the c-invariant diagonal points: a bent injection P + eps A + eps^2 B is the point (P, A, A, 2B) of T^2 V; when the
       class vanishes the tetrads move at order eps^3, otherwise at order eps^2
"""
import itertools, json, os, sys
import numpy as np
from scipy.linalg import null_space
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import tangent as TG
from pce.tangent import HD
from pce.lab import law
from pce import injection as IJ
from pce import lifting as LF
from pce import duality as DU

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
r6 = lambda x: float(np.round(x, 6))
W = LF.we(); P = W["P"]; p8 = IJ.as_vec(P); CHI = IJ.chi(3, (0, 1, 2))
zs = lambda n: (lambda z: z - z.mean())(rng.normal(size=n))

# ------------------------------------------------------------------------------------------------ TG1
ua = np.array([1.0 if x == (0, 0, 0) else 0.0 for x in TG.X3]); ub = np.array([0.872 - float(x[0] or x[1]) for x in TG.X3]); ell = ua - ub
c_post = list(W["ps"][0] - np.r_[W["UP"], 0.0])
def F_r(p): return TG.rhat_hd(HD.var(p, []) if not isinstance(p, HD) else p)
def F_dec(Ph): return TG.poly_hd(TG.rhat_hd(Ph), c_post)
def real_f(fun, p): return fun(HD.var(p, [])).c[0]
worst1 = 0.0; worst2 = 0.0; worst_sym = 0.0; h = 1e-5
for fun in (lambda Ph: TG.rhat_hd(Ph), F_dec, lambda Ph: (Ph * ell).sum()):
    for _ in range(5):
        u, v = zs(8), zs(8); T2 = fun(HD.var(p8, [u, v])); T2s = fun(HD.var(p8, [v, u]))
        fd1 = (real_f(fun, p8 + h * u) - real_f(fun, p8 - h * u)) / (2 * h)
        fd2 = (real_f(fun, p8 + h * u + h * v) - real_f(fun, p8 + h * u - h * v) - real_f(fun, p8 - h * u + h * v) + real_f(fun, p8 - h * u - h * v)) / (4 * h * h)
        worst1 = max(worst1, abs(T2.c[1] - fd1) / max(1, abs(fd1))); worst2 = max(worst2, abs(T2.c[3] - fd2) / max(1, abs(fd2))); worst_sym = max(worst_sym, abs(T2.c[3] - T2s.c[3]))
G4 = law(.35, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])); p16 = IJ.as_vec(G4); J4, H4 = TG.tetrad_jets(p16)
Jfd = IJ.tetrad_jacobian(G4); Jdir = np.array([J4 @ (np.eye(16)[k] - p16) for k in range(16)]).T
check("TG1 Weil arithmetic is exact: T and T^2 coefficients equal finite differences; T^2 f symmetric (Schwarz = naturality of c); tetrad J matches",
      worst1 < 1e-6 and worst2 < 1e-4 and worst_sym < 1e-12 and np.abs(Jdir - Jfd).max() < 1e-7,
      f"first (rel) {worst1:.1e}; second (rel) {worst2:.1e}; asym {worst_sym:.0e}; tetrad J {np.abs(Jdir - Jfd).max():.1e}")

# ------------------------------------------------------------------------------------------------ TG2
agree = {}; counts = {}
for k in (1, 2, 3):
    ok = 0; mem = 0
    for c in itertools.product((-1, 0, 1), repeat=2 ** k):
        a = TG.in_TH(np.array(c, float), k); b = TG.in_T_def(list(c), k); ok += (a == b); mem += a
    agree[k] = ok == 3 ** (2 ** k); counts[k] = mem
OUT["lex_counts"] = counts; OUT["lex_order_T2"] = [bin(S) for S in TG.lex_order(3)]
check("TG2 lexicographic lemma = definition of T^k of the half-line, exhaustively on {-1,0,1}^(2^k), k = 1, 2, 3", all(agree.values()),
      f"members {counts}")

# ------------------------------------------------------------------------------------------------ TG3
def eqmaps(f, g, k, shape=(5,)):
    worst = 0.0
    for _ in range(20):
        c = rng.normal(size=(2 ** k,) + shape); worst = max(worst, np.abs(f(c) - g(c)).max())
    return worst < 1e-14
C = TG.compose
ax = {
    "cc=1": eqmaps(C(TG.flip(2, 0), TG.flip(2, 0)), lambda c: c, 2),
    "cl=l": eqmaps(C(TG.lift(1, 0), TG.flip(2, 0)), C(TG.lift(1, 0)), 1),
    "lT l = T(l) l": eqmaps(C(TG.lift(1, 0), TG.lift(2, 1)), C(TG.lift(1, 0), TG.lift(2, 0)), 1),
    "Yang-Baxter": eqmaps(C(TG.flip(3, 1), TG.flip(3, 0), TG.flip(3, 1)), C(TG.flip(3, 0), TG.flip(3, 1), TG.flip(3, 0)), 3),
    "cT T(c) lT = T(l) c": eqmaps(C(TG.lift(2, 1), TG.flip(3, 0), TG.flip(3, 1)), C(TG.flip(2, 0), TG.lift(2, 0)), 2),
    "T(p) l = 0 p": eqmaps(C(TG.lift(1, 0), TG.proj(2, 0)), C(TG.proj(1, 0), TG.zero(0, 0)), 1),
    "p_T c = T(p)": eqmaps(C(TG.flip(2, 0), TG.proj(2, 1)), C(TG.proj(2, 0)), 2),
    "c T(0) = 0_T": eqmaps(C(TG.zero(1, 0), TG.flip(2, 0)), C(TG.zero(1, 1)), 1),
    "p l = 0 p (pT l)": eqmaps(C(TG.lift(1, 0), TG.proj(2, 1)), C(TG.proj(1, 0), TG.zero(0, 0)), 1),
}
# additivity of l and c (in coordinates): l(x, u + v) = T(+)(l(x,u), l(x,v)); c(x, a + a', b, d + d') = c(x,a,b,d) +_T c(x,a',b,d')
x, u, v, a, a2, b, d, d2 = (rng.normal(size=4) for _ in range(8))
lu = TG.apply(np.array([x, u]), TG.lift(1, 0)); lv = TG.apply(np.array([x, v]), TG.lift(1, 0)); luv = TG.apply(np.array([x, u + v]), TG.lift(1, 0))
Tplus = lambda A, B: np.array([A[0], A[1] + B[1], A[2], A[3] + B[3]])              # adds the inner fibre (same T(p) image)
plusT = lambda A, B: np.array([A[0], A[1], A[2] + B[2], A[3] + B[3]])              # adds the outer fibre (same p_T image)
cc = lambda A: TG.apply(A, TG.flip(2, 0))
ax["l additive"] = np.allclose(Tplus(lu, lv), luv)
ax["c additive"] = np.allclose(cc(np.array([x, a + a2, b, d + d2])), plusT(cc(np.array([x, a, b, d])), cc(np.array([x, a2, b, d2]))))
check("TG3 equational axioms (substitutions in W_k): " + ", ".join(ax), all(ax.values()), f"{sum(ax.values())}/{len(ax)}")

# ------------------------------------------------------------------------------------------------ TG4
def faces(n):
    out = {}
    for k in range(1, n + 1):
        for supp in itertools.combinations(range(n), k):
            x = np.zeros(n); x[list(supp)] = 1.0 / k; out[supp] = x
    return out

def random_member(x, k, tries=4000):
    n = len(x)
    for _ in range(tries):
        c = np.zeros((2 ** k, n)); c[0] = x
        for S in range(1, 2 ** k):
            z = rng.integers(-2, 3, size=n).astype(float); z[-1] -= z.sum(); c[S] = z
        if TG.in_T(c, k): return c
    return None

maps_k = {1: [("l", TG.lift(1, 0)), ("p", TG.proj(1, 0)), ("0", TG.zero(1, 0))],
          2: [("c", TG.flip(2, 0)), ("lT", TG.lift(2, 1)), ("T(l)", TG.lift(2, 0)), ("T(p)", TG.proj(2, 0)), ("pT", TG.proj(2, 1)),
              ("T(0)", TG.zero(2, 0)), ("0T", TG.zero(2, 2)), ("T0T", TG.zero(2, 1))],
          3: [("cT", TG.flip(3, 1)), ("T(c)", TG.flip(3, 0)), ("T(p)T", TG.proj(3, 1))]}
viol = []; tested = 0
for n in (3, 4):
    for supp, x in faces(n).items():
        for k, mp in maps_k.items():
            for _ in range(25):
                c = random_member(x, k)
                if c is None: continue
                for name, m in mp:
                    tested += 1
                    if not TG.in_T(TG.apply(c, m), m[1]): viol.append((n, supp, name))
# the addition: T_2 points (x, u, v) with u, v in C_x; u + v in C_x; T(+) on T(T_2): pairs of T^2 points with the same T(p) image
add_ok = True; noneg = {}
for n in (3, 4):
    for supp, x in faces(n).items():
        for _ in range(30):
            c1, c2 = random_member(x, 1), random_member(x, 1)
            add_ok &= TG.in_T(np.array([x, c1[1] + c2[1]]), 1)
            add_ok &= np.allclose(c1[1] + c2[1], c2[1] + c1[1])
        e = np.zeros(n); i_in = supp[0]; outs = [y for y in range(n) if y not in supp]
        if outs: e[outs[0]] = 1.0; e[i_in] = -1.0
        if outs: noneg[(n, supp)] = TG.in_T(np.array([x, e]), 1) and not TG.in_T(np.array([x, -e]), 1)
        else: noneg[(n, supp)] = all(TG.in_T(np.array([x, -random_member(x, 1)[1]]), 1) for _ in range(10)) is False
OUT["no_negatives"] = {f"D{n} {s}": v for (n, s), v in noneg.items()}
check("TG4 on Delta_3 and Delta_4 at every face type, p, 0, l, c and whiskerings preserve T^k Delta (k <= 3); + is a commutative monoid; negatives fail exactly off the interior",
      not viol and add_ok and all(v == (len(s) < n) for (n, s), v in noneg.items()),
      f"{tested} map applications, violations {len(viol)}")

# ------------------------------------------------------------------------------------------------ TG5
def stoch(nin, nout): return rng.dirichlet(np.ones(nout), nin).T                     # nout x nin, columns sum to 1
aff_ok = True; prod_ok = True
for n_in, n_out in ((3, 3), (3, 4), (4, 2), (4, 4)):
    for supp, x in faces(n_in).items():
        for k in (1, 2, 3):
            for _ in range(6):
                c = random_member(x, k); K = stoch(n_in, n_out)
                if c is not None: aff_ok &= TG.in_T(c @ K.T, k)
for k in (1, 2):
    for _ in range(80):
        x1 = np.zeros(2); x1[rng.integers(2)] = 1.0
        if rng.random() < 0.5: x1 = np.array([0.5, 0.5])
        x2 = np.zeros(2); x2[rng.integers(2)] = 1.0
        c1, c2 = random_member(x1, k), random_member(x2, k)
        if c1 is None or c2 is None: continue
        A, B = HD(c1, k), HD(c2, k); Pp = HD(np.einsum("si,tj->stij", c1, c2).reshape(2 ** k, 2 ** k, 4), k)
        prod = A[0] * B[0], A[0] * B[1], A[1] * B[0], A[1] * B[1]; cc_ = np.stack([q.c for q in prod], axis=1)
        prod_ok &= TG.in_T(cc_, k)
# the non-affine map f(q) = (1 - (q1 - q2)^2, (q1 - q2)^2, 0) at the vertex (1, 0, 0)
xv = np.array([1.0, 0, 0]); a_ = np.array([-1.0, 0, 1]); b_ = np.array([-1.0, 1, 0]); pt = TG.T2_point(xv, a_, b_, np.zeros(3))
def f_na(c, k):
    q = HD(c, k); dq = q[1] - q[2]; return np.stack([(1 - dq * dq).c, (dq * dq).c, (q[0] * 0.0).c], axis=1)
img = f_na(pt, 2); f_ok = TG.in_T(pt, 2) and not TG.in_T(img, 2)
first_ok = all(TG.in_T(f_na(c_, 1), 1) for c_ in [random_member(xv, 1) for _ in range(30)] + [random_member(np.array([.2, .5, .3]), 1)] if c_ is not None)
OUT["nonaffine"] = {"point": pt.tolist(), "image": img.tolist()}
check("TG5 stochastic matrices and products of laws preserve T^k Delta (k <= 3); f = (1-(q1-q2)^2, (q1-q2)^2, 0) preserves T Delta but T^2 f leaves T^2 Delta at the vertex",
      aff_ok and prod_ok and f_ok and first_ok, f"image cell 1 coefficients {img[:, 1].tolist()} (order 1, e1, e2, e1e2)")

# ------------------------------------------------------------------------------------------------ TG6
def zs_grid(n, lo=-1, hi=1):
    return [np.array(z, float) for z in itertools.product(range(lo, hi + 1), repeat=n) if sum(z) == 0]
univ = {}; repair = True
for n in (3, 4):
    G = zs_grid(n)
    for supp, x in faces(n).items():
        fib, img = TG.vertical_fibre(x, G, G); fs = {(tuple(u), tuple(w)) for u, w in fib}; ims = {(tuple(u), tuple(w)) for u, w in img}
        rep = {(tuple(u), tuple(w)) for u in G for w in G if TG.feasible_cone(x, u) and TG.second_cone(x, u, w)}
        univ[(n, supp)] = (len(fs), len(ims), ims <= fs, fs == ims); repair &= fs == rep
iff = all((v[3]) == (len(s) == n) for (n, s), v in univ.items()) and all(v[2] for v in univ.values())
OUT["universality"] = {f"Delta{n} supp {s}": {"fibre": v[0], "image": v[1]} for (n, s), v in univ.items()}
check("TG6 universality of the vertical lift holds exactly at interior points of Delta_3 and Delta_4; elsewhere the fibre of T(p) over 0 is strictly larger and equals {u in C_x, w in C_{x,u}}",
      iff and repair, "; ".join(f"D{n}{''.join(map(str, s))}: {v[0]}/{v[1]}" for (n, s), v in list(univ.items())[:7]))

# a polyhedron that is not a simplex: the cube [0,1]^3 cut by x1 + x2 + x3 <= 2 (rows g.x >= h), at interior, facet, edge, vertex
Gp = np.vstack([np.eye(3), -np.eye(3), -np.ones((1, 3))]); hp = np.r_[np.zeros(3), -np.ones(3), -2.0]
def lexrow(*t): return TG.lex_nonneg(list(t), 1e-12)
def inT2_poly(x, a, b, d):
    return all(lexrow(g @ x - h) and lexrow(g @ x - h, g @ a) and lexrow(g @ x - h, g @ b) and lexrow(g @ x - h, g @ b, g @ a, g @ d) for g, h in zip(Gp, hp))
G3 = [np.array(z, float) for z in itertools.product((-1, 0, 1), repeat=3)]; poly = {}
for name, xp in {"interior": [.5, .5, .5], "facet": [0, .5, .5], "edge": [0, 0, .5], "vertex": [0, 0, 0], "cut vertex": [1, 1, 0]}.items():
    xp = np.array(xp, float); fib = sum(inT2_poly(xp, u, 0 * u, w) for u in G3 for w in G3)
    img = sum(inT2_poly(xp, u, 0 * u, 0 * u) and inT2_poly(xp, w, 0 * w, 0 * w) for u in G3 for w in G3); poly[name] = (int(fib), int(img))
OUT["universality_polyhedron"] = poly
check("TG6b on a polyhedron that is not a simplex (a cut cube), the vertical lift is universal at x iff x is interior",
      all((f == i) == (k == "interior") for k, (f, i) in poly.items()), "; ".join(f"{k} {f}/{i}" for k, (f, i) in poly.items()))

# ------------------------------------------------------------------------------------------------ TG7
xe = np.array([0.5, 0.5, 0.0]); ue = np.array([-1.0, 0.0, 1.0]); we_ = np.array([0.0, 1.0, -1.0]); ptc = TG.T2_point(xe, ue, np.zeros(3), we_)
in_fib = TG.in_T(ptc, 2); in_img = TG.feasible_cone(xe, ue) and TG.feasible_cone(xe, we_)
fam = [(s, t, (xe + t * ue + s * t * we_)[2]) for s, t in ((0.1, 1e-3), (0.01, 1e-4), (0.5, 0.01))]
fam_ok = all(c3 >= 0 for *_, c3 in fam) and all(abs(c3 - t * (1 - s)) < 1e-15 for s, t, c3 in fam)
OUT["counterexample"] = {"x": xe.tolist(), "u": ue.tolist(), "w": we_.tolist(), "cell3_lex": [0.0, 0.0, 1.0, -1.0], "family": fam}
check("TG7 the counterexample on the edge of Delta_3: (x, u, 0, w) = ((1/2,1/2,0), (-1,0,1), 0, (0,1,-1)) is in the fibre but w is not in C_x; the family x + t u + s t w stays in Delta (cell 3 = t(1 - s))",
      in_fib and not in_img and fam_ok)

# ------------------------------------------------------------------------------------------------ TG8
from pce import segment_decision as SD
rfun = lambda L: SD.from_tables(SD.exact_tables(L))[2]
gfun = lambda L: float(np.sum(ell * IJ.as_vec(L)))
IFr = IJ.influence(rfun, P).ravel(); IFg = IJ.influence(gfun, P).ravel()
worst = 0.0; worst_g = 0.0
for _ in range(20):
    Q = rng.dirichlet(np.ones(8)); Tr = TG.rhat_hd(HD.var(p8, [Q - p8])).c[1]; Tg = (HD.var(p8, [Q - p8]) * ell).sum().c[1]
    worst = max(worst, abs(Tr - Q @ IFr)); worst_g = max(worst_g, abs(Tg - Q @ IFg))
r_real = TG.rhat_hd(HD.var(p8, [])).c[0]; same_r = abs(r_real - rfun(P)) < 1e-12
blow = []
for tv in (0.3, 0.1, 0.03, 0.01):
    a1 = np.array([.5, .1, .4]); a2 = np.array([.05, .5, .1]); a2c = a2.copy(); a2c[2] = a1[2] - tv * (a1[2] - a2[2])   # v_2 -> 0
    Lc = IJ.as_vec(law(0.6, a1, a2c)); u = zs(8); u /= np.linalg.norm(u)
    blow.append(max(abs(TG.rhat_hd(HD.var(Lc, [e - e.mean()])).c[1]) for e in np.eye(8)))
OUT["pipeline"] = {"rhat": r6(r_real), "Tr_blowup": [r6(b) for b in blow]}
check("TG8 the pipeline is a morphism off the coincidence strata: T F(P, Q - P) = sum_x Q(x) IF(x) (Theorem E(2)) for rhat and the gap; |T rhat| grows without bound as v_2 -> 0",
      worst < 1e-6 and worst_g < 1e-9 and same_r and all(np.diff(blow) > 0) and blow[-1] > 20 * blow[0], f"max |TF - Q.IF| {worst:.1e}; |T rhat| {['%.2f' % b for b in blow]}")

# ------------------------------------------------------------------------------------------------ TG9
Kmap = np.vstack([TG.marginal_matrix3(w) for w in ((0, 1), (1, 2), (0, 2))]); B8 = null_space(np.ones((1, 8)))
kerK = null_space(Kmap @ B8); kvec = B8 @ kerK[:, 0]; kvec /= kvec[0] / CHI[0]
lo, hi = IJ.stealth_range(P, (0, 1, 2)); ends = {}
for s in (lo, hi):
    Pe = p8 + s * CHI; ray = [t for t in (-1.0, 1.0) if TG.feasible_cone(Pe, t * CHI)]; ends[r6(s)] = ray
ray_ok = ends[r6(lo)] == [1.0] and ends[r6(hi)] == [-1.0] and kerK.shape[1] == 1 and np.allclose(kvec, CHI)
OUT["vertical"] = {"stealth_range": [r6(lo), r6(hi)], "rays": {str(k): v for k, v in ends.items()}}
check("TG9 stealth directions are vertical: ker T(K) = span(chi) at interior P; at the ends of the stealth range it is the inward ray",
      ray_ok, f"range [{lo:.3f}, {hi:.3f}]; rays {ends}")

# ------------------------------------------------------------------------------------------------ TG10
def DV(Pv, xi, h=1e-7):
    V = lambda z: max(ua @ z, ub @ z); return (V(Pv + h * xi) - V(Pv)) / h
s_w = 0.128 - p8[TG.X3.index((0, 0, 1))]; Pw = p8 + s_w * CHI; gap_w = ell @ Pw
defs_w, defs_c = [], []
for _ in range(30):
    xi = zs(8); xi2 = zs(8)
    defs_w.append(abs(DV(Pw, xi) + DV(Pw, -xi) - abs(ell @ xi))); defs_c.append(abs(DV(p8, xi) + DV(p8, xi2) - DV(p8, xi + xi2)))
d_chi = DV(Pw, CHI) + DV(Pw, -CHI)
OUT["wall"] = {"s_wall": r6(s_w), "defect_chi": r6(d_chi), "ell_chi": r6(ell @ CHI)}
check("TG10 walls: DV(P; .) is additive inside the chamber and fails additivity on the wall by exactly |l(xi)|; along chi the defect is |l(chi)| = 1",
      abs(gap_w) < 1e-15 and max(defs_w) < 1e-5 and max(defs_c) < 1e-5 and abs(d_chi - 1.0) < 1e-5, f"wall at s = {s_w:.4f}; defect along chi {d_chi:.6f}")

# ------------------------------------------------------------------------------------------------ TG11-13
truths = {"generic": ([.8, .3, .7, .25], [.2, .7, .15, .7]), "one": ([.8, .3, .7, .5], [.2, .7, .15, .5]), "two": ([.8, .3, .4, .5], [.2, .7, .4, .5])}
B16 = null_space(np.ones((1, 16))); res = {}
for name, (b1, b2) in truths.items():
    L4 = law(.35, np.array(b1), np.array(b2)); pp = IJ.as_vec(L4); J, H = TG.tetrad_jets(pp); JB = J @ B16
    K = B16 @ null_space(JB, rcond=1e-9); dk = K.shape[1]
    # TG11: the fibre of T(p) over 0 in T^2 V: (u, w) with J u = 0 and J w + H(u, 0) = J w = 0, computed from the W_2 coefficients
    rows = []
    for i in range(30):
        basis = np.zeros(30); basis[i] = 1.0; u = B16 @ basis[:15]; w = B16 @ basis[15:]
        T = TG.tetrads_hd(HD(np.array([pp, u, np.zeros(16), w]), 2)); rows.append(np.r_[T[0].c[1], T[1].c[1], T[0].c[3], T[1].c[3], T[0].c[2], T[1].c[2]])
    fib_dim = 30 - np.linalg.matrix_rank(np.array(rows).T, tol=1e-10)
    # TG12: completing pairs
    obs = []; diag = []; qv = []
    for _ in range(12):
        u = K @ rng.normal(size=dk); v = K @ rng.normal(size=dk); o, dd = TG.lift_pair(J, H, u, v, B16); obs.append(o)
        if dd is not None:
            T = TG.tetrads_hd(HD(np.array([pp, u, v, dd]), 2)); obs[-1] = max(obs[-1], abs(T[0].c[3]) + abs(T[1].c[3]) > 1e-9 and 1.0 or 0.0)
        A = K @ rng.normal(size=dk); o_d, _ = TG.lift_pair(J, H, A, A, B16); pB, BB, Jd, Hd = DU.jet(L4); oG, _ = DU.obstruction(Jd, Hd, BB.T @ A)
        diag.append((o_d, oG))
        # polarised quadric on the two-coincidence stratum: q(u, v) from the covariance differentials
        if name == "two":
            def dM(z, i, j):
                xs = np.array(TG.X4, float); mu = xs.T @ pp; dmu = xs.T @ z
                return float(z @ (xs[:, i] * xs[:, j]) - dmu[i] * mu[j] - mu[i] * dmu[j])
            qq = 0.5 * (dM(u, 0, 3) * dM(v, 1, 2) + dM(v, 0, 3) * dM(u, 1, 2) - dM(u, 0, 2) * dM(v, 1, 3) - dM(v, 0, 2) * dM(u, 1, 3))
            qv.append((obs[-1], abs(qq)))
    # a pair on the quadric: v chosen in ker J with q(u, v) = 0 (solve one linear condition on v given u)
    quad_zero = None
    if name == "two":
        u = K @ rng.normal(size=dk); cand = [K[:, i] for i in range(dk)]
        vals = np.array([TG.lift_pair(J, H, u, c, B16)[0] for c in cand])
        Uq = np.array([np.einsum("kij,i,j->k", H, u, c) for c in cand])           # H(u, .) on the basis
        UU, sv, _ = np.linalg.svd(J @ B16); co = UU[:, int(np.sum(sv > 1e-9 * sv[0])):]
        lin = (co.T @ Uq.T).ravel(); ns = null_space(lin[None, :]); v = K @ ns[:, 0]; quad_zero = TG.lift_pair(J, H, u, v, B16)[0]
    # TG13: bent injections
    A = K @ rng.normal(size=dk); A /= np.linalg.norm(A); o_A, d2 = TG.lift_pair(J, H, A, A, B16)
    Bb = 0.5 * d2 if d2 is not None else 0.5 * B16 @ np.linalg.lstsq(J @ B16, -np.einsum("kij,i,j->k", H, A, A), rcond=None)[0]
    es = np.array([1e-3, 3e-3, 1e-2]); mv = [np.abs(IJ.tetrads(IJ.as_law(pp + e * A + e * e * Bb, 4))).max() for e in es]
    slope = np.polyfit(np.log(es), np.log(mv), 1)[0]
    res[name] = dict(rankJ=int(np.linalg.matrix_rank(JB, tol=1e-9)), dimker=dk, fibre=int(fib_dim), max_obs=max(obs), min_obs=min(obs),
                     diag=[(r6(a), r6(b)) for a, b in diag[:3]], diag_ratio=max(abs(a - 2 * b) for a, b in diag), slope=float(slope),
                     quad_zero=quad_zero, qratio=[float(o / max(qq, 1e-300)) for o, qq in qv])
OUT["scheme"] = {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}
check("TG11 tetrad scheme, Weil tangent structure: the vertical lift is universal at all three truths (fibre of T(p) over 0 = T_2 V, dim = 2 dim ker J)",
      all(v["fibre"] == 2 * v["dimker"] for v in res.values()), "; ".join(f"{k}: rank J {v['rankJ']}, ker {v['dimker']}, fibre {v['fibre']}" for k, v in res.items()))
cone_onto = True
for n in (3, 4):
    for supp, x in faces(n).items():
        for _ in range(20):
            c1, c2 = random_member(x, 1), random_member(x, 1); cone_onto &= TG.in_T(TG.T2_point(x, c1[1], c2[1], np.zeros(n)), 2)
check("TG12 (p_T, T p) : T^2 V -> T_2 V is onto at generic and one-coincidence truths, not on the two-coincidence stratum (class [H(u,v)] in coker J, zero on the polarised quadric; diagonal = 2 x Theorem G(4)); onto on the cone side",
      res["generic"]["max_obs"] < 1e-8 and res["one"]["max_obs"] < 1e-8 and res["two"]["min_obs"] > 1e-4 and res["two"]["quad_zero"] < 1e-10
      and np.allclose(res["two"]["qratio"], np.sqrt(2), atol=1e-8)
      and all(v["diag_ratio"] < 1e-4 for v in res.values()) and cone_onto,
      f"two: min class {res['two']['min_obs']:.2e}, on the quadric {res['two']['quad_zero']:.1e}; class / (q(u,v)/2) = {res['two']['qratio'][0]:.6f}; diag vs 2 ob {max(v['diag_ratio'] for v in res.values()):.1e}")
check("TG13 2-jets are the c-invariant diagonal points (P, A, A, 2B): bent injections keep the tetrads to order eps^3 where the class vanishes, order eps^2 on the two-coincidence stratum",
      res["generic"]["slope"] > 2.8 and res["one"]["slope"] > 2.8 and res["two"]["slope"] < 2.3, "; ".join(f"{k}: slope {v['slope']:.2f}" for k, v in res.items()))

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "tangent.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
print(f"{sum(results)}/{len(results)} passed")
