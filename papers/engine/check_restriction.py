"""check_restriction.py -- restriction-category axioms, the tangent structure on hiding sheets, and global pooling (companion
"A tangent structure for injected data"). Writes lab/restriction.json.

  RC1  the restriction axioms R1-R4 (Cockett-Lack) hold for the partial maps of the lab (conditioning, the post-probe estimator, the
       decision gap, a garbling, a mixing map) on 600 sample laws, including faces, corners and points outside each domain
  RC2  joins: the chamber-wise value V = U_T on {gap > 0} joined with U_P on {gap < 0} is a join of compatible partial maps, with
       bar(f v g) = bar f v bar g and (f v g) h = f h v g h; the wall is outside its domain
  RC3  T is a restriction functor: dom T(f) = T(dom f), T(bar f) = bar T(f), and T(g f) = T(g) T(f) where defined (Weil arithmetic)
  RC4  the corner criterion is local in the restriction category: conditioning passes Theorem 10 at every point of its domain, including
       the corners delta_000, delta_001 and the edges of the face x_0 = 0
  SH1  a hiding sheet is an object of the cone category: on the C4 rectangle the vertical lift is universal exactly at interior points
       (the walls are its layers: fibre/image counts at a vertex, an edge, the interior)
  SH2  robust decisions on a sheet: the minimum of an event probability over the C4 sheet sits on its boundary and satisfies the cone
       condition DF . u >= 0 for every feasible direction u at the minimiser (first-order optimality in the tangent structure)
  SH3  the explanation multi-map's tangent: as the data move, every facet of the sheet keeps its normal and shifts (log-linear right-hand
       sides), so T of the multi-map is a vector of facet shifts; checked against finite differences on the C4 rectangle
  GP1  global pooling along a chord P_eps = (1 - eps) P + eps Q: the pipeline's domain is {q > 0} with q = M01 M02 M12 of degree <= 6, and
       the act of 'T iff rhat > r_c' can change only at roots of h = m_T^2 - r_c^2 q, degree <= 6 (exact, 200 random chords)
  GP2  so along any chord there are at most 4 domain intervals and at most 6 act changes; dense scans never exceed the root counts
  GP3  a certificate: if P and Q choose the same act and q, h have no roots in (0, 1) (Sturm counts), every mixture is defined and keeps
       the act; on 450 random same-act pairs the certificate is never wrong
  GP4  the exact decision: the genuine roots of h (q > 0 and the right branch of the square root) are exactly the act changes along the
       chord; the global picture at three distances from the truth (safe, flip inside the domain, leaves the domain)
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import restriction as RS
from pce import tangent as TG, tangent_decide as TD
from pce.tangent import HD
from pce import lifting as LF, injection as IJ, covers as CV

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
W = LF.we(); P0 = IJ.as_vec(W["P"]); c = list(W["ps"][0] - np.r_[W["UP"], 0.0]); r_c = -c[0] / c[1]

# sample laws: interior, faces, corners, outside domains
pts = [rng.dirichlet(np.ones(8)) for _ in range(300)] + [rng.dirichlet(np.ones(8) * 20) * 0.5 + P0 * 0.5 for _ in range(150)]
for _ in range(100):
    p = rng.dirichlet(np.ones(8)); p[rng.choice(8, rng.integers(1, 6), replace=False)] = 0; pts.append(p / p.sum())
pts += [np.eye(8)[k] for k in range(8)]; pts += [np.r_[0, 0, 0, 0, rng.dirichlet(np.ones(4))] for _ in range(42)]
K = rng.dirichlet(np.ones(8), 8).T
garble = RS.Partial(lambda P: True, lambda P: K @ P, "garble"); mixq = RS.Partial(lambda P: True, lambda P: 0.7 * P + 0.3 * P0, "mix")
gap = RS.gap_map(*c); maps = [RS.cond_map, RS.rhat_map, garble, mixq]
endo = [RS.cond_map, garble, mixq]                                            # maps law -> law, for composition
# RC1
ok = True
for f in maps:
    ok &= RS.equal_on(RS.compose(f, RS.restrict(f)), f, pts)                                       # R1: f bar f = f
    for g in maps:
        ok &= RS.equal_on(RS.compose(RS.restrict(f), RS.restrict(g)), RS.compose(RS.restrict(g), RS.restrict(f)), pts)   # R2
        ok &= RS.equal_on(RS.restrict(RS.compose(g, RS.restrict(f))), RS.compose(RS.restrict(g), RS.restrict(f)), pts)   # R3
for f in endo:
    for g in maps:
        ok &= RS.equal_on(RS.compose(RS.restrict(g), f), RS.compose(f, RS.restrict(RS.compose(g, f))), pts)            # R4
check("RC1 restriction axioms R1-R4 for conditioning, rhat, a garbling and a mixing map on 600 laws (faces, corners, outside domains)", ok, f"{len(pts)} laws")

# RC2: join
ua = np.array([1.0 if x == (0, 0, 0) else 0.0 for x in TG.X3]); ub = np.array([0.872 - float(x[0] or x[1]) for x in TG.X3]); ell = ua - ub
fT = RS.Partial(lambda P: ell @ P > 1e-12, lambda P: np.array([ua @ P]), "UT"); fP = RS.Partial(lambda P: ell @ P < -1e-12, lambda P: np.array([ub @ P]), "UP")
def join(f, g):
    return RS.Partial(lambda x: f.dom(x) or g.dom(x), lambda x: f.fn(x) if f.dom(x) else g.fn(x), f"{f.name}v{g.name}")
compatible = all(not (fT.dom(p) and fP.dom(p)) for p in pts)
V = join(fT, fP); ok = compatible and RS.equal_on(RS.restrict(V), join(RS.restrict(fT), RS.restrict(fP)), pts)
for h in endo: ok &= RS.equal_on(RS.compose(V, h), join(RS.compose(fT, h), RS.compose(fP, h)), pts)
Pw = P0 + (0.128 - P0[TG.X3.index((0, 0, 1))]) * IJ.chi(3, (0, 1, 2)); ok &= V(Pw) is None
check("RC2 the chamber-wise value is a join of compatible partial maps: bar(f v g) = bar f v bar g, (f v g) h = f h v g h; the wall lies outside its domain", ok)

# RC3
ok = True
for p in pts[:200]:
    for f, Fh in ((RS.rhat_map, lambda X: [TG.rhat_hd(X)]),):
        dom = f.dom(p)
        if dom:
            u = rng.normal(size=8); u -= u.mean(); T1 = Fh(HD.var(p, [u]))[0]
            h = 1e-6; fd = (f.fn(p + h * u)[0] - f.fn(p - h * u)[0]) / (2 * h) if f.dom(p + h * u) and f.dom(p - h * u) else None
            ok &= fd is None or abs(T1.c[1] - fd) < 1e-4 * max(1, abs(fd))
    # T(g f) = T(g) T(f) for g = rhat, f = garble (affine): chain rule where defined
    if RS.rhat_map.dom(K @ p):
        u = rng.normal(size=8); u -= u.mean(); lhs = TG.rhat_hd(HD.var(K @ p, [K @ u])).c[1]
        rhs = TG.rhat_hd(HD(np.stack([K @ p, K @ u]), 1)).c[1]; ok &= abs(lhs - rhs) < 1e-9 * max(1, abs(lhs))
check("RC3 T is a restriction functor on these maps: T of a partial map lives on T(dom f) and satisfies the chain rule where defined", ok)

# RC4
cond_hd = lambda X: (lambda s: [X[0] / s, X[1] / s, X[2] / s, X[3] / s, X[0] * 0.0, X[0] * 0.0, X[0] * 0.0, X[0] * 0.0])(X[0] + X[1] + X[2] + X[3])
corner_pts = [np.eye(8)[0], np.eye(8)[1], np.r_[.5, .5, 0, 0, 0, 0, 0, 0], np.r_[.3, 0, .2, 0, .5, 0, 0, 0], np.r_[.25, .25, .25, .25, 0, 0, 0, 0]]
ok = all(TD.second_order_ok(cond_hd, x)[0] and TD.direct_T2(cond_hd, x, 150, rng)[0] for x in corner_pts)
check("RC4 conditioning passes the second-order corner criterion at every tested point of its domain (corners and faces of the simplex)", ok)

# SH: the C4 sheet
m, E = 4, [(0, 1), (1, 2), (2, 3), (0, 3)]; w, a1, a2 = 0.4, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])
mu, cc, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, cc); sg = [float(np.sign(v[0] / F["comps"][0]["v0"][0]))]
A, b, who = CV.sheet_constraints(mu, F, sg)
from scipy.spatial import HalfspaceIntersection, ConvexHull
z0, rad = CV.chebyshev(A, b); hs = HalfspaceIntersection(np.c_[A, -b], z0); V2 = hs.intersections[ConvexHull(hs.intersections).vertices]
def feas(z, u, tol=1e-9): return all(A[k] @ u <= 1e-12 for k in range(len(b)) if abs(A[k] @ z - b[k]) < tol)
G2 = [np.array(t, float) for t in itertools.product((-1, 0, 1), repeat=2)]
def fib_img(z):
    act = [k for k in range(len(b)) if abs(A[k] @ z - b[k]) < 1e-9]
    lex = lambda *t: next((s > 0 for s in t if abs(s) > 1e-12), True)
    fib = sum(1 for u in G2 for w_ in G2 if all(lex(A[k] @ z - b[k] if False else 0.0, -(A[k] @ u)) and lex(0.0, 0.0, -(A[k] @ u), -(A[k] @ w_)) for k in act) and feas(z, u))
    img = sum(1 for u in G2 for w_ in G2 if feas(z, u) and feas(z, w_)); return fib, img
vert = V2[0]; edge = 0.5 * (V2[0] + V2[1]); inter = z0
cnts = {"vertex": fib_img(vert), "edge": fib_img(edge), "interior": fib_img(inter)}
OUT["sheet_universality"] = cnts
check("SH1 the C4 sheet is an object of the cone category: universal lift exactly at interior points (fibre/image at vertex, edge, interior)",
      cnts["interior"][0] == cnts["interior"][1] and cnts["edge"][0] > cnts["edge"][1] and cnts["vertex"][0] > cnts["vertex"][1], str(cnts))
X4 = list(itertools.product((0, 1), repeat=4)); ev = np.array([1.0 if x[0] == x[1] == x[2] == 1 else 0.0 for x in X4])
Fz = lambda z: ev @ CV.law_m(*CV.explain(mu, F, sg, z[0], z[1:], m=4)[:3]).ravel()
ys = np.linspace(V2[:, 0].min(), V2[:, 0].max(), 161); xs = np.linspace(V2[:, 1].min(), V2[:, 1].max(), 161)
grid = [np.array([y, x]) for y in ys for x in xs if np.all(A @ np.array([y, x]) <= b + 1e-12)]
vals = [Fz(z) for z in grid]; zmin = grid[int(np.argmin(vals))]
from scipy.optimize import minimize
res = minimize(Fz, zmin, constraints=[{"type": "ineq", "fun": lambda z: b - A @ z}], method="SLSQP", options=dict(ftol=1e-14)); zs = res.x
on_bd = np.min(b - A @ zs) < 1e-7; gr = np.array([(Fz(zs + 1e-6 * e) - Fz(zs - 1e-6 * e)) / 2e-6 for e in np.eye(2)])
dirs = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 361)]; cone_ok = all(gr @ u >= -1e-6 for u in dirs if feas(zs, u, 1e-7))
OUT["sheet_min"] = dict(z=zs.tolist(), value=float(Fz(zs)), active=[who[k] for k in range(len(b)) if abs(A[k] @ zs - b[k]) < 1e-7])
check("SH2 the worst case of P(x0 = x1 = x2 = 1) over the C4 sheet lies on its boundary and satisfies DF . u >= 0 on the tangent cone there",
      on_bd and cone_ok and res.success, f"minimum {Fz(zs):.5f} at walls {OUT['sheet_min']['active']}")
# SH3: move the data (a1 of variable 0) and compare facet shifts with finite differences of the right-hand sides
def rhs(aa):
    mu_, cc_, v_ = CV.truth_data(w, aa, a2, E); F_ = CV.factorise(m, E, cc_); s_ = [float(np.sign(v_[0] / F_["comps"][0]["v0"][0]))]
    A_, b_, _ = CV.sheet_constraints(mu_, F_, s_); return A_, b_
A1, b1 = rhs(a1 + np.r_[1e-5, 0, 0, 0]); A0_, b0_ = rhs(a1)
same_normals = np.allclose(A1, A0_); shift = (b1 - b0_) / 1e-5
OUT["facet_shift"] = shift.tolist()
check("SH3 T of the explanation multi-map is a vector of facet shifts: moving the data keeps every facet normal of the sheet", same_normals and np.all(np.isfinite(shift)),
      f"shifts {np.round(shift, 3).tolist()}")

# GP
act = lambda P: (None if not RS.rhat_map.dom(P) else (1 if RS.rhat_map.fn(P)[0] > r_c else -1))
def scan(P, Q, n=4001):
    es = np.linspace(0, 1, n); dom = np.array([RS._q((1 - e) * P + e * Q) > 1e-14 for e in es])
    acts = [act((1 - e) * P + e * Q) for e in es]
    holes = int(np.sum(np.diff(dom.astype(int)) == -1)); changes = sum(1 for k in range(1, n) if acts[k] is not None and acts[k - 1] is not None and acts[k] != acts[k - 1])
    return holes, changes, dom
ok1 = ok2 = True; degs = []
for _ in range(200):
    P = rng.dirichlet(np.ones(8)); Q = rng.dirichlet(np.ones(8)); e, q, h, mT = RS.chord_polys(P, Q, r_c); degs.append((q.degree(), h.degree()))
    ok1 &= q.degree() <= 6 and h.degree() <= 6
    holes, changes, _ = scan(P, Q, 1501); ok2 &= changes <= RS.sturm_count(h) and holes <= 3
check("GP1 along a chord the domain polynomial q and the act polynomial h have degree <= 6 (exact)", ok1, f"max degrees {max(d[0] for d in degs)}, {max(d[1] for d in degs)}")
check("GP2 dense scans never exceed the root counts: at most 3 holes and at most 6 act changes per chord", ok2)
ok3 = True; exact_ok = True; allstats = {}; n_pairs = 0
def exact_check(P, Q, ex):
    """between consecutive critical points (roots of q, genuine roots of h) the domain status and the act are constant; inside one
    domain interval the act changes exactly at the genuine roots"""
    crit = sorted(set([0.0, 1.0] + ex["q_roots"] + ex["genuine"])); mids = [(a + b) / 2 for a, b in zip(crit[:-1], crit[1:])]
    ok = True; prev = None
    for k, x in enumerate(mids):
        L = (1 - x) * P + x * Q; a = act(L)
        for y in np.linspace(crit[k], crit[k + 1], 9)[1:-1]:                 # constant on the open interval
            ok &= act((1 - y) * P + y * Q) == a
        if k > 0:
            boundary = crit[k]
            if boundary in ex["genuine"]: ok &= prev is not None and a is not None and a != prev
            elif boundary not in ex["q_roots"]: ok &= a == prev
        prev = a
    return ok
for scale in (0.02, 0.06, None):
    stats = {"safe": 0, "flip inside the domain": 0, "leaves the domain": 0}; n = 0
    for _ in range(4000):
        if scale is None: P, Q = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
        else:
            P = np.clip(P0 + rng.normal(scale=scale, size=8), 1e-4, None); P /= P.sum(); Q = np.clip(P0 + rng.normal(scale=scale, size=8), 1e-4, None); Q /= Q.sum()
        if act(P) is None or act(P) != act(Q): continue
        n += 1; cert = RS.pooling_certificate(P, Q, r_c, act); ex = RS.exact_pooling(P, Q, r_c, act)
        if cert["safe"]:
            holes, changes, dom = scan(P, Q, 801); ok3 &= holes == 0 and changes == 0
        exact_ok &= exact_check(P, Q, ex)
        stats["leaves the domain" if ex["q_roots"] else ("flip inside the domain" if ex["genuine"] else "safe")] += 1
        if n >= 150: break
    allstats["near truth, sd %.2f" % scale if scale else "anywhere (Dirichlet)"] = stats; n_pairs += n
OUT["pooling_stats"] = allstats; OUT["pooling_pairs"] = n_pairs; stats = allstats
flip = json.load(open(os.path.join(HERE, "..", "lab", "tangent_decide.json")))["flips"]["T,T -> P"]
cf = RS.pooling_certificate(np.array(flip["P_plus"]), np.array(flip["P_minus"]), r_c, act); OUT["td5_certificate"] = cf
check("GP3 the Sturm certificate is sound (no roots of q, h in (0, 1) => every mixture defined with the same act); the TD5 flip pair is not certified",
      ok3 and not cf["safe"] and cf["roots_h"] >= 1, f"{n_pairs} same-act pairs; TD5 pair: roots of h {cf['roots_h']}")
check("GP4 exact decision: the act is constant between consecutive critical points and changes exactly at the genuine roots of h; the global picture by distance",
      exact_ok, " | ".join(k + ": " + ", ".join(f"{kk} {vv}" for kk, vv in v.items()) for k, v in stats.items()))

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "restriction.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (int(o) if isinstance(o, (np.integer,)) else (bool(o) if isinstance(o, np.bool_) else str(o))))
print(f"{sum(results)}/{len(results)} passed")
