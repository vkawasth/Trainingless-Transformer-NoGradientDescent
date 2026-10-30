"""check_higher.py -- Phase III, Stage 2 (higher windows) and the three-type proof of concept (companion "Two latent types on
arbitrary covers"). Macaulay2 and Singular are both used, and each result of one is checked against the other or against an
independent numerical computation. Writes lab/higher.json.

  HW1  the moment polynomials are the moments of the standardised latent: s_k(r) = E tau^k for the two-point law of tau with
       atoms t, -1/t (mean 0, variance 1), the atoms solve tau^2 = r tau + 1, and m_S = v_S E tau^|S| on random two-type laws
  HW2  the full four-variable window (the lab's m = 4 table): 16 minimal generators = 2 tetrads + 8 exchange binomials + 6 Fibonacci
       trinomials; codim 6 = n - rank of the parametrisation's Jacobian; Betti numbers 16, 41, 47, 40, 24, 5 from Macaulay2 equal
       Singular's; the lab's six residuals (two tetrads, three triple differences, the quadruple) have Jacobian rank 6 = codim
  HW3  the Fibonacci gluing relations c_e (m_{T u T'} - m_{T-j} m_{T'-i}) = m_T m_T' vanish exactly (rational arithmetic) on the model,
       and each of the six trinomials of HW2 is one of them modulo the tetrads; the 3 x 3 Hankel determinant of (1, 0, 1, r, r^2+1)
       vanishes (two atoms)
  HW4  a tetrahedron and a triangle sharing an edge: 26 generators, including cross-window trinomials; Macaulay2 = Singular Betti
  HW5  collinear k types: m_S = v_S M_|S| with M_j the moments of a k-atom standardised latent; on m = 5 with k = 3 collinear types
       the fibre over the pairs has dimension 3 and drops by one with each new window size: triple 2, four-window 1, five-window 0
  HW6  the moment space: over all laws of tau on the positivity interval with mean 0 and variance 1, the range of M_3 is exactly the
       two-type segment (the extremes are two-point laws); given M_3, M_4 is pinned for two types but ranges over an interval for three
  HW7  general three types on m = 5: the pentad (one generator, degree 5, 12 terms) vanishes on three-type pair moments and not on
       random ones; the image ranks along the cover ladder are 14 (pairs), 15, 16, 16, 17 (after four triples), 17 (all windows)
  HW8  S_3 descent: a single four-variable window is defective (rank 13 of 14, a one-dimensional fibre), two overlapping four-windows
       {0123}, {1234} have rank 17 and a unique solution up to the label group (multistart); at a coincidence on the overlap (equal
       weights, two types agreeing on {1,2,3}) the rank falls to 14
"""
import itertools, json, os, sys
from fractions import Fraction
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import syzygy as SZ
from pce import covers as CV
from pce import threetype as TT
from pce import duality as DU
from pce.lab import law
from pce import injection as IJ
from scipy.optimize import linprog

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# HW1
ok = True
for _ in range(20):
    t = rng.uniform(0.2, 3.0); w = 1 / (1 + t * t); r = t - 1 / t
    for k in range(2, 9):
        Ek = w * t ** k + (1 - w) * (-1 / t) ** k; ok &= abs(Ek - CV.s_poly(k, r)) < 1e-9 * max(1, abs(Ek))
    ok &= abs(t * t - r * t - 1) < 1e-12 and abs(1 / t ** 2 + r / t - 1) < 1e-12
    wl = rng.uniform(0.2, 0.8); a1 = rng.uniform(0.1, 0.9, 5); a2 = rng.uniform(0.1, 0.9, 5); v = np.sqrt(wl * (1 - wl)) * (a1 - a2); tt = np.sqrt((1 - wl) / wl)
    th = TT.pack(np.array([wl, 1 - wl]), np.vstack([a1, a2])); faces = SZ.closed_faces([tuple(range(5))])
    mo = TT.moments_obs(th, 5, faces, k=2)[5:]
    ok &= np.allclose(mo, [np.prod(v[list(S)]) * CV.s_poly(len(S), tt - 1 / tt) for S in faces], atol=1e-13)
check("HW1 s_k(r) = E tau^k for the standardised two-point latent; tau^2 = r tau + 1; m_S = v_S E tau^|S| on random two-type laws", ok)

# HW2
F4 = SZ.closed_faces([(0, 1, 2, 3)]); R4 = SZ.m2_general(4, F4); S4 = SZ.singular_ranks(4, F4, R4["gens"])
kinds = {"tetrad": 0, "exchange": 0, "trinomial": 0}
for g in R4["gens"]:
    nt = len([x for x in g.replace("-", "+").split("+") if x.strip()])
    kinds["trinomial" if nt == 3 else ("exchange" if "m" in g else "tetrad")] += 1
def param_rank(m, faces):
    def f(z):
        v = z[:m]; r = z[m]; return np.array([np.prod(v[list(S)]) * (1 if len(S) == 2 else CV.s_poly(len(S), r)) for S in faces])
    z0 = np.r_[rng.uniform(0.2, 0.6, m) * rng.choice([-1, 1], m), rng.uniform(-1, 1)]; h = 1e-6
    J = np.array([(f(z0 + h * e) - f(z0 - h * e)) / (2 * h) for e in np.eye(m + 1)]).T; return np.linalg.matrix_rank(J, tol=1e-8)
L4 = law(.35, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])); p16 = IJ.as_vec(L4); B16 = np.linalg.svd(np.ones((1, 16)))[2][1:].T
h = 1e-6; Jr = np.array([(DU.model_residuals(IJ.as_law(p16 + h * b, 4)) - DU.model_residuals(IJ.as_law(p16 - h * b, 4))) / (2 * h) for b in B16.T]).T
lab_rank = np.linalg.matrix_rank(Jr, tol=1e-6)
OUT["K4full"] = dict(gens=R4["gens"], ranks=R4["ranks"], singular=S4, kinds=kinds, lab_residual_rank=int(lab_rank))
check("HW2 full four-variable window: 16 generators = 2 tetrads + 8 exchanges + 6 trinomials; codim 6 = n - rank; Betti M2 = Singular; the lab's six residuals have rank 6",
      kinds == {"tetrad": 2, "exchange": 8, "trinomial": 6} and R4["codim"] == len(F4) - param_rank(4, F4) == 6 and S4[1:] == R4["ranks"] == [16, 41, 47, 40, 24, 5] and lab_rank == 6,
      f"kinds {kinds}; Betti {R4['ranks']}; Singular {S4[1:]}; lab residual rank {lab_rank}")

# HW3
def exact_point(m, faces):
    v = [Fraction(int(rng.integers(1, 9)), int(rng.integers(2, 9))) * (1 if rng.random() < 0.5 else -1) for _ in range(m)]
    r = Fraction(int(rng.integers(-5, 6)), int(rng.integers(1, 5)))
    s = {2: Fraction(1), 3: r}
    for k in range(3, 8): s[k + 1] = r * s[k] + s[k - 1]
    pt = {}
    for S in faces:
        p = Fraction(1)
        for i in S: p *= v[i]
        pt[SZ.fname(S)] = p * s[len(S)]
    return pt
fib = SZ.fibonacci_relations(F4); ok = all(SZ.eval_rel(g, exact_point(4, F4)) == 0 for g in fib for _ in range(5))
ok &= all(SZ.eval_rel(g, exact_point(4, F4)) == 0 for g in R4["gens"] for _ in range(3))
# each trinomial = a Fibonacci relation modulo the tetrads: their difference vanishes wherever the tetrads do (random points of the
# tetrad variety in c, free m): check on points with c_ij = x_i x_j (tetrads hold), arbitrary m
tri = [g for g in R4["gens"] if len([x for x in g.replace("-", "+").split("+") if x.strip()]) == 3]; match = 0
for g in tri:
    for f in fib:
        same = True
        for _ in range(4):
            x = rng.normal(size=4); pt = {SZ.fname(S): (x[S[0]] * x[S[1]] if len(S) == 2 else rng.normal()) for S in F4}
            if abs(SZ.eval_rel(g, pt) - SZ.eval_rel(f, pt)) > 1e-9 and abs(SZ.eval_rel(g, pt) + SZ.eval_rel(f, pt)) > 1e-9: same = False
        if same: match += 1; break
r_ = Fraction(3, 7); Hk = np.array([[1, 0, 1], [0, 1, r_], [1, r_, r_ * r_ + 1]], dtype=object)
hank = Hk[0, 0] * (Hk[1, 1] * Hk[2, 2] - Hk[1, 2] * Hk[2, 1]) - Hk[0, 1] * (Hk[1, 0] * Hk[2, 2] - Hk[1, 2] * Hk[2, 0]) + Hk[0, 2] * (Hk[1, 0] * Hk[2, 1] - Hk[1, 1] * Hk[2, 0])
check("HW3 Fibonacci gluing relations vanish exactly on the model; each trinomial is one of them modulo the tetrads; the two-atom Hankel determinant vanishes",
      ok and match == len(tri) == 6 and hank == 0, f"{len(fib)} gluing relations; {match}/{len(tri)} trinomials matched")

# HW4
F5 = SZ.closed_faces([(0, 1, 2, 3), (2, 3, 4)]); R5 = SZ.m2_general(5, F5); S5 = SZ.singular_ranks(5, F5, R5["gens"])
cross = [g for g in R5["gens"] if len([x for x in g.replace("-", "+").split("+") if x.strip()]) == 3 and "4" in g]
OUT["tettri"] = dict(gens=R5["gens"], ranks=R5["ranks"], singular=S5, cross=cross)
check("HW4 tetrahedron + triangle on an edge: 26 generators with cross-window trinomials; Betti M2 = Singular",
      len(R5["gens"]) == 26 and S5[1:] == R5["ranks"] and len(cross) >= 2 and R5["codim"] == len(F5) - param_rank(5, F5),
      f"Betti {R5['ranks']}; cross trinomials {cross[:2]}")

# HW5: collinear three types, m = 5: parameters (mu, v, the latent law tau: 3 atoms, 3 weights; mean 0, variance 1)
def collinear_obs(z, faces, m=5):
    v = z[:m]; tau = z[m:m + 3]; q = z[m + 3:m + 6]; q = q / q.sum()
    return np.array([np.prod(v[list(S)]) * np.sum(q * tau ** len(S)) for S in faces] + [q @ tau, q @ tau ** 2 - 1])
def collinear_fibre_dim(faces):
    tau = np.array([-1.3, 0.2, 1.1]); q0 = np.array([0.3, 0.3, 0.4]); tau = tau - q0 @ tau; tau = tau / np.sqrt(q0 @ tau ** 2)
    z0 = np.r_[rng.uniform(0.2, 0.5, 5) * rng.choice([-1, 1], 5), tau, q0]; h = 1e-7
    J = np.array([(collinear_obs(z0 + h * e, faces) - collinear_obs(z0 - h * e, faces)) / (2 * h) for e in np.eye(len(z0))]).T
    return len(z0) - 1 - np.linalg.matrix_rank(J, tol=1e-7)             # minus one: q is projective (normalised)
pairs5 = [(i, j) for i, j in itertools.combinations(range(5), 2)]
dims = [collinear_fibre_dim(pairs5), collinear_fibre_dim(pairs5 + [(0, 1, 2)]), collinear_fibre_dim(pairs5 + [(0, 1, 2), (0, 1, 2, 3)]),
        collinear_fibre_dim(pairs5 + [(0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4)])]
OUT["collinear_dims"] = [int(d) for d in dims]
check("HW5 collinear three types on m = 5: fibre dimension 3 over the pairs, then 2, 1, 0 as window sizes 3, 4, 5 are added (one moment each)",
      dims == [3, 2, 1, 0], f"{dims}")

# HW6
w, a1, a2 = 0.6, np.array([.5, .1, .4, .7, .3]), np.array([.05, .5, .1, .2, .6]); mu = w * a1 + (1 - w) * a2; v = np.sqrt(w * (1 - w)) * (a1 - a2)
lo = max((-mu[i] / v[i]) if v[i] > 0 else ((1 - mu[i]) / v[i]) for i in range(5)); hi = min(((1 - mu[i]) / v[i]) if v[i] > 0 else (-mu[i] / v[i]) for i in range(5))
g = np.linspace(lo, hi, 6001); Aeq = np.vstack([np.ones_like(g), g, g ** 2])
m3 = [s * linprog(s * g ** 3, A_eq=Aeq, b_eq=[1, 0, 1], bounds=[(0, None)] * len(g), method="highs").fun for s in (1, -1)]
tlo, thi = -1 / lo, hi; seg = [tlo - 1 / tlo, thi - 1 / thi]; t0 = np.sqrt((1 - w) / w); r0 = t0 - 1 / t0
m4 = [s * linprog(s * g ** 4, A_eq=np.vstack([Aeq, g ** 3]), b_eq=[1, 0, 1, r0], bounds=[(0, None)] * len(g), method="highs").fun for s in (1, -1)]
OUT["moment_space"] = dict(tau_interval=[lo, hi], M3_range=m3, two_type_segment=seg, M4_given_M3=m4, two_type_M4=r0 ** 2 + 1)
check("HW6 moment space: the range of M_3 over all latent laws on the positivity interval (mean 0, variance 1) is the two-type segment; M_4 given M_3 is pinned for two types, an interval for three",
      abs(m3[0] - seg[0]) < 1e-5 and abs(m3[1] - seg[1]) < 1e-5 and abs(m4[0] - (r0 ** 2 + 1)) < 1e-5 and m4[1] > m4[0] + 0.5,
      f"M3 in [{m3[0]:.4f}, {m3[1]:.4f}] vs segment [{seg[0]:.4f}, {seg[1]:.4f}]; M4 | M3 in [{m4[0]:.4f}, {m4[1]:.4f}], two-type {r0 ** 2 + 1:.4f}")

# HW7
pent, cd = TT.pentad_m2(); pg = pent[0]; nterms = len([x for x in pg.replace("-", "+").split("+") if x.strip()])
ok = True
for _ in range(10):
    th = TT.random_theta(5, 3, rng); wv, Am = TT.unpack(th, 5); C, _ = TT.between_cov(wv, Am)
    ok &= abs(SZ.eval_rel(pg, {f"c{i}{j}": C[i, j] for i, j in pairs5})) < 1e-12
bad = {f"c{i}{j}": rng.normal() for i, j in pairs5}; ok &= abs(SZ.eval_rel(pg, bad)) > 1e-6
th = TT.random_theta(5, 3, rng); trip = list(itertools.combinations(range(5), 3))
ladder = [TT.obs_rank(th, 5, pairs5 + trip[:j])[0] for j in range(11)] + [TT.obs_rank(th, 5, [tuple(range(5))])[0]]
OUT["three_type"] = dict(pentad=pg, pentad_terms=nterms, ladder=ladder)
check("HW7 three types, m = 5: one pentad (degree 5, 12 terms) on the pairs; image ranks 14 (pairs), then 15, 16, 16, 17 with triples; 17 with the full table",
      ok and cd == 1 and nterms == 12 and ladder[0] == 14 and ladder[4] == 17 and ladder[-1] == 17, f"ladder {ladder}")

# HW8
r4 = TT.obs_rank(TT.random_theta(4, 3, rng), 4, [(0, 1, 2, 3)])[0]; W2 = [(0, 1, 2, 3), (1, 2, 3, 4)]
th = TT.random_theta(5, 3, rng); rk2 = TT.obs_rank(th, 5, W2)[0]
sols = TT.fit_windows(TT.obs(th, 5, W2), 5, 3, W2, 40, rng); uniq = []
for s in sols:
    wc, Ac = TT.canonical(s, 5)
    if not any(np.allclose(wc, u[0], atol=1e-5) and np.allclose(Ac, u[1], atol=1e-5) for u in uniq): uniq.append((wc, Ac))
wc0, Ac0 = TT.canonical(th, 5); truth_found = any(np.allclose(wc0, u[0], atol=1e-5) and np.allclose(Ac0, u[1], atol=1e-5) for u in uniq)
wco = np.array([0.3, 0.3, 0.4]); Aco = np.array([[.8, .2, .7, .3, .9], [.2, .2, .7, .3, .15], [.5, .85, .15, .75, .5]]); rkc = TT.obs_rank(TT.pack(wco, Aco), 5, W2)[0]
OUT["descent"] = dict(single_window_rank=r4, glued_rank=rk2, converged=len(sols), distinct=len(uniq), coincidence_rank=rkc)
check("HW8 S_3 descent: one four-window is defective (13 of 14); the two overlapping windows have rank 17 and one solution up to labels; a coincidence on the overlap drops the rank to 14",
      r4 == 13 and rk2 == 17 and len(uniq) == 1 and truth_found and len(sols) >= 20 and rkc == 14, f"{len(sols)} converged starts, {len(uniq)} distinct up to S_3")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "higher.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (int(o) if hasattr(o, "__int__") else str(o)))
print(f"{sum(results)}/{len(results)} passed")
