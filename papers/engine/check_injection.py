"""check_injection.py -- data injection on the hiding segment (lab paper, Part II: Theorems E-G). Whole-record injection:
the observed law is P_eps = (1 - eps) P + eps Q.

  IN1 the injected set is a latent actor: a product Q gives exactly the three-type law with weights (1-eps)w, (1-eps)(1-w), eps; at
      m = 4 it breaks the tetrads unless mu_Q - mu_P is parallel to v, in which case the tetrads stay exactly zero for every eps
  IN2 slopes are linear in the injected law: d/d eps F(P_eps) at 0 equals sum_x Q(x) IF(x) (pre-probe constraints and rhat)
  IN3 the vertex attack: the first-order budget min_x margin / (-IF(x)) is attained at a point mass, and the exact budget of the best
      point mass over the first-order budget tends to 1 as the margin shrinks; random laws Q never beat it by more than O(margin)
  IN4 the hidden channel (m = 3): the injections leaving every single and pair table unchanged are exactly Q = P + s chi (kernel of the
      pair-marginal map is spanned by chi); they leave R and the pre-probe act unchanged and move rhat by exactly eps s / v_T
  IN5 budgets: the flip budget eps* = (distance to the oracle switch) |v_T| / |s| and the detection budget eps_det = (distance to the
      wall) |v_T| / |s| (same direction) agree with bisection on the regulator's pipeline
  IN6 m = 4: Q = P + s chi_S leaves all pair tables and the other triple moments unchanged and moves m_S by 2 eps s, so rhat_S moves by
      2 eps s / v_S; the poisoning budget of probe S is proportional to |v_S| (the factor that sets its value and tempo)
  IN7 first-order stealth (m = 4): an injection in the kernel of the tetrad Jacobian moves the tetrads only at second order (log-log
      slope ~2), a vertex attack at first order (slope ~1); the stealth-tilt frontier max{tilt : |N(Q - P)| <= delta} is concave and
      piecewise linear in delta
  IN9 one line (m = 3): the laws with P's pair tables form the line P + s chi; the regulator's hiding segment lies on it, inside the
      positivity segment (triple-moment coordinates: v_T R is strictly inside m_T + [s_lo, s_hi])
  IN10 one geometry (m = 4): the pair fibre has dimension 5; the directions in it that keep every model equation (tetrads, triple
      consistency, quadruple) to first order form a line, the tangent of the regulator's hiding segment; a single-triple channel
      chi_S breaks triple consistency at first order (a second triple sees it), while injection along the segment tangent moves the
      residuals only at second order
  IN11 obstruction (Theorem G(4)): at a generic truth and at a single coincidence every first-order tetrad-stealthy direction extends
      (obstruction 0; a bent injection keeps the tetrads to third order), while on the singular stratum (two coincidences) the
      obstruction is non-zero and the tetrads move at second order along every bent injection
  IN12 the second-triple defence (Corollary): on the stated m = 4 truth, half-capacity chi_{012} injection at eps = 0.05 moves
      rhat_{012} by exactly 2 eps s / v_S (0.629 -> 0.533) and leaves the other three triples at 0.629
  IN13 the stealth-tilt frontier's closed form (Theorem G(2)) on Study 12's H4 truth: the dual min over lambda equals the LP at every
      delta; optimal injections use at most three records; the frontier has exactly two breakpoints (0.219 and 2.744)
  IN14 channel capacity (Corollary): cap_+ and cap_- are the rarest cells of each sign of chi_S; on Study 11's m = 4 truth G the order
      of the triples by |v_S|/sigma_S equals their order by the smallest largest shift 2 cap / |v_S|
  IN8 strata: the tetrad Jacobian has rank 2 at a generic truth and at one coincidence (v_i = 0), rank 1 when two coordinates of v
      vanish, rank 0 at the merge (v = 0): first-order stealthy directions grow on the strata
Run: python check_injection.py
"""
import itertools, os, sys, warnings
import numpy as np
from numpy.polynomial import polynomial as Pn
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.lab import law, moments, fit_v
from pce import segment_decision as SD
from pce import injection as IJ

rng = np.random.default_rng(20261028); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

def truth(m, sep=0.15):
    while True:
        w = rng.uniform(0.2, 0.8); a1 = rng.uniform(0.1, 0.9, m); a2 = rng.uniform(0.1, 0.9, m)
        if np.min(np.abs(a1 - a2)) > sep: return w, a1, a2

def rand_u(k):
    tab = {x: rng.uniform(-1, 1) for x in itertools.product((0, 1), repeat=k)}; return lambda x: tab[tuple(x)]

def product_law(p):
    return law(1.0, np.asarray(p), np.asarray(p))

# IN1
err_mix, tet_generic, tet_axis = 0.0, [], 0.0
for _ in range(5):
    w, a1, a2 = truth(4); P = law(w, a1, a2); b = rng.uniform(0.1, 0.9, 4); Q = product_law(b)
    for e in (0.05, 0.2):
        three = (1 - e) * w * IJ.as_vec(product_law(a1)) + (1 - e) * (1 - w) * IJ.as_vec(product_law(a2)) + e * IJ.as_vec(Q)
        err_mix = max(err_mix, np.max(np.abs(IJ.as_vec(IJ.mix(P, Q, e)) - three)))
    tet_generic.append(np.max(np.abs(IJ.tetrads(IJ.mix(P, Q, 0.1)))))
    mu, M, _ = moments(P); v = fit_v(mu, M); c = 0.5 * min(np.min(mu / np.abs(v)), np.min((1 - mu) / np.abs(v)))
    Qa = product_law(mu + c * v)
    for e in (0.05, 0.1, 0.3): tet_axis = max(tet_axis, np.max(np.abs(IJ.tetrads(IJ.mix(P, Qa, e)))))
check("IN1 a product injection is a third latent type; tetrads break unless the injected mean moves along v", err_mix < 1e-14 and min(tet_generic) > 1e-6 and tet_axis < 1e-15,
      f"mix err {err_mix:.1e}; generic |tetrad| >= {min(tet_generic):.1e}; along v {tet_axis:.1e}")

# helpers for m = 3 decisions
def decision(m_margin=None):
    while True:
        w, a1, a2 = truth(3); P = law(w, a1, a2); acts = [rand_u(3) for _ in range(3)]; out = IJ.pipeline(P, acts, -10)
        if out is None: continue
        W = [SD.worst(p, out["R"]) for p in out["polys"]]; a = int(np.argmax(W)); gap = sorted(W)[-1] - sorted(W)[-2]
        if gap < 0.02: continue
        return P, acts, W, a

# IN2
err2 = 0.0
for _ in range(5):
    P, acts, W, a = decision(); u0 = W[a] - 0.05; F = lambda X: np.append(IJ.constraints(X, acts, u0, a), IJ.pipeline(X, acts, u0)["rhat"])
    IF = IJ.influence(F, P)
    for _ in range(3):
        q = rng.dirichlet(np.ones(8)); Q = IJ.as_law(q, 3); e = 1e-6
        dd = (F(IJ.mix(P, Q, e)) - F(IJ.mix(P, Q, -e))) / (2 * e); err2 = max(err2, np.max(np.abs(dd - q @ IF)) / (1 + np.max(np.abs(dd))))
check("IN2 the slope along any injected law is sum_x Q(x) IF(x)", err2 < 1e-6, f"max rel err {err2:.1e}")

# IN3
errs = {1e-2: [], 1e-3: []}; beaten = 0; tried = 0
for _ in range(6):
    P, acts, W, a = decision()
    for delta in (1e-2, 1e-3):
        u0 = W[a] - delta; F = lambda X: IJ.constraints(X, acts, u0, a); e1, k, j = IJ.first_order_budget(P, F)
        D = np.zeros_like(P); D[IJ.outcomes(3)[k]] = 1.0
        flipped = lambda X: (lambda o: o is None or o["pre"] != a)(IJ.pipeline(X, acts, u0))
        ex = IJ.exact_budget(P, D, flipped, emax=min(0.5, 20 * e1), grid=400); errs[delta].append(abs(ex / e1 - 1))
        if delta == 1e-3:
            for _ in range(20):
                Q = IJ.as_law(rng.dirichlet(np.ones(8) * 0.3), 3); tried += 1
                if IJ.exact_budget(P, Q, flipped, emax=2 * e1, grid=200) < e1 * (1 - 0.05): beaten += 1
check("IN3 vertex attack: exact/first-order budget -> 1 as the margin shrinks; random injections do not beat it", np.median(errs[1e-3]) < np.median(errs[1e-2]) and np.median(errs[1e-3]) < 0.02 and beaten == 0,
      f"median |ratio-1|: margin 1e-2 {np.median(errs[1e-2]):.3f}, 1e-3 {np.median(errs[1e-3]):.4f}; random beats {beaten}/{tried}")

# IN4
A = []
for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
    A.append([float(all(x[i] == 1 for i in S)) for x in IJ.outcomes(3)])
A.append([1.0] * 8); A = np.array(A); ker = np.linalg.svd(A)[2][-1]; c3 = IJ.chi(3, (0, 1, 2))
kerdim = 8 - np.linalg.matrix_rank(A); align = abs(ker @ c3) / np.linalg.norm(c3)
err4 = 0.0; unchanged = True
for _ in range(10):
    P, acts, W, a = decision(); u0 = W[a] - 0.01; base = IJ.pipeline(P, acts, u0); lo, hi = IJ.stealth_range(P, (0, 1, 2))
    for s in (lo, hi):
        Q = IJ.as_law(IJ.as_vec(P) + s * c3, 3)
        for e in (0.03, 0.1):
            o = IJ.pipeline(IJ.mix(P, Q, e), acts, u0)
            unchanged &= np.allclose(o["R"], base["R"]) and o["pre"] == base["pre"] and np.allclose(o["mu"], base["mu"])
            err4 = max(err4, abs(o["rhat"] - (base["rhat"] + e * s / np.prod(base["v"]))))
check("IN4 hidden channel: kernel of the pair-marginal map = span(chi); R and the pre-probe act unchanged; rhat moves by eps s / v_T", kerdim == 1 and align > 1 - 1e-12 and unchanged and err4 < 1e-12,
      f"kernel dim {kerdim}, |cos| {align:.6f}; max rhat err {err4:.1e}")

# IN5
err5 = 0.0; used = 0
for _ in range(40):
    P, acts0, W, a = decision(); base = IJ.pipeline(P, acts0, -10); r0 = base["rhat"]; R = base["R"]; vT = np.prod(base["v"])
    ps0 = base["polys"]; rc = rng.uniform(R[0] + 0.05, R[1] - 0.05)               # make act 1 cross act 0 at rc
    shift = Pn.polyval(rc, ps0[0]) - Pn.polyval(rc, ps0[1]); acts = [acts0[0], (lambda f, c: (lambda x: f(x) + c))(acts0[1], shift)]
    base = IJ.pipeline(P, acts, -10); ps = base["polys"]; u0 = -10.0; a0 = SD.oracle(ps, r0, u0)
    lo, hi = IJ.stealth_range(P, (0, 1, 2)); c3 = IJ.chi(3, (0, 1, 2))
    for s_ in (lo, hi):
        dirn = np.sign(s_ / vT); wall = (R[1] if dirn > 0 else R[0]); e_det = abs(wall - r0) * abs(vT) / abs(s_)
        Q = IJ.as_law(IJ.as_vec(P) + s_ * c3, 3)
        det = IJ.exact_budget(P, Q, lambda X: (lambda o: o is None or o["post"] == "empty")(IJ.pipeline(X, acts, u0)), emax=min(1.0, 2 * e_det), grid=400)
        err5 = max(err5, abs(det - e_det) / e_det)
        if np.sign(rc - r0) == dirn and abs(ps[0][1] - ps[1][1]) > 1e-6:
            e_star = abs(rc - r0) * abs(vT) / abs(s_)
            fl = IJ.exact_budget(P, Q, lambda X: (lambda o: o is not None and o["post"] not in (a0, "empty"))(IJ.pipeline(X, acts, u0)), emax=min(1.0, 2 * e_star + 0.01), grid=400)
            if np.isfinite(fl): err5 = max(err5, abs(fl - e_star) / e_star); used += 1
check("IN5 flip and detection budgets eps* and eps_det agree with bisection on the pipeline", used >= 10 and err5 < 1e-3, f"{used} flips checked, max rel err {err5:.1e}")

# IN6
err6 = 0.0; other_ok = True; ratio = []
for _ in range(5):
    w, a1, a2 = truth(4); P = law(w, a1, a2); mu, M, mS = moments(P); v = fit_v(mu, M)
    for S in itertools.combinations(range(4), 3):
        lo, hi = IJ.stealth_range(P, S); s = 0.5 * hi; Q = IJ.as_law(IJ.as_vec(P) + s * IJ.chi(4, S), 4); e = 0.1
        mu2, M2, mS2 = moments(IJ.mix(P, Q, e)); other_ok &= np.allclose(M2, M) and np.allclose(mu2, mu)
        for S2 in itertools.combinations(range(4), 3):
            exp = mS(S2) + (2 * e * s if S2 == S else 0.0); err6 = max(err6, abs(mS2(S2) - exp))
        v2 = fit_v(mu2, M2); err6 = max(err6, abs(mS2(S) / np.prod(v2[list(S)]) - (mS(S) + 2 * e * s) / np.prod(v[list(S)])))
check("IN6 m = 4: chi_S injection moves only m_S, by 2 eps s; rhat_S moves by 2 eps s / v_S", other_ok and err6 < 1e-12, f"max err {err6:.1e}")

# IN7
slopes_st, slopes_vx, concave = [], [], True
for _ in range(4):
    w, a1, a2 = truth(4); P = law(w, a1, a2); N = IJ.tetrad_jacobian(P)
    tiltF = lambda X: (lambda mu, M, mS: mS((0, 1, 2)) / np.prod(fit_v(mu, M)[:3]))(*moments(X))
    tilt = IJ.influence(tiltF, P).ravel(); val0, q0 = IJ.stealth_tilt_lp(P, tilt, N, 0.0)
    Qs = IJ.as_law(q0, 4); k = int(np.argmax(np.abs(tilt))); Qv = np.zeros_like(P); Qv[IJ.outcomes(4)[k]] = 1.0
    es = np.array([1e-3, 3e-3, 1e-2, 3e-2])
    for Q, L in ((Qs, slopes_st), (Qv, slopes_vx)):
        t = [np.max(np.abs(IJ.tetrads(IJ.mix(P, Q, e)))) for e in es]; L.append(np.polyfit(np.log(es), np.log(t), 1)[0])
    ds = np.linspace(0, 2 * np.max(np.abs(N)), 21); vals = np.array([IJ.stealth_tilt_lp(P, tilt, N, d)[0] for d in ds])
    concave &= bool(np.all(np.diff(vals, 2) <= 1e-9 * (1 + np.abs(vals).max())))
check("IN7 tangent-stealthy injection moves the tetrads at second order, a vertex attack at first; the frontier is concave",
      min(slopes_st) > 1.8 and max(slopes_vx) < 1.2 and concave, f"stealthy slopes {np.round(slopes_st, 2)}; vertex slopes {np.round(slopes_vx, 2)}")

# IN8
def rank_at(a1, a2, w=0.35):
    J = IJ.tetrad_jacobian(law(w, np.array(a1), np.array(a2))); return int(np.linalg.matrix_rank(J, tol=1e-9))
a1b, a2b = [.8, .3, .7, .25], [.2, .7, .15, .7]; r_gen = rank_at(a1b, a2b)
def tie(idx):
    b1, b2 = list(a1b), list(a2b)
    for i in idx: b2[i] = b1[i]
    return rank_at(b1, b2)
r_one = sorted({tie((i,)) for i in range(4)}); r_two = sorted({tie(c) for c in itertools.combinations(range(4), 2)})
r_merge = rank_at([.6, .3, .4, .5], [.6, .3, .4, .5]); r_one = r_one[0] if len(r_one) == 1 else r_one; r_two = r_two[0] if len(r_two) == 1 else r_two
check("IN8 tetrad Jacobian rank 2 (generic), 2 (every single coincidence), 1 (every pair of coincidences), 0 (merge)", (r_gen, r_one, r_two, r_merge) == (2, 2, 1, 0), f"ranks {(r_gen, r_one, r_two, r_merge)}")

# IN9
from pce import duality as DU
from scipy.linalg import null_space
inside = True; line_ok = True
for _ in range(10):
    w, a1, a2 = truth(3); P = law(w, a1, a2); (m_lo, m_hi), (p_lo, p_hi), m0 = DU.m3_line(P)
    inside &= p_lo < m_lo < m0 < m_hi < p_hi or p_lo < m_lo <= m0 <= m_hi < p_hi
    pts, rs = DU.segment_points(P); K = DU.pair_fibre_basis(3)
    for X in pts:
        d = IJ.as_vec(X) - IJ.as_vec(P); line_ok &= np.linalg.norm(d - K @ (K.T @ d)) < 1e-8 and K.shape[1] == 1
check("IN9 one line: pair fibre = P + s chi; the hiding segment lies on it, strictly inside the positivity segment", inside and line_ok)

# IN10
dims, single_first, tangent_second = [], [], []
for _ in range(4):
    w, a1, a2 = truth(4); P = law(w, a1, a2); K = DU.pair_fibre_basis(4); dims.append(K.shape[1])
    R0 = DU.model_residuals(P); h = 1e-6
    Jr = np.array([(DU.model_residuals(IJ.as_law(IJ.as_vec(P) + h * K[:, i], 4)) - DU.model_residuals(IJ.as_law(IJ.as_vec(P) - h * K[:, i], 4))) / (2 * h) for i in range(K.shape[1])]).T
    ker = null_space(Jr, rcond=1e-7); tg = DU.segment_tangent(P); tgK = K.T @ tg
    aligned = ker.shape[1] == 1 and abs(ker[:, 0] @ tgK) / np.linalg.norm(tgK) > 1 - 1e-6
    lo, hi = IJ.stealth_range(P, (0, 1, 2)); chiS = 0.5 * hi * IJ.chi(4, (0, 1, 2))
    es = np.array([1e-4, 3e-4, 1e-3])
    def slope(dirn):
        t = [np.max(np.abs(DU.model_residuals(IJ.as_law(IJ.as_vec(P) + e * dirn, 4)) - R0)) for e in es]; return np.polyfit(np.log(es), np.log(t), 1)[0]
    single_first.append(slope(chiS)); tangent_second.append(slope(tg / np.linalg.norm(tg)) if aligned else 0)
check("IN10 one geometry (m = 4): pair fibre dim 5; model-invisible directions = the hiding segment's tangent; a single channel is seen by a second triple",
      all(d == 5 for d in dims) and min(tangent_second) > 1.8 and max(single_first) < 1.2, f"dims {dims}; single-channel slopes {np.round(single_first, 2)}; tangent slopes {np.round(tangent_second, 2)}")

# IN11
res = {}
for name, (b1, b2) in {"generic": ([.8, .3, .7, .25], [.2, .7, .15, .7]), "one": ([.8, .3, .7, .5], [.2, .7, .15, .5]), "two": ([.8, .3, .4, .5], [.2, .7, .4, .5])}.items():
    P = law(.35, np.array(b1), np.array(b2)); p, B, J, H = DU.jet(P); U, sv, Vt = np.linalg.svd(J); rk = int(np.sum(sv > 1e-7 * sv[0]))
    K = null_space(J, rcond=1e-7); obs, sl = [], []
    for _ in range(6):
        A = K @ rng.normal(size=K.shape[1]); A /= np.linalg.norm(A); o, _ = DU.obstruction(J, H, A); obs.append(o)
        q = 0.5 * np.einsum("kij,i,j->k", H, A, A); b = np.linalg.pinv(J, rcond=1e-7) @ (-q); es = np.array([1e-3, 3e-3, 1e-2])
        t = [np.max(np.abs(DU._tet_vec(p + B @ (e * A + e * e * b)))) for e in es]; sl.append(np.polyfit(np.log(es), np.log(t), 1)[0])
    res[name] = (max(obs), min(sl), max(sl))
check("IN11 obstruction: zero (bent stealth to third order) at generic and single-coincidence truths; non-zero on the two-coincidence stratum",
      res["generic"][0] < 1e-6 and res["one"][0] < 1e-6 and res["generic"][1] > 2.8 and res["one"][1] > 2.8 and min(o for o in [res["two"][0]]) > 1e-3 and res["two"][2] < 2.6,
      "; ".join(f"{k}: max obstruction {v[0]:.1e}, bent slopes {v[1]:.2f}-{v[2]:.2f}" for k, v in res.items()))

# IN12
P = law(.35, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])); mu, M, mS = moments(P); v = fit_v(mu, M); S = (0, 1, 2)
lo, hi = IJ.stealth_range(P, S); s_ = 0.5 * hi; e = 0.05; Q = IJ.as_law(IJ.as_vec(P) + s_ * IJ.chi(4, S), 4)
mu2, M2, mS2 = moments(IJ.mix(P, Q, e)); v2 = fit_v(mu2, M2)
rb = {T: mS(T) / np.prod(v[list(T)]) for T in itertools.combinations(range(4), 3)}; ra = {T: mS2(T) / np.prod(v2[list(T)]) for T in rb}
ok12 = abs(ra[S] - rb[S] - 2 * e * s_ / np.prod(v[list(S)])) < 1e-10 and all(abs(ra[T] - rb[T]) < 1e-10 for T in rb if T != S) and abs(rb[S] - 0.629) < 1e-3 and abs(ra[S] - 0.5333) < 1e-3
check("IN12 second-triple defence: chi_S moves r_S alone, by 2 eps s / v_S", ok12, f"r_012 {rb[S]:.4f} -> {ra[S]:.4f}; others {[round(ra[T], 4) for T in rb if T != S]}")

# IN13
sys.path.insert(0, os.path.join(HERE, "study12_inject")); sys.path.insert(0, os.path.join(HERE, "study11b_settle"))
import design12 as _D12, run_study12 as _R12
from scipy.optimize import linprog
T_ = _D12.tetrad_design(); P4 = T_["P"]; Nj = IJ.tetrad_jacobian(P4)
tiltF = lambda X: (lambda mu, M, mS: mS((0, 1, 2)) / np.prod(fit_v(mu, M)[:3]))(*moments(X)); tilt = IJ.influence(tiltF, P4).ravel()
_, SigT = _R12.tetrad_stat(_R12.exact_tabs(P4, _R12.PAIRS4)); Nw = np.linalg.solve(np.linalg.cholesky(SigT), Nj)
def dual(d):
    A_ = [np.r_[-Nw[:, x], Nw[:, x], -1.0] for x in range(Nw.shape[1])]
    return linprog(np.r_[d * np.ones(4), 1.0], A_ub=np.array(A_), b_ub=-tilt, bounds=[(0, None)] * 4 + [(None, None)], method="highs").fun
ds = np.linspace(0, 3.2, 161); prim = [IJ.stealth_tilt_lp(P4, tilt, Nw, d) for d in ds]
gap = max(abs(p_[0] - dual(d)) for p_, d in zip(prim, ds)); supp = max(int(np.sum(p_[1] > 1e-9)) for p_ in prim)
vals = np.array([p_[0] for p_ in prim]); sl = np.diff(vals) / np.diff(ds); idx = np.where(np.abs(np.diff(sl)) > 1e-4)[0]
kinks = int(len(idx) and 1 + np.sum(np.diff(idx) > 1)); kink_at = [round(float(ds[i + 1]), 2) for i in idx]
check("IN13 frontier closed form: dual = LP at every delta; at most three records; two breakpoints", gap < 1e-8 and supp <= 3 and kinks == 2,
      f"max |primal - dual| {gap:.1e}; largest support {supp}; breakpoints {kinks} near {kink_at}")

# IN14
P = law(0.30, np.array([.70, .30, .80, .55]), np.array([.20, .65, .10, .25])); mu, M, mS = moments(P); v = fit_v(mu, M); rows = []
for S in itertools.combinations(range(4), 3):
    lo, hi = IJ.stealth_range(P, S); c_ = IJ.chi(4, S); p_ = IJ.as_vec(P)
    capok = abs(hi - p_[c_ < 0].min()) < 1e-15 and abs(-lo - p_[c_ > 0].min()) < 1e-15
    vS = abs(np.prod(v[list(S)])); sig = np.sqrt(sum(P[x] * np.prod([(x[i] - mu[i]) for i in S]) ** 2 for x in IJ.outcomes(4)))
    rows.append((S, vS / sig, 2 * max(hi, -lo) / vS, capok))
o1 = [r_[0] for r_ in sorted(rows, key=lambda r_: -r_[1])]; o2 = [r_[0] for r_ in sorted(rows, key=lambda r_: r_[2])]
check("IN14 capacity = rarest cell of each sign; best probe = hardest to poison (same order of triples) on truth G", all(r_[3] for r_ in rows) and o1 == o2,
      "; ".join(f"{''.join(map(str, r_[0]))}: |v|/sigma {r_[1]:.3f}, shift {r_[2]:.1f}" for r_ in rows))

print(f"{sum(results)}/{len(results)} passed")
