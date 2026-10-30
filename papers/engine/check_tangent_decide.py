"""check_tangent_decide.py -- the tangent structure applied to decision boundaries (companion "A tangent structure for injected data",
Sections "Non-affine maps at corners" and "Decision boundaries"). Writes lab/tangent_decide.json.

  TD1  the second-order criterion (Hessian of each vanishing output cell zero on L x K and >= 0 on pairs of extreme rays of K) equals
       the direct test (T^2 f of sampled points of T^2 Delta, lexicographic membership) on 60 random quadratic self-maps of Delta_3 at
       seven points each; the counterexample fails on a pair of rays (-2) though not on any single ray
  TD2  a subcategory: affine maps, products and Bayesian conditioning pass at every face type; composites of passing maps pass
  TD3  Bayesian conditioning P -> P(. | x_0 = 0) on Delta_4 is a partial map (a restriction-category morphism, defined where
       P(x_0 = 0) > 0) and passes the criterion at every point of its domain tested, corners included
  TD4  the lab's post-probe decision boundary {U_T(rhat) = U_P} is curved: its second fundamental form at a boundary point next to the
       worked truth has eigenvalues of both signs
  TD5  mixture flips: along the most positive (negative) curvature direction there are two laws that both choose T (P) whose 50:50
       mixture chooses P (T); all six are laws in the pipeline's domain
  TD6  affine decision functionals (expected utility of fixed acts) never flip: on 2000 random pairs and shares, F(P), F(Q) > 0 implies
       F(mixture) > 0
  TD7  second-order budgets: for the vertex attacks that flip the act, eps2 (from the 2-jet) is closer to the exact budget than eps1;
       curvature reverses the first-order verdict for two records (010: eps1 = 0.43 but no flip; 101: no first-order flip, exact
       0.46); the hidden channel is affine (F'' = 0) and its first-order budget is exact
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import tangent as TG
from pce import tangent_decide as TD
from pce.tangent import HD
from pce import lifting as LF
from pce import injection as IJ

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
r6 = lambda x: float(np.round(x, 6))
PTS3 = [np.array(p, float) for p in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.3, .3, .4)]]
grid3 = [np.array(q) for q in itertools.product(np.linspace(0, 1, 21), repeat=3) if abs(sum(q) - 1) < 1e-9]

def rand_map():
    forms = []
    for _ in range(2):
        t = rng.integers(3)
        forms.append(("sq", rng.integers(-2, 3, 3).astype(float)) if t == 0 else ("prod", tuple(rng.choice(3, 2, replace=False))) if t == 1 else ("sqi", int(rng.integers(3))))
    def ev(fm, X):
        k, p = fm
        if k == "sq": z = X[0] * p[0] + X[1] * p[1] + X[2] * p[2]; return z * z
        if k == "prod": return X[p[0]] * X[p[1]]
        return X[p] * X[p]
    mx = max(sum(ev(fm, HD.var(q, [])).c[0] for fm in forms) for q in grid3); a = 0.9 / mx if mx > 0 else 1.0
    def f(X):
        o = [ev(fm, X) * a for fm in forms]; s = X[0] + X[1] + X[2]; return [s * s - o[0] - o[1], o[0], o[1]]
    return f

# TD1
f_ce = lambda X: (lambda dq: [1 - dq * dq, dq * dq, X[0] * 0.0])(X[1] - X[2])
ok_ce, det_ce = TD.second_order_ok(f_ce, np.array([1.0, 0, 0])); xv = np.array([1.0, 0, 0])
g, H = TD.jets_scalar(lambda X: f_ce(X)[1], xv, 3); L, R = TD.cone_K(xv, g); diag = [float(r @ H @ r) for r in R]; off = float(R[0] @ H @ R[1])
agree = 0; tot = 0; fails = 0
for _ in range(60):
    f = rand_map()
    for x in PTS3:
        ok, _ = TD.second_order_ok(f, x); d, ntest, nbad = TD.direct_T2(f, x, 200, rng); tot += 1; agree += (ok == d); fails += (not ok)
OUT["criterion"] = dict(agree=agree, total=tot, fails=fails, counterexample_rays_diag=diag, counterexample_rays_off=off)
check("TD1 the Hessian criterion equals the direct T^2 test on 420 random quadratic maps and points; the counterexample fails on a pair of rays only",
      agree == tot and fails >= 10 and not ok_ce and min(diag) >= 0 and off < 0, f"{agree}/{tot} agree, {fails} fail; rays: diagonal {diag}, pair {off}")

# TD2
def stoch_map(K): return lambda X: [sum(X[j] * K[i, j] for j in range(K.shape[1])) for i in range(K.shape[0])]
def prod_map(X):                                        # Delta_2 x Delta_2 inside Delta_4 coordinates (x0, x1, y0, y1) -> 2 x 2 table
    return [X[0] * X[2], X[0] * X[3], X[1] * X[2], X[1] * X[3]]
cond = lambda X: (lambda s: [X[0] / s, X[1] / s])(X[0] + X[1])
ok = True
for x in PTS3:
    K = rng.dirichlet(np.ones(3), 3).T; ok &= TD.second_order_ok(stoch_map(K), x)[0]
for x1 in ([1, 0], [0, 1], [.5, .5]):
    for x2 in ([1, 0], [0, 1], [.4, .6]):
        x = np.array(x1 + x2, float)
        # the product map lives on Delta_2 x Delta_2; test it with the direct T^2 test on that polytope: points of T^2 of each factor
        for _ in range(60):
            cs = []
            for xx in (np.array(x1, float), np.array(x2, float)):
                while True:
                    c = np.zeros((4, 2)); c[0] = xx
                    for S in (1, 2, 3): z = rng.integers(-2, 3, size=2).astype(float); z[-1] -= z.sum(); c[S] = z
                    if TG.in_T(c, 2): break
                cs.append(c)
            img = TD.outputs(prod_map, HD(np.concatenate(cs, axis=1), 2)); ok &= TG.in_T(img, 2)
comp_ok = True; ncomp = 0
for _ in range(40):
    f1, f2 = rand_map(), rand_map()
    for x in PTS3:
        y = np.array([v.c[0] for v in f1(HD.var(x, []))])
        if TD.second_order_ok(f1, x)[0] and TD.second_order_ok(f2, y)[0]:
            ncomp += 1; comp_ok &= TD.second_order_ok(lambda X: f2(HD(np.stack([v.c for v in f1(X)], axis=1), X.k)), x)[0]
check("TD2 affine maps, products and conditioning pass; composites of passing maps pass (a subcategory)", ok and comp_ok and ncomp > 50, f"{ncomp} composites tested")

# TD3
ok = True; pts4 = [(1, 0, 0, 0), (0, 1, 0, 0), (.5, .5, 0, 0), (.5, 0, .5, 0), (.4, 0, 0, .6), (0, .3, .7, 0), (.25, .25, .25, .25)]
for x in pts4:
    x = np.array(x, float); ok &= TD.second_order_ok(cond, x)[0] and TD.direct_T2(cond, x, 200, rng)[0]
check("TD3 Bayesian conditioning is a restriction-category morphism (domain P(x_0 = 0) > 0) passing the criterion at every point tested, corners included", ok)

# TD4, TD5
W = LF.we(); P = IJ.as_vec(W["P"]); c = list(W["ps"][0] - np.r_[W["UP"], 0.0])
F = lambda X: TG.poly_hd(TG.rhat_hd(X), c); B = np.linalg.svd(np.ones((1, 8)))[2][1:].T
gP, HP = TD.grad_hess(F, P, B); d = -B @ gP / np.linalg.norm(gP)
s0 = brentq(lambda s: F(HD.var(P + s * d, [])).c[0], 0, 0.01); P0 = P + s0 * d
ev, V, g0, H0 = TD.second_form(F, P0, B)
OUT["boundary"] = dict(P0=P0.tolist(), II=ev.tolist(), shift=r6(s0))
check("TD4 the post-probe decision boundary is curved both ways at a boundary point next to the truth", ev.min() < -0.5 and ev.max() > 0.5, f"II eigenvalues {np.round(ev, 3).tolist()}")
flips = {}; ok = True
for k, name in ((int(np.argmax(ev)), "T,T -> P"), (int(np.argmin(ev)), "P,P -> T")):
    delta = 0.02; eta = (-0.25 if ev[k] > 0 else 0.25) * abs(ev[k]) * delta ** 2
    Pp, Pm, Mid, vals = TD.mixture_flip(F, P0, B, V[:, k], delta, eta)
    valid = min(Pp.min(), Pm.min()) > 0 and all(TG.rhat_hd(HD.var(p, [])).c[0] == TG.rhat_hd(HD.var(p, [])).c[0] for p in (Pp, Pm, Mid))
    sgn = 1 if ev[k] > 0 else -1; ok &= valid and sgn * vals[0] > 0 and sgn * vals[1] > 0 and sgn * vals[2] < 0
    es = np.linspace(0, 1, 2001); Fe = [F(HD.var((1 - e) * Pp + e * Pm, [])).c[0] for e in es]; bad = [e for e, v in zip(es, Fe) if sgn * v < 0]
    flips[name] = dict(P_plus=Pp.tolist(), P_minus=Pm.tolist(), values=[r6(v) for v in vals], flip_shares=[r6(min(bad)), r6(max(bad))] if bad else None, curvature=r6(ev[k]))
OUT["flips"] = flips
check("TD5 mixture flips both ways: two laws choosing the same act whose mixture chooses the other (valid laws in the domain)",
      ok, "; ".join(f"{k}: F = {v['values']}, flips for shares {v['flip_shares']}" for k, v in flips.items()))

# TD6
ua = np.array([1.0 if x == (0, 0, 0) else 0.0 for x in TG.X3]); ub = np.array([0.872 - float(x[0] or x[1]) for x in TG.X3]); ell = ua - ub
ok = True; n = 0
for _ in range(2000):
    Pq, Qq = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8)); e = rng.uniform()
    if ell @ Pq > 0 and ell @ Qq > 0: n += 1; ok &= ell @ ((1 - e) * Pq + e * Qq) > 0
check("TD6 affine decision functionals never flip under mixing (a half-space is convex)", ok and n > 50, f"{n} same-act pairs")

# TD7
bud = {}
for k, x in enumerate(TG.X3):
    u = np.eye(8)[k] - P; r = TD.budget_jets(F, P, u); bud["".join(map(str, x))] = {kk: (r6(v) if np.isfinite(v) else None) for kk, v in r.items()}
chi = IJ.chi(3, (0, 1, 2)); rc = TD.budget_jets(F, P, -chi if TG.rhat_hd(HD.var(P, [chi])).c[1] * c[1] > 0 else chi, emax=0.03)
OUT["budgets"] = bud; OUT["budget_chi"] = {kk: r6(v) for kk, v in rc.items()}
better = all(abs(b["eps2"] - b["exact"]) < abs(b["eps1"] - b["exact"]) for b in bud.values() if b["eps1"] is not None and b["exact"] is not None and b["eps1"] < 0.1)
rev = bud["010"]["eps1"] is not None and bud["010"]["exact"] is None and bud["101"]["eps1"] is None and bud["101"]["exact"] is not None
check("TD7 second-order budgets improve on first order; curvature reverses the first-order verdict for 010 and 101; the hidden channel is affine and exact",
      better and rev and abs(rc["d2F"]) < 1e-12 and abs(rc["eps1"] - rc["exact"]) < 1e-12,
      "; ".join(f"{k}: eps1 {v['eps1']}, eps2 {v['eps2']}, exact {v['exact']}" for k, v in bud.items() if v["eps1"] is not None or v["exact"] is not None))

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "tangent_decide.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
