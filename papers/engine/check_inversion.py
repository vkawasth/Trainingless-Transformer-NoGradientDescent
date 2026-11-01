"""Checks for Bayesian inversion (lab Section "What does not matter", subsection "Inversion").
  IV1  the Bayesian inverse of (p, K) has the same coupling, swapped; inverting twice returns K on the support of p; inversion reverses
       composition, (L o K)dag = Kdag o Ldag (almost surely)
  IV2  inversion is partial: defined exactly where q(y) > 0, so its domain ends at a positivity wall
  IV3  the conditioning operator is integration along the inverse of the marginal map: C_X f = sum_x P(x) f(P | x) agrees with the
       relevance module, and C_X C_Y = C_XY (inversion composes)
  IV4  latent inversion: the posterior over types given a record is permuted, not changed, by relabelling; posterior-predictive decisions
       on observables are invariant
  IV5  the defect sandwich: 0 <= D_X(F) <= ante(F) - ante(Rect_X F), with ante(Rect) equal to the worst case of the updated policy on the
       hull (Epstein-Schneider), on hiding segments and on random non-rectangular sets; both defects vanish when F is rectangular
  IV6  the upper bound is attained in a positive fraction of cases, so the sandwich cannot be tightened to an equality or to zero
  IV7  inversion along an observed variable is affine on the fibre (its marginal q(y) is constant there): posterior of a mixture of two
       ends of a hiding segment = the same mixture of posteriors
  IV8  in general inversion is linear-fractional (posterior of a mixture = mixture of posteriors with weights proportional to q(y)), and its
       derivative maps feasible directions at a corner to feasible directions
  IV9  inversion creates no pooling flips: if the posteriors of two laws choose the same act, so does the posterior of every mixture
  IV10 at the domain wall the derivative of inversion grows exactly like 1 / q(y)
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import inversion as IV, relevance as RV, three as TH
rng = np.random.default_rng(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def stoch(n, m, zeros=False):
    K = rng.dirichlet(np.ones(m), size=n)
    if zeros: K[:, -1] = 0; K /= K.sum(axis=1, keepdims=True)
    return K

# IV1
err = 0.0
for _ in range(100):
    p = rng.dirichlet(np.ones(4)); K = stoch(4, 3); L = stoch(3, 5)
    q, Kd = IV.bayes_inverse(p, K); err = max(err, np.abs(IV.joint(q, Kd) - IV.joint(p, K).T).max())
    p2, Kdd = IV.bayes_inverse(q, Kd); err = max(err, np.abs(Kdd - K).max())
    r, LKd = IV.bayes_inverse(p, K @ L); _, Ld = IV.bayes_inverse(q, L); err = max(err, np.abs(LKd - Ld @ Kd).max())
check("IV1 inversion swaps the coupling, is involutive on the support, and reverses composition", err < 1e-12, f"{err:.1e}")

# IV2
p = rng.dirichlet(np.ones(4)); K = stoch(4, 3, zeros=True); q, Kd = IV.bayes_inverse(p, K)
ok2 = np.isnan(Kd[-1]).all() and not np.isnan(Kd[:-1]).any()
for eps in (1e-3, 1e-6, 0.0):
    K2 = stoch(4, 3); K2[:, -1] = eps; K2 /= K2.sum(axis=1, keepdims=True); q2, Kd2 = IV.bayes_inverse(p, K2); ok2 &= (np.isnan(Kd2[-1]).all()) == (eps == 0.0)
check("IV2 inversion is partial: defined exactly where q(y) > 0; the domain ends at the positivity wall q(y) = 0", ok2)

# IV3
L = [rng.normal(size=(2,) * 4) for _ in range(3)]; err3 = 0.0
for _ in range(100):
    P = rng.dirichlet(np.ones(16)).reshape((2,) * 4); R = lambda Q: RV.bayes_risk(Q, L)
    err3 = max(err3, abs(IV.cond_op(R, P, [0]) - RV.act_on(R, P, [0])))
    err3 = max(err3, abs(IV.cond_op(lambda Q: IV.cond_op(R, Q, [1]), P, [0]) - IV.cond_op(R, P, [0, 1])))
check("IV3 the conditioning operator is integration along the inverse of the marginal map, and C_X C_Y = C_XY", err3 < 1e-12, f"{err3:.1e}")

# IV4
ok4 = True
for _ in range(30):
    w = rng.dirichlet([2] * 3); a = rng.uniform(0.05, 0.95, (3, 4)); x = rng.integers(0, 2, 4); u = rng.normal(size=(2,) * 4)
    post = IV.posterior_types(w, a, x)
    for perm in itertools.permutations(range(3)):
        p2 = IV.posterior_types(w[list(perm)], a[list(perm)], x); ok4 &= np.allclose(p2, post[list(perm)])
    # posterior predictive of an observable act given X0 = x0
    P = TH.law(w, a); Pc = RV.condition(P, [0], (x[0],))[0]
    Pp = TH.law(w[[2, 0, 1]], a[[2, 0, 1]]); Ppc = RV.condition(Pp, [0], (x[0],))[0]; ok4 &= abs((Pc * u).sum() - (Ppc * u).sum()) < 1e-14
check("IV4 latent inversion: relabelling permutes the posterior over types; posterior-predictive observable decisions are invariant", ok4)

# IV5, IV6
chi = np.array([[[(2 * a - 1) * (2 * b - 1) * (2 * c - 1) for c in (0, 1)] for b in (0, 1)] for a in (0, 1)], float)
def segment():
    while True:
        P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); smin = -P[chi > 0].min(); smax = P[chi < 0].min()
        if smax - smin > 0.05: return [P + smin * chi, P + smax * chi]
def rect_set():
    px = [rng.dirichlet([2, 2]) for _ in range(2)]; q = {x: [rng.dirichlet(np.ones(4)).reshape(2, 2) for _ in range(2)] for x in (0, 1)}; out = []
    for pp in px:
        for q0 in q[0]:
            for q1 in q[1]:
                P = np.zeros((2, 2, 2)); P[0] = pp[0] * q0; P[1] = pp[1] * q1; out.append(P)
    return out
viol = pos = tight = 0; es = 0.0; rect_zero = 0.0
for t in range(300):
    kind = t % 3
    F = segment() if kind == 0 else [rng.dirichlet(np.ones(8)).reshape(2, 2, 2) for _ in range(3)] if kind == 1 else rect_set()
    U = [rng.normal(size=(2, 2, 2)) for _ in range(3)]
    ante, upd = RV.ante_and_updated(F, [0], U); H = IV.rect_hull(F, [0]); anteH, updH = RV.ante_and_updated(H, [0], U)
    D, G = ante - upd, ante - anteH; es = max(es, abs(anteH - updH))
    viol += D < -1e-9 or D > G + 1e-9
    if kind == 2: rect_zero = max(rect_zero, abs(D), abs(G))
    if D > 1e-9: pos += 1; tight += abs(D - G) < 1e-9
OUT["IV5_6"] = dict(violations=int(viol), positive=pos, tight=tight, es_on_hull=es, rect_zero=rect_zero)
check("IV5 the defect sandwich 0 <= D_X(F) <= ante(F) - ante(Rect_X F); Epstein-Schneider exact on the hull; both zero on rectangular sets",
      viol == 0 and es < 1e-9 and rect_zero < 1e-9, f"{viol} violations of 300; hull gap {es:.1e}; rectangular {rect_zero:.1e}")
check("IV6 the upper bound is attained in a positive fraction of cases with D > 0, and is strict in others", 0 < tight < pos, f"tight in {tight} of {pos}")
# IV7-IV10
post = lambda P, ax, y: RV.condition(P, ax, y)[0]
e7 = 0.0
for _ in range(100):
    A, B = segment()
    for lam in rng.uniform(0, 1, 5):
        M = lam * A + (1 - lam) * B
        for y in (0, 1): e7 = max(e7, np.abs(post(M, [0], (y,)) - (lam * post(A, [0], (y,)) + (1 - lam) * post(B, [0], (y,)))).max())
check("IV7 inversion along an observed variable is affine on the fibre (q(y) constant along the hiding segment)", e7 < 1e-12, f"{e7:.1e}")
e8 = 0.0; feas = True
for _ in range(200):
    A = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); B = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); lam = rng.uniform()
    M = lam * A + (1 - lam) * B; qa, qb = A[0].sum(), B[0].sum(); mu = lam * qa / (lam * qa + (1 - lam) * qb)
    e8 = max(e8, np.abs(post(M, [0], (0,)) - (mu * post(A, [0], (0,)) + (1 - mu) * post(B, [0], (0,)))).max())
    P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); P[0, 1, 0] = 0; P[0, 0, 1] = 0; P /= P.sum()
    u = rng.normal(size=(2, 2, 2)); u[P == 0] = np.abs(u[P == 0]) + 0.01; u.flat[int(np.argmax(P))] -= u.sum()
    h = 1e-7; d = (post(P + h * u, [0], (0,)) - post(P, [0], (0,))) / h; Z = post(P, [0], (0,)) == 0
    feas &= bool(np.all(d[Z] >= -1e-6)) and abs(d.sum()) < 1e-5
check("IV8 inversion is linear-fractional, and its derivative maps feasible directions at a corner to feasible directions", e8 < 1e-12 and feas, f"{e8:.1e}")
flips = tests = 0
for _ in range(2000):
    A = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); B = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); U = [rng.normal(size=(2, 2, 2)) for _ in range(3)]
    act = lambda Q: int(np.argmax([(Q * u).sum() for u in U])); pa, pb = post(A, [0], (1,)), post(B, [0], (1,))
    if act(pa) != act(pb): continue
    for lam in np.linspace(0.05, 0.95, 7): tests += 1; flips += act(post(lam * A + (1 - lam) * B, [0], (1,))) != act(pa)
check("IV9 inversion creates no pooling flips (decision chambers are convex and inversion maps segments to segments)", flips == 0 and tests > 1000, f"{flips} flips in {tests}")
P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); u = rng.normal(size=(2, 2, 2)); u -= u.mean(); rs = []
for q in (1e-1, 1e-2, 1e-3, 1e-4):
    Q = P.copy(); Q[0] *= q / Q[0].sum(); Q[1] *= (1 - q) / Q[1].sum(); h = 1e-9
    rs.append(np.abs((post(Q + h * u, [0], (0,)) - post(Q, [0], (0,))) / h).max() * q)
check("IV10 at the domain wall the derivative of inversion grows like 1 / q(y)", np.ptp(rs) < 1e-3 * max(rs), f"|D| q(y) = {np.round(rs, 4).tolist()}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "inversion.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
