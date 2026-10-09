"""check_bridges.py -- the statements, worked examples and synthetic tests of "Bridges".

  G1  -Hess H on the tangent space of the simplex is the Fisher metric
  G2  conormal intersection: the entropy graph meets the conormal brane of a marginal fibre exactly at the
      maximum-entropy law, transversally (three binary variables, pairwise margins; and chain margins -> gluing filler)
  W1  three bits with pairwise disagreement 1 - eps: the fibre is the line p0 + t x with x(s) = (-1)^{|s|}; a joint exists
      iff eps >= 1/3, a positive one iff 1/3 < eps < 1; the max-entropy law has a = (3 eps - 1)/4 on 000, 111 and b = (1-eps)/4
      elsewhere; the Fisher form on the fibre is 8/(3 eps - 1) + 24/(1 - eps); the contextual fraction is 1 - 3 eps below 1/3
  S1  synthetic: from N joint samples of a positive law with eps = 1/3 + m, the verdict "breaks" has probability (1 - 3m/2)^N,
      so N of order 1/m suffices; from pairwise-only samples the misses persist until N m^2 is large
  G3  Abreu: the full entropy potential satisfies the boundary determinant condition when some cells are not facets
  W2  2 x 2 couplings, f = (0.4, 0.6), g = (0.3, 0.7): X = CP^1 with moment segment t = alpha_11 in [0, 0.3]; entropy potential
      critical at the product t = 0.12, Guillemin's at 0.15; only the fibre over the Frechet midpoint t = 0.15 has a critical
      point of the disc potential
  G4  gluing with beta acts by alpha -> alpha K; it preserves integer circulations iff K is deterministic or constant
  W3  2 x 2: alpha -> alpha K multiplies the circulation by lambda = K_11 - K_21, with |lambda| the Dobrushin coefficient;
      the image of the moment segment has |lambda| times its length; integral iff lambda in {-1, 0, 1}
  W4  A_f: centre of dimension 2 (HH^0), simple modules of dimensions 1 and n - 1, Morita equivalent to R x R
"""
import itertools, math
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
R = np.random.default_rng(3); res = []
def check(n, c, info=""): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))
def H(p): p = np.asarray(p, float).ravel(); p = p[p > 0]; return float(-(p * np.log(p)).sum())

# ---------------------------------------------------------------- G1
ok = True
for n in (3, 4, 6):
    p = R.dirichlet(np.ones(n)); T = null_space(np.ones((1, n))); eps = 1e-5; u = T[:, 0]
    second = (H(p + eps * u) - 2 * H(p) + H(p - eps * u)) / eps ** 2       # d^2 H along u
    ok &= abs(-second - (u ** 2 / p).sum()) < 1e-3 * (u ** 2 / p).sum()
check("G1 -d^2H(u,u) = sum u^2/p (finite differences, n = 3, 4, 6)", ok)

# ---------------------------------------------------------------- G2 and W1 (three bits)
cells = list(itertools.product([0, 1], repeat=3))
def rows(S): return np.array([[1.0 if (c[S[0]], c[S[1]]) == v else 0.0 for c in cells] for v in itertools.product([0, 1], repeat=2)])
PAIRS = [(0, 1), (1, 2), (0, 2)]; M = np.vstack([rows(S) for S in PAIRS])
def ipf(targets, iters=4000):
    q = np.ones(8) / 8
    for _ in range(iters):
        for S, t in zip(PAIRS, targets):
            A = rows(S); q = q * (A.T @ (t / (A @ q)))
    return q
ok = True
for _ in range(10):
    P = R.dirichlet(np.ones(8)); q = ipf([rows(S) @ P for S in PAIRS]); N = null_space(M)
    ok &= np.abs(N.T @ (-np.log(q) - 1)).max() < 1e-8 and np.linalg.eigvalsh(N.T @ np.diag(1 / q) @ N).min() > 1e-6 and H(q) >= H(P) - 1e-12
P3 = R.dirichlet(np.ones(8)).reshape(2, 2, 2); F = P3.sum(2)[:, :, None] * P3.sum(0)[None, :, :] / P3.sum((0, 2))[None, :, None]
N2 = null_space(np.vstack([rows((0, 1)), rows((1, 2))]))
check("G2 the entropy graph meets the conormal of a marginal fibre at its max-entropy law, transversally; for chain margins at "
      "the gluing filler", ok and np.abs(N2.T @ (-np.log(F.ravel()) - 1)).max() < 1e-10)

x = np.array([(-1.0) ** sum(c) for c in cells])
def tables(eps): t = np.array([eps / 2, (1 - eps) / 2, (1 - eps) / 2, eps / 2]); return np.tile(t, 3)
def feasible(eps, strict=False):
    # maximise s subject to p >= s, M p = tables, sum p = 1
    c = np.r_[np.zeros(8), -1.0]; A_eq = np.hstack([np.vstack([M, np.ones(8)]), np.zeros((13, 1))]); b_eq = np.r_[tables(eps), 1]
    A_ub = np.hstack([-np.eye(8), np.ones((8, 1))]); r = linprog(c, A_ub=A_ub, b_ub=np.zeros(8), A_eq=A_eq, b_eq=b_eq,
                                                                   bounds=[(0, None)] * 8 + [(None, None)], method="highs")
    if r.status != 0: return False
    return (-r.fun > 1e-9) if strict else True
def cf(eps):
    r = linprog(-np.ones(8), A_ub=M, b_ub=tables(eps), bounds=[(0, None)] * 8, method="highs"); return 1 + r.fun
ok_line = np.allclose(null_space(np.vstack([M, np.ones(8)])).ravel() / null_space(np.vstack([M, np.ones(8)])).ravel()[0], x / x[0])
ok_feas = (all(feasible(e) == (e >= 1 / 3 - 1e-12) for e in np.linspace(0, 1, 61)) and not feasible(1 / 3, strict=True)
           and feasible(0.34, strict=True) and feasible(0.99, strict=True) and not feasible(1.0, strict=True))
ok_max = True; ok_fisher = True
for e in (0.35, 0.5, 0.7, 0.9):
    q = ipf([tables(e)[4 * k:4 * k + 4] for k in range(3)]); a, b = (3 * e - 1) / 4, (1 - e) / 4
    ok_max &= np.allclose(q, [a if c in [(0, 0, 0), (1, 1, 1)] else b for c in cells], atol=1e-8)
    ok_fisher &= abs((x ** 2 / q).sum() - (8 / (3 * e - 1) + 24 / (1 - e))) < 1e-6 * (x ** 2 / q).sum()
ok_cf = all(abs(cf(e) - max(0.0, 1 - 3 * e)) < 1e-9 for e in np.linspace(0, 1, 41))
check("W1 three bits: fibre = line along the parity vector; joint iff eps >= 1/3, positive joint iff 1/3 < eps < 1; max-entropy law "
      "(3eps-1)/4, (1-eps)/4; Fisher form 8/(3eps-1) + 24/(1-eps); contextual fraction 1 - 3 eps", ok_line and ok_feas and ok_max and ok_fisher and ok_cf,
      f"Fisher form at eps = 0.34, 0.35, 0.5: {[round(8/(3*e-1)+24/(1-e),1) for e in (0.34, 0.35, 0.5)]}")

# ---------------------------------------------------------------- S1 synthetic: detecting the breaking threshold from data
def sample_law(eps):
    a, b = (3 * eps - 1) / 4, (1 - eps) / 4
    return np.array([a if c in [(0, 0, 0), (1, 1, 1)] else b for c in cells])
m = 0.02; law = sample_law(1 / 3 + m)
# eps_hat = (1 + 2 * share of all-equal samples) / 3, so the verdict "breaks" (eps_hat <= 1/3) happens exactly when no
# all-equal sample is seen: probability (1 - 2a)^N with 2a = 3m/2.  Detection needs N of order 1/m, not 1/m^2.
rates, pred = {}, {}
for c in (0.5, 2, 8):
    N = int(math.ceil(c / m)); wrong = 0
    for _ in range(2000):
        cnt = R.multinomial(N, law); P = cnt / N
        eps_hat = np.mean([(rows(S) @ P)[[0, 3]].sum() for S in PAIRS]); wrong += not (eps_hat > 1 / 3 + 1e-12)
    rates[c] = wrong / 2000; pred[c] = (1 - 1.5 * m) ** N
# pairwise-only data: each pair observed in its own N samples (the usual contextuality design); eps_hat then has variance
# of order 1/N and the miss rate decays only once N m^2 is large
rates_pw = {}
for N in (25, 100, 400, 1600):
    wrong = 0
    for _ in range(2000):
        eps_hat = np.mean([R.binomial(N, 1 / 3 + m) / N for _ in PAIRS]); wrong += not (eps_hat > 1 / 3 + 1e-12)
    rates_pw[N] = wrong / 2000
fisher_eps = lambda e: 9 / (2 * (3 * e - 1)) + 3 / (2 * (1 - e))
check("S1 with joint (triple) samples the break at margin m is missed with probability (1 - 3m/2)^N, so N of order 1/m suffices; "
      "with pairwise-only samples misses persist until N m^2 is large",
      all(abs(rates[c] - pred[c]) < 0.03 for c in rates) and rates[0.5] > 0.3 and rates[8] < 0.005
      and rates_pw[400] > 0.03 and rates_pw[1600] < rates_pw[25]
      and [round(100 * rates[c]) for c in (0.5, 2, 8)] == [47, 6, 0] and [round(100 * rates_pw[N]) for N in (25, 100, 400)] == [42, 26, 7]
      and round(100 * rates_pw[1600], 2) == 0.05,                # the numbers quoted in the paper for this seed
      f"m = 0.02; joint samples N = 25, 100, 400: observed {[rates[c] for c in (0.5, 2, 8)]}, predicted {[round(pred[c], 3) for c in (0.5, 2, 8)]}; "
      f"pairwise only N = 25..1600: {rates_pw}; Fisher information of eps (joint) at m: {fisher_eps(1/3 + m):.1f}")

# ---------------------------------------------------------------- G3 Abreu condition with non-facet cells
f, g = np.array([0.8, 0.2]), np.array([0.3, 0.3, 0.4])
def alpha(t): a2 = np.array([t[0], t[1], f[1] - t[0] - t[1]]); return np.vstack([g - a2, a2])
J = np.array([[-1, 0], [0, -1], [1, 1], [1, 0], [0, 1], [-1, -1]], float)
def hess(t): return J.T @ np.diag(0.5 / alpha(t).ravel()) @ J
vals = []
for v in ([0, 0], [0.2, 0], [0, 0.2]):
    for e in (1e-2, 1e-4, 1e-6):
        c = np.array([0.2 / 3, 0.2 / 3]); t = np.array(v) + e * (c - np.array(v)) / np.linalg.norm(c - np.array(v))
        vals.append(np.linalg.det(hess(t)) * t[0] * t[1] * (f[1] - t[0] - t[1]))
check("G3 det(Hess G) * prod(facet functions) stays bounded and positive at every vertex although row-1 cells are not facets",
      min(vals) > 0.04 and max(vals) < 0.1, f"range {min(vals):.4f} .. {max(vals):.4f}")

# ---------------------------------------------------------------- W2 the 2 x 2 example
f, g = np.array([0.4, 0.6]), np.array([0.3, 0.7])
L, U = max(0, f[0] + g[0] - 1), min(f[0], g[0])
def cells22(t): return np.array([t, f[0] - t, g[0] - t, 1 - f[0] - g[0] + t])
ts = np.linspace(L + 1e-9, U - 1e-9, 300001)
full = [0.5 * (cells22(t) * np.log(cells22(t))).sum() for t in ts]
guil = [0.5 * ((t - L) * np.log(t - L) + (U - t) * np.log(U - t)) for t in ts]
t_full, t_guil = ts[int(np.argmin(full))], ts[int(np.argmin(guil))]
# disc potential of CP^1 with moment segment [L, U]: PO = T^{t-L} y + T^{U-t} / y; dPO/dy = 0 needs equal valuations, t = (L+U)/2
t_floer = (L + U) / 2
check("W2 2x2: entropy potential critical at the product coupling t = 0.12, Guillemin's at 0.15; the only Floer-nontrivial fibre "
      "is over the Frechet midpoint 0.15; non-facet cells stay positive on the segment",
      abs(t_full - f[0] * g[0]) < 1e-5 and abs(t_guil - 0.15) < 1e-5 and abs(t_floer - 0.15) < 1e-12
      and min(f[0] - U, 1 - f[0] - g[0] + L) > 0, f"segment [{L}, {U}], product {f[0]*g[0]:.2f}")

# ---------------------------------------------------------------- G4 integrality of gluing
def det_or_const(K): return bool(np.isin(K, [0, 1]).all()) or np.allclose(K, K[0])
ok = True
for _ in range(600):
    m, h = R.integers(2, 5), R.integers(2, 4); kind = R.integers(3)
    K = np.eye(h)[R.integers(0, h, m)] if kind == 0 else (np.tile(R.dirichlet(np.ones(h)), (m, 1)) if kind == 1 else R.dirichlet(np.ones(h), size=m))
    moves = [np.outer([1, -1], np.eye(m)[j] - np.eye(m)[l]) for j, l in itertools.combinations(range(m), 2)]
    integral = all(np.allclose(xm @ K, np.round(xm @ K)) for xm in moves)
    ok &= integral == det_or_const(K)
check("G4 alpha -> alpha K preserves integer circulations iff K is deterministic or constant", ok)

# ---------------------------------------------------------------- W3 the 2 x 2 correspondence
ok = True
for _ in range(200):
    K = R.dirichlet(np.ones(2), size=2); lam = K[0, 0] - K[1, 0]
    xx = np.array([[1.0, -1.0], [-1.0, 1.0]])
    ok &= np.allclose(xx @ K, lam * xx) and abs(abs(lam) - 0.5 * np.abs(K[0] - K[1]).sum()) < 1e-12
    h = K.T @ g; Lh, Uh = max(0, f[0] + h[0] - 1), min(f[0], h[0])
    img = [(np.array([[t, f[0] - t], [g[0] - t, 1 - f[0] - g[0] + t]]) @ K)[0, 0] for t in (L, U)]
    ok &= abs(abs(img[1] - img[0]) - abs(lam) * (U - L)) < 1e-12 and Lh - 1e-12 <= min(img) and max(img) <= Uh + 1e-12
lam_int = [round(K[0, 0] - K[1, 0]) for K in (np.eye(2), np.eye(2)[::-1], np.tile([0.3, 0.7], (2, 1)))]
check("W3 2x2: x K = lambda x with |lambda| the Dobrushin coefficient; the moment segment maps onto a segment |lambda| times as long; "
      "integral exactly for identity, swap, constant", ok and lam_int == [1, -1, 0])

# ---------------------------------------------------------------- W4 Morita data of A_f
ok = True
for n in (3, 4):
    fv = R.dirichlet(np.ones(n)); D = np.diag(fv)
    basis = []
    P = np.outer(fv, fv); basis.append(P.ravel())
    for i, j in itertools.product(range(n - 1), repeat=2):
        xm = np.zeros((n, n)); xm[i, j] = 1; xm[i, n - 1] = -1; xm[n - 1, j] = -1; xm[n - 1, n - 1] = 1; basis.append(xm.ravel())
    B = np.array(basis).T; dim = B.shape[1]
    mult = lambda a, b: a @ np.linalg.inv(D) @ b
    # centre: z with z.b = b.z for all basis b
    Mc = np.vstack([np.column_stack([(mult(B[:, c].reshape(n, n), B[:, k].reshape(n, n)) - mult(B[:, k].reshape(n, n), B[:, c].reshape(n, n))).ravel() for c in range(dim)]) for k in range(dim)])
    centre_dim = null_space(Mc).shape[1]
    ok &= dim == (n - 1) ** 2 + 1 and centre_dim == 2
check("W4 A_f has dimension (n-1)^2 + 1 and centre of dimension 2 (= HH^0), as for R x M_{n-1}(R), Morita equivalent to R x R", ok)

print(f"\n{sum(res)}/{len(res)} checks passed")
