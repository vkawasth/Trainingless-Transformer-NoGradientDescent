"""check_barr.py -- checks of Lemma "D, eta and mu are monotone for the Barr lifting" in
"Shadows, Fibres and Secants of the Posterior Tower" (Section 6).

Barr lifting: Pi barD(R) Pi'  iff  some coupling of Pi and Pi' is supported in R. Membership is decided by a linear-programming
feasibility problem (HiGHS); every instance below is built so that the expected answer is known, and negative controls are included.

  R1  (a) barD(Delta) is equality; barD R ; barD S is contained in barD(R;S); barD of a preorder is a preorder (200 random instances;
      negative controls: unrelated point masses are rejected)
  R2  (b) D is monotone: (Df x Dg)(barD R) is contained in barD((f x g)R)  (200 random instances)
  R3  (c) delta_x barD(R) delta_y iff x R y  (all pairs, 50 random relations)
  R4  (d) mu is monotone: Pi barD(barD R) Pi' implies mu Pi barD(R) mu Pi'  (200 random instances); and the converse fails
      (explicit counterexample with R the diagonal)
Run: python check_barr.py
"""
import numpy as np, itertools
from scipy.optimize import linprog
rng = np.random.default_rng(20261013)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

def related(P, Q, R):
    """is there a coupling of P (on range(len(P))) and Q supported in the boolean matrix R?"""
    pairs = [(i, j) for i in range(len(P)) for j in range(len(Q)) if R[i, j]]
    if not pairs: return False
    A = np.zeros((len(P) + len(Q), len(pairs)))
    for k, (i, j) in enumerate(pairs): A[i, k] = 1; A[len(P) + j, k] = 1
    r = linprog(np.zeros(len(pairs)), A_eq=A, b_eq=np.concatenate([P, Q]), bounds=(0, None), method="highs")
    return r.status == 0
def rand_rel(n, m, p=0.4, reflexive=False):
    R = rng.random((n, m)) < p
    if reflexive: np.fill_diagonal(R, True)
    for i in range(n):
        if not R[i].any(): R[i, rng.integers(m)] = True
    return R
def rand_preorder(n):
    R = rand_rel(n, n, 0.3, reflexive=True)
    for k in range(n):                                   # transitive closure
        R = R | (R[:, [k]] & R[[k], :])
    return R
def coupling_in(R, P=None):
    """random coupling supported in R; if P is given, its first marginal is P (needs every row of R non-empty)."""
    n, m = R.shape
    if P is None: P = rng.dirichlet(np.ones(n))
    W = np.zeros((n, m))
    for i in range(n):
        idx = np.flatnonzero(R[i]); W[i, idx] = P[i] * rng.dirichlet(np.ones(len(idx)))
    return W

# R1
ok = True; neg = 0
for _ in range(200):
    n = rng.integers(3, 6); R = rand_preorder(n); S = rand_preorder(n)
    P = rng.dirichlet(np.ones(n)); Q = rng.dirichlet(np.ones(n))
    ok &= related(P, P, np.eye(n, dtype=bool)) and not related(P, Q, np.eye(n, dtype=bool))   # barD(Delta) = equality
    W1 = coupling_in(R); W2 = coupling_in(S, W1.sum(0))
    ok &= related(W1.sum(1), W2.sum(0), (R.astype(int) @ S.astype(int)) > 0)                 # composition
    ok &= related(P, P, R)                                                                   # reflexive
    W3 = coupling_in(R); W4 = coupling_in(R, W3.sum(0))
    ok &= related(W3.sum(1), W4.sum(0), R)                                                   # transitive (R preorder)
    x, y = np.argwhere(~R)[0] if (~R).any() else (None, None)
    if x is not None:
        neg += 1; ok &= not related(np.eye(n)[x], np.eye(n)[y], R)                           # negative control
check("R1 barD(Delta) = equality; barD R ; barD S within barD(R;S); barD(preorder) is a preorder", ok, f"{neg} negative controls rejected")

# R2
ok = True
for _ in range(200):
    n, m, n2, m2 = rng.integers(2, 6, 4); R = rand_rel(n, m)
    f, g = rng.integers(0, n2, n), rng.integers(0, m2, m)
    R2 = np.zeros((n2, m2), bool)
    for i, j in np.argwhere(R): R2[f[i], g[j]] = True
    W = coupling_in(R); P, Q = W.sum(1), W.sum(0)
    Pf = np.bincount(f, weights=P, minlength=n2); Qg = np.bincount(g, weights=Q, minlength=m2)
    ok &= related(P, Q, R) and related(Pf, Qg, R2)
check("R2 D is monotone along relation-preserving maps", ok)

# R3
ok = True
for _ in range(50):
    n, m = rng.integers(2, 6, 2); R = rand_rel(n, m)
    ok &= all(related(np.eye(n)[x], np.eye(m)[y], R) == bool(R[x, y]) for x in range(n) for y in range(m))
check("R3 delta_x barD(R) delta_y iff x R y", ok)

# R4
ok = True
for _ in range(200):
    n, m, k = rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5)
    R = rand_rel(n, m)
    pairs = [coupling_in(R) for _ in range(k)]                          # (p_i, p'_i) with p_i barD(R) p'_i
    w = rng.dirichlet(np.ones(k))
    muP = sum(wi * W.sum(1) for wi, W in zip(w, pairs)); muQ = sum(wi * W.sum(0) for wi, W in zip(w, pairs))
    ok &= related(muP, muQ, R)
# converse fails: X = Y = {0,1}, R = diagonal; Pi = 1/2 delta_{delta_0} + 1/2 delta_{delta_1}, Pi' = delta_{uniform}.
# mu Pi = mu Pi' = uniform, so mu Pi barD(Delta) mu Pi'; but barD(Delta) is equality on D X, and the only laws on laws related
# to Pi' under barD(equality) is Pi' itself, which differs from Pi.
laws = [np.array([1., 0.]), np.array([0., 1.]), np.array([.5, .5])]    # points of D X used by Pi and Pi'
Rlift = np.array([[related(p, q, np.eye(2, dtype=bool)) for q in laws] for p in laws])
Pi, Pi2 = np.array([.5, .5, 0.]), np.array([0., 0., 1.])
conv_fails = related(Pi @ np.array(laws), Pi2 @ np.array(laws), np.eye(2, dtype=bool)) and not related(Pi, Pi2, Rlift)
check("R4 mu is monotone for barD (200 instances); the converse fails (explicit counterexample)", ok and conv_fails)

print(f"\n{sum(results)}/{len(results)} checks passed")
