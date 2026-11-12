"""check_infogeom.py -- checks for Section 14.5 (a unified coefficient system) of
"Shadows, Fibres and Secants of the Posterior Tower".

  I1  tangent presheaf: for P_t = P (1 + t v) with E_P v = 0, the derivative at t = 0 of log P_t(sigma-cell) is E[v | sigma]; so
      marginalisation differentiates to conditional expectation, and the Fisher metric is the L^2(P) inner product (finite differences,
      300 random cases)
  I2  Laplacian: for the gluing complex delta0 f = (E_A f, E_B f), the operator delta0* delta0 = E_A + E_B has spectrum
      {0 on ker delta0} u {1 - cos t_k, 1 + cos t_k} u {1} u {2 on L^2(A ^ B)}; its smallest non-zero eigenvalue is 1 - c, and the
      non-trivial eigenvalues of E_A E_B E_A are cos^2 t_k (300 random pairs of partitions)
  I3  KL Pythagoras: with P* = P_AC P_CB / P_C, KL(P||Q) = KL(P||P*) + KL(P*||Q) for every Q with A _||_ B | C; P* is the minimiser,
      and KL(P||P*) = I(A;B|C) (500 random laws)
  I4  local expansion: I(A;B|C) / (1/2 sum_c P(c) sum_k sigma_k(c)^2) -> 1 as P approaches the Beck-Chevalley model; far from it the
      ratio is reported, not asserted
  I5  positivity is the only obstruction: every compatible family of pair tables on the triangle of binary variables has a signed global
      section (rank argument), while a positive one exists only for some (linear programming); 400 random compatible families
Run: python check_infogeom.py
"""
import itertools, math
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(20261013)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

def Emat(lab, p):
    """conditional expectation onto the partition lab, as a matrix acting on functions (rows: points)."""
    n = len(p); M = np.zeros((n, n))
    for k in set(lab):
        idx = [i for i in range(n) if lab[i] == k]; w = p[idx] / p[idx].sum()
        for i in idx: M[i, idx] = w
    return M
def sym(M, p):
    """the same operator in an orthonormal frame of L^2(p)."""
    D = np.diag(np.sqrt(p)); return D @ M @ np.diag(1 / np.sqrt(p))

# I1
ok = True
for _ in range(300):
    n = 10; p = rng.dirichlet(np.ones(n)); lab = list(rng.integers(0, 3, n))
    v = rng.normal(size=n); v -= p @ v
    t = 1e-6; pt = p * (1 + t * v)
    E = Emat(lab, p) @ v
    for i in range(n):
        cell = [j for j in range(n) if lab[j] == lab[i]]
        d = (math.log(pt[cell].sum()) - math.log(p[cell].sum())) / t
        ok &= abs(d - E[i]) < 1e-5
check("I1 marginalisation differentiates to conditional expectation (tangent presheaf = L^2 presheaf with the Fisher metric)", ok)

# I2
ok = True; worst = 0.0
for _ in range(300):
    n = 12; p = rng.dirichlet(np.ones(n)); A = list(rng.integers(0, 4, n)); B = list(rng.integers(0, 4, n))
    EA, EB = sym(Emat(A, p), p), sym(Emat(B, p), p)
    EA, EB = (EA + EA.T) / 2, (EB + EB.T) / 2
    lap = np.sort(np.linalg.eigvalsh(EA + EB))
    s = np.linalg.svd(EA @ EB, compute_uv=False)
    cos = s[(s > 1e-9) & (s < 1 - 1e-9)]
    c = cos.max() if len(cos) else 0.0
    nz = lap[lap > 1e-9]
    if len(cos):
        worst = max(worst, abs(nz.min() - (1 - c)))
        for x in cos:                                   # both 1 - cos and 1 + cos occur
            ok &= np.abs(lap - (1 - x)).min() < 1e-9 and np.abs(lap - (1 + x)).min() < 1e-9
    eig3 = np.sort(np.linalg.eigvalsh(EA @ EB @ EA))
    ok &= all(np.abs(eig3 - x * x).min() < 1e-9 for x in cos)
    allowed = [0.0, 1.0, 2.0] + [1 - x for x in cos] + [1 + x for x in cos]
    ok &= all(min(abs(l - a) for a in allowed) < 1e-9 for l in lap)
check("I2 Laplacian E_A + E_B: spectrum {0, 1 -+ cos t_k, 1, 2}, smallest non-zero eigenvalue 1 - c; E_A E_B E_A has eigenvalues cos^2 t_k",
      ok and worst < 1e-9, f"max |gap - (1 - c)| = {worst:.1e}")

# I3
def kl(p, q): return float((p * np.log(p / q)).sum())
ok = True
for _ in range(500):
    P = rng.dirichlet(np.ones(27)).reshape(3, 3, 3)                      # axes (A, C, B)
    Pac, Pcb, Pc = P.sum(2), P.sum(0), P.sum((0, 2))
    Ps = np.einsum("ac,cb->acb", Pac, Pcb) / Pc[None, :, None]
    Q = np.einsum("c,ca,cb->acb", rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3), 3), rng.dirichlet(np.ones(3), 3))
    I = sum(Pc[c] * kl(P[:, c, :] / Pc[c], np.outer(Pac[:, c], Pcb[c]) / Pc[c] ** 2) for c in range(3))
    ok &= abs(kl(P, Q) - kl(P, Ps) - kl(Ps, Q)) < 1e-12 and kl(P, Q) >= kl(P, Ps) - 1e-12 and abs(kl(P, Ps) - I) < 1e-12
check("I3 KL(P||Q) = KL(P||P*) + KL(P*||Q) for all Q with A _||_ B | C; P* minimises; KL(P||P*) = I(A;B|C)", ok)

# I4
def ratio(P):
    Pac, Pcb, Pc = P.sum(2), P.sum(0), P.sum((0, 2))
    Ps = np.einsum("ac,cb->acb", Pac, Pcb) / Pc[None, :, None]
    s2 = 0.0
    for c in range(P.shape[1]):
        T = P[:, c, :] / P[:, c, :].sum(); K = T / np.sqrt(np.outer(T.sum(1), T.sum(0)))
        s2 += Pc[c] * (np.linalg.svd(K, compute_uv=False)[1:] ** 2).sum()
    return kl(P, Ps) / (s2 / 2)
local = []
for t in (0.3, 0.1, 0.03, 0.01):
    rs = []
    for _ in range(50):
        base = np.einsum("c,ca,cb->acb", rng.dirichlet(np.ones(3) * 5), rng.dirichlet(np.ones(3) * 5, 3), rng.dirichlet(np.ones(3) * 5, 3))
        D = rng.normal(size=(3, 3, 3)); P = base * np.exp(t * D); P /= P.sum(); rs.append(ratio(P))
    local.append((t, float(np.max(np.abs(np.array(rs) - 1)))))
far = [ratio(rng.dirichlet(np.ones(27)).reshape(3, 3, 3)) for _ in range(300)]
check("I4 I(A;B|C) / (sum P(c) sigma_k(c)^2 / 2) -> 1 near the Beck-Chevalley model", local[-1][1] < 0.02 and local[-1][1] < local[0][1],
      "max |ratio - 1| at t = " + ", ".join(f"{t}: {e:.3f}" for t, e in local) + f"; far from the model the ratio ranges over [{min(far):.2f}, {max(far):.2f}]")

# I5
ATOMS = list(itertools.product((0, 1), repeat=3)); PAIRS = [(0, 1), (1, 2), (0, 2)]
rows = []
for i, j in PAIRS:
    for u, v in itertools.product((0, 1), repeat=2):
        rows.append([1.0 if (x[i], x[j]) == (u, v) else 0.0 for x in ATOMS])
M = np.array(rows + [[1.0] * 8])
signed_ok = True; positive = 0
for _ in range(400):
    m = rng.uniform(0.1, 0.9, 3)                                             # P(X_i = 1)
    b = []
    for i, j in PAIRS:
        lo, hi = max(0, m[i] + m[j] - 1), min(m[i], m[j]); p11 = rng.uniform(lo, hi)
        b += [1 - m[i] - m[j] + p11, m[j] - p11, m[i] - p11, p11]
    b = np.array(b + [1.0])
    x, *_ = np.linalg.lstsq(M, b, rcond=None); signed_ok &= np.abs(M @ x - b).max() < 1e-10
    r = linprog(np.zeros(8), A_eq=M, b_eq=b, bounds=(0, None), method="highs"); positive += r.status == 0
check("I5 every compatible family on the triangle has a signed global section; only some have a positive one",
      signed_ok and 0 < positive < 400, f"positive global law for {positive} of 400 compatible families")

print(f"\n{sum(results)}/{len(results)} checks passed")
