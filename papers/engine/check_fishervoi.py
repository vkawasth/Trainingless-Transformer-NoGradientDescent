"""Checks for Fisher geometry against decision relevance (prop:fisher and rem:fisher of section 12.23, module pce/fishervoi.py).
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = exact symbolic computation (sympy);
NUMERICAL = floating-point, with tolerances stated.
  FV1  IDENTITY     phases are invisible to decisions: amplitudes sqrt(P) e^{i theta} with the same P and different phases are far apart in
                    Fubini-Study distance, while every expected utility, R and VOI is the same; on real amplitudes Fubini-Study = Fisher-Rao / 2
  FV2  IDENTITY     the null space of the observed Fisher information (one reading per observed context) is exactly the fibre's direction space,
                    while the complete-data Fisher information is positive definite; the determination criterion of thm:determination is
                    "every full-dimensional cell's value functional is Fisher-visible", and it agrees with the exact VOI envelope
  FV3  IDENTITY     decision value is Lipschitz in Fisher-Rao distance: |u.P - u.Q| <= 2 TV <= 2 sqrt(1 - BC^2) = 2 sin(d_FR / 2) for |u| <= 1
  FV4  NUMERICAL    the converse fails: two laws of one fibre at large Fisher-Rao distance with the same VOI; and two laws at Fisher-Rao distance
                    -> 0 on either side of a policy wall with different actions (values continuous, actions not)
  FV5  CERTIFICATE  the volume coefficient sqrt(det g) of the normal family depends on the coordinates: in (mu, sigma^k) it is
                    sqrt(2) / (|k| sigma^(k+1)), decreasing in sigma for k = 1, constant for k = -1, increasing for k = -2; the affine maps act by
                    isometries and transitively, so no isometry-invariant function of a point prefers small sigma
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import voigeom as V
from pce import poisonflip as P
from pce import fishervoi as F
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261108); M4 = np.array(G.M, float); C = G.CELLS; TAU = 0.45
T = np.array([1.0 if c[0] == c[1] else 0.0 for c in C]); zB0 = [c[2] for c in C]; Uaa = V.act_abstain_utilities(TAU)

# FV1
fs, same, rel = [], 0.0, 0.0
for s in range(50):
    p = rng.dirichlet(np.ones(16)); th1, th2 = rng.uniform(0, 2 * np.pi, (2, 16))
    z1, z2 = np.sqrt(p) * np.exp(1j * th1), np.sqrt(p) * np.exp(1j * th2); fs.append(F.fubini_study(z1, z2))
    p1, p2 = np.abs(z1) ** 2, np.abs(z2) ** 2
    same = max(same, abs(V.voi_at(p1, Uaa, zB0) - V.voi_at(p2, Uaa, zB0)), max(abs(u @ p1 - u @ p2) for u in rng.uniform(-1, 1, (20, 16))))
    q = rng.dirichlet(np.ones(16)); rel = max(rel, abs(F.fubini_study(np.sqrt(p), np.sqrt(q)) - F.fisher_rao(p, q) / 2))
check("FV1 IDENTITY phases change the Fubini-Study position but no expected utility or VOI; on real amplitudes FS = Fisher-Rao / 2",
      min(fs) > 0.1 and same < 1e-12 and rel < 1e-12, f"50 random laws with two random phase vectors: FS distance {min(fs):.3f}-{max(fs):.3f}, largest change of "
      f"VOI or of 20 random expected utilities {same:.1e}; |FS - FR/2| on real amplitudes {rel:.1e}")

# FV2
ok = True; dims = []
for s in range(30):
    x = rng.dirichlet(np.ones(16)); d = M4 @ x
    N = F.null_directions(x); D = V.fibre_directions(d); Ifull, _ = F.full_fisher(x)
    same_space = N.shape[0] == D.shape[0] and np.linalg.matrix_rank(np.vstack([N, D]), 1e-8) == D.shape[0]
    ok &= same_space and np.linalg.eigvalsh(Ifull).min() > 0; dims.append(N.shape[0])
agree = tot = det = 0
def additive(rg):
    u = np.zeros(16)
    for (i, j) in G.CTX:
        f = rg.normal(size=(2, 2)); u += np.array([f[c[i]][c[j]] for c in C])
    return u
for s in range(40):
    x = rng.dirichlet(np.ones(16)); d = M4 @ x; ut = s % 3
    U = [additive(rng) * 0.3 for _ in range(2)] if ut == 0 else [rng.normal(size=16) * 0.3 for _ in range(2)] if ut == 1 else Uaa
    z = [c[s % 4] for c in C]; N = F.null_directions(x)
    cells = V.full_dimensional_cells(d, U, z); visible = all(np.linalg.norm(N @ V.value_functional(U, z, p)) < 1e-8 for p in cells)
    lo, hi = V.voi_envelope(d, U, z); agree += visible == (hi - lo < 1e-8); tot += 1; det += visible
check("FV2 IDENTITY fibre directions = null space of the observed Fisher information; VOI determined iff its cell functionals are Fisher-visible",
      ok and agree == tot and 0 < det < tot, f"30 interior laws: null space dimension {sorted(set(dims))} equals the fibre's direction space, complete-data "
      f"Fisher positive definite; 40 decision problems: visibility and the exact envelope agree in {agree}/{tot} ({det} determined)")

# FV3
worst = 0.0; worst_tv = 0.0
for s in range(2000):
    p, q = rng.dirichlet(np.ones(16) * rng.uniform(0.2, 3), 2); u = rng.uniform(-1, 1, 16); b = F.bc(p, q)
    worst = max(worst, abs(u @ p - u @ q) - 2 * np.sin(F.fisher_rao(p, q) / 2)); worst_tv = max(worst_tv, F.tv(p, q) - np.sqrt(1 - b * b))
check("FV3 IDENTITY |u.P - u.Q| <= 2 TV <= 2 sin(d_FR / 2) for utilities in [-1, 1]",
      worst <= 1e-12 and worst_tv <= 1e-12, f"2000 random pairs: largest excess over the Fisher-Rao bound {worst:.2e}, of TV over sqrt(1 - BC^2) {worst_tv:.2e}")

# FV4
d = M4 @ np.random.default_rng(1).dirichlet(np.ones(16)); loc = np.array([1.0 if c[0] == c[2] else 0.0 for c in C]); Ul = [TAU - loc, loc - TAU]
Vt = V.fibre_vertices(d); best = max(((F.fisher_rao(a, b), i, j) for i, a in enumerate(Vt) for j, b in enumerate(Vt) if i < j))
a, b = Vt[best[1]], Vt[best[2]]; va, vb = V.voi_at(a, Ul, zB0), V.voi_at(b, Ul, zB0)
x0 = np.random.default_rng(2).dirichlet(np.ones(16) * 4); v = T - T.mean(); v -= v.mean(); x0 = x0 + (TAU - T @ x0) / (T @ v) * v
rows = []
for eps in (1e-2, 1e-3, 1e-4):
    xp, xm = x0 + eps * v, x0 - eps * v
    rows.append((eps, F.fisher_rao(xp, xm), int(T @ xp >= TAU), int(T @ xm >= TAU), abs(V.voi_at(xp, Uaa, zB0) - V.voi_at(xm, Uaa, zB0))))
ok = abs(va - vb) < 1e-10 and best[0] > 1.0 and all(r[2] != r[3] for r in rows) and rows[-1][1] < 1e-2 and rows[-1][4] < 1e-3 and x0.min() > 0
check("FV4 NUMERICAL far in Fisher-Rao with equal VOI; near in Fisher-Rao with different actions",
      ok, f"one fibre (dimension {G.fibre_dim(d)}), decision and signal inside (A0, B0): two vertices at Fisher-Rao distance {best[0]:.3f} (TV {F.tv(a, b):.3f}) "
      f"with VOI {va:.4f} and {vb:.4f}; across the wall P(A0 = A1) = {TAU}: (eps, d_FR, action +, action -, VOI gap) "
      + "; ".join(f"({e:g}, {f_:.1e}, {p_}, {m_}, {g_:.1e})" for e, f_, p_, m_, g_ in rows))

# FV5
import sympy as sp
s_ = sp.symbols("sigma", positive=True)
cs = {k: F.normal_complexity(k) for k in (1, -1, -2)}; cl = F.normal_complexity(0); iso = F.normal_isometry()
expect = {k: sp.sqrt(2) / (abs(k) * s_ ** (k + 1)) for k in (1, -1, -2)}
ok = all(sp.simplify(cs[k] - expect[k]) == 0 for k in cs) and sp.simplify(cl - sp.sqrt(2) / s_) == 0 and iso == sp.zeros(2, 2)
check("FV5 CERTIFICATE the normal family's volume coefficient depends on the coordinates; the affine maps are transitive isometries",
      ok, f"sqrt(det g) in (mu, sigma): {cs[1]}; (mu, log sigma): {cl}; (mu, 1/sigma): {cs[-1]}; (mu, 1/sigma^2): {cs[-2]}; "
      f"pullback under (mu, sigma) -> (a mu + b, a sigma) minus the metric: {iso.tolist()}")
print(f"{sum(results)}/{len(results)} passed")
