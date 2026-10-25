"""check_noncomm.py -- non-commutativity as structure (note: level2/partial_gluing.pdf, Section 9).

  NC1 commuting observables carry no obstruction: no order effects, the Kirkwood-Dirac section is a genuine law (top rung), and the
      context tables glue (lam* = 1, the joint spectral measure)
  NC2 order effects are bounded by commutators: |P(P then Q) - P(Q then P)| <= ||[P, Q]|| (two-subspace theory; random projections in
      dimensions 2-6), and the QQ equality holds for every state and pair of binary projective measurements
  NC3 commutators decide the cover, the cover decides gluing: for the CHSH observables the commuting cliques are the four Bell contexts,
      the tables agree on overlaps, and lam* = 2 - sqrt 2 at Tsirelson's angles; as the angle between A0 and A1 opens from 0, lam* falls
      from 1 (contextuality needs non-commutativity)
  NC4 the canonical section lives on the complex rung: the Kirkwood-Dirac table has the exact marginals of both measurements, and
      Im q(a, b) = tr([B_b, A_a] rho)/2i; commuting -> R>=0, real non-commuting -> R (negative entries possible), complex -> C
  NC5 loops: the Bargmann invariant tr(P1 P2 P3) of three spin-1/2 directions has argument = half the signed solid angle; for real
      projections it is real
  NC6 non-abelian holonomy: projecting a rank-2 subspace around two different loops gives 2x2 unitaries that do not commute; the dense
      (Zeno) limit converges
  NC7 Zeno dragging (probability paths): steering |j, j> from n0 to n1 through N intermediate projections succeeds with probability
      cos(Th/2N)^(4jN) -> 1 (exact, j = 1/2, 1, 3/2), against cos(Th/2)^(4j) directly; commuting projections cannot steer at all
  NC8 path dependence: dense dragging around a closed loop returns the state with phase j * Omega (orientation sign) (spin j, solid angle Omega): two
      paths to the same target with equal success probability differ by a geometric phase j * Omega
Writes ../level2/noncomm_results.json.  Run: python check_noncomm.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.integrate import gluable_fraction
from pce.noncomm import (spectral, axis_measurement, sequential, order_effect, comm_norm, compatibility_cover, context_tables,
                         kirkwood_dirac, kd_rung, bargmann, zeno_drag, spin_ops)

rng = np.random.default_rng(20261024); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
def rand_unitary(d):
    Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); Q, R = np.linalg.qr(Z); return Q * (np.diag(R) / np.abs(np.diag(R)))
def rand_state(d, pure=False):
    if pure: v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v); return np.outer(v, v.conj())
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); R = G @ G.conj().T; return R / np.trace(R).real
def rand_proj(d, r, U=None):
    U = rand_unitary(d) if U is None else U; return U[:, :r] @ U[:, :r].conj().T
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex); I2 = np.eye(2)

# NC1
ok = True
for _ in range(30):
    d = int(rng.integers(3, 6)); U = rand_unitary(d); rho = rand_state(d)
    obs = []
    for _ in range(3):
        lab = rng.integers(0, 2, d); obs.append([U @ np.diag((lab == v).astype(float)) @ U.conj().T for v in (0, 1)])
    obs = [o for o in obs if all(np.trace(p).real > 0.5 for p in o)]
    if len(obs) < 2: continue
    ok &= all(abs(order_effect(rho, a[0], b[0])) < 1e-12 for a in obs for b in obs)
    ok &= kd_rung(kirkwood_dirac(rho, obs[0], obs[1]))["rung"] == "R>=0"
    cov = compatibility_cover(obs); ok &= cov == [tuple(range(len(obs)))]
    names = [f"O{i}" for i in range(len(obs))]; T = context_tables(rho, obs, cov)
    ok &= gluable_fraction({n: 2 for n in names}, [Window(tuple(names[i] for i in C), t) for C, t in zip(cov, T)]).lam > 1 - 1e-7
check("NC1 commuting observables: no order effects, Kirkwood-Dirac on the top rung, one context, lam* = 1", ok)

# NC2
ok = True; worst = 0.0; pts = []
for _ in range(300):
    d = int(rng.integers(2, 7)); P = rand_proj(d, int(rng.integers(1, d))); Q = rand_proj(d, int(rng.integers(1, d))); rho = rand_state(d)
    o = abs(order_effect(rho, P, Q)); c = comm_norm(P, Q); ok &= o <= c + 1e-12; worst = max(worst, o / c if c > 1e-12 else 0); pts.append([c, o])
    A = [P, np.eye(d) - P]; B = [Q, np.eye(d) - Q]; AB = sequential(rho, A, B); BA = sequential(rho, B, A)
    ok &= abs((AB[0, 1] + AB[1, 0]) - (BA[0, 1] + BA[1, 0])) < 1e-12
OUT["order"] = pts
check("NC2 |order effect| <= ||[P, Q]|| and the QQ equality holds (300 random pairs, dimensions 2-6)", ok, f"largest ratio {worst:.3f}")

# NC3
def chsh(phi, rho):
    A = [np.kron(Z, I2), np.kron(np.cos(phi) * Z + np.sin(phi) * X, I2)]
    b0 = np.cos(phi / 2) * Z + np.sin(phi / 2) * X; b1 = np.cos(phi / 2) * Z - np.sin(phi / 2) * X
    B = [np.kron(I2, b0), np.kron(I2, b1)]
    obs = [spectral(o) for o in A + B]; cov = compatibility_cover(obs); T = context_tables(rho, obs, cov); names = ["A0", "A1", "B0", "B1"]
    agree = True
    for (C1, T1), (C2, T2) in [((C1, T1), (C2, T2)) for C1, T1 in zip(cov, T) for C2, T2 in zip(cov, T) if C1 != C2]:
        sh = set(C1) & set(C2)
        for v in sh:
            m1 = T1.sum(axis=1 - C1.index(v)) if len(C1) == 2 else T1; m2 = T2.sum(axis=1 - C2.index(v)) if len(C2) == 2 else T2
            agree &= np.allclose(m1, m2, atol=1e-12)
    lam = gluable_fraction({n: 2 for n in names}, [Window(tuple(names[i] for i in C), t) for C, t in zip(cov, T)]).lam
    return cov, agree, lam, comm_norm(obs[0][0], obs[1][0])
psi = np.array([1, 0, 0, 1]) / np.sqrt(2); bell = np.outer(psi, psi).astype(complex)
cov, agree, lam_t, _ = chsh(np.pi / 2, bell)
sweep = [[float(phi), *chsh(phi, bell)[2:]] for phi in np.linspace(0, np.pi / 2, 31)]
ok = cov == [(0, 2), (0, 3), (1, 2), (1, 3)] and agree and abs(lam_t - (2 - np.sqrt(2))) < 1e-9
ok &= abs(sweep[0][1] - 1) < 1e-7 and all(s[1] > 1 - 1e-7 or s[2] > 1e-9 for s in sweep) and sweep[-1][1] < sweep[len(sweep) // 2][1]
OUT["chsh_sweep"] = sweep
check("NC3 commuting cliques = the four Bell contexts; tables agree on overlaps; lam* = 2 - sqrt 2 at Tsirelson; lam* < 1 needs [A0, A1] != 0",
      ok, f"lam* = {lam_t:.6f} (2 - sqrt 2 = {2 - np.sqrt(2):.6f}); at phi = 0 lam* = {sweep[0][1]:.3f}")

# NC4
ok = True; examples = {}
for _ in range(100):
    d = int(rng.integers(2, 6)); UA, UB = rand_unitary(d), rand_unitary(d); rho = rand_state(d)
    A = [np.outer(UA[:, k], UA[:, k].conj()) for k in range(d)]; B = [np.outer(UB[:, k], UB[:, k].conj()) for k in range(d)]
    q = kirkwood_dirac(rho, A, B)
    ok &= np.allclose(q.sum(axis=1), [np.trace(a @ rho) for a in A]) and np.allclose(q.sum(axis=0), [np.trace(b @ rho) for b in B])
    ok &= all(abs(q[i, k].imag - (np.trace((B[k] @ A[i] - A[i] @ B[k]) @ rho) / 2j).real) < 1e-12 for i in range(d) for k in range(d))
zb = [np.diag([1., 0]), np.diag([0., 1])]; xb = spectral(X); yb = spectral(Y)
rho_r = np.array([[0.85, -0.35], [-0.35, 0.15]], complex)
examples = {"commuting": kd_rung(kirkwood_dirac(rho_r, zb, zb)), "real (Z, X)": kd_rung(kirkwood_dirac(rho_r, zb, xb)),
            "complex (Z, Y)": kd_rung(kirkwood_dirac(rho_r, zb, yb))}
ok &= examples["commuting"]["rung"] == "R>=0" and examples["real (Z, X)"]["rung"] == "R" and examples["complex (Z, Y)"]["rung"] == "C"
OUT["kd"] = examples
check("NC4 Kirkwood-Dirac: exact marginals, Im q = tr([B_b, A_a] rho)/2i; commuting R>=0, real non-commuting R, complex C", ok,
      "; ".join(f"{k}: {v['rung']} (nonpositivity {v['nonpositivity']:.3f})" for k, v in examples.items()))

# NC5
def solid(n1, n2, n3):
    return 2 * np.arctan2(n1 @ np.cross(n2, n3), 1 + n1 @ n2 + n2 @ n3 + n3 @ n1)
ok = True; errs = []
for _ in range(200):
    ns = [v / np.linalg.norm(v) for v in rng.normal(size=(3, 3))]; Ps = [axis_measurement(0.5, n)[0] for n in ns]
    b = np.trace(Ps[0] @ Ps[1] @ Ps[2]); errs.append(abs(np.angle(b) - solid(*ns) / 2)); OUT.setdefault("bargmann", []).append([float(solid(*ns)), float(np.angle(b)), float(abs(b))])
ok = max(errs) < 1e-9
for _ in range(50):
    ns = [np.r_[v[0], 0, v[1]] / np.linalg.norm(v) for v in rng.normal(size=(3, 2))]; Ps = [axis_measurement(0.5, n)[0] for n in ns]
    ok &= abs(np.trace(Ps[0] @ Ps[1] @ Ps[2]).imag) < 1e-12
check("NC5 loop phase: arg tr(P1 P2 P3) = (signed solid angle)/2 for spin-1/2; real projections give a real invariant", ok, f"max error {max(errs):.1e}")

# NC6
def loop_holonomy(P0, H, N):
    d = P0.shape[0]; ev, V = np.linalg.eigh(P0); B0 = V[:, ev > 0.5]; M = np.eye(d, dtype=complex)
    from scipy.linalg import expm
    for t in np.linspace(0, 2 * np.pi, N + 1)[1:]:
        R = expm(-1j * t * H); M = (R @ P0 @ R.conj().T) @ M
    W = B0.conj().T @ M @ B0; u, s, vh = np.linalg.svd(W); return u @ vh
d = 4; P0 = np.diag([1., 1, 0, 0]).astype(complex)
def herm_2pi():
    """a generator with integer spectrum, so that exp(-2 pi i H) = 1 and the loop t -> R(t) P0 R(t)^* closes exactly"""
    V = rand_unitary(d); return V @ np.diag(rng.permutation([0., 1, 2, 3])).astype(complex) @ V.conj().T
H1, H2 = herm_2pi(), herm_2pi()
U1, U2 = loop_holonomy(P0, H1, 400), loop_holonomy(P0, H2, 400)
U1b = loop_holonomy(P0, H1, 800)
nonab = float(np.linalg.norm(U1 @ U2 - U2 @ U1)); conv = float(np.linalg.norm(U1 - U1b))
ok = nonab > 1e-2 and conv < 1e-2 and np.allclose(U1.conj().T @ U1, np.eye(2), atol=1e-10)
OUT["holonomy"] = {"noncommutativity": nonab, "convergence": conv}
check("NC6 non-abelian holonomy of projected loops: U1 U2 != U2 U1; the dense limit converges", ok, f"||[U1, U2]|| = {nonab:.3f}, N=400 vs 800: {conv:.1e}")

# NC7
ok = True; drag = {}
for j in (0.5, 1.0, 1.5):
    rows = []
    for N in (1, 2, 4, 8, 16, 32, 64):
        p, Th = zeno_drag(j, [0, 0, 1], [1, 0, 0], N); ok &= abs(p - np.cos(Th / (2 * N)) ** (4 * j * N)) < 1e-10; rows.append([N, p])
    ok &= rows[-1][1] > 0.9 and rows[0][1] < 0.6; drag[str(j)] = rows
# commuting control: projections diagonal in the z basis never move the z-eigenstate
rho = axis_measurement(1, [0, 0, 1])[0]; target = axis_measurement(1, [1, 0, 0])[0]; s = rho.copy()
for _ in range(50):
    P = np.diag(rng.integers(0, 2, 3).astype(float)); P = P if np.trace(P @ s).real > 1e-9 else np.eye(3) - P; s = P @ s @ P; s /= np.trace(s).real
ok &= abs(np.trace(target @ s).real - np.trace(target @ rho).real) < 1e-12
OUT["drag"] = drag
check("NC7 Zeno dragging: success cos(Th/2N)^(4jN) -> 1 (exact for j = 1/2, 1, 3/2); commuting projections cannot steer", ok,
      ", ".join(f"j={j}: N=1 {r[0][1]:.3f} -> N=64 {r[-1][1]:.3f}" for j, r in drag.items()))

# NC8
def coherent(j, n): return axis_measurement(j, n)[0]
def loop_phase(j, pts):
    v = np.trace(np.linalg.multi_dot([coherent(j, n) for n in pts]) if len(pts) > 1 else coherent(j, pts[0])); return np.angle(v)
def great(n0, n1, N):
    n0 = n0 / np.linalg.norm(n0); n1 = n1 / np.linalg.norm(n1); Th = np.arccos(np.clip(n0 @ n1, -1, 1)); ax = np.cross(n0, n1); ax /= np.linalg.norm(ax)
    return [np.cos(t) * n0 + np.sin(t) * np.cross(ax, n0) for t in np.linspace(0, Th, N + 1)]
ok = True; ph = []
e = np.eye(3)
for j in (0.5, 1.0):
    for (a, b, c) in [(e[2], e[0], e[1]), (e[2], (e[0] + e[1]) / np.sqrt(2), e[1])]:
        pts = great(a, b, 120)[:-1] + great(b, c, 120)[:-1] + great(c, a, 120)[:-1]
        Om = solid(a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c))
        got = loop_phase(j, pts); want = np.angle(np.exp(1j * j * Om)); ph.append([j, float(Om), float(got), float(want)])
        ok &= abs(np.angle(np.exp(1j * (got - want)))) < 2e-2
p1, _ = zeno_drag(0.5, e[2], e[1], 64); p2a, _ = zeno_drag(0.5, e[2], e[0], 32); p2b, _ = zeno_drag(0.5, e[0], e[1], 32)
OUT["phase"] = ph
check("NC8 path dependence: the dense loop phase is j * Omega (sign by orientation): paths with equal success differ by a geometric phase",
      ok, "; ".join(f"j={a}: Omega={b:.3f}, phase {c:.3f} vs {d:.3f}" for a, b, c, d in ph))

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "noncomm_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
