"""check_qglue.py -- quantum graded gluing as the positivity domain of a hereditary quadratic (note: level2/partial_gluing.pdf, Section 11).

A window w with density matrix rho_w gives f_w = rho_w^T - K^*K, a self-adjoint hereditary quadratic matrix polynomial in the entries of an
unnormalised global vector psi; f_w(psi) >= 0 iff Tr_rest(psi psi^*) <= rho_w. A family gives f = (+)_w f_w, whose positivity domain is the
set of quantum explanations (Islam, Volcic and Woerdeman, arXiv:2610.14053).

  QG1 the hypotheses of the paper hold: f_w evaluates to rho_w^T - Tr_rest(psi psi^*)^T; its quadratic coefficients generate the full matrix
      algebra (coefficient-irreducible); the line system is minimal; the linearisation [[I, K], [K^*, rho^T]] generates the full algebra
      (irreducible pencil, so f_w is an atom); f_w(0) > 0 iff rho_w has full rank; f_w is unsignatured; at level 2 the domain is a free set
      (closed under direct sums, unitary similarity and compression to a common invariant subspace)
  QG2 the walls: at the boundary point s* u of every ray, the active window has det f_w = 0 (Prop. 2.21) and the real gradient of det f_w
      is non-zero there, a regular real point (so the real locus is Zariski dense in that component, Prop. 3.14); every window of the
      family is active somewhere (positivity-minimal)
  QG3 the classical limit: for diagonal (classical) windows the quantum fraction (a semidefinite program) equals the classical lam* (a
      linear program)
  QG4 the dephasing bound: for every product basis, the quantum fraction is at most the classical lam* of the tables in that basis; for
      pairwise Werner windows on three qubits the quantum fraction is min(1, 3(1 - p)/2) and equals the Z-basis lam* (monogamy: full gluing
      only for p <= 1/3)
  QG5 the semidefinite dual certifies each quantum fraction (duality gap < 1e-6)
  QG6 pure (level-1, single-actor) explanations: the best pure fraction found is feasible and at most the mixed fraction, often far below
  QG7 the Werner formula, proved: by symmetry (U x U x U and qubit permutations) the optimum is attained on a Pi_{3/2} + b Pi_{1/2}; the
      partial traces of the two projectors and a two-variable linear program give lam_Q = min(1, 3(1-p)/2), which also equals the Z-basis
      lam* (proved by counting equal pairs); checked for p in [-1/3, 0.95]
Writes ../level2/qglue_results.json.  Run: python check_qglue.py
"""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.integrate import gluable_fraction
from pce.qglue import ptrace, Kcoeffs, f_eval, linearisation, algebra_dim, radial, lam_mixed, ascend, _reorder

rng = np.random.default_rng(20261024); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
def rstate(d, mix=0.3):
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); R = G @ G.conj().T; R /= np.trace(R).real
    return (1 - mix) * R + mix * np.eye(d) / d
def runit(D): u = rng.normal(size=D) + 1j * rng.normal(size=D); return u / np.linalg.norm(u)
m = 3; PAIRS = [[0, 1], [1, 2], [0, 2]]

# QG1
ok = True; info = []
for keep in ([0, 1], [1], [0, 2]):
    rho = rstate(2 ** len(keep)); Ks = Kcoeffs(keep, m)
    for _ in range(5):
        u = runit(8) * rng.uniform(0.5, 2); ok &= np.allclose(f_eval(rho, keep, m, u.reshape(8, 1, 1)), (rho - ptrace(np.outer(u, u.conj()), keep, m)).T)
    C = [Kj.conj().T @ Kl for Kj in Ks for Kl in Ks]; dk = 2 ** len(keep)
    ok &= algebra_dim([c for c in C if np.linalg.norm(c) > 0]) == dk * dk
    kerK = np.linalg.matrix_rank(np.vstack(Ks)); kerKs = np.linalg.matrix_rank(np.hstack([Kj for Kj in Ks]))
    ok &= kerK == Ks[0].shape[1] and kerKs == Ks[0].shape[0]                       # common kernels of K_j and of K_j^* are trivial
    C0, A, As = linearisation(rho, keep, m); inv = np.linalg.inv(C0); dimA = algebra_dim([inv @ M for M in A + As])
    ok &= dimA == C0.shape[0] ** 2; info.append(f"window {keep}: pencil algebra {dimA}/{C0.shape[0] ** 2}")
    ok &= np.linalg.eigvalsh(rho).min() > 0 and np.linalg.eigvalsh(f_eval(rho, keep, m, np.zeros((8, 1, 1)))).min() > 0
    big = f_eval(rho, keep, m, (10 * runit(8)).reshape(8, 1, 1)); ok &= np.linalg.eigvalsh(big).min() < 0     # unsignatured
    sing = np.zeros((dk, dk)); sing[0, 0] = 1; ok &= np.linalg.eigvalsh(f_eval(sing, keep, m, np.zeros((8, 1, 1)))).min() <= 1e-12
# free-set properties at level 2
rho = rstate(4); keep = [0, 1]
def inD(X): return np.linalg.eigvalsh(f_eval(rho, keep, m, X)).min() >= -1e-10
s1 = radial([(keep, rho)], m, (u1 := runit(8))); s2 = radial([(keep, rho)], m, (u2 := runit(8)))
x1 = 0.9 * np.sqrt(s1) * u1; x2 = 0.9 * np.sqrt(s2) * u2
Xsum = np.zeros((8, 2, 2), complex); Xsum[:, 0, 0] = x1; Xsum[:, 1, 1] = x2; ok &= inD(Xsum)
U = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]; ok &= inD(np.einsum("ab,jbc,cd->jad", U.conj().T, Xsum, U))
Xtri = Xsum.copy(); Xtri[:, 0, 1] = 0.3 * runit(8)
if inD(Xtri): ok &= inD(Xtri[:, :1, :1])                                          # compression to the invariant subspace e_1
check("QG1 hypotheses of the paper: evaluation, coefficient-irreducible, minimal line system, irreducible linearisation (atom), "
      "f(0) > 0 iff full rank, unsignatured, free set at level 2", ok, "; ".join(info))

# QG2
ok = True; regs = []; active_all = True
for _ in range(6):
    wins = [(k, rstate(4)) for k in PAIRS]
    for _ in range(10):
        u = runit(8); s = radial(wins, m, u); x = np.sqrt(s) * u
        dets = [np.linalg.det(f_eval(rho, k, m, x.reshape(8, 1, 1))).real for k, rho in wins]; j = int(np.argmin(np.abs(dets)))
        ok &= abs(dets[j]) < 1e-9 and all(np.linalg.eigvalsh(f_eval(rho, k, m, x.reshape(8, 1, 1))).min() > -1e-9 for k, rho in wins)
        k, rho = wins[j]; z = np.r_[x.real, x.imag]; g = np.zeros(16)
        for t in range(16):
            e = np.zeros(16); e[t] = 1e-6
            fp = np.linalg.det(f_eval(rho, k, m, ((z + e)[:8] + 1j * (z + e)[8:]).reshape(8, 1, 1))).real
            fm = np.linalg.det(f_eval(rho, k, m, ((z - e)[:8] + 1j * (z - e)[8:]).reshape(8, 1, 1))).real
            g[t] = (fp - fm) / 2e-6
        regs.append(np.linalg.norm(g))
    act = set()
    for _ in range(300):
        u = runit(8); act.add(int(np.argmin([radial([w], m, u) for w in wins])))
    active_all &= act == {0, 1, 2}
ok &= min(regs) > 1e-6 and active_all
check("QG2 walls: boundary points lie on det f_w = 0 of the active window (Prop. 2.21) and are regular real points (non-zero gradient); "
      "every window is active somewhere (positivity-minimal)", ok, f"smallest |grad det| at a boundary point {min(regs):.2e} over {len(regs)} points")

# QG3
ok = True; diffs = []
for _ in range(12):
    T = [rng.dirichlet(np.ones(4)).reshape(2, 2) * 0.9 + 0.025 for _ in range(3)]
    G = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); T = [G.sum(axis=2), G.sum(axis=0), G.sum(axis=1)] if rng.random() < 0.3 else T
    lc = gluable_fraction({"A": 2, "B": 2, "C": 2}, [Window(("A", "B"), T[0]), Window(("B", "C"), T[1]), Window(("A", "C"), T[2])]).lam
    lq, _ = lam_mixed([(k, np.diag(t.reshape(-1)).astype(complex)) for k, t in zip(PAIRS, T)], m); diffs.append(abs(lq - lc))
ok = max(diffs) < 1e-6
check("QG3 classical limit: for diagonal windows the quantum fraction equals the classical lam*", ok, f"largest |lam_Q - lam*| {max(diffs):.1e} over 12 families")

# QG4
def local_basis_tables(wins, Us):
    tabs = []
    for k, rho in wins:
        Uw = np.kron(Us[k[0]], Us[k[1]]); tabs.append(np.real(np.diag(Uw.conj().T @ rho @ Uw)).reshape(2, 2))
    return tabs
def haar2():
    return np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
ok = True; gaps = []
for _ in range(6):
    wins = [(k, rstate(4, 0.4)) for k in PAIRS]; lq, _ = lam_mixed(wins, m); best = 1.0
    for _ in range(20):
        Us = [haar2() for _ in range(3)]; T = local_basis_tables(wins, Us)
        lc = gluable_fraction({"A": 2, "B": 2, "C": 2}, [Window(("A", "B"), T[0]), Window(("B", "C"), T[1]), Window(("A", "C"), T[2])]).lam
        ok &= lq <= lc + 1e-6; best = min(best, lc); OUT.setdefault("dephasing_pairs", []).append([float(lq), float(lc)])
    gaps.append(best - lq)
sing = np.array([0, 1, -1, 0]) / np.sqrt(2); S = np.outer(sing, sing); wer = []
for p in np.linspace(0.05, 0.95, 19):
    W = p * S + (1 - p) * np.eye(4) / 4; lq, _ = lam_mixed([(k, W) for k in PAIRS], m); d = np.real(np.diag(W)).reshape(2, 2)
    lc = gluable_fraction({"A": 2, "B": 2, "C": 2}, [Window(("A", "B"), d), Window(("B", "C"), d), Window(("A", "C"), d)]).lam
    wer.append([float(p), lq, lc, min(1, 1.5 * (1 - p))])
ok &= all(abs(r[1] - r[3]) < 1e-6 and abs(r[1] - r[2]) < 1e-6 for r in wer)
OUT["werner"] = wer; OUT["dephasing_gaps"] = gaps
check("QG4 dephasing bound: lam_Q <= lam* of the tables in every product basis (6 random families x 20 bases); Werner triangle: "
      "lam_Q = min(1, 3(1-p)/2) = lam*(Z basis), full gluing iff p <= 1/3", ok,
      "smallest gap to the best sampled basis per family: " + ", ".join(f"{g:.3f}" for g in gaps))

# QG5
import cvxpy as cp
ok = True; dg = []
for trial in range(5):
    wins = [(k, rstate(4, 0.3)) for k in PAIRS]
    D = 8; Sv = cp.Variable((D, D), hermitian=True); cons = [Sv >> 0]; duals = []
    for keep, rho in wins:
        rest = [q for q in range(m) if q not in keep]; T = cp.partial_trace(Sv, [2, 2, 2], axis=rest[0]); c = rho - T >> 0; cons.append(c); duals.append(c)
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Sv))), cons); prob.solve(solver="SCS", eps=1e-9, max_iters=200000)
    Y = [c.dual_value for c in duals]; dual_obj = sum(np.real(np.trace(y @ rho)) for y, (k, rho) in zip(Y, wins))
    # dual feasibility: sum_w (Y_w (x) I_rest) - I >= 0 on the full space, in the original qubit order
    Mtot = np.zeros((8, 8), complex)
    for y, (k, rho) in zip(Y, wins):
        rest = [q for q in range(m) if q not in k]; Big = np.kron(y, np.eye(2)).reshape([2] * 6)
        order = k + rest; inv = np.argsort(order); Big = Big.transpose(list(inv) + [i + 3 for i in inv]).reshape(8, 8); Mtot += Big
    ok &= np.linalg.eigvalsh(Mtot - np.eye(8)).min() > -1e-6 and all(np.linalg.eigvalsh(y).min() > -1e-7 for y in Y)
    dg.append(abs(dual_obj - prob.value))
ok &= max(dg) < 1e-6
check("QG5 the semidefinite dual (Y_w >= 0 with sum_w Y_w (x) I >= I) certifies each quantum fraction", ok, f"largest duality gap {max(dg):.1e}")

# QG6
ok = True; rows = []
for p in (0.2, 0.5, 0.8):
    W = p * S + (1 - p) * np.eye(4) / 4; wins = [(k, W) for k in PAIRS]; lq, _ = lam_mixed(wins, m)
    best = 0; ub = None
    for st in range(6):
        v, u = ascend(wins, m, runit(8), iters=6000, beta=2000.0, lr=0.02)
        if v > best: best, ub = v, u
    x = np.sqrt(best) * ub; feas = all(np.linalg.eigvalsh(f_eval(rho, k, m, x.reshape(8, 1, 1))).min() > -1e-8 for k, rho in wins)
    ok &= feas and best <= lq + 1e-6; rows.append([p, best, lq])
OUT["pure_vs_mixed"] = rows
check("QG6 pure single-actor explanations: the best found is feasible and below the mixed fraction", ok,
      "; ".join(f"p={a}: pure >= {b:.3f}, mixed {c:.3f}" for a, b, c in rows))

# QG7 the Werner formula, proved by symmetry: the optimum is attained on a Pi_{3/2} + b Pi_{1/2}
perms = list(itertools.permutations(range(3)))
def perm_op(pm):
    M = np.zeros((8, 8))
    for i in range(8):
        bb = [(i >> (2 - k)) & 1 for k in range(3)]; nb = [bb[pm[k]] for k in range(3)]; M[sum(nb[k] << (2 - k) for k in range(3)), i] = 1
    return M
Psym = sum(perm_op(pm) for pm in perms) / 6; Phalf = np.eye(8) - Psym; Ps = S; Pt = np.eye(4) - Ps
ok = np.linalg.matrix_rank(Psym) == 4 and all(np.allclose(ptrace(Psym, k, m), 4 / 3 * Pt) and np.allclose(ptrace(Phalf, k, m), 2 / 3 * Pt + 2 * Ps) for k in PAIRS)
rows7 = []
for p in np.linspace(-1 / 3, 0.95, 24):
    W = p * Ps + (1 - p) * np.eye(4) / 4; sv, tv = (1 + 3 * p) / 4, (1 - p) / 4
    b = min(sv / 2, 3 * tv / 2); a = max(0.0, (3 * tv - 2 * b) / 4); val = 4 * a + 4 * b; sig = a * Psym + b * Phalf
    feas = all(np.linalg.eigvalsh(W - ptrace(sig, k, m)).min() > -1e-12 for k in PAIRS)
    lq, _ = lam_mixed([(k, W) for k in PAIRS], m); d = np.real(np.diag(W)).reshape(2, 2)
    lc = gluable_fraction({"A": 2, "B": 2, "C": 2}, [Window(("A", "B"), d), Window(("B", "C"), d), Window(("A", "C"), d)]).lam
    f = min(1, 1.5 * (1 - p)); ok &= feas and abs(val - f) < 1e-12 and abs(lq - f) < 1e-6 and abs(lc - f) < 1e-6; rows7.append([float(p), val, lq, lc])
OUT["werner_proof"] = rows7
check("QG7 Werner formula proved: Tr_3 Pi_{3/2} = (4/3) Pi_trip, Tr_3 Pi_{1/2} = (2/3) Pi_trip + 2 Pi_sing; the invariant two-variable LP gives "
      "min(1, 3(1-p)/2), its optimiser is feasible, and it equals the SDP and the Z-basis lam* for p in [-1/3, 0.95]", ok, f"{len(rows7)} values of p")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "qglue_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
