"""check_cbdfrac.py -- gluing fractions that allow for direct influence (note: level2/partial_gluing.pdf, Section 13).

  CF1 the relaxation over the right-angled Coxeter group of an n-cycle reproduces the quantum bound n cos(pi/n) of the odd-sign correlator
      sum for n = 4 (Tsirelson), 6, 8, at level 1+E and level 2
  CF2 the classical fraction of the consistified system is 1 - CNTF (Dzhafarov, Kujala and Cervantes); on cyclic systems it equals 1 - c,
      so the CbD degree c of pce.cbd coincides with CNTF (random contextual and noncontextual cyclic systems of rank 3, 4, 5)
  CF3 consistently connected systems: lam_C,cbd = lam*, and with bipartite commutation lam_Q,cbd = lam_Q (Bell NPA); with minimal
      commutation (copies commute only with what they are measured with) the quantum fraction is strictly larger on some boxes
  CF4 lam_C,cbd = 1 iff c = 0: signalling noncontextual systems give 1, contextual ones less
  CF5 ordering on random signalling boxes: lam* <= lam_C,cbd <= lam_Q,cbd(bipartite) <= lam_Q,cbd(minimal) <= 1, lam_Q <= lam_Q,cbd(bipartite)
  CF6 a quantum model with direct influence (Tsirelson state; Alice's output forced to +1 with probability eps when Bob's setting is 1,
      through an ancilla, so her two copies commute and are maximally coupled): lam_Q,cbd = 1 although lam_C,cbd < 1 and lam_Q < 1
  CF7 a PR box with added direct influence stays beyond quantum: lam_Q,cbd = lam_C,cbd < 1 (quantum adds nothing); recoding any
      variable leaves every fraction unchanged (to the solver accuracy of about 2e-4: the program is not strictly feasible)
Writes ../level2/cbdfrac_results.json.  Run: python check_cbdfrac.py
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.cbd import cbd
from pce.cbdfrac import edge_npa_value, lam_classical_cbd, lam_quantum_cbd, consistify, bipartite
from pce.integrate import gluable_fraction
from pce.npa import npa_fraction, behaviour

rng = np.random.default_rng(20261025); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
DOM = {"A1": 2, "A2": 2, "B1": 2, "B2": 2}; KEYS = [(0, 0), (0, 1), (1, 0), (1, 1)]
def bell_windows(P):
    return [Window((f"A{x+1}", f"B{y+1}"), np.asarray(P[x, y], float)) for x, y in KEYS]

# CF1
vals = {}
for n in (4, 6, 8):
    E = [(i, (i + 1) % n, None) for i in range(n)]; co = [1] * (n - 1) + [-1]
    for lv in ("1+E", 2) if n < 8 else ("1+E",):
        vals[f"n={n}, level {lv}"] = edge_npa_value(n, E, co, lv)
ok = all(abs(v - int(k.split(",")[0][2:]) * np.cos(np.pi / int(k.split(",")[0][2:]))) < 1e-6 for k, v in vals.items())
OUT["cycle_bounds"] = vals
check("CF1 n-cycle relaxation reproduces n cos(pi/n) (n = 4, 6, 8)", ok, "; ".join(f"{k}: {v:.6f}" for k, v in vals.items()))

# CF2
def cyclic(n, contextual):
    W = []
    for i in range(n):
        anti = contextual and i == n - 1
        T = np.array([[0, 1], [1, 0]]) / 2 if anti else np.eye(2) / 2
        T = (1 - 0.25) * T + 0.25 * rng.dirichlet(np.ones(4)).reshape(2, 2) if contextual else rng.dirichlet(np.ones(4) * 0.6).reshape(2, 2)
        a = rng.uniform(0, 0.08); T = (1 - a) * T + a * np.outer([1, 0], T.sum(axis=0))
        W.append(Window((f"R{i}", f"R{(i+1)%n}"), T))
    return W
dev = []; cs = []
for n in (3, 4, 5):
    for t in range(8):
        W = cyclic(n, t % 2 == 0); c = cbd(W)["c"]; lc = lam_classical_cbd(W); dev.append(abs(lc - (1 - c))); cs.append(c)
OUT["cntf_dev"] = max(dev)
check("CF2 lam_C,cbd = 1 - CNTF = 1 - c on cyclic systems (rank 3, 4, 5; 24 random systems)", max(dev) < 1e-7 and sum(c > 1e-3 for c in cs) >= 6,
      f"largest deviation {max(dev):.1e}; {sum(c > 1e-6 for c in cs)} of 24 contextual, c up to {max(cs):.3f}")

# random no-signalling boxes
def det_box(a0, a1, b0, b1):
    return behaviour({"A": [a0, a1], "B": [b0, b1], "AB": [[a0 * b0, a0 * b1], [a1 * b0, a1 * b1]]})
LOC = [det_box(*s) for s in itertools.product((1, -1), repeat=4)]
iso = lambda v: behaviour({"A": [0, 0], "B": [0, 0], "AB": [[v, v], [v, -v]]})
PR, TS = iso(1.0), iso(1 / np.sqrt(2))
def ns_box():
    w = rng.dirichlet(np.ones(3)); wl = rng.dirichlet(np.ones(16)); return w[0] * PR + w[1] * TS + w[2] * sum(wl[k] * LOC[k] for k in range(16))

# CF3
ok = True; d3 = []; larger = 0
for _ in range(8):
    P = ns_box(); W = bell_windows(P)
    a, b = gluable_fraction(DOM, W).lam, lam_classical_cbd(W); c, d = npa_fraction(P, "1+AB"), lam_quantum_cbd(W, parties=bipartite)
    m = lam_quantum_cbd(W); larger += m > d + 1e-3
    d3.append(max(abs(a - b), abs(c - d))); ok &= abs(a - b) < 1e-7 and abs(c - d) < 2e-5 and m >= d - 2e-5
ok &= larger > 0
check("CF3 consistently connected: lam_C,cbd = lam*, lam_Q,cbd(bipartite) = lam_Q; minimal commutation is larger (8 boxes)", ok,
      f"largest deviation {max(d3):.1e}; minimal commutation strictly larger on {larger} of 8")

# signalling boxes: a no-signalling box with each party's output pushed toward +1 by an amount depending on the other party's setting
def push(P, eA, eB):
    Q = np.array(P, float)
    for x, y in KEYS:
        T = Q[x, y].copy()
        a = eA[x][y]; T = (1 - a) * T + a * np.outer([1, 0], T.sum(axis=0))       # Alice's output forced to + with prob a
        b = eB[y][x]; T = (1 - b) * T + b * np.outer(T.sum(axis=1), [1, 0])       # Bob's output forced to + with prob b
        Q[x, y] = T
    return Q

# CF4
ok = True; rows = []
for t in range(12):
    if t < 6: P = sum(w * LOC[k] for k, w in enumerate(rng.dirichlet(np.ones(16))))                 # local, then signalling
    else: P = 0.5 * PR + 0.5 * ns_box()
    S = push(P, rng.uniform(0, 0.06, (2, 2)), rng.uniform(0, 0.06, (2, 2))); W = bell_windows(S)
    c = cbd(W)["c"]; lc = lam_classical_cbd(W); rows.append((c, lc)); ok &= (c < 1e-7) == (lc > 1 - 1e-7)
ok &= sum(c >= 1e-7 for c, _ in rows) >= 4 and sum(c < 1e-7 for c, _ in rows) >= 4
check("CF4 lam_C,cbd = 1 iff c = 0 (12 signalling systems)", ok,
      f"{sum(c < 1e-7 for c, _ in rows)} noncontextual (all lam_C,cbd = 1), {sum(c >= 1e-7 for c, _ in rows)} contextual (max lam_C,cbd {max([l for c, l in rows if c >= 1e-7] or [0]):.3f})")

# CF5
ok = True; chain = []
for _ in range(10):
    S = push(0.6 * PR + 0.4 * ns_box(), rng.uniform(0, 0.08, (2, 2)), rng.uniform(0, 0.08, (2, 2))); W = bell_windows(S)
    v = [gluable_fraction(DOM, W).lam, lam_classical_cbd(W), lam_quantum_cbd(W, parties=bipartite), lam_quantum_cbd(W)]; q = npa_fraction(S, "1+AB")
    ok &= v[0] <= v[1] + 1e-7 and v[1] <= v[2] + 2e-5 and v[2] <= v[3] + 2e-5 and v[3] <= 1 + 1e-6 and q <= v[2] + 2e-5; chain.append(v + [q])
OUT["chain"] = chain
check("CF5 lam* <= lam_C,cbd <= lam_Q,cbd(bipartite) <= lam_Q,cbd(minimal) <= 1; lam_Q <= lam_Q,cbd(bipartite) (10 signalling boxes)", ok,
      "e.g. " + ", ".join(f"{x:.3f}" for x in chain[0]))

# CF6
X = np.array([[0, 1], [1, 0]], float); Z = np.diag([1.0, -1.0]); I2 = np.eye(2)
def proj(M):
    w, V = np.linalg.eigh(M); return [V[:, w > 0] @ V[:, w > 0].T, V[:, w < 0] @ V[:, w < 0].T]
def quantum_di(eps):
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2); anc = np.array([np.sqrt(1 - eps), np.sqrt(eps)])
    psi = np.kron(anc, phi)                                      # ancilla (Alice's), Alice's qubit, Bob's qubit
    Aop = [Z, X]; Bop = [(Z + X) / np.sqrt(2), (Z - X) / np.sqrt(2)]
    P0, P1 = np.diag([1.0, 0]), np.diag([0, 1.0])
    P = np.zeros((2, 2, 2, 2))
    for x, y in KEYS:
        A = np.kron(P0, Aop[x]) + np.kron(P1, I2) if y == 1 else np.kron(I2, Aop[x])    # copy of A_x in context y
        A = np.kron(A, I2); B = np.kron(np.eye(4), Bop[y])
        for ia, PA in enumerate(proj(A)):
            for ib, PB in enumerate(proj(B)): P[x, y, ia, ib] = psi @ PA @ PB @ psi
    return P
rows = []; ok = True
for eps in (0.02, 0.05):
    S = quantum_di(eps); W = bell_windows(S)
    lc, lq, q0 = lam_classical_cbd(W), lam_quantum_cbd(W, parties=bipartite), npa_fraction(S, "1+AB"); lq2 = lam_quantum_cbd(W, 2, parties=bipartite)
    rows.append([eps, lc, lq, lq2, q0]); ok &= lq > 1 - 2e-5 and lq2 > 1 - 2e-5 and lc < 1 - 1e-3 and q0 < 1 - 1e-3
OUT["quantum_di"] = rows
check("CF6 quantum model with direct influence: lam_Q,cbd = 1 (levels 1+E and 2) while lam_C,cbd < 1 and lam_Q < 1", ok,
      "; ".join(f"eps {r[0]}: lam_C,cbd {r[1]:.3f}, lam_Q,cbd {r[2]:.5f}/{r[3]:.5f}, lam_Q {r[4]:.3f}" for r in rows))

# CF7
S = push(PR, [[0, 0.2], [0, 0.2]], [[0.1, 0], [0.1, 0]]); W = bell_windows(S); lq = lam_quantum_cbd(W, parties=bipartite); lc = lam_classical_cbd(W)
ok = lq < 1 - 1e-3 and abs(lq - lc) < 1e-3
def recode(W, var):
    out = []
    for w in W:
        T = np.asarray(w.table, float).copy()
        if w.vars[0] == var: T = T[::-1, :]
        if w.vars[1] == var: T = T[:, ::-1]
        out.append(Window(w.vars, T))
    return out
inv = max(max(abs(lam_classical_cbd(recode(W, v)) - lc), abs(lam_quantum_cbd(recode(W, v), parties=bipartite) - lq)) for v in DOM)
ok &= inv < 5e-4
check("CF7 PR box with direct influence stays beyond quantum (lam_Q,cbd < 1); fractions invariant under recoding", ok,
      f"lam_C,cbd {lc:.3f}, lam_Q,cbd {lq:.3f}; recoding deviation {inv:.1e}")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "cbdfrac_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
