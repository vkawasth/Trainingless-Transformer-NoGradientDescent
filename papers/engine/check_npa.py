"""check_npa.py -- the categorical home, computed (note: level2/partial_gluing.pdf, Section 12).

  NP1 positivity on truncated free modules (the NPA hierarchy) reproduces Tsirelson's bound: max CHSH = 2 sqrt 2 at levels 1, 1+AB, 2
  NP2 the dimension-free quantum rung against the classical one, for isotropic PR boxes of strength v (correlators +-v): lam* = 2(1 - v)
      (from v = 1/2 on), the quantum fraction is 1 up to v = 1/sqrt 2 and (1 - v)/(1 - 1/sqrt 2) beyond (a scaled Tsirelson box); the
      Tsirelson box itself: lam* = 2 - sqrt 2, quantum 1; the PR box: both 0
  NP3 the hierarchy is ordered: lam* <= lam_NPA(2) <= lam_NPA(1+AB) <= lam_NPA(1) <= 1 on random no-signalling boxes
  NP4 data processing (Perrone's principle for the gluing fractions): local classical post-processing of the outcomes never decreases lam*
      (theorem: push the optimal sub-measure through the channels) nor the level-(1+AB) quantum fraction (checked numerically)
  NP5 holonomy against finite dimension: invertible n x n matrices with U V = q V U exist iff q^n = 1 (determinant), so for q = e^(2 pi i k/N)
      they exist exactly in dimensions divisible by N (clock and shift) and for q = e^(2 pi i sqrt 2) in no dimension; numerically, the best
      unitary residual ||UV - qVU|| is 0 when N | n and bounded away from 0 otherwise
  NP6 Kochen-Specker against Gleason (de Silva-Barbosa's reading) on the Peres-Mermin square: the commutation cliques are the six lines; no
      +-1 assignment is consistent with all of them (no global points); for every state the six context tables agree on overlaps (linear
      gap 0: compatible probabilities, a signed global section) yet lam* = 0 (no measure on points), state-independently
Writes ../level2/npa_results.json.  Run: python check_npa.py
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.npa import npa_value, npa_fraction, local_fraction, behaviour
from pce.noncomm import spectral, compatibility_cover, context_tables
from pce.ladder import level1
from scipy.linalg import expm
from scipy.optimize import minimize

rng = np.random.default_rng(20261024); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
r2 = 1 / np.sqrt(2)

# NP1
vals = {lv: npa_value([[1, 1], [1, -1]], lv) for lv in ("1", "1+AB", "2")}
check("NP1 NPA reproduces Tsirelson's bound 2 sqrt 2 at levels 1, 1+AB, 2", all(abs(v - 2 * np.sqrt(2)) < 1e-6 for v in vals.values()),
      ", ".join(f"level {k}: {v:.9f}" for k, v in vals.items()))

# NP2
iso = lambda v: behaviour({"A": [0, 0], "B": [0, 0], "AB": [[v, v], [v, -v]]})
rows = []; ok = True
for v in np.linspace(0.5, 1.0, 21):
    P = iso(v); lc = local_fraction(P); lq = npa_fraction(P, "1+AB"); fq = 1.0 if v <= r2 else (1 - v) / (1 - r2)
    ok &= abs(lc - 2 * (1 - v)) < 1e-6 and abs(lq - fq) < 1e-5; rows.append([float(v), lc, lq, fq])
ts = iso(r2); ok &= abs(local_fraction(ts) - (2 - np.sqrt(2))) < 1e-6 and abs(npa_fraction(ts) - 1) < 1e-6
ok &= abs(local_fraction(iso(1.0))) < 1e-9 and abs(npa_fraction(iso(1.0))) < 1e-6
OUT["isotropic"] = rows
check("NP2 isotropic PR boxes: lam* = 2(1-v); quantum fraction 1 up to v = 1/sqrt 2, then (1-v)/(1-1/sqrt 2); Tsirelson box 2-sqrt 2 vs 1; PR box 0",
      ok, f"21 values of v in [0.5, 1]; largest deviation from the formulas {max(max(abs(a[1] - 2 * (1 - a[0])), abs(a[2] - a[3])) for a in rows):.1e}")

# NP3
def det_box(a0, a1, b0, b1):
    return behaviour({"A": [a0, a1], "B": [b0, b1], "AB": [[a0 * b0, a0 * b1], [a1 * b0, a1 * b1]]})
LOC = [det_box(*s) for s in itertools.product((1, -1), repeat=4)]
PR = iso(1.0); TS = iso(r2)
ok = True; chain = []
for _ in range(12):
    wts = rng.dirichlet(np.ones(3)); wl = rng.dirichlet(np.ones(16)); P = wts[0] * PR + wts[1] * TS + wts[2] * sum(wl[k] * LOC[k] for k in range(16))
    assert np.allclose(P.sum(axis=(2, 3)), 1); c = [local_fraction(P), npa_fraction(P, "2"), npa_fraction(P, "1+AB"), npa_fraction(P, "1")]
    ok &= all(c[i] <= c[i + 1] + 1e-6 for i in range(3)) and c[-1] <= 1 + 1e-6; chain.append(c)
OUT["chain"] = chain
check("NP3 hierarchy ordered: lam* <= NPA(2) <= NPA(1+AB) <= NPA(1) <= 1 (12 random no-signalling boxes)", ok,
      "e.g. " + ", ".join(f"{x:.3f}" for x in chain[0]))

# NP4
def post(P, KA, KB):
    out = np.zeros_like(P)
    for x, y in itertools.product(range(2), range(2)): out[x, y] = KA[x] @ P[x, y] @ KB[y].T
    return out
ok = True; dl = []; dq = []
for _ in range(10):
    wts = rng.dirichlet(np.ones(3)); P = wts[0] * PR + wts[1] * TS + wts[2] * LOC[int(rng.integers(16))]
    KA = [rng.dirichlet(np.ones(2), 2).T for _ in range(2)]; KB = [rng.dirichlet(np.ones(2), 2).T for _ in range(2)]
    Q = post(P, KA, KB); a, b = local_fraction(P), local_fraction(Q); c, d = npa_fraction(P), npa_fraction(Q)
    ok &= b >= a - 1e-7 and d >= c - 1e-6; dl.append(b - a); dq.append(d - c)
check("NP4 data processing: local post-processing of outcomes never decreases lam* (theorem) or the quantum fraction (numerical)", ok,
      f"smallest increase: classical {min(dl):.2e}, quantum {min(dq):.2e} over 10 boxes")

# NP5
def clock_shift(N, m, k):
    q = np.exp(2j * np.pi * k / N); C = np.diag(q ** np.arange(N)); S = np.roll(np.eye(N), 1, axis=0)
    return np.kron(C, np.eye(m)), np.kron(S, np.eye(m)), q
ok = True
for N, k in ((3, 1), (4, 1), (5, 2)):
    for m in (1, 2):
        U, V, q = clock_shift(N, m, k); ok &= np.linalg.norm(U @ V - q * V @ U) < 1e-12 or np.linalg.norm(V @ U - q * U @ V) < 1e-12
def best_residual(n, q, starts=6):
    def unit(z):
        H = z.reshape(2, n, n); H = H[0] + 1j * H[1]; H = (H + H.conj().T) / 2; return expm(1j * H)
    best = np.inf
    for _ in range(starts):
        z0 = rng.normal(size=4 * n * n)
        f = lambda z: np.linalg.norm(unit(z[:2 * n * n]) @ unit(z[2 * n * n:]) - q * unit(z[2 * n * n:]) @ unit(z[:2 * n * n])) ** 2
        r = minimize(f, z0, method="L-BFGS-B", options={"maxiter": 3000}); best = min(best, np.sqrt(r.fun))
    return best
res = {}
q3 = np.exp(2j * np.pi / 3); qirr = np.exp(2j * np.pi * np.sqrt(2))
for n in (2, 3, 4): res[f"q=e^(2pi i/3), n={n}"] = best_residual(n, q3)
for n in (2, 3): res[f"q=e^(2pi i sqrt2), n={n}"] = best_residual(n, qirr)
ok &= res["q=e^(2pi i/3), n=3"] < 1e-5 and res["q=e^(2pi i/3), n=2"] > 1e-2 and res["q=e^(2pi i/3), n=4"] > 1e-2 and all(
    v > 1e-2 for k, v in res.items() if "sqrt2" in k)
OUT["holonomy"] = res
check("NP5 holonomy vs dimension: UV = qVU has invertible n x n solutions iff q^n = 1 (clock-shift when N | n); none for irrational phase", ok,
      "; ".join(f"{k}: min residual {v:.3g}" for k, v in res.items()))

# NP6
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex); I = np.eye(2)
K = lambda a, b: np.kron(a, b)
PM = [K(X, I), K(I, X), K(X, X), K(I, Z), K(Z, I), K(Z, Z), K(X, Z), K(Z, X), K(Y, Y)]
obs = [spectral(o) for o in PM]; cov = compatibility_cover(obs)
lines = sorted([(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)])
ok = sorted(cov) == lines
sign = []
for C in cov:
    Mprod = PM[C[0]] @ PM[C[1]] @ PM[C[2]]; sign.append(int(round(np.real(Mprod[0, 0]))))
ok &= all(np.allclose(PM[C[0]] @ PM[C[1]] @ PM[C[2]], s * np.eye(4)) for C, s in zip(cov, sign))
glob = 0
for s in itertools.product((1, -1), repeat=9):
    glob += all(s[C[0]] * s[C[1]] * s[C[2]] == sg for C, sg in zip(cov, sign))
ok &= glob == 0
lamr = []
names = [f"O{i}" for i in range(9)]
for t in range(6):
    if t == 0: rho = np.eye(4) / 4
    else:
        G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); rho = G @ G.conj().T; rho /= np.trace(rho).real
    T = context_tables(rho, obs, cov); W = [Window(tuple(names[i] for i in C), tb) for C, tb in zip(cov, T)]
    lv = level1({n: 2 for n in names}, W); lamr.append((lv["linear_gap"], lv["lam"], lv["rung"]))
ok &= all(g < 1e-7 and l < 1e-7 and r == "R" for g, l, r in lamr)
OUT["peres_mermin"] = {"signs": sign, "global_assignments": glob, "states": lamr}
check("NP6 Peres-Mermin: cliques = six lines; no global +-1 assignment (Kochen-Specker); every state's tables are compatible (linear gap 0, "
      "signed section) yet lam* = 0 (Gleason without points)", ok, f"line products {sign}; consistent assignments {glob}; 6 states: rung R, lam* 0")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "npa_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
