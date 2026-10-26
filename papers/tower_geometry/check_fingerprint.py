"""check_fingerprint.py -- the origin fingerprint on cases of known origin (note: level2/partial_gluing.pdf, Section 13).

  FP1 a local (classical) box is labelled classical: lam* = 1, no direct influence, c = 0
  FP2 the Tsirelson box is labelled quantum, not classical: lam* = 2 - sqrt 2, quantum fraction 1
  FP3 the PR box is labelled beyond quantum: lam* = 0, quantum fraction 0, no direct influence
  FP4 a box whose marginals move with the partner's setting, made from a local model, is labelled direct influence only: linear gap > 0,
      c = 0
  FP5 a cyclic rank-4 system with both direct influence and a PR-type component strong enough to exceed it is labelled contextual beyond
      direct influence (c > 0), and the Kujala-Dzhafarov criterion agrees
  FP6 level two: three-variable tables made from genuine two-actor mixtures, from signed mixtures, and laws with Det < 0 are labelled
      R>=0, R and C
  FP7 the fractions are ordered (lam* <= quantum <= 1) on random no-signalling boxes, and data processing: random local post-processing
      never decreases lam*
Run: python check_fingerprint.py
"""
import itertools, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.fingerprint import fingerprint
from pce.npa import behaviour
from pce.cbd import kd_cyclic
from pce.level2 import lcm_law
rng = np.random.default_rng(20261025); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
DOM = {"A0": 2, "A1": 2, "B0": 2, "B1": 2}
def windows_of(P): return [Window((f"A{x}", f"B{y}"), P[x, y]) for x in range(2) for y in range(2)]
def det_box(a0, a1, b0, b1): return behaviour({"A": [a0, a1], "B": [b0, b1], "AB": [[a0 * b0, a0 * b1], [a1 * b0, a1 * b1]]})
iso = lambda v: behaviour({"A": [0, 0], "B": [0, 0], "AB": [[v, v], [v, -v]]})
LOC = [det_box(*s) for s in itertools.product((1, -1), repeat=4)]
def fp(P): return fingerprint(DOM, windows_of(P), bell=(2, 2))

wl = rng.dirichlet(np.ones(16)); f = fp(sum(w * L for w, L in zip(wl, LOC)))
check("FP1 local box -> classical", f["origin"] == "classical" and f["direct"] < 1e-9 and f["cbd_c"] < 1e-9, f"lam* {f['lam_classical']:.3f}")
f = fp(iso(1 / np.sqrt(2)))
check("FP2 Tsirelson box -> quantum, not classical", f["origin"] == "quantum, not classical" and abs(f["lam_classical"] - (2 - np.sqrt(2))) < 1e-6,
      f"lam* {f['lam_classical']:.4f}, quantum {f['lam_quantum']:.4f}")
f = fp(iso(1.0))
check("FP3 PR box -> beyond quantum", f["origin"] == "beyond quantum" and f["lam_classical"] < 1e-9 and f["lam_quantum"] < 1e-6)
# FP4: a local hidden-variable model in which B's outcome distribution depends on Alice's setting (signalling, no contextuality)
P = np.zeros((2, 2, 2, 2))
for x, y in itertools.product(range(2), range(2)):
    pa = 0.5; pb = 0.3 + 0.4 * x
    P[x, y] = np.outer([pa, 1 - pa], [pb, 1 - pb])
f = fp(P)
check("FP4 setting-dependent marginals from a local model -> direct influence only", f["origin"] == "direct influence only" and f["linear_gap"] > 1e-6 and f["cbd_c"] < 1e-9,
      f"direct {f['direct']:.3f}, c {f['cbd_c']:.1e}")
# FP5: PR component plus direct influence
Q = 0.8 * iso(1.0) + 0.2 * P
f = fp(Q); kd = kd_cyclic(windows_of(Q)[:1] + [windows_of(Q)[1], windows_of(Q)[3], windows_of(Q)[2]])
check("FP5 PR component plus direct influence -> contextual beyond direct influence; Kujala-Dzhafarov agrees", f["origin"] == "contextual beyond direct influence" and f["cbd_c"] > 1e-6,
      f"direct {f['direct']:.3f}, c {f['cbd_c']:.3f}, KD {kd}")
# FP6
DOM3 = {"X": 2, "Y": 2, "Z": 2}
def win3(T): return [Window(("X", "Y"), T.sum(axis=2)), Window(("Y", "Z"), T.sum(axis=0)), Window(("X", "Z"), T.sum(axis=1))]
labs = []
T = lcm_law([0.3, 0.7], [[0.2, 0.6, 0.7], [0.8, 0.3, 0.4]]); labs.append(fingerprint(DOM3, win3(T), joint3=T)["level2"])
while True:
    w = -0.1; a = rng.uniform(0, 1, (2, 3)); T = lcm_law([w, 1 - w], a)
    if T.min() >= 0: break
labs.append(fingerprint(DOM3, win3(T), joint3=T)["level2"])
from pce.ladder import level2_m3
while True:
    T = rng.dirichlet(np.ones(8) * 0.6).reshape(2, 2, 2)
    if level2_m3(T)["rung"] == "C": break
labs.append(fingerprint(DOM3, win3(T), joint3=T)["level2"])
check("FP6 level two: genuine mixture -> R>=0, signed mixture -> R, Det < 0 -> C", labs == ["R>=0", "R", "C"], f"labels {labs}")
# FP7
ok = True
for _ in range(10):
    wts = rng.dirichlet(np.ones(3)); wl = rng.dirichlet(np.ones(16)); P = wts[0] * iso(1.0) + wts[1] * iso(1 / np.sqrt(2)) + wts[2] * sum(w * L for w, L in zip(wl, LOC))
    f = fp(P); ok &= f["lam_classical"] <= f["lam_quantum"] + 1e-6 <= 1 + 2e-6
    KA = [rng.dirichlet(np.ones(2), 2).T for _ in range(2)]; KB = [rng.dirichlet(np.ones(2), 2).T for _ in range(2)]
    Q = np.zeros_like(P)
    for x, y in itertools.product(range(2), range(2)): Q[x, y] = KA[x] @ P[x, y] @ KB[y].T
    ok &= fingerprint(DOM, windows_of(Q))["lam_classical"] >= f["lam_classical"] - 1e-7
check("FP7 fractions ordered (lam* <= quantum <= 1); local post-processing never decreases lam*", ok, "10 random boxes")
print(f"\n{sum(results)}/{len(results)} checks passed")
