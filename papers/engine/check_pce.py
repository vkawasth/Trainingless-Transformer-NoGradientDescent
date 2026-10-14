"""check_pce.py -- checks for the Posterior Coherence Engine (pce).

  P1  three-token model of the paper (Prop. 16.1): incoherent iff eps < 1/3; coherent and non-unique for 1/3 < eps < 1, with fibre
      width 2 min(a, b), a = (3eps-1)/4, b = (1-eps)/4; unique at eps = 1/3 (61 grid values)
  P2  certificates are valid: for 300 random incoherent families, the certificate is >= 0 on every deterministic assignment (hence on
      every global law) and negative on the data; and the radius is 0 exactly for families generated from a global law (300)
  P3  the three-level split is exact: total = between + within + noise, all terms >= 0 (500 random hierarchies)
  P4  pooling: the defect is 0 for the Markov glue and positive for generic laws; on the three-token section J it equals
      |2 eps - 1| / (2 eps) (Prop. 16.2)
  P5  KL ball: the radius reproduces the companion note's example (0.0424 bits for the independent coupling, 0 for the garbling)
  P6  decisions: incoherent evidence is flagged, wide non-unique evidence is flagged when asked, coherent evidence is answered or
      withheld by the score threshold
Run: python check_pce.py   (from the engine folder)
"""
import itertools, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce import Window, coherence, split, defect, markov_glue, kl_ball, decide

rng = np.random.default_rng(20261014)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

DOM3 = {"X0": 2, "X1": 2, "X2": 2}
def three(eps):
    t = np.array([[eps / 2, (1 - eps) / 2], [(1 - eps) / 2, eps / 2]])
    return [Window(("X0", "X1"), t), Window(("X1", "X2"), t), Window(("X0", "X2"), t)]

# P1
ok = True; worst = 0.0
for k in range(1, 62):
    eps = k / 62; r = coherence(DOM3, three(eps))
    a, b = (3 * eps - 1) / 4, (1 - eps) / 4
    if eps < 1 / 3 - 1e-12: ok &= r.status == "incoherent"
    else:
        ok &= r.status in ("coherent-nonunique", "coherent-unique"); worst = max(worst, abs(r.width - 2 * min(a, b)))
r13 = coherence(DOM3, three(1 / 3))
check("P1 three-token: incoherent iff eps < 1/3; width 2 min(a, b) above; unique at eps = 1/3", ok and worst < 1e-7 and r13.status == "coherent-unique",
      f"max width error {worst:.1e}")

# P2
def rand_windows_from_law(P, pairs, names):
    return [Window((names[i], names[j]), P.sum(axis=tuple(k for k in range(P.ndim) if k not in (i, j)))) for i, j in pairs]
names = ["X0", "X1", "X2"]; pairs = [(0, 1), (1, 2), (0, 2)]
ok_cert = ok_zero = True; n_incoh = 0
for _ in range(300):
    P = rng.dirichlet(np.ones(8) * 0.5).reshape(2, 2, 2)
    ok_zero &= coherence(DOM3, rand_windows_from_law(P, pairs, names)).radius < 1e-9
    m = rng.uniform(0.1, 0.9, 3); tabs = []                               # random compatible pair tables (Frechet bounds)
    for i, j in pairs:
        lo, hi = max(0, m[i] + m[j] - 1), min(m[i], m[j]); p11 = rng.uniform(lo, hi)
        tabs.append(np.array([[1 - m[i] - m[j] + p11, m[j] - p11], [m[i] - p11, p11]]))
    wins = [Window((names[i], names[j]), t) for (i, j), t in zip(pairs, tabs)]
    r = coherence(DOM3, wins)
    if r.status == "incoherent":
        n_incoh += 1
        val = sum(float((r.certificate.get(w, 0) * wins[w].table).sum()) for w in range(3))
        for atom in itertools.product((0, 1), repeat=3):                  # deterministic assignments
            ok_cert &= sum(float(r.certificate.get(w, np.zeros((2, 2)))[atom[i], atom[j]]) for w, (i, j) in enumerate(pairs)) >= -1e-9
        ok_cert &= val < -1e-9
check("P2 certificates are >= 0 on every deterministic assignment and < 0 on the data; radius 0 for families from a global law",
      ok_cert and ok_zero and n_incoh > 20, f"{n_incoh} incoherent of 300 random compatible families")

# P3
ok = True
for _ in range(500):
    G = rng.integers(1, 5); K = rng.integers(2, 5)
    groups = [[(rng.uniform(0.2, 1), rng.dirichlet(np.ones(K))) for _ in range(rng.integers(1, 5))] for _ in range(G)]
    s = split(groups, rng.dirichlet(np.ones(G)))
    ok &= abs(s.total - (s.between + s.within + s.noise)) < 1e-10 and min(s.between, s.within, s.noise) > -1e-12
check("P3 three-level split: total = between + within + noise, all non-negative", ok, "500 random hierarchies")

# P4
ok = True
for _ in range(200):
    P = rng.dirichlet(np.ones(27)).reshape(3, 3, 3)
    c0, I0 = defect(markov_glue(P)); c1, I1 = defect(P)
    ok &= c0 < 1e-9 and abs(I0) < 1e-9 and c1 > 1e-6 and I1 > 1e-9
okJ = True
for eps in (0.4, 0.45, 0.6, 0.75, 0.9):
    a, b = (3 * eps - 1) / 4, (1 - eps) / 4
    J = np.array([[[a, b], [b, b]], [[b, b], [b, a]]])                      # J[x0, x1, x2]: a on 000 and 111, b elsewhere
    okJ &= abs(defect(J)[0] - abs(2 * eps - 1) / (2 * eps)) < 1e-12
check("P4 defect 0 for the Markov glue, positive for generic laws; equals |2eps-1|/(2eps) on the section J", ok and okJ)

# P5
def joint(coupling):
    P = np.zeros((2, 2, 2))
    for x, a, b in itertools.product((0, 1), repeat=3):
        pa = 0.8 if a == x else 0.2
        pb = (0.75 if b == a else 0.25) if coupling == "g" else (0.65 if b == x else 0.35)
        P[x, a, b] = 0.5 * pa * pb
    return P
Ii, _ = kl_ball(joint("i")); Ig, _ = kl_ball(joint("g"))
check("P5 KL-ball radius: 0.0424 bits for the independent coupling, 0 for the garbling (companion Example 8)",
      abs(Ii / math.log(2) - 0.0424) < 5e-4 and abs(Ig) < 1e-12, f"{Ii / math.log(2):.4f} bits")

# P6
s = split([[(1, [0.9, 0.1])], [(1, [0.8, 0.2])]])
d1 = decide(s, coherence(DOM3, three(0.25)))
d2 = decide(s, coherence(DOM3, three(0.5)), width_tol=0.01)
d3 = decide(s, coherence(DOM3, three(0.5)), score=0.9, threshold=0.5)
d4 = decide(s, None, score=0.2, threshold=0.5)
check("P6 decisions: flag incoherent, flag wide non-unique on request, answer when confident, abstain below threshold",
      d1.action == "flag-incoherent" and d2.action == "flag-nonunique" and d3.action == "answer" and d4.action == "abstain")

print(f"\n{sum(results)}/{len(results)} checks passed")
