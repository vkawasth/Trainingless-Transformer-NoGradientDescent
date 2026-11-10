"""Checks for 'Pooling and conditioning' (companion note; module pce/poolcond.py).
  PC1 IDENTITY  linear pooling: conditioning the pool equals pooling the conditioned sources with weights w_i P_i(E) / sum_j w_j P_j(E)
  PC2 IDENTITY  with fixed weights, linear pooling and conditioning commute exactly when the sources give the evidence equal probability;
                the example with P_A(E) = P_B(E) = 0.82 has defect 0; changing B's likelihoods to (0.3, 0.6) gives 0.604 against 0.520
  PC3 IDENTITY  geometric pooling commutes with conditioning when the pool conditions on the pooled likelihood (externally Bayesian)
  PC4 IDENTITY  the decision maker's own operation, intersecting the fibres of trusted cells, does not depend on the order of the cells
  PC5 FINDING   planted evidence: under pool-then-condition (Bayesian model averaging) the planting source takes all the weight and drives
                P(X) to 1 although X is false; fixed linear weights stay at 0.355 and geometric pooling reaches 0.842; without planting all
                three end near 0.005
"""
import os, sys, itertools, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import poolcond as PO, tamper as T, bridgetoy as BT, miscal as MC
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(3)
def rand_dist(m): x = rng.random(m) + 0.05; return x / x.sum()

# PC1
worst = 0.0
for _ in range(300):
    m = int(rng.integers(2, 6)); n = int(rng.integers(2, 5)); Ps = [rand_dist(m) for _ in range(n)]; w = rand_dist(n); liks = [rng.random(m) + 0.01 for _ in range(n)]
    _, after, _ = PO.defect(Ps, w, liks); alt = PO.linear([PO.condition(P, l) for P, l in zip(Ps, liks)], PO.reweighted(w, Ps, liks)); worst = max(worst, PO.tv(after, alt))
check("PC1 IDENTITY condition-after-pool = pool-after-condition with likelihood-reweighted weights", worst < 1e-12, f"300 random cases; largest difference {worst:.1e}")

# PC2
PA = np.array([0.9, 0.1]); PB = np.array([0.1, 0.9]); lA = np.array([0.9, 0.1]); lB = np.array([0.1, 0.9]); lB2 = np.array([0.3, 0.6])
d0 = PO.defect([PA, PB], [0.5, 0.5], [lA, lB])[0]; d1, a1, b1 = PO.defect([PA, PB], [0.5, 0.5], [lA, lB2])
eq_ok = True
for _ in range(200):                                                     # construct sources with equal P(E): commute; unequal: do not
    m = 3; Ps = [rand_dist(m) for _ in range(2)]; l0 = rng.random(m) + 0.01; l1 = rng.random(m) + 0.01
    l1 = l1 * PO.evidence_prob(Ps[0], l0) / PO.evidence_prob(Ps[1], l1)
    if l1.max() <= 1: eq_ok &= PO.defect(Ps, [0.4, 0.6], [l0, l1])[0] < 1e-12
    l2 = l1 * 0.7
    if PO.evidence_prob(Ps[1], l2) != PO.evidence_prob(Ps[0], l0): eq_ok &= PO.defect(Ps, [0.4, 0.6], [l0, l2])[0] > 1e-9
check("PC2 IDENTITY fixed-weight linear pooling commutes with conditioning exactly when P_i(E) are equal", d0 < 1e-12 and abs(a1[0] - 0.6043) < 1e-3 and abs(b1[0] - 0.5202) < 1e-3 and eq_ok,
      f"memo example: P_A(E) = P_B(E) = {PO.evidence_prob(PA, lA):.2f}, defect {d0:.1e}; modified: P_B(E) = {PO.evidence_prob(PB, lB2):.2f}, P(X | E) {a1[0]:.4f} (pool first) vs {b1[0]:.4f} (condition first), defect {d1:.4f}")

# PC3
worst = 0.0
for _ in range(300):
    m = int(rng.integers(2, 6)); n = int(rng.integers(2, 5)); Ps = [rand_dist(m) for _ in range(n)]; w = rand_dist(n); liks = [rng.random(m) + 0.01 for _ in range(n)]
    worst = max(worst, PO.defect(Ps, w, liks, "geometric")[0])
check("PC3 IDENTITY geometric pooling is externally Bayesian", worst < 1e-12, f"300 random cases, different likelihoods per source; largest defect {worst:.1e}")

# PC4
P = MC.ising(np.full(5, 0.7), np.array([0.15, 0, 0, 0, 0])); cells = list(MC.reports(P).values()); g = BT.world()[2]; I0 = T.interval(5, cells, g); worst = 0.0
for perm in itertools.permutations(range(len(cells))):
    I = T.interval(5, [cells[i] for i in perm], g); worst = max(worst, abs(I[0] - I0[0]) + abs(I[1] - I0[1]))
check("PC4 IDENTITY intersecting fibres is order-free", worst < 1e-9, f"all 120 orders of the five cycle reports; largest change of the range of B {worst:.1e}")

# PC5
op = PO.stream(planted=True); on = PO.stream(planted=False)
json.dump(dict(planted=op, honest=on), open(os.path.join(HERE, "poolcond_stream.json"), "w"), indent=1)
check("PC5 FINDING planted evidence captures the weight under pool-then-condition and flips P(X); fixed and geometric pooling resist less or more",
      op["wK"][-1] > 0.99 and op["bma"][-1] > 0.99 and abs(op["fixed"][-1] - 0.355) < 0.01 and abs(op["geometric"][-1] - 0.842) < 0.01 and max(on["bma"][-1], on["fixed"][-1], on["geometric"][-1]) < 0.01,
      f"planted: weight of K {op['wK'][-1]:.3f}, P(X) BMA {op['bma'][-1]:.3f}, fixed {op['fixed'][-1]:.3f}, geometric {op['geometric'][-1]:.3f}; without planting: weight of K {on['wK'][-1]:.3f}, P(X) {on['bma'][-1]:.3f}, {on['fixed'][-1]:.3f}, {on['geometric'][-1]:.3f}")
print(f"{sum(results)}/{len(results)} passed")
