"""check_invariant.py -- the second-order birth invariant (note: level2/partial_gluing.pdf, Section 15).

At a classical point X0 of a cycle's Hermitian representation scheme, the odd-sign cycle operators are diagonal with top eigenvalue n - 2 on
the characters that saturate them. Along a curve X(t) = X0 + t Y + t^2 Y2 (Y an extendable tangent vector, Y2 its second-order lift) the top
eigenvalue is (n - 2) + t lambda_1 + t^2 lambda_2 + ..., with lambda_1 the top eigenvalue of P B1 P and lambda_2 that of
P B2 P + P B1 (lambda0 - B0)^+ B1 P (degenerate perturbation theory). Lambda_2 = max over odd sign patterns and unit extendable Y.

  INV1 the first-order term vanishes at every classical point (CHSH Rep_4: 67 classes; KCBS Rep_3: 33; KCBS Rep_4: 242)
  INV2 Lambda_2 > 0 exactly at the birth classes for CHSH (Rep_4: Lambda_2 = 1/2 at the one birth class) and KCBS (Rep_3: Lambda_2 = 3/4 at
       the one birth class), and Lambda_2 equals the observed constant of the witness growth (local maximum / eps^2)
  INV3 KCBS Rep_4: Lambda_2 > 0 implies birth (no false positive among the 213 classes without birth), and every birth class with
       Lambda_2 = 0 is a higher-order birth: the largest witness of sampled nearby models scales like d^p with p clearly above 2 (third
       or fourth order; estimated from random samples, so p is approximate)
Run: python check_invariant.py (Rep_3 and CHSH live; KCBS Rep_4 from ../level2/c5_invariant_n4.json, written by scan_invariant.py c5n4)
"""
import json, os, sys, warnings
import numpy as np
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.birth2 import setup, invariant, first_order

rng = np.random.default_rng(20261026); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
L2 = lambda p: json.load(open(os.path.join(HERE, "..", "level2", p)))

# KCBS Rep_3, live
S3 = setup(5, 3); R3 = L2("c5_scan_n3.json"); rows3 = []
for r in R3:
    X0 = [np.diag([float(r["class"][j][v]) for j in range(3)]) for v in range(5)]
    inv = invariant(S3, X0, 200, rng)
    rows3.append({"birth": r["local_max_witness"] > 1e-6 or r["random_ctx_share"] > 0, "Lambda2": inv["Lambda2"], "first": first_order(S3, X0),
                  "observed": r["local_max_witness"] / 0.01})
C4 = L2("c4_invariant_n4.json"); C5 = L2("c5_invariant_n4.json")

# INV1
fo = max([x["first"] for x in rows3] + [x["first_order"] for x in C4] + [x["first_order"] for x in C5])
check("INV1 the first-order term vanishes at every classical point (CHSH Rep_4, KCBS Rep_3 and Rep_4)", fo < 1e-10,
      f"{len(C4)} + {len(rows3)} + {len(C5)} classes; largest first-order coefficient {fo:.1e}")

# INV2
ag3 = sum((x["Lambda2"] > 1e-6) == x["birth"] for x in rows3); ag4 = sum((x["Lambda2"] > 1e-6) == x["birth"] for x in C4)
b3 = [x for x in rows3 if x["birth"]]; b4 = [x for x in C4 if x["birth"]]
ok = ag3 == len(rows3) and ag4 == len(C4) and all(abs(x["Lambda2"] - 0.75) < 1e-3 for x in b3) and all(abs(x["Lambda2"] - 0.5) < 1e-3 for x in b4) \
     and all(abs(x["Lambda2"] - x["observed"]) < 0.02 for x in b3)
OUT["rep3"] = rows3
check("INV2 Lambda_2 > 0 exactly at the birth classes: CHSH Rep_4 (1/2) and KCBS Rep_3 (3/4); it matches the observed growth constant", ok,
      f"agreement CHSH {ag4}/{len(C4)}, KCBS Rep_3 {ag3}/{len(rows3)}; KCBS: Lambda_2 {b3[0]['Lambda2']:.4f} vs local max / eps^2 {b3[0]['observed']:.4f}")

# INV3
fp = [x for x in C5 if x["Lambda2"] > 1e-6 and not x["birth"]]; second = [x for x in C5 if x["Lambda2"] > 1e-6]
hi = [x for x in C5 if x["birth"] and x["Lambda2"] <= 1e-6]; nb = [x for x in C5 if not x["birth"]]
S4 = setup(5, 4); exps = []
def wit(X):
    return max(np.linalg.eigvalsh((lambda B: (B + B.T) / 2)(sum(g[i] * X[i] @ X[(i + 1) % 5] for i in range(5)))).max() for g in S4.G) - 3
for o in hi:
    X0 = [np.diag([float(o["class"][j][v]) for j in range(4)]) for v in range(5)]; z0 = S4.pack(X0); pts = []
    for eps in (0.2, 0.1, 0.05, 0.025):
        best = (0.0, 0.0)
        for _ in range(250):
            z = least_squares(S4.F, z0 + eps * rng.normal(size=z0.size) / np.sqrt(len(z0)), xtol=1e-15, ftol=1e-15, gtol=1e-15).x
            if np.linalg.norm(S4.F(z)) < 1e-10:
                w = wit(S4.unpack(z))
                if w > best[0]: best = (w, np.linalg.norm(z - z0))
        if best[0] > 0: pts.append(best)
    d = np.log([p[1] for p in pts]); w = np.log([p[0] for p in pts]); exps.append(float(np.polyfit(d, w, 1)[0]))
OUT["rep4"] = {"second_order_births": len(second), "false_positives": len(fp), "higher_order_births": len(hi), "exponents": exps,
               "Lambda2_values": sorted({round(x["Lambda2"], 4) for x in second})}
check("INV3 KCBS Rep_4: Lambda_2 > 0 implies birth; the other birth classes are higher order (witness ~ d^p, p > 2.5)",
      len(fp) == 0 and len(second) > 0 and len(hi) > 0 and all(2.5 < p < 5.0 for p in exps),
      f"{len(second)} second-order birth classes (Lambda_2 in {sorted({round(x['Lambda2'], 3) for x in second})}), 0 false positives among "
      f"{len(nb)} classes without birth; {len(hi)} higher-order birth classes with exponents {', '.join(f'{p:.2f}' for p in exps)}")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "invariant_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
