"""scan_invariant.py -- the second-order birth invariant Lambda_2 (pce/birth2.py) at every class of classical points: python scan_invariant.py c5n4
(Rep_4 of the 5-cycle, about 20 minutes; needs ../level2/c5_scan_n4.json from scan_c5.py) or python scan_invariant.py c4 (Rep_4 of the 4-cycle, about
3 minutes). Writes ../level2/c5_invariant_n4.json or ../level2/c4_invariant_n4.json."""
import sys, os, json, time, itertools, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce.birth2 import setup, invariant, first_order
from pce.coxeter import diagonal_points, analyse
which = sys.argv[1]; rng = np.random.default_rng(0); t = time.time(); out = []
if which == "c5n4":
    R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "level2", "c5_scan_n4.json"))); S = setup(5, 4)
    for i, r in enumerate(R):
        X0 = [np.diag([float(r["class"][j][v]) for j in range(4)]) for v in range(5)]
        inv = invariant(S, X0, 150, rng); fo = first_order(S, X0)
        out.append({"class": r["class"], "size": r["size"], "birth": r["local_max_witness"] > 1e-6 or r["random_ctx_share"] > 0,
                    "singular": r["singular"], "Lambda2": inv["Lambda2"], "first_order": fo, "smooth2": inv["smooth"], "localmax": r["local_max_witness"]})
        if i % 20 == 0: print(i, f"{time.time()-t:.0f}s", flush=True)
else:   # C4 control, Rep_4: classes up to D4 x sign flips x reordering; birth by the exact block criterion
    k = 4; G = (list("abcd"), [(i, (i + 1) % k) for i in range(k)]); S = setup(4, 4)
    A = lambda u: (u[0], u[2]); B = lambda u: (u[1], u[3])
    def birth(key):
        ch = [tuple(c) for c in key]
        for quad in itertools.combinations(range(len(ch)), 4):
            Sq = [ch[m] for m in quad]
            if len(set(Sq)) < 4: continue
            for p in itertools.permutations(Sq):
                if A(p[0]) == A(p[2]) and A(p[1]) == A(p[3]) and B(p[0]) == B(p[1]) and B(p[2]) == B(p[3]) and \
                   tuple(np.multiply(A(p[0]), A(p[1]))) == (-1, -1) and tuple(np.multiply(B(p[0]), B(p[2]))) == (-1, -1): return True
        return False
    def canon(key):
        best = None; cols = [tuple(int(x) for x in c) for c in key]
        for r_ in range(k):
            for refl in (False, True):
                perm = [((-i if refl else i) + r_) % k for i in range(k)]
                for fl in itertools.product((1, -1), repeat=k):
                    cc = tuple(sorted(tuple(fl[v] * c[perm[v]] for v in range(k)) for c in cols))
                    if best is None or cc < best: best = cc
        return best
    reps = {}; sizes = {}
    for X, key in diagonal_points(G, 4):
        c = canon(key); reps.setdefault(c, (X, key)); sizes[c] = sizes.get(c, 0) + 1
    for i, (c, (X, key)) in enumerate(reps.items()):
        inv = invariant(S, X, 150, rng); fo = first_order(S, X)
        out.append({"class": [list(x) for x in c], "size": sizes[c], "birth": birth(key), "singular": analyse(G, X, 3, rng)["obstructed"],
                    "Lambda2": inv["Lambda2"], "first_order": fo, "smooth2": inv["smooth"]})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "level2", {"c5n4": "c5_invariant_n4.json", "c4": "c4_invariant_n4.json"}[which]), "w"), default=float)
pred = [o["Lambda2"] > 1e-6 for o in out]; obs = [o["birth"] for o in out]
print(which, "classes", len(out), "agreement", sum(p == q for p, q in zip(pred, obs)), "| birth", sum(obs), "| predicted", sum(pred),
      "| max first order", max(o["first_order"] for o in out), f"| {time.time()-t:.0f}s", flush=True)
print("Lambda2 at birth classes:", sorted({round(o["Lambda2"], 4) for o in out if o["birth"]}), "| at others max:", max([o["Lambda2"] for o in out if not o["birth"]] or [0]))
