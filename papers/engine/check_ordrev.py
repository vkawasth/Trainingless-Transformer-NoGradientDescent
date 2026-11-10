"""Checks for 'Ordered co-revision' (companion note; module pce/ordrev.py). Exact enumeration; no data, no fitting.
IDENTITY = an exact instance of a statement proved or cited in the text; FINDING = a computed fact about the enumerated families.
Family T (three sources, pairs 1 = {1,2}, 2 = {2,3}): every ordering of two adoptions and two withdrawals on each pair (2520 histories).
Family F (four sources, pairs 1, 2, 3 on a line): every ordering of one adoption and one withdrawal on each pair (720 histories), and every
ordering of two adoptions on each pair (90 positive histories).
  OR1 IDENTITY  the geometric realisation (clockwise full turns about midpoints for adoption, unit spacing) gives the algebraic braid
  OR2 IDENTITY  family T shares counts, co-revision membership, net windings and pairwise linking numbers (one class)
  OR3 IDENTITY  Milnor's triple invariant equals the signed area of the cumulative count path (level-2 signature): 9 classes
  OR4 IDENTITY  braid classes = free-reduction classes = full-signature classes (A_12, A_23 generate a free group; the signature
                determines a path up to backtracking); in family T the level-5 truncation already separates all 297
  OR5 IDENTITY  degree-3 Magnus coefficients of the longitudes and the level-3 signature induce the same partition (195 classes)
  OR6 FINDING   the closure polynomials (Jones, Alexander) induce 10 classes, not comparable with the triple invariant, refined by the
                level-4 signature but not the level-3 one
  OR7 IDENTITY  family F positive histories: two histories have the same braid iff they have the same trace (Cartier-Foata normal form
                under the independence of disjoint pairs); the path signature, which ignores independence, separates more
  OR8 FINDING   family F signed histories: the signature partition strictly refines the braid partition; the braid is the signature
                modulo the swapping of independent events
"""
import os, sys, itertools, json
import numpy as np, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import ordrev as OR, bundlebraid as BB, braidknot as BK
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def part(H, f):
    c = {}
    for h in H: c.setdefault(f(h), []).append(h)
    return c
def refines(a, b):
    lab = {h: i for i, v in enumerate(b.values()) for h in v}
    return all(len({lab[h] for h in v}) == 1 for v in a.values())
def same(a, b): return refines(a, b) and refines(b, a)

T = OR.arrangements({(1, 1): 2, (1, -1): 2, (2, 1): 2, (2, -1): 2})
F = OR.arrangements({(1, 1): 1, (1, -1): 1, (2, 1): 1, (2, -1): 1, (3, 1): 1, (3, -1): 1})
Fp = OR.arrangements({(1, 1): 2, (2, 1): 2, (3, 1): 2})

# OR1
rng = np.random.default_rng(1); ok = True; n = 0
for fam, k in ((T, 3), (F, 4)):
    for idx in rng.choice(len(fam), 60, replace=False):
        h = fam[idx]; P = OR.realise(h, k); w = BB.braid(P + 1e-7 * np.exp(1j * np.arange(k))[None, :], 0.123)[0]
        ok &= BK.artin(w, k) == BK.artin(OR.braid_word(h), k); n += 1
check("OR1 IDENTITY the geometric realisation gives the algebraic braid", ok, f"{n} histories (60 from each family); a clockwise turn is A = sigma^2 under the reading convention of pce/bundlebraid")

# OR2
cnt = part(T, lambda h: tuple(sorted(h))); lk = part(T, lambda h: tuple(sorted((i, tuple(sorted(d.items()))) for i, d in OR.milnor3(h)[0].items())))
check("OR2 IDENTITY family T: counts, membership, net windings and linking numbers are the same for all 2520 orderings", len(T) == 2520 and len(cnt) == 1 and len(lk) == 1,
      f"{len(T)} histories; classes by counts {len(cnt)}, by linking numbers {len(lk)}")

# OR3
area = lambda h: (lambda S: (S[2][0, 1] - S[2][1, 0]) / 2)(OR.signature(h, 2, 2))
mu = part(T, lambda h: OR.milnor3(h)[1][3][0]); s2 = part(T, lambda h: OR.sig_key(h, 2, 2))
check("OR3 IDENTITY triple invariant = signed area of the count path = level-2 signature", all(OR.milnor3(h)[1][3][0] == round(area(h)) and abs(area(h) - round(area(h))) < 1e-9 for h in T) and same(mu, s2),
      f"classes {len(mu)}; values {sorted(mu)}")

# OR4
br = part(T, lambda h: OR.braid_key(h, 3)); fr = part(T, OR.free_reduce); sig = {L: part(T, lambda h, L=L: OR.sig_key(h, 2, L)) for L in (3, 4, 5, 8)}
check("OR4 IDENTITY braid = free reduction = full signature; level 5 suffices in family T", same(br, fr) and same(br, sig[8]) and same(br, sig[5]) and len(br) == 297,
      f"classes: braid {len(br)}, free reduction {len(fr)}, signature levels 3, 4, 5, 8: {[len(sig[L]) for L in (3, 4, 5, 8)]}")

# OR5
mg = part(T, lambda h: OR.magnus_key(h, 3))
check("OR5 IDENTITY degree-3 Magnus coefficients of the longitudes and the level-3 signature give the same partition", same(mg, sig[3]), f"{len(mg)} classes")

# OR6
reps = {k: v[0] for k, v in br.items()}
A = {k: str(OR.alexander3(h)) for k, h in reps.items()}; J = {k: str(OR.jones3(h)) for k, h in reps.items()}
cl = part(T, lambda h: (A[OR.braid_key(h, 3)], J[OR.braid_key(h, 3)])); jo = part(T, lambda h: J[OR.braid_key(h, 3)])
ok = len(cl) == 10 and not refines(cl, mu) and not refines(mu, cl) and refines(sig[4], cl) and not refines(sig[3], cl) and refines(br, cl)
check("OR6 FINDING closure polynomials: 10 classes, not comparable with the triple invariant, functions of the level-4 signature", ok,
      f"Jones classes {len(jo)}, Jones and Alexander together {len(cl)}")
json.dump({"family_T": {"histories": len(T), "counts": len(cnt), "triple": len(mu), "sig3": len(sig[3]), "sig4": len(sig[4]), "braid": len(br), "closure_polys": len(cl)}},
          open(os.path.join(HERE, "ordrev_summary.json"), "w"), indent=1)

# OR7
bp = part(Fp, lambda h: OR.braid_key(h, 4)); tp = part(Fp, OR.trace_key); sp_ = part(Fp, lambda h: OR.sig_key(h, 3, 6))
check("OR7 IDENTITY four sources, positive histories: same braid iff same trace; the signature separates more", same(bp, tp) and refines(sp_, bp) and len(sp_) > len(bp),
      f"{len(Fp)} histories; braid classes {len(bp)}, traces {len(tp)}, signature classes {len(sp_)}")

# OR8
bf = part(F, lambda h: OR.braid_key(h, 4)); sf = part(F, lambda h: OR.sig_key(h, 3, 6))
split = sum(1 for v in bf.values() if len({OR.sig_key(h, 3, 6) for h in v}) > 1)
check("OR8 FINDING four sources, signed histories: the signature strictly refines the braid (it ignores independence)", refines(sf, bf) and len(sf) > len(bf),
      f"{len(F)} histories; braid classes {len(bf)}, signature classes {len(sf)}; braid classes split by the signature {split}")
json.load(open(os.path.join(HERE, "ordrev_summary.json")))
s = json.load(open(os.path.join(HERE, "ordrev_summary.json"))); s["family_F"] = {"signed": len(F), "braid": len(bf), "signature": len(sf), "positive": len(Fp), "positive_braid": len(bp), "positive_traces": len(tp), "positive_signature": len(sp_)}
json.dump(s, open(os.path.join(HERE, "ordrev_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
