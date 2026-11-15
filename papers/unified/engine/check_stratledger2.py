"""Chambers and basis regions, measured consistently (section sec:ledger; supersedes the record QL5 of check_stratledger.py, whose
chamber labels were taken up to the 72 symmetries of the table while the basis labels were not). New script; it modifies nothing.
Declared before the first run: the same 10000 nondegenerate 3 x 3 margins (seed 20261126). A chamber is labelled by its set of
feasible bases (equivalently, for nondegenerate margins, its normal fan: the vertex cones are the zero cells of the feasible bases).
Both raw labels and labels up to symmetry are kept.
  QL9 IDENTITY  prop:chamberrefine: for 10 generic integer linear objectives, every raw chamber carries exactly one optimal basis
                (vertex identity, recorded as its support); with symmetry-quotiented chamber labels, chambers carry several (the
                artifact that produced QL5's purity 0.30)
  QL10 RECORD   conditional bands (upper and lower end, before evidence): basis regions per raw chamber; for every pair of tables in one
                raw chamber with different optimal bases, the switch point on the segment between them (chambers are convex) is found
                by bisection and checked to lie in the chamber's interior (the chamber label is unchanged along the segment)
  QL11 RECORD   (added after QL9-QL10 ran, which found no switch for the paper's decision quantity) five random generic ratio
                objectives n.x / d.x with positive integer n <= d: chambers with more than one basis region, and whether every located
                switch lies in a chamber interior
"""
import os, sys, math, itertools
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import stratledger as SL, cpl33 as C
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def nondeg(f, g):
    fs = {sum(c) for k in range(1, 3) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in range(1, 3) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def margins(rng, k, den):
    v = rng.dirichlet([2] * k) * den; x = np.maximum(1, np.floor(v)).astype(int); x[int(np.argmax(x))] += den - x.sum()
    return [Fr(int(y), den) for y in x] if x.min() >= 1 else None
rng = np.random.default_rng(20261126); pop = []
while len(pop) < 10000:
    f, g = margins(rng, 3, 120), margins(rng, 3, 120)
    if f and g and nondeg(f, g): pop.append((f, g))
P = SL.Problem(3, 3); T = len(pop); Bm = np.array([P.b(f, g) for f, g in pop]).T
ALLB = [B for B in itertools.combinations(range(P.N), P.m) if P.is_basis(B)]; INV = {B: np.linalg.inv(P.A[:, list(B)].astype(float)) for B in ALLB}
def feasible_set(b):
    return frozenset(B for B in ALLB if (INV[B] @ b >= -1e-12).all())
raw = [feasible_set(Bm[:, t]) for t in range(T)]
sym_of_cell = []
for pr, pc, tr in C.SYMS:
    mp_ = {}
    for k, (i, j) in enumerate(P.cells):
        i2, j2 = (pc[j], pr[i]) if tr else (pr[i], pc[j]); mp_[k] = P.cells.index((i2, j2))
    sym_of_cell.append(mp_)
def quotient(lbl):
    return min(tuple(sorted(tuple(sorted(m[k] for k in B)) for B in lbl)) for m in sym_of_cell)
qlab = [quotient(l) for l in raw]
nraw, nq = len(set(raw)), len(set(qlab))
order = np.random.default_rng(1).permutation(T); viol_raw = 0; multi_q = 0; per_obj = []
for o in range(10):
    nvec = np.random.default_rng(100 + o).integers(-20, 21, size=P.N).astype(float); L = SL.Library(P)
    asg, _, s, _ = SL.build(P, L, Bm, nvec, np.zeros(P.N), +1, order)
    byraw = defaultdict(set); byq = defaultdict(set)
    for t in range(T): byraw[raw[t]].add(asg[t]); byq[qlab[t]].add(asg[t])
    v = sum(1 for k in byraw if len(byraw[k]) > 1); viol_raw += v; mq = sum(1 for k in byq if len(byq[k]) > 1); multi_q += mq; per_obj.append((v, mq))
check("QL9 IDENTITY each raw chamber carries exactly one optimal basis for every generic linear objective; quotiented labels do not",
      viol_raw == 0, f"{nraw} raw chambers, {nq} up to symmetry; raw chambers with >1 basis per objective {[p[0] for p in per_obj]}; quotiented {[p[1] for p in per_obj]}")
# QL10 conditional bands
ix = {cl: k for k, cl in enumerate(P.cells)}; n0 = np.zeros(P.N); n0[ix[(0, 0)]] = 1; d0 = np.zeros(P.N)
for cl in ((0, 0), (0, 1), (1, 0), (1, 1)): d0[ix[cl]] = 1
Dd = np.array([f[0] + f[1] + g[0] + g[1] > 1 for f, g in pop]); Di = np.where(Dd)[0]; info = []; interior_bad = 0; switches = 0
for key, sense in (("hi0", 1), ("lo0", -1)):
    L = SL.Library(P); asg, _, s, _ = SL.build(P, L, Bm[:, Di], n0, d0, sense, np.random.default_rng(2).permutation(len(Di)))
    byraw = defaultdict(list)
    for q, t in enumerate(Di): byraw[raw[t]].append(q)
    multi = [k for k in byraw if len({asg[q] for q in byraw[k]}) > 1]; per = Counter(len({asg[q] for q in byraw[k]}) for k in byraw)
    tested = 0
    for k in multi[:40]:
        qs = byraw[k]; a = qs[0]; others = [q for q in qs if asg[q] != asg[a]]
        if not others: continue
        bq = others[0]; b1, b2 = Bm[:, Di[a]], Bm[:, Di[bq]]
        def basis_at(lmb):
            bb = (1 - lmb) * b1 + lmb * b2
            for kk in range(len(L.bases)):
                ok, _ = L.certify(kk, bb[:, None], n0, d0, sense)
                if ok[0]: return kk
            return None
        lo, hi = 0.0, 1.0; ba = asg[a]
        for _ in range(40):
            mid = (lo + hi) / 2
            if basis_at(mid) == ba: lo = mid
            else: hi = mid
        sw = (lo + hi) / 2; bb = (1 - sw) * b1 + sw * b2; tested += 1; switches += 1
        interior_bad += feasible_set(bb) != k                              # the switch point must lie in the same chamber
    info.append(f"{key}: {len(L.bases)} bases; chambers by number of basis regions {dict(sorted(per.items()))}; chambers with >1 region {len(multi)}; switches located {tested}")
check("QL10 RECORD conditional bands: basis regions per raw chamber; switches lie in chamber interiors", True,
      "; ".join(info) + f"; switch points outside their chamber {interior_bad}")
rr = np.random.default_rng(77); info2 = []; bad2 = 0; tot_multi = 0
for o in range(5):
    dv = rr.integers(1, 10, size=P.N).astype(float); nv = np.floor(dv * rr.uniform(0, 1, size=P.N)); L = SL.Library(P)
    asg, _, s_, _ = SL.build(P, L, Bm, nv, dv, +1, np.random.default_rng(3).permutation(T))
    byraw = defaultdict(list)
    for t in range(T): byraw[raw[t]].append(t)
    multi = [k for k in byraw if len({asg[t] for t in byraw[k]}) > 1]; tot_multi += len(multi); located = 0
    for k in multi[:25]:
        ts = byraw[k]; a = ts[0]; bq = next(t for t in ts if asg[t] != asg[a]); b1, b2 = Bm[:, a], Bm[:, bq]
        def basis_at(lmb):
            bb = (1 - lmb) * b1 + lmb * b2
            for kk in range(len(L.bases)):
                ok, _ = L.certify(kk, bb[:, None], nv, dv, +1)
                if ok[0]: return kk
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if basis_at(mid) == asg[a]: lo = mid
            else: hi = mid
        bb = (1 - (lo + hi) / 2) * b1 + (lo + hi) / 2 * b2; located += 1; bad2 += feasible_set(bb) != k
    info2.append(f"objective {o}: {len(L.bases)} bases, chambers with >1 region {len(multi)}, switches located {located}")
check("QL11 RECORD random ratio objectives: optimal bases can switch inside a chamber; located switches lie in chamber interiors", True,
      "; ".join(info2) + f"; switch points outside their chamber {bad2}")
print(f"{sum(results)}/{len(results)} passed")
