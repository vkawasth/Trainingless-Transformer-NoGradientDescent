"""Checks for the masking proposition (module pce/masking.py). Depth of a newcomer's defect = the lowest layer of invariants that sees it
(lowest nonzero Magnus degree = lower-central-series depth).
  HD1 IDENTITY  exhaustive search over all reduced words of length <= 14 in F_2: the shortest nontrivial word of depth >= k has length
                1, 4, 8, 14, 14 for k = 1..5 (none of depth >= 6 up to length 14); all are >= k (Malestein-Putman); depths confirmed by an
                independent Magnus computation; each word is realised as a newcomer's history whose defect has exactly that loop word
  HD2 IDENTITY  the defect is the product of the newcomer's events, each conjugated by the old history up to that moment
                (Delta = prod_i c_i x_{j_i}^{+-1} c_i^{-1}); its depth does not change when the newcomer's whole activity is reconjugated by
                any old history
  HD3 IDENTITY  a lone newcomer (one block of newcomer events between old histories) has the depth of its own event word, so hiding below
                layer k with a nontrivial defect needs at least as many events as the shortest word of depth >= k
  HD4 IDENTITY  riding: turn once around source j, let the old sources braid (w), undo the turn: depth(Delta) = 1 + depth(longitude of j
                in w), and Delta = 1 iff that longitude is trivial
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import masking as MK, stranddefect as S, braidknot as BK
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
inv, red = BK._inv, BK._red
rng = np.random.default_rng(7)

# HD1
best = MK.shortest_by_depth(2, 14, 5)
d = {k: min(len(v) for kk, v in best.items() if kk >= k) for k in range(1, 6)}
ok = d == {1: 1, 2: 4, 3: 8, 4: 14, 5: 14} and 6 not in best and all(d[k] >= k for k in d)
for k, w in best.items():
    M = BK.magnus(w, 6); nz = min(len(a) for a, c in M.items() if a and abs(c) > 1e-9); ok &= nz == k
    bp = S.from_loop_word(w, 2); ok &= S.loop_word(S.defect(bp, S.forget(bp, 3, 3), 2), 2) == w
check("HD1 IDENTITY shortest hidden newcomer words in F_2: lengths 1, 4, 8, 14, 14 for depths >= 1..5", ok,
      f"shortest by depth {d}; words (x, y, X = x^-1, Y = y^-1): " + str({k: "".join("xyXY"[[1, 2, -1, -2].index(g)] for g in w) for k, w in sorted(best.items())}))

# HD2
def rold(n, L):
    w = []
    for _ in range(L):
        i, j = sorted(rng.choice(np.arange(1, n + 1), 2, replace=False).tolist()); g = S.pure_gen(i, j); w += g if rng.random() < 0.5 else inv(g)
    return w
ok = True; cnt = 0
for n in (2, 3):
    for _ in range(25):
        bp = []
        for _ in range(int(rng.integers(2, 5))):
            if rng.random() < 0.5: bp.append(("old", rold(n, 1)))
            else: bp.append(("new", int(rng.integers(1, n + 1)), int(rng.choice([1, -1]))))
        word, old = MK.build(bp, n); f, prod = MK.conj_decomposition(bp, n)
        D = S.defect(word, S.forget(word, n + 1, n + 1), n); ok &= S.loop_word(D, n) == prod
        w = rold(n, 1); D2 = S.section_far(w, n) + D + inv(S.section_far(w, n)); lw2 = S.loop_word(D2, n)
        ok &= (MK.depth(prod, n, 5) if prod else 99) == (MK.depth(lw2, n, 5) if lw2 else 99); cnt += 1
check("HD2 IDENTITY defect = product of conjugated newcomer events; depth invariant under reconjugation by old histories", ok, f"{cnt} random histories")

# HD3
ok = True
for k, w in best.items():
    for n in (2,):
        for _ in range(3):
            b1 = rold(n, 1); b2 = rold(n, 1)
            bp = S.section_far(b1, n) + S.from_loop_word(w, n) + S.section_far(b2, n)
            D = S.defect(bp, S.forget(bp, n + 1, n + 1), n); ok &= MK.depth(S.loop_word(D, n), n, 6) == MK.depth(w, n, 6)
check("HD3 IDENTITY lone newcomer between old histories: defect depth = depth of its own event word", ok)

# HD4
A = lambda i, j: S.pure_gen(i, j); com = lambda u, v: u + v + inv(u) + inv(v)
olds = {"A12": A(1, 2), "[A12,A23]": com(A(1, 2), A(2, 3)), "[A13,[A12,A23]]": com(A(1, 3), com(A(1, 2), A(2, 3))), "[A12,[A12,A23]]": com(A(1, 2), com(A(1, 2), A(2, 3)))}
rows = []; ok = True
for name, w in olds.items():
    L = BK.longitudes(w, 3)
    for j in (1, 2, 3):
        _, prod = MK.conj_decomposition([("new", j, 1), ("old", w), ("new", j, -1)], 3)
        dD = MK.depth(prod, 3, 6) if prod else None; dl = MK.depth(L[j], 3, 6) if L[j] else None
        ok &= (dD is None and dl is None) or (dD is not None and dl is not None and dD == dl + 1); rows.append((name, j, dl, dD))
check("HD4 IDENTITY riding: two newcomer events reach depth 1 + depth(longitude of j in the old history)", ok, f"(old history, source, longitude depth, defect depth): {rows}")
json.dump(dict(shortest={str(k): v for k, v in best.items()}, riding=rows), open(os.path.join(HERE, "masking_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
