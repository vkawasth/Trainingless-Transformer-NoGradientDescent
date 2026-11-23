"""POST HOC (after the registered PW2 PASS; not a check): the thinnest PW2 match (ratio 0.907) is the same-t step from the frozen Study 81
configuration at t = 0.01 to the AMEND1/AMEND2 configuration (peg-free curves moved inside their cylinders). Subdivide that isotopy
(fractions 0, 1/8, ..., 1 of each curve's transverse move), match consecutive configurations by the registered rule, and compare the
composite map with the direct match used by PW2."""
import os, sys, itertools, numpy as np
E = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, E); os.chdir(E)
sys.argv = ["x", "0.01"]; os.environ["STUDY72_T"] = "0.01"
import wall81_bside as W, wall82 as X
from check_path82 import match
tc = W.taut_curves(W.transported()); T = W.T
targets = []
for q, tg in enumerate(tc["curves"]):
    X.POS = q; nt, sh = X.core(tg, T); targets.append((tg, nt))
def config(frac):
    cur = []
    for tg, nt in targets:
        d = dict(tg); d["verts"] = list(np.array(tg["verts"]) + frac * (np.array(nt["verts"]) - np.array(tg["verts"]))); cur.append(d)
    cv0 = W.g.build(cur, W.r0); cv = [cv0[q] for q in W.PERM]; pr = W.h.floer_chain(cv, T, W.tol)
    return {k: [complex(x["z"]) for x in pr[k]["points"]] for k in pr}
fr = np.linspace(0, 1, 9); C = [config(f) for f in fr]; PAIRS = list(itertools.combinations(range(6), 2))
comp = {pq: list(range(len(C[0][pq]))) for pq in PAIRS}; worst = 0; ok_all = True
for a, b in zip(C[:-1], C[1:]):
    for pq in PAIRS:
        m, d, s, ok = match(a[pq], b[pq], T.wa, T.wb, T.wa, T.wb); ok_all &= ok
        if np.isfinite(s) and s > 0: worst = max(worst, d / (0.5 * s))
        if m is not None: comp[pq] = [m[x] for x in comp[pq]]
direct = {pq: match(C[0][pq], C[-1][pq], T.wa, T.wb, T.wa, T.wb)[0] for pq in PAIRS}
same = all((direct[pq] or []) == comp[pq] for pq in PAIRS)
print(f"POST HOC: subdivided isotopy unambiguous {ok_all} (worst ratio {worst:.3f}); composite map = direct match on all pairs: {same}")
