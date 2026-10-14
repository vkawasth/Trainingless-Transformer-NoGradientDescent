"""fibre_scan.py -- Betti numbers of tropical Hardy-Weinberg 2-mixture fibres over a grid of targets (writes betti_scan.json).
b1 is computed whenever the cover has at most 24 nonempty cells; otherwise only b0 is reported (b1 = null)."""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibre_topology import ALPHAS, SIGNS, cell_constraints, feasible, betti
from fractions import Fraction as F
from multiprocessing import Pool

def job(ac):
    a, c = F(ac[0]), F(ac[1]); v = [F(0), a, c]; m = min(v); v = [x - m for x in v]
    cells = [(al, s) for al in ALPHAS for s in SIGNS]; C = {cc: cell_constraints(cc[0], cc[1], v) for cc in cells}
    verts = [cc for cc in cells if feasible([C[cc]])]
    simp = {0: [(cc,) for cc in verts], 1: [s for s in itertools.combinations(verts, 2) if feasible([C[x] for x in s])]}
    if len(verts) <= 24:
        e1 = set(simp[1])
        simp[2] = [s for s in itertools.combinations(verts, 3)
                   if all(f in e1 for f in itertools.combinations(s, 2)) and feasible([C[x] for x in s])]
        b = betti(simp, 1)
    else:
        simp[2] = []; b = [betti(simp, 0)[0], None]
    return (str(a), str(c), b, len(verts))

if __name__ == "__main__":
    grid = [(str(F(ai, 2)), str(F(ci, 2))) for ai in range(-8, 9) for ci in range(-12, 15)]
    t = time.time()
    with Pool(2) as p:
        out = p.map(job, grid, chunksize=1)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "betti_scan.json"), "w"))
    from collections import Counter
    print(f"{time.time() - t:.0f}s", Counter((tuple(o[2]), o[3] > 24) for o in out))
