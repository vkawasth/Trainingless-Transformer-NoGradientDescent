"""Small exact example for sec:lexdescent (illustration, exact rational arithmetic; not a test). 2 x 4 tables, columns merged {0,1},{2,3}
or {0},{1,2,3}; rows not merged. Weighted leximin: maximise lexicographically the sorted vector (w_k x_k) over the fibre."""
import os, sys, json
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.cpl33 import certify_max
def wleximin(r, c, f, g, w):
    cells = [(i, j) for i in range(r) for j in range(c)]; n = len(cells)
    A = [[Fr(1) if cl[0] == i else Fr(0) for cl in cells] for i in range(r)] + [[Fr(1) if cl[1] == j else Fr(0) for cl in cells] for j in range(c - 1)]
    b = list(f) + list(g[:-1]); fixed = {}; free = set(range(n)); levels = []
    while free:
        cvec = [Fr(0)] * n + [Fr(1)]; A_eq = [row + [Fr(0)] for row in A] + [[Fr(1) if k == i else Fr(0) for k in range(n)] + [Fr(0)] for i in fixed]; b_eq = b + [fixed[i] for i in fixed]
        A_ub = [[(-w[i] if k == i else Fr(0)) for k in range(n)] + [Fr(1)] for i in sorted(free)]; t, _ = certify_max(cvec, A_eq, b_eq, A_ub, [Fr(0)] * len(free), free={n})
        newly = []
        for i in sorted(free):
            c2 = [Fr(1) if k == i else Fr(0) for k in range(n)]; A_eq2 = A + [[Fr(1) if k == j else Fr(0) for k in range(n)] for j in fixed]; b_eq2 = b + [fixed[j] for j in fixed]
            A_ub2 = [[(-w[j] if k == j else Fr(0)) for k in range(n)] for j in sorted(free)]; v, _ = certify_max(c2, A_eq2, b_eq2, A_ub2, [-t] * len(free), free=set())
            if v * w[i] == t: newly.append(i)
        for i in newly: fixed[i] = t / w[i]; free.discard(i)
        levels.append((t, [cells[i] for i in newly]))
    return {cells[k]: v for k, v in fixed.items()}, levels
def show(x, r, c): return [[str(x[(i, j)]) for j in range(c)] for i in range(r)]
def merge(x, r, groups): return [[sum(x[(i, j)] for j in G) for G in groups] for i in range(r)]
out = {}
f = [Fr(3, 10), Fr(7, 10)]
cases = {"symmetric": ([Fr(1, 4), Fr(1, 4), Fr(1, 4), Fr(1, 4)], [[0, 1], [2, 3]]),
         "asymmetric": ([Fr(1, 12), Fr(5, 12), Fr(1, 4), Fr(1, 4)], [[0, 1], [2, 3]]),
         "unequal blocks": ([Fr(1, 4), Fr(1, 4), Fr(1, 4), Fr(1, 4)], [[0], [1, 2, 3]])}
for name, (g, groups) in cases.items():
    xf, lf = wleximin(2, 4, f, g, [Fr(1)] * 8); gc = [sum(g[j] for j in G) for G in groups]; k = len(groups)
    xc, _ = wleximin(2, k, f, gc, [Fr(1)] * (2 * k)); xa, _ = wleximin(2, k, f, gc, [Fr(1, len(groups[j])) for i in range(2) for j in range(k)])
    mf = merge(xf, 2, groups); plain = [[xc[(i, j)] for j in range(k)] for i in range(2)]; avg = [[xa[(i, j)] for j in range(k)] for i in range(2)]
    out[name] = dict(g=[str(v) for v in g], groups=groups, fine=show(xf, 2, 4), merged_fine=[[str(v) for v in row] for row in mf],
                     coarse_plain=[[str(v) for v in row] for row in plain], coarse_average=[[str(v) for v in row] for row in avg],
                     plain_descends=mf == plain, average_descends=mf == avg, levels=[str(h) for h, _ in lf])
    print(name, json.dumps(out[name]))
json.dump(out, open(os.path.join(HERE, "study35_four6", "descent_example.json"), "w"), indent=1)
