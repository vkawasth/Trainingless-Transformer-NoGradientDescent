"""Study 31 addendum (written AFTER check_credalshape.py ran and the figure showed, for all three figure tables, that the before-evidence
outcome region equals the product of its three coordinate bands while the after-evidence region does not). Post hoc; a record, not a test.
  SO7 RECORD (post hoc)  for 100 random defined P332 tables: ratio (area of the outcome region)/(area of the product of its coordinate
                         bands) before evidence, after evidence, and whether every edge of the before-evidence region is parallel to an
                         edge of the simplex (a generalized permutohedron)
"""
import os, sys, itertools
import numpy as np
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "check_credalshape.py")).read()
pre = src.split("# ------------------------------------------------------------------ SO1")[0]; exec(pre)
mid = src.split("# ------------------------------------------------------------------ SO3-SO6 on P332")[1].split("ok3 = True")[0]; exec(mid)
def tern(q): q = np.atleast_2d(q); return np.column_stack([q[:, 1] + q[:, 2] / 2, q[:, 2] * np.sqrt(3) / 2])
def ratio(P):
    P = np.array(P); cb = [[P[:, a].min(), P[:, a].max()] for a in range(3)]; pts = []
    for a, b in itertools.combinations(range(3), 2):
        for va in cb[a]:
            for vb in cb[b]:
                q = np.zeros(3); q[a] = va; q[b] = vb; q[3 - a - b] = 1 - va - vb
                if all(cb[i][0] - 1e-12 <= q[i] <= cb[i][1] + 1e-12 for i in range(3)): pts.append(q)
    A = ConvexHull(tern(P)).volume if np.linalg.matrix_rank(P[1:] - P[0]) >= 2 else 0.0; Bp = ConvexHull(tern(np.array(pts))).volume if len(pts) >= 3 else 0.0
    return A / Bp if Bp > 0 else 1.0
def permutohedral(P):
    P = np.array(P)
    if np.linalg.matrix_rank(P[1:] - P[0]) < 2: return True
    h = ConvexHull(tern(P)); V = P[h.vertices]; ok = True
    for i in range(len(V)):
        d = V[(i + 1) % len(V)] - V[i]; ok &= int((np.abs(d) < 1e-10).sum()) == 1
    return bool(ok)
rng = np.random.default_rng(71); r0 = []; r1 = []; perm = 0; n = 0
for q in rng.choice(len(R["Di"]), 100, replace=False):
    xb = exact_vertex(q, "lo0"); V = enumerate_vertices(xb); Q0 = np.array([to_q(x)[1] for x in V]); Q1 = np.array([to_q(x, Lc)[1] for x in V])
    r0.append(ratio(Q0)); r1.append(ratio(Q1)); perm += permutohedral(Q0); n += 1
r0, r1 = np.array(r0), np.array(r1)
check("SO7 RECORD (post hoc) outcome region against the product of its bands", True,
      f"{n} tables; before evidence: ratio min {r0.min():.6f} max {r0.max():.6f}, equal to the product in {(r0 > 1 - 1e-9).sum()}, edges parallel to the simplex in {perm}; "
      f"after evidence: ratio min {r1.min():.3f} median {np.median(r1):.3f} max {r1.max():.3f}, equal in {(r1 > 1 - 1e-9).sum()}")
print(f"{sum(results)}/{len(results)} passed")
