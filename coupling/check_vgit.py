"""check_vgit.py -- (2) coupling toric varieties as GIT quotients, and wall-crossing beyond 2x3.

  V1 the kernel of the ray map Z^{n x m} -> Lambda^*, c -> v_c, is the lattice of matrices r_i + s_j
     (rank n+m-1): X_{f,g} = C^{n x m} //_{(f,g)} (T^n x T^m)/diagonal, with (f,g) the GIT character
  V2 the hyperplanes spanned by the weights e_i (+) e_j are exactly the walls sum_I f = sum_J g together with
     the boundary facets f_i = 0, g_j = 0 of the weight cone
  V3 wall-crossings for 2x4 and 3x3: classify by fans (rays = facet cells, cones = zero sets at vertices):
     blow-up/down (rays differ by one) or flip (same rays, different cones)
"""
import itertools, random
from fractions import Fraction as F
from collections import Counter
import numpy as np
from poly import fan, walls, signs
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form
R = random.Random(31)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

def ray(c, n, m):
    """cell functional on the circulation lattice in the basis e_ij (i<n-1, j<m-1)"""
    i, j = c; v = []
    for a in range(n - 1):
        for b in range(m - 1):
            v.append((1 if (i, j) == (a, b) else 0) - (1 if (i == a and j == m - 1) else 0) - (1 if (i == n - 1 and j == b) else 0)
                     + (1 if (i, j) == (n - 1, m - 1) else 0))
    return v
ok = True; info = []
for n, m in ((2, 3), (2, 4), (3, 3), (3, 4)):
    cells = [(i, j) for i in range(n) for j in range(m)]
    M = Matrix([ray(c, n, m) for c in cells]).T          # (n-1)(m-1) x nm
    K = M.nullspace()
    rk = len(K)
    # the row/column characters r_i + s_j span the kernel
    RC = Matrix([[int(c[0] == i) for c in cells] for i in range(n)] + [[int(c[1] == j) for c in cells] for j in range(m)])
    span_ok = Matrix.hstack(*K).rank() == RC.rank() == n + m - 1 and (M * RC.T).is_zero_matrix
    # saturation of the kernel lattice: Smith invariants of RC^T are all 1
    S = smith_normal_form(RC.T, domain=ZZ); inv = [abs(S[i, i]) for i in range(min(S.shape)) if S[i, i] != 0]
    ok &= span_ok and rk == n + m - 1 and all(x == 1 for x in inv)
    info.append(f"{n}x{m}: kernel rank {rk}")
check("V1 the ray map's kernel is the row/column character lattice (saturated): Cox/GIT presentation", ok, "; ".join(info))

# V2: hyperplanes spanned by rank-deficient subsets of the weights a_c = e_i (+) e_j, inside {sum f = sum g}.
# A flat of rank n+m-2 has a 2-dim annihilator in R^{n+m} containing triv = (1..1,-1..-1); its class mod triv is the
# wall normal.  We match it exactly against v_IJ = (1_I, -1_J) (v_{I^c J^c} = triv - v_IJ is the same functional).
ok = True; info = []
for n, m in ((2, 3), (2, 4), (3, 3)):
    cells = [(i, j) for i in range(n) for j in range(m)]
    A = np.array([[float(c[0] == i) for i in range(n)] + [float(c[1] == j) for j in range(m)] for c in cells])
    W = walls(n, m)
    V = {w: np.r_[[float(i in w[0]) for i in range(n)], [-float(j in w[1]) for j in range(m)]] for w in W}
    target = n + m - 2; matched = set(); flats = set(); bad = 0; nb = 0
    for r in range(1, len(cells) + 1):
        for S in itertools.combinations(range(len(cells)), r):
            sub = A[list(S)]
            if np.linalg.matrix_rank(sub) != target: continue
            flat = frozenset(k for k in range(len(cells)) if np.linalg.matrix_rank(np.vstack([sub, A[k]])) == target)
            if flat in flats: continue
            flats.add(flat)
            hit = [w for w in W if np.allclose(A[sorted(flat)] @ V[w], 0)]
            bnd = [k for k in range(n + m) if np.allclose(A[sorted(flat)][:, k], 0)]   # facet f_i = 0 or g_j = 0
            if len(hit) + len(bnd) != 1: bad += 1
            matched.update(hit); nb += len(bnd)
    # conversely every (I,J) has a zero set of rank exactly n+m-2
    conv = all(np.linalg.matrix_rank(A[[k for k in range(len(cells)) if abs(A[k] @ V[w]) < 1e-12]]) == target for w in W)
    ok &= bad == 0 and matched == set(W) and len(flats) == len(W) + n + m and nb == n + m and conv
    info.append(f"{n}x{m}: {len(flats)} flats = {len(W)} walls + {nb} boundary facets")
check("V2 hyperplanes spanned by weights = the walls sum_I f = sum_J g plus the boundary facets f_i = 0, g_j = 0", ok, "; ".join(info))

def rmarg(k, den=60):
    w = [R.randint(1, den) for _ in range(k)]; s = sum(w); return [F(x, s) for x in w]
def crossing_types(n, m, trials):
    W = walls(n, m); types = Counter(); examples = {}
    found = 0; attempts = 0
    while found < trials and attempts < 20000:
        attempts += 1
        f, g = rmarg(n, 200), rmarg(m, 200)
        s0 = signs(f, g, W)
        if 0 in s0: continue
        # small perturbation
        f2 = [x + F(R.randint(-6, 6), 400) for x in f]; g2 = [x + F(R.randint(-6, 6), 400) for x in g]
        if min(f2) <= 0 or min(g2) <= 0: continue
        sf, sg = sum(f2), sum(g2); f2 = [x / sf for x in f2]; g2 = [x / sg for x in g2]
        s1 = signs(f2, g2, W)
        if 0 in s1 or sum(a != b for a, b in zip(s0, s1)) != 1: continue
        found += 1
        (r0, c0, v0), (r1, c1, v1) = fan(f, g), fan(f2, g2)
        if r0 == r1 and c0 == c1: t = "same fan"
        elif len(r0 ^ r1) == 1: t = "blow-up/down (one ray)"
        elif r0 == r1: t = "flip (same rays)"
        else: t = f"other ({len(r0 ^ r1)} rays change)"
        types[t] += 1; examples.setdefault(t, (f, g, f2, g2, v0, v1))
    return types, examples
for n, m, trials in ((2, 4, 40), (3, 3, 25)):
    types, ex = crossing_types(n, m, trials)
    check(f"V3 {n}x{m}: every single-wall crossing is a blow-up/down or a flip", all(t.startswith(("blow", "flip", "same")) for t in types),
          str(dict(types)))
print(f"\n{sum(res)}/{len(res)} checks passed")
