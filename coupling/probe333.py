import itertools, random, math
from fractions import Fraction as F
from collections import Counter, deque
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_form
import numpy as np
from scipy.optimize import linprog
R = random.Random(5)
def build(shape, p):
    cells = list(itertools.product(*[range(s) for s in shape])); rows = []; rhs = []
    for keep in ((0,1),(0,2),(1,2)):
        for k in itertools.product(*[range(shape[i]) for i in keep]):
            rows.append([int(tuple(c[i] for i in keep) == k) for c in cells]); rhs.append(sum(p[c] for c in cells if tuple(c[i] for i in keep) == k))
    M = Matrix(rows); rk = M.rank()
    # independent rows
    _, piv = M.T.rref(); idx = list(piv)
    A = [rows[i] for i in idx]; b = [rhs[i] for i in idx]
    return cells, A, b
def solve_basis(A, b, B):
    M = Matrix([[A[i][j] for j in B] for i in range(len(A))]); x = M.LUsolve(Matrix(b))
    return [F(int(v.p), int(v.q)) for v in x]
def vertices(shape, p):
    cells, A, b = build(shape, p); m, N = len(A), len(cells)
    lp = linprog(np.zeros(N), A_eq=np.array(A, float), b_eq=np.array([float(x) for x in b]), bounds=[(0, None)]*N, method="highs-ds")
    x = lp.x; B = sorted(range(N), key=lambda j: -x[j])[:m]
    # make B a basis
    while Matrix([[A[i][j] for j in B] for i in range(m)]).rank() < m:
        B = sorted(R.sample(range(N), m))
    B = sorted(B); xb = solve_basis(A, b, B)
    assert all(v >= 0 for v in xb), "start basis infeasible"
    start = frozenset(B); seen = {start: xb}; queue = deque([start]); V = {}
    while queue:
        B = sorted(queue.popleft()); xb = seen[frozenset(B)]
        v = [F(0)]*N
        for j, val in zip(B, xb): v[j] = val
        V[tuple(v)] = None
        AB = Matrix([[A[i][j] for j in B] for i in range(m)]); ABinv = AB.inv()
        for j in range(N):
            if j in B: continue
            d = ABinv * Matrix([A[i][j] for i in range(m)])
            d = [F(int(t.p), int(t.q)) for t in d]
            ratios = [(xb[k] / d[k], k) for k in range(m) if d[k] > 0]
            if not ratios: continue
            tmin = min(r for r, _ in ratios)
            for r, k in ratios:
                if r == tmin:
                    nb = B[:]; nb[k] = j; key = frozenset(nb)
                    if key not in seen:
                        nx = solve_basis(A, b, sorted(nb)); seen[key] = nx; queue.append(key)
    return cells, A, list(V)
def analyse(cells, V, dim):
    Z = [frozenset(i for i, x in enumerate(v) if x == 0) for v in V]
    simple = smooth = True; idxs = []
    for i, v in enumerate(V):
        nb = [j for j in range(len(V)) if j != i and not any(k not in (i, j) and Z[k] >= (Z[i] & Z[j]) for k in range(len(V)))]
        if len(nb) != dim: simple = False; continue
        E = []
        for j in nb:
            w = [V[j][k] - v[k] for k in range(len(cells))]; den = 1
            for x in w: den = den * x.denominator // math.gcd(den, x.denominator)
            ints = [int(x * den) for x in w]; g = 0
            for x in ints: g = math.gcd(g, abs(x))
            E.append([x // g for x in ints])
        S = smith_normal_form(Matrix(E).T)
        ind = 1
        for t in range(dim): ind *= abs(S[t, t])
        idxs.append(ind)
        if ind != 1: smooth = False
    return simple, smooth, Counter(idxs)
for shape in ((2,3,3),(3,3,3)):
    for t in range(4):
        cells0 = list(itertools.product(*[range(s) for s in shape]))
        w = [F(R.randint(1, 10**6)) for _ in cells0]; s = sum(w); p = {c: x/s for c, x in zip(cells0, w)}
        cells, A, V = vertices(shape, p); dim = len(cells) - len(A)
        zc = Counter(sum(1 for x in v if x == 0) for v in V)
        si, sm, idx = analyse(cells, V, dim)
        print(shape, "dim", dim, "vertices", len(V), "zeros/vertex", dict(zc), "simple", si, "smooth", sm, "indices", dict(idx), flush=True)
