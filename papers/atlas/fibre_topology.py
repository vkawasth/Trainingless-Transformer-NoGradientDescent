"""fibre_topology.py -- tropical fibres of 2-component Hardy-Weinberg mixtures, their assignment-cell covers, nerves and Cech cohomology.

Parameters theta = (beta, d1, d2) in R^3.  Weights (tropical, normalised): w1 = max(0,-beta), w2 = max(0,beta).
Component rates: u(d) = (0, d, 2d) - min(0, 2d)  (the positive tropicalisation of r(s,t) = (s^2, 2st, t^2)/(s+t)^2, d = val t - val s).
Mixture rates: m(x) = min(w1 + u(d1)(x), w2 + u(d2)(x)).  Fibre over v: {theta : m = v}.

Cover: closed convex cells P_{alpha,sigma}, alpha: X -> {1,2} the designated attainer, sigma in {+,-}^3 the sign region of (beta,d1,d2).
Within a sign region every term is affine in theta, so each cell is a polyhedron; their union is the fibre.
"""
from fractions import Fraction as F
import itertools
import numpy as np
from scipy.optimize import linprog

X = (0, 1, 2)
SIGNS = list(itertools.product((1, -1), repeat=3))
ALPHAS = list(itertools.product((0, 1), repeat=3))

def affine_terms(sig):
    """return T[i][x] = (coef vector over (beta,d1,d2), constant) valid in sign region sig"""
    sb, s1, s2 = sig
    # weights: beta>=0: w1=0, w2=beta ; beta<=0: w1=-beta, w2=0
    w = [((0, 0, 0), 0), ((1, 0, 0), 0)] if sb > 0 else [((-1, 0, 0), 0), ((0, 0, 0), 0)]
    def u(idx, s):  # idx = 1 or 2 -> coordinate of d
        e = [0, 0, 0]; e[idx] = 1
        if s > 0:   # (0, d, 2d)
            return [((0, 0, 0), 0), (tuple(e), 0), (tuple(2 * c for c in e), 0)]
        return [(tuple(-2 * c for c in e), 0), (tuple(-c for c in e), 0), ((0, 0, 0), 0)]
    T = []
    for i, (idx, s) in enumerate(((1, s1), (2, s2))):
        uu = u(idx, s)
        T.append([(tuple(a + b for a, b in zip(w[i][0], uu[x][0])), w[i][1] + uu[x][1]) for x in X])
    return T

def cell_constraints(alpha, sig, v):
    """equalities A_eq th = b_eq, inequalities A_ub th <= b_ub (lists of Fractions)"""
    T = affine_terms(sig); Aeq, beq, Aub, bub = [], [], [], []
    for x in X:
        a = alpha[x]; c, k = T[a][x]
        Aeq.append(list(c)); beq.append(F(v[x]) - k)
        j = 1 - a; c2, k2 = T[j][x]
        Aub.append([-q for q in c2]); bub.append(k2 - F(v[x]))   # T_j >= v  <=>  -c2.th <= k2 - v
    for i, s in enumerate(sig):   # s*theta_i >= 0
        row = [0, 0, 0]; row[i] = -s; Aub.append(row); bub.append(F(0))
    return Aeq, beq, Aub, bub

def feasible(cons_list, return_point=False):
    Aeq, beq, Aub, bub = [], [], [], []
    for c in cons_list:
        Aeq += c[0]; beq += c[1]; Aub += c[2]; bub += c[3]
    res = linprog(np.zeros(3), A_ub=np.array(Aub, float) if Aub else None, b_ub=np.array(bub, float) if bub else None,
                  A_eq=np.array(Aeq, float) if Aeq else None, b_eq=np.array(beq, float) if beq else None,
                  bounds=[(None, None)] * 3, method="highs")
    ok = res.status == 0
    return (ok, res.x if ok else None) if return_point else ok

def rates(theta):
    b, d1, d2 = theta
    w1, w2 = max(0, -b), max(0, b)
    u = lambda d: [x - min(0, 2 * d) for x in (0, d, 2 * d)]
    return [min(w1 + a, w2 + c) for a, c in zip(u(d1), u(d2))]

def nerve(v, maxdim=2):
    cells = [(a, s) for a in ALPHAS for s in SIGNS]
    C = {c: cell_constraints(c[0], c[1], v) for c in cells}
    verts = [c for c in cells if feasible([C[c]])]
    simplices = {0: [(c,) for c in verts]}
    for k in range(1, maxdim + 1):
        simplices[k] = []
        for s in itertools.combinations(verts, k + 1):
            # all faces must be present
            if k >= 2 and not all(f in set(simplices[k - 1]) for f in itertools.combinations(s, k)): continue
            if feasible([C[c] for c in s]): simplices[k].append(s)
    return verts, simplices, C

def rank_Q(M):
    """exact rank over Q of an integer matrix (list of lists)"""
    A = [[F(x) for x in r] for r in M]; rk = 0
    if not A: return 0
    nc = len(A[0])
    for c in range(nc):
        piv = next((r for r in range(rk, len(A)) if A[r][c] != 0), None)
        if piv is None: continue
        A[rk], A[piv] = A[piv], A[rk]
        for r in range(len(A)):
            if r != rk and A[r][c] != 0:
                f = A[r][c] / A[rk][c]; A[r] = [A[r][j] - f * A[rk][j] for j in range(nc)]
        rk += 1
    return rk

def betti(simplices, upto=1):
    """Cech/simplicial Betti numbers b_0..b_upto of the nerve (rational coefficients)"""
    idx = {k: {s: i for i, s in enumerate(simplices[k])} for k in simplices}
    def delta(k):  # coboundary C^k -> C^{k+1}, matrix rows = (k+1)-simplices
        rows = []
        for s in simplices.get(k + 1, []):
            r = [0] * len(simplices[k])
            for j in range(len(s)):
                face = s[:j] + s[j + 1:]
                r[idx[k][face]] += (-1) ** j
            rows.append(r)
        return rows
    ranks = {k: (rank_Q(delta(k)) if simplices.get(k + 1) and simplices[k] else 0) for k in range(upto + 1)}
    b = []
    for k in range(upto + 1):
        dimCk = len(simplices[k]); rk_out = ranks[k]; rk_in = ranks[k - 1] if k > 0 else 0
        b.append(dimCk - rk_out - rk_in)
    return b

def fibre_samples(v, n=4000, rng=None):
    """sample points of the fibre: random interior points of each nonempty cell (via LP with random objective)"""
    rng = rng or np.random.default_rng(0)
    verts, simp, C = nerve(v, maxdim=1)
    pts = []
    for c in verts:
        Aeq, beq, Aub, bub = C[c]
        for _ in range(12):
            obj = rng.normal(size=3)
            res = linprog(obj, A_ub=np.array(Aub, float), b_ub=np.array(bub, float), A_eq=np.array(Aeq, float),
                          b_eq=np.array(beq, float), bounds=[(-6, 6)] * 3, method="highs")
            if res.status == 0: pts.append((c, res.x))
    return verts, simp, pts
