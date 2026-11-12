"""check_hallucination.py -- the three-token hallucination example of "Shadows, Fibres and Secants of the Posterior Tower".

Model: binary tokens X0, X1, X2, each pair agreeing with probability eps (pairwise table [[eps/2,(1-eps)/2],[(1-eps)/2,eps/2]]).
Sources s = A, B, U with eps_A = 1/5, eps_B = 3/10, eps_U = 2/5 and mixture weights w; eps(w) = sum_s w_s eps_s.
Exact rational arithmetic except the singular-value checks L4b/L7 (double precision, tolerance 1e-9) and the fibre minimisation L5.

  L1  a joint (global section) exists iff eps >= 1/3; for eps < 1/3 the functional sum_{i<j} P(X_i != X_j) <= 2 (valid on every
      joint, checked on all 8 atoms) is violated by exactly 1 - 3 eps
  L2  the global sections are J + s chi (chi = parity, the one-dimensional kernel of the pairwise-margin map), |s| <= min(a, b),
      with J symmetric: a = (3eps-1)/4 on 000, 111 and b = (1-eps)/4 elsewhere
  L3  for a full-support joint, the meet of sigma(X0,X1) and sigma(X1,X2) is sigma(X1)
  L4  Beck-Chevalley for the window square holds iff X0 _||_ X2 | X1; at the symmetric joint iff eps = 1/2
  L4b at the symmetric joint the Beck-Chevalley defect norm is c = |2eps-1|/(2eps) = max_{x1} |corr(X0, X2 | X1 = x1)|
  L5  a Markov (Beck-Chevalley) joint in the fibre exists iff eps = 1/2 (eps in (1/3,1)); over 99 sections s on the fibre the
      defect c is smallest at the symmetric joint s = 0
  L6  the 12-token stream: obstructed exactly for t = 1..7 and liftable for t = 8..12 (the reverse of the draft's table)
  L7  alternating the two window shadows converges to E[.|X1] at rate c^(2n-1)
  L8  valuation commutes with marginalisation on positive Puiseux joints; a scaled family with agreement rate a > 0 is tropically
      non-liftable (support certificate), and with a = 0 it is tropically liftable for every leading coefficient
"""
from fractions import Fraction as F
import itertools, json, hashlib, sys
import numpy as np

OUT = {}
def check(name, cond, info=""):
    OUT[name] = bool(cond); print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
ATOMS = list(itertools.product((0, 1), repeat=3)); PAIRS = [(0, 1), (1, 2), (0, 2)]
chi = {x: (-1) ** sum(x) for x in ATOMS}

def tables(P):  # pairwise tables of a joint
    return {(i, j): {(u, v): sum(P[x] for x in ATOMS if x[i] == u and x[j] == v) for u in (0, 1) for v in (0, 1)} for i, j in PAIRS}
def model(eps): return {(u, v): (eps / 2 if u == v else (1 - eps) / 2) for u in (0, 1) for v in (0, 1)}
def sym_joint(eps):
    a, b = (3 * eps - 1) / 4, (1 - eps) / 4
    return {x: (a if len(set(x)) == 1 else b) for x in ATOMS}, a, b

# L1
okL1 = True
for k in range(0, 41):
    eps = F(k, 40); J, a, b = sym_joint(eps)
    exists = eps >= F(1, 3)
    if exists: okL1 &= all(J[x] >= 0 for x in ATOMS) and all(tables(J)[p] == model(eps) for p in PAIRS)
    # the functional: every atom has at most two disagreeing pairs
    okL1 &= all(sum(1 for i, j in PAIRS if x[i] != x[j]) <= 2 for x in ATOMS)
    viol = 3 * (1 - eps) - 2
    okL1 &= (viol > 0) == (not exists) and viol == 1 - 3 * eps
check("L1 global section exists iff eps >= 1/3; certificate sum P(X_i != X_j) <= 2 violated by 1 - 3 eps", okL1)

# L2: kernel of the margin map is spanned by chi; range of s
def rank(rows):
    A = [list(r) for r in rows]; rk = 0; nc = len(A[0])
    for c in range(nc):
        piv = next((r for r in range(rk, len(A)) if A[r][c] != 0), None)
        if piv is None: continue
        A[rk], A[piv] = A[piv], A[rk]
        for r in range(len(A)):
            if r != rk and A[r][c] != 0:
                f = A[r][c] / A[rk][c]; A[r] = [A[r][j] - f * A[rk][j] for j in range(nc)]
        rk += 1
    return rk
M = [[F(1) if (x[i], x[j]) == cell else F(0) for x in ATOMS] for (i, j) in PAIRS for cell in itertools.product((0, 1), repeat=2)]
okL2 = rank(M) == 7 and all(sum(r[k] * chi[x] for k, x in enumerate(ATOMS)) == 0 for r in M)
for k in range(14, 40):
    eps = F(k, 40); J, a, b = sym_joint(eps); m = min(a, b)
    for s in (m, -m, m / 2):
        P = {x: J[x] + s * chi[x] for x in ATOMS}
        okL2 &= all(v >= 0 for v in P.values()) and all(tables(P)[p] == model(eps) for p in PAIRS)
    if m > 0:
        P = {x: J[x] + (m + F(1, 1000)) * chi[x] for x in ATOMS}; okL2 &= min(P.values()) < 0
check("L2 global sections = J + s*chi, |s| <= min(a, b); chi spans the kernel of the margin map (rank 7)", okL2)

# finite L2 machinery
def canon(lab):
    d = {}; return tuple(d.setdefault(v, len(d)) for v in lab)
def meet(a, b):
    n = len(a); par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for lab in (a, b):
        first = {}
        for i, v in enumerate(lab):
            if v in first: par[find(i)] = find(first[v])
            else: first[v] = i
    return canon([find(i) for i in range(n)])
def Emat(lab, P):
    n = len(P); Mx = [[F(0)] * n for _ in range(n)]
    for c in set(lab):
        cell = [i for i in range(n) if lab[i] == c]; m = sum(P[j] for j in cell)
        for i in cell:
            for j in cell: Mx[i][j] = P[j] / m
    return Mx
def app(Mx, f): return [sum(Mx[i][j] * f[j] for j in range(len(f))) for i in range(len(Mx))]
W01 = canon([(x[0], x[1]) for x in ATOMS]); W12 = canon([(x[1], x[2]) for x in ATOMS]); X1 = canon([x[1] for x in ATOMS])
okL3 = meet(W01, W12) == X1
check("L3 meet of the two context windows sigma(X0,X1), sigma(X1,X2) is sigma(X1)", okL3)

def bc_holds(P):
    Pl = [P[x] for x in ATOMS]; EA, EC = Emat(W01, Pl), Emat(X1, Pl)
    basisB = [[F(1) if W12[i] == c else F(0) for i in range(8)] for c in set(W12)]
    return all(app(EA, f) == app(EC, f) for f in basisB)
def cond_indep(P):
    for x1 in (0, 1):
        Z = sum(P[x] for x in ATOMS if x[1] == x1)
        p = lambda u, v: P[(u, x1, v)] / Z
        for u in (0, 1):
            for v in (0, 1):
                if p(u, v) != (p(u, 0) + p(u, 1)) * (p(0, v) + p(1, v)): return False
    return True
def c_norm(P):
    Pl = np.array([float(P[x]) for x in ATOMS])
    D, Di = np.diag(np.sqrt(Pl)), np.diag(1 / np.sqrt(Pl))
    E = lambda lab: np.array([[float(v) for v in r] for r in Emat(lab, [P[x] for x in ATOMS])])
    return np.linalg.norm(D @ (E(W01) - E(X1)) @ E(W12) @ Di, 2)
def cond_corr(P, x1):
    Z = sum(P[x] for x in ATOMS if x[1] == x1); p = {(u, v): float(P[(u, x1, v)] / Z) for u in (0, 1) for v in (0, 1)}
    r0, r1, c0, c1 = p[0, 0] + p[0, 1], p[1, 0] + p[1, 1], p[0, 0] + p[1, 0], p[0, 1] + p[1, 1]
    return (p[0, 0] * p[1, 1] - p[0, 1] * p[1, 0]) / np.sqrt(r0 * r1 * c0 * c1)

okL4 = okL4b = True
for k in range(14, 40):
    eps = F(k, 40); J, a, b = sym_joint(eps)
    okL4 &= bc_holds(J) == cond_indep(J) == (eps == F(1, 2))
    cf = abs(2 * float(eps) - 1) / (2 * float(eps))
    okL4b &= abs(c_norm(J) - cf) < 1e-9 and abs(max(abs(cond_corr(J, 0)), abs(cond_corr(J, 1))) - cf) < 1e-9
check("L4 window Beck-Chevalley <=> X0 _||_ X2 | X1; at the symmetric joint iff eps = 1/2", okL4)
check("L4b symmetric joint: defect norm c = |2eps-1|/(2eps) = max conditional correlation", okL4b)

okL5 = True; minc = {}
for k in range(14, 40):
    eps = F(k, 40); J, a, b = sym_joint(eps); m = min(a, b)
    ss = [m * F(j, 50) for j in range(-49, 50)]
    vals = []; anyBC = False
    for s in ss:
        P = {x: J[x] + s * chi[x] for x in ATOMS}
        anyBC |= cond_indep(P); vals.append(c_norm(P))
    okL5 &= anyBC == (eps == F(1, 2))
    okL5 &= abs(min(vals) - abs(2 * float(eps) - 1) / (2 * float(eps))) < 1e-9   # minimum attained at the symmetric joint (s = 0)
    minc[str(eps)] = round(min(vals), 4)
check("L5 a Beck-Chevalley (Markov) joint exists in the fibre iff eps = 1/2; on the s-grid c is minimised at the symmetric joint", okL5, f"min_s c at eps=2/5: {minc['2/5']}, 7/20: {minc['7/20']}, 3/5: {minc['3/5']}")

eA, eB, eU = F(1, 5), F(3, 10), F(2, 5)
S = [("0.10", "0.10"), ("0.20", "0.15"), ("0.30", "0.20"), ("0.40", "0.30"), ("0.50", "0.40"), ("0.60", "0.50"), ("0.70", "0.60"),
     ("0.80", "0.65"), ("0.90", "0.70"), ("0.95", "0.75"), ("0.50", "0.80"), ("0.30", "0.90")]
stream = []
for t, (th, mu) in enumerate(S, 1):
    th, mu = F(th), F(mu); e = (1 - mu) * (th * eA + (1 - th) * eB) + mu * eU; mb = (F(1, 3) + th) / (1 + th)
    stream.append((t, e, e >= F(1, 3), (mu >= mb) == (e >= F(1, 3))))
okL6 = [s[0] for s in stream if not s[2]] == list(range(1, 8)) and all(s[3] for s in stream)
check("L6 stream: obstructed for t = 1..7, liftable for t = 8..12; liftable side is mu >= mu(theta)", okL6,
      ", ".join(f"t{t}: eps={float(e):.4f}" for t, e, _, _ in stream[5:9]))

okL7 = True
for eps in (F(2, 5), F(7, 20), F(3, 5), F(3, 4)):
    J, a, b = sym_joint(eps); Pl = np.array([float(J[x]) for x in ATOMS]); D, Di = np.diag(np.sqrt(Pl)), np.diag(1 / np.sqrt(Pl))
    E = lambda lab: np.array([[float(v) for v in r] for r in Emat(lab, [J[x] for x in ATOMS])])
    AB = E(W01) @ E(W12); EM = E(X1); cf = abs(2 * float(eps) - 1) / (2 * float(eps)); T = np.eye(8)
    for n in range(1, 7):
        T = T @ AB; okL7 &= abs(np.linalg.norm(D @ (T - EM) @ Di, 2) - cf ** (2 * n - 1)) < 1e-9
check("L7 alternating the window shadows converges to E[.|X1] at rate c^(2n-1)", okL7)

# L8 tropical
def padd(a, b):
    c = dict(a)
    for e, x in b.items():
        c[e] = c.get(e, 0) + x
        if c[e] == 0: del c[e]
    return c
val = lambda a: min(a)
import random; rng = random.Random(5)
okL8 = True
for t in range(200):
    P = {x: {rng.randint(0, 4): F(rng.randint(1, 5)), 5: F(rng.randint(-2, 2) or 1)} for x in ATOMS}
    for (i, j) in PAIRS:
        for u in (0, 1):
            for v in (0, 1):
                tot = {}
                for x in ATOMS:
                    if x[i] == u and x[j] == v: tot = padd(tot, P[x])
                okL8 &= val(tot) == min(val(P[x]) for x in ATOMS if x[i] == u and x[j] == v)
def trop_liftable(a):
    # agree cells have rate a, disagree cells rate 0; a tropical joint r must have min over each cell equal to the cell rate.
    # every atom lies in an agree cell of some pair, so r >= a on all atoms; feasible iff the disagree cells can reach 0, i.e. a == 0
    lower = {x: (a if any(x[i] == x[j] for i, j in PAIRS) else 0) for x in ATOMS}
    r = dict(lower)   # the smallest candidate
    ok = True
    for (i, j) in PAIRS:
        for u in (0, 1):
            for v in (0, 1):
                target = a if u == v else 0
                ok &= min(r[x] for x in ATOMS if x[i] == u and x[j] == v) == target
    return ok
okL8 &= trop_liftable(0) and not any(trop_liftable(F(k, 4)) for k in range(1, 9))
check("L8 val commutes with marginals; agreement rate a > 0 is tropically non-liftable, a = 0 tropically liftable", okL8)

npass = sum(OUT.values()); print(f"\n{npass}/{len(OUT)} checks pass")
print("results SHA-256:", hashlib.sha256(json.dumps(OUT, sort_keys=True).encode()).hexdigest())
sys.exit(0 if npass == len(OUT) else 1)
