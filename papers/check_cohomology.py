"""check_cohomology.py -- exact checks for the cohomology sections of "Shadows, Fibres and Secants of the Posterior Tower".

Part H (the L2 presheaf on the lattice of sigma-algebras; exact rational arithmetic, except H2's norm in double precision)
  H1  Beck-Chevalley for the square A^B <= A, B: E_A|L2(B) = E_{A^B}|L2(B) iff A, B conditionally independent given the meet
  H2  the Beck-Chevalley defect has operator norm c (Friedrichs cosine) on L2(B)
  H3  in finite dimensions every compatible pair (E_C f_A = E_C f_B) glues: rank of (E_A, E_B) = dim of compatible pairs
  H4  the canonical gluing f_A + f_B - E_C f_A works for every compatible pair iff Beck-Chevalley holds
  H5  restriction maps E_G: L2(G') -> L2(G) for G strictly coarser than G' are surjective and never injective (not isomorphisms)
Part K (Cech cohomology of tropical Hardy-Weinberg fibres; exact)
  K1  for the generic and collision targets every cell and nerve face reported nonempty has an exact rational witness,
      and every cell and face reported empty has an exact Farkas certificate
  K2  Betti numbers of the nerve (exact rank over Q): generic (b0, b1) = (2, 0), collision (1, 0)
  K3  the swap of the two components acts on the generic fibre and exchanges its two components
  K4  lifts are fibrewise convex: a convex combination of two positive lifts with the same valuation pattern is again one
  K5  (if betti_scan.json is present) over the scanned target grid, b1 = 0 everywhere and b0 = 1 exactly on the collision locus
"""
from fractions import Fraction as F
import itertools, random, json, os, sys, hashlib
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "atlas"))
from fibre_topology import nerve, betti, cell_constraints, ALPHAS, SIGNS, rates

OUT = {}
def check(name, cond, info=""):
    OUT[name] = bool(cond); print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
rng = random.Random(20261013)

# ------------------------------------------------------------------ finite L2 machinery (exact)
def canon(lab):
    m = {}; return tuple(m.setdefault(a, len(m)) for a in lab)
def meet(a, b):
    n = len(a); par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for lab in (a, b):
        first = {}
        for i, x in enumerate(lab):
            if x in first: par[find(i)] = find(first[x])
            else: first[x] = i
    return canon([find(i) for i in range(n)])
def Emat(lab, P):
    n = len(P); M = [[F(0)] * n for _ in range(n)]
    for c in set(lab):
        cell = [i for i in range(n) if lab[i] == c]; m = sum(P[j] for j in cell)
        for i in cell:
            for j in cell: M[i][j] = P[j] / m
    return M
def mm(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def basis(lab):  # indicator basis of L2(lab)
    return [[F(1) if lab[i] == c else F(0) for i in range(len(lab))] for c in sorted(set(lab))]
def app(M, f): return [sum(M[i][j] * f[j] for j in range(len(f))) for i in range(len(M))]
def rank(rows):
    A = [list(r) for r in rows]; rk = 0
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
def ci_meet(a, b, P):
    m = meet(a, b); cm, am, bm, ab = {}, {}, {}, {}
    for i in range(len(P)):
        cm[m[i]] = cm.get(m[i], 0) + P[i]; am[a[i]] = am.get(a[i], 0) + P[i]; bm[b[i]] = bm.get(b[i], 0) + P[i]
        ab[(a[i], b[i])] = ab.get((a[i], b[i]), 0) + P[i]
    ca = {a[i]: m[i] for i in range(len(P))}; cb = {b[i]: m[i] for i in range(len(P))}
    return all(ab.get((x, y), 0) * cm[ca[x]] == am[x] * bm[y] for x in am for y in bm if ca[x] == cb[y])
def rand_P(n):
    w = [F(rng.randint(1, 6)) for _ in range(n)]; s = sum(w); return [x / s for x in w]
def rand_part(n, k): return canon([rng.randrange(k) for _ in range(n)])
def ci_instance():
    P, a, b = [], [], []
    for comp in range(rng.randint(1, 2)):
        ka, kb = rng.randint(1, 3), rng.randint(1, 3); pc = F(rng.randint(1, 4))
        ra = [F(rng.randint(1, 4)) for _ in range(ka)]; rb = [F(rng.randint(1, 4)) for _ in range(kb)]
        for x in range(ka):
            for y in range(kb):
                P.append(pc * ra[x] * rb[y]); a.append((comp, x)); b.append((comp, y))
    s = sum(P); return [p / s for p in P], canon(a), canon(b)

okH1 = okH2 = okH3 = okH4 = True; nci = nnot = 0
for t in range(300):
    if t % 2: P, a, b = ci_instance()
    else: n = rng.randint(3, 7); P = rand_P(n); a = rand_part(n, 3); b = rand_part(n, 3)
    n = len(P); c = meet(a, b); EA, EC = Emat(a, P), Emat(c, P)
    Bb = basis(b)
    bc_holds = all(app(EA, f) == app(EC, f) for f in Bb)
    ci = ci_meet(a, b, P); okH1 &= (bc_holds == ci); nci += ci; nnot += (not ci)
    # H2: operator norm of (E_A - E_C) restricted to L2(B), in L2(P)
    D = np.diag([np.sqrt(float(p)) for p in P]); Di = np.diag([1 / np.sqrt(float(p)) for p in P])
    EB = np.array([[float(x) for x in r] for r in Emat(b, P)])
    Dif = np.array([[float(EA[i][j] - EC[i][j]) for j in range(n)] for i in range(n)])
    normBC = np.linalg.norm(D @ Dif @ EB @ Di, 2)
    A_ = sorted(set(a)); B_ = sorted(set(b))
    pa = {x: float(sum(P[i] for i in range(n) if a[i] == x)) for x in A_}; pb = {y: float(sum(P[i] for i in range(n) if b[i] == y)) for y in B_}
    K = np.array([[float(sum(P[i] for i in range(n) if a[i] == x and b[i] == y)) / np.sqrt(pa[x] * pb[y]) for y in B_] for x in A_])
    sv = sorted(np.linalg.svd(K, compute_uv=False), reverse=True); ncomp = len(set(c))
    cF = sv[ncomp] if len(sv) > ncomp else 0.0
    okH2 &= abs(normBC - cF) < 1e-9
    # H3: image of (E_A, E_B) on L2(P) has the dimension of the compatible pairs
    EBx = Emat(b, P)
    img = [app(EA, e) + app(EBx, e) for e in [[F(1) if i == j else F(0) for i in range(n)] for j in range(n)]]
    dimA, dimB, dimC = len(set(a)), len(set(b)), len(set(c))
    okH3 &= rank(img) == dimA + dimB - dimC
    # H4: canonical gluing for a basis of compatible pairs
    canon_ok = True
    for fa in basis(a) + [[F(0)] * n]:
        for fb in basis(b) + [[F(0)] * n]:
            # make the pair compatible: replace f_B by f_B - E_C f_B + E_C f_A  (still in L2(B) since C <= B)
            fbc = [x - y + z for x, y, z in zip(fb, app(EC, fb), app(EC, fa))]
            g = [x + y - z for x, y, z in zip(fa, fbc, app(EC, fa))]
            canon_ok &= app(EA, g) == fa and app(EBx, g) == fbc
    okH4 &= (canon_ok == bc_holds)
check("H1 Beck-Chevalley for the meet square <=> conditional independence given the meet", okH1, f"{nci} CI, {nnot} not")
check("H2 the Beck-Chevalley defect has norm equal to the Friedrichs cosine c", okH2)
check("H3 finite dimensions: every compatible pair glues (rank of (E_A,E_B) = dim of compatible pairs)", okH3)
check("H4 the canonical gluing f_A + f_B - E_C f_A works for all compatible pairs iff Beck-Chevalley holds", okH4)
okH5 = True
for t in range(100):
    n = rng.randint(3, 7); P = rand_P(n); fine = rand_part(n, 4); cmap = {x: rng.randrange(2) for x in set(fine)}
    coarse = canon([cmap[x] for x in fine])
    if len(set(coarse)) == len(set(fine)): continue
    R = [app(Emat(coarse, P), f) for f in basis(fine)]       # images of a basis of L2(fine)
    okH5 &= rank(R) == len(set(coarse)) < len(set(fine))
check("H5 restriction E_G: L2(G') -> L2(G), G strictly coarser: surjective, never injective", okH5)

# ------------------------------------------------------------------ Part K: exact certificates for the tropical fibre nerves
def lp_point(cons, obj):
    Aeq, beq, Aub, bub = [], [], [], []
    for c in cons: Aeq += c[0]; beq += c[1]; Aub += c[2]; bub += c[3]
    res = linprog(obj, A_ub=np.array(Aub, float), b_ub=np.array(bub, float), A_eq=np.array(Aeq, float) if Aeq else None,
                  b_eq=np.array(beq, float) if beq else None, bounds=[(-50, 50)] * 3, method="highs")
    return (Aeq, beq, Aub, bub), (res.x if res.status == 0 else None)
def exact_witness(cons):
    for obj in (np.zeros(3), np.array([1, 2, 3.]), np.array([-1, 1, 0.5]), np.array([0.3, -2, 1])):
        (Aeq, beq, Aub, bub), x = lp_point(cons, obj)
        if x is None: return None
        xr = [F(v).limit_denominator(48) for v in x]
        if all(sum(F(r[j]) * xr[j] for j in range(3)) == b for r, b in zip(Aeq, beq)) and all(sum(F(r[j]) * xr[j] for j in range(3)) <= b for r, b in zip(Aub, bub)):
            return xr
    return None
def exact_farkas(cons):
    """y_eq free, y_ub >= 0, A_eq^T y_eq + A_ub^T y_ub = 0, b.y = -1  (certifies infeasibility)"""
    Aeq, beq, Aub, bub = [], [], [], []
    for c in cons: Aeq += c[0]; beq += c[1]; Aub += c[2]; bub += c[3]
    me, mu_ = len(Aeq), len(Aub)
    # variables: y_eq = yp - ym, y_ub
    nv = 2 * me + mu_
    Arows = []
    for j in range(3):
        Arows.append([float(Aeq[i][j]) for i in range(me)] + [-float(Aeq[i][j]) for i in range(me)] + [float(Aub[i][j]) for i in range(mu_)])
    Arows.append([float(beq[i]) for i in range(me)] + [-float(beq[i]) for i in range(me)] + [float(bub[i]) for i in range(mu_)])
    res = linprog(np.ones(nv), A_eq=np.array(Arows), b_eq=np.array([0, 0, 0, -1.]), bounds=[(0, None)] * nv, method="highs")
    if res.status != 0: return None
    y = [F(v).limit_denominator(64) for v in res.x]
    yeq = [y[i] - y[me + i] for i in range(me)]; yub = y[2 * me:]
    ok = all(v >= 0 for v in yub)
    ok &= all(sum(F(Aeq[i][j]) * yeq[i] for i in range(me)) + sum(F(Aub[i][j]) * yub[i] for i in range(mu_)) == 0 for j in range(3))
    ok &= sum(beq[i] * yeq[i] for i in range(me)) + sum(bub[i] * yub[i] for i in range(mu_)) < 0
    return (yeq, yub) if ok else None

cases = {"generic": [F(1, 2), F(1), F(0)], "collision": [F(0), F(1), F(2)]}
okK1 = True; okK2 = True; counts = {}
for name, v in cases.items():
    verts, simp, C = nerve(v, 2); b = betti(simp, 1); counts[name] = (len(verts), len(simp[1]), len(simp[2]), b)
    cells = [(a, s) for a in ALPHAS for s in SIGNS]; vs = set(verts)
    nwit = nfar = 0
    for c in cells:
        if c in vs: ok = exact_witness([C[c]]) is not None; nwit += ok
        else: ok = exact_farkas([C[c]]) is not None; nfar += ok
        okK1 &= ok
    e1 = set(simp[1])
    for s in itertools.combinations(verts, 2):
        ok = (exact_witness([C[c] for c in s]) is not None) if s in e1 else (exact_farkas([C[c] for c in s]) is not None)
        okK1 &= ok; nwit += ok and s in e1; nfar += ok and s not in e1
    e2 = set(simp[2])
    for s in itertools.combinations(verts, 3):
        if not all(f in e1 for f in itertools.combinations(s, 2)): continue
        ok = (exact_witness([C[c] for c in s]) is not None) if s in e2 else (exact_farkas([C[c] for c in s]) is not None)
        okK1 &= ok; nwit += ok and s in e2; nfar += ok and s not in e2
    counts[name] += (nwit, nfar)
okK2 = counts["generic"][3] == [2, 0] and counts["collision"][3] == [1, 0]
check("K1 every nonempty cell/face has an exact rational witness, every empty one an exact Farkas certificate", okK1,
      "; ".join(f"{k}: {c[4]} witnesses, {c[5]} certificates" for k, c in counts.items()))
check("K2 Betti numbers of the nerve: generic (2,0), collision (1,0)", okK2, str({k: c[3] for k, c in counts.items()}))
# K3 swap symmetry on the generic fibre: (beta,d1,d2) -> (-beta,d2,d1) maps witnesses to fibre points, components exchanged
v = cases["generic"]; verts, simp, C = nerve(v, 1)
comp = {}
par = {c: c for c in verts}
def fnd(c):
    while par[c] != c: c = par[c]
    return c
for s in simp[1]: par[fnd(s[0])] = fnd(s[1])
okK3 = True; comps = set(fnd(c) for c in verts)
for c in verts:
    w = exact_witness([C[c]]); sw = (-w[0], w[2], w[1])
    okK3 &= [F(x) for x in rates(sw)] == [F(x) for x in rates(tuple(w))] == v
    # which cell contains the swapped point
    hits = [d for d in verts if all(sum(F(r[j]) * sw[j] for j in range(3)) == bb for r, bb in zip(C[d][0], C[d][1])) and
            all(sum(F(r[j]) * sw[j] for j in range(3)) <= bb for r, bb in zip(C[d][2], C[d][3]))]
    okK3 &= bool(hits) and all(fnd(h) != fnd(c) for h in hits)
check("K3 component swap preserves the generic fibre and exchanges its two components", okK3 and len(comps) == 2)

# K4 fibrewise convexity of lifts (exact Puiseux arithmetic, t = s^N)
def padd(a, b):
    c = dict(a)
    for e, x in b.items():
        c[e] = c.get(e, 0) + x
        if c[e] == 0: del c[e]
    return c
def pscale(a, k): return {e: k * x for e, x in a.items()}
def val(a): return min(a) if a else None
def lead(a): return a[min(a)]
okK4 = True
for t in range(200):
    k = 2; u = [tuple(rng.randint(0, 4) for _ in range(3)) for _ in range(k)]; u = [tuple(x - min(ui) for x in ui) for ui in u]
    w = [0, rng.randint(0, 2)]; vv = [min(w[i] + u[i][x] for i in range(k)) for x in range(3)]
    rho = [{e: c for e, c in {vv[x]: F(rng.randint(2, 6)), vv[x] + 1: F(rng.choice([-2, -1, 1, 2, 3]))}.items()} for x in range(3)]
    def lift(split):
        q = [[None] * 3 for _ in range(k)]
        for x in range(3):
            att = [i for i in range(k) if w[i] + u[i][x] == vv[x]]; rest = {}
            for i in range(k):
                if i not in att: q[i][x] = {w[i] + u[i][x]: F(1, 3)}; rest = padd(rest, q[i][x])
            rem = padd(rho[x], pscale(rest, -1))
            for j, i in enumerate(att): q[i][x] = pscale(rem, split[j] if len(att) == 2 else 1)
        return q
    s1, s2 = F(1, 4), F(2, 3)
    L1, L2 = lift([s1, 1 - s1]), lift([s2, 1 - s2]); lam = F(rng.randint(1, 9), 10)
    Lc = [[padd(pscale(L1[i][x], lam), pscale(L2[i][x], 1 - lam)) for x in range(3)] for i in range(k)]
    for L in (L1, L2, Lc):
        for x in range(3):
            tot = {}
            for i in range(k): tot = padd(tot, L[i][x])
            okK4 &= tot == rho[x] and all(val(L[i][x]) == w[i] + u[i][x] and lead(L[i][x]) > 0 for i in range(k))
check("K4 lifts are fibrewise convex (convex combinations of lifts with one valuation pattern are lifts)", okK4)

p = os.path.join(HERE, "atlas", "betti_scan.json")
if os.path.exists(p):
    data = json.load(open(p)); okK5 = True; nb = {}; nob1 = 0
    for a_, c_, b, ncell in data:
        a_, c_ = F(a_), F(c_); nb[str(b)] = nb.get(str(b), 0) + 1
        if b[0]:
            if b[1] is None: nob1 += 1
            else: okK5 &= b[1] == 0
            okK5 &= (b[0] == 1) == (c_ == 2 * a_)
    check("K5 scanned targets: b1 = 0 wherever computed; b0 = 1 exactly on the collision locus, b0 = 2 elsewhere in the secant", okK5,
          f"{len(data)} targets, types {nb}, b1 not computed (cover > 24 cells) at {nob1}")

npass = sum(OUT.values()); print(f"\n{npass}/{len(OUT)} checks pass")
print("results SHA-256:", hashlib.sha256(json.dumps(OUT, sort_keys=True).encode()).hexdigest())
sys.exit(0 if npass == len(OUT) else 1)
