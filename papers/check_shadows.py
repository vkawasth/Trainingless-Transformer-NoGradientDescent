"""check_shadows.py -- exact checks for "Conditional Expectations, Beck Defects, and the Shadow Algebra of the Posterior Tower".

Exact rational arithmetic (fractions.Fraction) everywhere except the spectral checks A4/A4b, which use numpy in double precision
with tolerance 1e-9.  Tower definitions (lambda-sigma, C3, eta, mu) are imported verbatim from check_tower.py (same folder, or the parent folder).

Part A  finite conditional-expectation algebra
  A1  for finite partitions A, B of a full-support finite probability space the following agree on every trial:
      (i) E_A E_B = E_B E_A, (ii) E_A E_B = E_{A meet B}, (iii) A, B conditionally independent given A meet B,
      (iv) every connected component of the bipartite cell graph has a rank-one contingency block,
      (v) every singular value of K = D_A^{-1/2} N D_B^{-1/2} lies in {0, 1}
  A2  E_A E_B = E_{A join B} holds iff A = B
  A3  the paper's Example (P = 3/8,1/8,1/8,3/8): (E_R E_L Z)(1) = 15/8, (E_L E_R Z)(1) = 2; meet trivial, not independent
  A4  ||(E_A E_B)^n - E_{A meet B}|| = c^(2n-1), c = largest singular value of K below 1, n = 1..6
  A5  lax interchange: E_{meet} E_A E_B = E_{meet} = E_{meet} E_B E_A exactly, so both composites sit above E_meet in convex order;
      when A, B do not commute the two composites differ and both strictly exceed the meet in L2 energy
  A6  a sub-sigma-algebra C contained in A and B with A, B conditionally independent given C forces E_A E_B = E_C = E_B E_A and C = meet
  A7  Gaussian-type correction: channels X_L, X_R conditionally independent given H; sigma(X_L), sigma(X_R) need not commute,
      sigma(X_L,H), sigma(X_R,H) always commute with product E_{sigma(H)}
Part B  the canonical shadows of a tower context
  B1  the star space of a coherent context (Pi~ - r - {x, theta}) is coherent (mu Omega = Pi, D(q) Sigma = Pi, mu Pi = rho)
  B2  the shadow algebra G_T = sigma(r, leaves in T) is a commutative band: E_{G_T} E_{G_U} = E_{G_{T cap U}} for all T, U
  B3  without the centre, sigma(Pi~) and sigma(theta) fail to commute on some contexts
  B4  martingale: laws of E[Z|trivial], E[Z|Pi~], E[Z|Pi~,r], Z are delta_rho, D(mu)Omega, Pi, D(eta)rho
  B5  Markov absorption: E[Z|theta,r] = E[Z|r] for Z = delta_x; E_r E_theta Z = E_theta Z fails on some contexts
  B6  the two split idempotents e_bot = eta mu, e_top = D(eta) mu: idempotent, left-zero band, same splitting mu, natural;
      e_top affine, e_bot not affine; delta_rho <=cx Pi <=cx D(eta) rho with explicit martingale couplings
  B7  tower link: the level-one unit defect of lambda-sigma is (e_bot Pi, e_top Pi); the level-one multiplication defect is
      (Law W, Law E[W|group]) on the sample-index space
Part C  diagnosis of the uploaded scripts (their pooling function reproduced verbatim)
  C1  in the coherent generator the L and R cells coincide on every input type (committed regime)
  C2  a label-defined measure-valued composite R(L(nu)) is a single pooling onto a coarsening of L's partition
  C3  the naive composite can fail to interchange, and the fixed-space composite then also differs
  C4  nearest-atom pooling is discontinuous across a Voronoi wall: inputs 1/1000 apart in W1, outputs 501/2000 apart
  C5  check_span: (Cm) "defect equals sum" compares a pair with itself; "not inverse" holds because output keys change type
  C6  non-coherent Sigma: duplicated predictives tie and the first key wins, so R's partition equals L's
  C7  3-cell check: atoms of type D^2 against inputs of type D^1 are all at TV distance 1, so the pooling is total (e_bot)
Part D  moment varieties
  D1  the order-r moment map of a 2-component univariate Gaussian mixture has Jacobian rank min(r,5) at a generic rational point
      (order 2 fills the plane, so second moments cannot see mixtures)
  D2  affine statistics are constant on barycentre fibres: regrouping a mixture of mixtures does not change its moments
Part E  E1 the Kleisli-to-convex functor T is fully faithful on finite simplices (affine maps = stochastic matrices)
        E2 T preserves Kleisli composition and is l1 non-expansive
Part G  G1 the Gaussian-model fusion identity (discrete channels)
        G2 naturality of shadows along measure-preserving maps that pull back the partitions
        G3 star space functorial along context maps with D(f) injective on supp Pi; canonical shadows natural
"""
from fractions import Fraction as F
import itertools, random, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
def check(name, cond, info=""):
    OUT[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (("  [" + info + "]") if info else ""))

rng = random.Random(20261013)

# ============================================================ Part A: finite algebra
def cells(lab):
    d = {}
    for i, a in enumerate(lab): d.setdefault(a, []).append(i)
    return list(d.values())

def Emat(lab, P):
    n = len(P); M = [[F(0)] * n for _ in range(n)]
    for c in cells(lab):
        m = sum(P[j] for j in c)
        for i in c:
            for j in c: M[i][j] = P[j] / m
    return M

def mm(A, B):
    n = len(A); return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def apply(M, z):  # z: list of vectors (tuples) or scalars
    n = len(M)
    if isinstance(z[0], tuple):
        return [tuple(sum(M[i][k] * z[k][d] for k in range(n)) for d in range(len(z[0]))) for i in range(n)]
    return [sum(M[i][k] * z[k] for k in range(n)) for i in range(n)]

def canon(lab):  # canonical relabelling of a partition
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

def join(a, b): return canon(list(zip(a, b)))

def ci_given_meet(a, b, P):
    m = meet(a, b)
    comp_mass = {}; amass = {}; bmass = {}; ab = {}
    for i in range(len(P)):
        comp_mass[m[i]] = comp_mass.get(m[i], 0) + P[i]
        amass[a[i]] = amass.get(a[i], 0) + P[i]; bmass[b[i]] = bmass.get(b[i], 0) + P[i]
        ab[(a[i], b[i])] = ab.get((a[i], b[i]), 0) + P[i]
    comp_of_a = {a[i]: m[i] for i in range(len(P))}; comp_of_b = {b[i]: m[i] for i in range(len(P))}
    for x in amass:
        for y in bmass:
            if comp_of_a[x] != comp_of_b[y]: continue
            if ab.get((x, y), 0) * comp_mass[comp_of_a[x]] != amass[x] * bmass[y]: return False
    return True

def rank_frac(rows):
    M = [list(r) for r in rows]; rk = 0; ncol = len(M[0]) if M else 0
    for c in range(ncol):
        piv = next((r for r in range(rk, len(M)) if M[r][c] != 0), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]
        for r in range(len(M)):
            if r != rk and M[r][c] != 0:
                f = M[r][c] / M[rk][c]; M[r] = [M[r][k] - f * M[rk][k] for k in range(ncol)]
        rk += 1
    return rk

def rank_one_blocks(a, b, P):
    m = meet(a, b)
    for comp in set(m):
        idx = [i for i in range(len(P)) if m[i] == comp]
        A = sorted(set(a[i] for i in idx)); B = sorted(set(b[i] for i in idx))
        N = [[sum(P[i] for i in idx if a[i] == x and b[i] == y) for y in B] for x in A]
        if rank_frac(N) != 1: return False
    return True

def K_matrix(a, b, P):
    A = sorted(set(a)); B = sorted(set(b))
    pa = {x: float(sum(P[i] for i in range(len(P)) if a[i] == x)) for x in A}
    pb = {y: float(sum(P[i] for i in range(len(P)) if b[i] == y)) for y in B}
    N = np.array([[float(sum(P[i] for i in range(len(P)) if a[i] == x and b[i] == y)) for y in B] for x in A])
    return np.diag([1 / np.sqrt(pa[x]) for x in A]) @ N @ np.diag([1 / np.sqrt(pb[y]) for y in B])

def rand_P(n):
    w = [F(rng.randint(1, 6)) for _ in range(n)]; s = sum(w); return [x / s for x in w]

def rand_part(n, k): return canon([rng.randrange(k) for _ in range(n)])

def ci_instance():
    """components x (a-cells) x (b-cells), product weights inside each component -> CI given meet by construction"""
    idx = []; P = []; a = []; b = []
    for comp in range(rng.randint(1, 2)):
        ka, kb = rng.randint(1, 3), rng.randint(1, 3)
        pc = F(rng.randint(1, 4)); ra = [F(rng.randint(1, 4)) for _ in range(ka)]; rb = [F(rng.randint(1, 4)) for _ in range(kb)]
        for x in range(ka):
            for y in range(kb):
                reps = rng.randint(1, 2)
                for r in range(reps):
                    P.append(pc * ra[x] * rb[y] / reps); a.append((comp, x)); b.append((comp, y))
    s = sum(P); return [p / s for p in P], canon(a), canon(b)

def nested_instance(n):
    a = rand_part(n, 4); coarse = {x: rng.randrange(2) for x in set(a)}
    return rand_P(n), a, canon([coarse[x] for x in a])

stats = {"commute": 0, "noncommute": 0}
okA1 = True
for t in range(600):
    kind = t % 3
    if kind == 0: n = rng.randint(3, 7); P = rand_P(n); a = rand_part(n, 3); b = rand_part(n, 3)
    elif kind == 1: P, a, b = ci_instance()
    else: n = rng.randint(3, 7); P, a, b = nested_instance(n)
    EA, EB, EM = Emat(a, P), Emat(b, P), Emat(meet(a, b), P)
    i1 = mm(EA, EB) == mm(EB, EA); i2 = mm(EA, EB) == EM; i3 = ci_given_meet(a, b, P); i4 = rank_one_blocks(a, b, P)
    sv = np.linalg.svd(K_matrix(a, b, P), compute_uv=False)
    i5 = all(abs(s) < 1e-9 or abs(s - 1) < 1e-9 for s in sv)
    okA1 &= (i1 == i2 == i3 == i4 == i5)
    stats["commute" if i1 else "noncommute"] += 1
check("A1 commute <=> product is E_meet <=> CI given meet <=> rank-one blocks <=> singular values in {0,1}", okA1,
      f"{stats['commute']} commuting, {stats['noncommute']} non-commuting trials")

okA2 = True; eq_cnt = 0
for t in range(400):
    n = rng.randint(2, 6); P = rand_P(n); a = rand_part(n, 3)
    b = a if t % 4 == 0 else rand_part(n, 3)
    holds = mm(Emat(a, P), Emat(b, P)) == Emat(join(a, b), P)
    okA2 &= (holds == (canon(a) == canon(b))); eq_cnt += holds
check("A2 E_A E_B = E_{A join B} iff A = B", okA2, f"identity held in {eq_cnt}/400 trials, all with A = B")

P46 = [F(3, 8), F(1, 8), F(1, 8), F(3, 8)]; Z46 = [F(1), F(2), F(3), F(4)]
aL = (0, 0, 1, 1); aR = (0, 1, 0, 1)
RL = apply(mm(Emat(aR, P46), Emat(aL, P46)), Z46); LR = apply(mm(Emat(aL, P46), Emat(aR, P46)), Z46)
check("A3 Example: (E_R E_L Z)(1) = 15/8, (E_L E_R Z)(1) = 2, meet trivial, not CI",
      RL[0] == F(15, 8) and LR[0] == 2 and len(set(meet(aL, aR))) == 1 and not ci_given_meet(aL, aR, P46))

def l2norm_op(M, P):
    D = np.diag([np.sqrt(float(p)) for p in P]); Di = np.diag([1 / np.sqrt(float(p)) for p in P])
    return np.linalg.norm(D @ np.array([[float(x) for x in r] for r in M]) @ Di, 2)

okA4 = True; tested = 0
for t in range(200):
    n = rng.randint(4, 7); P = rand_P(n); a = rand_part(n, 3); b = rand_part(n, 3)
    EA, EB, EM = Emat(a, P), Emat(b, P), Emat(meet(a, b), P)
    if mm(EA, EB) == EM: continue
    sv = sorted(np.linalg.svd(K_matrix(a, b, P), compute_uv=False), reverse=True)
    ncomp = len(set(meet(a, b))); c = sv[ncomp] if len(sv) > ncomp else 0.0
    T = np.eye(n); AB = np.array([[float(x) for x in r] for r in mm(EA, EB)]); EMf = np.array([[float(x) for x in r] for r in EM])
    D = np.diag([np.sqrt(float(p)) for p in P]); Di = np.diag([1 / np.sqrt(float(p)) for p in P])
    for k in range(1, 7):
        T = T @ AB
        val = np.linalg.norm(D @ (T - EMf) @ Di, 2)
        okA4 &= abs(val - c ** (2 * k - 1)) < 1e-9
    tested += 1
check("A4 ||(E_A E_B)^n - E_meet|| = c^(2n-1), c = Friedrichs cosine, n = 1..6", okA4 and tested > 50, f"{tested} non-commuting pairs")

okA5 = True; strict = 0; differ = 0; ntest = 0
for t in range(300):
    n = rng.randint(3, 7); P = rand_P(n); a = rand_part(n, 3); b = rand_part(n, 3)
    EA, EB, EM = Emat(a, P), Emat(b, P), Emat(meet(a, b), P)
    okA5 &= mm(EM, mm(EA, EB)) == EM and mm(EM, mm(EB, EA)) == EM
    if mm(EA, EB) != mm(EB, EA):
        ntest += 1
        Z = [tuple(F(rng.randint(0, 5)) for _ in range(2)) for _ in range(n)]
        u, v, w = apply(mm(EA, EB), Z), apply(mm(EB, EA), Z), apply(EM, Z)
        en = lambda y: sum(P[i] * sum(c * c for c in y[i]) for i in range(n))
        differ += (u != v); strict += (en(u) > en(w) or u == w) and (en(v) > en(w) or v == w)
check("A5 lax interchange: E_meet E_A E_B = E_meet = E_meet E_B E_A; composites above the meet", okA5 and strict == ntest,
      f"{ntest} non-commuting pairs, composites differ on a random vector in {differ}")

okA6 = True
for t in range(200):
    # random C-structure: index = (c, u, v) with P = p(c) p(u|c) p(v|c); A = sigma(c,u), B = sigma(c,v)
    idx = []; P = []
    for c in range(rng.randint(1, 3)):
        pc = F(rng.randint(1, 4)); us = [F(rng.randint(1, 4)) for _ in range(rng.randint(1, 3))]; vs = [F(rng.randint(1, 4)) for _ in range(rng.randint(1, 3))]
        for iu, pu in enumerate(us):
            for iv, pv in enumerate(vs): idx.append((c, iu, iv)); P.append(pc * pu / sum(us) * pv / sum(vs))
    s = sum(P); P = [p / s for p in P]
    a = canon([(c, u) for c, u, v in idx]); b = canon([(c, v) for c, u, v in idx]); cc = canon([c for c, u, v in idx])
    EA, EB, EC = Emat(a, P), Emat(b, P), Emat(cc, P)
    okA6 &= mm(EA, EB) == EC == mm(EB, EA) and canon(meet(a, b)) == cc
check("A6 CI given some C inside A and B => E_A E_B = E_C = E_B E_A and C = meet", okA6)

okA7a = False; okA7b = True
for t in range(200):
    H = [0, 1, 2]; pH = rand_P(3)
    chL = [rand_P(2) for _ in H]; chR = [rand_P(2) for _ in H]
    idx = [(h, xl, xr) for h in H for xl in (0, 1) for xr in (0, 1)]
    P = [pH[h] * chL[h][xl] * chR[h][xr] for h, xl, xr in idx]
    aXL = canon([xl for h, xl, xr in idx]); aXR = canon([xr for h, xl, xr in idx])
    aHL = canon([(h, xl) for h, xl, xr in idx]); aHR = canon([(h, xr) for h, xl, xr in idx]); aH = canon([h for h, xl, xr in idx])
    EXL, EXR = Emat(aXL, P), Emat(aXR, P)
    if mm(EXL, EXR) != mm(EXR, EXL): okA7a = True
    okA7b &= mm(Emat(aHL, P), Emat(aHR, P)) == Emat(aH, P) == mm(Emat(aHR, P), Emat(aHL, P))
check("A7 X_L, X_R CI given H: sigma(X_L), sigma(X_R) do not commute (some instance)", okA7a)
check("A7b sigma(X_L,H), sigma(X_R,H) always commute with product E_sigma(H)", okA7b)

# ============================================================ Part B: canonical shadows of a context
# tower definitions verbatim
_tw = os.path.join(HERE, 'check_tower.py')
src = open(_tw if os.path.exists(_tw) else os.path.join(HERE, '..', 'check_tower.py')).read()
_defs = src.split('# ---------- comonad ----------')[0]
_ns = {}
exec(_defs, _ns)
exec(src[src.index('def lam_bar(M):'):src.index('okn = oku = okc = okm = True')], _ns)
exec(src[src.index('def K1(h):'):src.index('oksn = oksc = True')], _ns)
Dist, delta, Dmap, mu, mix = _ns['Dist'], _ns['delta'], _ns['Dmap'], _ns['mu'], _ns['mix']
C, E, eta, lam_sig = _ns['C'], _ns['E'], _ns['eta'], _ns['lam_sig']

Xb = (0, 1, 2)
def rdist(vals):
    k = rng.randint(1, min(3, len(vals))); vs = rng.sample(list(vals), k); ws = [F(rng.randint(1, 5)) for _ in vs]; s = sum(ws)
    return Dist([(v, w / s) for v, w in zip(vs, ws)])

def star_context():
    atoms = []
    while len(atoms) < 3:
        r = rdist(Xb)
        if r not in atoms: atoms.append(r)
    nlev = rng.randint(1, 3)
    Pis = [Dist([(r, w) for r, w in zip(atoms, ws)]) for ws in
           [[F(x) / s for x in xs] for xs in [[rng.randint(0, 3) or 1 for _ in atoms] for _ in range(nlev)] for s in [sum(xs)]]]
    wO = [F(rng.randint(1, 4)) for _ in Pis]; sO = sum(wO)
    Om = Dist([(p, w / sO) for p, w in zip(Pis, wO)])
    Pi = mu(Om); rho = mu(Pi)
    thetas = ("t1", "t2", "t3")
    kappa = {r: rdist(thetas) for r in atoms}  # theta | r
    Sig = Dist([((th, r), Pi.m.get(r, 0) * q) for r in atoms for th, q in kappa[r].items()])
    # star space: (Pt, r, th, x) with weight Om(Pt) Pt(r) kappa(th|r) r(x)
    idx, P = [], []
    for Pt, wo in Om.items():
        for r, wr in Pt.items():
            for th, wt in kappa[r].items():
                for x, wx in r.items():
                    idx.append((Pt, r, th, x)); P.append(wo * wr * wt * wx)
    return dict(rho=rho, Pi=Pi, Om=Om, Sig=Sig, idx=idx, P=P)

LEAVES = ("lev2", "hyp", "obs")
def G(ctx, T, centre=True):
    pos = {"lev2": 0, "hyp": 2, "obs": 3}
    keys = ([1] if centre else []) + [pos[l] for l in T]
    return canon([tuple(repr(t[k]) for k in keys) for t in ctx["idx"]])

def law(vals, P):
    return Dist([(v, p) for v, p in zip(vals, P)])

def vec(d):
    return tuple(d.m.get(x, F(0)) for x in Xb)
def todist(v):
    return Dist([(x, q) for x, q in zip(Xb, v) if q])

okB1 = okB2 = okB4 = okB5a = True; B3 = 0; B5b = 0; NB = 60
for t in range(NB):
    ctx = star_context(); P = ctx["P"]; idx = ctx["idx"]
    okB1 &= mu(ctx["Om"]) == ctx["Pi"] and Dmap(lambda h: h[1], ctx["Sig"]) == ctx["Pi"] and mu(ctx["Pi"]) == ctx["rho"]
    okB1 &= sum(P) == 1
    subsets = [frozenset(s) for k in range(4) for s in itertools.combinations(LEAVES, k)]
    Es = {T: Emat(G(ctx, T), P) for T in subsets}
    for T in subsets:
        for U in subsets:
            okB2 &= mm(Es[T], Es[U]) == Es[T & U]
    EPt, Eth = Emat(G(ctx, ["lev2"], centre=False), P), Emat(G(ctx, ["hyp"], centre=False), P)
    B3 += mm(EPt, Eth) != mm(Eth, EPt)
    Z = [vec(delta(t[3])) for t in idx]
    triv = canon([0] * len(idx))
    lawof = lambda M: Dmap(todist, law(apply(M, Z), P))
    okB4 &= lawof(Emat(triv, P)) == delta(ctx["rho"])
    okB4 &= lawof(Emat(G(ctx, ["lev2"], centre=False), P)) == Dmap(mu, ctx["Om"])
    okB4 &= lawof(Emat(G(ctx, ["lev2"]), P)) == ctx["Pi"]
    okB4 &= Dmap(todist, law(Z, P)) == Dmap(eta, ctx["rho"])
    Er = Emat(G(ctx, []), P)
    okB5a &= apply(Es[frozenset(["hyp"])], Z) == apply(Er, Z)
    Eth_only = Emat(G(ctx, ["hyp"], centre=False), P)
    B5b += apply(mm(Er, Eth_only), Z) != apply(Eth_only, Z)
check("B1 star space of a coherent context: mu Omega = Pi, D(q) Sigma = Pi, mu Pi = rho, total mass 1", okB1, f"{NB} contexts")
check("B2 shadow algebra: E_{G_T} E_{G_U} = E_{G_{T cap U}} for all leaf sets T, U (commutative band)", okB2)
check("B3 without the centre, sigma(Pi~) and sigma(theta) fail to commute on some contexts", B3 > 0, f"{B3}/{NB}")
check("B4 martingale laws: delta_rho, D(mu)Omega, Pi, D(eta)rho", okB4)
check("B5 Markov absorption E[Z|theta,r] = E[Z|r]; E_r E_theta Z != E_theta Z on some contexts", okB5a and B5b > 0, f"{B5b}/{NB}")

e_bot = lambda Pi: delta(mu(Pi))
e_top = lambda Pi: Dmap(eta, mu(Pi))
okB6 = True; nonaff = 0
fmap = {0: "a", 1: "a", 2: "b"}.get
for t in range(80):
    Pi = rdist([rdist(Xb) for _ in range(4)]); Pi2 = rdist([rdist(Xb) for _ in range(4)])
    okB6 &= e_bot(e_bot(Pi)) == e_bot(Pi) and e_top(e_top(Pi)) == e_top(Pi)
    okB6 &= e_bot(e_top(Pi)) == e_bot(Pi) and e_top(e_bot(Pi)) == e_top(Pi)
    okB6 &= mu(e_bot(Pi)) == mu(Pi) == mu(e_top(Pi))
    D2f = lambda Q: Dmap(lambda r: Dmap(fmap, r), Q)
    okB6 &= D2f(e_bot(Pi)) == e_bot(D2f(Pi)) and D2f(e_top(Pi)) == e_top(D2f(Pi))
    s = F(rng.randint(1, 3), 4)
    m = mix([(s, Pi), (1 - s, Pi2)])
    okB6 &= e_top(m) == mix([(s, e_top(Pi)), (1 - s, e_top(Pi2))])
    nonaff += e_bot(m) != mix([(s, e_bot(Pi)), (1 - s, e_bot(Pi2))])
    # martingale couplings: delta_rho -> Pi (all mass of rho to atoms of Pi), Pi -> D(eta)rho (r -> delta_x with prob r(x))
    rho = mu(Pi)
    okB6 &= mix([(w, r) for r, w in Pi.items()]) == rho
    cpl = {}
    for r, w in Pi.items():
        for x, q in r.items(): cpl[(r, x)] = cpl.get((r, x), 0) + w * q
    okB6 &= all(sum(cpl[(r, x)] * (1 if x == y else 0) for x in Xb if (r, x) in cpl) / w == r.m.get(y, 0)
                for r, w in Pi.items() for y in Xb)
    okB6 &= Dist([(delta(x), p) for (r, x), p in cpl.items()]) == Dmap(eta, rho)
check("B6 e_bot, e_top: idempotent, left-zero band, same splitting mu, natural, e_top affine, sandwich couplings", okB6)
check("B6b e_bot is not affine (some mixture)", nonaff > 0, f"{nonaff}/80")

okB7 = True
for t in range(30):
    p = _ns['rpoint'](_ns['X'])
    lhs = lam_sig(delta(p)); rhs = C(eta)(p)
    Pi1 = rhs[1][0]
    okB7 &= lhs[1][0] == e_bot(Pi1) and rhs[1][0] == e_top(lhs[1][0])
    # multiplication: M' in D^2(C3 X)
    Ms = [_ns['rM'](_ns['X'], 2) for _ in range(2)]
    Mp = Dist([(m, F(1, 2)) for m in Ms]) if Ms[0] != Ms[1] else delta(Ms[0])
    L = lam_sig(mu(Mp))[1][0]
    Rr = C(mu)(lam_sig(Dmap(lam_sig, Mp)))[1][0]
    idx = []; P = []
    for m, nu in Mp.items():
        for (x, c), a in m.items(): idx.append((m, c[0])); P.append(nu * a)
    grp = canon([repr(i[0]) for i in idx])
    W = [vec(i[1]) for i in idx]
    okB7 &= Dmap(todist, law(W, P)) == L
    okB7 &= Dmap(todist, law(apply(Emat(grp, P), W), P)) == Rr
check("B7 tower link: (U) defect of lambda-sigma = (e_bot Pi, e_top Pi); (M) defect = (Law W, Law E[W|group])", okB7)

# ============================================================ Part C: diagnosis of the uploaded scripts
def canonical(d): return tuple(sorted(d.items(), key=lambda kv: repr(kv[0])))
def mk(items):
    d = {}
    for k, p in items: d[k] = d.get(k, F(0)) + F(p)
    return canonical(d)
def barycenter(dd):
    out = {}
    for rk, p in dd:
        for b, q in rk: out[b] = out.get(b, F(0)) + p * q
    return canonical(out)
def tv(d1, d2):
    a, b = dict(d1), dict(d2); return sum(abs(a.get(k, F(0)) - b.get(k, F(0))) for k in set(a) | set(b)) / 2
def is_dist(x):
    return isinstance(x, tuple) and len(x) > 0 and all(isinstance(i, tuple) and len(i) == 2 for i in x)
def is_hyp(x): return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], int) and is_dist(x[1])
def dist_of(x): return x[1] if is_hyp(x) else x
def tad(a, b):
    if is_dist(a) and is_dist(b): return tv(a, b)
    if is_hyp(a) and is_hyp(b): return F(1) if a[0] != b[0] else tv(a[1], b[1])
    if (is_dist(a) and is_hyp(b)) or (is_hyp(a) and is_dist(b)): return tv(dist_of(a), dist_of(b))
    return F(1)
def assign(nu, atoms):
    g = {}
    for k, p in dict(nu).items():
        best, bd = None, None
        for a in atoms:
            d = tad(k, a)
            if bd is None or d < bd: bd, best = d, a
        g.setdefault(best, []).append(k)
    return g
def pooling(nu, atoms):
    d = dict(nu); out = {}
    for a, ks in assign(nu, atoms).items():
        mass = sum(d[k] for k in ks); w = {}
        for k in ks:
            for b, q in dict(dist_of(k)).items(): w[b] = w.get(b, F(0)) + d[k] / mass * q
        bc = canonical(w); out[bc] = out.get(bc, F(0)) + mass
    return canonical(out)
def random_rho(n, r): return mk([(i, r.randint(1, 5)) for i in range(n)])
def coh_ctx(n, r):
    r1 = random_rho(n, r); r2 = random_rho(n, r)
    while r1 == r2: r2 = random_rho(n, r)
    w11 = F(r.randint(1, 4), 5); P1 = mk([(r1, w11), (r2, 1 - w11)])
    w21 = F(r.randint(1, 4), 5); P2 = mk([(r1, w21), (r2, 1 - w21)])
    o = F(r.randint(1, 4), 5); Om = mk([(P1, o), (P2, 1 - o)]); Pi = barycenter(Om)
    Sig = mk([((i % 2, j), w) for i, (j, w) in enumerate(dict(Pi).items())])
    return (barycenter(Pi), Pi, Om, Sig)
def partition_of(nu, atoms): return sorted(sorted(map(repr, ks)) for ks in assign(nu, atoms).values())

okC1 = True; r = random.Random(42); natoms = set()
for n in (2, 3, 4):
    for t in range(300):
        c = coh_ctx(n, r); Pi, Sig = c[1], c[3]; natoms.add(len(Pi))
        for nu in (mk([(mk([(b, 1)]), F(1, n)) for b in range(n)]),
                   mk([(h, F(1, len(Sig))) for h, _ in Sig])):
            okC1 &= partition_of(nu, [k for k, _ in Pi]) == partition_of(nu, [k for k, _ in Sig])
        okC1 &= len(set(th for (th, _), _ in Sig)) == len(Sig) == len(Pi)
check("C1 coherent generator: Pi has at most 2 atoms, descriptors biject with atoms, L and R cells coincide", okC1 and natoms <= {1, 2},
      f"atom counts seen {sorted(natoms)}")

# C2/C3: threshold labels on the segment D({0,1}) = [0,1] (coordinate = mass on 1)
def seg(s): return mk([(0, 1 - s), (1, s)]) if 0 < s < 1 else mk([(1 if s == 1 else 0, 1)])
def thr_pool(nu, t):  # pool by label s < t vs s >= t
    d = dict(nu); g = {}
    for k, p in d.items(): g.setdefault(dict(k).get(1, F(0)) >= t, []).append(k)
    out = {}
    for lab, ks in g.items():
        m = sum(d[k] for k in ks); bc = canonical({b: sum(d[k] * dict(k).get(b, F(0)) for k in ks) / m for b in (0, 1)})
        bc = canonical({b: q for b, q in dict(bc).items() if q}); out[bc] = out.get(bc, F(0)) + m
    return canonical(out)
pts = [F(1, 10), F(3, 10), F(7, 10), F(9, 10)]
nu = mk([(seg(s), F(1, 4)) for s in pts])
tA, tB = F(1, 2), F(1, 4)
RL_naive = thr_pool(thr_pool(nu, tA), tB); LR_naive = thr_pool(thr_pool(nu, tB), tA)
labA = canon([s >= tA for s in pts]); labB = canon([s >= tB for s in pts]); Pn = [F(1, 4)] * 4
Zs = [s for s in pts]
fixed_RL = apply(mm(Emat(labB, Pn), Emat(labA, Pn)), Zs); fixed_LR = apply(mm(Emat(labA, Pn), Emat(labB, Pn)), Zs)
okC2 = True
for t in range(200):  # composite = one pooling onto a coarsening of the first partition
    k = rng.randint(3, 6); ss = sorted(set(F(rng.randint(1, 19), 20) for _ in range(k)))
    nu_ = mk([(seg(s), F(1, len(ss))) for s in ss]); t1, t2 = F(rng.randint(1, 19), 20), F(rng.randint(1, 19), 20)
    lab1 = [s >= t1 for s in ss]; P_ = [F(1, len(ss))] * len(ss)
    m1 = apply(Emat(canon(lab1), P_), ss); lab2 = [x >= t2 for x in m1]
    coarse = canon(list(zip(lab1, lab2)))
    okC2 &= all(len(set(lab2[i] for i in range(len(ss)) if lab1[i] == v)) == 1 for v in set(lab1))  # sigma(a2 o m1) is coarser than sigma(a1)
    m2 = apply(Emat(canon(lab2), P_), m1)
    okC2 &= thr_pool(thr_pool(nu_, t1), t2) == mk([(seg(v), p) for v, p in zip(m2, P_)])
    okC2 &= m2 == apply(Emat(canon(lab2), P_), ss)  # single pooling of the ORIGINAL points onto sigma(a2 o m1), coarser than sigma(a1)
check("C2 naive composite R(L(nu)) = Law(E_{sigma(a_R o m_L)} Z), a single pooling onto a coarsening of L's partition", okC2)
check("C3 naive interchange fails on the threshold example; fixed-space composites also differ",
      RL_naive != LR_naive and fixed_RL != fixed_LR, f"R(L nu)={[(str(dict(k).get(1,0)), str(p)) for k,p in RL_naive]}, "
      f"L(R nu)={[(str(dict(k).get(1,0)), str(p)) for k,p in LR_naive]}")

def W1_seg(n1, n2):  # W1 on [0,1] between laws of the coordinate s
    pts_ = sorted(set([dict(k).get(1, F(0)) for k, _ in n1] + [dict(k).get(1, F(0)) for k, _ in n2]))
    cdf = lambda nn, x: sum(p for k, p in nn if dict(k).get(1, F(0)) <= x)
    return sum(abs(cdf(n1, pts_[i]) - cdf(n2, pts_[i])) * (pts_[i + 1] - pts_[i]) for i in range(len(pts_) - 1))
atoms_C4 = [seg(F(2, 3)), seg(F(1, 3))]
def nu_t(tt): return mk([(seg(F(1, 2) + tt), F(1, 2)), (seg(F(1)), F(1, 2))])
jump = W1_seg(pooling(nu_t(F(1, 1000)), atoms_C4), pooling(nu_t(F(-1, 1000)), atoms_C4))
inp = W1_seg(nu_t(F(1, 1000)), nu_t(F(-1, 1000)))
check("C4 nearest-atom pooling is discontinuous: inputs 1/1000 apart in W1, outputs more than 1/4 apart", inp == F(1, 1000) and jump > F(1, 4),
      f"W1(in)={inp}, W1(out)={jump}")

okC5 = True; r = random.Random(42)
for t in range(300):
    c = coh_ctx(3, r); Sig = c[3]; nu = mk([(h, F(1, len(Sig))) for h, _ in Sig])
    Lout = pooling(nu, [k for k, _ in c[1]])
    okC5 &= not (set(k for k, _ in Lout) & set(k for k, _ in nu)) and all(not is_hyp(k) for k, _ in Lout)
    okC5 &= (Lout, pooling(nu, [k for k, _ in Sig])) == (pooling(nu, [k for k, _ in c[1]]), pooling(nu, [k for k, _ in Sig]))
check("C5 check_span: outputs drop theta (keys change type, so 'not inverse' is automatic); (Cm) test compares a pair with itself", okC5)

okC6 = True; r = random.Random(43)
for t in range(300):
    n = 3; r1 = random_rho(n, r); r2 = random_rho(n, r)
    while r1 == r2: r2 = random_rho(n, r)
    Sig = mk([((0, r1), F(1, 3)), ((1, r1), F(1, 3)), ((0, r2), F(1, 3))])
    nu = mk([(mk([(b, 1)]), F(1, n)) for b in range(n)])
    g = assign(nu, [k for k, _ in Sig])
    okC6 &= (1, r1) not in g
    okC6 &= partition_of(nu, [k for k, _ in Sig]) == partition_of(nu, [r1, r2])
check("C6 non-coherent Sigma: tied duplicate (1,rho1) never receives mass; R cells = L cells", okC6)

okC7 = True; r = random.Random(44)
for t in range(300):
    c = coh_ctx(3, r); Pi = c[1]
    atoms5 = [k for k, _ in mk([(mk([(rr, 1)]), w) for rr, w in Pi])]
    nu = mk([(mk([(b, 1)]), F(1, 3)) for b in range(3)])
    okC7 &= pooling(nu, atoms5) == mk([(barycenter(nu), 1)])
check("C7 3-cell check: pooling onto D^2-typed atoms is total pooling e_bot (all TV distances 1)", okC7)

# ============================================================ Part D: moment varieties
import sympy as sp
w, m1, s1, m2, s2, x = sp.symbols('w m1 s1 m2 s2 x')
def gauss_moments(m, s, r):  # raw moments of N(m, s) (s = variance), via recursion E[X^k] = m E[X^{k-1}] + (k-1) s E[X^{k-2}]
    M = [sp.Integer(1), m]
    for k in range(2, r + 1): M.append(sp.expand(m * M[k - 1] + (k - 1) * s * M[k - 2]))
    return M[1:r + 1]
pt = {w: sp.Rational(1, 3), m1: sp.Rational(-1, 2), s1: sp.Rational(2, 3), m2: sp.Rational(3, 4), s2: sp.Rational(5, 4)}
ranks = {}
for rr in range(2, 7):
    mix_m = [w * a + (1 - w) * b for a, b in zip(gauss_moments(m1, s1, rr), gauss_moments(m2, s2, rr))]
    J = sp.Matrix(mix_m).jacobian([w, m1, s1, m2, s2]).subs(pt)
    ranks[rr] = J.rank()
check("D1 Jacobian rank of the order-r moment map of a 2-Gaussian mixture is min(r,5)", all(ranks[k] == min(k, 5) for k in ranks),
      str(ranks))
okD2 = True
for t in range(50):
    comps = [rdist(Xb) for _ in range(4)]; wts = rand_P(4)
    flat = mix(list(zip(wts, comps)))
    g1 = [(wts[0] + wts[1], mix([(wts[0] / (wts[0] + wts[1]), comps[0]), (wts[1] / (wts[0] + wts[1]), comps[1])])),
          (wts[2] + wts[3], mix([(wts[2] / (wts[2] + wts[3]), comps[2]), (wts[3] / (wts[2] + wts[3]), comps[3])]))]
    hier = Dist([(d, p) for p, d in g1]) if g1[0][1] != g1[1][1] else delta(g1[0][1])
    okD2 &= mu(hier) == flat
check("D2 regrouping a mixture of mixtures leaves the flattened law (hence every affine statistic) unchanged", okD2)

# ============================================================ Part E
okE = True
for t in range(50):
    nx, ny = rng.randint(1, 4), rng.randint(1, 4)
    K = [rand_P(ny) for _ in range(nx)]  # K[x][y]
    T = lambda p, K=K: [sum(p[x] * K[x][y] for x in range(len(p))) for y in range(len(K[0]))]
    okE &= all(T([F(1) if i == x else F(0) for i in range(nx)]) == K[x] for x in range(nx))
    p, q, s = rand_P(nx), rand_P(nx), F(rng.randint(0, 4), 4)
    okE &= T([s * a + (1 - s) * b for a, b in zip(p, q)]) == [s * a + (1 - s) * b for a, b in zip(T(p), T(q))]
check("E1 affine maps of finite simplices are exactly stochastic matrices (T fully faithful)", okE)

okE2 = True
for t in range(100):
    nx, ny, nz = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)
    K = [rand_P(ny) for _ in range(nx)]; L = [rand_P(nz) for _ in range(ny)]
    TK = lambda p: [sum(p[x] * K[x][y] for x in range(nx)) for y in range(ny)]
    TL = lambda p: [sum(p[y] * L[y][z] for y in range(ny)) for z in range(nz)]
    LK = [[sum(K[x][y] * L[y][z] for y in range(ny)) for z in range(nz)] for x in range(nx)]
    p, q = rand_P(nx), rand_P(nx)
    okE2 &= [sum(p[x] * LK[x][z] for x in range(nx)) for z in range(nz)] == TL(TK(p))
    okE2 &= sum(abs(a - b) for a, b in zip(TK(p), TK(q))) <= sum(abs(a - b) for a, b in zip(p, q))
check("E2 T preserves Kleisli composition and is l1 non-expansive", okE2)

# ============================================================ Part G: restored material
okG1 = True
for t in range(200):
    H = [0, 1, 2]; pi = rand_P(3)
    chL = [rand_P(3) for _ in H]; chR = [rand_P(3) for _ in H]
    for xl in range(3):
        for xr in range(3):
            qL = [pi[h] * chL[h][xl] for h in H]; s1 = sum(qL); qL = [a / s1 for a in qL]
            qR = [pi[h] * chR[h][xr] for h in H]; s2 = sum(qR); qR = [a / s2 for a in qR]
            qLR = [pi[h] * chL[h][xl] * chR[h][xr] for h in H]; s3 = sum(qLR); qLR = [a / s3 for a in qLR]
            fu = [qL[h] * qR[h] / pi[h] for h in H]; s4 = sum(fu); fu = [a / s4 for a in fu]
            okG1 &= fu == qLR
check("G1 fusion identity Q_LR = Fuse(Q_L, Q_R) under X_L, X_R conditionally independent given H", okG1)

okG2 = True
for t in range(200):
    # f: I -> I' measure preserving (I' = cells of a random map), partitions A', B' on I' pulled back
    n2 = rng.randint(2, 5); P2 = rand_P(n2)
    n = rng.randint(n2, n2 + 4); fmap = list(range(n2)) + [rng.randrange(n2) for _ in range(n - n2)]
    # split the mass of each i' among its preimages
    P = [F(0)] * n
    for i2 in range(n2):
        pre = [i for i in range(n) if fmap[i] == i2]; ws = [F(rng.randint(1, 4)) for _ in pre]
        for i, wgt in zip(pre, ws): P[i] = P2[i2] * wgt / sum(ws)
    A2 = rand_part(n2, 3); B2 = rand_part(n2, 3)
    A = canon([A2[fmap[i]] for i in range(n)]); B = canon([B2[fmap[i]] for i in range(n)])
    Z2 = [tuple(F(rng.randint(0, 6)) for _ in range(2)) for _ in range(n2)]
    Z = [Z2[fmap[i]] for i in range(n)]
    for a, a2 in ((A, A2), (B, B2)):
        lhs = apply(Emat(a, P), Z); rhs = [apply(Emat(a2, P2), Z2)[fmap[i]] for i in range(n)]
        okG2 &= lhs == rhs
check("G2 naturality: E_{f^-1 A'}(Z' o f) = (E_{A'} Z') o f for measure-preserving f", okG2)

okG3 = True; ntested = 0
Yb = ("a", "b")
for t in range(40):
    ctx = star_context()
    fm = {0: "a", 1: "a", 2: "b"} if t % 2 else {0: "a", 1: "b", 2: "b"}
    Df = lambda r: Dmap(fm.get, r)
    atoms = [r for r, _ in ctx["Pi"].items()]
    if len(set(Df(r) for r in atoms)) < len(atoms): continue   # need Df injective on supp Pi
    ntested += 1
    phi = lambda pt: (Dmap(Df, pt[0]), Df(pt[1]), pt[2], fm[pt[3]])
    push = {}
    for pt, w in zip(ctx["idx"], ctx["P"]): push[phi(pt)] = push.get(phi(pt), 0) + w
    # star space of the pushed-forward context
    Om2 = Dmap(lambda Pt: Dmap(Df, Pt), ctx["Om"]); Pi2 = mu(Om2)
    Sig2 = Dmap(lambda h: (h[0], Df(h[1])), ctx["Sig"])
    star2 = {}
    for Pt, wo in Om2.items():
        for r, wr in Pt.items():
            for (th, r2), ws in Sig2.items():
                if r2 != r: continue
                for x, wx in r.items():
                    star2[(Pt, r, th, x)] = star2.get((Pt, r, th, x), 0) + wo * wr * (ws / Pi2.m[r]) * wx
    okG3 &= push == star2
    # naturality of L_c and R_c along phi: pulled-back partitions agree
    idx = ctx["idx"]; P = ctx["P"]
    keys2 = list(push.keys()); pos = {k: i for i, k in enumerate(keys2)}; P2 = [push[k] for k in keys2]
    for leaves in (["lev2"], ["hyp"]):
        cols = {"lev2": 0, "hyp": 2}
        lab = canon([tuple([repr(pt[1])] + [repr(pt[cols[l]]) for l in leaves]) for pt in idx])
        lab2 = canon([tuple([repr(k[1])] + [repr(k[cols[l]]) for l in leaves]) for k in keys2])
        pulled = canon([lab2[pos[phi(pt)]] for pt in idx])
        okG3 &= pulled == lab
        Z2 = [tuple(F(rng.randint(0, 5)) for _ in range(2)) for _ in keys2]
        lhs = apply(Emat(lab, P), [Z2[pos[phi(pt)]] for pt in idx])
        rhs = [apply(Emat(lab2, P2), Z2)[pos[phi(pt)]] for pt in idx]
        okG3 &= lhs == rhs
check("G3 star space is functorial along context maps with Df injective on supp Pi; L_c, R_c are natural", okG3 and ntested > 5,
      f"{ntested} context maps")

npass = sum(OUT.values()); print(f"\n{npass}/{len(OUT)} checks pass")
blob = json.dumps(OUT, sort_keys=True).encode()
print("results SHA-256:", hashlib.sha256(blob).hexdigest())
sys.exit(0 if npass == len(OUT) else 1)
