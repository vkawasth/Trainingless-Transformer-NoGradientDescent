"""check_tower_horns.py -- the obstruction tower and the coupling nerve: what is identified, and what is not.

Horn encoding.  An n-horn on bits with faces F_i (i != k) becomes a level-one claim on the base B = disjoint union of the
face assignment sets: x1 = (1/n) sum_i (i, data_i).  A global assignment s gives rho_s = (1/n) sum_i delta_(i, s|F_i).
Admissibility data: S1 = {rho_s} u (compatible face families, here the parity families of every parity pattern),
S2 = {delta_rho : rho = rho_s}  (only global mechanisms are effective).  Then L1 = conv S1 is a local test,
E1 = conv{rho_s} is the set of fillable horn data, and the fibre of level-two lifts is the set of decompositions.

  T1 compatible horn data pass the local test: x1 in L1 (parity horns n = 3,4,5; mixtures; 2-horns)
  T2 x1 in E1 iff the horn fills iff CF(x1) = 0; every 2-horn lies in E1
  T3 a) with S1 = all vertices of the local polytope, o3^-(x1) = 2 CF(x1) (= 2 - 2^{3-n} for parity), d(x1,E1) <= 2 CF;
     b) with fewer contextual mechanisms in S1, o3^- can exceed 2 CF (parity families only: o3^- = 2)
     c) for the parity horns d(x1,E1) = 2 eps*(n), the best approximate filler distance
  T4 the level of the tower is not the dimension of the horn: every n-horn obstruction, n = 3,4,5, is the secondary
     obstruction of a level-one claim (extension-obstructed type), and the primary test L1 never fails on compatible data
  T5 Fisher / Guillemin bound: for y in E1 with full support, d(x1,E1)^2 <= ||x1 - y||_1^2 <= chi^2(x1||y) = Fisher_y(x1-y),
     and ||x1-y||_1^2 <= 2 KL(x1||y) (Pinsker; KL is the Bregman divergence of sum y log y).  Equality in the first
     iff |x1-y|/y is constant on the support (the tower example W1 attains it); no lower bound (KL can be infinite)
  T6 directions of indeterminacy: for a 2-horn the fibre of level-two lifts over S1 = {rho_s} is the filler polytope, whose
     tangent space is the slice-wise circulations (conditional covariances), of dimension sum_{y_k}(|Y_i|-1)(|Y_j|-1)
  T7 transport: o3^- can only fall under change of base along a kernel (data processing), random general admissibility data
"""
import itertools, random
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
R = random.Random(17); res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))
def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, nv=None):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
    assert r.status == 0, r.message
    return r

def encode(n, k=0):
    faces = [i for i in range(n + 1) if i != k]
    idx = {i: tuple(j for j in range(n + 1) if j != i) for i in faces}
    B = [(i, a) for i in faces for a in itertools.product((0, 1), repeat=n)]
    pos = {b: t for t, b in enumerate(B)}
    def vec(face_laws):                          # face_laws: {i: {a: p}}
        v = np.zeros(len(B))
        for i, law in face_laws.items():
            for a, p in law.items(): v[pos[(i, a)]] += float(p) / len(faces)
        return v
    def glob(s): return vec({i: {tuple(s[j] for j in idx[i]): 1} for i in faces})
    det = [glob(s) for s in itertools.product((0, 1), repeat=n + 1)]
    def parity(pat):
        return vec({i: {a: Fr(1, 2 ** (n - 1)) for a in itertools.product((0, 1), repeat=n) if sum(a) % 2 == pat[t]}
                    for t, i in enumerate(faces)})
    pats = list(itertools.product((0, 1), repeat=len(faces)))
    ctx = [parity(p) for p in pats]
    odd = parity(tuple(1 if t == 0 else 0 for t in range(len(faces))))
    return dict(B=B, det=np.array(det), ctx=np.array(ctx), odd=odd, faces=faces, idx=idx, vec=vec, glob=glob)

def in_conv(x, P):                               # feasibility of x in conv(rows of P)
    m = len(P); r = linprog(np.zeros(m), A_eq=np.vstack([P.T, np.ones(m)]), b_eq=np.r_[x, 1], bounds=[(0, None)] * m, method="highs")
    return r.status == 0
def dist1(x, P):                                 # l1 distance from x to conv(rows of P)
    m, d = P.shape; nv = m + d                   # w (m), t (d): |x - P^T w| <= t
    A = np.block([[P.T, -np.eye(d)], [-P.T, -np.eye(d)]]); b = np.r_[x, -x]
    r = lp(np.r_[np.zeros(m), np.ones(d)], A, b, np.r_[np.ones(m), np.zeros(d)][None, :], [1], nv); return r.fun
def cf(x, P):                                    # unmodeled weight: 1 - max sum lam, P^T lam <= x
    m = len(P); r = lp(-np.ones(m), P.T, x, nv=m); return 1 + r.fun
def o3min(x, S1, E2rows):
    """min over Pi in D(S1) with mu(Pi) = x of the l1 distance from Pi to conv(E2rows) (rows are vectors on S1)"""
    m, d = S1.shape; q = len(E2rows); nv = m + q + m        # Pi (m), w (q), t (m)
    A_ub = np.block([[np.eye(m), -E2rows.T, -np.eye(m)], [-np.eye(m), E2rows.T, -np.eye(m)]]); b_ub = np.zeros(2 * m)
    A_eq = np.vstack([np.c_[S1.T, np.zeros((d, q + m))], np.r_[np.ones(m), np.zeros(q + m)], np.r_[np.zeros(m), np.ones(q), np.zeros(m)]])
    r = lp(np.r_[np.zeros(m + q), np.ones(m)], A_ub, b_ub, A_eq, np.r_[x, 1, 1], nv); return r.fun


def compat_rows(H, n):
    """linear equations (rows) on R^B saying a vector is a compatible face family with equal face masses"""
    B, faces, idx = H["B"], H["faces"], H["idx"]; pos = {b: t for t, b in enumerate(B)}; R_ = []
    for i, j in itertools.combinations(faces, 2):
        common = sorted(set(idx[i]) & set(idx[j]))
        for val in itertools.product((0, 1), repeat=len(common)):
            r = np.zeros(len(B))
            for (f, a) in B:
                if f in (i, j) and all(a[idx[f].index(c)] == v for c, v in zip(common, val)): r[pos[(f, a)]] += 1 if f == i else -1
            R_.append(r)
    return np.array(R_)
def o3min_local(x, H, n):
    """S1 = vertices of the local polytope, S2 = Diracs on global mechanisms: o3^- = 2 min{mass of r : x - r in cone(det), r >= 0 compatible}"""
    det = H["det"]; m, d = det.shape; C = compat_rows(H, n)
    nv = m + d                                   # lam (m), r (d)
    A_eq = np.vstack([np.c_[det.T, np.eye(d)], np.c_[np.zeros((len(C), m)), C]])
    r = lp(np.r_[np.zeros(m), np.ones(d)], A_eq=A_eq, b_eq=np.r_[x, np.zeros(len(C))], nv=nv); return 2 * r.fun

# ---------------------------------------------------------------- T1-T4
rows = []; ok1 = ok2 = ok3 = ok3b = ok4 = True; dpar = {}
for n in (3, 4, 5):
    H = encode(n); det, ctx = H["det"], H["ctx"]; S1 = np.vstack([det, ctx]); md = len(det)
    E2 = np.c_[np.eye(md), np.zeros((md, len(ctx)))]          # Diracs on the global mechanisms, as vectors on S1
    claims = [("parity", H["odd"])]
    for _ in range(3 if n < 5 else 1):
        t = R.random(); lam = np.array([R.random() for _ in range(md)]); lam /= lam.sum()
        claims.append((f"mix t={t:.2f}", t * H["odd"] + (1 - t) * lam @ det))
    for name, x in claims:
        inL1 = in_conv(x, S1); c = cf(x, det); d = dist1(x, det); o = o3min(x, S1, E2); oL = o3min_local(x, H, n)
        ok1 &= inL1; ok2 &= (c < 1e-9) == in_conv(x, det)
        ok3 &= d <= 2 * c + 1e-9 and abs(oL - 2 * c) < 1e-8 and oL <= o + 1e-9
        if name == "parity":
            ok3 &= abs(c - (1 - 2 ** (2 - n))) < 1e-9; ok3b &= abs(o - 2) < 1e-9 and o > 2 * c + 1e-6; ok4 &= c > 0 and inL1; dpar[n] = d
        rows.append((n, name, round(c, 4), round(d, 4), round(oL, 4), round(o, 4)))
# 2-horns: faces {0,1}, {1,2} of three variables with sizes (2,3,2), random compatible data: always fillable
def two_horn_ok(trials=20):
    ok = True
    for _ in range(trials):
        p = np.random.default_rng(R.randint(0, 10 ** 6)).dirichlet(np.ones(12)).reshape(2, 3, 2)
        x = np.r_[p.sum(2).ravel(), p.sum(0).ravel()] / 2                     # compatible by construction
        G = []
        for s in itertools.product(range(2), range(3), range(2)):
            v = np.zeros(12); v[s[0] * 3 + s[1]] += .5; v[6 + s[1] * 2 + s[2]] += .5; G.append(v)
        ok &= in_conv(x, np.array(G)) and cf(x, np.array(G)) < 1e-9
    return ok
ok2 &= two_horn_ok()
for r in rows: print("     n=%d %-12s CF=%-7s d(x1,E1)=%-7s o3^-[local polytope]=%-7s o3^-[parity families]=%s" % r)
check("T1 compatible horn data pass the local test x1 in L1 = conv S1", ok1)
check("T2 x1 in E1 iff CF = 0 (iff the horn fills); every compatible 2-horn lies in E1", ok2)
check("T3a with S1 = vertices of the local polytope, o3^- = 2 CF exactly (parity: 2 - 2^{3-n}); d(x1,E1) <= 2 CF", ok3)
check("T3b with S1 = global mechanisms + parity families only, o3^- = 2 > 2 CF on the parity horns: the secondary "
      "obstruction depends on which contextual mechanisms are locally admissible", ok3b)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import check_horns_n as CH
eps = {n: CH.eps_star(CH.horn(n, 0), n) for n in (3, 4, 5)}
check("T3c for the parity horns d(x1,E1) = 2 eps*(n): the l1 distance to the fillable horns is twice the best "
      "approximate filler distance of check_horns_n (n = 3,4,5)", all(abs(dpar[n] - 2 * eps[n]) < 1e-8 for n in eps),
      "; ".join(f"n={n}: d = {dpar[n]:.6f}, 2 eps* = {2 * eps[n]:.6f}" for n in eps))
check("T4 every unfillable n-horn (n = 3,4,5) is an extension-obstructed level-one claim: primary test passes, the "
      "secondary obstruction is positive -- tower level is not horn dimension", ok4)

# ---------------------------------------------------------------- T5 Fisher / Guillemin
ok = True; worst = 0
for _ in range(300):
    d = R.randint(2, 6); P = np.random.default_rng(R.randint(0, 10 ** 6)).dirichlet(np.ones(d), size=R.randint(1, 4))
    x = np.random.default_rng(R.randint(0, 10 ** 6)).dirichlet(np.ones(d) * .5)
    lam = np.random.default_rng(R.randint(0, 10 ** 6)).dirichlet(np.ones(len(P))); y = lam @ P
    D = dist1(x, P); l1 = np.abs(x - y).sum(); chi2 = ((x - y) ** 2 / y).sum()
    kl = sum(a * np.log(a / b) for a, b in zip(x, y) if a > 0)
    ok &= D ** 2 <= l1 ** 2 + 1e-12 and l1 ** 2 <= chi2 + 1e-12 and l1 ** 2 <= 2 * kl + 1e-12
    worst = max(worst, l1 ** 2 / chi2)
x = np.array([.7, .3]); h = np.array([.5, .5])                                  # tower example W1
eqW1 = abs(np.abs(x - h).sum() ** 2 - ((x - h) ** 2 / h).sum()) < 1e-15
check("T5 d(x1,E1)^2 <= ||x1-y||_1^2 <= Fisher_y(x1-y) = chi^2 and <= 2 KL(x1||y) for y in E1 (300 random cases); "
      "equality in the Fisher bound at the tower example W1", ok and eqW1, f"max ratio ||.||_1^2 / chi^2 = {worst:.4f}; W1: 0.16 = 0.16")

# ---------------------------------------------------------------- T6 directions of indeterminacy for 2-horns
ok = True; dims = []
for (a, b, c) in ((2, 2, 2), (2, 3, 2), (3, 2, 4)):
    cells = list(itertools.product(range(a), range(b), range(c)))
    M = np.array([[float(s[0] == i and s[1] == j) for s in cells] for i in range(a) for j in range(b)] +
                 [[float(s[1] == j and s[2] == l) for s in cells] for j in range(b) for l in range(c)])
    _, sv, vt = np.linalg.svd(M); K = vt[np.sum(sv > 1e-9):]
    ok &= len(K) == b * (a - 1) * (c - 1)
    # each kernel vector is, slice by slice in y_1, a circulation on Y0 x Y2
    for v in K:
        T = v.reshape(a, b, c)
        ok &= np.allclose(T.sum(2), 0) and np.allclose(T.sum(0), 0)
    dims.append(f"{a}x{b}x{c}: {len(K)}")
check("T6 for 2-horns the fibre of level-two lifts is the filler polytope; its directions are slice-wise circulations, "
      "dimension sum_{y1} (|Y0|-1)(|Y2|-1)", ok, "; ".join(dims))

# ---------------------------------------------------------------- T7 transport monotonicity of o3^-
def o3min_general(x, S1, S2):                    # S2 rows are laws on S1
    return o3min(x, S1, S2)
ok = True; drops = 0; trials = 0
while trials < 60:
    nB, m, q = R.randint(2, 4), R.randint(2, 5), R.randint(1, 3)
    g = np.random.default_rng(R.randint(0, 10 ** 6))
    S1 = g.dirichlet(np.ones(nB) * .7, size=m); S2 = g.dirichlet(np.ones(m) * .7, size=q)
    Pi = g.dirichlet(np.ones(m)); x = Pi @ S1                              # x1 in L1
    k = g.dirichlet(np.ones(R.randint(2, 4)), size=nB)                    # kernel B -> B'
    try: before = o3min(x, S1, S2); after = o3min(x @ k, S1 @ k, S2)      # S1 pushed forward; S2 indexes the same S1
    except AssertionError: continue
    trials += 1; ok &= after <= before + 1e-9; drops += after < before - 1e-9
check("T7 change of base along a kernel never increases o3^- (60 random general admissibility data)", ok,
      f"strict decrease in {drops} of {trials}")
print(f"\n{sum(res)}/{len(res)} checks passed")
