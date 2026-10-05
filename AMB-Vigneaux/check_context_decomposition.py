"""Computational companion to Sections 5.2-5.3 (identifiability and contamination) and 10.6 (sieving).

Exact (rational) checks, plus linear programs whose answers are verified exactly where possible.

  A. Identifiability of decompositions x = sum_i lambda_i A_i
     * affinely independent sources: unique weights (exact linear solve; LP min = max)
     * affinely dependent sources: an exact dependency moves the weights inside W(x)
     * exclusive-region formula lambda_i = x(R_i)/A_i(R_i); per-observation attribution
  B. Supports-only model
     * two sources: feasible weights are exactly [x(R_A), 1 - x(R_B)] (endpoints attained, beyond infeasible)
     * k sources: allocations of the mass function = core of the belief function (LP vs inequalities)
  C. Contamination x = sum lambda_i A_i + mu U, U unknown
     * mu_min via LP; dual certificate y >= 0 rationalized and verified exactly
     * feasible mu form [mu_min, 1]
     * d(x, conv A) <= 2 mu_min; exact counterexample to the converse
     * several unknown sources merge into one (counting is not identifiable)
     * mu_min = 0 iff x in conv A
  D. Sieving (illustration)
     * iterated conditioning = single conditioning; survivors have Omega <= k once z^(k+1) > N;
       survivors are 1 and primes once z > sqrt(N)
     * Selberg's parity example: weights 1 +- lambda(n) have pure parity but nearly equal sieve data
"""
import contextlib, io, random, itertools
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
with contextlib.redirect_stdout(io.StringIO()):
    from check_tower import Dist, delta, Dmap, mu, mix

R = random.Random(41)
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)

def rvec(m, zeros=0):
    w = [F(R.randint(1, 6)) for _ in range(m)]
    for i in R.sample(range(m), zeros): w[i] = F(0)
    s = sum(w); return [a / s for a in w]
def comb(lam, As): return [sum(l * A[b] for l, A in zip(lam, As)) for b in range(len(As[0]))]
def l1(u, v): return sum(abs(a - b) for a, b in zip(u, v))

def solve_exact(M, y):
    """Gaussian elimination over Fractions; returns one solution of M z = y (M: rows), or None."""
    n, m = len(M), len(M[0])
    A = [row[:] + [y[i]] for i, row in enumerate(M)]
    piv, r = [], 0
    for c in range(m):
        p = next((i for i in range(r, n) if A[i][c] != 0), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]
        A[r] = [v / A[r][c] for v in A[r]]
        for i in range(n):
            if i != r and A[i][c] != 0:
                A[i] = [a - A[i][c] * b for a, b in zip(A[i], A[r])]
        piv.append(c); r += 1
    if any(all(v == 0 for v in A[i][:-1]) and A[i][-1] != 0 for i in range(n)): return None
    z = [F(0)] * m
    for i, c in enumerate(piv): z[c] = A[i][-1]
    return z, piv

def null_vector(M):
    """a nonzero rational vector c with M c = 0 (exists when M has more columns than rank)."""
    n, m = len(M), len(M[0])
    A = [row[:] for row in M]
    piv, r = [], 0
    for c in range(m):
        p = next((i for i in range(r, n) if A[i][c] != 0), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]
        A[r] = [v / A[r][c] for v in A[r]]
        for i in range(n):
            if i != r and A[i][c] != 0:
                A[i] = [a - A[i][c] * b for a, b in zip(A[i], A[r])]
        piv.append(c); r += 1
    free = next(c for c in range(m) if c not in piv)
    v = [F(0)] * m; v[free] = F(1)
    for i, c in enumerate(piv): v[c] = -A[i][free]
    return v

def weight_range(x, As, i, extra=None):
    """min and max of lambda_i over {lambda >= 0, sum lambda = 1, sum lambda_j A_j = x} (LP)."""
    k, m = len(As), len(x)
    Aeq = np.r_[np.array([[float(A[b]) for A in As] for b in range(m)]), np.ones((1, k))]
    beq = np.r_[np.array([float(v) for v in x]), 1.0]
    out = []
    for sgn in (1, -1):
        c = np.zeros(k); c[i] = sgn
        r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * k)
        out.append(sgn * r.fun)
    return out

# ======================= A. identifiability =======================
m = 4
ok_unique = ok_range = True
for _ in range(25):
    k = R.randint(2, m)
    while True:
        As = [rvec(m, R.randint(0, 1)) for _ in range(k)]
        # affine independence <=> differences A_i - A_0 linearly independent
        D = [[As[i][b] - As[0][b] for i in range(1, k)] for b in range(m)]
        sol = solve_exact(D, [F(0)] * m)
        if len(sol[1]) == k - 1: break
    lam = rvec(k)
    x = comb(lam, As)
    Msys = [[A[b] for A in As] for b in range(m)] + [[F(1)] * k]
    z, piv = solve_exact(Msys, x + [F(1)])
    ok_unique &= z == lam and len(piv) == k
    lo, hi = weight_range(x, As, 0)
    ok_range &= abs(lo - hi) < 1e-7
check("affinely independent sources: weights recovered exactly and uniquely", ok_unique)
check("affinely independent sources: LP min = max for each weight", ok_range)

ok_dep = ok_dep_lp = True
for _ in range(25):
    k = m + 1                                           # more sources than outcomes: dependent
    As = [rvec(m) for _ in range(k)]
    # dependency: sum c_i A_i = 0, sum c_i = 0, c != 0  (null space of Msys)
    Msys = [[A[b] for A in As] for b in range(m)] + [[F(1)] * k]
    c = null_vector(Msys)
    ok_dep &= all(v == 0 for v in comb(c, As)) and sum(c) == 0
    lam = [F(1, k)] * k
    x = comb(lam, As)
    t = min(F(1, k) / abs(v) for v in c if v != 0) / 2
    lam2 = [a + t * v for a, v in zip(lam, c)]
    ok_dep &= all(v >= 0 for v in lam2) and comb(lam2, As) == x and lam2 != lam
    i = next(j for j, v in enumerate(c) if v != 0)       # a weight the dependency moves
    lo, hi = weight_range(x, As, i)
    ok_dep_lp &= hi - lo > 1e-6
check("affinely dependent sources: an exact dependency gives two decompositions of x", ok_dep)
check("affinely dependent sources: LP weight interval is nondegenerate", ok_dep_lp)

# exclusive regions; the example from the discussion
A = {"b1": F(3, 4), "b2": F(1, 4)}; Bs = {"b2": F(1, 2), "b3": F(1, 2)}
lt = F(3, 5)
x = {b: lt * A.get(b, 0) + (1 - lt) * Bs.get(b, 0) for b in ("b1", "b2", "b3")}
check("exclusive-region formula: lambda = x(R_A)/A(R_A) = 3/5 = 1 - x(R_B)/B(R_B)",
      x["b1"] / A["b1"] == lt and 1 - x["b3"] / Bs["b3"] == lt)
check("per-observation attribution: P(A | b2) = lambda A(b2)/x(b2) = 3/7",
      lt * A["b2"] / x["b2"] == F(3, 7))

# ======================= B. supports only =======================
ok_two = True
for _ in range(40):
    outs = list(range(5))
    SA = set(R.sample(outs, R.randint(1, 4))); SB = set(R.sample(outs, R.randint(1, 4)))
    if not (SA - SB) and not (SB - SA): continue
    U = sorted(SA | SB)
    w = rvec(len(U)); x = dict(zip(U, w))
    RA, RB, RAB = SA - SB, SB - SA, SA & SB
    xA = sum(x[b] for b in RA); xB = sum(x[b] for b in RB); xO = sum(x[b] for b in RAB)
    lo, hi = xA, 1 - xB
    def feasible(l):
        """exact: build A, B with the given supports and x = l A + (1-l) B, or return False"""
        if l < xA or 1 - l < xB: return False
        a_over, b_over = l - xA, (1 - l) - xB
        if xO == 0 and (a_over != 0 or b_over != 0): return False
        Ad = {b: x[b] for b in RA}; Bd = {b: x[b] for b in RB}
        for b in RAB:
            Ad[b] = x[b] * a_over / xO if xO else F(0)
            Bd[b] = x[b] * b_over / xO if xO else F(0)
        ok = all(v >= 0 for v in list(Ad.values()) + list(Bd.values()))
        rec = {b: Ad.get(b, 0) + Bd.get(b, 0) for b in U}
        okA = l == 0 or sum(Ad.values()) == l
        okB = l == 1 or sum(Bd.values()) == 1 - l
        return ok and rec == x and okA and okB
    ok_two &= feasible(lo) and feasible(hi) and feasible((lo + hi) / 2)
    ok_two &= not (lo > 0 and feasible(lo - F(1, 1000))) and not (hi < 1 and feasible(hi + F(1, 1000)))
check("supports only: feasible weights are exactly [x(R_A), 1 - x(R_B)] (endpoints attained, beyond not)", ok_two)

ok_core = True; n_tested = 0
for _ in range(30):
    k, outs = 3, list(range(5))
    supp = [set(R.sample(outs, R.randint(2, 4))) for _ in range(k)]
    covered = sorted(set().union(*supp))
    x = dict(zip(covered, rvec(len(covered))))
    T = {b: frozenset(i for i in range(k) if b in supp[i]) for b in covered}
    mass = {}
    for b in covered: mass[T[b]] = mass.get(T[b], 0) + x[b]
    def Bel(S): return sum(v for t, v in mass.items() if t <= S)
    for _ in range(6):
        lam = rvec(k)
        in_core = all(sum(lam[i] for i in S) >= Bel(frozenset(S)) - F(0)
                      for r in range(1, k + 1) for S in itertools.combinations(range(k), r))
        # allocation LP: z[b,i] >= 0 for i in T[b], sum_i z[b,i] = x(b), sum_b z[b,i] = lam_i
        var = [(b, i) for b in covered for i in T[b]]
        Aeq, beq = [], []
        for b in covered:
            Aeq.append([1.0 if v[0] == b else 0.0 for v in var]); beq.append(float(x[b]))
        for i in range(k):
            Aeq.append([1.0 if v[1] == i else 0.0 for v in var]); beq.append(float(lam[i]))
        r = linprog(np.zeros(len(var)), A_eq=np.array(Aeq), b_eq=np.array(beq),
                    bounds=[(0, None)] * len(var))
        ok_core &= (r.status == 0) == in_core; n_tested += 1
check(f"supports only, 3 sources: allocations = core of the belief function ({n_tested} weight vectors)", ok_core)

# ======================= C. contamination =======================
def mu_min_lp(x, As):
    k, m = len(As), len(x)
    M = np.array([[float(A[b]) for A in As] for b in range(m)])
    r = linprog(-np.ones(k), A_ub=M, b_ub=np.array([float(v) for v in x]), bounds=[(0, None)] * k)
    d = linprog(np.array([float(v) for v in x]), A_ub=-M.T, b_ub=-np.ones(k), bounds=[(0, None)] * m)
    return 1 + r.fun, r.x, d.x
def dist_conv(x, As):
    k, m = len(As), len(x)
    M = np.array([[float(A[b]) for A in As] for b in range(m)]); xv = np.array([float(v) for v in x])
    c = np.r_[np.zeros(k), np.ones(m)]
    Aub = np.r_[np.c_[-M, -np.eye(m)], np.c_[M, -np.eye(m)]]
    r = linprog(c, A_ub=Aub, b_ub=np.r_[-xv, xv], A_eq=np.r_[np.ones(k), np.zeros(m)][None],
                b_eq=[1], bounds=[(0, None)] * (k + m))
    return r.fun

ok_cert = ok_up = ok_dist = ok_zero = True
for _ in range(40):
    m, k = 4, R.randint(1, 3)
    As = [rvec(m, R.randint(0, 2)) for _ in range(k)]
    if R.random() < 0.3:
        x = comb(rvec(k), As)                          # inside conv A
    else:
        x = rvec(m, R.randint(0, 2))
    mm, lam, y = mu_min_lp(x, As)
    # exact certificate: y >= 0 with <y, A_i> >= 1 gives mu >= 1 - <y, x>
    yq = [max(F(v).limit_denominator(10**6), F(0)) for v in y]
    scale = max(F(1) / sum(a * b for a, b in zip(yq, A)) for A in As)
    yq = [v * max(scale, F(1)) for v in yq]
    lower = 1 - sum(a * b for a, b in zip(yq, x))
    lq = [max(F(v).limit_denominator(10**6), F(0)) for v in lam]
    s = comb(lq, As)
    shrink = min([F(1)] + [x[b] / s[b] for b in range(m) if s[b] > x[b]])
    lq = [v * shrink for v in lq]
    upper = 1 - sum(lq)
    ok_cert &= all(v >= 0 for v in yq) and all(sum(a * b for a, b in zip(yq, A)) >= 1 for A in As)
    ok_cert &= all(a <= b for a, b in zip(comb(lq, As), x)) and lower <= upper and upper - lower < 1e-5
    # upward closed: scaling a feasible lambda raises mu continuously to 1
    for t in (F(1, 2), F(0)):
        lt_ = [v * t for v in lq]
        ok_up &= all(a <= b for a, b in zip(comb(lt_, As), x))
    d = dist_conv(x, As)
    ok_dist &= d <= 2 * mm + 1e-7
    ok_zero &= (mm < 1e-8) == (d < 1e-8)
check("mu_min: exact primal-dual certificates bracket the LP value", ok_cert)
check("feasible unknown weights form [mu_min, 1] (scaling a feasible lambda)", ok_up)
check("d(x, conv A) <= 2 mu_min  (distance test is a valid sufficient test)", ok_dist)
check("mu_min = 0 iff x in conv A", ok_zero)

# exact counterexample to the converse
A1 = [F(1, 3)] * 3; x = [F(1, 2), F(1, 2), F(0)]
y = [F(0), F(0), F(3)]
check("counterexample: d(x, A) = 2/3", l1(x, A1) == F(2, 3))
check("counterexample: certificate y = (0,0,3) gives mu_min >= 1 - <y,x> = 1, so mu_min = 1",
      sum(a * b for a, b in zip(y, A1)) == 1 and 1 - sum(a * b for a, b in zip(y, x)) == 1)
check("counterexample: with budget 0.7 the distance test accepts (2/3 <= 0.7) but x is not explainable",
      F(2, 3) <= F(7, 10) and F(1) > F(7, 10))

# counting: two unknown sources merge into one
ok_merge = True
for _ in range(20):
    As = [rvec(4)]
    U1, U2 = rvec(4), rvec(4); l, n1, n2 = F(1, 2), F(1, 5), F(3, 10)
    x2 = [l * a + n1 * u + n2 * v for a, u, v in zip(As[0], U1, U2)]
    U = [(n1 * u + n2 * v) / (n1 + n2) for u, v in zip(U1, U2)]
    x1 = [l * a + (n1 + n2) * w for a, w in zip(As[0], U)]
    ok_merge &= x1 == x2 and abs(sum(U) - 1) == 0 and all(v >= 0 for v in U)
check("two unknown sources with weights n1, n2 equal one unknown source with weight n1 + n2", ok_merge)

# ======================= D. sieving =======================
N = 3000
def factor(n):
    out, p = [], 2
    while p * p <= n:
        while n % p == 0: out.append(p); n //= p
        p += 1
    if n > 1: out.append(n)
    return out
primes = [p for p in range(2, N + 1) if len(factor(p)) == 1]
prior = Dist([(n, F(1, N)) for n in range(1, N + 1)])
def cond(P, pred):
    z = sum(p for v, p in P.items() if pred(v))
    return Dist([(v, p / z) for v, p in P.items() if pred(v)])
z = 11
step = prior
for p in [q for q in primes if q < z]:
    step = cond(step, lambda n, p=p: n % p != 0)
once = cond(prior, lambda n: all(n % q for q in primes if q < z))
check("iterated conditioning on p !| n (p < z) equals single conditioning on their intersection", step == once)
ok_k = True
for k in (1, 2, 3, 4):
    zk = int(N ** (1 / (k + 1))) + 1
    while zk ** (k + 1) <= N: zk += 1
    surv = [n for n in range(1, N + 1) if all(n % q for q in primes if q < zk)]
    ok_k &= all(len(factor(n)) <= k for n in surv if n > 1)
check("survivors of sieving up to z with z^(k+1) > N have at most k prime factors (k = 1..4)", ok_k)
zs = int(N ** 0.5) + 1
surv = [n for n in range(1, N + 1) if all(n % q for q in primes if q < zs)]
check("for z > sqrt(N) the survivors are exactly 1 and the primes in [z, N]",
      surv == [1] + [p for p in primes if p >= zs])
liou = {n: (-1) ** len(factor(n)) for n in range(1, N + 1)}
w_even = {n: 1 + liou[n] for n in range(1, N + 1)}; w_odd = {n: 1 - liou[n] for n in range(1, N + 1)}
pure = all((w_even[n] == 0 or len(factor(n)) % 2 == 0) and (w_odd[n] == 0 or len(factor(n)) % 2 == 1)
           for n in range(1, N + 1))
gaps = []
for d in range(1, 31):
    if any(d % (p * p) == 0 for p in primes if p * p <= d): continue
    se = sum(w_even[n] for n in range(d, N + 1, d)); so = sum(w_odd[n] for n in range(d, N + 1, d))
    gaps.append(abs(se - so) / (2 * (N // d)))
check(f"parity example: weights 1+lambda / 1-lambda have pure even / odd Omega", pure)
print(f"     relative gap in sieve data |S_+(d) - S_-(d)| / (2 floor(N/d)), squarefree d <= 30: "
      f"max {max(gaps):.3f}  (illustration, N = {N})")

print(f"\n{sum(res)}/{len(res)} checks passed")
