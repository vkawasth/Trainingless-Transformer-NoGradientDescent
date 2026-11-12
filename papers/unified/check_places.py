"""check_places.py -- exact checks for Section 17 (places of the context presheaf) of
"Shadows, Fibres and Secants of the Posterior Tower".

The three-token model of Section 16: binary X0, X1, X2, uniform singletons, every pair agrees with probability eps.
With a = (3 eps - 1)/4 and b = (1 - eps)/4, the symmetric section is J(000) = J(111) = a and J = b elsewhere; chi(x) = (-1)^(x0+x1+x2).

  M1  centre (trivial seminorm, germs = Q): for every rational eps, J is a signed global section over Q (total mass 1, all three pair
      tables correct), also for eps < 1/3 where a < 0
  M2  the kernel of the pair-marginal map Q^8 -> Q^12 is the line Q.chi (rank 7), so the fibre over any field of characteristic 0 is
      J + K.chi; the rank is 7 over F_p as well (p = 2, 3, 5, 7, 11; over F_2, chi reduces to the all-ones vector), but J itself has
      denominator 4 and does not reduce mod 2
  M3  positivity lives only at the archimedean place: the positive fibre is non-empty iff eps >= 1/3 (exact), while for every prime
      p <= 47, -1 is a sum of four squares in Q_p (Hensel witness mod p^12), so Q_p admits no ordering and "positive law" has no meaning
  M4  Beck-Chevalley is decided at the centre: the conditional-independence determinants Delta_0, Delta_1 are polynomials in (eps, s)
      with rational coefficients; on the fibre family they vanish together only at (1/2, 0) and the degenerate (1, 0)
  M5  the size of the defect is archimedean and algebraic: c = |2eps-1|/(2eps) at s = 0; c is rational on the whole symmetric family
      (row and column margins agree); for a general 2x2 table c^2 is rational and c lies in a quadratic field
  M6  finite places measure the determinant: v_p(Delta) at the primes, and the product formula prod_v |Delta|_v = 1 recovers the
      archimedean size |Delta|_inf from the p-adic sizes
  M7  endpoints of the p-branches: for p-integral data, J reduces to a section over F_p of the reduced pair tables; its support mod p
      is the zero pattern seen at the end of the branch (e.g. eps = 2/5: mod 3 only 000 and 111 survive)
  M8  topology: the Beck-Chevalley cover {X0X1, X1X2} has nerve an edge (b0, b1) = (1, 0), the three-window cover a hollow triangle
      (1, 1); a finite truncation of the tree M(Z) has b1 = 0
  E1  the interactive explorer (explorer/context_sheaf.html) at eps = 0.40, s = 0 shows Phi = 1.8, c = 0.25, I(X0;X2|X1) = 0.0464 bits
Run: python check_places.py
"""
from fractions import Fraction as F
import itertools, math
import sympy as sp

results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

ATOMS = list(itertools.product((0, 1), repeat=3))
def section(eps, s=F(0)):
    a, b = (3 * eps - 1) / 4, (1 - eps) / 4
    return {x: {0: a + s, 3: a - s, 1: b - s, 2: b + s}[sum(x)] for x in ATOMS}
def pair(p, i, j):
    t = [[F(0)] * 2 for _ in range(2)]
    for x, v in p.items(): t[x[i]][x[j]] += v
    return t
PAIRS = [(0, 1), (1, 2), (0, 2)]

# M1
grid = [F(k, 60) for k in range(1, 60)]
ok = all(sum(section(e).values()) == 1 and all(pair(section(e), i, j) == [[e / 2, (1 - e) / 2], [(1 - e) / 2, e / 2]] for i, j in PAIRS)
         for e in grid)
neg = [e for e in grid if section(e)[(0, 0, 0)] < 0]
check("M1 signed global sections over Q exist for every rational eps (59 values), including eps < 1/3", ok and max(neg) < F(1, 3),
      f"a < 0 exactly for eps < 1/3 ({len(neg)} of 59 grid values)")

# M2
rows = []
for i, j in PAIRS:
    for u, v in itertools.product((0, 1), repeat=2):
        rows.append([1 if (x[i], x[j]) == (u, v) else 0 for x in ATOMS])
Mq = sp.Matrix(rows); ker = Mq.nullspace()
chi = sp.Matrix([(-1) ** sum(x) for x in ATOMS])
def rank_mod(M, p):
    A = [[int(v) % p for v in r] for r in M.tolist()]; r = 0; cols = len(A[0])
    for c in range(cols):
        piv = next((i for i in range(r, len(A)) if A[i][c] % p), None)
        if piv is None: continue
        A[r], A[piv] = A[piv], A[r]; inv = pow(A[r][c], -1, p)
        A[r] = [v * inv % p for v in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]: A[i] = [(vi - A[i][c] * vr) % p for vi, vr in zip(A[i], A[r])]
        r += 1
    return r
rk = {p: rank_mod(Mq, p) for p in (2, 3, 5, 7, 11)}
check("M2 kernel of the pair-marginal map is the line spanned by chi (rank 7 over Q and over F_p)",
      Mq.rank() == 7 and len(ker) == 1 and (ker[0] / ker[0][0] - chi / chi[0]).is_zero_matrix and all(rk[p] == 7 for p in rk),
      f"ranks mod p: {rk}")

# M3
def pos_fibre_nonempty(e):
    J = section(e); a, b = J[(0, 0, 0)], J[(1, 0, 0)]
    return min(a, b) >= 0                      # |s| <= min(a, b) is the positive fibre
okpos = all(pos_fibre_nonempty(e) == (e >= F(1, 3)) for e in grid + [F(1, 3)])
def sqrt_mod_pk(n, p, k):
    """x with x^2 = n mod p^k by Hensel lifting (p odd, n a nonzero square mod p)."""
    x = next(x for x in range(1, p) if (x * x - n) % p == 0); m = p
    for _ in range(k - 1):
        m2 = m * p; x = (x - (x * x - n) * pow(2 * x, -1, m2)) % m2; m = m2
    return x
def minus_one_sum_of_squares(p, k=12):
    mod = p ** k
    if p == 2:                                  # -7 is a square in Z_2; -1 = (sqrt(-7))^2 + 2^2 + 1^2 + 1^2
        x = 1
        for j in range(3, k + 1):               # lift x^2 = -7 mod 2^j
            if (x * x + 7) % (2 ** j): x += 2 ** (j - 2)
        sq = [x, 2, 1, 1]; mod = 2 ** (k - 1)
    else:
        a = next(a for a in range(p) if any((a * a + b * b + 1) % p == 0 for b in range(p)))
        b = next(b for b in range(p) if (a * a + b * b + 1) % p == 0)
        if b % p == 0: a, b = b, a
        b = sqrt_mod_pk((-1 - a * a) % mod, p, k) if (-1 - a * a) % p else 0
        sq = [a, b, 0, 0]
        if (a * a + b * b + 1) % mod: return False, None
    return (sum(v * v for v in sq) + 1) % mod == 0, sq
primes = [p for p in range(2, 48) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
wit = {p: minus_one_sum_of_squares(p) for p in primes}
check("M3 positive fibre non-empty iff eps >= 1/3; in Q_p (all p <= 47) -1 is a sum of squares, so Q_p has no ordering",
      okpos and all(w[0] for w in wit.values()), f"e.g. p = 3: -1 = {wit[3][1][0]}^2 + {wit[3][1][1]}^2 mod 3^12; p = 2 via sqrt(-7)")

# M4
e, s = sp.symbols("eps s")
a, b = (3 * e - 1) / 4, (1 - e) / 4
def Psym(x): return {0: a + s, 3: a - s, 1: b - s, 2: b + s}[sum(x)]
Delta = [sp.expand(Psym((0, x1, 0)) * Psym((1, x1, 1)) - Psym((0, x1, 1)) * Psym((1, x1, 0))) for x1 in (0, 1)]
sol = sp.solve(Delta, [e, s], dict=True)
coeffs_rational = all(c.is_rational for D in Delta for c in sp.Poly(D, e, s).coeffs())
check("M4 CI determinants have rational coefficients; on the fibre family they vanish together only at (1/2, 0) and (1, 0)",
      coeffs_rational and sorted((d[e], d[s]) for d in sol) == [(sp.Rational(1, 2), 0), (1, 0)],
      f"Delta_0 = {sp.factor(Delta[0])}, Delta_1 = {sp.factor(Delta[1])}")

# M5
def c_sq(eps, s_):
    J = section(eps, s_); best = F(0)
    for x1 in (0, 1):
        t = [[J[(i, x1, j)] for j in (0, 1)] for i in (0, 1)]
        r0, r1, c0, c1 = t[0][0] + t[0][1], t[1][0] + t[1][1], t[0][0] + t[1][0], t[0][1] + t[1][1]
        if min(r0, r1, c0, c1) > 0: best = max(best, (t[0][0] * t[1][1] - t[0][1] * t[1][0]) ** 2 / (r0 * r1 * c0 * c1))
    return best
ok5 = all(c_sq(ep, F(0)) == ((2 * ep - 1) / (2 * ep)) ** 2 for ep in grid if ep > F(1, 3) and ep < 1)
# on the symmetric family row and column margins agree in each stratum, so c = |Delta| / (r0 r1) is rational; for a general table
# c^2 is rational and c lies in a quadratic field, e.g. the table [[1, 2], [3, 4]] / 10 has c^2 = 1/126
fam_rational = all(math.isqrt((q := c_sq(ep, sv)).numerator) ** 2 == q.numerator and math.isqrt(q.denominator) ** 2 == q.denominator
                   for ep in grid if F(1, 3) < ep < 1 for sv in (F(0), F(1, 200), F(-1, 300)) if abs(sv) <= min(section(ep)[(0, 0, 0)], section(ep)[(1, 0, 0)]))
t = [[F(1, 10), F(2, 10)], [F(3, 10), F(4, 10)]]
q = (t[0][0] * t[1][1] - t[0][1] * t[1][0]) ** 2 / ((t[0][0] + t[0][1]) * (t[1][0] + t[1][1]) * (t[0][0] + t[1][0]) * (t[0][1] + t[1][1]))
is_square = math.isqrt(q.numerator) ** 2 == q.numerator and math.isqrt(q.denominator) ** 2 == q.denominator
check("M5 c = |2eps-1|/(2eps) at s = 0; c is rational on the whole symmetric family; for a general table c^2 in Q, c quadratic",
      ok5 and fam_rational and not is_square, f"general table [[1,2],[3,4]]/10: c^2 = {q}")

# M6
def vp(x, p):
    x = F(x)
    if x == 0: return math.inf
    n, d, v = x.numerator, x.denominator, 0
    while n % p == 0: n //= p; v += 1
    while d % p == 0: d //= p; v -= 1
    return v
rows6 = []; okprod = True
for ep, sv in ((F(2, 5), F(0)), (F(3, 5), F(0)), (F(3, 4), F(1, 40)), (F(1, 2), F(0))):
    D0 = Delta[0].subs({e: sp.Rational(ep.numerator, ep.denominator), s: sp.Rational(sv.numerator, sv.denominator)})
    D0 = F(int(sp.fraction(D0)[0]), int(sp.fraction(D0)[1]))
    if D0 != 0:
        ps = sp.primefactors(D0.numerator) + sp.primefactors(D0.denominator)
        prod = abs(D0)                                       # |D0|_inf
        for p in ps: prod *= F(p) ** (-vp(D0, p))            # times |D0|_p = p^(-v_p)
        okprod &= prod == 1
    rows6.append((str(ep), str(sv), str(D0), {p: vp(D0, p) for p in (2, 3, 5, 7)}))
check("M6 p-adic valuations of Delta_0 and the product formula |Delta|_inf * prod_p |Delta|_p = 1", okprod,
      "; ".join(f"eps={r[0]}, s={r[1]}: Delta0={r[2]}, v_p={r[3]}" for r in rows6))

# M7
def reduce_mod(x, p): return x.numerator * pow(x.denominator, -1, p) % p
okred = True; supp = {}
for ep, p in ((F(2, 5), 3), (F(2, 5), 7), (F(3, 7), 5), (F(4, 5), 3)):
    J = section(ep)
    if any(v.denominator % p == 0 for v in J.values()) or ep.denominator % p == 0: continue
    Jp = {x: reduce_mod(v, p) for x, v in J.items()}
    okred &= sum(Jp.values()) % p == 1
    for i, j in PAIRS:
        t = [[0, 0], [0, 0]]
        for x, v in Jp.items(): t[x[i]][x[j]] = (t[x[i]][x[j]] + v) % p
        okred &= t == [[reduce_mod(ep / 2, p), reduce_mod((1 - ep) / 2, p)], [reduce_mod((1 - ep) / 2, p), reduce_mod(ep / 2, p)]]
    supp[(str(ep), p)] = ["".join(map(str, x)) for x, v in Jp.items() if v]
check("M7 J reduces to a section over F_p of the reduced pair tables; support mod p", okred and supp[("2/5", 3)] == ["000", "111"],
      str(supp))

# M8
def betti(vertices, simplices):
    """Betti numbers b0, b1 over Q of the simplicial complex given by maximal simplices (as frozensets)."""
    faces = {k: set() for k in range(3)}
    for S in simplices:
        for k in range(min(3, len(S))):
            for f in itertools.combinations(sorted(S), k + 1): faces[k].add(f)
    V, E, T = sorted(faces[0]), sorted(faces[1]), sorted(faces[2])
    d1 = sp.zeros(len(V), len(E))
    for j, (u, v) in enumerate(E): d1[V.index((u,)), j] = -1; d1[V.index((v,)), j] = 1
    d2 = sp.zeros(len(E), len(T))
    for j, (u, v, w) in enumerate(T):
        for sgn, ed in ((1, (v, w)), (-1, (u, w)), (1, (u, v))): d2[E.index(ed), j] = sgn
    r1 = d1.rank() if E else 0; r2 = d2.rank() if T else 0
    return len(V) - r1, len(E) - r1 - r2
def nerve(cover):
    idx = range(len(cover)); simp = []
    for k in (3, 2, 1):
        for S in itertools.combinations(idx, k):
            if set.intersection(*[cover[i] for i in S]): simp.append(frozenset(S))
    return simp
bc = betti(None, nerve([{0, 1}, {1, 2}]))
tri = betti(None, nerve([{0, 1}, {1, 2}, {0, 2}]))
tree = betti(None, [frozenset(("c", f"p{i}")) for i in range(6)] + [frozenset((f"p{i}", f"q{i}")) for i in range(6)])
check("M8 nerves: Beck-Chevalley cover (1, 0); three-window cover (1, 1); truncated M(Z) tree has b1 = 0",
      bc == (1, 0) and tri == (1, 1) and tree[1] == 0, f"BC {bc}, three windows {tri}, tree {tree}")

# E1
J = section(F(2, 5)); eps = 0.40
phi = 3 * (1 - eps); c = math.sqrt(c_sq(F(2, 5), F(0)))
cmi = 0.0
for x1 in (0, 1):
    t = [[float(J[(i, x1, j)]) for j in (0, 1)] for i in (0, 1)]; n = sum(map(sum, t))
    for i in (0, 1):
        for j in (0, 1):
            if t[i][j] > 0: cmi += t[i][j] * math.log2(t[i][j] * n / (sum(t[i]) * (t[0][j] + t[1][j])))
check("E1 explorer values at eps = 0.40, s = 0", abs(phi - 1.8) < 1e-12 and abs(c - 0.25) < 1e-12 and abs(cmi - 0.0464) < 5e-5,
      f"Phi = {phi:.3f}, c = {c:.3f}, I = {cmi:.4f} bits")

print(f"\n{sum(results)}/{len(results)} checks passed")
