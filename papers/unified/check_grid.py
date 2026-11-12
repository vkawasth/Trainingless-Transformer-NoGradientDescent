"""check_grid.py -- the two towers as one grid: tower depth (D, D^2, D^3, shadow) against the base semiring (C > R > R>=0, and the
tropical semiring T). Note: level2/partial_gluing.pdf, Section 16; main paper Sections 2 and 8.

The posterior tower is built from one monad D (finitely supported weights summing to one) and makes sense over any semiring S. This file
re-implements the tower code of check_tower.py with the semiring as a parameter and checks, row by row:

  TG1 the comonad laws of C_3 (counits, coassociativity, naturality) in every row
  TG2 the averaging law: natural, unit, counit, multiplication hold, and comultiplication fails on the paper's counterexample, in every row
  TG3 the shift law: natural and counit hold, unit and multiplication fail, in every row
  TG4 change of base commutes with the tower: the semiring maps R>=0 -> R -> C carry coherent contexts to coherent contexts and commute
      with both laws (C(D phi) o lambda^S = lambda^S' o D(phi)); for the tropical row this is the valuation monad morphism of the main paper
      (check_tropical.py T1, T2)
  TG5 what needs positivity: over R a coherent level-two law with the right barycentre violates Jensen (it is not above delta_rho in convex
      order), its second moment can fall below that of the barycentre (no martingale), and the signed barycentre fibre is unbounded (no
      convex-order diameter, so the unit defect of the shift law has no size)
  TG6 contextuality as a row split at fixed depth: at level one the PR box has a section in the R row and none in R>=0; at level two a
      three-variable table has a complex but no real two-type fibre
Run: python check_grid.py
"""
from fractions import Fraction as F
import itertools, random, sys, os

# ---------------- semirings ----------------
class Gauss:
    """Gaussian rationals a + b i with Fraction parts (exact, hashable)"""
    __slots__ = ("a", "b")
    def __init__(self, a, b=0): self.a = F(a); self.b = F(b)
    def __add__(self, o): o = o if isinstance(o, Gauss) else Gauss(o); return Gauss(self.a + o.a, self.b + o.b)
    __radd__ = __add__
    def __mul__(self, o): o = o if isinstance(o, Gauss) else Gauss(o); return Gauss(self.a * o.a - self.b * o.b, self.a * o.b + self.b * o.a)
    __rmul__ = __mul__
    def __truediv__(self, o):
        o = o if isinstance(o, Gauss) else Gauss(o); n = o.a * o.a + o.b * o.b
        return Gauss((self.a * o.a + self.b * o.b) / n, (self.b * o.a - self.a * o.b) / n)
    def __eq__(self, o): o = o if isinstance(o, Gauss) else Gauss(o); return self.a == o.a and self.b == o.b
    def __hash__(self): return hash((self.a, self.b))
    def __bool__(self): return bool(self.a) or bool(self.b)
    def __repr__(self): return f"({self.a}{'+' if self.b >= 0 else '-'}{abs(self.b)}i)"

INF = float("inf")
class Semiring:
    def __init__(self, name, zero, one, add, mul, is_zero, rand_weights, from_frac=None):
        self.name = name; self.zero = zero; self.one = one; self.add = add; self.mul = mul; self.is_zero = is_zero
        self.rand_weights = rand_weights; self.from_frac = from_frac
def _pos(R, k):
    ws = [F(R.randint(1, 4)) for _ in range(k)]; s = sum(ws); return [w / s for w in ws]
def _signed(R, k):
    while True:
        ws = [F(R.randint(-3, 4)) for _ in range(k)]; s = sum(ws)
        if s != 0 and all(ws): return [w / s for w in ws]
def _complex(R, k):
    while True:
        ws = [Gauss(R.randint(-3, 4), R.randint(-3, 3)) for _ in range(k)]; s = sum(ws, Gauss(0))
        if s and all(ws): return [w / s for w in ws]
def _trop(R, k):
    ws = [F(R.randint(0, 5)) for _ in range(k)]; m = min(ws); return [w - m for w in ws]     # normalised: min = 0 (the tropical one)
ROWS = {
    "R>=0": Semiring("R>=0", F(0), F(1), lambda a, b: a + b, lambda a, b: a * b, lambda a: a == 0, _pos, lambda q: q),
    "R": Semiring("R", F(0), F(1), lambda a, b: a + b, lambda a, b: a * b, lambda a: a == 0, _signed, lambda q: q),
    "C": Semiring("C", Gauss(0), Gauss(1), lambda a, b: a + b, lambda a, b: a * b, lambda a: not a, _complex, lambda q: Gauss(q)),
    "T": Semiring("T", INF, F(0), lambda a, b: min(a, b), lambda a, b: a + b, lambda a: a == INF, _trop, None),
}
S = ROWS["R>=0"]

# ---------------- the tower, over S ----------------
class Dist:
    __slots__ = ("m",)
    def __init__(self, items):
        m = {}
        for v, p in items:
            m[v] = S.add(m[v], p) if v in m else p
        self.m = {v: p for v, p in m.items() if not S.is_zero(p)}
    def __eq__(self, o): return isinstance(o, Dist) and self.m == o.m
    def __hash__(self): return hash(frozenset(self.m.items()))
    def items(self): return self.m.items()
    def total(self):
        t = S.zero
        for _, p in self.m.items(): t = S.add(t, p)
        return t
def delta(v): return Dist([(v, S.one)])
def Dmap(f, d): return Dist([(f(v), p) for v, p in d.items()])
def mu(dd): return Dist([(v, S.mul(p, q)) for d, p in dd.items() for v, q in d.items()])
def mix(pairs): return Dist([(v, S.mul(w, p)) for w, d in pairs for v, p in d.items()])
def Dk(k, f):
    g = f
    for _ in range(k): g = (lambda h: (lambda d: Dmap(h, d)))(g)
    return g
THETA = ("t1", "t2")
def Hyp(f): return lambda h: (h[0], Dmap(f, h[1]))
def Sh(f): return lambda s: Dmap(Hyp(f), s)
def E(f): return lambda c: (Dk(1, f)(c[0]), Dk(2, f)(c[1]), Dk(3, f)(c[2]), Sh(f)(c[3]))
def Emix(pairs): return tuple(mix([(w, c[i]) for w, c in pairs]) for i in range(4))
def C(f): return lambda p: (f(p[0]), E(f)(p[1]))
def eps(p): return p[0]
def j(c): return lambda y: (y, c)
def dlt(p): x, c = p; return (p, E(j(c))(c))
eta = delta
def lam_bar(M):
    return (Dmap(eps, M), Emix([(a, E(eta)(p[1])) for p, a in M.items()]))
def K1(h): return (h[0], delta(h[1]))
def lam_sig(M):
    P = Dmap(eps, M)
    L = [Dist([(p[1][i], a) for p, a in M.items()]) for i in range(3)]
    Sg = mix([(a, Dmap(K1, p[1][3])) for p, a in M.items()])
    return (P, (L[0], L[1], L[2], Sg))

R_ = random.Random(7)
def rdist(vals, k=2):
    vs = R_.sample(vals, min(k, len(vals))); return Dist(list(zip(vs, S.rand_weights(R_, len(vs)))))
def rctx(X):
    D1 = [delta(x) for x in X] + [rdist(X) for _ in range(2)]
    D2 = [rdist(D1) for _ in range(3)]; D3 = [rdist(D2) for _ in range(3)]
    hyps = [(R_.choice(THETA), R_.choice(D1)) for _ in range(3)]
    return (R_.choice(D1), R_.choice(D2), R_.choice(D3), rdist(hyps))
def coherent_ctx(X):
    D1 = [delta(x) for x in X] + [rdist(X) for _ in range(2)]
    D2 = [rdist(D1) for _ in range(3)]; Om = rdist([rdist(D2) for _ in range(3)])
    Pi = mu(Om); rho = mu(Pi); Sig = Dmap(lambda r: ("t1", r), Pi)
    return (rho, Pi, Om, Sig)
def is_coherent(c):
    rho, Pi, Om, Sig = c
    return mu(Pi) == rho and mu(Om) == Pi and Dmap(lambda h: h[1], Sig) == Pi
def rpoint(X): return (R_.choice(X), rctx(X))
def rM(Xs, n=2): return rdist([rpoint(Xs) for _ in range(4)], n)
X = (0, 1)
f = {0: "a", 1: "a"}.get; g = {0: "a", 1: "b"}.get
def ctx(rho): return (rho, delta(delta(0)), delta(delta(delta(0))), delta(("t1", delta(0))))
def half():
    return S.rand_weights.__name__ == "_trop" and [F(0), F(0)] or ([F(1, 2), F(1, 2)] if S.name != "C" else [Gauss(F(1, 2)), Gauss(F(1, 2))])

def beck_table():
    """the main paper's Beck table (check_tower.py), in the current row; returns {name: bool}"""
    out = {}
    ok = {k: True for k in ("counit1", "counit2", "coassoc", "nat_eps", "nat_delta")}
    for _ in range(25):
        p = rpoint(X)
        ok["counit1"] &= eps(dlt(p)) == p; ok["counit2"] &= C(eps)(dlt(p)) == p; ok["coassoc"] &= C(dlt)(dlt(p)) == dlt(dlt(p))
        for h in (f, g): ok["nat_eps"] &= eps(C(h)(p)) == h(eps(p)); ok["nat_delta"] &= dlt(C(h)(p)) == C(C(h))(dlt(p))
    out["comonad"] = all(ok.values())
    okn = oku = okc = okm = True
    for _ in range(15):
        M = rM(X)
        for h in (f, g): okn &= C(Dk(1, h))(lam_bar(M)) == lam_bar(Dmap(C(h), M))
        okc &= eps(lam_bar(M)) == Dmap(eps, M); p = rpoint(X); oku &= lam_bar(delta(p)) == C(eta)(p)
        Mp = rdist([rM(X) for _ in range(3)], 2); okm &= lam_bar(mu(Mp)) == C(mu)(lam_bar(Dmap(lam_bar, Mp)))
    out.update({"avg natural": okn, "avg unit": oku, "avg counit": okc, "avg multiplication": okm})
    w = half(); c1, c2 = ctx(delta(0)), ctx(delta(1)); M = Dist([((0, c1), w[0]), ((0, c2), w[1])])
    out["avg comultiplication fails"] = dlt(lam_bar(M)) != C(lam_bar)(lam_bar(Dmap(dlt, M)))
    oksn = oksc = True
    for _ in range(15):
        Mr = rM(X)
        for h in (f, g): oksn &= C(Dk(1, h))(lam_sig(Mr)) == lam_sig(Dmap(C(h), Mr))
        oksc &= eps(lam_sig(Mr)) == Dmap(eps, Mr)
    out.update({"shift natural": oksn, "shift counit": oksc})
    pt = (0, (Dist([(0, w[0]), (1, w[1])]),) + ctx(delta(0))[1:])
    out["shift unit fails"] = lam_sig(delta(pt)) != C(eta)(pt)
    cA, cB = ctx(delta(0)), ctx(delta(1)); Mp = delta(Dist([((0, cA), w[0]), ((1, cB), w[1])]))
    out["shift multiplication fails"] = lam_sig(mu(Mp)) != C(mu)(lam_sig(Dmap(lam_sig, Mp)))
    return out

results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

tables = {}
for nm in ("R>=0", "R", "C", "T"):
    S = ROWS[nm]; R_ = random.Random(7); tables[nm] = beck_table()
keys = list(tables["R>=0"])
print("Beck table by row:"); print("  " + " | ".join(f"{k}" for k in ROWS))
for k in keys: print(f"  {k:28s} " + "  ".join("yes" if tables[r][k] else "NO " for r in ROWS))
same = all(tables[r] == tables["R>=0"] for r in ROWS)
check("TG1 comonad laws of C_3 hold in every row", all(tables[r]["comonad"] for r in ROWS))
check("TG2 averaging law: natural, unit, counit, multiplication hold; comultiplication fails -- in every row",
      all(tables[r][k] for r in ROWS for k in ("avg natural", "avg unit", "avg counit", "avg multiplication", "avg comultiplication fails")))
check("TG3 shift law: natural, counit hold; unit and multiplication fail -- in every row",
      all(tables[r][k] for r in ROWS for k in ("shift natural", "shift counit", "shift unit fails", "shift multiplication fails")),
      "the Beck table is row-independent" if same else "rows differ")

# TG4: change of base R>=0 -> R -> C
def deep(phi, obj):
    if isinstance(obj, Dist): return Dist([(deep(phi, v), phi(p)) for v, p in obj.items()])
    if isinstance(obj, tuple): return tuple(deep(phi, o) for o in obj)
    return obj
ok = True
for src, dst in (("R>=0", "R"), ("R", "C")):
    for _ in range(12):
        S = ROWS[src]; R_ = random.Random(R_.randint(0, 10**6)); c = coherent_ctx(X); M = rM(X)
        lb, ls = lam_bar(M), lam_sig(M); coh = is_coherent(c)
        S = ROWS[dst]; phi = ROWS[dst].from_frac if src != "C" else None
        cd = deep(phi, c); Md = deep(phi, M)
        ok &= coh and is_coherent(cd) and deep(phi, lb) == lam_bar(Md) and deep(phi, ls) == lam_sig(Md)
check("TG4 change of base R>=0 -> R -> C preserves coherence and commutes with both laws", ok,
      "tropical row: the valuation monad morphism of the main paper (check_tropical.py T1, T2)")

# TG5: positivity-dependent statements fail over R
S = ROWS["R"]
import numpy as np
rho = np.array([0.5, 0.5]); r1 = np.array([0.5, 0.5]) + np.array([0.25, -0.25]); r2 = np.array([0.5, 0.5]) + np.array([0.5, -0.5])
w = (2.0, -1.0)                                    # 2 r1 - r2 = rho: a signed level-two law with barycentre rho
bary = w[0] * r1 + w[1] * r2
phi = lambda r: float(np.sum(r ** 2))               # convex
jensen = w[0] * phi(r1) + w[1] * phi(r2) - phi(rho)
sec = [phi(rho + t * np.array([1, -1]) * 0.1) for t in (0, 1)]
v = np.array([0.25, -0.25]); fib = []
for t in (0, 1, 10, 100, 1000):
    # Pi_t = (1+t)(1/2 delta_{rho+v} + 1/2 delta_{rho-v}) - t delta_rho: a signed level-two law with barycentre rho, for every t;
    # and its mirror (1+t) delta_rho - t (1/2 delta_{rho+v} + 1/2 delta_{rho-v})
    atoms = [(rho + v, (1 + t) / 2), (rho - v, (1 + t) / 2), (rho, -t)]
    assert np.allclose(sum(w_ * r for r, w_ in atoms), rho)
    up = sum(w_ * phi(r) for r, w_ in atoms) - phi(rho)
    down = (1 + t) * phi(rho) - t * (phi(rho + v) + phi(rho - v)) / 2 - phi(rho)
    fib.append((t, up, down))
ok = np.allclose(bary, rho) and jensen < 0 and fib[-1][1] > 100 and fib[-1][2] < -100 and fib[0][1] >= 0
check("TG5 over R: Jensen fails for a coherent signed level-two law (no convex order, no martingale); the signed fibre is unbounded", ok,
      f"sum w phi(r) - phi(rho) = {jensen:.3f} < 0; Jensen gap along the fibre from {fib[0][1]:.3f} to +{fib[-1][1]:.0f} and {fib[-1][2]:.0f}")

# TG6: row splits at fixed depth
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
from pce import Window
from pce.ladder import level1, level2_m3
pr = lambda par: np.array([[0.5, 0], [0, 0.5]]) if par else np.array([[0, 0.5], [0.5, 0]])
W = [Window(("A0", "B0"), pr(1)), Window(("A0", "B1"), pr(1)), Window(("A1", "B0"), pr(1)), Window(("A1", "B1"), pr(0))]
l1 = level1({"A0": 2, "A1": 2, "B0": 2, "B1": 2}, W)
rng = np.random.default_rng(0); cpx = None
for _ in range(5000):
    Q = rng.dirichlet(np.ones(8)).reshape(2, 2, 2)
    if level2_m3(Q)["rung"] == "C": cpx = Q; break
check("TG6 row splits: level one (PR box) has an R-section and no R>=0-section; level two has a C-fibre and no R-fibre",
      l1["linear_gap"] < 1e-9 and l1["negativity"] > 0.1 and l1["lam"] < 1e-9 and cpx is not None,
      f"PR box: linear gap {l1['linear_gap']:.1e}, negativity {l1['negativity']:.2f}, lam* {l1['lam']:.2f}; complex-only table found")

print(f"\n{sum(results)}/{len(results)} checks passed")
