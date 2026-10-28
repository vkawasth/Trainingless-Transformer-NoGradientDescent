"""check_scheme.py -- scheme structure of the two-type latent-class model (note: level2/partial_gluing.pdf, Section 7). Exact (sympy).

Coordinates: weights w1 (w2 = 1 - w1), mean mu = w1 a1 + w2 a2, difference d = a1 - a2 (a polynomial automorphism:
a1 = mu + w2 d, a2 = mu - w1 d); observed side: mean and central moments m_S, |S| >= 2 (a unitriangular polynomial change of the law).

  SC1 moment normal form: m_S = c_|S|(w) d_S with c_k(w) = w1 w2^k + w2 (-w1)^k (m = 3, 4); for fixed w a monomial map in d
  SC2 the fibre over a product law is the Stanley-Reisner scheme: its ideal is (mu - c) + (w1 w2 d_i d_j : i < j) and equals the
      intersection of the primes (w1), (w2), (d_j : j != i) (m = 3, 4); hence reduced
  SC3 m = 3, exact inverse: p = w1 w2 = Delta / Det, q = w2 - w1 = +/- sqrt(1 - 4p), d_i = m123 / (q m_jk), with Delta = m12 m13 m23 and
      Det = m123^2 + 4 Delta; the Jacobian determinant is -(w1 w2)^3 (d1 d2 d3)^2
  SC4 Det is Cayley's hyperdeterminant of the 2x2x2 table
  SC5 m = 3 flattening stratification: generic laws have fibre 2 reduced points; Delta > 0 -> real weights in (0, 1); Det > 0 > Delta ->
      real with weights outside [0, 1]; Det < 0 -> complex-conjugate pair; one variable independent -> a surface; one pairwise moment
      zero alone -> empty
  SC6 m = 4: a variable independent of the rest (d4 = 0) is a flat point: full rank and fibre of 2 points
  SC7 pairwise windows: with v = sqrt(w1 w2) d and r = (w2 - w1)/sqrt(w1 w2), m_S = v_S s_|S|(r) where s_2 = 1, s_3 = r and
      s_(k+1) = r s_k + s_(k-1) (exact, k <= 7); so the pairwise tables see mu and v but not w (rank 2m of 2m + 1 for m = 3, 4, 5), every
      event probability is a polynomial in r of degree <= m - 2, and one triple window with v_S != 0 pins r
Writes ../level2/scheme_results.json (region map and a path across the walls for the figure).  Run: python check_scheme.py
"""
import itertools, json, os
import numpy as np
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); R = sp.Rational
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

w1 = sp.Symbol("w1"); w2 = 1 - w1
ck = lambda k: w1 * w2 ** k + w2 * (-w1) ** k
def coords(m): return sp.symbols(f"mu1:{m+1}"), sp.symbols(f"d1:{m+1}")
def law(m, mu, d, c=None):
    a1 = [mu[i] + w2 * d[i] for i in range(m)]; a2 = [mu[i] - w1 * d[i] for i in range(m)]
    pr = lambda a, x: sp.Mul(*[(a[i] if x[i] else 1 - a[i]) for i in range(m)])
    return a1, a2, {x: sp.expand(w1 * pr(a1, x) + w2 * pr(a2, x) - (pr(c, x) if c else 0)) for x in itertools.product([0, 1], repeat=m)}

# SC1
ok = True
for m in (3, 4):
    mu, d = coords(m); a1, a2, _ = law(m, mu, d)
    for k in range(2, m + 1):
        for S in itertools.combinations(range(m), k):
            cm = w1 * sp.Mul(*[a1[i] - mu[i] for i in S]) + w2 * sp.Mul(*[a2[i] - mu[i] for i in S])
            ok &= sp.expand(cm - ck(k) * sp.Mul(*[d[i] for i in S])) == 0
check("SC1 moment normal form m_S = c_|S|(w) d_S (toric in d for fixed weights)", ok, "m = 3, 4")

# SC2
def intersect(I, J, vars_):
    t = sp.Symbol("t_elim"); G = sp.groebner([t * f for f in I] + [(1 - t) * g for g in J], t, *vars_, order="lex")
    return [g for g in G.exprs if not g.has(t)]
ok = True
for m in (3, 4):
    mu, d = coords(m); c = [R(2, 5), R(1, 2), R(3, 5), R(1, 3)][:m]; vars_ = list(d) + [w1] + list(mu)
    _, _, gens = law(m, mu, d, c)
    G1 = sp.groebner(list(gens.values()), *vars_, order="grevlex")
    G2 = sp.groebner([mu[i] - c[i] for i in range(m)] + [w1 * w2 * d[i] * d[j] for i, j in itertools.combinations(range(m), 2)], *vars_, order="grevlex")
    ok &= G1.exprs == G2.exprs
    comps = [[w1], [w2]] + [[d[j] for j in range(m) if j != i] for i in range(m)]; acc = comps[0]
    for C in comps[1:]: acc = intersect(acc, C, list(d) + [w1])
    ok &= sp.groebner(acc, *(list(d) + [w1]), order="grevlex").exprs == sp.groebner([w1 * w2 * d[i] * d[j] for i, j in itertools.combinations(range(m), 2)], *(list(d) + [w1]), order="grevlex").exprs
check("SC2 fibre over a product law = Stanley-Reisner scheme (mu - c, w1 w2 d_i d_j) = (w1) n (w2) n_i (d_j: j != i), reduced", ok, "m = 3, 4")

# SC3
d1, d2, d3 = sp.symbols("d1 d2 d3"); p = w1 * w2; q = w2 - w1
Mf = [p * d1 * d2, p * d1 * d3, p * d2 * d3, p * q * d1 * d2 * d3]
m12, m13, m23, m123 = sp.symbols("m12 m13 m23 m123"); Dl = m12 * m13 * m23; Det = m123 ** 2 + 4 * Dl; sub = dict(zip([m12, m13, m23, m123], Mf))
ok = sp.simplify((Dl / Det).subs(sub) - p) == 0 and sp.simplify((1 - 4 * Dl / Det).subs(sub) - q ** 2) == 0
ok &= all(sp.simplify((m123 / (q * mjk)).subs(sub) - di) == 0 for di, mjk in [(d1, m23), (d2, m13), (d3, m12)])
jac = sp.factor(sp.Matrix(Mf).jacobian([w1, d1, d2, d3]).det()); ok &= sp.expand(jac + (w1 * w2) ** 3 * (d1 * d2 * d3) ** 2) == 0
check("SC3 m = 3 exact inverse p = Delta/Det, q = +/-sqrt(1-4p), d_i = m123/(q m_jk); Jacobian det = -(w1 w2)^3 (d1 d2 d3)^2", ok, f"det = {jac}")

# SC4
mu3 = sp.symbols("mu1:4"); mom = {(0, 1): m12, (0, 2): m13, (1, 2): m23, (0, 1, 2): m123}
def P(x):
    e = 0
    for k in range(4):
        for S in itertools.combinations(range(3), k):
            f = mom.get(S, 0) if k >= 2 else (1 if k == 0 else 0)
            e += f * sp.Mul(*[(mu3[i] if x[i] else 1 - mu3[i]) for i in range(3) if i not in S]) * sp.Mul(*[(1 if x[i] else -1) for i in S])
    return sp.expand(e)
a = {x: P(x) for x in itertools.product([0, 1], repeat=3)}; A = lambda i, j, k: a[(i, j, k)]
H = (A(0,0,0)**2*A(1,1,1)**2 + A(0,0,1)**2*A(1,1,0)**2 + A(0,1,0)**2*A(1,0,1)**2 + A(1,0,0)**2*A(0,1,1)**2
     - 2*(A(0,0,0)*A(0,0,1)*A(1,1,0)*A(1,1,1) + A(0,0,0)*A(0,1,0)*A(1,0,1)*A(1,1,1) + A(0,0,0)*A(1,0,0)*A(0,1,1)*A(1,1,1)
          + A(0,0,1)*A(0,1,0)*A(1,0,1)*A(1,1,0) + A(0,0,1)*A(1,0,0)*A(0,1,1)*A(1,1,0) + A(0,1,0)*A(1,0,0)*A(0,1,1)*A(1,0,1))
     + 4*(A(0,0,0)*A(0,1,1)*A(1,0,1)*A(1,1,0) + A(0,0,1)*A(0,1,0)*A(1,0,0)*A(1,1,1)))
check("SC4 Det = m123^2 + 4 m12 m13 m23 is Cayley's hyperdeterminant of the 2x2x2 table", sp.expand(H - Det) == 0 and sp.simplify(sum(a.values()) - 1) == 0)

# SC5
X = [d1, d2, d3, w1]
def fibre(vals): return sp.groebner([sp.expand(f - v) for f, v in zip(Mf, vals)], *X, order="lex")
def pts(G): return sp.solve(G.exprs, X, dict=True)
def real(s): return all(abs(complex(sp.N(v)).imag) < 1e-12 for v in s.values())
v_mix = [f.subs({w1: R(3, 10), d1: R(1, 2), d2: R(-1, 5), d3: R(3, 10)}) for f in Mf]
S1 = pts(fibre(v_mix)); ok = len(S1) == 2 and all(real(s) and 0 < s[w1] < 1 for s in S1)
S2 = pts(fibre([R(-1, 50), R(1, 40), R(1, 60), R(1, 10)])); ok &= len(S2) == 2 and all(real(s) and not (0 <= s[w1] <= 1) for s in S2)
S3 = pts(fibre([-v_mix[0], v_mix[1], v_mix[2], v_mix[3]])); ok &= len(S3) == 2 and not any(real(s) for s in S3)
ok &= fibre([R(1, 20), 0, 0, 0]).exprs == sp.groebner([d3, sp.expand(p * d1 * d2 - R(1, 20))], *X, order="lex").exprs
ok &= fibre([0, R(1, 20), R(1, 30), R(1, 100)]).exprs == [1]
check("SC5 m = 3 strata: 2 points (mixture / signed / complex pair by the signs of Delta and Det); one variable independent: a surface; "
      "one pairwise moment zero alone: empty", ok, f"signed example w1 = {sp.N(S2[0][w1], 4)}, {sp.N(S2[1][w1], 4)}")

# SC6
d4 = sp.symbols("d1:5"); X4 = list(d4) + [w1]
M4 = [ck(len(S)) * sp.Mul(*[d4[i] for i in S]) for k in range(2, 5) for S in itertools.combinations(range(4), k)]
pt = {w1: R(3, 10), d4[0]: R(1, 2), d4[1]: R(-1, 5), d4[2]: R(3, 10), d4[3]: 0}
rk = sp.Matrix(M4).jacobian(X4).subs(pt).rank()
S4 = sp.solve(sp.groebner([sp.expand(f - f.subs(pt)) for f in M4], *X4, order="lex").exprs, X4, dict=True)
check("SC6 m = 4: a variable independent of the rest is a flat point (full rank, fibre of 2 points)", rk == 5 and len(S4) == 2, f"rank {rk}/5, {len(S4)} points")

# SC7 pairwise windows
w1 = sp.symbols("w1", positive=True); w2 = 1 - w1; rr = (w2 - w1) / sp.sqrt(w1 * w2)
ck = lambda k: w1 * w2 ** k + w2 * (-w1) ** k
sk = [None, None, sp.Integer(1), rr]
for k in range(4, 8): sk.append(sp.expand(rr * sk[k - 1] + sk[k - 2]))
ok7 = all(sp.simplify(ck(k) / (w1 * w2) ** sp.Rational(k, 2) - sk[k]) == 0 for k in range(2, 8))
rng7 = np.random.default_rng(7); ranks = []
for m in (3, 4, 5):
    def Fm(th, wins):
        w = th[0]; a1 = th[1:1 + m]; a2 = th[1 + m:]
        return np.array([w * np.prod(a1[list(S)]) + (1 - w) * np.prod(a2[list(S)]) for S in wins])
    sing = [(i,) for i in range(m)]; pairs = list(itertools.combinations(range(m), 2)); tri = [tuple(range(3))]
    th = np.r_[0.35, rng7.uniform(.1, .9, 2 * m)]; h = 1e-6
    def rk(wins):
        J = np.array([(Fm(th + h * e, wins) - Fm(th - h * e, wins)) / (2 * h) for e in np.eye(2 * m + 1)]).T
        sv = np.linalg.svd(J, compute_uv=False); return int((sv > 1e-7 * sv[0]).sum())
    ranks.append((m, rk(sing + pairs), rk(sing + pairs + tri)))
okr = all(a == 2 * m and b == 2 * m + 1 for m, a, b in ranks)
check("SC7 pairwise windows see mu and sqrt(w1 w2) d but not w: m_S = v_S s_|S|(r) with s_(k+1) = r s_k + s_(k-1); one triple pins r",
      ok7 and okr, "recurrence exact for k <= 7; ranks (m, pairs, pairs + triple): " + ", ".join(f"({m},{a},{b})" for m, a, b in ranks))

# figure data: region map on the slice m13 = m23 = k, and a path m12 = s across the walls (m123 fixed)
k = 0.03; m123v = 0.002; G12 = np.linspace(-0.004, 0.004, 401); G123 = np.linspace(-0.012, 0.012, 401)
reg = [[(2 if x > 0 else (1 if y * y + 4 * x * k * k > 0 else 0)) for x in G12] for y in G123]
path = []
for s in np.linspace(-0.004, 0.004, 801):
    if abs(s) < 1e-12: continue
    D = s * k * k; De = m123v ** 2 + 4 * D; pp = D / De; qq2 = 1 - 4 * pp
    if qq2 < 0: path.append([float(s), None, None, None]); continue
    qq = np.sqrt(qq2); ww = (1 - qq) / 2; dmax = max(abs(m123v / (qq * k)), abs(m123v / (qq * s)))
    path.append([float(s), float(ww), float(1 - ww), float(dmax)])
json.dump({"k": k, "m123": m123v, "x": G12.tolist(), "y": G123.tolist(), "region": reg, "path": path},
          open(os.path.join(HERE, "..", "level2", "scheme_results.json"), "w"))
print(f"\n{sum(results)}/{len(results)} checks passed")
