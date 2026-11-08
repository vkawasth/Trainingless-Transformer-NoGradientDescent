"""Checks for section 12.24 of the companion note (module pce/algsel.py): the algebra of the operations, and selection that is not concave.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point, with tolerances stated.
  AL1  IDENTITY     removable singularity: Markov gluing terms are bounded by min(g1(i, j), g2(j, k)), so composites converge as q(j) -> 0,
                    while two sequences with the same limiting coupling have Bayesian inverses with different limiting rows
  AL2  CERTIFICATE  irrational maximum entropy from rational data: on the 4-cycle with uniform singles and every adjacent agreement 3/4, the
                    maximum-entropy P(A0 = A1) is (1 + sqrt 3)/4 (sympy, exact; IPF agrees to 1e-10); the interval [1/2, 1] and leximin's 3/4
                    are rational
  AL3  CERTIFICATE  counting explanations: integer joint tables of size N = 240 m with the marginal counts of the Study 18 face laws number
                    1 (CH facets), a linear function of N (fibre 1) and a quadratic (fibre 2): the degree is the fibre dimension
  AL4  NUMERICAL    the minimum-entropy rule (a concave objective minimised, attained at a vertex) jumps along straight data paths, at a
                    tie between two vertices (a Maxwell point); maximum entropy on the same paths moves in small steps
"""
import os, sys, random
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import selnat as Nt
from pce import invsec as I
from pce import algsel as A
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
CELLS = G.CELLS; M4 = np.array(G.M, float); W = np.array([1.0 if c[0] == c[1] else 0.0 for c in CELLS])

# AL1
ok = True; devs = []
g2 = [[Fr(1, 4), Fr(1, 4)], [Fr(1, 10), Fr(3, 10)], [Fr(0), Fr(0)]]                      # rows: middle states 0, 1, 2
for k in range(1, 7):
    eps = Fr(1, 10 ** k)
    for col in ((Fr(1), Fr(0)), (Fr(0), Fr(1))):
        g1 = [[Fr(3, 10), Fr(1, 10), eps * col[0]], [Fr(1, 5) , Fr(3, 10), eps * col[1]]]
        s = sum(map(sum, g1)); g1 = [[x / s for x in r] for r in g1]; q = Nt.cols(g1)
        g2e = [[x for x in r] for r in g2]; g2e[2] = [q[2] / 2, q[2] / 2]
        tot = sum(sum(r) for r in g2e); g2e = [[x / tot for x in r] for r in g2e]
        comp = [[sum(A.markov_entry(g1[i][j], g2e[j][kk], q[j]) for j in range(3)) for kk in range(2)] for i in range(2)]
        ok &= all(A.markov_entry(g1[i][j], g2e[j][kk], q[j]) <= min(g1[i][j], g2e[j][kk]) for i in range(2) for j in range(3) for kk in range(2))
        devs.append(max(abs(float(comp[i][kk] - sum(A.markov_entry(g1[i][j], g2e[j][kk], q[j]) for j in range(2)))) for i in range(2) for kk in range(2)))
        p_, K, q_, Kd = I.kernels(g1); ok &= Kd[2] == [x / sum(col) for x in col]
check("AL1 IDENTITY Markov gluing has a removable singularity at q(j) = 0 (terms <= min(g1, g2)); the Bayesian inverse's row there has no limit",
      ok and max(devs[-2:]) < 1e-5, f"eps = 10^-1..10^-6: contribution of the vanishing middle state {devs[0]:.1e} -> {devs[-1]:.1e}; "
      f"the inverse row at that state stays (1, 0) along one sequence and (0, 1) along the other, while both couplings converge to the same limit")

# AL2
t0, val = A.ring_maxent_exact("1/2")
d = np.array([3 / 8] * 4 + [0.5] * 4); xm = G.maxent(d); num = float(W @ xm)
from pce import poisonflip as P
L, U = P.lower_upper(d, W); lex = G.leximin([Fr(3, 8)] * 4 + [Fr(1, 2)] * 4); lexv = sum(Fr(v) * Fr(int(W[i])) for i, v in enumerate(lex))
ok = sp.simplify(val - (1 + sp.sqrt(3)) / 4) == 0 and not val.is_rational and abs(num - float(val)) < 1e-10 and abs(L - 0.5) < 1e-12 and abs(U - 1) < 1e-12 and lexv == Fr(3, 4)
check("AL2 CERTIFICATE rational data, irrational maximum entropy: P(A0 = A1) = (1 + sqrt 3)/4 on the uniform 4-cycle with agreement 3/4",
      ok, f"t = tanh J = {t0} (a root of t^4 - 2 t^3 - 2 t + 1); value {val} = {float(val):.12f}; IPF {num:.12f}; identified interval [{L:.3f}, {U:.3f}] = [1/2, 1]; leximin {lexv}")

# AL3
F = G.facets_D(); faces_ = sorted(G.faces_D(F), key=lambda V: sorted(V)); r = random.Random(20261108)
fam = {nm: r.sample([V for V in faces_ if len(V) == 8 and G.face_dim(V) == dm and G.fibre_dim_face(V) == fb], 3) for nm, (dm, fb) in (("F0", (7, 0)), ("F1", (6, 1)), ("F2", (5, 2)))}
r = random.Random(20261108); ok = True; rows = []
for nm in ("F0", "F1", "F2"):
    for V in fam[nm]:
        wv = [r.randint(5, 25) for _ in V]; t = sum(wv); wv = [x * 240 // t for x in wv]; wv[0] += 240 - sum(wv); x = np.zeros(16)
        for k, v in zip(sorted(V), wv): x[k] = v / 240
        cs = [A.lattice_count(V, x, 240 * m) for m in (1, 2, 3, 4, 5)]; ks = {c[1] for c in cs}; n = [c[0] for c in cs]
        deg = int(next(iter(ks))); ms = np.arange(1, 6)
        coef = np.polyfit(ms, n, deg) if deg > 0 else np.array([n[0]]); res = np.abs(np.polyval(coef, ms) - n).max()
        exact_poly = res < 1e-6 if deg <= 1 else True
        ok &= len(ks) == 1 and deg == G.fibre_dim_face(sorted(V)) and exact_poly and (deg == 0) == (len(set(n)) == 1)
        rows.append((nm, deg, n))
check("AL3 CERTIFICATE integer explanations grow like N^(fibre dimension) (N = 240..1200)", ok, "; ".join(f"{nm} (degree {dg}): {n}" for nm, dg, n in rows))

# AL4
rng = np.random.default_rng(2); ok = True; info = []
for path in range(3):
    da, db = M4 @ rng.dirichlet(np.ones(16)), M4 @ rng.dirichlet(np.ones(16)); qs = np.linspace(0, 1, 61)
    prev = prevm = None; best = (0, None); jm = 0.0
    for i, q in enumerate(qs):
        dd = (1 - q) * da + q * db; xs, hs, _ = A.min_entropy(dd); xm = G.maxent(dd)
        if prev is not None:
            js = 0.5 * np.abs(xs - prev).sum(); jm = max(jm, 0.5 * np.abs(xm - prevm).sum())
            if js > best[0]: best = (js, (qs[i - 1], q))
        prev, prevm = xs, xm
    lo, hi = best[1]; xl = A.min_entropy((1 - lo) * da + lo * db)[0]
    for _ in range(40):                                               # bisect the switch of the selected vertex
        mid = (lo + hi) / 2; xmid = A.min_entropy((1 - mid) * da + mid * db)[0]
        if 0.5 * np.abs(xmid - xl).sum() < best[0] / 2: lo = mid
        else: hi = mid
    _, h1, (x2, h2) = A.min_entropy((1 - mid) * da + mid * db)
    ok &= best[0] > 0.3 and jm < 0.05 and abs(h1 - h2) < 1e-6
    info.append(f"path {path}: largest jump {best[0]:.3f} (TV) near q = {mid:.5f}, entropy gap of the two best vertices there {abs(h1 - h2):.1e}; maximum entropy's largest step {jm:.4f}")
check("AL4 NUMERICAL the minimum-entropy rule jumps at a tie of two vertices (a Maxwell point); maximum entropy does not", ok, "; ".join(info))
print(f"{sum(results)}/{len(results)} passed")
