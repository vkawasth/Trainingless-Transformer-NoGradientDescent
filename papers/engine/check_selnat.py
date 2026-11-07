"""Checks for sections 12.13-12.14 of the companion note (module pce/selnat.py): a small dagger category for finite probability, and how
little it takes to change a composed decision. Each check is marked IDENTITY (an exact instance of a statement proved in the text) or
CERTIFICATE (an exact counterexample or an exact computed value, with all inputs printed).
  NC1  IDENTITY     Markov gluing on random rational chains: associative, unital, a dagger (transpose reverses composition), and monoidal
                    (the tensor product of couplings is functorial); it commutes with coarse-graining an end space
  NC2  IDENTITY     selection: closed (the composite lies in Pi(p, r)), unital on both sides, reversal-compatible, relabelling-equivariant
  NC3  IDENTITY     selection commutes with tensoring by an identity coupling diag(w) on both sides (a copied factor)
  NC4  CERTIFICATE  selection is not monoidal: sigma(g x g', d x d') != sigma(g, d) x sigma(g', d'), while Markov gluing is
  NC5  CERTIFICATE  selection does not commute with coarse-graining the end space, while Markov gluing does
  NC6  CERTIFICATE  coarse-graining the middle space changes both compositions (one exact chain each; selection on 15 random chains)
  NC7  IDENTITY     binary ends: the composites compatible with the data fill [sum L_b, sum U_b] (LP over the fibre, 20 chains); selection is the
                    midpoint, so its worst-case error over the fibre is the half-width, never more than Markov's
  NC8  CERTIFICATE  the umbrella example: the interval, the two composites, the worst-case errors, and the smallest margin-preserving edits that
                    flip each decision at tau = 1/2 (exact roots, checked on both sides)
  NC9  CERTIFICATE  zero-cost changes in the umbrella example: refining the hidden middle flips the Markov decision and never the selected one
                    (and only shrinks the interval); recording the end space more finely, or attaching an independent factor, moves selection and
                    leaves Markov gluing fixed
"""
import os, sys, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import selnat as N
from pce import selcomp as S
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(13)
def rnd(r, c):
    x = [[Fr(int(rng.integers(1, 10))) for _ in range(c)] for _ in range(r)]; t = sum(map(sum, x)); return [[v / t for v in row] for row in x]
def after(g, c):                                       # a random coupling whose row marginal is the column marginal of g
    q = N.cols(g); k = [[Fr(int(rng.integers(1, 10))) for _ in range(c)] for _ in q]
    return [[q[j] * v / sum(k[j]) for v in k[j]] for j in range(len(q))]

# NC1
ok = True
for _ in range(12):
    g1 = rnd(2, 3); g2 = after(g1, 2); g3 = after(g2, 3)
    ok &= N.eq(N.markov(N.markov(g1, g2), g3), N.markov(g1, N.markov(g2, g3)))
    ok &= N.eq(N.markov(N.diag(N.rows(g1)), g1), g1) and N.eq(N.markov(g1, N.diag(N.cols(g1))), g1)
    ok &= N.eq(N.T(N.markov(g1, g2)), N.markov(N.T(g2), N.T(g1)))
    h1 = rnd(2, 2); h2 = after(h1, 2)
    ok &= N.eq(N.markov(N.kron(g1, h1), N.kron(g2, h2)), N.kron(N.markov(g1, g2), N.markov(h1, h2))) and N.eq(N.T(N.kron(g1, h1)), N.kron(N.T(g1), N.T(h1)))
    ok &= N.eq(N.markov(g2, N.merge_cols(g3, [[0], [1, 2]])), N.merge_cols(N.markov(g2, g3), [[0], [1, 2]]))
check("NC1 IDENTITY Markov gluing: associative, unital, dagger, monoidal, natural for end coarse-graining (12 random chains)", ok)

# NC2
ok = True; n = 0
for _ in range(8):
    g1 = rnd(2, 3); g2 = after(g1, 2); s = N.sel(g1, g2); n += 1
    ok &= N.rows(s) == N.rows(g1) and N.cols(s) == N.cols(g2) and min(min(r) for r in s) >= 0
    ok &= N.eq(N.sel(g1, N.diag(N.cols(g1))), g1) and N.eq(N.sel(N.diag(N.rows(g1)), g1), g1)
    ok &= N.eq(N.sel(N.T(g2), N.T(g1)), N.T(s))
    P = [1, 2, 0]                                       # relabel the middle states; and swap the two end states of X_0
    gp = [[g1[i][P[j]] for j in range(3)] for i in (1, 0)]; dp = [g2[P[j]] for j in range(3)]
    ok &= N.eq(N.sel(gp, dp), [s[1], s[0]])
check("NC2 IDENTITY selection: closed, unital on both sides, reversal-compatible, relabelling-equivariant", ok, f"{n} chains")

# NC3
ok = True
g, d = N.umbrella()
for w in ([Fr(1, 4), Fr(3, 4)], [Fr(2, 5), Fr(3, 5)]):
    ok &= N.eq(N.sel(N.kron(g, N.diag(w)), N.kron(d, N.diag(w))), N.kron(N.sel(g, d), N.diag(w)))
g1 = rnd(2, 3); g2 = after(g1, 2); w = [Fr(1, 3), Fr(2, 3)]
ok &= N.eq(N.sel(N.kron(g1, N.diag(w)), N.kron(g2, N.diag(w))), N.kron(N.sel(g1, g2), N.diag(w)))
check("NC3 IDENTITY selection commutes with tensoring by an identity coupling (a copied factor)", ok)

# NC4
g, d = N.umbrella(); gq = N.cpl(Fr(1, 4), Fr(1, 2), Fr(1, 5)); dq = N.cpl(Fr(1, 2), Fr(3, 5), Fr(1, 10))
lhs = N.sel(N.kron(g, gq), N.kron(d, dq)); rhs = N.kron(N.sel(g, d), N.sel(gq, dq))
mk = N.eq(N.markov(N.kron(g, gq), N.kron(d, dq)), N.kron(N.markov(g, d), N.markov(gq, dq)))
check("NC4 CERTIFICATE selection is not monoidal; Markov gluing is", lhs[0][0] == Fr(29, 1400) and rhs[0][0] == Fr(31, 1200) and mk,
      f"g = {N.show(g)}, d = {N.show(d)}, g' = {N.show(gq)}, d' = {N.show(dq)}; 00 entries {lhs[0][0]} vs {rhs[0][0]}")

# NC5
d3 = [[Fr(1, 10), Fr(1, 10), Fr(2, 15)], [Fr(1, 5), Fr(1, 5), Fr(4, 15)]]; G = [[0], [1, 2]]
a = N.sel(g, N.merge_cols(d3, G)); b = N.merge_cols(N.sel(g, d3), G)
mk = N.eq(N.markov(g, N.merge_cols(d3, G)), N.merge_cols(N.markov(g, d3), G))
check("NC5 CERTIFICATE selection does not commute with coarse-graining the end space; Markov gluing does",
      a[0][0] == Fr(17, 120) and b[0][0] == Fr(11, 90) and mk, f"g = {N.show(g)}, d = {N.show(d3)}, merge Z states 1, 2; 00 entries {a[0][0]} vs {b[0][0]}")

# NC6
gm = [[Fr(1, 10), Fr(1, 10), Fr(1, 5)], [Fr(1, 5), Fr(3, 20), Fr(1, 4)]]; dm = [[Fr(1, 5), Fr(1, 10)], [Fr(1, 20), Fr(1, 5)], [Fr(1, 5), Fr(1, 4)]]
M1, M2 = N.markov(gm, dm)[0][0], N.markov(N.merge_cols(gm, G), N.merge_rows(dm, G))[0][0]
gs_ = [[Fr(1, 8), Fr(9, 32), Fr(3, 32)], [Fr(3, 16), Fr(1, 4), Fr(1, 16)]]; ds_ = [[Fr(5, 144), Fr(5, 18)], [Fr(85, 448), Fr(153, 448)], [Fr(5, 64), Fr(5, 64)]]
S1, S2 = N.sel(gs_, ds_)[0][0], N.sel(N.merge_cols(gs_, G), N.merge_rows(ds_, G))[0][0]
cnt = 0
for _ in range(15):
    g1 = rnd(2, 3); g2 = after(g1, 2); cnt += not N.eq(N.sel(g1, g2), N.sel(N.merge_cols(g1, G), N.merge_rows(g2, G)))
check("NC6 CERTIFICATE coarse-graining the middle changes both compositions",
      (M1, M2) == (Fr(79, 450), Fr(73, 420)) and (S1, S2) == (Fr(1283, 8064), Fr(305, 2016)) and cnt > 0,
      f"Markov: g = {N.show(gm)}, d = {N.show(dm)}: {M1} vs {M2}; selection: g = {N.show(gs_)}, d = {N.show(ds_)}: {S1} vs {S2}; selection changed on {cnt}/15 random chains")

# NC7
ok = True
for _ in range(20):
    g1 = rnd(2, 3); g2 = after(g1, 2); lo, hi = N.identified(g1, g2)
    shape, cells, A, bb = S.fibre_constraints([g1, g2]); c = np.array([1.0 if (x[0] == 0 and x[2] == 0) else 0.0 for x in cells])
    Af = np.array(A, float); bf = np.array([float(x) for x in bb])
    rmin = linprog(c, A_eq=Af, b_eq=bf, bounds=[(0, None)] * len(cells), method="highs"); rmax = linprog(-c, A_eq=Af, b_eq=bf, bounds=[(0, None)] * len(cells), method="highs")
    ok &= abs(rmin.fun - float(lo)) < 1e-9 and abs(-rmax.fun - float(hi)) < 1e-9
    s = N.sel(g1, g2)[0][0]; m = N.markov(g1, g2)[0][0]
    ok &= s == (lo + hi) / 2 and lo <= m <= hi and max(m - lo, hi - m) >= (hi - lo) / 2
check("NC7 IDENTITY binary ends: the compatible composites fill [sum L_b, sum U_b]; selection is its midpoint, worst-case error the half-width", ok, "20 chains, LP over the fibre")

# NC8
E = N.UMBRELLA; p, q, r, t1, t2, tau = E["p"], E["q"], E["r"], E["t1"], E["t2"], E["tau"]; g, d = N.umbrella()
lo, hi = N.identified(g, d); us = N.sel(g, d)[0][0]; um = N.markov(g, d)[0][0]
ok = (lo, hi, us, um) == (Fr(7, 60), Fr(2, 5), Fr(31, 120), Fr(23, 112))
ok &= (hi - lo) / 2 == Fr(17, 120) and max(um - lo, hi - um) == Fr(109, 560)
def lohi(a, b): return N.identified(N.cpl(p, q, a), N.cpl(q, r, b))
rules = {"Markov": lambda a, b: S.u_markov(p, q, r, a, b), "selection": lambda a, b: S.u_bin(p, q, r, a, b),
         "lower end": lambda a, b: lohi(a, b)[0], "upper end": lambda a, b: lohi(a, b)[1]}
want = {("Markov", 1): Fr(-1, 76), ("Markov", 2): Fr(-1, 70), ("selection", 1): Fr(-9, 140), ("selection", 2): Fr(-1, 15),
        ("lower end", 1): Fr(1, 12), ("lower end", 2): Fr(1, 12), ("upper end", 1): Fr(-89, 420), ("upper end", 2): Fr(-13, 60)}
got = {}; c = tau * p; eps = Fr(1, 10 ** 7)
for nm, f in rules.items():
    s1 = N.flip_edit(lambda t: f(t, t2), t1, max(Fr(0), p + q - 1), min(p, q), c); s2 = N.flip_edit(lambda t: f(t1, t), t2, max(Fr(0), q + r - 1), min(q, r), c)
    got[(nm, 1)], got[(nm, 2)] = s1, s2
    for (k, s) in ((1, s1), (2, s2)):                    # the decision is unchanged strictly before the root and changed just after it
        F = (lambda u: f(t1 + u, t2)) if k == 1 else (lambda u: f(t1, t2 + u))
        sg = 1 if s > 0 else -1
        ok &= F(s) == c and ((F(s - sg * eps) > c) == (F(Fr(0)) > c)) and ((F(s + sg * eps) > c) != (F(Fr(0)) > c))
ok &= got == want
check("NC8 CERTIFICATE the umbrella example: interval, composites, worst-case errors, smallest flipping edits (TV = 2|s|)", ok,
      f"interval {lo}..{hi}, selection {us}, Markov {um}; edits " + ", ".join(f"{k[0]} on g{k[1]}: {v}" for k, v in got.items()))

# NC9
ok = True; nref = 0; mflip = []; smin = None; inside = True
for x in [Fr(k, 120) for k in range(21)]:
    for y in [Fr(k, 120) for k in range(21)]:
        gf = [[x, g[0][0] - x, g[0][1]], [Fr(1, 6) - x, g[1][0] - (Fr(1, 6) - x), g[1][1]]]
        df = [[y, Fr(1, 6) - y], [d[0][0] - y, d[0][1] - (Fr(1, 6) - y)], d[1]]
        if min(min(rr) for rr in gf) < 0 or min(min(rr) for rr in df) < 0: continue
        nref += 1; L, U = N.identified(gf, df); ms = N.markov(gf, df)[0][0]; ss = S.select_binary(gf, df)
        ok &= lo <= L <= U <= hi and ss == (L + U) / 2 and N.eq(N.merge_cols(gf, [[0, 1], [2]]), g) and N.eq(N.merge_rows(df, [[0, 1], [2]]), d)
        inside &= L < c < U
        if ms < c: mflip.append((x, y, ms))
        smin = ss if smin is None else min(smin, ss)
ok &= len(mflip) > 0 and smin > c and inside
x0, y0, m0 = mflip[0]
# record the end space more finely: split 'wet' into two states
fine = []; GW = [[0, 1], [2]]
for i in range(1, 8):
    for j in range(1, 8):
        a_, b_ = t2 * Fr(i, 8), (r - t2) * Fr(j, 8); df = [[a_, t2 - a_, q - t2], [b_, r - t2 - b_, 1 - q - r + t2]]
        ok &= N.merge_cols(N.markov(g, df), GW)[0][0] == um
        fine.append(N.merge_cols(N.sel(g, df), GW)[0][0])
moved = sum(v != us for v in fine)
# attach an independent factor W (a coin with its own two couplings) and read off the (forecast, wet) marginal
W1 = N.cpl(Fr(4, 5), Fr(2, 5), Fr(9, 25)); W2 = N.cpl(Fr(2, 5), Fr(7, 10), Fr(13, 100))
P = N.sel(N.kron(g, W1), N.kron(d, W2)); ind = P[0][0] + P[0][1] + P[1][0] + P[1][1]
Pm = N.markov(N.kron(g, W1), N.kron(d, W2)); indm = Pm[0][0] + Pm[0][1] + Pm[1][0] + Pm[1][1]
ok &= moved > 0 and ind != us and indm == um and ind > c
check("NC9 CERTIFICATE zero-cost changes: hidden refinement flips Markov, never selection; finer end and an independent factor move selection only", ok,
      f"{nref} refinements, Markov flips on {len(mflip)} (e.g. x = {x0}, y = {y0}: P(wet | rain) = {m0 / p}), selection min {smin / p}, interval always straddles tau; "
      f"finer end moves selection on {moved}/49 (range {min(fine) / p}..{max(fine) / p}); independent factor: selection {ind / p}, Markov {indm / p}")
print(f"{sum(results)}/{len(results)} passed")
