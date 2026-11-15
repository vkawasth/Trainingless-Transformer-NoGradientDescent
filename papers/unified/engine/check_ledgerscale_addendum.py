"""Study 28 addendum (written AFTER the first run of check_ledgerscale.py, which failed LS2 at 2 x 8: 9 solves for one objective against
C(8, 1) = 8). prop:basiscount(i) is stated for a GENERIC linear objective; the pre-registered LS2 drew integer objectives in [-20, 20]
without testing genericity, and the offending objective has tied row differences (n_{0j} - n_{1j} = 13 for j = 5 and 7), so a 2 x 2
cycle has zero reduced cost and the optimum is not unique. LS2 stays FAIL as pre-registered; this addendum is labelled post hoc.
  LS8 IDENTITY (post hoc)  for the same 45 objectives (15 shapes x 3) and populations: an objective is dual degenerate when some basis in
                           its library has a nonbasic cell of zero reduced cost; every objective whose solves exceed C(r+c-2, r-1) is dual
                           degenerate, and every non-degenerate objective respects the bound
"""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "check_ledgerscale.py")).read().split("SHAPES =")[0]; exec(src)
SHAPES = [(2, n) for n in range(2, 9)] + [(3, n) for n in range(3, 8)] + [(4, n) for n in range(4, 7)]
over = []; degen = []; ok = True
for si, (r, c) in enumerate(SHAPES):
    P = SL.Problem(r, c); pop = population(r, c, 1009, 20261200 + 10 * r + c, 10000); Bm = np.array([P.b(f, g) for f, g in pop]).T
    cap = math.comb(r + c - 2, r - 1)
    for o in range(3):
        nv = np.random.default_rng(7 + si * 10 + o).integers(-20, 21, size=P.N).astype(float); L = SL.Library(P)
        a, _, s, _, _, _ = timed_build(P, L, Bm, nv, np.zeros(P.N), +1, np.random.default_rng(1).permutation(10000))
        dg = False
        for k, B in enumerate(L.bases):
            B = list(B); Nn = [j for j in range(P.N) if j not in B]; rc = nv[Nn] - nv[B] @ (L.inv[k] @ P.A[:, Nn]); dg |= bool((abs(rc) < 1e-9).any())
        if dg: degen.append(f"{r}x{c}#{o} ({s}/{cap})")
        if s > cap: over.append(f"{r}x{c}#{o}")
        ok &= (s <= cap) or dg
check("LS8 IDENTITY (post hoc) every objective above the bound is dual degenerate (degenerate ones may still meet it)", ok and all(any(x.startswith(o) for x in degen) for o in over),
      f"above bound {over}; dual-degenerate objectives {degen}")
print(f"{sum(results)}/{len(results)} passed")
