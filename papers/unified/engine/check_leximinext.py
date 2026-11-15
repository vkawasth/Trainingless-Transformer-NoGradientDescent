"""Further evidence on cor:leximin (facet-leximin = leximin over all cells) and a first-level argument. New script; modifies nothing.
  LQ1 IDENTITY  on random nondegenerate margins of shapes 2x3, 3x3, 2x4 and 3x4 (entries >= 0.001), the leximin over the facet cells and
                the leximin over all cells give the same coupling (to 1e-7)
  LQ2 IDENTITY  on the same margins, every redundant cell c dominates some facet cell k on the whole polytope (x_c >= x_k everywhere;
                one linear programme per pair)
  LQ3 IDENTITY  domination gives the first level: min over all cells equals min over facet cells at every point (checked at the
                vertices and at random points of each polytope), so the first leximin level and its optimal set agree
Not proved: that domination persists on each later optimal face, which is what equality at every level would need.
"""
import os, sys
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.couplinglp import margins_A, facets, leximin_levels
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261118); shapes = [(2, 3), (3, 3), (2, 4), (3, 4)]
diff = 0; cases = 0; red = 0; dom = 0; lvl1 = 0; per = {s: 0 for s in shapes}
for it in range(400):
    n, m = shapes[it % 4]; conc = float(rng.choice([0.3, 0.7, 1, 3]))
    f = rng.dirichlet([conc] * n); g = rng.dirichlet([conc] * m)
    if min(f.min(), g.min()) < 1e-3: continue
    N = n * m; A = margins_A(n, m); b = np.r_[f, g]; Fc = facets(f, g); R = [c for c in range(N) if c not in Fc]
    xa, _ = leximin_levels(f, g, range(N)); xf, _ = leximin_levels(f, g, Fc); cases += 1; per[(n, m)] += 1
    diff += np.abs(xa - xf).max() > 1e-7
    for c in R:
        red += 1
        for k in Fc:
            cc = np.zeros(N); cc[c] = 1; cc[k] = -1
            if linprog(cc, A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs").fun >= -1e-10: dom += 1; break
    for _ in range(20):
        x = linprog(rng.normal(size=N), A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs").x
        y = linprog(rng.normal(size=N), A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs").x; t = rng.random(); z = t * x + (1 - t) * y
        lvl1 += abs(z.min() - z[Fc].min()) > 1e-9
check("LQ1 IDENTITY facet-leximin equals all-cell leximin on nondegenerate margins", diff == 0 and cases > 300, f"{cases} margins {per}; disagreements {diff}")
check("LQ2 IDENTITY every redundant cell dominates some facet cell on the whole polytope", dom == red and red > 100, f"{red} redundant cells; dominated {dom}")
check("LQ3 IDENTITY the minimum over all cells equals the minimum over facet cells at every point tested (first level and first optimal set agree)", lvl1 == 0, f"{20 * cases} points; differences {lvl1}")
print(f"{sum(results)}/{len(results)} passed")
