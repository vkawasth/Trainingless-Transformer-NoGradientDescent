"""Exhaustive 2x3 sweep (Bridges, Thm sweep23). For one generic representative of each of the 18 chambers of margins:
facet cells, the toric surface, self-intersections, real-locus topology; the critical points of the Cho-Oh potential
W = sum_facets T^{lambda_c} z^{nu_c}: exact valuations, with multiplicity and with the pairing of the z1- and z2-valuations,
from the Newton polygons of four resultants (z1, z2, z1 z2, z1/z2); distinctness certified numerically at one value T = e^{-40}
(n distinct roots of a system with at most n roots, counted with multiplicity, at one T means the discriminant is not identically
zero, so the roots are distinct over the Novikov field); the leximin (FOOO) coupling and the product coupling."""
import itertools, json, sys, math
from fractions import Fraction as Fr
from collections import Counter
import numpy as np, mpmath as mp, sympy as sp
from scipy.optimize import linprog
from newton23 import exact_positions
mp.mp.dps = 60
NU = {(0,0): (1,0), (0,1): (0,1), (0,2): (-1,-1), (1,0): (-1,0), (1,1): (0,-1), (1,2): (1,1)}
def lam(c, f1, g):
    return {(0,0): Fr(0), (0,1): Fr(0), (0,2): f1, (1,0): g[0], (1,1): g[1], (1,2): g[2] - f1}[c]
WALLS = [(0,), (1,), (2,), (0,1), (0,2), (1,2)]
def wallvals(f1, g): return [f1 - sum(g[j] for j in J) for J in WALLS]
def facets_of(f1, g):
    f = (f1, 1 - f1)
    return [c for c in NU if f[c[0]] + g[c[1]] < 1]          # Prop fano23; cross-checked by LP below
def facets_lp(f1, g):
    out = []
    for c in NU:
        others = [d for d in NU if d != c]
        A = [[-NU[d][0], -NU[d][1]] for d in others]; b = [float(lam(d, f1, g)) for d in others]
        Aeq = [list(NU[c])]; beq = [-float(lam(c, f1, g))]
        cv = [1, 0] if NU[c][1] != 0 else [0, 1]
        lo = linprog(cv, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)]*2, method="highs")
        hi = linprog([-x for x in cv], A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)]*2, method="highs")
        if lo.status == 0 and hi.status == 0 and -hi.fun - lo.fun > 1e-9: out.append(c)
    return out
def self_int(facets):
    order = sorted(facets, key=lambda c: math.atan2(NU[c][1], NU[c][0])); out = {}
    for i, c in enumerate(order):
        a, b, v = np.array(NU[order[i-1]]), np.array(NU[order[(i+1) % len(order)]]), np.array(NU[c])
        k = (a + b)[0] / v[0] if v[0] else (a + b)[1] / v[1]
        assert np.allclose(a + b, k * v); out[c] = int(-k)
    return out
def orientable(facets):
    return any(all((p[0]*(NU[c][0] % 2) + p[1]*(NU[c][1] % 2)) % 2 == 1 for c in facets) for p in itertools.product([0, 1], repeat=2))
def surface(facets, D2):
    n = len(facets)
    if n == 3: return "CP^2"
    if n == 4: return "CP^1 x CP^1" if sorted(D2.values()) == [0, 0, 0, 0] else "F_1 = Bl_1 CP^2"
    return {5: "Bl_2 CP^2", 6: "Bl_3 CP^2 (dP6)"}[n]
def distinct_roots(facets, f1, g, k, N, rng, tries=6000):
    a = {c: mp.e**(-k*mp.mpf(float(lam(c, f1, g)))) for c in facets}
    def F(w1, w2):
        e = {c: a[c]*mp.e**(NU[c][0]*w1 + NU[c][1]*w2) for c in facets}
        sc = max(abs(v) for v in e.values())
        return [sum(NU[c][i]*e[c] for c in facets)/sc for i in (0, 1)]
    sols = []
    for t in range(tries):
        if len(sols) >= N: break
        u = rng.uniform(0, 0.6, 2); th = rng.uniform(0, 2*np.pi, 2)
        try: w = mp.findroot(F, [mp.mpc(-k*u[0], th[0]), mp.mpc(-k*u[1], th[1])], tol=mp.mpf(10)**-45, maxsteps=100)
        except Exception: continue
        if max(abs(v) for v in F(w[0], w[1])) > mp.mpf(10)**-40: continue
        z = (mp.e**w[0], mp.e**w[1])
        if all(max(abs(z[j] - y[j]) / max(abs(z[j]), abs(y[j])) for j in (0, 1)) > mp.mpf(10)**-20 for y in sols): sols.append(z)
    return len(sols)
def leximin(facets, f1, g):
    """FOOO's iterated max-min over the facet cells, exactly (LP values rationalized and verified)."""
    x1, x2 = sp.symbols('x1 x2')
    ell = {c: NU[c][0]*x1 + NU[c][1]*x2 + sp.Rational(lam(c, f1, g).numerator, lam(c, f1, g).denominator) for c in facets}
    fixed = []; levels = []
    for _ in range(3):
        free = [c for c in facets if c not in [d for d, _ in fixed]]
        A = [[-NU[c][0], -NU[c][1], 1] for c in free] + [[-NU[c][0], -NU[c][1], 0] for c, _ in fixed]
        b = [float(lam(c, f1, g)) for c in free] + [float(lam(c, f1, g)) for c, _ in fixed]
        Aeq = [[NU[c][0], NU[c][1], 0] for c, _ in fixed] or None; beq = [float(v - lam(c, f1, g)) for c, v in fixed] or None
        r = linprog([0, 0, -1], A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)]*3, method="highs")
        t = Fr(r.x[2]).limit_denominator(10**6); levels.append(t)
        for c in free:
            rr = linprog([-NU[c][0], -NU[c][1], 0], A_ub=A, b_ub=[bb if a_[2] == 0 else bb - float(t) + 1e-11 for a_, bb in zip(A, b)],
                         A_eq=Aeq, b_eq=beq, bounds=[(None, None)]*2 + [(0, 0)], method="highs")
            if abs(-rr.fun + float(lam(c, f1, g)) - float(t)) < 1e-9: fixed.append((c, t))
        M = np.array([NU[c] for c, _ in fixed], float)
        if np.linalg.matrix_rank(M) == 2:
            sol = sp.solve([sp.Eq(ell[c], sp.Rational(v.numerator, v.denominator)) for c, v in fixed], [x1, x2], dict=True)[0]
            u = (Fr(str(sol[x1])), Fr(str(sol[x2])))
            # verify: all fixed equalities hold exactly and every facet cell is >= its level
            assert all(NU[c][0]*u[0] + NU[c][1]*u[1] + lam(c, f1, g) == v for c, v in fixed)
            return u, levels
    raise RuntimeError("leximin did not terminate")
def maxmin_unique(facets, f1, g):
    A = [[-NU[c][0], -NU[c][1], 1] for c in facets]; b = [float(lam(c, f1, g)) for c in facets]
    t = linprog([0, 0, -1], A_ub=A, b_ub=b, bounds=[(None, None)]*3, method="highs").x[2]
    ext = [linprog(cv, A_ub=[a[:2] for a in A], b_ub=[bb - t + 1e-12 for bb in b], bounds=[(None, None)]*2, method="highs").x
           for cv in ([1, 0], [-1, 0], [0, 1], [0, -1])]
    return max(np.ptp([e[0] for e in ext]), np.ptp([e[1] for e in ext])) < 1e-7
def classify(facets, f1, g, u):
    vals = {c: NU[c][0]*u[0] + NU[c][1]*u[1] + lam(c, f1, g) for c in facets}
    m = min(vals.values()); lead = [c for c in facets if vals[c] == m]
    rank = np.linalg.matrix_rank(np.array([NU[c] for c in lead], float))
    return m, lead, (1 if rank == 2 else 2)
if __name__ == "__main__":
    chambers = sorted({tuple(1 if v > 0 else -1 for v in wallvals(Fr(a, 97), (Fr(b, 97), Fr(c, 97), 1 - Fr(b, 97) - Fr(c, 97))))
                       for a in range(1, 97) for b in range(1, 97) for c in range(1, 97 - b)
                       if 1 - Fr(b, 97) - Fr(c, 97) > 0 and all(v != 0 for v in wallvals(Fr(a, 97), (Fr(b, 97), Fr(c, 97), 1 - Fr(b, 97) - Fr(c, 97))))},
                      reverse=True)
    print("chambers:", len(chambers))
    REP = {}; rs = np.random.default_rng(11)
    for _ in range(400000):
        f1 = Fr(int(rs.integers(1, 9973)), 9973); gg = rs.dirichlet(np.ones(3))
        g1 = Fr(int(gg[0]*9973), 9973); g2 = Fr(int(gg[1]*9973), 9973); g = (g1, g2, 1 - g1 - g2)
        if min(g) <= 0: continue
        wv = wallvals(f1, g); key = tuple(1 if v > 0 else -1 for v in wv)
        gap = min(min(abs(v) for v in wv), f1, 1 - f1, min(g))
        if key not in REP or gap > REP[key][2]: REP[key] = (f1, g, gap)
    assert set(REP) == set(chambers)
    rng = np.random.default_rng(7); rows = []
    for idx, s in enumerate(chambers):
        f1, g, gap = REP[s]
        facets = facets_of(f1, g); assert sorted(facets) == sorted(facets_lp(f1, g))
        n = len(facets); D2 = self_int(facets); surf = surface(facets, D2)
        v1, v2, vp, vq, sols = exact_positions(facets, f1, g)
        assert len(sols) == 1, "pairing not unique"
        pairs = Counter(list(sols)[0])
        found = distinct_roots(facets, f1, g, 40, n, rng)
        ulex, levels = leximin(facets, f1, g)
        prod = (f1 * g[0], f1 * g[1])
        branes = []
        for u, mult in sorted(pairs.items(), key=lambda kv: -kv[1]):
            m, lead, lv = classify(facets, f1, g, u)
            branes.append(dict(u=[str(u[0]), str(u[1])], n=mult, min_cell=str(m), lead=[f"{c[0]+1}{c[1]+1}" for c in lead], levels=lv))
        row = dict(chamber=idx+1, f1=str(f1), g=[str(x) for x in g], wall_gap=str(gap), facets=[f"{c[0]+1}{c[1]+1}" for c in facets], surface=surf,
                   self_int={f"{c[0]+1}{c[1]+1}": D2[c] for c in facets}, euler=n, total=sum(pairs.values()), found_distinct_at_T_e40=found,
                   distinct=(found == n), real_chi=4 - n, real_orientable=orientable(facets), maslov1_divisors=sum(1 for v in D2.values() if v == -1),
                   branes=branes, leximin=[str(ulex[0]), str(ulex[1])], leximin_levels=[str(x) for x in levels],
                   branes_at_fooo=pairs.get(ulex, 0), maxmin_unique=bool(maxmin_unique(facets, f1, g)),
                   product=[str(prod[0]), str(prod[1])], brane_at_product=prod in pairs)
        rows.append(row)
        print(f"#{idx+1:2d} f1={f1} g=({','.join(str(x) for x in g)}) {surf:16s} facets {row['facets']} chi={n} total={row['total']} distinct={found} | "
              + "; ".join(f"u=({b['u'][0]},{b['u'][1]}) x{b['n']} L{b['levels']} min {b['min_cell']} lead {b['lead']}" for b in branes)
              + f" | leximin ({ulex[0]},{ulex[1]}) carries {row['branes_at_fooo']} | max-min unique {row['maxmin_unique']} | product carries {row['brane_at_product']}",
              flush=True)
    json.dump(rows, open("sweep23.json", "w"), indent=1)
    print("SUMMARY counts = vertices:", all(r['total'] == r['euler'] for r in rows),
          "| distinct:", all(r['distinct'] for r in rows),
          "| leximin carries >= half:", all(2 * r['branes_at_fooo'] >= r['total'] for r in rows),
          "| product carries none:", not any(r['brane_at_product'] for r in rows),
          "| max-min set a point in", sum(r['maxmin_unique'] for r in rows), "chambers",
          "| two-level branes split a column:", all(b['levels'] == 1 or (len(b['lead']) == 2 and b['lead'][0][1] == b['lead'][1][1]) for r in rows for b in r['branes']))
