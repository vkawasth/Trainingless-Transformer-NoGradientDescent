"""Checks for Step 4f of "Probability, Context and Differentiation: The Layers": recursive refinement of degenerate initial forms.
  DG1  the recursion is exact: along random Puiseux arcs (generic ones and ones that follow degenerate directions), the recursive limit
       equals the limit by direct expansion, on random non-negative families
  DG2  the degeneracy criterion: at an edge slope w and coefficient c > 0 the arc needs refining iff the edge polynomials of all leading
       cells vanish at c, iff P_A's edge polynomial does; when only some vanish, those cells drop out and the limit is read off at once
  DG3  the worked example (y - x - x^2)^2 against x^5, exactly: two refinements, multiplicity 2 at both; along y = x + x^2 + b x^{5/2} the
       share of source 1 is b^2/(1+b^2); along the exact branch y = x + x^2 it is 0; arcs leaving earlier give 1
  DG4  random families built from known roots: every node of the refinement tree is a truncation of a root of some cell, and the
       recursion agrees with direct expansion along arcs that follow each root to a random depth and then leave it
  DG5  termination: along every chain of the tree the multiplicity is non-increasing, and the depth is finite (no chain reaches the cap
       unless the cells share an exact root)
  DG6  an infinite root cut by a competitor: (y^2 - x^2 - x^3)^2 against x^K follows the series y = x sqrt(1+x) = x + x^2/2 - x^3/8 + ...
       for exactly (K - 2)/2 - 1 refinements (K = 8, 12, 16), with multiplicity 2 throughout; without the competitor the chain runs to the cap
"""
import os, sys, json, random
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import degen as DG
random.seed(20261101); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
sq = lambda f: DG._pmul(f, f)
def root_poly(phi):                                    # y - phi(x), phi = [(exponent int, coeff)]
    f = {(0, 1): Fr(1)}
    for e, c in phi: f[(e, 0)] = f.get((e, 0), Fr(0)) - c
    return DG.clean(f)
def rphi(n):
    es = sorted(random.sample(range(1, 6), n)); return [(e, Fr(random.randint(1, 4), random.randint(1, 3)) * (1 if i == 0 else random.choice([1, -1]))) for i, e in enumerate(es)]
def path_to_arc(path):
    """nested (w, c*) pairs to an arc y = sum c_i x^{e_i}"""
    arc = []; scale = Fr(1); shift = Fr(0)
    for w, c in path:
        e = shift + w * scale; arc.append((e, c)); shift = e; scale = scale / Fr(w).denominator
    return arc

# DG1
ok = True; n = 0
for _ in range(300):
    cells = {}
    for l in range(random.randint(2, 3)):
        f = sq(root_poly(rphi(random.randint(1, 2))))
        if random.random() < 0.5: f = DG._pmul(f, {(random.randint(0, 2), random.randint(0, 1)): Fr(1)})
        cells[l] = f
    cells[9] = {(random.randint(3, 8), 0): Fr(1)}
    arcs = [[(Fr(random.randint(1, 4), random.randint(1, 2)), Fr(random.randint(1, 3)))]]
    for path, m in DG.tree(cells, 4):
        a = path_to_arc(path); arcs.append(a); arcs.append(a + [(a[-1][0] + Fr(random.randint(1, 3), 2), Fr(random.choice([1, -2, 3])))])
    for a in arcs:
        if a[0][1] <= 0: continue
        d, r = DG.arc_direct(cells, a), DG.arc_recursive(cells, a); ok &= d == r; n += 1
check("DG1 the recursion is exact: recursive limit = direct expansion along random generic and degenerate arcs", ok, f"{n} arcs")
OUT["DG1_arcs"] = n

# DG2
ok = True; partial = 0
for _ in range(300):
    phi = rphi(1); cells = {1: sq(root_poly(phi)), 2: DG._pmul({(0, 1): Fr(1)}, {(phi[0][0], 0): Fr(random.randint(1, 3))}), 3: {(7, 0): Fr(1)}}
    w = Fr(phi[0][0]); c = phi[0][1]
    if c <= 0: continue
    P = DG.total(cells); Gc = sum(v * c ** b for b, v in DG.edge_poly(P, w).items())
    mu = DG.weight_min(P, w)
    lead = {k: sum(v * c ** b for b, v in DG.edge_poly(f, w).items()) for k, f in cells.items() if DG.weight_min(f, w) == mu}
    allzero = all(v == 0 for v in lead.values()); ok &= allzero == (Gc == 0)
    if not allzero and any(v == 0 for v in lead.values()):
        partial += 1; a = [(w, c), (w + 1, Fr(1))]; ok &= DG.arc_direct(cells, a) == DG.arc_recursive(cells, a) and all(lead[k] != 0 for k in DG.arc_direct(cells, a))
check("DG2 refine iff all leading cells vanish iff P_A's edge polynomial does; partially vanishing cells drop out", ok, f"{partial} partial cases")

# DG3
cells = {1: sq(root_poly([(1, Fr(1)), (2, Fr(1))])), 2: {(5, 0): Fr(1)}}
t = DG.tree(cells, 6); ok = [m for _, m in t] == [2, 2] and [p[-1] for p, _ in t] == [(Fr(1), Fr(1)), (Fr(1), Fr(1))]
for b in (Fr(1), Fr(2), Fr(1, 3), Fr(-3)):
    a = [(Fr(1), Fr(1)), (Fr(2), Fr(1)), (Fr(5, 2), b)]; ok &= DG.arc_direct(cells, a)[1] == b * b / (1 + b * b) == DG.arc_recursive(cells, a)[1]
ok &= DG.arc_recursive(cells, [(Fr(1), Fr(1)), (Fr(2), Fr(1))]) == {2: Fr(1)}
ok &= DG.arc_direct(cells, [(Fr(1), Fr(1)), (Fr(2), Fr(1)), (Fr(2), Fr(0)), (Fr(9, 4), Fr(1))]) in ({1: Fr(1)},)
ok &= DG.arc_direct(cells, [(Fr(1), Fr(1)), (Fr(2), Fr(1)), (Fr(3), Fr(1))]) == {2: Fr(1)}
check("DG3 worked example: two refinements of multiplicity 2; share b^2/(1+b^2) along y = x + x^2 + b x^(5/2); exact branch 0; earlier exits 1", ok,
      f"tree {[(tuple(map(str, p[-1])), m) for p, m in t]}")

# DG4, DG5
ok4 = True; ok5 = True; nodes = 0; arcsn = 0; maxdepth = 0
for _ in range(200):
    base = [(e, Fr(random.randint(1, 4), random.randint(1, 3)) * (1 if i == 0 else random.choice([1, -1]))) for i, e in enumerate(sorted(random.sample(range(1, 5), 3)))]
    phis = []
    for _ in range(random.randint(2, 3)):                  # roots sharing a prefix of a common base root, then branching off
        j = random.randint(1, 3); p = base[:j]
        p = p + [(p[-1][0] + random.randint(1, 2), Fr(random.choice([1, 2, -1, 3])))]
        phis.append(p)
    cells = {l: sq(root_poly(p)) for l, p in enumerate(phis)}; cells[9] = {(random.randint(10, 16), 0): Fr(1)}
    tr = DG.tree(cells, 7)
    for path, m in tr:
        nodes += 1; a = path_to_arc(path); maxdepth = max(maxdepth, len(path))
        ok4 &= any([(Fr(e), c) for e, c in p[:len(a)]] == a for p in phis)
    chains = {}
    for path, m in tr: chains[path] = m
    for path, m in tr:
        if len(path) > 1: ok5 &= m <= chains[path[:-1]]
    ok5 &= all(len(p) < 7 for p, _ in tr)
    for p in phis:
        k = random.randint(1, len(p)); a = [(Fr(e), c) for e, c in p[:k]]
        if a[0][1] <= 0: continue
        a2 = a + [(a[-1][0] + Fr(random.randint(1, 4), 2), Fr(random.choice([1, 2, -1])))]
        for arc in (a, a2): ok4 &= DG.arc_direct(cells, arc) == DG.arc_recursive(cells, arc); arcsn += 1
check("DG4 tree nodes are truncations of the cells' roots; recursion = direct expansion along arcs following each root then leaving", ok4, f"{nodes} nodes, {arcsn} arcs")
check("DG5 termination: multiplicity non-increasing along chains; every chain finite (max depth seen {})".format(maxdepth), ok5)

# DG6
ok = True; rows = []; series = [Fr(1), Fr(1, 2), Fr(-1, 8), Fr(1, 16), Fr(-5, 128), Fr(7, 256), Fr(-21, 1024)]
g = {(0, 2): Fr(1), (2, 0): Fr(-1), (3, 0): Fr(-1)}
for K in (8, 12, 16):
    cells = {1: sq(g), 2: {(K, 0): Fr(1)}}; tr = DG.tree(cells, 10); depth = max(len(p) for p, _ in tr)
    arc = path_to_arc(max(tr, key=lambda pm: len(pm[0]))[0]); coeffs = [c for _, c in arc]
    ok &= depth == (K - 2) // 2 - 1 and coeffs == series[:depth] and all(m == 2 for _, m in tr); rows.append((K, depth, [str(c) for c in coeffs]))
    e = Fr((K - 2), 2); ae = series[int(e) - 1]
    for c in (Fr(0), Fr(1), Fr(-1, 2)):
        full = [(Fr(i + 1), series[i]) for i in range(int(e) - 1)] + [(e, c)]
        ok &= DG.arc_direct(cells, full)[1] == 4 * (c - ae) ** 2 / (4 * (c - ae) ** 2 + 1)
trinf = DG.tree({1: sq(g)}, 6); ok &= max(len(p) for p, _ in trinf) == 6 and all(m == 2 for _, m in trinf)
check("DG6 an infinite root x sqrt(1+x) is followed for (K-2)/2 - 1 refinements before x^K cuts it; without x^K it runs to the cap", ok, f"{rows}")
OUT["DG6"] = rows
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "degen.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
