"""check_coxeter.py -- contextuality and formal smoothness of the observable algebra (note: level2/partial_gluing.pdf, Section 15).

The observables of a compatibility graph G generate the right-angled Coxeter group W_G. Group theory: G chordal <=> W_G virtually free
<=> Q[W_G] formally smooth (so every Rep_n is smooth). Contextuality theory: quantum contextuality is possible iff G is not chordal. The
checks test the matrix side of this correspondence.

  CX1 chordal graphs (P4, star, paw, diamond): no direct sum of characters in Rep_2 or Rep_3 has an obstructed tangent vector
  CX2 chordless cycles C4, C5, C6: no obstructed direct sum of characters in Rep_2, but obstructed points in Rep_3; for C4 an exact rational certificate (sympy): a tangent
      vector that does not extend to second order, next to one that does (the extendable vectors form a cone, not a space)
  CX3 for C4 = K_{2,2} (W = D_inf x D_inf) the obstructed points of Rep_3 are exactly those predicted by the Kunneth formula: coordinates
      i, j, k with H^1(chi_i chi_k) from the first factor, H^1(chi_k chi_j) from the second, and their cup product landing in
      H^2(chi_i chi_j) = H^1 (x) H^1; and dim H^1(W, End V) = tangent - orbit dimension at every point
  CX4 every graph on 4 and 5 vertices (all 45 isomorphism classes): Rep_3 has an obstructed diagonal point iff the graph is not chordal
  CX5 the quantum side (NPA level 2): on every chordal graph on 4 and 5 vertices the quantum maximum of random correlator functionals
      equals the classical one; on every non-chordal one an induced chordless k-cycle
      reaches the quantum n-cycle bound (k cos(pi/k) for even k, (3k cos(pi/k) - k)/(1 + cos(pi/k)) for odd k; Araujo et al.) > k - 2
  CX6 exact rational certificates (sympy) of a singular point of Rep_3 for C5 and C6 as well; with the retraction W_G -> W_C onto an induced
      cycle (other generators -> 1), Rep_n(W_C) is a retract of Rep_n(W_G), and a retract of a smooth scheme is smooth, so every graph with a
      chordless cycle of length 4, 5 or 6 has a singular Rep_3; also checked directly by extending the C4 point trivially to the 5-vertex graphs
Writes ../level2/coxeter_results.json.  Run: python check_coxeter.py  (about 8 minutes)
"""
import itertools, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import networkx as nx
import sympy as sp
from pce.coxeter import analyse, diagonal_points, linearisation, exact_obstruction, is_chordal
from pce.cbdfrac import edge_npa_value

rng = np.random.default_rng(20261025); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def G_of(nxg):
    nodes = sorted(nxg.nodes()); ix = {v: i for i, v in enumerate(nodes)}
    return ([str(v) for v in nodes], [(ix[u], ix[v]) for u, v in nxg.edges()])
def scan(G, n):
    pts = obs = 0
    for X, key in diagonal_points(G, n):
        pts += 1; obs += analyse(G, X, 2, rng)["obstructed"]
    return pts, obs
named = {"P4": nx.path_graph(4), "star": nx.star_graph(3), "paw": nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)]),
         "diamond": nx.Graph([(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)]), "C4": nx.cycle_graph(4), "C5": nx.cycle_graph(5), "C6": nx.cycle_graph(6)}

# CX1
rows = {}
for nm in ("P4", "star", "paw", "diamond"):
    G = G_of(named[nm]); rows[nm] = [scan(G, 2), scan(G, 3)]
ok = all(r[0][1] == 0 and r[1][1] == 0 for r in rows.values()); OUT["chordal"] = rows
check("CX1 chordal graphs: no obstructed point in Rep_2 or Rep_3", ok, "; ".join(f"{k}: {v[0][0]}+{v[1][0]} points" for k, v in rows.items()))

# CX2
rows = {}
for nm in ("C4", "C5", "C6"):
    G = G_of(named[nm]); rows[nm] = [scan(G, 2), scan(G, 3)]
ok = all(r[0][1] == 0 and r[1][1] > 0 for r in rows.values())
G4 = G_of(named["C4"]); cert = None
for X, key in diagonal_points(G4, 3):
    if analyse(G4, X, 2, rng)["obstructed"]:
        L = sp.Matrix(np.rint(linearisation(G4, X)).astype(int)); NS = L.nullspace(); res = []
        for coeffs in ([1] * len(NS), list(range(1, len(NS) + 1))):
            y = sum((c * v for c, v in zip(coeffs, NS)), sp.zeros(L.shape[1], 1))
            res.append(exact_obstruction(G4, X, [np.array(y[i * 9:(i + 1) * 9]).reshape(3, 3) for i in range(4)]))
        cert = {"point": [list(map(int, c)) for c in key], "extends": [not r for r in res]}; break
ok &= cert is not None and cert["extends"] == [True, False]; OUT["cycles"] = rows; OUT["certificate"] = cert
check("CX2 chordless cycles: no obstructed direct sum of characters at n = 2, obstructed points at n = 3; exact certificate for C4", ok,
      "; ".join(f"{k}: {v[1][1]} of {v[1][0]} points obstructed at n=3" for k, v in rows.items()) + "; C4 certificate: one vector extends, one does not")

# CX3
def h(s): return (1 if s == (1, 1) else 0), (1 if s == (-1, -1) else 0)
def kun(ch):
    h0A, h1A = h((ch[0], ch[2])); h0B, h1B = h((ch[1], ch[3])); return h1A * h0B + h0A * h1B, h1A * h1B
def pred(key):
    cols = [np.array(c) for c in key]
    for i, j, k in itertools.permutations(range(len(cols)), 3):
        a = tuple(cols[i] * cols[k]); b = tuple(cols[k] * cols[j]); ab = tuple(cols[i] * cols[j])
        if h((a[0], a[2]))[1] and h((a[1], a[3]))[0] and h((b[0], b[2]))[0] and h((b[1], b[3]))[1] and kun(ab)[1]: return True
    return False
agree = tot = 0; h1ok = True
for X, key in diagonal_points(G4, 3):
    a = analyse(G4, X, 2, rng); tot += 1; agree += a["obstructed"] == pred(key)
    h1ok &= sum(kun(tuple(np.array(ci) * np.array(cj)))[0] for ci in key for cj in key) == a["H1"]
check("CX3 C4: obstructed points of Rep_3 = Kunneth cup-product prediction; dim H^1 = tangent - orbit", agree == tot and h1ok,
      f"agreement at {agree} of {tot} points")

# CX4, CX5
classes = [g for g in nx.graph_atlas_g() if g.number_of_nodes() in (4, 5)]
ok4 = True; ok5 = True; table = []
for g in classes:
    G = G_of(g); ch = nx.is_chordal(g); assert ch == is_chordal(G)
    pts, obs = scan(G, 3); ok4 &= (obs > 0) == (not ch)
    E = [(u, v, None) for u, v in G[1]]; gap = None
    if E:
        if ch:
            gaps = []
            for _ in range(6):
                c = rng.normal(size=len(E)); q = edge_npa_value(len(G[0]), E, c, 2)
                cl = max(sum(ci * s[u] * s[v] for ci, (u, v, _) in zip(c, E)) for s in itertools.product((1, -1), repeat=len(G[0])))
                gaps.append(q - cl)
            gap = max(gaps); ok5 &= gap < 1e-5
        else:
            cyc = [cy for cy in nx.chordless_cycles(g) if len(cy) >= 4][0]; k = len(cyc)
            ce = {(min(cyc[i], cyc[(i + 1) % k]), max(cyc[i], cyc[(i + 1) % k])) for i in range(k)}
            c = [(1.0 if (min(u, v), max(u, v)) in ce else 0.0) for u, v, _ in E]
            first = [i for i, (u, v, _) in enumerate(E) if c[i] > 0][0]; c[first] = -1.0
            qb = k * np.cos(np.pi / k) if k % 2 == 0 else (3 * k * np.cos(np.pi / k) - k) / (1 + np.cos(np.pi / k))   # Araujo et al.
            gap = edge_npa_value(len(G[0]), E, c, 2) - (k - 2); ok5 &= abs(gap - (qb - (k - 2))) < 1e-5
    table.append({"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "chordal": ch, "points": pts, "obstructed": obs, "quantum_gap": gap})
OUT["all_graphs"] = table
nc = sum(not t["chordal"] for t in table)
check("CX4 all graphs on 4 and 5 vertices: an obstructed point in Rep_3 iff not chordal", ok4, f"{len(table)} classes, {nc} non-chordal")
check("CX5 NPA: chordal -> quantum = classical on random correlator functionals; non-chordal -> the n-cycle quantum bound minus k - 2", ok5,
      f"largest chordal gap {max(t['quantum_gap'] for t in table if t['chordal'] and t['quantum_gap'] is not None):.1e}")

# CX6
def certificate(G, n=3):
    for X, key in diagonal_points(G, n):
        if analyse(G, X, 2, rng)["obstructed"]:
            L = sp.Matrix(np.rint(linearisation(G, X)).astype(int)); NS = L.nullspace(); N = len(G[0])
            for coeffs in (list(range(1, len(NS) + 1)), [1] * len(NS), [(-1) ** i * (i + 2) for i in range(len(NS))]):
                y = sum((c * v for c, v in zip(coeffs, NS)), sp.zeros(L.shape[1], 1))
                if exact_obstruction(G, X, [np.array(y[i * n * n:(i + 1) * n * n]).reshape(n, n) for i in range(N)]): return [list(map(int, c)) for c in key]
    return None
certs = {nm: certificate(G_of(named[nm])) for nm in ("C5", "C6")}
ext_ok = True
for g in classes:
    if g.number_of_nodes() == 5 and not nx.is_chordal(g) and any(len(cy) == 4 for cy in nx.chordless_cycles(g)):
        cyc = [cy for cy in nx.chordless_cycles(g) if len(cy) == 4][0]; G = G_of(g); nodes = sorted(g.nodes())
        # the C4 point on the cycle, identity on the other vertex
        key = cert["point"]; Xc = [np.diag([key[j][i] for j in range(3)]).astype(float) for i in range(4)]
        X = [np.eye(3) for _ in nodes]
        for pos, v in enumerate(cyc): X[nodes.index(v)] = Xc[pos]
        ext_ok &= analyse(G, X, 3, rng)["obstructed"]
OUT["certificates_C5_C6"] = certs
check("CX6 exact certificates for C5 and C6; the C4 singular point extended trivially stays singular in every 5-vertex graph containing an induced C4",
      all(v is not None for v in certs.values()) and ext_ok, ", ".join(f"{k}: certified" for k, v in certs.items() if v))

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "coxeter_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
