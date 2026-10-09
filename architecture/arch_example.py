"""Architecture note: one 2x3 margin traced through every layer, and the deformation facts.
f = (0.4, 0.6), g = (0.3, 0.43, 0.27)."""
import itertools, math, sys
from fractions import Fraction as Fr
from collections import Counter
import numpy as np, sympy as sp
from scipy.optimize import linprog
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bridges', 'fukaya'))
from newton23 import exact_positions
from sweep23 import facets_of, self_int, surface, leximin, classify, orientable, NU, lam, wallvals, distinct_roots
f1 = Fr(2, 5); g = (Fr(3, 10), Fr(43, 100), Fr(27, 100)); f = (f1, 1 - f1)
print("== fibre")
cells = [(i, j) for i in range(2) for j in range(3)]
fac = facets_of(f1, g); print("facet cells", [f"{c[0]+1}{c[1]+1}" for c in fac], "| forced lower bounds", {f"{c[0]+1}{c[1]+1}": str(max(Fr(0), f[c[0]] + g[c[1]] - 1)) for c in cells if c not in fac})
# vertices of the polygon in (a11, a12)
x1, x2 = sp.symbols('x1 x2'); verts = set()
for a, b in itertools.combinations(fac, 2):
    s = sp.solve([NU[a][0]*x1 + NU[a][1]*x2 + sp.Rational(lam(a, f1, g)), NU[b][0]*x1 + NU[b][1]*x2 + sp.Rational(lam(b, f1, g))], [x1, x2], dict=True)
    if s and all(NU[c][0]*s[0][x1] + NU[c][1]*s[0][x2] + sp.Rational(lam(c, f1, g)) >= 0 for c in fac): verts.add((s[0][x1], s[0][x2]))
verts = sorted(verts); print("vertices", len(verts), [(str(a), str(b)) for a, b in verts])
area = abs(sum(verts[i][0]*verts[(i+1) % len(verts)][1] - verts[(i+1) % len(verts)][0]*verts[i][1]
               for i in range(len(verts))) / 2) if False else None
hull = sorted(verts, key=lambda v: math.atan2(float(v[1]) - float(sum(p[1] for p in verts))/len(verts), float(v[0]) - float(sum(p[0] for p in verts))/len(verts)))
area = abs(sum(hull[i][0]*hull[(i+1) % len(hull)][1] - hull[(i+1) % len(hull)][0]*hull[i][1] for i in range(len(hull)))) / 2
print("area (symplectic volume)", area)
print("== base: chambers, tear and break margins")
wv = wallvals(f1, g); print("wall values f1 - g(J):", [str(v) for v in wv], "| tear margin m_T =", str(min(abs(v) for v in wv)),
                           "| break margin =", str(min(min(f), min(g))))
print("== entropy")
prod = (f1*g[0], f1*g[1]); P = np.array([[float(f[i]*g[j]) for j in range(3)] for i in range(2)])
print("product (max-entropy) coupling (a11, a12) =", str(prod[0]), str(prod[1]), "| H =", round(float(-(P*np.log(P)).sum()), 4), "nats =",
      round(float(-(P*np.log2(P)).sum()), 4), "bits")
print("== toric")
D2 = self_int(fac); print("surface", surface(fac, D2), "| self-intersections", {f"{c[0]+1}{c[1]+1}": D2[c] for c in fac})
print("Kaehler class coefficients f_i g_j on facets:", {f"{c[0]+1}{c[1]+1}": str(f[c[0]]*g[c[1]]) for c in fac})
print("== Floer")
v1, v2, vp, vq, sols = exact_positions(fac, f1, g); pairs = Counter(list(sols)[0]); ul, lv = leximin(fac, f1, g)
for u, m in sorted(pairs.items(), key=lambda kv: -kv[1]):
    mm, lead, l = classify(fac, f1, g, u)
    print(f"  {m} branes over ({u[0]}, {u[1]}), smallest cells {[f'{c[0]+1}{c[1]+1}' for c in lead]} = {mm}, level {l}")
print("leximin", (str(ul[0]), str(ul[1])), "carries", pairs.get(ul, 0), "| product carries", pairs.get(prod, 0),
      "| smallest facet product", min((str(f[c[0]]*g[c[1]]), f"{c[0]+1}{c[1]+1}") for c in fac), "unique:",
      sorted(f[c[0]]*g[c[1]] for c in fac)[0] < sorted(f[c[0]]*g[c[1]] for c in fac)[1])
print("distinct critical points at T=e^-40:", distinct_roots(fac, f1, g, 40, len(fac), np.random.default_rng(1)))
print("== real locus: chi", 4 - len(fac), "orientable", orientable(fac), "Maslov-1 half-discs from", sum(1 for v in D2.values() if v == -1), "(-1)-curves")
# ---- rigidity of the 2x3 toric surfaces: h^0(T) = 2 + #Demazure roots, chi(T) = 14 - 2n (HRR), h^2(T) = 0, so h^1 = h^0 - chi
print("== rigidity of the 2x3 toric surfaces")
types = {"CP^2": [(0,0),(0,1),(0,2)], "CP^1 x CP^1": [(0,1),(0,2),(1,1),(1,2)], "F_1": [(0,0),(0,1),(0,2),(1,2)],
         "Bl_2": [(0,0),(0,1),(0,2),(1,1),(1,2)], "Bl_3": list(NU)}
for name, F in types.items():
    rays = [np.array(NU[c]) for c in F]; roots = 0
    for m in itertools.product(range(-3, 4), repeat=2):
        vals = [int(np.dot(m, v)) for v in rays]
        if sorted(vals)[0] == -1 and sum(1 for x in vals if x == -1) == 1 and all(x >= 0 for x in vals if x != -1): roots += 1
    n = len(F); h0 = 2 + roots; chiT = 14 - 2*n; D2 = self_int(F)
    ilten = sum(max(0, -d - 1) for d in D2.values())
    print(f"  {name:12s} n={n}: Demazure roots {roots}, h0(T)={h0}, chi(T)=14-2n={chiT}, h1(T)=h0-chi={h0 - chiT}, Ilten sum {ilten}")
# ---- margins -> Kaehler moduli: rank of the linear part of (f1,g1,g2) -> [omega] in H^2 = R^facets / M^*
print("== margins into Kaehler moduli")
F1, G1, G2 = sp.symbols('F1 G1 G2')
for name, F in types.items():
    lamsym = {(0,0): 0, (0,1): 0, (0,2): F1, (1,0): G1, (1,1): G2, (1,2): (1 - G1 - G2) - F1}
    vec = sp.Matrix([lamsym[c] for c in F]); N = sp.Matrix([list(NU[c]) for c in F])
    # quotient by the column space of N (linear functions): project onto its orthogonal complement
    Q = sp.eye(len(F)) - N*(N.T*N).inv()*N.T
    J = (Q*vec).jacobian([F1, G1, G2])
    print(f"  {name:12s} dim H^2 = {len(F) - 2}, rank of margins -> [omega] = {J.rank()}")
# degree-6 del Pezzo: the six normals sum to zero, so the sum of support numbers is c1.[omega]; on that chamber it is identically 1
lamsym = {(0,0): 0, (0,1): 0, (0,2): F1, (1,0): G1, (1,1): G2, (1,2): (1 - G1 - G2) - F1}
print("dP6: sum of normals", tuple(int(x) for x in sum(np.array(NU[c]) for c in NU)), "| c1.[omega] =", sp.simplify(sum(lamsym.values())))
