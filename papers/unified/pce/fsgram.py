"""The categorical layer per chamber (section sec:fsgram): toric surfaces of 2 x 3 coupling polygons, line-bundle cohomology, exceptional
collections, their Gram (Euler) matrices and mutations. New module; it modifies nothing.
  cpl23_facets(f, g)           the cells whose inequality is a facet of the 2 x 3 coupling polygon (fibre coordinates u in Z^2)
  NORMALS                      the facet normals n_ij in the fibre lattice: x_ij = x0_ij + <n_ij, u>
  surface(rays)                the toric surface of a complete fan in Z^2: rays in cyclic order, self-intersections, canonical class
  cohomology(S, a)             h^0, h^1, h^2 of O(sum a_i D_i), by the lattice-point formula (H^i = sum over m of reduced cohomology of
                               the subcomplex of the fan on the rays with <m, v_i> < -a_i)
  chi(S, a)                    Riemann-Roch: 1 + (D.D - K.D) / 2
  toric_collection(S)          O, O(D_1), O(D_1 + D_2), ..., O(D_1 + ... + D_{m-1}) (the toric system A_i = D_i of Hille and Perling)
  gram(S, coll)                chi(E_i, E_j) = chi(O(L_j - L_i))
  mutate(G, i)                 the Gram matrix after the left mutation of the pair (i, i+1): (E_i, E_i+1) -> (L_{E_i} E_i+1, E_i)
  invariants(G)                isometry invariants of the bilinear lattice: characteristic polynomial of the Coxeter map -G^-1 G^T, Smith
                               normal forms of G + G^T and G - G^T
"""
from __future__ import annotations
import itertools, math
import numpy as np
import sympy as sp
NORMALS = {(0, 0): (1, 0), (0, 1): (-1, 1), (0, 2): (0, -1), (1, 0): (-1, 0), (1, 1): (1, -1), (1, 2): (0, 1)}
def _x0(f, g):
    """a base point of the coupling polytope (north-west corner rule; any point of the affine fibre works)."""
    f = list(f); g = list(g); x = np.zeros((2, 3)); i = j = 0
    while i < 2 and j < 3:
        t = min(f[i], g[j]); x[i, j] = t; f[i] -= t; g[j] -= t
        if f[i] <= 1e-15: i += 1
        else: j += 1
    return x
def cpl23_facets(f, g, tol=1e-12):
    """cells whose constraint <n, u> >= -x0 bounds an edge of positive length of the polygon."""
    x0 = _x0(f, g); cons = [(np.array(NORMALS[c], float), -x0[c]) for c in NORMALS]
    verts = []
    for (n1, b1), (n2, b2) in itertools.combinations(cons, 2):
        A = np.array([n1, n2])
        if abs(np.linalg.det(A)) < 1e-12: continue
        u = np.linalg.solve(A, [b1, b2])
        if all(n @ u >= b - 1e-10 for n, b in cons): verts.append(u)
    out = []
    for c, (n, b) in zip(NORMALS, cons):
        on = [u for u in verts if abs(n @ u - b) < 1e-9]
        if len(on) >= 2 and max(np.linalg.norm(p - q) for p in on for q in on) > tol: out.append(c)
    return out
def surface(rays):
    rays = sorted([tuple(r) for r in rays], key=lambda v: math.atan2(v[1], v[0])); m = len(rays)
    b = []
    for i in range(m):
        p, c, n = np.array(rays[i - 1]), np.array(rays[i]), np.array(rays[(i + 1) % m]); s = p + n
        k = s[0] / c[0] if c[0] != 0 else s[1] / c[1]
        assert np.allclose(s, k * c), "fan is not smooth"; b.append(int(round(k)))
    I = np.zeros((m, m), int)
    for i in range(m):
        I[i, i] = -b[i]; I[i, (i + 1) % m] += 1; I[(i + 1) % m, i] += 1
    if m == 3: I = np.ones((3, 3), int)                                        # P^2: every pair of lines meets once
    smooth = all(abs(rays[i][0] * rays[(i + 1) % m][1] - rays[i][1] * rays[(i + 1) % m][0]) == 1 for i in range(m))
    return dict(rays=rays, m=m, I=I, K=-np.ones(m, int), smooth=smooth)
def chi(S, a):
    a = np.asarray(a); I = S["I"]; K = S["K"]; return int(1 + (a @ I @ a - K @ I @ a) // 2) if (a @ I @ a - K @ I @ a) % 2 == 0 else None
def cohomology(S, a):
    rays = S["rays"]; m = S["m"]; a = np.asarray(a); R = 2 * int(np.abs(a).sum()) + 3; h = [0, 0, 0]
    for m1 in range(-R, R + 1):
        for m2 in range(-R, R + 1):
            neg = [(m1 * v[0] + m2 * v[1]) < -a[i] for i, v in enumerate(rays)]
            k = sum(neg)
            if k == 0: h[0] += 1
            elif k == m: h[2] += 1
            else:
                arcs = sum(1 for i in range(m) if neg[i] and not neg[i - 1])  # maximal runs on the cycle
                h[1] += arcs - 1
    return h
def toric_collection(S):
    m = S["m"]; out = [np.zeros(m, int)]
    for k in range(1, m):
        d = out[-1].copy(); d[k - 1] += 1; out.append(d)
    return out
def gram(S, coll):
    n = len(coll); return np.array([[chi(S, coll[j] - coll[i]) for j in range(n)] for i in range(n)], int)
def mutate(G, i):
    """left mutation at (i, i+1): new classes e'_i = e_i+1 - G[i, i+1] e_i, e'_i+1 = e_i (classes in the old basis; Gram = B G B^T).
    Exact Python integers (entries can grow quickly under repeated mutation)."""
    G = [[int(x) for x in row] for row in G]; n = len(G); c = G[i][i + 1]
    B = [[1 if r == k else 0 for k in range(n)] for r in range(n)]
    B[i] = [0] * n; B[i][i + 1] = 1; B[i][i] = -c; B[i + 1] = [0] * n; B[i + 1][i] = 1
    BG = [[sum(B[r][k] * G[k][q] for k in range(n) if B[r][k]) for q in range(n)] for r in range(n)]
    return [[sum(BG[r][k] * B[q][k] for k in range(n) if B[q][k]) for q in range(n)] for r in range(n)]
def invariants(G):
    from sympy.matrices.normalforms import smith_normal_form
    Gm = sp.Matrix(G); x = sp.symbols("x")
    cox = -(Gm.inv() * Gm.T); cp = sp.Poly(cox.charpoly(x).as_expr(), x).all_coeffs()
    s = smith_normal_form(Gm + Gm.T, domain=sp.ZZ); t = smith_normal_form(Gm - Gm.T, domain=sp.ZZ)
    return dict(coxeter_charpoly=tuple(int(c) for c in cp), snf_sym=tuple(abs(int(s[i, i])) for i in range(len(G))),
                snf_anti=tuple(abs(int(t[i, i])) for i in range(len(G))))
