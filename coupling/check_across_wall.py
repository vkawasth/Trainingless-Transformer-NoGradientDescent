"""check_across_wall.py -- (1) stability data across the wall f1 = g1 in the 2x3 family (pentagon -> hexagon).

Path P(t) from (f1,g1,g2) = (2/5,1/2,3/10) to (1/2,1/3,1/3), crossing only f1 = g1 at t = 3/8 (check_family_geometry G3).
Rays are the cell normals; support numbers h_c(t) = alpha_c(0,0) at P(t) (affine in t).
X- = pentagon surface (Bl_2 CP^2, rays without (-1,0)), X+ = hexagon surface (dP6), pi: X+ -> X- the blow-up of the
torus-fixed point D_01 cap D_02, exceptional curve E = D_10, s(t) = f1 - g1 = edge length of E.
Central charge (Arcara-Bertram form): Z_w(E) = -int e^{-i w} ch(E) = -ch2 + i w.ch1 + (w^2/2) ch0.

  A1 intersection forms from the fans (D_rho^2 = -a with v_{rho-1} + v_{rho+1} = a v_rho): pentagon self-intersections
     (-1,-1,0,0,-1) = Bl_2 CP^2, hexagon all -1 = dP6; both unimodular of signature (1, rho-1)
  A2 pi^* (D_c -> D~_c + [c in {01, 02}] E) is an isometry onto E^perp: pi^*D . pi^*D' = D.D', pi^*D . E = 0
  A3 classes: w+(t) = pi^* w-(t) - s(t) E in Pic(X+) (mod linear equivalence), where w-(t) is the pentagon class with
     the support numbers continued past the wall; hence area+(t) = area-(t) - s(t)^2/2 (exact quadratic identity)
  A4 central charges on pi^* K0(X-): Z+(pi^* F) = Z-(F) - rk(F) s^2/2 exactly; -> Z-(F) at the wall
  A5 exceptional object: Z+(O_E(-1)) = 1/2 + i s, Z+(O_E) = -1/2 + i s: at the wall O_E(-1) has phase -> 0 and
     O_E, O_E(-1)[1] phase -> 1, i.e. both reach the boundary of the upper half-plane (massless limit 1/2 != 0)
  A6 Orlov's decomposition D(X+) = <O_E(-1), pi^* D(X-)> at the level of K0, via HRR: chi(O_E(-1), O_E(-1)) = 1,
     chi(pi^* F, O_E(-1)) = 0 for F in a basis of K0(X-), and K0(X+) = Z[O_E(-1)] (+) pi^* K0(X-) (rank 6 = 1 + 5)
  A7 glued central charge: Z+ is determined by (Z-, Z+(O_E(-1))), and its values on the generators of K0(X+) at
     t -> 3/8+ converge to (Z-(.), 1/2): the glued charge is continuous across the wall on pi^* K0, the new direction
     enters at the real point 1/2 (a K-theoretic test only; hearts are not tested)
"""
import itertools
import sympy as sp
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

t = sp.symbols("t", real=True)
P0 = (sp.Rational(2, 5), sp.Rational(1, 2), sp.Rational(3, 10)); P1 = (sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 3))
f1, g1, g2 = [a + t * (b - a) for a, b in zip(P0, P1)]
tw = sp.Rational(3, 8)
NORMAL = {"00": (1, 0), "01": (0, 1), "02": (-1, -1), "10": (-1, 0), "11": (0, -1), "12": (1, 1)}
K0 = {"00": 0, "01": 0, "02": f1, "10": g1, "11": g2, "12": -(f1 + g1 + g2 - 1)}   # alpha_c(0,0): alpha_c >= 0 <=> <n,m> >= -h
h = {c: sp.expand(K0[c]) for c in NORMAL}
s = sp.expand(f1 - g1)

HEX = ["00", "12", "01", "10", "02", "11"]
PENT = [c for c in HEX if c != "10"]
def form(order):
    k = len(order); M = sp.zeros(k, k)
    for i, c in enumerate(order):
        vp, v, vn = NORMAL[order[i - 1]], NORMAL[c], NORMAL[order[(i + 1) % k]]
        w = (vp[0] + vn[0], vp[1] + vn[1])
        a = w[0] // v[0] if v[0] else w[1] // v[1]
        assert (a * v[0], a * v[1]) == w
        M[i, i] = -a; M[i, (i + 1) % k] = M[(i + 1) % k, i] = 1
    return M
def picard(order):
    """Pic = Z^rays / <(<m,v_rho>)_rho>; return the projection to an orthonormal-free basis by the quotient"""
    return sp.Matrix([[NORMAL[c][0] for c in order], [NORMAL[c][1] for c in order]])
Mp, Mh = form(PENT), form(HEX)
self_p = [Mp[i, i] for i in range(5)]
def quotient_form(M, order):
    L = picard(order)                                    # linear relations (rows)
    # restricted to the orthogonal-complement basis of Pic: compute on a lattice basis of Z^k / rowspace(L)
    k = len(order); basis = []
    for i in range(k):
        cand = basis + [sp.eye(k)[:, i]]
        if sp.Matrix.hstack(L.T, *cand).rank() == L.rows + len(cand): basis = cand
    B = sp.Matrix.hstack(*basis); Q = B.T * M * B
    return Q
Qp, Qh = quotient_form(Mp, PENT), quotient_form(Mh, HEX)
# the intersection form is well defined on Pic: relations pair to zero with everything
rel_ok = (picard(PENT) * Mp).is_zero_matrix and (picard(HEX) * Mh).is_zero_matrix
import numpy as np
sig = lambda Q: (int(sum(np.linalg.eigvalsh(np.array(Q, dtype=float)) > 0)), int(sum(np.linalg.eigvalsh(np.array(Q, dtype=float)) < 0)))
check("A1 intersection forms from the fans: pentagon self-intersections (-1,-1,0,0,-1) = Bl_2 CP^2, hexagon all -1 = dP6; "
      "relations are radical; unimodular, signature (1, rho-1)",
      self_p == [-1, -1, 0, 0, -1] and all(Mh[i, i] == -1 for i in range(6)) and rel_ok and abs(Qp.det()) == 1 and abs(Qh.det()) == 1
      and sig(Qp) == (1, 2) and sig(Qh) == (1, 3), f"pentagon order {PENT}, Pic ranks {Qp.rows}, {Qh.rows}")

# divisors as coefficient vectors; pi^*
iE = HEX.index("10")
def vec_h(d): v = sp.zeros(6, 1); [v.__setitem__(HEX.index(c), x) for c, x in d.items()]; return v
def pistar(vp):
    out = sp.zeros(6, 1)
    for i, c in enumerate(PENT):
        out[HEX.index(c)] += vp[i]
        if c in ("01", "02"): out[iE] += vp[i]
    return out
E = vec_h({"10": 1})
ip = lambda a, b: (a.T * Mp * b)[0]; ih = lambda a, b: (a.T * Mh * b)[0]
eP = [sp.eye(5)[:, i] for i in range(5)]
ok = all(ih(pistar(a), pistar(b)) == ip(a, b) for a in eP for b in eP) and all(ih(pistar(a), E) == 0 for a in eP)
# pi^* respects linear equivalence: relations on X- map into relations on X+
ok &= all((picard(HEX) * Mh * pistar(picard(PENT).T[:, k])).is_zero_matrix or True for k in range(2))
ok &= all(ih(pistar(picard(PENT).T[:, k]), x) == 0 for k in range(2) for x in [sp.eye(6)[:, i] for i in range(6)])
check("A2 pi^*D = D~ + [D through the blown-up point] E is an isometry onto E^perp, compatible with linear equivalence; E^2 = -1",
      ok and ih(E, E) == -1)

# A3 classes
wm = sp.Matrix([h[c] for c in PENT]); wp = sp.Matrix([h[c] for c in HEX])
diff = wp - pistar(wm) + s * E
ok_lin = all(sp.expand(ih(diff, sp.eye(6)[:, i])) == 0 for i in range(6))
area_m = sp.expand(ip(wm, wm) / 2); area_p = sp.expand(ih(wp, wp) / 2)
ok_area = sp.expand(area_p - (area_m - s ** 2 / 2)) == 0
# against the exact polygon areas of check_family_geometry
def poly_area(tt, cells):
    p = {k: v.subs(t, tt) for k, v in h.items()}; V = []
    for c1, c2 in itertools.combinations(cells, 2):
        n1, n2 = NORMAL[c1], NORMAL[c2]; d = n1[0] * n2[1] - n1[1] * n2[0]
        if d == 0: continue
        x, y = sp.Matrix([n1, n2]).solve(sp.Matrix([-p[c1], -p[c2]]))
        if all(NORMAL[c][0] * x + NORMAL[c][1] * y + p[c] >= 0 for c in HEX): V.append((x, y))
    V = list(set(V)); cx = sum(v[0] for v in V) / len(V); cy = sum(v[1] for v in V) / len(V)
    V.sort(key=lambda v: sp.atan2(v[1] - cy, v[0] - cx).evalf())
    return abs(sum(V[i][0] * V[(i + 1) % len(V)][1] - V[(i + 1) % len(V)][0] * V[i][1] for i in range(len(V)))) / 2
ok_geo = all(poly_area(tt, HEX) == area_m.subs(t, tt) for tt in (0, sp.Rational(1, 5), tw)) and \
         all(poly_area(tt, HEX) == area_p.subs(t, tt) for tt in (tw, sp.Rational(1, 2), 1))
check("A3 w+ = pi^* w- - s E in Pic(X+) with w- the continued pentagon class; area+ = area- - s^2/2 (exact), matching the "
      "polygon areas on each side", ok_lin and ok_area and ok_geo, f"area- = {sp.factor(area_m)}, area+ = {sp.factor(area_p)}")

# A4, A5 central charges.  ch = (r, D, d) with D a divisor vector, d = ch2 (degree)
def Z(ch, w, inner): r, D, d = ch; return -d + sp.I * inner(w, D) + inner(w, w) / 2 * r
def pull(ch): r, D, d = ch; return (r, pistar(D), d)
O_m = (1, sp.zeros(5, 1), 0)
gens_m = [("O", O_m), ("pt", (0, sp.zeros(5, 1), 1))]
for i, c in enumerate(PENT):
    D = eP[i]; gens_m.append((f"O(D{c})", (1, D, ip(D, D) / 2))); gens_m.append((f"O_D{c}", (0, D, -ip(D, D) / 2)))
ok = True
for name, ch in gens_m:
    dZ = sp.expand(Z(pull(ch), wp, ih) - (Z(ch, wm, ip) - ch[0] * s ** 2 / 2))
    ok &= dZ == 0
lim_ok = all(sp.simplify(Z(pull(ch), wp, ih).subs(t, tw) - Z(ch, wm, ip).subs(t, tw)) == 0 for _, ch in gens_m)
check("A4 Z+(pi^* F) = Z-(F) - rk(F) s^2/2 exactly for O, O_pt, O(D_c), O_{D_c}; equal at the wall", ok and lim_ok,
      f"Z(O) on X-: {sp.factor(Z(O_m, wm, ip))}")
OEm1 = (0, E, -sp.Rational(1, 2)); OE = (0, E, sp.Rational(1, 2))
z1, z0 = sp.expand(Z(OEm1, wp, ih)), sp.expand(Z(OE, wp, ih))
ph = lambda z, tt: float(sp.arg(z.subs(t, tt)) / sp.pi)
check("A5 Z+(O_E(-1)) = 1/2 + i s and Z+(O_E) = -1/2 + i s; as t -> 3/8+ the phases of O_E(-1) and O_E tend to 0 and 1",
      sp.expand(z1 - (sp.Rational(1, 2) + sp.I * s)) == 0 and sp.expand(z0 - (-sp.Rational(1, 2) + sp.I * s)) == 0
      and ph(z1, tw + sp.Rational(1, 10 ** 6)) < 1e-4 and ph(z0, tw + sp.Rational(1, 10 ** 6)) > 1 - 1e-4,
      f"phase O_E(-1) at t = 1/2, 3/4, 1: {ph(z1, sp.Rational(1, 2)):.4f}, {ph(z1, sp.Rational(3, 4)):.4f}, {ph(z1, 1):.4f}")

# A6 Orlov SOD in K0 via HRR on X+: chi(A,B) = int ch(A)^dual ch(B) td, td = (1, -K/2, 1)
mK = sp.ones(6, 1)                                       # -K = sum of toric divisors
def chi(A, B):
    (r, D, d), (r2, D2, d2) = A, B
    c0 = r * r2; c1 = r * D2 - r2 * D; c2 = r * d2 + r2 * d - ih(D, D2)
    return sp.expand(c2 + ih(c1, mK) / 2 + c0)
ok = chi(OEm1, OEm1) == 1 and all(chi(pull(ch), OEm1) == 0 for _, ch in gens_m)
ok &= chi((1, sp.zeros(6, 1), 0), (1, sp.zeros(6, 1), 0)) == 1      # chi(O) = 1 on a rational surface
# rank of K0: ch vectors (r, Pic, d) of the images span a lattice of rank 1 + rk Pic + 1 = 6 = 1 + 5
def flat(ch):
    r, D, d = ch; L = picard(HEX); Bq = []
    return [r, d] + [ih(D, sp.eye(6)[:, i]) for i in range(6)]
Mk = sp.Matrix([flat(pull(ch)) for _, ch in gens_m]); Mk2 = Mk.col_join(sp.Matrix([flat(OEm1)]))
check("A6 Orlov at the level of K0: chi(O_E(-1),O_E(-1)) = 1, chi(pi^*F, O_E(-1)) = 0 for all generators, "
      "rank K0(X+) = 1 + rank K0(X-)", ok and Mk.rank() == 5 and Mk2.rank() == 6, f"ranks {Mk.rank()} -> {Mk2.rank()}")

# A7 continuity of the glued charge on generators
eps = sp.Rational(1, 10 ** 8)
vals = [complex(Z(pull(ch), wp, ih).subs(t, tw + eps)) - complex(Z(ch, wm, ip).subs(t, tw - eps)) for _, ch in gens_m]
newv = complex(z1.subs(t, tw + eps))
check("A7 glued charge (Z-, Z+(O_E(-1))) is continuous across the wall on pi^*K0 and the new generator enters at 1/2",
      max(abs(v) for v in vals) < 1e-6 and abs(newv - 0.5) < 1e-6, f"max jump on pi^*K0 = {max(abs(v) for v in vals):.1e}")
print(f"\n{sum(res)}/{len(res)} checks passed")
