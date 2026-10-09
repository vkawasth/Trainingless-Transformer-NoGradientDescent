"""Exact valuations of the critical points of W = sum_facets T^{lambda_c} z^{nu_c} (2x3 couplings) by Newton polygons of
resultants over the Puiseux/Novikov field: val(z2) from Res_{z1}, val(z1) from Res_{z2}. Coefficients are polynomials in
a_c = T^{lambda_c}; their valuations are computed exactly with cancellation."""
import itertools, sympy as sp
from fractions import Fraction as Fr
NU = {(0,0): (1,0), (0,1): (0,1), (0,2): (-1,-1), (1,0): (-1,0), (1,1): (0,-1), (1,2): (1,1)}
CELLS = list(NU)
z1, z2 = sp.symbols('z1 z2'); A = sp.symbols('a0:6')
_res = {}
def resultants(facets):
    key = tuple(facets)
    if key in _res: return _res[key]
    W = sum(A[CELLS.index(c)] * z1**NU[c][0] * z2**NU[c][1] for c in facets)
    def poly(e):
        e = sp.expand(e * z1**3 * z2**3); p = sp.Poly(e, z1, z2)
        m1 = min(m[0] for m in p.monoms()); m2 = min(m[1] for m in p.monoms())
        return sp.expand(e / (z1**m1 * z2**m2))
    e1 = poly(z1*sp.diff(W, z1)); e2 = poly(z2*sp.diff(W, z2))
    R2 = sp.Poly(sp.resultant(e1, e2, z1), z2); R1 = sp.Poly(sp.resultant(e1, e2, z2), z1)
    _res[key] = (R1, R2); return R1, R2
def val(coef, lam):
    """valuation of a polynomial in a_c with a_c = T^{lam_c}, exact (with cancellation)."""
    p = sp.Poly(sp.expand(coef), *A); bucket = {}
    for mon, c in zip(p.monoms(), p.coeffs()):
        e = sum(Fr(m) * lam[i] for i, m in enumerate(mon) if m)
        bucket[e] = bucket.get(e, 0) + c
    nz = [e for e, c in bucket.items() if c != 0]
    return min(nz) if nz else None
def newton_slopes(R, lam):
    pts = [(i, val(c, lam)) for i, c in enumerate(reversed(R.all_coeffs()))]
    pts = [(i, v) for i, v in pts if v is not None]
    i0 = pts[0][0]  # drop roots at 0
    pts = [(i - i0, v) for i, v in pts]
    # lower hull
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    out = []
    for a, b in zip(hull, hull[1:]):
        slope = Fr(b[1] - a[1]) / (b[0] - a[0]); out.append((-slope, b[0] - a[0]))  # root valuation, multiplicity
    return out
def lam_of(f1, g):
    L = {(0,0): Fr(0), (0,1): Fr(0), (0,2): f1, (1,0): g[0], (1,1): g[1], (1,2): g[2] - f1}
    return L
def exact_vals(facets, f1, g):
    R1, R2 = resultants(facets); L = lam_of(f1, g)
    lam = [L[c] if c in facets else Fr(0) for c in CELLS]
    return newton_slopes(R1, lam), newton_slopes(R2, lam)
if __name__ == "__main__":
    import time; t = time.time()
    f1 = Fr(3754, 9973); g = (Fr(5012, 9973), Fr(2557, 9973), Fr(2404, 9973))
    facets = [(0,0), (0,1), (0,2), (1,1), (1,2)]
    v1, v2 = exact_vals(facets, f1, g)
    print("val z1:", [(str(a), m) for a, m in v1]); print("val z2:", [(str(a), m) for a, m in v2]); print(time.time() - t)

# ---- exact pairing of the z1- and z2-valuations via the valuations of z1*z2 and z1/z2
s_ = sp.symbols('s')
_res2 = {}
def product_resultants(facets):
    key = tuple(facets)
    if key in _res2: return _res2[key]
    W = sum(A[CELLS.index(c)] * z1**NU[c][0] * z2**NU[c][1] for c in facets)
    def poly(e):
        e = sp.expand(e * z1**3 * z2**3); p = sp.Poly(e, z1, z2)
        m1 = min(m[0] for m in p.monoms()); m2 = min(m[1] for m in p.monoms())
        return sp.expand(e / (z1**m1 * z2**m2))
    e1 = poly(z1*sp.diff(W, z1)); e2 = poly(z2*sp.diff(W, z2))
    out = []
    for subs in ({z2: s_/z1}, {z2: z1/s_}):   # s = z1 z2,  s = z1 / z2
        f1_ = sp.numer(sp.together(e1.subs(subs))); f2_ = sp.numer(sp.together(e2.subs(subs)))
        out.append(sp.Poly(sp.resultant(sp.expand(f1_), sp.expand(f2_), z1), s_))
    _res2[key] = out; return out
def exact_positions(facets, f1, g):
    """Exact valuations (u1, u2) of all critical points, with multiplicity, from Newton polygons of four resultants."""
    from collections import Counter
    v1, v2 = exact_vals(facets, f1, g)
    L = lam_of(f1, g); lam = [L[c] if c in facets else Fr(0) for c in CELLS]
    Rp, Rq = product_resultants(facets)
    vp = Counter(); vq = Counter()
    for v, m in newton_slopes(Rp, lam): vp[v] += m
    for v, m in newton_slopes(Rq, lam): vq[v] += m
    A1 = [v for v, m in v1 for _ in range(m)]; A2 = [v for v, m in v2 for _ in range(m)]
    n = len(A1)
    sols = set()
    for perm in set(itertools.permutations(A2)):
        pairs = list(zip(A1, perm))
        P = Counter(a + b for a, b in pairs); Q = Counter(a - b for a, b in pairs)
        # the s-resultants may carry extra roots at s = 0 / infinity only; compare the finite nonzero parts
        if P == +vp and Q == +vq: sols.add(tuple(sorted(pairs)))
    return v1, v2, vp, vq, sols
