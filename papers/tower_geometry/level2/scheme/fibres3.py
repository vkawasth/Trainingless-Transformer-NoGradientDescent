import sympy as sp
w1, d1, d2, d3 = sp.symbols("w1 d1 d2 d3"); w2 = 1 - w1; p = w1 * w2; q = w2 - w1; X = [d1, d2, d3, w1]
Mf = [p * d1 * d2, p * d1 * d3, p * d2 * d3, p * q * d1 * d2 * d3]
def fibre(vals):
    G = sp.groebner([sp.expand(f - v) for f, v in zip(Mf, vals)], *X, order="lex"); return G
def pts(G):
    return sp.solve(G.exprs, X, dict=True)
R = sp.Rational
# generic real law (from w1 = 3/10, d = (1/2, -1/5, 3/10)): two real points
v = [f.subs({w1: R(3, 10), d1: R(1, 2), d2: R(-1, 5), d3: R(3, 10)}) for f in Mf]
S = pts(fibre(v)); print("generic, Delta>0:", len(S), "points:", S)
# a law with Delta < 0 (flip the sign of m12 only): two complex-conjugate points
v2 = [-v[0], v[1], v[2], v[3]]; S2 = pts(fibre(v2)); print("Delta<0:", len(S2), "points; real?", [all(sp.im(sp.N(x)) == 0 for x in s.values()) for s in S2])
print("   ", [{k: sp.N(x, 4) for k, x in s.items()} for s in S2])
# stratum (ii): X3 independent of (X1, X2): m13 = m23 = m123 = 0, m12 != 0
G3 = fibre([R(1, 20), 0, 0, 0]); tgt = sp.groebner([d3, sp.expand(p * d1 * d2 - R(1, 20))], *X, order="lex")
print("stratum ii fibre ideal == (d3, w1 w2 d1 d2 - m12):", G3.exprs == tgt.exprs, "  -> a surface (dimension 2)")
# closure points not in the image: one pairwise moment zero, the others not
G4 = fibre([0, R(1, 20), R(1, 30), R(1, 100)]); print("m12 = 0 only: fibre ideal =", G4.exprs, "(empty fibre)")
