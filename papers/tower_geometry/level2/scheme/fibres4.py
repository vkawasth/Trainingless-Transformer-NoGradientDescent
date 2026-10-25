import itertools, sympy as sp
w1 = sp.Symbol("w1"); d = sp.symbols("d1:5"); w2 = 1 - w1; X = list(d) + [w1]; R = sp.Rational
ck = lambda k: w1 * w2 ** k + w2 * (-w1) ** k
Mf = [ck(len(S)) * sp.Mul(*[d[i] for i in S]) for k in range(2, 5) for S in itertools.combinations(range(4), k)]
pt = {w1: R(3, 10), d[0]: R(1, 2), d[1]: R(-1, 5), d[2]: R(3, 10), d[3]: 0}
J = sp.Matrix(Mf).jacobian(X); print("m=4, X4 independent: rank of d(moments)/d(w, d) =", J.subs(pt).rank(), "of 5 (full)")
G = sp.groebner([sp.expand(f - f.subs(pt)) for f in Mf], *X, order="lex"); S = sp.solve(G.exprs, X, dict=True)
print("fibre:", len(S), "points:", S)
