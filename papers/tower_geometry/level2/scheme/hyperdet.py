import itertools, sympy as sp
# law on {0,1}^3 from mean and central moments; Cayley hyperdeterminant of the 2x2x2 table
mu = sp.symbols("mu1:4"); m12, m13, m23, m123 = sp.symbols("m12 m13 m23 m123"); mom = {(0, 1): m12, (0, 2): m13, (1, 2): m23, (0, 1, 2): m123}
def P(x):  # P(x) = E prod (indicator) expanded via central moments
    e = 0
    for k in range(4):
        for S in itertools.combinations(range(3), k):
            f = mom.get(S, 0) if k >= 2 else (1 if k == 0 else 0)
            rest = sp.Mul(*[(mu[i] if x[i] else 1 - mu[i]) for i in range(3) if i not in S])
            e += f * rest * sp.Mul(*[(1 if x[i] else -1) for i in S])
    return sp.expand(e)
a = {x: P(x) for x in itertools.product([0, 1], repeat=3)}
print("law sums to 1:", sp.simplify(sum(a.values()) - 1) == 0)
A = lambda i, j, k: a[(i, j, k)]
Det = (A(0,0,0)**2*A(1,1,1)**2 + A(0,0,1)**2*A(1,1,0)**2 + A(0,1,0)**2*A(1,0,1)**2 + A(1,0,0)**2*A(0,1,1)**2
       - 2*(A(0,0,0)*A(0,0,1)*A(1,1,0)*A(1,1,1) + A(0,0,0)*A(0,1,0)*A(1,0,1)*A(1,1,1) + A(0,0,0)*A(1,0,0)*A(0,1,1)*A(1,1,1)
            + A(0,0,1)*A(0,1,0)*A(1,0,1)*A(1,1,0) + A(0,0,1)*A(1,0,0)*A(0,1,1)*A(1,1,0) + A(0,1,0)*A(1,0,0)*A(0,1,1)*A(1,0,1))
       + 4*(A(0,0,0)*A(0,1,1)*A(1,0,1)*A(1,1,0) + A(0,0,1)*A(0,1,0)*A(1,0,0)*A(1,1,1)))
print("hyperdeterminant == m123^2 + 4 m12 m13 m23:", sp.expand(Det - (m123**2 + 4*m12*m13*m23)) == 0)
# the three real regions: an example with Det > 0 but m12 m13 m23 < 0 -> real explanation with weights outside [0, 1]
w1, d1, d2, d3 = sp.symbols("w1 d1 d2 d3"); w2 = 1 - w1; p = w1*w2; q = w2 - w1
Mf = [p*d1*d2, p*d1*d3, p*d2*d3, p*q*d1*d2*d3]; R = sp.Rational
v = [R(-1, 50), R(1, 40), R(1, 60), R(1, 10)]; print("example Det =", v[3]**2 + 4*v[0]*v[1]*v[2], " Delta =", v[0]*v[1]*v[2])
S = sp.solve(sp.groebner([sp.expand(f - x) for f, x in zip(Mf, v)], d1, d2, d3, w1, order="lex").exprs, [d1, d2, d3, w1], dict=True)
print("  explanations:", [{k: sp.N(x, 4) for k, x in s.items()} for s in S])
print("Det =", sp.factor(Det))
