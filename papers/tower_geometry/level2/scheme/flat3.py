"""The two-type latent-class map on three binary variables as a family over the space of observed laws (exact, sympy).
Coordinates: source (w1, mu, d); target (mu, m12, m13, m23, m123) = mean and central moments, a polynomial change of the law."""
import sympy as sp
w1, d1, d2, d3 = sp.symbols("w1 d1 d2 d3"); w2 = 1 - w1; p = w1 * w2; q = w2 - w1
M = sp.Matrix([p * d1 * d2, p * d1 * d3, p * d2 * d3, p * q * d1 * d2 * d3])          # (m12, m13, m23, m123)
J = M.jacobian([w1, d1, d2, d3]); det = sp.factor(J.det())
print("Jacobian determinant (mu block is the identity):", det)
# exact inverse
m12, m13, m23, m123 = sp.symbols("m12 m13 m23 m123"); D = m12 * m13 * m23
P_ = D / (m123 ** 2 + 4 * D)
sub = dict(zip([m12, m13, m23, m123], M))
print("p recovered:", sp.simplify(P_.subs(sub) - p) == 0)
print("q^2 recovered:", sp.simplify((1 - 4 * P_).subs(sub) - q ** 2) == 0)
for di, mjk in [(d1, m23), (d2, m13), (d3, m12)]:
    print(f"{di} = m123/(q m_jk):", sp.simplify((m123 / (q * mjk)).subs(sub) - di) == 0)
