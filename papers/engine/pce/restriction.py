"""Restriction categories, the tangent structure on hiding sheets, and global pooling (companion "A tangent structure for injected
data", Sections "Restriction axioms", "On the hiding polytopes", "Global pooling"). New module.

Partial maps are pairs (domain predicate, function) on laws. The restriction of f is the identity on dom f.
  Partial, compose(g, f), restrict(f), equal_on(f, g, pts)
  cond_map, rhat_map, gap_map            Bayesian conditioning, the post-probe estimator, the decision gap (partial maps on simplices)
  chord_polys(P, Q, r_c)                 along P_eps = (1 - eps) P + eps Q: q(eps) = M01 M02 M12 (degree <= 6, the domain) and
                                         h(eps) = m_T^2 - r_c^2 q (degree <= 6, the act changes of 'T iff rhat > r_c'), exact coefficients
  sturm_count(p, a, b)                   number of distinct real roots of p in (a, b) (Sturm's theorem, sympy)
  pooling_certificate(P, Q, r_c)         'safe' iff q and h have no roots in (0, 1) and P, Q are in the domain with the same act
"""
from __future__ import annotations
import itertools
import numpy as np
from . import injection as IJ
from .lab import moments

X3 = list(itertools.product((0, 1), repeat=3))


class Partial:
    def __init__(self, dom, fn, name=""):
        self.dom, self.fn, self.name = dom, fn, name

    def __call__(self, x): return self.fn(x) if self.dom(x) else None


def compose(g, f):
    return Partial(lambda x: f.dom(x) and g.dom(f.fn(x)), lambda x: g.fn(f.fn(x)), f"{g.name}.{f.name}")


def restrict(f): return Partial(f.dom, lambda x: x, f"bar({f.name})")


def equal_on(f, g, pts, tol=1e-12):
    for x in pts:
        a, b = f(x), g(x)
        if (a is None) != (b is None): return False
        if a is not None and np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) > tol: return False
    return True


def _M(P):
    _, M, mS = moments(IJ.as_law(np.asarray(P, float), 3)); return M, mS


cond_map = Partial(lambda P: P[0] + P[1] + P[2] + P[3] > 1e-12,
                   lambda P: np.r_[np.asarray(P[:4]) / np.sum(P[:4]), np.zeros(4)], "cond")      # condition on x_0 = 0, as a law on X3


def _q(P):
    M, _ = _M(P); return M[0, 1] * M[0, 2] * M[1, 2]


rhat_map = Partial(lambda P: _q(P) > 1e-14, lambda P: np.array([_M(P)[1]((0, 1, 2)) / (-np.sqrt(_q(P)))]), "rhat")


def gap_map(c0, c1):
    return Partial(rhat_map.dom, lambda P: np.array([c0 + c1 * rhat_map.fn(P)[0]]), "gap")


# ------------------------------------------------------------------------------------------------ global pooling along a chord
def chord_polys(P, Q, r_c):
    """exact polynomial coefficients (in eps) of q = M01 M02 M12 and h = m_T^2 - r_c^2 q along (1 - eps) P + eps Q (sympy)"""
    import sympy
    e = sympy.symbols("e")
    rat = lambda V: (lambda R: R[:-1] + [1 - sum(R[:-1])])([sympy.Rational(round(float(x) * 10 ** 12), 10 ** 12) for x in V])   # exact laws (sum 1)
    Pr, Qr = rat(P), rat(Q); L = [p * (1 - e) + q * e for p, q in zip(Pr, Qr)]
    xs = np.array(X3)
    E = lambda f: sum(L[k] * f(xs[k]) for k in range(8))
    mu = [E(lambda x, i=i: x[i]) for i in range(3)]
    Mij = lambda i, j: E(lambda x: x[i] * x[j]) - mu[i] * mu[j]
    mT = E(lambda x: (x[0] - mu[0]) * (x[1] - mu[1]) * (x[2] - mu[2]))
    q = sympy.expand(Mij(0, 1) * Mij(0, 2) * Mij(1, 2)); rc = sympy.nsimplify(r_c, rational=True)
    h = sympy.expand(sympy.expand(mT) ** 2 - rc ** 2 * q); return e, sympy.Poly(q, e), sympy.Poly(h, e), sympy.Poly(sympy.expand(mT), e)


def sturm_count(p, a=0, b=1):
    """distinct real roots of p in the open interval (a, b) (sympy's exact root counting, Sturm sequences)"""
    if p.is_zero: return np.inf
    sq = p.sqf_part() if p.degree() > 0 else p
    return int(sq.count_roots(a, b)) - int(sq.eval(a) == 0) - int(sq.eval(b) == 0) if p.degree() > 0 else 0


def pooling_certificate(P, Q, r_c, act):
    """act(P) in {+1, -1, None}. Returns dict with the counts and the verdict"""
    e, q, h, mT = chord_polys(P, Q, r_c)
    aP, aQ = act(P), act(Q)
    nq, nh = sturm_count(q), sturm_count(h)
    safe = aP is not None and aQ is not None and aP == aQ and nq == 0 and nh == 0
    return dict(act_P=aP, act_Q=aQ, roots_q=nq, roots_h=nh, deg_q=q.degree(), deg_h=h.degree(), safe=safe)


def genuine_crossings(P, Q, r_c):
    """exact decision along the chord: real roots of q and h in (0, 1) (isolated by sympy); a root of h is a genuine act change iff
    q > 0 there and m_T has the sign that makes rhat = m_T / (-sqrt q) equal r_c (squaring adds the other branch)"""
    import sympy
    e, q, h, mT = chord_polys(P, Q, r_c)
    def roots(p):
        if p.degree() <= 0: return []
        return sorted({float(r) for r in sympy.Poly(p.sqf_part()).real_roots() if 0 < float(r) < 1})
    qr = roots(q); hr = roots(h); genuine = []
    for x in hr:
        qv = float(q.eval(x)); mv = float(mT.eval(x))
        if qv > 0 and abs(mv / (-np.sqrt(qv)) - r_c) < 1e-6: genuine.append(x)
    return dict(q_roots=qr, h_roots=hr, genuine=genuine)


def exact_pooling(P, Q, r_c, act):
    g = genuine_crossings(P, Q, r_c); aP, aQ = act(P), act(Q)
    return dict(**g, safe=bool(aP is not None and aP == aQ and not g["q_roots"] and not g["genuine"]))
