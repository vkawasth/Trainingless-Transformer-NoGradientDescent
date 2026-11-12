"""The cut quotient as an elimination ideal, and the cost of cutting a decision path (rem:elimquot of the companion note). New module;
it modifies nothing.
For an outcome f, a polynomial in crossing (or switch) variables e_1..e_n with e_i^2 = 1, and a cut (S, S^c):
  coeff_map(f, S, n)         write f = sum_m a_m(x_S) m(y), m running over the multilinear monomials of the right-hand variables; return the
                             list of coefficient polynomials a_m in the left variables (only the nonzero ones)
  elimination(f, S, n, dom)  the reduced Groebner basis (lex, left variables first) of <e_i^2 - 1 (i in S), u_m - a_m(x)>, and the part
                             free of the left variables: the elimination ideal in u, whose zeros are the images of the coefficient map
  degree(G_u, u)             the number of standard monomials of the zero-dimensional ideal G_u (= its number of points, counted with
                             multiplicity; the ideal of a finite set of points is radical, so this is the number of classes)
  rank(f, S, n)              the dimension of the span of the coefficient vectors over all left assignments
  cut_cost(f, n, order, k)   for switches visited in the given time order, the (left, right) class counts at the cut after k switches
The classes of the cut quotient (pce/boundary.classes) are the fibres of x -> (a_m(x)): two left assignments give the same outcome against
every right completion iff their coefficient vectors agree, since the right monomials are linearly independent functions.
"""
from __future__ import annotations
import itertools
import sympy as sp

E = sp.symbols("e1:9")


def _reduce(p, vars_):
    p = sp.Poly(sp.expand(p), *vars_); out = 0
    for mon, c in p.terms():
        out += c * sp.prod([v ** (k % 2) for v, k in zip(vars_, mon)])
    return sp.expand(out)


def coeff_map(f, S, n):
    xs = [E[i] for i in S]; ys = [E[i] for i in range(n) if i not in S]
    g = _reduce(f, list(E[:n])); P = sp.Poly(g, *ys) if ys else None
    if P is None: return [g]
    return [sp.expand(c) for mon, c in P.terms() if sp.expand(c) != 0]


def elimination(f, S, n, domain="QQ"):
    xs = [E[i] for i in S]; a = coeff_map(f, S, n); u = sp.symbols(f"u0:{len(a)}")
    polys = [x ** 2 - 1 for x in xs] + [ui - ai for ui, ai in zip(u, a)]
    G = sp.groebner(polys, *xs, *u, order="lex", domain=domain)
    Gu = [g for g in G.exprs if not (sp.sympify(g).free_symbols & set(xs))]
    return G, Gu, u, a


def degree(Gu, u, domain="QQ"):
    if not u: return 1
    G = sp.groebner(Gu, *u, order="grevlex", domain=domain)
    lead = [sp.Poly(g, *u).monoms(order="grevlex")[0] for g in G.exprs]
    bounds = []
    for i in range(len(u)):                                             # a pure power u_i^d must lead some element (zero-dimensional)
        ds = [m[i] for m in lead if all(m[j] == 0 for j in range(len(u)) if j != i)]
        bounds.append(min(ds) if ds else None)
    if any(b is None for b in bounds): raise ValueError("not zero-dimensional")
    count = 0
    for mon in itertools.product(*[range(b) for b in bounds]):
        if not any(all(mon[j] >= m[j] for j in range(len(u))) for m in lead): count += 1
    return count


def rank(f, S, n, subs=None):
    xs = [E[i] for i in S]; a = coeff_map(f, S, n); rows = []
    for x in itertools.product((1, -1), repeat=len(xs)):
        r = [ai.subs(dict(zip(xs, x))) for ai in a]
        rows.append([sp.sympify(v).subs(subs) if subs else v for v in r])
    return sp.Matrix(rows).rank()


def cut_cost(f, n, order, k):
    """switch order[t] is visited at time t; cut after k switches: (classes of the past, classes of the future)"""
    past = list(order[:k]); fut = list(order[k:])
    _, Gl, ul, _ = elimination(f, past, n); _, Gr, ur, _ = elimination(f, fut, n)
    return degree(Gl, ul), degree(Gr, ur)
