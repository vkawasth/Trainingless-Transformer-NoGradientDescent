"""Step 1 of "Probability, Context and Differentiation: The Layers": the finite distribution monad, the context comonad, their
distributive law, marginalisation (total, affine) and conditioning (partial, nonlinear). New module. Exact arithmetic (Fraction).

Distributions are dicts {outcome: Fraction} with positive values summing to 1 (zero entries are dropped). A distribution of
distributions is a dict keyed by `freeze(p)`.

  freeze(p), thaw(k)          hashable form of a distribution and back
  dmap(f, p)                  the functor D on a map f: X -> Y (push-forward)
  eta(x), mu(Phi)             unit (point mass) and multiplication (flattening) of the monad D
  eps(lx), delta(lx)          counit (l, x) -> x and comultiplication (l, x) -> (l, (l, x)) of the comonad W = L x (-)
  lam(l, p)                   the distributive law W D -> D W, (l, p) -> delta_l (x) p
  bk_compose(f, g)            composite in the two-sided Kleisli category: mu . D g . lam . W f . delta
  marg(p, keep)               marginalisation of a law on tuples onto the coordinates `keep` (D of a projection)
  cond(p, A)                  conditioning on an event (a predicate); None where p(A) = 0 (partial)
  pool(pis, ps)               sum_l pi_l p_l
  bayes_weights(pis, ps, A)   pi_l^A = pi_l p_l(A) / sum_j pi_j p_j(A)
"""
from __future__ import annotations
from fractions import Fraction as Fr


def clean(p): return {k: v for k, v in p.items() if v != 0}
def freeze(p): return tuple(sorted(clean(p).items()))
def thaw(k): return dict(k)


def dmap(f, p):
    out = {}
    for x, v in p.items():
        y = f(x); out[y] = out.get(y, Fr(0)) + v
    return clean(out)


def eta(x): return {x: Fr(1)}


def mu(Phi):
    out = {}
    for k, w in Phi.items():
        for x, v in thaw(k).items(): out[x] = out.get(x, Fr(0)) + w * v
    return clean(out)


def eps(lx): return lx[1]
def delta(lx): return (lx[0], lx)
def lam(l, p): return clean({(l, x): v for x, v in p.items()})


def dmap_to_dists(f, p):
    """D applied to a map whose values are distributions: returns a distribution of distributions (keys frozen)"""
    out = {}
    for x, v in p.items():
        k = freeze(f(x)); out[k] = out.get(k, Fr(0)) + v
    return out


def bk_compose(f, g):
    """f: (l, x) -> D Y, g: (l, y) -> D Z; returns (l, x) -> D Z = mu . D g . lam . W f . delta"""
    def h(lx):
        l, inner = delta(lx)                       # (l, (l, x))
        Wf = (l, f(inner))                         # W f: (l, D Y)
        lamd = lam(*Wf)                            # D (L x Y)
        return mu(dmap_to_dists(g, lamd))          # D g then mu
    return h


def marg(p, keep):
    return dmap(lambda x: tuple(x[i] for i in keep), p)


def prob(p, A): return sum((v for x, v in p.items() if A(x)), Fr(0))


def cond(p, A):
    a = prob(p, A)
    if a == 0: return None
    return clean({x: v / a for x, v in p.items() if A(x)})


def pool(pis, ps):
    out = {}
    for w, p in zip(pis, ps):
        for x, v in p.items(): out[x] = out.get(x, Fr(0)) + w * v
    return clean(out)


def bayes_weights(pis, ps, A):
    num = [w * prob(p, A) for w, p in zip(pis, ps)]; s = sum(num, Fr(0)); return [n / s for n in num]
