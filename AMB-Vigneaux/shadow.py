"""Posterior + shadow on finite spaces: alternatives (+), joint factors (x), conditioning, and a context comonad. Exact (Fraction).

A shadowed posterior is a finite formal mixture of branches  sum_i w_i [mu_i, S_i]:  w_i > 0 summing to 1, mu_i a
distribution on X, S_i a shadow (here a frozenset of factor / evidence names). Branches with equal (mu, S) are merged.
  a (+)_lam b     alternatives: branches of a weighted lam, branches of b weighted 1 - lam; shadows are NOT merged
  a (x) b         joint factors: pairs (i, j) give the branch [mu_i mu_j / Z_ij, S_i u S_j] with weight w_i v_j Z_ij,
                  renormalised (Z_ij = sum_x mu_i(x) nu_j(x)); undefined if every Z_ij = 0
  cond(a, L, e)   = a (x) [L, {e}]  (L a likelihood, unnormalised): Bayes' rule on mu, and the branch weights move by
                  the branch evidence, so P(branch | e) comes out exactly
  U(a)            the plain mixture sum_i w_i mu_i (the Giry image)
Checked exactly (tests/test_shadow.py): (+) is commutative, associative (with the induced weights) and idempotent;
(x) is commutative and associative; (x) distributes over (+) only with the evidence-reweighted mixing weight
lam' = lam Z_a / (lam Z_a + (1 - lam) Z_b) -- Bayes -- and NOT with the fixed weight; U is a homomorphism for both.
Context comonad on X:  C(X) = {(x, rho, S) : rho(x) > 0},  eps(x, rho, S) = x,
  delta(x, rho, S) = ((x, rho, S), rho^, S)  where rho^ is the law of (x', rho, S), x' ~ rho  (the context redrawn),
  C(f)(x, rho, S) = (f x, f_* rho, S).  Counit and coassociativity laws are checked exactly.
"""
from fractions import Fraction as Fr


def _norm(d):
    d = {k: Fr(v) for k, v in d.items() if Fr(v) != 0}; z = sum(d.values())
    if z == 0: raise ZeroDivisionError("zero mass")
    return {k: v / z for k, v in d.items()}


def _key(mu): return tuple(sorted(mu.items(), key=lambda kv: repr(kv[0])))


class SP:
    """a shadowed posterior: {(mu_key, shadow): weight}"""
    def __init__(self, branches):
        acc = {}
        for w, mu, S in branches:
            if Fr(w) == 0: continue
            k = (_key(_norm(mu)), frozenset(S)); acc[k] = acc.get(k, 0) + Fr(w)
        z = sum(acc.values()); self.b = {k: v / z for k, v in acc.items()}
    @staticmethod
    def point(mu, S=()): return SP([(1, mu, S)])
    def __eq__(self, o): return isinstance(o, SP) and self.b == o.b
    def __repr__(self): return "SP(%s)" % self.b
    def branches(self): return [(w, dict(k[0]), k[1]) for k, w in self.b.items()]
    def U(self):
        out = {}
        for w, mu, _ in self.branches():
            for x, p in mu.items(): out[x] = out.get(x, 0) + w * p
        return out
    def evidence(self, L):
        """sum_i w_i sum_x mu_i(x) L(x): the marginal likelihood of the evidence under this posterior"""
        return sum(w * sum(p * Fr(L.get(x, 0)) for x, p in mu.items()) for w, mu, _ in self.branches())


def oplus(a, b, lam):
    lam = Fr(lam); return SP([(lam * w, mu, S) for w, mu, S in a.branches()] + [((1 - lam) * w, mu, S) for w, mu, S in b.branches()])


def otimes(a, b):
    out = []
    for w, mu, S in a.branches():
        for v, nu, T in b.branches():
            prod = {x: mu[x] * nu[x] for x in mu if x in nu and mu[x] * nu[x] != 0}; Z = sum(prod.values())
            if Z: out.append((w * v * Z, prod, S | T))
    if not out: raise ZeroDivisionError("no compatible pair of branches")
    return SP(out)


def cond(a, L, e):
    """Bayes on every branch, branch weights moved by the branch evidence; the shadow records e"""
    return otimes(a, SP([(1, {x: Fr(v) for x, v in L.items()}, {e})]))


def bayes_weight(a, b, lam, L):
    """the mixing weight that makes (a (+)_lam b) (x) L = (a (x) L) (+)_lam' (b (x) L): lam' = lam Z_a / (lam Z_a + (1-lam) Z_b)"""
    lam = Fr(lam); Za, Zb = a.evidence(L), b.evidence(L); return lam * Za / (lam * Za + (1 - lam) * Zb)


# ------------------------------------------------------------------------- context comonad (finite)
def ctx(x, rho, S=()):
    rho = _norm(rho)
    if rho.get(x, 0) == 0: raise ValueError("x must lie in the support of rho")
    return (x, _key(rho), frozenset(S))


def eps(c): return c[0]


def delta(c):
    x, rk, S = c; rho = dict(rk)
    return (c, _key({ctx(xp, rho, S): p for xp, p in rho.items()}), S)


def cmap(f, c):
    x, rk, S = c; out = {}
    for xp, p in dict(rk).items(): out[f(xp)] = out.get(f(xp), 0) + p
    return (f(x), _key(out), S)
