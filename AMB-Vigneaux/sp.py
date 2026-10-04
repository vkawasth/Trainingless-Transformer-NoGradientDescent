"""Structured posteriors on finite spaces: a Giry-style monad that carries factors, evidence and structural subspaces.

Everything is exact (fractions.Fraction); no floating point unless a caller asks for it.

Records live in a monoid M. Shipped monoids:
    FactorSet       frozenset of factor names under union          commutative, idempotent (a join-semilattice)
    FactorMultiset  multiset of factor names under sum              commutative, NOT idempotent
    Trace           tuple of evidence items under concatenation     NOT commutative, NOT idempotent
    Subspace        subspace of Q^n under sum (span), as RREF      commutative, idempotent
    Product(M1, ..)  componentwise

Two constructions of SP(X):
  * PerBranch  = G(X x M)  (the writer transformer on the finite distribution monad). Each outcome carries its own
    record; bind multiplies records along each branch. It is a monad for EVERY monoid M (laws hold exactly).
  * Global     = G(X) x M  with ONE record for the whole posterior. bind((mu, m), K) = (mu >>= nu, m . AGG_{x in supp mu} m_x),
    AGG the monoid product over the support. Unit and both identity laws hold for every M; ASSOCIATIVITY holds exactly
    iff M is commutative and idempotent on the records reached: a record reachable along two paths is counted twice
    otherwise. (Checked exactly in tests: holds for FactorSet / Subspace, fails for FactorMultiset / Trace.)
Both have a forgetful map U to the plain distribution monad with U(bind) = bind(U) (naturality, checked).
Conditioning on evidence e with likelihood L_e: mu_e = L_e mu / Z_e, factors get L_e, the trace gets e.
"""
from fractions import Fraction as Fr
from collections import Counter
from itertools import product as iproduct


# ----------------------------------------------------------------------------------------------- monoids
class FactorSet:
    unit = frozenset()
    commutative = idempotent = True
    @staticmethod
    def mul(a, b): return a | b
    @staticmethod
    def of(*names): return frozenset(names)


class FactorMultiset:
    unit = frozenset()                                    # a multiset as a frozenset of (name, count) pairs
    commutative, idempotent = True, False
    @staticmethod
    def mul(a, b):
        c = Counter(dict(a)); c.update(dict(b)); return frozenset(c.items())
    @staticmethod
    def of(*names): return frozenset(Counter(names).items())


class Trace:
    unit = ()
    commutative = idempotent = False
    @staticmethod
    def mul(a, b): return tuple(a) + tuple(b)
    @staticmethod
    def of(*items): return tuple(items)


def rref(rows, n):
    """reduced row echelon basis (tuple of tuples of Fractions) of the span of rows in Q^n"""
    A = [[Fr(x) for x in r] for r in rows if any(x != 0 for x in r)]; out = []; col = 0
    while A and col < n:
        piv = next((i for i, r in enumerate(A) if r[col] != 0), None)
        if piv is None: col += 1; continue
        r = A.pop(piv); r = [x / r[col] for x in r]
        A = [[x - a[col] * y for x, y in zip(a, r)] for a in A]; A = [a for a in A if any(x != 0 for x in a)]
        out = [[x - o[col] * y for x, y in zip(o, r)] for o in out]; out.append(r); col += 1
    return tuple(sorted(tuple(r) for r in out))


def Subspace(n):
    class _S:
        unit = ()
        commutative = idempotent = True
        dim = n
        @staticmethod
        def mul(a, b): return rref(list(a) + list(b), n)
        @staticmethod
        def of(*vecs): return rref(vecs, n)
    return _S


def Product(*Ms):
    class _P:
        unit = tuple(M.unit for M in Ms)
        commutative = all(M.commutative for M in Ms); idempotent = all(M.idempotent for M in Ms)
        @staticmethod
        def mul(a, b): return tuple(M.mul(x, y) for M, x, y in zip(Ms, a, b))
    _P.parts = Ms
    return _P


# ----------------------------------------------------------------------------------------------- distributions
def dist(d):
    """normalised finite distribution {outcome: Fraction}, zero masses dropped"""
    d = {k: Fr(v) for k, v in d.items() if Fr(v) != 0}; z = sum(d.values()); return {k: v / z for k, v in d.items()}


def giry_bind(mu, k):
    out = {}
    for x, p in mu.items():
        for y, q in k(x).items():
            out[y] = out.get(y, 0) + p * q
    return {y: v for y, v in out.items() if v != 0}


# ----------------------------------------------------------------------------------------------- per-branch G(X x M)
class PerBranch:
    """SP(X) = G(X x M): a distribution over (outcome, record) pairs."""
    def __init__(self, M): self.M = M
    def unit(self, x): return {(x, self.M.unit): Fr(1)}
    def bind(self, rho, K):
        out = {}
        for (x, m), p in rho.items():
            for (y, m2), q in K(x).items():
                key = (y, self.M.mul(m, m2)); out[key] = out.get(key, 0) + p * q
        return {k: v for k, v in out.items() if v != 0}
    def U(self, rho):
        out = {}
        for (x, _), p in rho.items(): out[x] = out.get(x, 0) + p
        return out
    def records(self, rho):
        out = {}
        for (_, m), p in rho.items(): out[m] = out.get(m, 0) + p
        return out
    def condition(self, rho, L, e=None, factor=None, trace_slot=None, factor_slot=None):
        """mu_e ∝ L(x) mu; each branch's record gains the factor / evidence item (slots index a Product monoid)"""
        w = {k: v * Fr(L(k[0])) for k, v in rho.items()}; Z = sum(w.values())
        if Z == 0: raise ZeroDivisionError("evidence has probability zero")
        return {(x, self._stamp(m, e, factor, trace_slot, factor_slot)): v / Z for (x, m), v in w.items() if v != 0}
    def _stamp(self, m, e, factor, ts, fs):
        if not hasattr(self.M, "parts"): return m
        m = list(m)
        if fs is not None and factor is not None: m[fs] = self.M.parts[fs].mul(m[fs], self.M.parts[fs].of(factor))
        if ts is not None and e is not None: m[ts] = self.M.parts[ts].mul(m[ts], self.M.parts[ts].of(e))
        return tuple(m)
    def given_record(self, rho, m):
        """the posterior of the branch that carries record m (exact)"""
        sub = {x: p for (x, mm), p in rho.items() if mm == m}; z = sum(sub.values()); return {x: p / z for x, p in sub.items()}


# ----------------------------------------------------------------------------------------------- global G(X) x M
class Global:
    """SP(X) = G(X) x M with ONE record per posterior; bind aggregates the kernel's records over the support."""
    def __init__(self, M): self.M = M
    def unit(self, x): return ({x: Fr(1)}, self.M.unit)
    def bind(self, rho, K):
        mu, m = rho; agg = self.M.unit
        for x in sorted(mu, key=repr):                    # a fixed order; matters only for non-commutative M
            agg = self.M.mul(agg, K(x)[1])
        return (giry_bind(mu, lambda x: K(x)[0]), self.M.mul(m, agg))
    def U(self, rho): return rho[0]
    def condition(self, rho, L, e=None, factor=None, trace_slot=None, factor_slot=None):
        mu, m = rho; w = {x: p * Fr(L(x)) for x, p in mu.items()}; Z = sum(w.values())
        if Z == 0: raise ZeroDivisionError("evidence has probability zero")
        return ({x: v / Z for x, v in w.items() if v != 0}, PerBranch(self.M)._stamp(m, e, factor, trace_slot, factor_slot))


# ----------------------------------------------------------------------------------------------- law checks (exact)
def check_laws(T, rho, K, L, xs):
    """exact monad laws for construction T (PerBranch or Global): left identity on each x in xs, right identity, associativity"""
    left = all(T.bind(T.unit(x), K) == K(x) for x in xs)
    right = T.bind(rho, T.unit) == rho
    assoc = T.bind(T.bind(rho, K), L) == T.bind(rho, lambda x: T.bind(K(x), L))
    nat = giry_bind(T.U(rho), lambda x: T.U(K(x))) == T.U(T.bind(rho, K))
    return dict(left_identity=left, right_identity=right, associativity=assoc, naturality_of_U=nat)


def mutual_information(joint):
    """exact-input, float-output I(A;B) in nats for a joint {(a, b): p}"""
    from math import log
    pa, pb = {}, {}
    for (a, b), p in joint.items(): pa[a] = pa.get(a, 0) + p; pb[b] = pb.get(b, 0) + p
    return sum(float(p) * log(float(p) / (float(pa[a]) * float(pb[b]))) for (a, b), p in joint.items() if p > 0)
