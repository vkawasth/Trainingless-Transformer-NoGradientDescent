"""Layer 2 — the probability layer, with Vigneaux / Baudot–Bennequin information
cohomology.

An *information structure* here is the set of observables of a finite joint
distribution: an observable is a tuple of variable names, and the product of
observables is their join (union of variables).  Observables form a monoid
acting on functions of probability laws by *conditioning*:

    (X · F)(P) = Σ_x P(X = x) · F(P | X = x)

(the module action of the doc's Phase 1, step 2).  With this action,
n-cochains F[X1|…|Xn](P) have the Hochschild-type coboundary

    δF[X1|…|Xn+1] = X1·F[X2|…|Xn+1]
                    + Σ_{i=1..n} (-1)^i F[X1|…|XiXi+1|…|Xn+1]
                    + (-1)^{n+1} F[X1|…|Xn],

and δ_t is the same operator with the trivial action in the first term.
Shannon entropy H is a 1-cocycle (δH = 0 is the chain rule, and up to a
constant it generates H¹ — Baudot–Bennequin 2015, Vigneaux 2020), while
δ_t H = I is mutual information.  Higher co-informations I_k arise the same
way and are provided in closed (inclusion–exclusion) form.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

Observable = Tuple[str, ...]


@dataclass(frozen=True)
class FiniteJoint:
    """A law P on ∏_v O_v, stored as an ndarray with one axis per variable."""

    variables: Tuple[str, ...]
    p: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.p, dtype=float)
        if p.ndim != len(self.variables):
            raise ValueError("one axis per variable required")
        object.__setattr__(self, "p", p)

    @classmethod
    def from_flat(cls, variables: Sequence[str], shape: Sequence[int], flat: np.ndarray) -> "FiniteJoint":
        return cls(tuple(variables), np.asarray(flat, float).reshape(tuple(shape)))

    def _axes(self, obs: Observable) -> Tuple[int, ...]:
        return tuple(self.variables.index(v) for v in obs)

    @staticmethod
    def join(*obs: Observable) -> Observable:
        """Monoid product XY of observables (join = union, order-stable)."""
        out: List[str] = []
        for o in obs:
            for v in o:
                if v not in out:
                    out.append(v)
        return tuple(out)

    def marginal(self, obs: Observable) -> np.ndarray:
        keep = self._axes(obs)
        drop = tuple(i for i in range(len(self.variables)) if i not in keep)
        m = self.p.sum(axis=drop) if drop else self.p
        # reorder the kept axes to follow `obs`
        order = sorted(keep)
        return np.moveaxis(m, [order.index(a) for a in keep], list(range(len(keep)))) if keep else np.asarray(m)

    def condition(self, obs: Observable, value: Tuple) -> "FiniteJoint | None":
        """P | (X = x), as a law on the same variables (None if P(X=x)=0)."""
        idx = [slice(None)] * len(self.variables)
        for a, v in zip(self._axes(obs), value):
            idx[a] = v
        mask = np.zeros_like(self.p)
        mask[tuple(idx)] = 1.0
        q = self.p * mask
        z = q.sum()
        return None if z <= 0 else FiniteJoint(self.variables, q / z)

    def values(self, obs: Observable) -> Iterable[Tuple]:
        return itertools.product(*(range(self.p.shape[a]) for a in self._axes(obs)))


# ---------------------------------------------------------------------------
# functionals, action, coboundaries
# ---------------------------------------------------------------------------
Cochain = Callable[..., float]  # F(P, X1, …, Xn) -> float


def shannon(q: np.ndarray, base: float = 2.0) -> float:
    q = np.asarray(q, float).ravel()
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(base))


def entropy(P: FiniteJoint, X: Observable) -> float:
    """H[X](P) — the entropy 1-cochain."""
    return shannon(P.marginal(X))


def act(X: Observable, F: Cochain) -> Cochain:
    """Conditioning action  (X·F)(P, …) = Σ_x P(X=x) F(P|X=x, …)."""
    def XF(P: FiniteJoint, *args):
        px = P.marginal(X)
        tot = 0.0
        for x in P.values(X):
            w = float(px[x]) if px.ndim else float(px)
            if w > 0:
                tot += w * F(P.condition(X, x), *args)
        return tot
    return XF


def coboundary(F: Cochain, n: int, trivial: bool = False) -> Cochain:
    """δF (or δ_t F if trivial=True) for an n-cochain F; returns an (n+1)-cochain."""
    def dF(P: FiniteJoint, *Xs: Observable) -> float:
        assert len(Xs) == n + 1, f"expected {n + 1} observables"
        first = F(P, *Xs[1:]) if trivial else act(Xs[0], F)(P, *Xs[1:])
        tot = first
        for i in range(1, n + 1):
            merged = Xs[:i - 1] + (FiniteJoint.join(Xs[i - 1], Xs[i]),) + Xs[i + 1:]
            tot += (-1) ** i * F(P, *merged)
        tot += (-1) ** (n + 1) * F(P, *Xs[:n])
        return tot
    return dF


def mutual_information(P: FiniteJoint, X: Observable, Y: Observable) -> float:
    """I(X;Y) = δ_t H [X|Y]."""
    return coboundary(entropy, 1, trivial=True)(P, X, Y)


def conditional_entropy(P: FiniteJoint, Y: Observable, X: Observable) -> float:
    """H(Y|X) = (X·H)[Y]."""
    return act(X, entropy)(P, Y)


def co_information(P: FiniteJoint, *Xs: Observable) -> float:
    """Multivariate co-information I_k(X1;…;Xk) = Σ_{∅≠S} (-1)^{|S|+1} H(X_S)."""
    k, tot = len(Xs), 0.0
    for r in range(1, k + 1):
        for S in itertools.combinations(Xs, r):
            tot += (-1) ** (r + 1) * entropy(P, FiniteJoint.join(*S))
    return tot


def cocycle_defect(F: Cochain, P: FiniteJoint, X: Observable, Y: Observable) -> float:
    """|δF[X|Y](P)| — zero for every P, X, Y iff F is a 1-cocycle."""
    return abs(coboundary(F, 1)(P, X, Y))


# ---------------------------------------------------------------------------
# per-context profile of an empirical model
# ---------------------------------------------------------------------------
def context_joint(model, C) -> FiniteJoint:
    sc = model.scenario
    shape = [len(sc.outcomes[m]) for m in C]
    return FiniteJoint.from_flat(C, shape, model.tables[C])


def information_profile(model) -> Dict:
    """Information-layer readout of each context: marginal entropies, joint entropy,
    total correlation / mutual information (δ_t H), co-information, and the entropy
    cocycle defect (a numerical certificate that the chain rule holds)."""
    out = {}
    for C in model.scenario.contexts:
        P = context_joint(model, C)
        singles = [(m,) for m in C]
        Hs = {m: entropy(P, (m,)) for m in C}
        H_joint = entropy(P, tuple(C))
        prof = {"H": Hs, "H_joint": H_joint, "total_correlation": sum(Hs.values()) - H_joint}
        if len(C) >= 2:
            prof["I_first_rest"] = mutual_information(P, singles[0], tuple(C[1:]))
            prof["cocycle_defect"] = cocycle_defect(entropy, P, singles[0], tuple(C[1:]))
            prof["co_information"] = co_information(P, *singles)
        out[C] = prof
    return out


def mean_mutual_information(model) -> float:
    """Scalar summary: average over contexts of the total correlation."""
    prof = information_profile(model)
    return float(np.mean([v["total_correlation"] for v in prof.values()]))
