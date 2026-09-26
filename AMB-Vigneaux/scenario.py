"""Measurement scenarios and empirical models.

A *measurement scenario* is a finite set of measurements X, each with a finite
outcome set O_x, together with a measurement cover M: a family of contexts
(subsets of X that can be measured jointly).

The *event sheaf* E assigns to U ⊆ X the set of sections s: U -> ∏ O_x.
An *empirical model* e assigns to every context C a probability distribution
e_C on E(C).  This is the object shared by all three layers of the engine:

* Outcome layer   – its support presheaf S_e ⊆ E (Boolean / Z-coefficients).
* Probability layer – the distributions e_C on the simplices Δ(E(C)).
* Information layer – parameters θ with e = P_θ (see geometry.py).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

Section = Tuple  # outcomes aligned with a tuple of measurement names
Context = Tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    """Measurement scenario ⟨X, M, O⟩."""

    outcomes: Mapping[str, Tuple]
    contexts: Tuple[Context, ...]

    def __post_init__(self):
        object.__setattr__(self, "outcomes", {k: tuple(v) for k, v in self.outcomes.items()})
        object.__setattr__(self, "contexts", tuple(tuple(c) for c in self.contexts))
        for c in self.contexts:
            for m in c:
                if m not in self.outcomes:
                    raise ValueError(f"context {c} uses unknown measurement {m!r}")

    # --- event sheaf -----------------------------------------------------
    @property
    def measurements(self) -> Tuple[str, ...]:
        return tuple(self.outcomes)

    def sections(self, U: Sequence[str]) -> List[Section]:
        """E(U): all joint outcome assignments on U, in lexicographic order."""
        return list(itertools.product(*(self.outcomes[m] for m in U)))

    def n_sections(self, U: Sequence[str]) -> int:
        return int(np.prod([len(self.outcomes[m]) for m in U])) if len(U) else 1

    @staticmethod
    def restrict(s: Section, C: Sequence[str], U: Sequence[str]) -> Section:
        """Restriction map E(C) -> E(U) for U ⊆ C."""
        idx = [list(C).index(m) for m in U]
        return tuple(s[i] for i in idx)

    def overlap(self, C: Sequence[str], D: Sequence[str]) -> Context:
        """C ∩ D, ordered as in the scenario's measurement list."""
        cs, ds = set(C), set(D)
        return tuple(m for m in self.measurements if m in cs and m in ds)

    def global_sections(self) -> List[Section]:
        return self.sections(self.measurements)

    def restriction_matrix(self, C: Sequence[str], U: Sequence[str]) -> np.ndarray:
        """0/1 matrix R with (R p)[u] = Σ_{s|_U = u} p[s]  (pushforward E(C)→E(U))."""
        secC, secU = self.sections(C), self.sections(U)
        pos = {u: i for i, u in enumerate(secU)}
        R = np.zeros((len(secU), len(secC)))
        for j, s in enumerate(secC):
            R[pos[self.restrict(s, C, U)], j] = 1.0
        return R


@dataclass
class EmpiricalModel:
    """A family {e_C ∈ Δ(E(C))}_{C ∈ M}."""

    scenario: Scenario
    tables: Dict[Context, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        sc = self.scenario
        clean = {}
        for C in sc.contexts:
            if C not in self.tables:
                raise ValueError(f"missing table for context {C}")
            t = np.asarray(self.tables[C], dtype=float).reshape(-1)
            if t.size != sc.n_sections(C):
                raise ValueError(f"context {C}: expected {sc.n_sections(C)} entries, got {t.size}")
            if np.any(t < -1e-12):
                raise ValueError(f"context {C}: negative probabilities")
            s = t.sum()
            if s <= 0:
                raise ValueError(f"context {C}: zero mass")
            clean[C] = np.clip(t, 0, None) / s
        self.tables = clean

    # --- constructors ------------------------------------------------------
    @classmethod
    def from_function(cls, scenario: Scenario, f) -> "EmpiricalModel":
        """Build from f(context, section) -> weight (normalised per context)."""
        return cls(scenario, {C: np.array([f(C, s) for s in scenario.sections(C)], float)
                              for C in scenario.contexts})

    @classmethod
    def from_counts(cls, scenario: Scenario, counts: Mapping[Context, np.ndarray],
                    pseudocount: float = 0.0) -> "EmpiricalModel":
        """Empirical measure P̂_N = (1/N) Σ δ_{x_i}, optionally Laplace-smoothed."""
        return cls(scenario, {C: np.asarray(counts[C], float) + pseudocount for C in scenario.contexts})

    @classmethod
    def from_global(cls, scenario: Scenario, p_global: np.ndarray) -> "EmpiricalModel":
        """Pushforward of a distribution on E(X) — a non-contextual model."""
        X = scenario.measurements
        return cls(scenario, {C: scenario.restriction_matrix(X, C) @ p_global for C in scenario.contexts})

    def mix(self, other: "EmpiricalModel", lam: float) -> "EmpiricalModel":
        """(1-λ)·self + λ·other."""
        return EmpiricalModel(self.scenario, {C: (1 - lam) * self.tables[C] + lam * other.tables[C]
                                              for C in self.scenario.contexts})

    # --- structure ---------------------------------------------------------
    def prob(self, C: Context, s: Section) -> float:
        return float(self.tables[C][self.scenario.sections(C).index(tuple(s))])

    def support(self, eps: float = 1e-12) -> Dict[Context, List[Section]]:
        """Support presheaf S_e(C) = {s : e_C(s) > ε}.  This is the Boolean shadow
        that the outcome layer (AMB cohomology) operates on."""
        sc = self.scenario
        return {C: [s for s, p in zip(sc.sections(C), self.tables[C]) if p > eps] for C in sc.contexts}

    def marginal(self, C: Context, U: Sequence[str]) -> np.ndarray:
        return self.scenario.restriction_matrix(C, U) @ self.tables[C]

    def signalling_defect(self) -> float:
        """max_{C,D} ‖e_C|_{C∩D} − e_D|_{C∩D}‖_∞ ; zero iff the model is no-signalling
        (i.e. the e_C form a compatible family of the distribution presheaf)."""
        sc, worst = self.scenario, 0.0
        for C, D in itertools.combinations(sc.contexts, 2):
            U = sc.overlap(C, D)
            if U:
                worst = max(worst, float(np.max(np.abs(self.marginal(C, U) - self.marginal(D, U)))))
        return worst

    def sample(self, n: int | Mapping[Context, int], rng: np.random.Generator) -> Dict[Context, np.ndarray]:
        """Realisation: draw N outcomes per context; returns count vectors."""
        out = {}
        for C in self.scenario.contexts:
            k = n[C] if isinstance(n, Mapping) else n
            out[C] = rng.multinomial(int(k), self.tables[C]).astype(float)
        return out

    def __repr__(self):
        lines = [f"EmpiricalModel({len(self.scenario.contexts)} contexts)"]
        for C in self.scenario.contexts:
            lines.append(f"  {C}: " + " ".join(f"{p:.3f}" for p in self.tables[C]))
        return "\n".join(lines)


@dataclass
class GlobalLaw:
    """A law on global assignments E(X).  Sampling it is SINGLE-SOURCE: one draw
    fixes every measurement at once, and every context is read off the same
    sample.  By T0 the resulting empirical model always has a global section
    (the empirical joint), so γ = 0 and CF = 0 on it, on any cover."""

    scenario: Scenario
    p: np.ndarray

    def __post_init__(self):
        p = np.clip(np.asarray(self.p, float).ravel(), 0, None)
        if p.size != self.scenario.n_sections(self.scenario.measurements):
            raise ValueError("law must live on E(X)")
        self.p = p / p.sum()

    def model(self) -> "EmpiricalModel":
        return EmpiricalModel.from_global(self.scenario, self.p)

    def sample(self, n: int, rng: np.random.Generator) -> Dict[Context, np.ndarray]:
        sc, X = self.scenario, self.scenario.measurements
        g = rng.multinomial(int(n), self.p).astype(float)
        return {C: sc.restriction_matrix(X, C) @ g for C in sc.contexts}


def bell_scenario(parties: int = 2, settings: int = 2, outcomes: Iterable = (0, 1)) -> Scenario:
    """n-partite Bell scenario: party k has measurements <k><i>; contexts pick one per party.
    Measurement names: 'a0','a1',…,'b0','b1',…,'c0',…"""
    letters = "abcdefghij"[:parties]
    outs = {f"{l}{i}": tuple(outcomes) for l in letters for i in range(settings)}
    ctxs = [tuple(f"{l}{i}" for l, i in zip(letters, choice))
            for choice in itertools.product(range(settings), repeat=parties)]
    return Scenario(outs, tuple(ctxs))


def cyclic_scenario(n: int, outcomes: Iterable = (0, 1)) -> Scenario:
    """n-cycle scenario (n=4 is CHSH, n=5 is KCBS): contexts {x_i, x_{i+1}}."""
    outs = {f"x{i}": tuple(outcomes) for i in range(n)}
    return Scenario(outs, tuple((f"x{i}", f"x{(i + 1) % n}") for i in range(n)))
