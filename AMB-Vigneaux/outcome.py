"""Layer 3 — the outcome layer: sheaf sections and obstructions (Abramsky et al.).

Everything here sees only the *support presheaf* S_e of an empirical model
(plus, for the contextual fraction, the probabilities themselves).  The
hierarchy of obstructions implemented:

* Logical contextuality at (C, s):  s ∈ S_e(C) is not the restriction of any
  global section g ∈ E(X) with g|_D ∈ S_e(D) for every context D.
* Strong contextuality:  S_e has no global section at all.
* Cohomological obstruction γ(s) ∈ Ȟ¹ (Abramsky–Mansfield–Barbosa 2011):
  γ(s) = 0  iff  there is a compatible family {r_D ∈ Z·S_e(D)}_D of
  Z-linear combinations of support sections with r_C = s.  Decided exactly
  by integer linear algebra.  γ(s) ≠ 0 ⇒ logical contextuality at s
  (the converse can fail — "false negatives" are reported explicitly).
* Contextual fraction CF(e) (Abramsky–Barbosa–Mansfield 2017): the minimal
  λ such that e = (1-λ)e^NC + λe^SC, via linear programming.  Its LP dual
  is an optimal (normalised) Bell inequality, also returned.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from .scenario import Context, EmpiricalModel, Scenario, Section
from .zlinalg import solve_integer, solve_mod_p, solve_rational

SupportMap = Dict[Context, List[Section]]


# ---------------------------------------------------------------------------
# possibilistic (Boolean) level
# ---------------------------------------------------------------------------
def compatible_global_sections(scenario: Scenario, support: SupportMap) -> List[Section]:
    """Global sections g ∈ E(X) whose every contextual restriction is possible."""
    X = scenario.measurements
    supp_sets = {C: set(support[C]) for C in scenario.contexts}
    out = []
    for g in scenario.sections(X):
        if all(scenario.restrict(g, X, C) in supp_sets[C] for C in scenario.contexts):
            out.append(g)
    return out


def extends_to_global(scenario: Scenario, support: SupportMap, C: Context, s: Section,
                      _globals: Optional[List[Section]] = None) -> bool:
    gs = compatible_global_sections(scenario, support) if _globals is None else _globals
    X = scenario.measurements
    return any(scenario.restrict(g, X, C) == tuple(s) for g in gs)


# ---------------------------------------------------------------------------
# Z-coefficient Čech obstruction
# ---------------------------------------------------------------------------
@dataclass
class CohomologyResult:
    context: Context
    section: Section
    obstruction_vanishes: bool
    witness: Optional[Dict[Context, Dict[Section, int]]] = None  # compatible Z-family if γ(s)=0


def cohomological_obstruction(scenario: Scenario, support: SupportMap,
                              C0: Context, s: Section, ring="Z") -> CohomologyResult:
    """Decide γ(s) = 0 for s ∈ S_e(C0) by searching for a compatible family
    {r_D ∈ R S_e(D)} with r_{C0} = s (AMB 2011, Prop. 4.2 / relative Čech H¹).

    ring: "Z" (default, exact Smith-type diagonalisation), "Q" (exact
    fractions), or a prime p for Z/p.  The class is ring-dependent: a mod-p
    contradiction is p-torsion, so it is seen over Z and Z/p and not over Q."""
    s = tuple(s)
    if s not in support[C0]:
        raise ValueError(f"{s} is not in the support over {C0}")
    ctxs = list(scenario.contexts)
    var = {}
    for C in ctxs:
        for t in support[C]:
            var[(C, t)] = len(var)
    rows, rhs = [], []
    # r_{C0} = s
    for t in support[C0]:
        row = [0] * len(var)
        row[var[(C0, t)]] = 1
        rows.append(row)
        rhs.append(1 if t == s else 0)
    # compatibility on every overlap (the empty overlap forces equal total weight)
    for C, D in itertools.combinations(ctxs, 2):
        U = scenario.overlap(C, D)
        for u in scenario.sections(U):
            row = [0] * len(var)
            for t in support[C]:
                if scenario.restrict(t, C, U) == u:
                    row[var[(C, t)]] += 1
            for t in support[D]:
                if scenario.restrict(t, D, U) == u:
                    row[var[(D, t)]] -= 1
            if any(row):
                rows.append(row)
                rhs.append(0)
    if ring == "Z":
        x = solve_integer(rows, rhs)
    elif ring == "Q":
        x = solve_rational(rows, rhs)
    else:
        x = solve_mod_p(rows, rhs, int(ring))
    if x is None:
        return CohomologyResult(C0, s, False)
    fam = {C: {t: x[var[(C, t)]] for t in support[C] if x[var[(C, t)]]} for C in ctxs}
    return CohomologyResult(C0, s, True, fam)


# ---------------------------------------------------------------------------
# probabilistic level: contextual fraction
# ---------------------------------------------------------------------------
@dataclass
class ContextualFraction:
    value: float                      # CF(e) ∈ [0,1]
    noncontextual_part: np.ndarray    # optimal b ≥ 0 on E(X) (mass 1-CF)
    bell_inequality: Dict[Context, np.ndarray]  # dual y ≥ 0: Σ_C y_C·e_C ≥ 1 for NC models... see note

    # Note: the dual optimum y satisfies  Σ_C ⟨y_C, d_C⟩ ≥ 1 for every deterministic
    # non-contextual model d, and Σ_C ⟨y_C, e_C⟩ = 1 − CF(e).


def contextual_fraction(model: EmpiricalModel) -> ContextualFraction:
    sc = model.scenario
    X = sc.measurements
    G = sc.global_sections()
    blocks = [sc.restriction_matrix(X, C) for C in sc.contexts]
    M = np.vstack(blocks)
    v = np.concatenate([model.tables[C] for C in sc.contexts])
    res = linprog(-np.ones(len(G)), A_ub=M, b_ub=v, bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover
        raise RuntimeError(res.message)
    ncf = -res.fun
    y = -res.ineqlin.marginals
    ineq, k = {}, 0
    for C, B in zip(sc.contexts, blocks):
        ineq[C] = y[k:k + B.shape[0]]
        k += B.shape[0]
    return ContextualFraction(float(np.clip(1 - ncf, 0, 1)), res.x, ineq)


# ---------------------------------------------------------------------------
# full outcome-layer report
# ---------------------------------------------------------------------------
@dataclass
class OutcomeReport:
    support: SupportMap
    n_global_sections: int
    strongly_contextual: bool
    logical_witnesses: List[Tuple[Context, Section]]        # sections not extending globally
    cohomological_witnesses: List[Tuple[Context, Section]]  # γ(s) ≠ 0
    false_negatives: List[Tuple[Context, Section]]          # logical but γ(s) = 0
    contextual_fraction: Optional[float] = None
    signalling_defect: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def logically_contextual(self) -> bool:
        return bool(self.logical_witnesses)

    @property
    def cohomologically_contextual(self) -> bool:
        return bool(self.cohomological_witnesses)

    @property
    def gamma_h1_nonzero(self) -> bool:
        """The engine's headline flag: a nonzero class γ ∈ Ȟ¹ has been triggered."""
        return self.cohomologically_contextual

    def level(self) -> str:
        if self.strongly_contextual:
            return "strong"
        if self.logically_contextual:
            return "logical"
        if self.contextual_fraction is not None and self.contextual_fraction > 1e-9:
            return "probabilistic"
        return "noncontextual"

    def summary(self) -> str:
        cf = "n/a" if self.contextual_fraction is None else f"{self.contextual_fraction:.4f}"
        return (f"level={self.level():13s} CF={cf}  γ≠0 at {len(self.cohomological_witnesses)} sections"
                f"  (logical at {len(self.logical_witnesses)}, false-neg {len(self.false_negatives)}),"
                f"  #global sections={self.n_global_sections}")


def analyse_outcomes(model: EmpiricalModel, eps: float = 1e-12, with_cf: bool = True,
                     support: Optional[SupportMap] = None) -> OutcomeReport:
    """Run the whole outcome-layer stack on a model (or on an explicit support)."""
    sc = model.scenario
    supp = model.support(eps) if support is None else support
    gs = compatible_global_sections(sc, supp)
    logical, coh, fneg = [], [], []
    for C in sc.contexts:
        for s in supp[C]:
            ext = extends_to_global(sc, supp, C, s, gs)
            if not ext:
                logical.append((C, s))
                r = cohomological_obstruction(sc, supp, C, s)
                if r.obstruction_vanishes:
                    fneg.append((C, s))
                else:
                    coh.append((C, s))
            # (if s extends to a global section then γ(s)=0 automatically)
    cf = contextual_fraction(model).value if with_cf else None
    return OutcomeReport(supp, len(gs), len(gs) == 0, logical, coh, fneg, cf, model.signalling_defect())
