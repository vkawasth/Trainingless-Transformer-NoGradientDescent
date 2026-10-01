"""The presheaf of Markov kernels on a measurement cover: descent, constructive gluing by conditional
products, and where the obstruction lives.

Presheaf.  For a context C, K(C) = Delta(E(C)), the laws on joint outcomes of C (a kernel from the
one-point 'choose C' to E(C); with explicit settings it is a Bell box p(o | x), and restriction
marginalises OUTPUTS with the settings held fixed, which is well defined exactly on the no-signalling
boxes). Restriction along C' in C is marginalisation. This is Abramsky--Brandenburger's D_R . E.

Descent.  A family (e_C) is compatible iff it agrees on every overlap (no-signalling). It glues iff
some global law P on E(X) restricts to every e_C.

Constructive gluing.  The conditional product p_U (x) p_{V | U cap V} glues TWO compatible laws, always
(it exists for every compatible pair in FinStoch). Iterated along a running-intersection order of the
contexts it is exactly ONE sweep of iterative proportional fitting from the uniform law, and it glues
the whole family (Vorob'ev, constructively). On a cyclic cover the same sweep glues every context of
a spanning chain but not the closing one: conditional products always exist, so their existence is
not the gluing criterion; the cycle is.

Adjunction.  R : Delta(E(X)) -> Fam(U), P |-> (P|_C), induces image -| preimage on subsets
(R_! S subseteq T  iff  S subseteq R^* T). The unit is always an inclusion; the counit
R_! R^* T subseteq T is an equality iff T lies in the image of R (the non-contextual families). The
failure of the counit is the obstruction, measured by CF (distance to the image, an LP whose dual is a
Bell inequality). Linearising (signed measures) makes R surjective onto the compatible families, so
the counit becomes an isomorphism and the obstruction disappears: positivity carries it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scenario import EmpiricalModel, Scenario


def restriction_maps(sc: Scenario) -> Dict[tuple, np.ndarray]:
    X = sc.measurements
    return {C: sc.restriction_matrix(X, C) for C in sc.contexts}


def descent_defect(model: EmpiricalModel) -> float:
    """max over overlapping pairs of the TV distance between the two restrictions to C cap D."""
    sc = model.scenario; worst = 0.0
    for i, C in enumerate(sc.contexts):
        for D in sc.contexts[i + 1:]:
            S = sc.overlap(C, D)
            if not S:
                continue
            a = sc.restriction_matrix(C, S) @ model.tables[C]
            b = sc.restriction_matrix(D, S) @ model.tables[D]
            worst = max(worst, 0.5 * float(np.abs(a - b).sum()))
    return worst


def rip_order(contexts: Sequence[Sequence[str]]) -> Optional[List[tuple]]:
    """An order C_1..C_n with the running-intersection property (each C_k cap (C_1 u..u C_{k-1}) lies in
    one earlier C_j), found by GYO elimination; None if the cover is cyclic."""
    ctx = [tuple(c) for c in contexts]
    alive = {c: set(c) for c in ctx}; removed: List[tuple] = []
    changed = True
    while changed and len(alive) > 1:
        changed = False
        for c in list(alive):
            others = [alive[d] for d in alive if d != c]
            shared = {m for m in alive[c] if any(m in o for o in others)}
            if any(shared <= o for o in others):          # an ear: its shared part sits in one other edge
                removed.append(c); del alive[c]; changed = True
                break
    if len(alive) > 1:
        return None
    return list(alive) + removed[::-1]


def sweep_glue(model: EmpiricalModel, order: Sequence[Sequence[str]]) -> np.ndarray:
    """Iterated conditional product along `order`: P <- P (x) e_C over the overlap with what has been glued.
    With the uniform law on not-yet-seen measurements this is one IPF sweep."""
    sc = model.scenario; R = restriction_maps(sc)
    n = next(iter(R.values())).shape[1]
    P = np.full(n, 1.0 / n)
    for C in order:
        C = tuple(C); m = R[C] @ P
        ratio = np.divide(model.tables[C], m, out=np.zeros_like(m), where=m > 0)
        P = P * (R[C].T @ ratio)
    t = P.sum()
    return P / t if t > 0 else P


def gluing_defects(model: EmpiricalModel, P: np.ndarray) -> Dict[tuple, float]:
    """TV distance between each context table and the restriction of the candidate global law P."""
    R = restriction_maps(model.scenario)
    return {C: 0.5 * float(np.abs(R[C] @ P - model.tables[C]).sum()) for C in model.scenario.contexts}


def signed_extension(model: EmpiricalModel) -> Tuple[np.ndarray, float]:
    """least-squares signed global measure; residual 0 iff the family is compatible (linear descent)."""
    sc = model.scenario; R = restriction_maps(sc)
    A = np.vstack([R[C] for C in sc.contexts]); b = np.concatenate([model.tables[C] for C in sc.contexts])
    q, *_ = np.linalg.lstsq(A, b, rcond=None)
    return q, float(np.abs(A @ q - b).max())
