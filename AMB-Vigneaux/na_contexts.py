"""N/A as context restriction: the gated contextuality test for a labelled stream.

Records are windows of k tokens.  Each token is one of the labels
T, F, could-T, could-F, U, or N/A.  N/A means "this variable is not in the
context of this record", so every record measures the set of its non-N/A
positions, and the observed measured sets form a hypergraph on the variables.

Pipeline (the order is the point: the cheap, decisive test runs first):

  1. GATE — GYO reduction of the hypergraph of MAXIMAL observed contexts.
     Acyclic (reduces to nothing): by Vorob'ev every CONSISTENT family
     extends, so any CF > 0 is inconsistency between contexts (signalling),
     not contextuality.  Cyclic: there is a core on which contextuality is
     possible.
  2. POSSIBILISTIC — support sections (cells seen ≥ min_count times): logical
     contextuality and the AMB obstruction γ over a chosen coefficient ring.
  3. PROBABILISTIC — CF by LP, with the signalling defect reported beside it.
  4. NULL — permute the N/A mask across records, independently per variable
     (keeps each variable's N/A rate, destroys any structure), recompute CF.
     If the null's CF distribution reaches the data's, stop.

Notes.  Contexts are estimated from DIFFERENT records, so this is a
multi-source design and T0 does not force CF = 0; that is exactly why the
permutation null is needed — sampling alone produces a CF floor.  Records whose
measured set is a strict subset of another observed set are not used for the
maximal-context tables (they are counted and reported).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .outcome import analyse_outcomes, cohomological_obstruction
from .scenario import EmpiricalModel, Scenario

LABELS = ("T", "F", "could-T", "could-F", "U", "n/a")
NA = -1          # MASK: the variable is not in this record's context
NA_VALUE = 5     # VALUE: the variable was measured and the answer is "not applicable"
                 # (e.g. negation applied to U).  The two are kept distinct.


# ---------------------------------------------------------------- GYO
def gyo_reduce(edges: Sequence[FrozenSet]) -> List[FrozenSet]:
    """Graham–Yu–Özsoyoğlu reduction.  Returns the irreducible core; the
    hypergraph is (alpha-)acyclic iff the core is empty."""
    E = [frozenset(e) for e in edges if e]
    changed = True
    while changed:
        changed = False
        # remove vertices that occur in exactly one edge
        count: Dict = {}
        for e in E:
            for v in e:
                count[v] = count.get(v, 0) + 1
        E2 = [frozenset(v for v in e if count[v] > 1) for e in E]
        if E2 != E:
            changed = True
        E = [e for e in E2 if e]
        # remove edges contained in another edge (keep one copy of duplicates)
        keep = []
        for i, e in enumerate(E):
            if any((e < f) or (e == f and j < i) for j, f in enumerate(E) if j != i):
                changed = True
                continue
            keep.append(e)
        E = keep
    return E


def is_acyclic(edges: Sequence[FrozenSet]) -> bool:
    return len(gyo_reduce(edges)) == 0


# ---------------------------------------------------------------- stream -> contexts
def measured_sets(records: np.ndarray) -> Dict[FrozenSet[int], int]:
    out: Dict[FrozenSet[int], int] = {}
    for r in records:
        s = frozenset(int(i) for i in np.nonzero(r != NA)[0])
        out[s] = out.get(s, 0) + 1
    return out


def maximal(sets: Sequence[FrozenSet]) -> List[FrozenSet]:
    return [s for s in sets if s and not any(s < t for t in sets)]


def empirical_from_records(records: np.ndarray, n_labels: int = 5) -> Tuple[EmpiricalModel, dict]:
    """Scenario = maximal observed contexts; each table counts the records whose
    measured set is exactly that context."""
    k = records.shape[1]
    ms = measured_sets(records)
    ctx = sorted(maximal(list(ms)), key=lambda s: sorted(s))
    names = [f"x{i}" for i in range(k)]
    used = sorted(set().union(*ctx)) if ctx else []
    sc = Scenario({names[i]: tuple(range(n_labels)) for i in used}, tuple(tuple(names[i] for i in sorted(C)) for C in ctx))
    counts = {C: np.zeros(sc.n_sections(C)) for C in sc.contexts}
    pos = {C: {s: j for j, s in enumerate(sc.sections(C))} for C in sc.contexts}
    by_set = {frozenset(int(n[1:]) for n in C): C for C in sc.contexts}
    unused = 0
    for r in records:
        s = frozenset(int(i) for i in np.nonzero(r != NA)[0])
        if s in by_set:
            C = by_set[s]
            counts[C][pos[C][tuple(int(r[int(n[1:])]) for n in C)]] += 1
        else:
            unused += 1
    info = dict(measured_sets={tuple(sorted(s)): c for s, c in ms.items()},
                maximal=[tuple(sorted(C)) for C in ctx], unused_records=unused)
    return EmpiricalModel.from_counts(sc, counts), info


# ---------------------------------------------------------------- the pipeline
@dataclass
class StreamVerdict:
    hypergraph: List[Tuple[int, ...]]
    gyo_core: List[Tuple[int, ...]]
    acyclic: bool
    unused_records: int
    cf: Optional[float] = None
    signalling: Optional[float] = None
    logical_witnesses: Optional[int] = None
    gamma_nonzero: Optional[int] = None
    support_full: Optional[bool] = None
    null_cf: List[float] = field(default_factory=list)
    deficit: Optional[float] = None
    deficit_split: Optional[Dict] = None
    p_value: Optional[float] = None
    verdict: str = ""


def missing_data_em(records: np.ndarray, n_labels: int, iters: int = 500, tol: float = 1e-10,
                    init: Optional[np.ndarray] = None) -> np.ndarray:
    """Maximum-likelihood global law P on E(X) from incomplete records
    (Dempster–Laird–Rubin EM; each record is observed on its non-masked
    positions).  This is the best CONSISTENT explanation of the stream."""
    k = records.shape[1]
    cells = np.array(list(itertools.product(range(n_labels), repeat=k)))
    groups: Dict[Tuple, int] = {}
    for r in records:
        key = tuple(int(v) for v in r)
        groups[key] = groups.get(key, 0) + 1
    match = []
    for key, c in groups.items():
        obs = [j for j in range(k) if key[j] != NA]
        m = np.all(cells[:, obs] == np.array([key[j] for j in obs]), axis=1) if obs else np.ones(len(cells), bool)
        match.append((m, c))
    P = np.full(len(cells), 1.0 / len(cells)) if init is None else np.asarray(init, float) / np.sum(init)
    for _ in range(iters):
        acc = np.zeros(len(cells))
        for m, c in match:
            w = P * m
            z = w.sum()
            if z > 0:
                acc += c * w / z
        new = acc / acc.sum()
        if np.abs(new - P).max() < tol:
            P = new
            break
        P = new
    return P


def bootstrap_records(records: np.ndarray, P: np.ndarray, n_labels: int, rng: np.random.Generator) -> np.ndarray:
    """Keep every record's mask; redraw its values from the consistent law P."""
    k = records.shape[1]
    cells = np.array(list(itertools.product(range(n_labels), repeat=k)))
    draw = cells[rng.choice(len(cells), size=len(records), p=P)]
    return np.where(records == NA, NA, draw)


def permute_mask(records: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Null: shuffle each variable's N/A indicator across records (same rate,
    no structure).  Values stay attached to their record."""
    full = records.copy()
    out = records.copy()
    for j in range(records.shape[1]):
        mask = records[:, j] == NA
        perm = rng.permutation(len(records))
        newmask = mask[perm]
        # a variable that becomes observed needs a value: borrow one from the same
        # record's original value if present, else from a random observed record
        obs_vals = full[~mask, j]
        vals = np.where(full[:, j] != NA, full[:, j], rng.choice(obs_vals, size=len(records)))
        out[:, j] = np.where(newmask, NA, vals)
    return out


def _support(model: EmpiricalModel, n_by_ctx: Dict, min_frac: float):
    sc = model.scenario
    return {C: [s for s, pr in zip(sc.sections(C), model.tables[C]) if pr > 0 and pr >= min_frac]
            for C in sc.contexts}


def analyse_stream(records: np.ndarray, rng: Optional[np.random.Generator] = None, n_null: int = 30,
                   min_frac: float = 0.0, ring="Z", n_labels: int = 6, force: bool = False) -> StreamVerdict:
    """min_frac: a cell is in the possibilistic support if it holds at least this
    fraction of its context's records (0 = any observation; the raw support)."""
    from .outcome import contextual_fraction
    rng = rng or np.random.default_rng(0)
    model, info = empirical_from_records(records, n_labels)
    hyper = [frozenset(C) for C in info["maximal"]]
    core = gyo_reduce(hyper)
    v = StreamVerdict([tuple(sorted(h)) for h in hyper], [tuple(sorted(c)) for c in core], not core,
                      info["unused_records"])
    if v.acyclic and not force:
        rep = analyse_outcomes(model, with_cf=True)
        v.cf, v.signalling = rep.contextual_fraction, rep.signalling_defect
        v.verdict = ("ACYCLIC cover: every consistent family extends (Vorob'ev). "
                     f"CF = {v.cf:.4f} can only be inconsistency (signalling {v.signalling:.4f}). AMB half empty.")
        return v
    sc = model.scenario
    support = _support(model, None, min_frac)
    rep = analyse_outcomes(model, support=support, with_cf=True)
    v.cf, v.signalling = rep.contextual_fraction, rep.signalling_defect
    v.logical_witnesses = len(rep.logical_witnesses)
    v.support_full = all(len(support[C]) == sc.n_sections(C) for C in sc.contexts)
    C0 = sc.contexts[0]
    v.gamma_nonzero = sum(not cohomological_obstruction(sc, support, C0, s, ring).obstruction_vanishes
                          for s in support[C0])
    # null: same masks, values redrawn from the ML consistent global law (ridge pinned)
    P = pinned_consistent_law(records, n_labels)
    v.deficit, v.deficit_split = deficit(records, n_labels)
    null_logical = 0
    for _ in range(n_null):
        m0, _ = empirical_from_records(bootstrap_records(records, P, n_labels, rng), n_labels)
        v.null_cf.append(contextual_fraction(m0).value)
        null_logical += len(analyse_outcomes(m0, support=_support(m0, None, min_frac), with_cf=False).logical_witnesses) > 0
    v.p_value = float((1 + sum(c >= v.cf for c in v.null_cf)) / (1 + len(v.null_cf)))
    stop = v.p_value > 0.05
    v.verdict = (f"CYCLIC core {v.gyo_core}. CF = {v.cf:.4f} (signalling {v.signalling:.4f}); "
                 f"null CF from the ML consistent law: mean {np.mean(v.null_cf):.4f}, max {max(v.null_cf):.4f}, "
                 f"p = {v.p_value:.3f}. " + ("STOP: indistinguishable from the null." if stop else "Above the null.")
                 + f"  Deficit of the best consistent law {v.deficit:.4f} nats/record, split "
                 + ", ".join(f"{''.join(c[1:] for c in C)}:{x:.3f}" for C, x in v.deficit_split.items()) + "."
                 + f"  Possibilistic at a SINGLE threshold (support ≥ {min_frac:g}; use threshold_sweep before reporting γ): {v.logical_witnesses} logical witnesses "
                 f"(null: {null_logical}/{n_null} replicates), γ≠0 at {v.gamma_nonzero} sections of {C0} over "
                 f"{ring}; support {'full' if v.support_full else 'sparse'}.")
    return v


# ---------------------------------------------------------------- synthetic streams
def chain_record(rng, k, eps, n_labels=5):
    """One latent window: x0 uniform, each next token copies the previous one,
    replaced by a uniform label with probability eps."""
    x = [int(rng.integers(n_labels))]
    for _ in range(k - 1):
        x.append(int(rng.integers(n_labels)) if rng.random() < eps else x[-1])
    return x


NEGATE = {0: 1, 1: 0, 2: 3, 3: 2, 4: NA_VALUE, NA_VALUE: NA_VALUE}   # U -> n/a; n/a absorbing


def structured_stream(n: int, rng: np.random.Generator, eps: float = 0.1, k: int = 3,
                      twist: str = "shift", n_labels: int = 5) -> np.ndarray:
    """(values are in 0..4; the negation twist can also emit the n/a value 5)"""
    """POSITIVE CONTROL.  Records measure a cyclic cover (one N/A per record for
    k = 3: contexts {x0,x1}, {x1,x2}, {x0,x2}).  The records that skip x1 come
    from a different process, in which x2 is a twisted copy of x0 (shift by 2,
    or negation).  The N/A pattern therefore carries the context."""
    out = np.full((n, k), NA)
    for r in range(n):
        which = r % k                       # which variable is N/A
        x = chain_record(rng, k, eps, n_labels)
        if which == 1:                      # the twisted context {x0, x2}
            base = x[0]
            tw = (base + 2) % n_labels if twist == "shift" else NEGATE[base]
            x[2] = int(rng.integers(n_labels)) if rng.random() < eps else tw
        row = list(x)
        row[which] = NA
        out[r] = row
    return out


def honest_stream(n: int, rng: np.random.Generator, eps: float = 0.1, k: int = 3,
                  na_rate: float = 0.3, one_na: bool = True, n_labels: int = 5) -> np.ndarray:
    """NULL.  One latent process for every record; N/A drawn at random, independent
    of the values.  one_na=True forces exactly one N/A per record (cyclic cover);
    otherwise each position is N/A independently (complete records appear, and
    the cover becomes acyclic)."""
    out = np.empty((n, k), dtype=int)
    for r in range(n):
        x = chain_record(rng, k, eps, n_labels)
        if one_na:
            x[int(rng.integers(k))] = NA
        else:
            for j in range(k):
                if rng.random() < na_rate:
                    x[j] = NA
        out[r] = x
    return out


# ---------------------------------------------------------------- pinned null, deficit, threshold sweep
def pinned_consistent_law(records: np.ndarray, n_labels: int, iters: int = 800,
                          init: Optional[np.ndarray] = None) -> np.ndarray:
    """ML consistent law with the ridge PINNED: fit by missing-data EM, then return
    the maximum-entropy law with the same observed-context marginals (IPF from
    uniform).  Different EM starts give different points on the flat ridge of
    unobserved higher-order interaction; the pinned law does not depend on the start.
    Needed before contexts larger than the interactions they can identify are
    used in a null."""
    k = records.shape[1]
    P = missing_data_em(records, n_labels, iters, init=init)
    cells = np.array(list(itertools.product(range(n_labels), repeat=k)))
    ctx = [sorted(s) for s in maximal(list(measured_sets(records)))]
    targets = []
    for C in ctx:
        idx = np.ravel_multi_index(cells[:, C].T, (n_labels,) * len(C))
        targets.append((idx, np.bincount(idx, weights=P, minlength=n_labels ** len(C))))
    Q = np.full(len(cells), 1.0 / len(cells))
    for _ in range(2000):
        for idx, t in targets:
            m = np.bincount(idx, weights=Q, minlength=len(t))
            Q = Q * np.divide(t, m, out=np.zeros_like(t), where=m > 0)[idx]
        if max(np.abs(np.bincount(idx, weights=Q, minlength=len(t)) - t).max() for idx, t in targets) < 1e-12:
            break
    return Q / Q.sum()


def deficit(records: np.ndarray, n_labels: int = 6) -> Tuple[float, Dict]:
    """Per-record log-likelihood lost by the best CONSISTENT law relative to the
    per-context fit, and its split by context (KL(data_C ‖ fitted_C)).  A KL
    distance to the non-contextual set: zero exactly when CF is (for consistent
    tables), but NOT a function of CF (on noisy-shift cycles at CF = 0.5 it is
    0.101, 0.068, 0.051 for L = 3, 4, 5; at ε = 0 it equals log(L/(L−1)))."""
    k = records.shape[1]
    P = missing_data_em(records, n_labels)
    cells = np.array(list(itertools.product(range(n_labels), repeat=k)))
    model, _ = empirical_from_records(records, n_labels)
    tot, n, split = 0.0, 0, {}
    for C in model.scenario.contexts:
        cols = [int(x[1:]) for x in C]
        fit = np.bincount(np.ravel_multi_index(cells[:, cols].T, (n_labels,) * len(cols)),
                          weights=P, minlength=n_labels ** len(cols))
        nc = sum(1 for r in records if frozenset(np.nonzero(r != NA)[0]) == frozenset(cols))
        e = model.tables[C]
        msk = e > 0
        kl = float((e[msk] * np.log(e[msk] / np.maximum(fit[msk], 1e-300))).sum())
        split[C] = kl
        tot += kl * nc
        n += nc
    return tot / max(n, 1), split


def threshold_sweep(records: np.ndarray, thresholds=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15),
                    n_null: int = 20, n_labels: int = 6, rng: Optional[np.random.Generator] = None):
    """Possibilistic test across support thresholds WITH its null band.  Returns
    rows (threshold, data logical witnesses, γ≠0 sections, null replicates with a
    logical witness).  Report the whole table: a γ claim is admissible only at
    thresholds where the null band is ~0 (near a cell's expected mass, sampling
    flips cells in and out of the support and manufactures logical witnesses)."""
    rng = rng or np.random.default_rng(0)
    P = pinned_consistent_law(records, n_labels)
    nulls = [bootstrap_records(records, P, n_labels, rng) for _ in range(n_null)]

    def possib(rec, t):
        m, _ = empirical_from_records(rec, n_labels)
        sc = m.scenario
        sup = {C: [s for s, pr in zip(sc.sections(C), m.tables[C]) if pr > 0 and pr >= t] for C in sc.contexts}
        if any(len(v) == 0 for v in sup.values()):
            return None, None
        r = analyse_outcomes(m, support=sup, with_cf=False)
        C0 = sc.contexts[0]
        g = sum(not cohomological_obstruction(sc, sup, C0, s).obstruction_vanishes for s in sup[C0])
        return len(r.logical_witnesses), g

    rows = []
    for t in thresholds:
        lw, g = possib(records, t)
        nl = sum(1 for nr in nulls if (possib(nr, t)[0] or 0) > 0)
        rows.append((t, lw, g, nl))
    return rows


def population_deficit(model: EmpiricalModel, iters: int = 3000, tol: float = 1e-13) -> float:
    """Population version of the deficit for an empirical model: min over global
    laws P of the context-averaged KL(e_C ‖ P_C), by EM with the tables as
    fractional records."""
    sc = model.scenario
    X = sc.measurements
    R = {C: sc.restriction_matrix(X, C) for C in sc.contexts}
    nG = R[sc.contexts[0]].shape[1]
    w = 1.0 / len(sc.contexts)
    P = np.full(nG, 1.0 / nG)
    for _ in range(iters):
        acc = np.zeros(nG)
        for C in sc.contexts:
            m = R[C] @ P
            ratio = np.divide(model.tables[C], m, out=np.zeros_like(m), where=m > 0)
            acc += w * P * (R[C].T @ ratio)
        new = acc / acc.sum()
        if np.abs(new - P).max() < tol:
            P = new
            break
        P = new
    d = 0.0
    for C in sc.contexts:
        e, q = model.tables[C], R[C] @ P
        msk = e > 0
        d += w * float((e[msk] * np.log(e[msk] / q[msk])).sum())
    return d
