"""The gluing lattice of a presheaf on a finite cover.

Given contexts C_1..C_n (e.g. targets, topics, outlets) and a test glues(S) -> p-value for "the local data on
the sub-cover S admit a global section" (for bias: the additive model 'lean + context effect' fits, i.e. every
loop inside S closes), the gluing lattice is the family of sub-covers that glue.

Exactly (without noise) gluing is DOWNWARD CLOSED: a global section on S restricts to one on any S' in S.
So the lattice is determined by its MAXIMAL glueable sub-covers, and the obstruction by its MINIMAL
non-glueable sub-covers (the minimal obstructed regions). Statistically, monotonicity can fail at the margin;
we report the number of violations (a subset rejected while a superset is accepted).
"""
from __future__ import annotations
import itertools


def lattice(contexts, glues, alpha=0.05, min_size=2, max_size=None, subsets=None):
    contexts = list(contexts)
    max_size = max_size or len(contexts)
    if subsets is None:
        subsets = [frozenset(c) for r in range(min_size, max_size + 1) for c in itertools.combinations(contexts, r)]
    res = {S: glues(S) for S in subsets}
    ok = {S for S, p in res.items() if p > alpha}
    bad = set(res) - ok
    maximal = [S for S in ok if not any(S < T for T in ok)]
    minimal_bad = [S for S in bad if not any(T < S for T in bad)]
    violations = sum(1 for S in bad for T in ok if S < T)
    return dict(p={" | ".join(sorted(S)): p for S, p in res.items()},
                maximal_glueable=[tuple(sorted(S)) for S in sorted(maximal, key=len, reverse=True)],
                minimal_obstructed=[tuple(sorted(S)) for S in sorted(minimal_bad, key=len)],
                n_tested=len(res), n_glue=len(ok), monotonicity_violations=violations)


def greedy_maximal(contexts, glues, alpha=0.05, starts=None):
    """for large n: grow glueable sets greedily from each start (adds the context with the largest p that
    keeps p > alpha); returns the distinct maximal sets found (a lower bound on the maximal family)."""
    contexts = list(contexts); found = set()
    for s in (starts or contexts):
        S = frozenset([s])
        while True:
            cand = [(glues(S | {c}), c) for c in contexts if c not in S]
            cand = [(p, c) for p, c in cand if p > alpha]
            if not cand:
                break
            S = S | {max(cand)[1]}
        found.add(S)
    return [tuple(sorted(S)) for S in sorted(found, key=len, reverse=True)]
