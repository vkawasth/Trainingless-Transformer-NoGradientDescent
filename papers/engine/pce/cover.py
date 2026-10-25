"""Gluing over a cover of contexts: scopes, context-dependent senses, competence as tolerance, graded gluing, holes, and robust actions.

An expert is a section over its own patch of context space. In each context the restriction maps tie its window to the world's
variables; where two experts' windows overlap the maps must agree, and resolving all these constraints at once is the gluing.

  Expert(name, vars, table, scope, senses)
      vars     the names the expert speaks about; table: its asserted law on them
      scope    the contexts in which its statements are meant to hold (its patch of the cover)
      senses   {context: {expert name -> world variable or None}}: what each of its words refers to in that context
               (None = the word has no referent there). Default: every word means the world variable of the same name.
  active(experts, c)                 the experts whose scope contains c, with their windows re-routed through the senses of c
  region_map(experts, contexts, query_vars)   per context: active experts, overlaps (shared world variables), and holes
               (query variables no active expert covers)
  competence(expert, truth_marginal, n, rng)  probe estimate of the expert's error: total variation between its table and an
               empirical sample of n cases with known answers
  glue_soft(domains, windows, eps)   theta* = smallest common scale such that some law q has TV(marginal of q, window) <= theta eps_i for
               every window (an LP). theta* = 0: exact gluing; theta* <= 1: glues within the tolerances; theta* > 1: obstruction
               of size theta* tolerances. Also returns lam* and the place layer of Section 13.
  robust_action(domains, windows, eps, utilities, evidence)   Gamma-maximin: for each action the smallest conditional expected
               utility over all laws within the tolerances (linear-fractional LP), and the action that maximises it
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog
from .coherence import Window, _constraints
from .integrate import gluable_fraction
from .places import layer


@dataclass
class Expert:
    name: str
    vars: tuple
    table: np.ndarray
    scope: set
    senses: dict = field(default_factory=dict)

    def window(self, c):
        """the expert's window in context c, or None if one of its words has no referent there."""
        m = self.senses.get(c, {})
        names = tuple(m.get(v, v) for v in self.vars)
        return None if any(n is None for n in names) else Window(names, np.asarray(self.table, float))


def active(experts, c):
    return [(e, e.window(c)) for e in experts if c in e.scope and e.window(c) is not None]


def region_map(experts, contexts, query_vars):
    out = {}
    for c in contexts:
        act = active(experts, c)
        cover = {v for _, w in act for v in w.vars}
        overlaps = {(a.name, b.name): sorted(set(wa.vars) & set(wb.vars))
                    for (a, wa), (b, wb) in itertools.combinations(act, 2) if set(wa.vars) & set(wb.vars)}
        out[c] = {"active": [e.name for e, _ in act], "talking_out_of_scope": [e.name for e in experts if c not in e.scope],
                  "overlaps": overlaps, "holes": sorted(set(query_vars) - cover)}
    return out


def competence(expert_table, truth_marginal, n, rng):
    """estimated total-variation error from n probe cases drawn from the true marginal of the expert's variables."""
    p = np.asarray(truth_marginal, float).ravel(); idx = rng.choice(len(p), n, p=p / p.sum())
    emp = np.bincount(idx, minlength=len(p)) / n
    return 0.5 * float(np.abs(np.asarray(expert_table, float).ravel() - emp).sum())


def glue_soft(domains, windows, eps):
    """theta*, a law attaining it, lam*, and the place layer of the family."""
    A, t, atoms, index = _constraints(domains, windows)
    n, m, k = A.shape[1], A.shape[0], len(windows)
    # variables: q (n), s+ (m), s- (m), theta (1)
    nv = n + 2 * m + 1; c = np.zeros(nv); c[-1] = 1.0
    Aeq = np.zeros((m + 1, nv)); Aeq[:m, :n] = A; Aeq[:m, n:n + m] = -np.eye(m); Aeq[:m, n + m:n + 2 * m] = np.eye(m)
    Aeq[m, :n] = 1.0; beq = np.concatenate([t, [1.0]])
    Aub = np.zeros((k, nv))
    for r, (wi, _) in enumerate(index):
        Aub[wi, n + r] = 0.5; Aub[wi, n + m + r] = 0.5
    for wi in range(k): Aub[wi, -1] = -max(eps[wi], 1e-12)
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(k), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    lam = gluable_fraction(domains, windows).lam
    lay = layer(domains, windows)[0]
    return {"theta": float(r.fun), "q": r.x[:n], "atoms": atoms, "lam": lam, "layer": lay}


def _feasible_set(domains, windows, eps):
    """constraints (A_ub, b_ub, A_eq, b_eq) in the variables (q, s+, s-) for the set of laws within the tolerances."""
    A, t, atoms, index = _constraints(domains, windows)
    n, m, k = A.shape[1], A.shape[0], len(windows)
    nv = n + 2 * m
    Aeq = np.zeros((m, nv)); Aeq[:, :n] = A; Aeq[:, n:n + m] = -np.eye(m); Aeq[:, n + m:] = np.eye(m)
    Aub = np.zeros((k, nv)); bub = np.array([e for e in eps], float)
    for r, (wi, _) in enumerate(index):
        Aub[wi, n + r] = 0.5; Aub[wi, n + m + r] = 0.5
    return A, t, atoms, Aub, bub, Aeq


def robust_action(domains, windows, eps, utilities, evidence=None):
    """utilities: {action: function(atom dict) -> utility}. evidence: {var: value} or None.
    Returns ({action: worst-case conditional expected utility}, best action). Laws are scaled (Charnes-Cooper): y = s q, P(evidence) = 1."""
    A, t, atoms, Aub, bub, Aeq = _feasible_set(domains, windows, eps)
    names = list(domains); n = A.shape[1]; m = A.shape[0]; nv = n + 2 * m
    inE = np.array([1.0 if evidence is None or all(a[names.index(v)] == val for v, val in evidence.items()) else 0.0 for a in atoms])
    vals = {}
    for act, u in utilities.items():
        uu = np.array([u(dict(zip(names, a))) if inE[k] else 0.0 for k, a in enumerate(atoms)])
        # variables (y, s+, s-, s): y = s q; equality A y - t s = s+ - s-; sum y = s; sum_{E} y = 1; TV constraints scaled by s
        c = np.concatenate([uu, np.zeros(2 * m), [0.0]])
        AeqC = np.zeros((m + 2, nv + 1)); AeqC[:m, :nv] = Aeq; AeqC[:m, -1] = -t
        AeqC[m, :n] = 1.0; AeqC[m, -1] = -1.0; AeqC[m + 1, :n] = inE
        beqC = np.concatenate([np.zeros(m + 1), [1.0]])
        AubC = np.hstack([Aub, -bub[:, None]])
        r = linprog(c, A_ub=AubC, b_ub=np.zeros(len(bub)), A_eq=AeqC, b_eq=beqC, bounds=(0, None), method="highs")
        vals[act] = float(r.fun) if r.status == 0 else float("nan")
    best = max(vals, key=lambda a: -np.inf if np.isnan(vals[a]) else vals[a])
    return vals, best


def expected_utility(P, names, utilities, evidence=None):
    """conditional expected utility of each action under a law P (array over names)."""
    out = {}
    for act, u in utilities.items():
        num = den = 0.0
        for a in itertools.product(*[range(s) for s in P.shape]):
            x = dict(zip(names, a))
            if evidence and any(x[v] != val for v, val in evidence.items()): continue
            num += P[a] * u(x); den += P[a]
        out[act] = num / den if den > 0 else float("nan")
    return out
