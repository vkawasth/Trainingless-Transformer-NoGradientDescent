"""Decision value of the braid under credal sets. New module; it modifies nothing.

Hypotheses K in {honest, coordinated}; actions flag, accept, inspect; utilities u(flag, honest) = -c_fp, u(accept, coordinated) = -c_fn,
u(inspect, .) = -c_in, others 0. Laws as in credalbraid (brokerage histories, m = 2): honest p_lam, coordinated q_{lam', beta}, prior pi.

  joint_laws(H, lams_h, lams_c, betas, shared, prior)   extreme joint laws on (K, history): arrays of shape (2, |H|)
  expost(lo, hi, costs)        per observed value: Gamma-maximin action and value from the posterior interval [lo, hi] of 'coordinated';
                               E-admissible action set
  exante(Pi, m, laws, costs)   normal-form Gamma-maximin value: max over randomised policies delta(a | y) of the minimum over extreme laws
                               of expected utility (a linear programme)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import credalbraid as CB

ACTIONS = ("flag", "accept", "inspect")


def utility(costs):
    c_fp, c_fn, c_in = costs
    return {"flag": (-c_fp, 0.0), "accept": (0.0, -c_fn), "inspect": (-c_in, -c_in)}      # (honest, coordinated)


def joint_laws(H, lams_h, lams_c, betas, shared, prior=0.5):
    out = []
    if shared:
        for lam in lams_h:
            for b in betas:
                p, q = CB.laws(H, lam, b); out.append(np.vstack([(1 - prior) * p, prior * q]))
    else:
        for lh in lams_h:
            p, _ = CB.laws(H, lh, 0.0)
            for lc in lams_c:
                for b in betas:
                    _, q = CB.laws(H, lc, b); out.append(np.vstack([(1 - prior) * p, prior * q]))
    return out


def posterior_interval(Pi, m, laws):
    post = []
    for L in laws:
        a = CB.push(L[1], Pi, m)[0]; h = CB.push(L[0], Pi, m)[0]; post.append(a / (a + h))
    post = np.array(post); return post.min(0), post.max(0)


def expost(lo, hi, costs):
    U = utility(costs); acts = []; vals = []; adm = []
    for l, h in zip(lo, hi):
        worst = {a: min((1 - q) * U[a][0] + q * U[a][1] for q in (l, h)) for a in ACTIONS}     # linear in q: extremes at ends
        best = max(worst, key=worst.get); acts.append(best); vals.append(worst[best])
        E = set()
        for q in np.linspace(l, h, 201):
            ex = {a: (1 - q) * U[a][0] + q * U[a][1] for a in ACTIONS}; mx = max(ex.values()); E |= {a for a in ACTIONS if ex[a] >= mx - 1e-12}
        adm.append(tuple(sorted(E)))
    return acts, np.array(vals), adm


def exante(Pi, m, laws, costs):
    U = utility(costs); A = len(ACTIONS); nv = m * A + 1
    c = np.zeros(nv); c[-1] = -1.0
    A_ub = []; b_ub = []
    for L in laws:
        ph = CB.push(L[0], Pi, m)[0]; pc = CB.push(L[1], Pi, m)[0]; row = np.zeros(nv)
        for y in range(m):
            for ai, a in enumerate(ACTIONS): row[y * A + ai] = -(ph[y] * U[a][0] + pc[y] * U[a][1])
        row[-1] = 1.0; A_ub.append(row); b_ub.append(0.0)
    A_eq = np.zeros((m, nv)); b_eq = np.ones(m)
    for y in range(m): A_eq[y, y * A:(y + 1) * A] = 1.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (nv - 1) + [(None, None)], method="highs")
    return float(-res.fun)
