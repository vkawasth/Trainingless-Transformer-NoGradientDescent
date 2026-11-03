"""An observed environment beside the latent source (layers paper, Section "An observed environment"). New module. Exact arithmetic.

A record carries a latent source l in L and an observed environment e in E. The law on L x E x X is
    Lambda(l, e, x) = rho_e * pi_{l|e} * p_{l,e}(x).
Observing e means conditioning on it; forgetting e means marginalising it out. Only a forgotten coordinate is reweighted by Bayes.

  joint(rho, pi, p)                Lambda as a dict {(l, e, x): mass}
  forget(Lam, keep)                push Lambda forward to the kept coordinates ('le', 'l', 'e', '') with x always kept
  condition(Lam, A)                C_{L x E x A} Lambda
  per_env_weights(Lam, A)          pi^{A,e}_l for each observed e
  env_posterior(Lam, A)            rho^A_e, the reweighting of a forgotten environment
  rates(Lam, A)                    p_{l,e}(A) and the pooled p_l(A) = sum_e rho_{e|l} p_{l,e}(A)
  pooled_order(m, r)               v_l = min_e (r_e + m_{l,e}): the order of source l on A once the environment is forgotten
"""
from __future__ import annotations
from fractions import Fraction as Fr


def joint(rho, pi, p):
    return {(l, e, x): Fr(rho[e]) * Fr(pi[e][l]) * Fr(p[(l, e)][x]) for e in rho for l in pi[e] for x in p[(l, e)]}


def forget(Lam, keep):
    out = {}
    for (l, e, x), v in Lam.items():
        k = (l if "l" in keep else None, e if "e" in keep else None, x); out[k] = out.get(k, 0) + v
    return out


def condition(Lam, A):
    tot = sum(v for (l, e, x), v in Lam.items() if x in A)
    return {k: v / tot for k, v in Lam.items() if k[2] in A}


def per_env_weights(Lam, A):
    out = {}
    for (l, e, x), v in Lam.items():
        if x in A: out.setdefault(e, {}); out[e][l] = out[e].get(l, 0) + v
    return {e: {l: v / sum(d.values()) for l, v in d.items()} for e, d in out.items()}


def env_posterior(Lam, A):
    m = {}
    for (l, e, x), v in Lam.items():
        if x in A: m[e] = m.get(e, 0) + v
    tot = sum(m.values()); return {e: v / tot for e, v in m.items()}


def rates(Lam, A):
    num, den = {}, {}
    for (l, e, x), v in Lam.items():
        den[(l, e)] = den.get((l, e), 0) + v
        if x in A: num[(l, e)] = num.get((l, e), 0) + v
    per = {k: num.get(k, 0) / den[k] for k in den}
    Ls = {l for l, e in den}; pooled = {}
    for l in Ls:
        tl = sum(den[k] for k in den if k[0] == l); pooled[l] = sum(num.get(k, 0) for k in den if k[0] == l) / tl
    return per, pooled


def pooled_order(m, r):
    Ls = {l for l, e in m}; return {l: min(r[e] + m[(l2, e)] for (l2, e) in m if l2 == l) for l in Ls}
