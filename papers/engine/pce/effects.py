"""Step 5 of "Probability, Context and Differentiation: The Layers": functions, duality and logs (module effects.py). New module.

States are laws p (arrays); effects are functions u on the same set; the pairing is <u, p> = sum u p. A kernel K (rows = inputs) acts on
states by p -> p K and on effects by u -> K u.

  push(K, p), pull(K, u)               the covariant and contravariant actions
  cond_effect(p, q)                    the update of p by an effect q >= 0: q p / <q, p> (None if <q, p> = 0)
  cond_exp(p, u, keep_mask_matrix)     E_p[u | X_S] as a function on X, through the marginalisation matrix M (rows = values of X_S)
  log_s(a, s)                          log a / log s (the log map at scale s)
  dual_cone_member(x, g)               g in C_x^*: constant on the support of x (gamma) and >= gamma on its empty cells
  cone_generators(x)                   lineality basis and extreme rays of C_x
"""
from __future__ import annotations
from fractions import Fraction as Fr
import math
import numpy as np


def pair(u, p): return sum(a * b for a, b in zip(u, p))


def push(K, p): return [sum(p[i] * K[i][j] for i in range(len(p))) for j in range(len(K[0]))]


def pull(K, u): return [sum(K[i][j] * u[j] for j in range(len(u))) for i in range(len(K))]


def compose(K, L): return [[sum(K[i][j] * L[j][k] for j in range(len(L))) for k in range(len(L[0]))] for i in range(len(K))]


def cond_effect(p, q):
    z = pair(q, p)
    if z == 0: return None
    return [a * b / z for a, b in zip(q, p)]


def cond_exp(p, u, M):
    """M: matrix (values of X_S) x (cells); returns E_p[u | X_S] as a function on cells (defined where the fibre has mass)"""
    out = [None] * len(p)
    for row in M:
        mass = sum(row[i] * p[i] for i in range(len(p)))
        if mass == 0: continue
        val = sum(row[i] * p[i] * u[i] for i in range(len(p))) / mass
        for i in range(len(p)):
            if row[i]: out[i] = val
    return out


def log_s(a, s): return math.log(a) / math.log(s)


def dual_cone_member(x, g, tol=1e-12):
    sup = [i for i in range(len(x)) if x[i] > 0]; gam = g[sup[0]]
    return all(abs(g[i] - gam) <= tol for i in sup) and all(g[i] >= gam - tol for i in range(len(x)) if x[i] == 0)


def cone_generators(x):
    n = len(x); I = np.eye(n); sup = [i for i in range(n) if x[i] > 0]; s0 = sup[0]
    return [I[s] - I[s0] for s in sup[1:]], [I[z] - I[s0] for z in range(n) if x[z] == 0]
