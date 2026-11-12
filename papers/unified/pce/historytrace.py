"""The history trace of an updater, and path discrepancy of kernel paths. New module; it modifies nothing (uses pce/kerneltransport.py).

On a fixed finite state space with strictly positive laws, every step an updater can take is multiplicative:
  a likelihood factor            P -> P L
  a Jeffrey update on context C  P -> P t(x_C) / P_C(x_C)              (a function of x_C)
  a refit onto data D_t          P -> Pi_{J_t}(P) = P exp(h_t), h_t in H_{D_t}   (Csiszar)
  a merge then reconstruction    P -> P(c, y) theta(x | c, y) = P theta(x | c, y) / P(x | c, y)
So after a final refit onto the current data D the updater holds Pi_J(P0 exp(tau)), tau = sum of the log factors, and only the class of
tau modulo H_D (constants plus functions of D's contexts) matters. The class is the history trace.

  span(ctxs, outs)          matrix whose columns span H for the given contexts (constants included)
  residual(f, B)            distance of f from the column span of B (least squares)
  quotient_dim(B, outs)     dim(F / H)
  step_* functions          one step of a history, each returning (new law, log factor)
  run(P0, steps)            apply a history; returns the pre-refit law and tau
  trace(R, R_honest)        log R - log R_honest (its class modulo H_D is the trace)
  dobrushin(K)              contraction coefficient of a kernel in total variation
  pathdisc(Ka, Kb, V)       max over the vertices V of TV(P Ka, P Kb) (as kerneltransport.hol)
"""
from __future__ import annotations
import numpy as np
from . import kerneltransport as KT


def span(ctxs, outs):
    cols = [np.ones(len(outs))]
    for ctx in ctxs:
        for lab in sorted({ctx(w) for w in outs}, key=repr):
            cols.append(np.array([1.0 if ctx(w) == lab else 0.0 for w in outs]))
    return np.array(cols).T


def residual(f, B):
    x = np.linalg.lstsq(B, f, rcond=None)[0]; return float(np.linalg.norm(B @ x - f))


def quotient_dim(B, outs): return len(outs) - int(np.linalg.matrix_rank(B))


def project_out(f, B):
    """the component of f orthogonal to span(B): a canonical representative of the class of f modulo H"""
    x = np.linalg.lstsq(B, f, rcond=None)[0]; return f - B @ x


def normalise(P): return P / P.sum()


def step_likelihood(P, L): return normalise(P * L), np.log(L)


def step_jeffrey(P, ctx, tab, outs):
    Q = KT.jeffrey(P, ctx, tab, outs); return Q, np.log(Q / P)


def step_refit(P, tables, outs):
    Q = KT.refit(P, tables, outs); return Q, np.log(Q / P)


def step_mergesplit(P, theta):
    Q = P @ KT.merge() @ KT.split(theta); return Q, np.log(Q / P)


def run(P0, steps):
    P = P0.copy(); tau = np.zeros(len(P0))
    for s in steps:
        P, lf = s(P); tau = tau + lf
    return P, tau


def trace(R, R_honest): return np.log(R) - np.log(R_honest)


def dobrushin(K):
    n = K.shape[0]; return max(0.5 * np.abs(K[i] - K[j]).sum() for i in range(n) for j in range(n))


def pathdisc(Ka, Kb, V): return max(KT.tv(P @ Ka, P @ Kb) for P in V)
