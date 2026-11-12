"""Outer reconciliation and the relay clock. New module; it modifies nothing (uses the Study 26 world, engine/study26_e2e/world26.py).

Outer reconciliation. Local data are exact tables on a cover; outer data add tables on further contexts. Extension of the local section
to the outer cover is feasibility of {p >= 0 : M p = d}; its failure has a Farkas certificate y with M^T y >= 0 and d^T y < 0, which is a
cover-additive function, non-negative on every state, whose expectation under the reported tables is negative.
  stack(blocks)               (M, d) from a list of (context, table)
  extends(blocks)             feasibility
  certificate(blocks)         a Farkas certificate (None if feasible), normalised so that d^T y = -1, entries bounded by 1e3
  minimal_infeasible(local, outer)   all minimal subsets W of the outer blocks for which local + W is infeasible
  rho(h_ctx, W_ctx, g, tau, P)        separation margin min ||M_W p - M_W P||_1 over p matching P on h_ctx with g(p) >= tau
  far_side_config(h_ctx, C_ctx, g, tau, P, rng)   a random coherent coalition configuration with g >= tau (a mixture of LP vertices)

The relay clock. Sources are vertices of a co-revision graph; an event is a signed edge ((i, j), s); two events are independent when their
edges share no vertex.
  layers(word)                Cartier-Foata layering: layer of an event = 1 + the largest layer of an earlier dependent event
  activation(word, start)     activation round of every source reached from 'start' by dependence chains (round 0 for start)
  reduce(word)                normal form in the right-angled Artin group: cancel an event against a later reversal when every event
                              between is independent of it, repeat; then sort each Cartier-Foata layer
  flip_bound(a, delta, Dfun)  T* = least T with sum_{j active by T} delta_j (T - a_j) >= D_{A(T)}
"""
from __future__ import annotations
import itertools, os, sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study26_e2e"))
import world26 as W

S, V = W.STATES, W.VARS; NS = len(S)


def ctxmat(ctx):
    idx = [V.index(v) for v in ctx]; A = np.zeros((2 ** len(ctx), NS))
    for i, s in enumerate(S): A[int("".join(str(s[j]) for j in idx), 2), i] = 1
    return A


def ev(pred): return np.array([1.0 if pred(dict(zip(V, s))) else 0.0 for s in S])


def stack(blocks):
    return np.vstack([ctxmat(c) for c, _ in blocks]), np.concatenate([np.asarray(t, float) for _, t in blocks])


def extends(blocks):
    M, d = stack(blocks); return linprog(np.zeros(NS), A_eq=M, b_eq=d, bounds=[(0, None)] * NS, method="highs").status == 0


def certificate(blocks):
    M, d = stack(blocks); m = len(d)
    r = linprog(np.zeros(m), A_ub=-M.T, b_ub=np.zeros(NS), A_eq=d[None], b_eq=[-1.0], bounds=[(-1e3, 1e3)] * m, method="highs")
    return None if r.status != 0 else r.x


def context_support(y, blocks, tol=1e-9):
    out, k = [], 0
    for c, t in blocks:
        n = len(t)
        if np.abs(y[k:k + n]).max() > tol: out.append(c)
        k += n
    return out


def minimal_infeasible(local, outer):
    found = []
    for r in range(len(outer) + 1):
        for Wset in itertools.combinations(range(len(outer)), r):
            if any(set(f) <= set(Wset) for f in found): continue
            if not extends(local + [outer[i] for i in Wset]): found.append(Wset)
    return found


def rho(h_ctx, W_ctx, g, tau, P):
    Mh = np.vstack([ctxmat(c) for c in h_ctx]); Mw = np.vstack([ctxmat(c) for c in W_ctx]); k = Mw.shape[0]; q = Mw @ P
    c = np.r_[np.zeros(NS), np.ones(k)]
    Aub = np.vstack([np.hstack([Mw, -np.eye(k)]), np.hstack([-Mw, -np.eye(k)]), np.r_[-g, np.zeros(k)][None]])
    bub = np.r_[q, -q, -tau]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=np.hstack([Mh, np.zeros((Mh.shape[0], k))]), b_eq=Mh @ P, bounds=[(0, None)] * (NS + k), method="highs")
    return (r.fun if r.status == 0 else np.inf), (r.x[:NS] if r.status == 0 else None)


def far_side_config(h_ctx, C_ctx, g, tau, P, rng, nvert=6):
    Mh = np.vstack([ctxmat(c) for c in h_ctx]); verts = []
    for _ in range(nvert):
        r = linprog(rng.normal(0, 1, NS), A_ub=-g[None], b_ub=[-tau], A_eq=Mh, b_eq=Mh @ P, bounds=[(0, None)] * NS, method="highs")
        if r.status == 0: verts.append(r.x)
    if not verts: return None, None
    p = rng.dirichlet(np.ones(len(verts))) @ np.array(verts); return [(c, ctxmat(c) @ p) for c in C_ctx], p


# ------------------------------------------------------------------ relay clock
def dependent(e, f): return bool(set(e[0]) & set(f[0]))


def layers(word):
    L = []
    for i, e in enumerate(word): L.append(1 + max([L[j] for j in range(i) if dependent(word[j], e)], default=0))
    return L


def activation(word, start):
    a = {start: 0}; L = layers(word)
    for e, t in sorted(zip(word, L), key=lambda x: x[1]):
        i, j = e[0]
        if i in a and a[i] < t and j not in a: a[j] = t
        elif j in a and a[j] < t and i not in a: a[i] = t
    return a


def reduce(word):
    w = list(word); changed = True
    while changed:
        changed = False
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if w[j][0] == w[i][0] and w[j][1] == -w[i][1] and all(not dependent(w[k], w[i]) for k in range(i + 1, j)):
                    del w[j]; del w[i]; changed = True; break
                if dependent(w[j], w[i]): break
            if changed: break
    L = layers(w); nf = []
    for t in range(1, max(L, default=0) + 1): nf.append(tuple(sorted(e for e, l in zip(w, L) if l == t)))
    return tuple(nf), w


def flip_bound(a, delta, Dfun, Tmax=10 ** 6):
    for T in range(0, Tmax):
        A = frozenset(j for j, t in a.items() if t < T and delta.get(j, 0) > 0)
        D = Dfun(A)
        if D <= 1e-12: return T
        if np.isfinite(D) and sum(delta[j] * (T - a[j]) for j in A) >= D - 1e-12: return T
    return np.inf
