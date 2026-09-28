"""Head + right-chain grammar (fixed depth L): the learner for Corpus V.

A level-l symbol b expands as
    k  ~ K_l[b, .]                 k in {0..KMAX}   (number of right children)
    c0 ~ H_l[b, .]                 left child
    c_t ~ R_l[b, c_{t-1}, .]       t = 1..k          (right children, Markov in the previous child)
Children are level-(l-1) symbols (tokens at l = 1).  Leaves are at depth L.

Exact inside with BANDED span tables: in_l[b, i, d] = P(b yields x[i:i+d]); spans of a level-l
symbol have length <= (KMAX+1)^l, and the top level needs only i = 0.  Expected counts for EM
are d log P / d log(theta) (autodiff), then rows are renormalised.
"""
from __future__ import annotations

import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
KMAX = 5


def init_params(L, V, A, rng, conc=1.0):
    P = {}
    for l in range(1, L + 1):
        C = A if l == 1 else V
        P[f"H{l}"] = rng.dirichlet(np.full(C, conc), V)
        P[f"K{l}"] = rng.dirichlet(np.full(KMAX + 1, conc), V)
        P[f"R{l}"] = rng.dirichlet(np.full(C, conc), (V, C))
    return P


def _level(inc, H, K, R, Dp, top):
    """inc: child inside (C, n, Dc+1) with inc[c, i, e] = inside of span [i, i+e).
    Returns parent inside (V, n_i, Dp+1) (n_i = 1 at the top level)."""
    C, n, Dc1 = inc.shape
    Dc = Dc1 - 1
    ni = 1 if top else n
    # W[c, i, d, e] = inc[c, i + d, e]  (0 outside the sentence)
    ii = jnp.arange(ni)[:, None] + jnp.arange(Dp + 1)[None, :]          # (ni, Dp+1)
    valid = ii < n
    W = jnp.where(valid[None, :, :, None], inc[:, jnp.clip(ii, 0, n - 1), :], 0.0)  # (C, ni, Dp+1, Dc+1)
    # f0[b, i, d, c] = H[b, c] inc[c, i, d]
    f = jnp.zeros((H.shape[0], ni, Dp + 1, C))
    f = f.at[:, :, :Dc + 1, :].set(H[:, None, None, :] * jnp.moveaxis(W[:, :, 0, :], 0, -1)[None])
    out = K[:, 0][:, None, None] * f.sum(-1)
    T = jnp.asarray((np.arange(Dp + 1)[:, None, None] + np.arange(Dc + 1)[None, :, None]
                     == np.arange(Dp + 1)[None, None, :]) & (np.arange(Dc + 1)[None, :, None] > 0), dtype=f.dtype)
    for t in range(1, KMAX + 1):
        g = jnp.einsum("bidc,bce->bide", f, R)                          # choose next child
        # advance by the next child's span length e:  f'[.., d+e, c'] += g[.., d, c'] W[c', i, d, e]
        f = jnp.einsum("bidc,cide,def->bifc", g, W, T)
        out = out + K[:, t][:, None, None] * f.sum(-1)
    return out


def make_loglik(L, V, A, nmax):
    D = {l: min((KMAX + 1) ** l, nmax) for l in range(1, L + 1)}

    def one(x, n, logP):
        P = {k: jnp.exp(v) for k, v in logP.items()}
        valid = (x >= 0).astype(jnp.float64)
        in0 = jnp.zeros((A, nmax, 2)).at[jnp.clip(x, 0, A - 1), jnp.arange(nmax), 1].add(valid)
        inc = in0
        for l in range(1, L + 1):
            top = l == L
            inc = _level(inc, P[f"H{l}"], P[f"K{l}"], P[f"R{l}"], nmax if top else D[l], top)
        prior = jnp.full(V, 1.0 / V)
        return jnp.log(prior @ inc[:, 0, n])

    b = jax.vmap(one, in_axes=(0, 0, None))
    f = jax.jit(lambda X, N, logP: b(X, N, logP).sum())
    gfun = jax.jit(jax.value_and_grad(lambda logP, X, N: b(X, N, logP).sum()))
    return f, gfun


def buckets(S, width=16, batch=32):
    by = {}
    for s in S:
        by.setdefault(int(np.ceil(len(s) / width) * width), []).append(s)
    out = []
    for nm, ss in sorted(by.items()):
        for s0 in range(0, len(ss), batch):
            part = ss[s0:s0 + batch]
            X = np.full((len(part), nm), -1, dtype=np.int64)
            for i, s in enumerate(part):
                X[i, :len(s)] = s
            out.append((nm, X, np.array([len(s) for s in part])))
    return out


class Learner:
    def __init__(self, L, V, A):
        self.L, self.V, self.A = L, V, A
        self.fns = {}

    def fn(self, nm):
        if nm not in self.fns:
            self.fns[nm] = make_loglik(self.L, self.V, self.A, nm)
        return self.fns[nm]

    def loglik(self, B, P):
        logP = {k: jnp.log(jnp.clip(v, 1e-300)) for k, v in P.items()}
        tot = sum(float(self.fn(nm)[0](X, N, logP)) for nm, X, N in B)
        return tot / sum(len(X) for _, X, _ in B)

    def em_step(self, B, P, smooth=1e-6):
        logP = {k: jnp.log(jnp.clip(v, 1e-300)) for k, v in P.items()}
        tot, C = 0.0, {k: np.zeros_like(v) for k, v in P.items()}
        for nm, X, N in B:
            val, gr = self.fn(nm)[1](logP, X, N)
            tot += float(val)
            for k in C:
                C[k] += np.asarray(gr[k])
        new = {k: (c + smooth) / (c + smooth).sum(-1, keepdims=True) for k, c in C.items()}
        return tot / sum(len(X) for _, X, _ in B), new


def gold_counts(trees, L, V, A):
    """Supervised MLE of the head + right-chain model from gold trees (best in class)."""
    C = {}
    for l in range(1, L + 1):
        c = A if l == 1 else V
        C[f"H{l}"] = np.zeros((V, c)); C[f"K{l}"] = np.zeros((V, KMAX + 1)); C[f"R{l}"] = np.zeros((V, c, c))

    def walk(node, l):
        b, kids = node
        syms = [k[1] if l == 1 else k[0] for k in kids]
        C[f"H{l}"][b, syms[0]] += 1; C[f"K{l}"][b, len(syms) - 1] += 1
        for a, c in zip(syms, syms[1:]):
            C[f"R{l}"][b, a, c] += 1
        if l > 1:
            for k in kids:
                walk(k, l - 1)
    for t in trees:
        walk(t, L)
    return {k: (v + 1e-6) / (v + 1e-6).sum(-1, keepdims=True) for k, v in C.items()}


# ---------------------------------------------------------------------------- positional variant
# b expands as  k ~ K[b,.],  c0 ~ H[b,k,.],  c_t ~ R[b,k,t-1,.]  (t = 1..k), children independent
# given (b, k, position).  It contains the true Corpus-V grammar whenever no symbol has two
# productions with the same number of right children.
def init_params_pos(L, V, A, rng, conc=1.0):
    P = {}
    for l in range(1, L + 1):
        C = A if l == 1 else V
        P[f"K{l}"] = rng.dirichlet(np.full(KMAX + 1, conc), V)
        P[f"H{l}"] = rng.dirichlet(np.full(C, conc), (V, KMAX + 1))
        P[f"R{l}"] = rng.dirichlet(np.full(C, conc), (V, KMAX + 1, KMAX))
    return P


def _level_pos(inc, H, K, R, Dp, top):
    C, n, Dc1 = inc.shape
    Dc = Dc1 - 1
    ni = 1 if top else n
    ii = jnp.arange(ni)[:, None] + jnp.arange(Dp + 1)[None, :]
    W = jnp.where((ii < n)[None, :, :, None], inc[:, jnp.clip(ii, 0, n - 1), :], 0.0)   # (C, ni, Dp+1, Dc+1)
    T = jnp.asarray((np.arange(Dp + 1)[:, None, None] + np.arange(Dc + 1)[None, :, None]
                     == np.arange(Dp + 1)[None, None, :]) & (np.arange(Dc + 1)[None, :, None] > 0), dtype=inc.dtype)
    base = jnp.zeros((C, ni, Dp + 1)).at[:, :, :Dc + 1].set(W[:, :, 0, :])            # a first child from i
    out = jnp.zeros((H.shape[0], ni, Dp + 1))
    for k in range(KMAX + 1):
        g = jnp.einsum("bc,cid->bid", H[:, k, :], base)                                # left child placed
        for t in range(k):
            g = jnp.einsum("bid,cide,def,bc->bif", g, W, T, R[:, k, t, :])            # next right child
        out = out + K[:, k][:, None, None] * g
    return out


def make_loglik_pos(L, V, A, nmax):
    D = {l: min((KMAX + 1) ** l, nmax) for l in range(1, L + 1)}

    def one(x, n, logP):
        P = {k: jnp.exp(v) for k, v in logP.items()}
        valid = (x >= 0).astype(jnp.float64)
        inc = jnp.zeros((A, nmax, 2)).at[jnp.clip(x, 0, A - 1), jnp.arange(nmax), 1].add(valid)
        for l in range(1, L + 1):
            top = l == L
            inc = _level_pos(inc, P[f"H{l}"], P[f"K{l}"], P[f"R{l}"], nmax if top else D[l], top)
        return jnp.log(jnp.full(V, 1.0 / V) @ inc[:, 0, n])

    b = jax.vmap(one, in_axes=(0, 0, None))
    return (jax.jit(lambda X, N, logP: b(X, N, logP).sum()),
            jax.jit(jax.value_and_grad(lambda logP, X, N: b(X, N, logP).sum())))


class LearnerPos(Learner):
    def fn(self, nm):
        if nm not in self.fns:
            self.fns[nm] = make_loglik_pos(self.L, self.V, self.A, nm)
        return self.fns[nm]


def gold_counts_pos(trees, L, V, A):
    C = {}
    for l in range(1, L + 1):
        c = A if l == 1 else V
        C[f"K{l}"] = np.zeros((V, KMAX + 1)); C[f"H{l}"] = np.zeros((V, KMAX + 1, c)); C[f"R{l}"] = np.zeros((V, KMAX + 1, KMAX, c))

    def walk(node, l):
        b, kids = node
        syms = [k[1] if l == 1 else k[0] for k in kids]
        k = len(syms) - 1
        C[f"K{l}"][b, k] += 1; C[f"H{l}"][b, k, syms[0]] += 1
        for t, c in enumerate(syms[1:]):
            C[f"R{l}"][b, k, t, c] += 1
        if l > 1:
            for kid in kids:
                walk(kid, l - 1)
    for t in trees:
        walk(t, L)
    return {k: (v + 1e-6) / (v + 1e-6).sum(-1, keepdims=True) for k, v in C.items()}
