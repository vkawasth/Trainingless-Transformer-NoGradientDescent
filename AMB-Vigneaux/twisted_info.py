"""Operation-twisted Vigneaux complex on a Z/n cycle: a finite computation.

Information structure (reduced, no trivial observable): variables c_0..c_{L-1} (values Z/n) and the
edge observables E_i = c_i c_{i+1}; products only inside a context: c_i c_{i+1} = E_i, X X = X,
c_i E_i = E_i.  Conditioning action (Vigneaux; alpha-deformed):
      (X ._a phi[Y])(P) = sum_x P(X=x)^a  phi[Y](P | X = x).
Coefficient module (finite, labelled): on an observable O with law Q,
      phi[O](Q) = c_O + sum_s w_O(s) Q(s)^a            (constants + labelled degree-a terms).
For a != 1 this module is closed under the a-action; for a = 1 it is the linear functionals
(expectations) with the Shannon action.
TWIST by an operation system g (edge i carries x -> x + g_i): inside context E_i the value of
c_{i+1} is read through the shift, i.e. phi[c_{i+1}] is applied to the RELABELLED marginal.
1-cocycle condition in every context and for every ordered pair X, Y of observables in it:
      phi[XY](P) = phi[X](P) + (X ._a phi[Y])(P)       for all laws P on the context.
Unknowns: (c_O, w_O) for all observables; the equations are linear in them, imposed at random laws.
"""
from __future__ import annotations

import itertools
import numpy as np


def _obs(L):
    return [("c", i) for i in range(L)] + [("E", i) for i in range(L)]


def _layout(L, n, logs=False):
    idx, k = {}, 0
    for o in _obs(L):
        size = n if o[0] == "c" else n * n
        width = 1 + size * (2 if logs else 1)
        idx[o] = (k, k + width); k += width
    return idx, k


def _xlogx(Q):
    return np.where(Q > 0, Q * np.log(np.where(Q > 0, Q, 1.0)), 0.0)


def cocycle_space_dim(L: int, n: int, g, alpha: float, samples: int = 60, seed: int = 0, logs: bool = False):
    """dim Z^1 of the twisted complex on the module above, and a basis (numerical nullspace).
    logs=True (alpha = 1 only) adds labelled Shannon terms  sum_s v_O(s) Q(s) log Q(s)."""
    rng = np.random.default_rng(seed)
    idx, N = _layout(L, n, logs)
    rows = []

    def phi_row(o, Q):
        """row vector r with r . theta = phi[o](Q)  (Q: law of o, flattened)."""
        r = np.zeros(N); a, b = idx[o]
        r[a] = 1.0
        m = len(Q)
        r[a + 1:a + 1 + m] = Q ** alpha
        if logs:
            r[a + 1 + m:a + 1 + 2 * m] = _xlogx(Q)
        return r

    for i in range(L):
        E, ca, cb = ("E", i), ("c", i), ("c", (i + 1) % L)
        shift = int(g[i]) % n
        for _ in range(samples):
            P = rng.dirichlet(np.ones(n * n)).reshape(n, n)          # law of (c_a, c_b) in context E_i
            Pa = P.sum(1)
            Pb = np.roll(P.sum(0), shift)                            # c_b read through the operation
            law = {E: P.ravel(), ca: Pa, cb: Pb}

            def cond(X, x):
                """law of every observable of the context, conditioned on X = x."""
                if X == ca:
                    Pc = np.zeros_like(P); Pc[x] = P[x] / P[x].sum()
                else:
                    Pc = np.zeros_like(P); Pc[:, (x - shift) % n] = P[:, (x - shift) % n] / P[:, (x - shift) % n].sum()
                return {E: Pc.ravel(), ca: Pc.sum(1), cb: np.roll(Pc.sum(0), shift)}

            def marg(X):
                return Pa if X == ca else Pb

            prod = {(ca, cb): E, (cb, ca): E, (ca, ca): ca, (cb, cb): cb, (ca, E): E, (E, ca): E,
                    (cb, E): E, (E, cb): E, (E, E): E}
            for (X, Y), XY in prod.items():
                lhs = phi_row(XY, law[XY])
                rhs = phi_row(X, law[X])
                if X == E:                                            # conditioning on the whole context
                    act = np.zeros(N)
                    for s in range(n * n):
                        if P.ravel()[s] <= 0:
                            continue
                        d = np.zeros(n * n); d[s] = 1.0; D = d.reshape(n, n)
                        cl = {E: d, ca: D.sum(1), cb: np.roll(D.sum(0), shift)}
                        act += P.ravel()[s] ** alpha * phi_row(Y, cl[Y])
                else:
                    act = np.zeros(N); m = marg(X)
                    for x in range(n):
                        if m[x] <= 0:
                            continue
                        act += m[x] ** alpha * phi_row(Y, cond(X, x)[Y])
                rows.append(lhs - rhs - act)
    A = np.array(rows)
    s = np.linalg.svd(A, compute_uv=False)
    tol = 1e-8 * max(1.0, s.max())
    null_dim = int((s < tol).sum()) + max(0, N - len(s))
    _, _, Vt = np.linalg.svd(A)
    basis = Vt[len(s) - int((s < tol).sum()):] if null_dim else np.zeros((0, N))
    return null_dim, basis, idx


def gauge_dim(L, n, alpha, logs=False):
    """Directions of theta that vanish identically on normalised laws (parametrisation kernel):
    at alpha = 1, (c_O, w_O) = (t, -t) gives phi = t - t sum Q = 0 -- one per observable."""
    return 2 * L if alpha == 1.0 else 0


def labelled_rank(basis, idx, n, logs=False):
    """Rank of the non-uniform (label-dependent) part of the cocycle space: for every observable
    and every coefficient block, subtract the block mean; a cocycle that depends on the labels
    (hence could see a relabelling twist) has nonzero remainder."""
    if len(basis) == 0:
        return 0
    parts = []
    for o, (a, b) in idx.items():
        m = n if o[0] == "c" else n * n
        blocks = [basis[:, a + 1:a + 1 + m]] + ([basis[:, a + 1 + m:a + 1 + 2 * m]] if logs else [])
        parts += [B - B.mean(1, keepdims=True) for B in blocks]
    return int(np.linalg.matrix_rank(np.hstack(parts), 1e-6))


def reduced_dim(L, n, g, alpha, logs=False, **kw):
    """(dim Z^1 modulo the parametrisation gauge, labelled rank)."""
    d, B, idx = cocycle_space_dim(L, n, g, alpha, logs=logs, **kw)
    return d - gauge_dim(L, n, alpha, logs), labelled_rank(B, idx, n, logs)


def band_twisting_is_trivial():
    """A monoid homomorphism from an idempotent monoid (XX = X) to a group is trivial:
    rho(X) = rho(XX) = rho(X)^2 implies rho(X) = 1."""
    return True


# ---------------------------------------------------------------------------------------------
# Non-symmetric coefficients: values in R^{Omega} (pointwise / random-variable valued cochains).
#   phi[O](Q) is a FUNCTION of the value o in Omega_O, i.e. an element of R^{Omega_O} pulled back
#   to R^{Omega_C}; relabellings permute the coordinates.  Pointwise conditioning action
#         (X . f)(P)(w) = f(P | X = X(w))(w)                              (no averaging)
#   (the band forces the action itself to be label-free: X.X.f = X.f leaves no room for a
#   permutation or a character in the action; the twist can only enter through GLUING, i.e. how
#   the shared variable c_{i+1} is read inside E_i).
#   Ansatz (label-dependent coefficients, local: depends on the law of O and on o only):
#         phi[O](Q)(v) = a0_O(v) + a1_O(v) log Q(v) + a2_O(v) Q(v) log Q(v)
#                        + sum_s b1_O(v,s) Q(s) + sum_s b2_O(v,s) Q(s)^2 .
#   Cocycle rule, pointwise at every w in the support:  phi[XY](P)(w) = phi[X](P)(w) + phi[Y](P|X=x(w))(w).
#   support = "full": Dirichlet laws on Omega_C.  support = "graph": laws on the graph of the
#   latent identity (c_{i+1} = c_i), i.e. the deterministic / single-outcome stratum.
#   support = "graph+eps": graph laws mixed with eps of full-support noise.
# ---------------------------------------------------------------------------------------------

def _pw_layout(L, n):
    idx, k = {}, 0
    for o in _obs(L):
        m = n if o[0] == "c" else n * n
        width = m * (3 + 2 * m)
        idx[o] = (k, m); k += width
    return idx, k


def pointwise_cocycles(L: int, n: int, g, support: str = "full", eps: float = 0.2,
                       samples: int = 40, seed: int = 0, spectrum: bool = False):
    """Effective dim Z^1 (modulo functionals that vanish on every law/value actually used)
    of the pointwise R^Omega complex on the twisted L-cycle."""
    rng = np.random.default_rng(seed)
    idx, N = _pw_layout(L, n)
    rows, evals = [], []

    def row(o, Q, v):
        r = np.zeros(N); base, m = idx[o]
        off = base + v * (3 + 2 * m)
        q = Q[v]
        r[off] = 1.0
        r[off + 1] = np.log(q)
        r[off + 2] = q * np.log(q)
        r[off + 3:off + 3 + m] = Q
        r[off + 3 + m:off + 3 + 2 * m] = Q ** 2
        evals.append(r)
        return r

    for i in range(L):
        E, ca, cb = ("E", i), ("c", i), ("c", (i + 1) % L)
        sh = int(g[i]) % n
        for _ in range(samples):
            if support == "full":
                P = rng.dirichlet(np.ones(n * n)).reshape(n, n)
            else:
                P = np.diag(rng.dirichlet(np.ones(n)))
                if support == "graph+eps":
                    P = (1 - eps) * P + eps * rng.dirichlet(np.ones(n * n)).reshape(n, n)

            def laws(Pm):
                return {E: Pm.ravel(), ca: Pm.sum(1), cb: np.roll(Pm.sum(0), sh)}

            def labels(a, b):                   # observed labels of w = (a, b_raw)
                return {E: a * n + b, ca: a, cb: (b + sh) % n}

            def cond(X, lab, a, b):
                if X == E:
                    D = np.zeros_like(P); D[a, b] = 1.0; return D
                if X == ca:
                    D = np.zeros_like(P); D[a] = P[a] / P[a].sum(); return D
                D = np.zeros_like(P); D[:, b] = P[:, b] / P[:, b].sum(); return D

            prod = {(ca, cb): E, (cb, ca): E, (ca, ca): ca, (cb, cb): cb, (ca, E): E, (E, ca): E,
                    (cb, E): E, (E, cb): E, (E, E): E}
            LW = laws(P)
            for a in range(n):
                for b in range(n):
                    if P[a, b] <= 1e-12:
                        continue
                    lab = labels(a, b)
                    for (X, Y), XY in prod.items():
                        PC = cond(X, lab, a, b)
                        LC = laws(PC)
                        r = row(XY, LW[XY], lab[XY]) - row(X, LW[X], lab[X]) - row(Y, LC[Y], lab[Y])
                        rows.append(r)
    A = np.array(rows)
    Ev = np.array(evals)
    if spectrum:
        # generalised singular values: min ||A th|| / ||Ev th|| over th outside the gauge
        _, se, Ve = np.linalg.svd(Ev, full_matrices=False)
        k = int((se > 1e-8 * se.max()).sum())
        W = Ve[:k].T / se[:k]
        return np.sort(np.linalg.svd(A @ W, compute_uv=False))
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = 1e-8 * max(1.0, s.max())
    r = int((s > tol).sum())
    Nsp = Vt[r:].T                                        # null space of the cocycle equations
    eff = int(np.linalg.matrix_rank(Ev @ Nsp, 1e-7)) if Nsp.size else 0
    return eff


def orbit_count(n, h):
    """number of orbits of x -> x + h on Z/n"""
    from math import gcd
    return gcd(n, h % n) if h % n else n
