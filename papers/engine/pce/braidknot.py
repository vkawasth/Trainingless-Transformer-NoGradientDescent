"""What braids and knots carry: exact tools for the companion note (section 'What braids and knots carry'). New module; it modifies
nothing.

Conventions (stated once, used everywhere).
  Braid words are lists of nonzero integers: +i is sigma_i, -i is sigma_i^{-1}, strands numbered 1..n by position, words read left to
  right in time. Artin's action on the free group F_n = <x_1..x_n> (right action, applied letter by letter):
      sigma_i : x_i -> x_i x_{i+1} x_i^{-1},  x_{i+1} -> x_i,  other generators fixed;  sigma_i^{-1} is its inverse.
  The action is faithful (Artin 1947), so two words give the same braid iff they give the same images.
  For a pure braid b, b(x_i) = l_i x_i l_i^{-1}; l_i is normalised so that the exponent sum of x_i in it is 0. Its Magnus expansion
  (x_j -> 1 + X_j) has coefficient of X_j = linking number of components i and j of the closure, and, when all linking numbers vanish,
  coefficient of X_j X_k = Milnor's invariant mu(j k i) of the closure (Milnor 1954; for closures of pure braids see Habegger-Lin 1990).
  Sign conventions of mu vary between authors; we report the coefficient and check only symmetry, sign reversal and vanishing.
Functions
  artin(word, n)                 images of x_1..x_n (as lists of +-generator indices, freely reduced)
  same_braid(w1, w2, n)          True iff the two words give the same braid
  permutation(word, n)           the permutation of positions; closure components = its cycles
  longitudes(word, n), magnus(w) as above; milnor(word) -> linking numbers and second-order coefficients for 3 strands
  commutator_path(seq, ...)      a planar realisation of products of full twists A = sigma_1^2, B = sigma_2^2 (and inverses): during
                                 each twist the two strands turn once about their midpoint; returns the positions over time
  winding_area(P)                net windings w_ij and the signed area enclosed by (theta_12, theta_23), divided by (2 pi)^2
  parity_laws(theta)             q_+ and q_- on {+1,-1}^4 (the parity pair); moments(q, S) = E_q[prod_{i in S} e_i]
"""
from __future__ import annotations
import itertools
import numpy as np


def _red(w):
    out = []
    for x in w:
        if out and out[-1] == -x: out.pop()
        else: out.append(x)
    return out


def _inv(w): return [-x for x in reversed(w)]


def artin(word, n):
    img = {j: [j] for j in range(1, n + 1)}
    for g in word:
        i = abs(g); new = dict(img)
        if g > 0: new[i] = _red(img[i] + img[i + 1] + _inv(img[i])); new[i + 1] = img[i]
        else: new[i] = img[i + 1]; new[i + 1] = _red(_inv(img[i + 1]) + img[i] + img[i + 1])
        img = new
    return img


def same_braid(w1, w2, n): return artin(w1, n) == artin(w2, n)


def permutation(word, n):
    p = list(range(n))                                                  # p[pos] = strand now at pos
    for g in word:
        i = abs(g) - 1; p[i], p[i + 1] = p[i + 1], p[i]
    return p


def cycles(p):
    seen, out = set(), []
    for i in range(len(p)):
        if i in seen: continue
        c, j = [], i
        while j not in seen: seen.add(j); c.append(j); j = p[j]
        out.append(c)
    return out


def longitudes(word, n):
    img = artin(word, n); L = {}
    for i in range(1, n + 1):
        w = img[i]; m = (len(w) - 1) // 2
        if w[m] != i or _red(w[:m] + [i] + _inv(w[:m])) != w: raise ValueError("not a pure braid")
        lam = w[:m]; k = sum(1 if x == i else -1 if x == -i else 0 for x in lam)
        L[i] = _red(lam + ([-i] * k if k > 0 else [i] * (-k)))
    return L


def magnus(w, deg=2):
    P = {(): 1.0}
    for x in w:
        g, e = abs(x), (1 if x > 0 else -1)
        f = {(): 1.0, (g,): float(e)}
        if e < 0:
            for k in range(2, deg + 1): f[(g,) * k] = 1.0 if k % 2 == 0 else -1.0
        Q = {}
        for a, ca in P.items():
            for b, cb in f.items():
                if len(a) + len(b) <= deg: Q[a + b] = Q.get(a + b, 0.0) + ca * cb
        P = Q
    return P


def milnor(word):
    """3 strands, pure braid: linking numbers lk[i][j] and second-order coefficients c[i] = (coef X_j X_k, coef X_k X_j), j < k != i"""
    L = longitudes(word, 3); lk, c = {}, {}
    for i in (1, 2, 3):
        M = magnus(L[i]); j, k = [x for x in (1, 2, 3) if x != i]
        lk[i] = {j: round(M.get((j,), 0.0)), k: round(M.get((k,), 0.0))}
        c[i] = (round(M.get((j, k), 0.0)), round(M.get((k, j), 0.0)))
    return lk, c


A, B = [1, 1], [2, 2]
AI, BI = [-1, -1], [-2, -2]


def commutator_path(seq, N=400, jitter=0.0, seed=0):
    """seq: list of (i, j, s), strands i, j (0-based) turn once about their midpoint, s = +1 or -1"""
    rng = np.random.default_rng(seed); z = np.array([0, 1, 2], complex); P = [z.copy()]
    for (i, j, s) in seq:
        z0 = P[-1]; c = (z0[i] + z0[j]) / 2; r = (z0[i] - z0[j]) / 2
        for u in np.linspace(0, 1, N)[1:]:
            w = z0.copy(); ph = np.exp(2j * np.pi * s * u); w[i] = c + r * ph; w[j] = c - r * ph; P.append(w)
    P = np.array(P)
    if jitter:
        bump = np.sin(np.linspace(0, np.pi, len(P)))[:, None]             # zero at both ends, so the loop stays closed
        P = P + jitter * bump * (rng.standard_normal(P.shape) + 1j * rng.standard_normal(P.shape))
    return P


TWIST = {"A": (0, 1, 1), "B": (1, 2, 1), "a": (0, 1, -1), "b": (1, 2, -1)}   # a, b = inverses


def winding_area(P):
    th = lambda i, j: np.unwrap(np.angle(P[:, j] - P[:, i]))
    t12, t23, t13 = th(0, 1), th(1, 2), th(0, 2)
    w = {k: (v[-1] - v[0]) / (2 * np.pi) for k, v in (("12", t12), ("23", t23), ("13", t13))}
    area = 0.5 * np.sum(t12[:-1] * np.diff(t23) - t23[:-1] * np.diff(t12)) / (2 * np.pi) ** 2
    dmin = min(np.abs(P[:, i] - P[:, j]).min() for i, j in ((0, 1), (1, 2), (0, 2)))
    return w, float(area), float(dmin)


SIGNS4 = list(itertools.product((1, -1), repeat=4))


def parity_laws(theta):
    qp = np.array([(1 + theta * np.prod(e)) / 16 for e in SIGNS4]); qm = np.array([(1 - theta * np.prod(e)) / 16 for e in SIGNS4])
    return qp, qm


def moments(q, S): return float(sum(qi * np.prod([e[i] for i in S]) for qi, e in zip(q, SIGNS4)))
