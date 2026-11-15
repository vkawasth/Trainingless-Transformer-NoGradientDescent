"""Braids induced by data paths, read through the dual braid monoid (Bessis, arXiv math/0101158) and the Hurwitz action (section
sec:stokesbraid, worked example). New module; it modifies nothing; uses pce/stokesbraid.py conventions.
Braid words are lists of (k, e): sigma_k^e, k = 1..r-1, e = +1 / -1.
  artin(word, r)                 Artin's faithful action on the free group F_r: the images of x_1..x_r as reduced words (tuples of +-i);
                                 two braids are equal iff their actions are equal
  band(t, s)                     the Birman-Ko-Lee band generator a_ts (t > s) as an Artin word
  delta(r)                       the dual Garside element sigma_(r-1) ... sigma_1
  simples(r)                     the simple elements of the dual monoid: one per non-crossing partition of {1..r} (Catalan(r) of them),
                                 each block {b_1 < ... < b_j} giving the descending cycle a_(b_j b_(j-1)) ... a_(b_2 b_1)
  dual_invariants(word, r, kmax) inf, sup and canonical length sup - inf in the dual Garside structure, by search over products of
                                 simples (exact equality through the free-group action); the factors of one minimal factorisation
  hurwitz(word, F)               the Hurwitz action on a tuple of transpositions: sigma_i sends (t_i, t_(i+1)) to (t_i t_(i+1) t_i, t_i)
                                 and sigma_i^-1 sends it to (t_(i+1), t_(i+1) t_i t_(i+1)); the product is preserved
  coxeter_factorisations(r)      all reduced factorisations of the Coxeter element (1 2 ... r+1) of S_(r+1) into r transpositions
                                 ((r+1)^(r-1) of them); Bessis, Prop. 1.6.1: the braid group B_r acts transitively on them
  path_braid(fun, params, theta) the braid word of a path of configurations, with adaptive bisection so that each step carries one
                                 Stokes event
"""
from __future__ import annotations
import itertools
import numpy as np
import mpmath as mp
from scipy.optimize import linear_sum_assignment
# ---------------- free group and Artin action ----------------
def _red(w):
    out = []
    for a in w:
        if out and out[-1] == -a: out.pop()
        else: out.append(a)
    return tuple(out)
def _inv(w): return tuple(-a for a in reversed(w))
def _subst(w, img):
    out = []
    for a in w: out.extend(img[a] if a > 0 else _inv(img[-a]))
    return _red(out)
def artin(word, r):
    """the automorphism of beta = s_1 s_2 ... (applied as x -> beta(x), first letter innermost): images of x_1..x_r."""
    img = {i: (i,) for i in range(1, r + 1)}
    for k, e in reversed(word):
        g = {i: (i,) for i in range(1, r + 1)}
        if e > 0: g[k] = (k, k + 1, -k); g[k + 1] = (k,)
        else: g[k] = (k + 1,); g[k + 1] = (-(k + 1), k, k + 1)
        img = {i: _subst(g[i], img) for i in img} if False else {i: _subst(img[i], g) for i in img}
    return tuple(img[i] for i in range(1, r + 1))
def inverse(word): return [(k, -e) for k, e in reversed(word)]
def band(t, s):
    """a_ts = (s_(t-1) ... s_(s+1)) s_s (s_(s+1)^-1 ... s_(t-1)^-1), t > s."""
    up = [(k, 1) for k in range(t - 1, s, -1)]; return up + [(s, 1)] + [(k, -1) for k in range(s + 1, t)]
def delta(r): return [(k, 1) for k in range(r - 1, 0, -1)]
def _noncrossing(r):
    pts = list(range(1, r + 1)); out = []
    def parts(rest):
        if not rest: yield []; return
        first = rest[0]
        for k in range(len(rest)):
            for blk in itertools.combinations(rest[1:], k):
                b = (first,) + blk; rem = [x for x in rest if x not in b]
                for p in parts(rem): yield [b] + p
    for p in parts(pts):
        ok = True
        for A, B in itertools.combinations(p, 2):
            for a1, a2 in itertools.combinations(A, 2):
                for b1, b2 in itertools.combinations(B, 2):
                    if a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2: ok = False
        if ok: out.append(p)
    return out
def simples(r):
    out = []
    for p in _noncrossing(r):
        w = []
        for b in p:
            b = sorted(b)
            for j in range(len(b) - 1, 0, -1): w += band(b[j], b[j - 1])
        out.append((p, w))
    return out
def _exp(word): return sum(e for _, e in word)
def dual_invariants(word, r, depth=6, half=3):
    """inf = max k with delta^-k beta a product of simples; sup - inf = the least number m of simples in such a product (for a Garside
    monoid the left-greedy normal form realises it). Search: products of exactly m non-trivial simples, m <= half stored, larger m
    by meeting in the middle (x = P Q, P of length half). Exact equality through the free-group action."""
    S = [(p, w) for p, w in simples(r) if w]
    levels = {0: {artin([], r): ()}}
    for m in range(1, half + 1):
        out = {}
        for a, ps in levels[m - 1].items():
            base = _word(ps, r)
            for p, w in S:
                key = artin(base + w, r)
                if key not in out: out[key] = ps + (p,)
        levels[m] = out
    e = _exp(word)
    for k in range(e // (r - 1) + 1, -depth - 2, -1):
        x = (inverse(delta(r)) * k + word) if k >= 0 else (delta(r) * (-k) + word)
        ex = e - k * (r - 1); ax = artin(x, r)
        for m in range(0, depth + 1):
            if ex < 0 or ex > m * (r - 1): continue
            if m <= half:
                if ax in levels[m]: return dict(inf=k, sup=k + m, canonical_length=m, factors=levels[m][ax])
                continue
            for bq, qs in levels[m - half].items():
                key = artin(x + inverse(_word(qs, r)), r)
                if key in levels[half]: return dict(inf=k, sup=k + m, canonical_length=m, factors=levels[half][key] + qs)
    return None
def _word(ps, r):
    w = []
    for p in ps:
        for b in p:
            b = sorted(b)
            for j in range(len(b) - 1, 0, -1): w += band(b[j], b[j - 1])
    return w
# ---------------- Hurwitz action ----------------
def _tmul(p, q):
    """composition of permutations given as tuples (images of 0..n-1): (p q)(x) = p(q(x))."""
    return tuple(p[q[i]] for i in range(len(p)))
def transposition(i, j, n):
    t = list(range(n)); t[i], t[j] = t[j], t[i]; return tuple(t)
def hurwitz(word, F):
    F = list(F)
    for k, e in word:
        a, b = F[k - 1], F[k]
        if e > 0: F[k - 1], F[k] = _tmul(_tmul(a, b), a), a
        else: F[k - 1], F[k] = b, _tmul(_tmul(b, a), b)
    return tuple(F)
def product(F):
    out = tuple(range(len(F[0])))
    for t in F: out = _tmul(out, t)
    return out
def coxeter_factorisations(r):
    n = r + 1; c = product([transposition(i, i + 1, n) for i in range(r)]); T = [transposition(i, j, n) for i, j in itertools.combinations(range(n), 2)]
    out = []
    def rec(pref, cur):
        if len(pref) == r:
            if cur == c: out.append(tuple(pref))
            return
        for t in T: rec(pref + [t], _tmul(cur, t))
    rec([], tuple(range(n))); return c, out
def perm_of(word, r):
    p = list(range(r))
    for k, e in word: p[k - 1], p[k] = p[k], p[k - 1]
    return tuple(p)
# ---------------- braids of paths ----------------
def _feat(lz):
    # (log|z|, cos arg z, sin arg z): continuous across the negative real axis, unlike the complex logarithm (whose imaginary part jumps
    # by 2 pi there; the first version matched on it and mis-assigned strands, found by check HZ3)
    return np.stack([lz.real, np.cos(lz.imag), np.sin(lz.imag)], axis=1)
def _match(prev_l, lz):
    a, b = _feat(prev_l), _feat(lz); C = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2); _, c = linear_sum_assignment(C); return c
def path_braid(fun, params, theta=0.1, maxdepth=22):
    rot = mp.e ** (1j * theta)
    def state(x):
        z = fun(x); return np.array([complex(mp.log(v)) for v in z]), z
    def events(za, zb):
        a = [rot * x for x in za]; b = [rot * x for x in zb]; order = sorted(range(len(a)), key=lambda i: mp.re(a[i])); ev = []
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if mp.sign(mp.re(a[i] - a[j])) != mp.sign(mp.re(b[i] - b[j])):
                    pi, pj = order.index(i), order.index(j); lo, hi = (i, j) if pi < pj else (j, i)
                    ev.append((min(pi, pj) + 1, 1 if mp.im(b[lo]) < mp.im(b[hi]) else -1))
        return ev
    word = []; where = []
    l0, z0 = state(params[0])
    def turn(za, zb):
        """the largest change of argument of a pairwise difference over a step (a pair can cross twice inside one step)."""
        m = 0.0
        for i in range(len(za)):
            for j in range(i + 1, len(za)):
                a, b = complex(za[i] - za[j]), complex(zb[i] - zb[j])
                if a == 0 or b == 0: continue
                m = max(m, abs(np.angle(b / a)))
        return m
    def seg(x0, lz0, zz0, x1, lz1, zz1, d):
        c = _match(lz0, lz1); lz1 = lz1[c]; zz1 = [zz1[i] for i in c]; ev = events(zz0, zz1)
        if (len(ev) <= 1 and turn(zz0, zz1) < 0.3) or d >= maxdepth:
            word.extend(ev); where.extend([x1] * len(ev)); return lz1, zz1
        xm = (x0 + x1) / 2; lm, zm = state(xm); cm = _match(lz0, lm); lm = lm[cm]; zm = [zm[i] for i in cm]
        lm, zm = seg(x0, lz0, zz0, xm, lm, zm, d + 1)
        return seg(xm, lm, zm, x1, lz1, zz1, d + 1)
    lz, zz = l0, z0
    for x0, x1 in zip(params[:-1], params[1:]):
        l1, z1 = state(x1); lz, zz = seg(x0, lz, zz, x1, l1, z1, 0)
    return word, where
