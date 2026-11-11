"""Masking: how deep can a newcomer's contribution hide, and what does hiding cost? New module; it modifies nothing.

Words in a free group F_n are lists of +-i (letter i = generator x_i; in the defect setting x_i = A_{i,n+1}, the newcomer's turn around
source i). The depth of g is the largest k with g in gamma_k (lower central series), infinity for g = 1. By Magnus (dimension subgroups
of free groups equal the lower central series), depth(g) is the lowest degree of a nonzero coefficient of M(g) - 1, M(x_i) = 1 + X_i.

  magnus_arrays(word, n, D)   truncated Magnus expansion as a list of dense arrays (degree d: shape (n,)*d)
  depth(word, n, D)           lowest nonzero degree <= D, or D + 1 if none
  shortest_by_depth(n, L, D)  exhaustive search over reduced words of length <= L: the shortest word of each depth 1..D, and under key
                              D + 1 the shortest of depth > D (depth-first, Magnus arrays updated letter by letter)
  conj_decomposition(bp, n)   for an enlarged history written as a word in Artin's pure generators with the newcomer at n + 1: the list of
                              (j, sign, c) such that the defect relative to s_out is prod_i c_i x_{j_i}^{sign_i} c_i^{-1}, c_i the image of
                              the old history before event i (Artin's action), and the product itself
"""
from __future__ import annotations
import numpy as np
from . import stranddefect as S, braidknot as BK

red, inv = BK._red, BK._inv


def _mul_gen(A, g, n, D):
    """right-multiply truncated series A (list over degrees) by M(x_|g|)^{+-1}"""
    i = abs(g) - 1; e = np.zeros(n); e[i] = 1.0; out = [a.copy() for a in A]
    if g > 0:
        for d in range(1, D + 1): out[d] = out[d] + np.multiply.outer(A[d - 1], e)
    else:
        for d in range(1, D + 1):
            pw = np.array(1.0)
            for j in range(1, d + 1):
                pw = np.multiply.outer(pw, -e); out[d] = out[d] + np.multiply.outer(A[d - j], pw)
    return out


def magnus_arrays(word, n, D):
    A = [np.ones(())] + [np.zeros((n,) * d) for d in range(1, D + 1)]
    for g in word: A = _mul_gen(A, g, n, D)
    return A


def _depth_of(A, D):
    for d in range(1, D + 1):
        if np.any(np.abs(A[d]) > 1e-9): return d
    return D + 1


def depth(word, n, D=6): return _depth_of(magnus_arrays(word, n, D), D)


def shortest_by_depth(n, L, D):
    best = {}
    letters = [s * i for i in range(1, n + 1) for s in (1, -1)]
    start = [np.ones(())] + [np.zeros((n,) * d) for d in range(1, D + 1)]
    stack = [([], start)]
    while stack:
        w, A = stack.pop()
        if w:
            k = _depth_of(A, D)
            if k not in best or len(w) < len(best[k]): best[k] = list(w)        # key D + 1 means depth > D
        if len(w) == L: continue
        for g in letters:
            if w and w[-1] == -g: continue
            stack.append((w + [g], _mul_gen(A, g, n, D)))
    return best


def conj_decomposition(bp, n):
    """bp: list of ('new', j, sign) and ('old', word-on-n-strands) blocks, in time order. Returns the factors and the defect word."""
    old_so_far = []; factors = []; prod = []
    for blk in bp:
        if blk[0] == "old": old_so_far = old_so_far + list(blk[1])
        else:
            _, j, s = blk
            g = S.pure_gen(j, n + 1) if s > 0 else inv(S.pure_gen(j, n + 1))
            img = S.loop_word(old_so_far + g + inv(old_so_far), n)        # = c x_j^{s} c^{-1} in the combing basis
            factors.append((j, s, img)); prod = red(prod + img)
    return factors, prod


def build(bp, n):
    """the enlarged braid word and the old history (forget) for a block description"""
    word = []; old = []
    for blk in bp:
        if blk[0] == "old": word += list(blk[1]); old += list(blk[1])
        else:
            _, j, s = blk; g = S.pure_gen(j, n + 1); word += g if s > 0 else inv(g)
    return word, old
