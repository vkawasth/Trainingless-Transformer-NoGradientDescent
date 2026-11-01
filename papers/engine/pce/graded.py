"""Step 4e of "Probability, Context and Differentiation: The Layers": the graded classes M_k of maps that keep feasibility at order k. New
module.

  rand_Tk_blocks(x, blocks, k, rng)    a random point of T^k of a product of simplices over x (coefficients on zero cells made non-negative
                                       with probability 0.6 per monomial, then filtered by the lexicographic lemma)
  direct_Tk(f, x, blocks, k, rng, n)   apply T^k f (f evaluated on Weil numbers) to n random points; returns (ok, tested, bad)
  zero_embed(c, k)                     the zero section T^k -> T^{k+1}: new coefficients on monomials containing e_{k+1} are 0
  sf_map(P)                            the subtraction-free map x -> (P_y(x) / sum_z P_z(x))_y for polynomials P_y with non-negative coefficients
"""
from __future__ import annotations
import numpy as np
from . import tangent as TG
from .tangent import HD


def in_blocks(c, blocks, k): return all(TG.in_T(c[:, b], k) for b in blocks)


def rand_Tk_blocks(x, blocks, k, rng, tries=5000):
    x = np.asarray(x, float); n = len(x)
    for _ in range(tries):
        c = np.zeros((2 ** k, n)); c[0] = x
        for b in blocks:
            xb = x[b]; sup = np.flatnonzero(xb > 0)[0]
            for S in range(1, 2 ** k):
                z = rng.integers(-2, 3, size=len(b)).astype(float)
                if rng.random() < 0.6: z[xb <= 0] = np.abs(z[xb <= 0])
                z[sup] -= z.sum(); c[S, b] = z
        if in_blocks(c, blocks, k): return c
    raise RuntimeError("no point found")


def direct_Tk(f, x, blocks, k, rng, n=200):
    bad = 0
    for _ in range(n):
        c = rand_Tk_blocks(x, blocks, k, rng); ys = f(HD(c, k)); img = np.stack([y.c for y in ys], axis=1); bad += not TG.in_T(img, k)
    return bad == 0, n, bad


def zero_embed(c, k):
    out = np.zeros((2 ** (k + 1),) + c.shape[1:]); out[:2 ** k] = c; return out


def sf_map(P):
    """P: list of (coefficient, exponent tuple) lists, one per output cell; coefficients > 0"""
    def f(X):
        vals = []
        for poly in P:
            tot = None
            for co, ex in poly:
                t = co
                for i, e in enumerate(ex):
                    for _ in range(e): t = t * X[i]
                tot = t if tot is None else tot + t
            vals.append(tot)
        Z = vals[0]
        for v in vals[1:]: Z = Z + v
        R = Z.recip(); return [v * R for v in vals]
    return f
