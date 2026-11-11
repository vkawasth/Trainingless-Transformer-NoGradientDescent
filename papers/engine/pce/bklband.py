"""Band generators of Birman, Ko and Lee, and the co-revision twists of 'Co-revision over any graph' in that language. New module; it
modifies nothing.

Conventions as in braidknot: words are lists of nonzero integers (+i = sigma_i, -i = its inverse), strands 1..n by position, read left to
right in time; equality of braids is decided by Artin's faithful action (braidknot.same_braid).

  band(t, s)        the band generator a_ts (n >= t > s >= 1) of Birman-Ko-Lee (1998), as an Artin word:
                    a_ts = (sigma_{t-1} ... sigma_{s+1}) sigma_s (sigma_{s+1}^{-1} ... sigma_{t-1}^{-1});  a_{s+1,s} = sigma_s
  crossing(t,s,r,q) do the chords ts and rq of points 1..n in convex position cross?
  delta(n)          a_{n,n-1} ... a_{21} (= sigma_{n-1} ... sigma_1); delta(n)^n is the full twist
  full_twist(B)     the full twist of the points in the block B (sorted), delta_B^{|B|} with delta_B = a_{b_k b_{k-1}} ... a_{b_2 b_1}
  convex_points(n)  points on the concave arc y = -x^2, numbered by x; with the projection angle of raagbraid their projection order is
                    their order along the arc, which is the circular order of Birman-Ko-Lee
  half_twist_path   the two endpoints of a chord exchange counterclockwise (sign +1) or clockwise (-1) inside a thin lens around it;
                    with these points the clockwise exchange reads exactly as band(t, s) (checked for every chord in check_bklband)
  geo_band(P, t, s, sign)  the Artin word read off that path by the exact braid reader (bundlebraid.braid)
  geo_twist(P, t, s)       the twist of raagbraid (one endpoint travels to the other, turns once around it, comes back), read the same way
"""
from __future__ import annotations
import numpy as np
from . import braidknot as BK, bundlebraid as BB, raagbraid as RG

PHI = RG.PHI


def band(t, s):
    if not t > s >= 1: raise ValueError("need t > s >= 1")
    pre = list(range(t - 1, s, -1))
    return pre + [s] + [-x for x in reversed(pre)]


def crossing(t, s, r, q):
    a, b = sorted((s, t)); c, d = sorted((q, r))
    return a < c < b < d or c < a < d < b


def delta(n):
    out = []
    for t in range(n, 1, -1): out += band(t, t - 1)
    return out


def delta_block(B):
    B = sorted(B); out = []
    for k in range(len(B) - 1, 0, -1): out += band(B[k], B[k - 1])
    return out


def full_twist(B): return delta_block(B) * len(B)


def power(w, k): return w * k if k >= 0 else BK._inv(w) * (-k)


def convex_points(n):
    x = np.linspace(-1.0, 1.0, n); return x - 1j * x ** 2


def half_twist_path(P, t, s, sign=-1, eps=0.12, steps=60):
    """exchange the points t and s (1-based) along their chord; the lens has half-width eps * half-length"""
    P = np.asarray(P, complex); i, j = s - 1, t - 1; m = (P[i] + P[j]) / 2; d = P[i] - m
    path = []
    for th in np.linspace(0, np.pi, steps):
        z = P.copy(); off = d * np.cos(th) + 1j * d * eps * np.sin(th) * sign
        z[i] = m + off; z[j] = m - off; path.append(z)
    return np.array(path)


def geo_band(P, t, s, sign=-1):
    w, _, generic, dmin, _ = BB.braid(half_twist_path(P, t, s, sign), PHI)
    if not generic: raise RuntimeError("non-generic projection")
    return w


def geo_twist(P, t, s, sign=1):
    D = dict(name=f"cv{len(P)}_{t}{s}_{sign}", P=np.asarray(P, complex), E=[(s - 1, t - 1)])
    return BB.braid(RG._twist_path(D, 0, sign), PHI)[0]
