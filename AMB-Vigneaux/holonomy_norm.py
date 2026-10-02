"""Statistical holonomy as a size of the cohomology class: what is true of 'KL is a norm on H^1'.

Setting. A cycle graph C_m (one independent loop, H^1(C_m, A) = A). An operation system puts a group element g_e on
each edge; A acts on laws on a finite set. Transporting a base law p around the loop from a base vertex returns
hol(g) . p with hol(g) = g_1 + ... + g_m (A abelian). The statistical holonomy is
        KL(g) = KL(p || hol(g) . p),
the information (nats per unit) lost by assuming that transport around the loop closes (flat transport).
Facts (checked in examples/holonomy_norm/run.py and tests/test_holonomy_norm.py):
  (2) Class function. Changing g by a coboundary (g_e -> g_e + c_{t(e)} - c_{s(e)}) or moving the base vertex leaves
      hol(g) and hence KL(g) unchanged: KL depends only on [g]. (Joint translation preserves KL.)
  (1) Zero set. KL(g) = 0 iff hol(g) fixes p. So KL(g) = 0 <=> [g] = 0 holds exactly when the action is FREE on p
      (stabiliser of p trivial). Counterexample otherwise: the uniform law is fixed by every shift, KL = 0 for all [g].
  (3) Not a norm. On A = Z/n there is no scaling; KL(p || tau_h p) != KL(p || tau_{-h} p) in general (asymmetric);
      KL(kh) != |k| KL(h) and != k^2 KL(h) beyond small h.
  What is true (tilts, A = R^k acting by exponential tilting with statistic T: (h . p)(x) ~ p(x) e^{h . T(x)}):
      KL(p || h . p) = psi(h) - h . grad psi(0),   psi(h) = log E_p e^{h . T}      (EXACT: a Bregman divergence)
                     = 1/2 h^T F h + O(|h|^3),     F = Cov_p(T) = the Fisher information of the tilt family.
      sqrt(2 KL) is, to second order, the Euclidean norm |h|_F on H^1(C_m; R^k); it is a norm iff F > 0 iff the
      action is faithful to first order (no component of T is a.s. constant-affine on supp p). For translations of a
      smooth law on a fine circle, F is the location Fisher information sum (p')^2 / p.
  The plaquette holonomy of infogeo2 (grids) is the tilt case with T = cell indicators: KL ~ 1/2 |H|^2_I.
"""
from __future__ import annotations

import numpy as np


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))


# ------------------------------------------------------------------------------------------- finite abelian: shifts on Z/n
def shift(p, h):
    return np.roll(p, int(h))


def hol(g, n):
    return int(np.sum(g)) % n


def kl_shift(p, g):
    """statistical holonomy of shift operations g (one per edge of the loop) on the base law p on Z/n"""
    return kl(p, shift(p, hol(g, len(p))))


def coboundary(c):
    """g_e = c_{v+1} - c_v around the cycle (vertex gauge); its holonomy is 0"""
    c = np.asarray(c); return np.roll(c, -1) - c


def stabiliser(p, tol=1e-12):
    n = len(p)
    return [h for h in range(n) if np.max(np.abs(shift(p, h) - p)) < tol]


# ------------------------------------------------------------------------------------------- tilts: A = R^k
def tilt(p, h, T):
    w = p * np.exp(T @ h); return w / w.sum()


def kl_tilt(p, h, T):
    return kl(p, tilt(p, h, T))


def bregman_psi(p, h, T):
    """psi(h) - h . grad psi(0), psi(h) = log E_p exp(h . T)"""
    psi = np.log(p @ np.exp(T @ h)); return float(psi - h @ (p @ T))


def fisher_tilt(p, T):
    m = p @ T; C = T - m
    return (C * p[:, None]).T @ C


def quad(p, h, T):
    return float(0.5 * h @ fisher_tilt(p, T) @ h)
