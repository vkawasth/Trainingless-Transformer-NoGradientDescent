"""Probability of pseudo-links: regularising directions, correlated persistent biases, and noise (companion note, 'A probabilistic
theory of pseudo-links'). New module; it modifies nothing.

Persistent biases. A regularising direction omega in C^k (pce/tiebraid) is read as a persistent bias of each source. A law of omega is
circular if omega and exp(i phi) omega have the same law for every phi.
  sample_circular_gaussian(Sigma, n)  omega = A (Z1 + i Z2)/sqrt(2), Sigma = A A^T real, Z independent standard normal: circular Gaussian
  sample_rotated(n, k)                omega = r * exp(i Theta): an arbitrary (non-Gaussian) radial vector r, rotated by a uniform Theta
  phase_diff_density(psi, rho)        density of arg(d1) - arg(d2) for jointly circular complex Gaussians with complex correlation rho:
                                      (1 - |rho|^2) / (2 pi (1 - b^2)) * (1 + b (pi/2 + arcsin b) / sqrt(1 - b^2)),  b = |rho| cos(psi - arg rho)
  joint_sector_prob(w1, w2, rho)      P(arg d1 in sector w1, arg d2 in sector w2) by one-dimensional quadrature (arg d1 is uniform and
                                      independent of the phase difference)
Noise. Two sources tied for T steps (their points coincide) are perturbed:
  iid jitter      independent circular offsets at every step: the winding over the tie, in turns, is a sum of T - 1 independent uniform
                  variables on (-1/2, 1/2]: mean 0, variance (T - 1)/12
  random walk     the offset performs a planar Gaussian random walk: by Spitzer's law the winding angle divided by (1/2) log T tends to a
                  standard Cauchy law
  persistent bias a fixed offset: the winding over the tie is 0, and the link takes one of finitely many values (prop:tiebridge)
Tie histories. broker_tie(order): three sources, two visitors that merge with the broker (a tie) and leave sideways, in interleaved or
nested order, closed back to the start; read with pce/tiebraid.
"""
from __future__ import annotations
import numpy as np
from . import tiebraid as TB


def sample_circular_gaussian(Sigma, n, rng):
    A = np.linalg.cholesky(np.asarray(Sigma, float)); k = A.shape[0]
    Z = (rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))) / np.sqrt(2)
    return Z @ A.T


def sample_rotated(n, k, rng):
    r = rng.exponential(1.0, (n, k)) * np.array([1.0, 2.0, 0.5, 1.5][:k] + [1.0] * max(0, k - 4))     # fixed shape, random sizes
    r = r * np.where(rng.random((n, k)) < 0.5, 1, 1j)                                                   # some structure in the phases
    return r * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, 1)))


def phase_diff_density(psi, rho):
    a = abs(rho); th = np.angle(rho); b = a * np.cos(psi - th)
    return (1 - a * a) / (2 * np.pi * (1 - b * b)) * (1 + b * (np.pi / 2 + np.arcsin(b)) / np.sqrt(1 - b * b))


def _in(arc, x):
    lo, hi = arc; x = (x - lo) % (2 * np.pi); return x < (hi - lo) % (2 * np.pi) or (hi - lo) % (2 * np.pi) == 0


def joint_sector_prob(w1, w2, rho, n=4000):
    """w1, w2: sectors as (start angle, end angle), counterclockwise"""
    a1 = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n; psi = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
    f = phase_diff_density(psi, rho); tot = 0.0
    m1 = np.array([_in(w1, x) for x in a1])
    for x in a1[m1]:
        m2 = np.array([_in(w2, y) for y in (x - psi)]); tot += (f[m2]).sum() * (2 * np.pi / n)
    return tot * (2 * np.pi / n) / (2 * np.pi)


def corr_of_differences(Sigma, x, y, z):
    """complex correlation of (omega_x - omega_y) and (omega_x - omega_z) for a real covariance Sigma"""
    S = np.asarray(Sigma, float); c = S[x, x] - S[x, z] - S[y, x] + S[y, z]
    v1 = S[x, x] + S[y, y] - 2 * S[x, y]; v2 = S[x, x] + S[z, z] - 2 * S[x, z]
    return c / np.sqrt(v1 * v2)


def winding_turns(points):
    """total unwrapped winding (in turns) of the polygon through the given complex points around 0"""
    ang = np.angle(points); d = np.diff(ang); d = (d + np.pi) % (2 * np.pi) - np.pi; return d.sum() / (2 * np.pi)


def broker_tie(order="interleaved"):
    """sources 0 (visitor 1), 1 (broker), 2 (visitor 3); a visit merges with the broker and leaves sideways (visitor 1 up, visitor 3 down)"""
    Z = [np.array([-1, 0, 1], complex)]
    def step(i, v): z = Z[-1].copy(); z[i] = v; Z.append(z)
    if order == "interleaved": seq = [("in", 0), ("in", 2), ("out", 0), ("out", 2)]
    elif order == "nested": seq = [("in", 0), ("in", 2), ("out", 2), ("out", 0)]
    else: seq = [("in", 0), ("out", 0), ("in", 2), ("out", 2)]
    for kind, i in seq:
        if kind == "in": step(i, 0)
        else:
            step(i, 0.6j if i == 0 else -0.6j); step(i, -1 if i == 0 else 1)
    Z = np.array(Z); return np.vstack([Z, Z[:1]])


def broker_visit(order="interleaved", n=24):
    """three sources: visitor 1 (index 0) at -1, broker (index 1) at 0, visitor 3 (index 2) at +1. A visit of visitor i travels to the far
    point of a circle of radius 0.3 that passes through the broker's position, goes once round it clockwise (opening; anticlockwise for
    closing), passing exactly through the broker (a tie), and returns. Whether the regularised loop encloses the broker depends on the side
    of the broker's bias relative to the visitor's: the opening and closing of one relationship share that wall."""
    base = np.array([-1, 0, 1], complex); Z = [base.copy()]
    seqs = {"interleaved": [(0, 1), (2, 1), (0, -1), (2, -1)], "nested": [(0, 1), (2, 1), (2, -1), (0, -1)], "sequential": [(0, 1), (0, -1), (2, 1), (2, -1)]}
    for i, s in seqs[order]:
        side = -1 if i == 0 else 1; c = 0.3 * side; far = c + 0.3 * side
        z = Z[-1].copy(); z[i] = far; Z.append(z)
        for t in np.linspace(0, 1, n + 1)[1:]:                            # start at the far point; the broker point is reached at t = 1/2
            z = Z[-1].copy(); z[i] = c + (far - c) * np.exp(-2j * np.pi * s * t); Z.append(z)
        z = Z[-1].copy(); z[i] = base[i]; Z.append(z)
    Z = np.array(Z); Z[np.abs(Z) < 1e-12] = 0; return np.vstack([Z, Z[:1]])
