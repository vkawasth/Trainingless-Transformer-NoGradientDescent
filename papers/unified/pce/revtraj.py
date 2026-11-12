"""Braids read from simulated revision trajectories, for comparison with the event-word braid of pce/ordrev (companion note,
'Ordered co-revision'). New module; it modifies nothing.

Four sources start at 0, 1, 2, 3 on the real line (their reports projected to the plane). An event (i, +1) is an adoption by the pair
(i, i+1), (i, -1) a withdrawal. Three revision geometries:
  partial      adoption moves both a sixth of the gap towards each other (a third in total) along the segment joining them; withdrawal moves them apart
               by the same rule (no ties ever)
  retrace      adoption moves both to their midpoint (a tie; repeated adoption of a tied pair does nothing); withdrawal of a tied pair
               returns both to where they were before they merged; of an untied pair, as in 'partial'
  sideways     as 'retrace', but a tied pair withdraws by separating perpendicular to the line (first strand up, second down, by 0.4)
               and then returning to the line at distance 0.5 from the midpoint, in its original order
The trajectory is closed by the straight path back to the start and read with pce/tiebraid (walls, resolutions, regularised braids). The
start is untied, so braids from different histories and regularisations share a base configuration and can be compared as elements.
"""
from __future__ import annotations
import numpy as np
from . import tiebraid as TB, braidknot as BK

K = 4


def trajectory(h, mode):
    z = np.arange(K, dtype=complex); P = [z.copy()]; pre = {}
    for (i, s) in h:
        a, b = i - 1, i; z = P[-1].copy(); tied = abs(z[a] - z[b]) < 1e-12
        if mode == "partial" or not tied and s < 0:                     # move along the joining segment
            d = z[b] - z[a]; f = 1 / 6 if s > 0 else -1 / 6
            z[a] += f * d; z[b] -= f * d; P.append(z); continue
        if s > 0:                                                         # full adoption: merge at the midpoint
            if not tied: pre[i] = (z[a], z[b]); m = (z[a] + z[b]) / 2; z[a] = z[b] = m
            P.append(z); continue
        m = z[a]                                                          # withdrawal of a tied pair
        if mode == "retrace":
            z[a], z[b] = pre.get(i, (m - 0.5, m + 0.5)); P.append(z)
        else:
            z1 = z.copy(); z1[a] = m + 0.4j; z1[b] = m - 0.4j; P.append(z1)
            z2 = z1.copy(); z2[a] = m - 0.5; z2[b] = m + 0.5; P.append(z2)
    return np.array(P)


def braids(h, mode, n=40, seed=0, eps=1e-5):
    """set of Artin keys of the regularised closed trajectory over n random directions; number of walls"""
    Z = TB.closed(trajectory(h, mode)); W = TB.walls(Z); rng = np.random.default_rng(seed); keys = set()
    for _ in range(n):
        om = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        w, init, gen, dmin = TB.regularised(Z, om, eps, 0.123)
        img = BK.artin(w, K); keys.add(tuple(tuple(img[j]) for j in sorted(img)))
    return keys, sum(len(v) for v in W.values())
