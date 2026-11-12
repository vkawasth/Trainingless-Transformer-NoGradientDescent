"""From source histories to braids to pseudo-knots (companion note, 'The bridge: ties, resolutions and probability'). New module; it
modifies nothing.

Data. Z[t, x] in C: one point per source and step (here z_x = s_x + i v_x of pce/bundlebraid, without the ring). Points within SNAP of one
another at a step are replaced by their mean, so ties are exact (in the toy, ties are below 1e-7 and genuine separations above 3.9e-3).
The history is closed by the straight path from the last configuration back to the first.
Regularisation. For a direction omega in C^k with distinct coordinates and eps > 0, the strands Z + eps * omega form a closed loop of k
labelled points; for generic omega it avoids collisions and is a pure braid, read by projection on exp(i phi).
Walls. For a pair (x, y) put D = z_x - z_y and delta = omega_x - omega_y. On a segment where D goes from d0 != 0 to 0 (a merge), the
regularised pair collides iff delta lies on the ray -R_+ d0; where D goes from 0 to d1 != 0 (a split), iff delta lies on -R_+ d1; where D
crosses 0 inside a segment, iff delta lies on the line through the segment's direction; where D stays 0, never. The walls of the pair are
these rays; the sector of delta between them is the pair's resolution. The resolution of omega is the vector of sectors.
Functions
  snap(Z), closed(Z), walls(Z), resolution(Z, W, omega)
  regularised(Z, omega, eps, phi)      the pure braid word and the strand order at the start
  linking(word, init, k)               pairwise linking numbers of the closure
  sub_braid(word, init, keep)          the braid on a subset of strands (forgetting the others), as a 3-strand word for milnor()
  triple(word, init, x, y, z)          Milnor's second-order coefficient of the closure of strands x, y, z (pce/braidknot.milnor)
  sector_probability(W_pair)           for isotropic Gaussian omega, the probability of each sector: its angle / 2 pi
"""
from __future__ import annotations
import itertools
import numpy as np
from . import bundlebraid as BB, braidknot as BK

SNAP = 1e-6


def snap(Z, tol=SNAP):
    Z = np.array(Z, complex).copy(); k = Z.shape[1]
    for t in range(len(Z)):
        lab = list(range(k))
        for a, b in itertools.combinations(range(k), 2):
            if abs(Z[t, a] - Z[t, b]) < tol:
                la, lb = lab[a], lab[b]; lab = [la if l == lb else l for l in lab]
        for l in set(lab):
            idx = [i for i in range(k) if lab[i] == l]; Z[t, idx] = Z[t, idx].mean()
    return Z


def closed(Z): Z = snap(Z); return np.vstack([Z, Z[:1]])


def walls(Z):
    """per pair (a, b): sorted list of wall angles in the delta-plane (a ray at angle theta is {r e^{i theta}: r > 0})"""
    k = Z.shape[1]; W = {}
    for a, b in itertools.combinations(range(k), 2):
        D = Z[:, a] - Z[:, b]; angs = set()
        for t in range(len(D) - 1):
            d0, d1 = D[t], D[t + 1]; z0, z1 = d0 == 0, d1 == 0
            if z0 and z1: continue
            if z1: angs.add(round(float(np.angle(-d0)), 9))
            elif z0: angs.add(round(float(np.angle(-d1)), 9))
            else:
                dd = d1 - d0; u = -np.real(np.conj(d0) * dd) / abs(dd) ** 2
                if 0 < u < 1 and abs(d0 + u * dd) < 1e-12:
                    angs.add(round(float(np.angle(dd)), 9)); angs.add(round(float(np.angle(-dd)), 9))
        W[(a, b)] = sorted(angs)
    return W


def _sector(delta, angs):
    if len(angs) < 2: return 0
    a = float(np.angle(delta)); return int(sum(a > x for x in angs)) % len(angs)


def resolution(W, omega):
    return tuple(_sector(omega[a] - omega[b], W[(a, b)]) for (a, b) in sorted(W) if len(W[(a, b)]) >= 2)


def regularised(Z, omega, eps=1e-5, phi=0.123):
    P = Z + eps * omega[None, :]
    word, log, gen, dmin, _ = BB.braid(P, phi)
    init = list(np.argsort(np.real(P[0] * np.exp(-1j * phi)), kind="stable"))
    return word, init, gen, dmin


def linking(word, init, k):
    pos = list(init); lk = np.zeros((k, k))
    for g in word:
        i = abs(g) - 1; x, y = pos[i], pos[i + 1]; lk[min(x, y), max(x, y)] += np.sign(g) / 2; pos[i], pos[i + 1] = y, x
    return tuple(int(round(v)) for v in lk[np.triu_indices(k, 1)])


def sub_braid(word, init, keep):
    """forget all strands not in keep (strand labels); the remaining crossings, renumbered by position among the kept strands"""
    pos = list(init); out = []
    for g in word:
        i = abs(g) - 1; x, y = pos[i], pos[i + 1]
        if x in keep and y in keep:
            kept = [s for s in pos if s in keep]; j = kept.index(x); out.append((j + 1) * (1 if g > 0 else -1))
        pos[i], pos[i + 1] = y, x
    start = [s for s in init if s in keep]
    return out, start


def triple(word, init, x, y, z):
    w, start = sub_braid(word, init, {x, y, z})
    lk, c = BK.milnor(w)
    return c[3][0]


def sector_probability(angs):
    if len(angs) < 2: return [1.0]
    a = list(angs) + [angs[0] + 2 * np.pi]
    widths = [(a[i + 1] - a[i]) / (2 * np.pi) for i in range(len(angs))]
    return [widths[-1]] + widths[:-1]                                   # sector 0 is the one containing angles above the last wall
