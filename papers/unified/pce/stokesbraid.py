"""Braids of critical values along data paths (section sec:stokesbraid). New module; it modifies nothing.
A data path lambda(s) moves the critical values of W_T (T = e^-N) in the plane. Ordering them by the real part of e^(i theta) z (a fixed
generic projection, the order of a distinguished basis of straight paths from a base point far in the direction e^(-i theta) i), a change
of order between two values is a Stokes event, and it contributes one Artin generator: sigma_k^(+1) when the strand at position k passes
in front (smaller projected imaginary part) of the strand at k + 1, sigma_k^(-1) otherwise.
  crit_values_F1(lam, N)          the four critical values of the F_1 potential (from the quartic of ex:F1)
  crit_values_CP2(lam, N)         the three critical values of the CP^2 potential y1 + y2 + T^(-l3) / (y1 y2) (with T^(-l1), T^(-l2))
  track(fun, lams, N, theta)      critical values along a path, matched step to step by log-distance; valuations v = log|z| / N
  stokes_events(tr, theta)        the crossings (step, i, j) and the braid word [(k, sign)]
"""
from __future__ import annotations
import numpy as np
import mpmath as mp
from scipy.optimize import linear_sum_assignment
def _prec(N): mp.mp.dps = max(50, int(0.6 * N) + 30)
def crit_values_F1(lam, N):
    _prec(N); T = mp.e ** (-N); l1, l2, l3, l4 = [mp.mpf(x) for x in lam]
    roots = mp.polyroots([T ** (-l2), 0, 0, -T ** (-(l1 + l3) / 2), -T ** (-l4)], maxsteps=400, extraprec=400)
    out = []
    for w in roots:
        y2 = w ** 2; y1 = T ** ((l1 - l3) / 2) / w; out.append(T ** (-l1) * y1 + T ** (-l2) * y2 + T ** (-l3) / (y1 * y2) + T ** (-l4) / y2)
    return out
def crit_values_CP2(lam, N):
    """W = a y1 + b y2 + c / (y1 y2): critical points y1 = (b c / a^2)^(1/3) zeta, y2 = a y1 / b; values 3 (a b c)^(1/3) zeta^k."""
    _prec(N); T = mp.e ** (-N); a, b, c = [T ** (-mp.mpf(x)) for x in lam]
    r = mp.cbrt(a * b * c); return [3 * r * mp.e ** (2j * mp.pi * k / 3) for k in range(3)]
def track(fun, lams, N):
    prev = None; out = []
    for lam in lams:
        z = fun(lam, N); lz = np.array([complex(mp.log(x)) for x in z])
        if prev is not None:
            f = lambda q: np.stack([q.real, np.cos(q.imag), np.sin(q.imag)], axis=1)     # continuous across the negative real axis
            C = np.linalg.norm(f(prev)[:, None, :] - f(lz)[None, :, :], axis=2); _, c = linear_sum_assignment(C); lz = lz[c]; z = [z[i] for i in c]
        out.append((lz, z)); prev = lz
    return out
def stokes_events(tr, theta=0.1):
    rot = mp.e ** (1j * theta); ev = []; word = []
    for k in range(1, len(tr)):
        a = [rot * x for x in tr[k - 1][1]]; b = [rot * x for x in tr[k][1]]
        order = sorted(range(len(a)), key=lambda i: mp.re(a[i]))
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if mp.sign(mp.re(a[i] - a[j])) != mp.sign(mp.re(b[i] - b[j])):
                    pi, pj = order.index(i), order.index(j); lo, hi = (i, j) if pi < pj else (j, i)
                    pos = min(pi, pj) + 1; front = mp.im(b[lo]) < mp.im(b[hi])
                    ev.append((k, i, j)); word.append((pos, 1 if front else -1))
    return ev, word
def valuations(tr, N): return np.array([[x.real / N for x in t[0]] for t in tr])
