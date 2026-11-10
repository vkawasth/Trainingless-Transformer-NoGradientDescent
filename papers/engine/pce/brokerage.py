"""Brokerage: a genuinely topological coordination experiment (companion note, 'Brokerage and Milnor's invariant'). New module; it
modifies nothing.

Three sources on a line: source 1, the broker 2, source 3. A relationship of source 1 with the broker is an interval: it opens (event A,
source 1 makes one full clockwise turn around the broker, the pure braid A = sigma_1^2) and later closes (A^{-1}, the turn undone). The
same for source 3 (B = sigma_2^2 with the broker). Each source's own history is fixed: source 1 has the interval sequence o, c, o, c, ...
(m intervals), and so has source 3; the broker never moves (pce/raagbraid realisation: the visiting source circles the broker). Only the
relative timing varies: a history is an interleaving of the two sources' own sequences.
Functions
  interleavings(m)              all merges of 1's and 3's sequences (each 2m events, own order kept): C(4m, 2m) histories
  event_word(h)                 the word in A^{+-1}, B^{+-1}
  intervals(h)                  the open/close times of each interval of 1 and of 3
  crossing_number(h)            sum over pairs (I of 1, J of 3) of +1 if I opens first and closes first while J is open (I and J cross,
                                1 first), -1 if J opens first and closes first, 0 if nested or disjoint
  mu123(h)                      Milnor's second-order coefficient of the closure (pce/braidknot.milnor)
  stats(h)                      what is held fixed or compared: counts, net windings, co-revision counts, who opens first, who closes
                                first, unsigned precedence counts, level-2 and level-3 signature of the count path, full braid
  realise(h)                    the planar trajectories (broker fixed); read back with pce/bundlebraid
"""
from __future__ import annotations
import itertools
import numpy as np
from . import braidknot as BK, ordrev as OR, bundlebraid as BB


def own(m): return [("o" if k % 2 == 0 else "c") for k in range(2 * m)]


def interleavings(m):
    n = 4 * m; out = []
    for pos in itertools.combinations(range(n), 2 * m):
        h = []; i1 = i3 = 0; s1 = own(m); s3 = own(m)
        for t in range(n):
            if t in pos: h.append(("1", s1[i1])); i1 += 1
            else: h.append(("3", s3[i3])); i3 += 1
        out.append(tuple(h))
    return out


def event_word(h):
    return [(1 if src == "1" else 2, 1 if kind == "o" else -1) for src, kind in h]


def braid_word(h): return OR.braid_word(event_word(h))


def intervals(h):
    out = {"1": [], "3": []}; openat = {}
    for t, (src, kind) in enumerate(h):
        if kind == "o": openat[src] = t
        else: out[src].append((openat.pop(src), t))
    return out


def crossing_number(h):
    I = intervals(h); c = 0
    for (a, b) in I["1"]:
        for (p, q) in I["3"]:
            if a < p < b < q: c += 1
            elif p < a < q < b: c -= 1
    return c


def mu123(h): return BK.milnor(braid_word(h))[1][3][0]


def linking(h): return BK.milnor(braid_word(h))[0]


def stats(h):
    w = event_word(h); I = intervals(h)
    opens1 = [a for a, _ in I["1"]]; opens3 = [p for p, _ in I["3"]]; closes1 = [b for _, b in I["1"]]; closes3 = [q for _, q in I["3"]]
    unsigned_prec = sum(1 for i in range(len(h)) for j in range(i + 1, len(h)) if h[i][0] == "1" and h[j][0] == "3")
    return dict(counts=tuple(sorted(h)), first_open="1" if opens1[0] < opens3[0] else "3", first_close="1" if closes1[0] < closes3[0] else "3",
                open_prec=sum(1 for a in opens1 for p in opens3 if a < p), close_prec=sum(1 for b in closes1 for q in closes3 if b < q),
                unsigned_prec=unsigned_prec, sig2=OR.sig_key(w, 2, 2), sig3=OR.sig_key(w, 2, 3), braid=OR.braid_key(w, 3))


def realise(h, n=40):
    """broker (index 1) fixed at 1; source 1 (index 0) at 0 and source 3 (index 2) at 2 circle it clockwise for an opening, back for a
    closing, at radius 0.3 after travelling along the line"""
    z = np.array([0, 1, 2], complex); P = [z.copy()]
    for src, kind in h:
        i = 0 if src == "1" else 2; s = 1 if kind == "o" else -1; base = z[i]; u = (z[1] - base) / abs(z[1] - base); a = z[1] - 0.3 * u
        for t in np.linspace(0, 1, 6)[1:]: q = P[-1].copy(); q[i] = base + t * (a - base); P.append(q)
        sign = s if src == "1" else s                                     # the same handedness for both visitors
        for t in np.linspace(0, 1, n)[1:]: q = P[-1].copy(); q[i] = z[1] - 0.3 * u * np.exp(-2j * np.pi * sign * t); P.append(q)
        for t in np.linspace(0, 1, 6)[1:]: q = P[-1].copy(); q[i] = a + t * (base - a); P.append(q)
    return np.array(P)
