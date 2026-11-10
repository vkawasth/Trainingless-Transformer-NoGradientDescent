"""Co-revision over an arbitrary graph: the right-angled Artin group A_G and its braid representations (companion note, 'Co-revision
over any graph'). New module; it modifies nothing.

A co-revision graph G has the sources as vertices and the pairs that co-revise as edges. Events are signed edges. Two events are independent
when their edges share no source. A_G = < a_e, e in E(G) | a_e a_f = a_f a_e for independent e, f > is the right-angled Artin group of
the independence graph (vertices E(G), adjacent when disjoint).
Word problem. raag_normal_form(w): repeatedly cancel a letter against a later inverse when every letter between them commutes with it
(this yields a reduced word), then take the Cartier-Foata normal form; two words are equal in A_G iff the normal forms agree.
Braid representation. Sources are points in the plane (a drawing of G with straight edges). The twist of edge e = (i, j) is realised
geometrically: source i travels along the segment towards j, turns once clockwise around j at radius 0.15, and returns; all other sources
stay. The resulting pure braid (read with pce/bundlebraid, projection angle 0.123) is the Dehn twist about the boundary of a thin
neighbourhood of the segment. twist_word(D, e, N) is its N-th power as an Artin word; braid_key(D, w, N) the Artin images of a history.
Drawings (D): positions, edges.
Partially commutative signature: pcsig_key(w, D, level) as in pce/ordrev, with independence = disjoint edges.
"""
from __future__ import annotations
import itertools
import numpy as np
from . import bundlebraid as BB, braidknot as BK

R = 0.15; PHI = 0.123


def drawing(name):
    if name == "path4": P = [0, 1, 2, 3]; E = [(0, 1), (1, 2), (2, 3)]
    elif name == "star3": P = [0, np.exp(1j * np.pi / 2), np.exp(1j * 7 * np.pi / 6), np.exp(1j * 11 * np.pi / 6)]; E = [(0, 1), (0, 2), (0, 3)]
    elif name == "triangle": P = [0, 1, 0.5 + 0.9j]; E = [(0, 1), (1, 2), (0, 2)]
    elif name == "square": P = [0, 1, 1 + 1j, 1j]; E = [(0, 1), (1, 2), (2, 3), (0, 3)]
    elif name == "k4_planar": P = [0, 2, 1 + 1.8j, 1 + 0.6j]; E = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    elif name == "k4_crossed": P = [0, 1, 1 + 1j, 1j]; E = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
    elif name == "path5": P = [0, 1, 2, 3, 4]; E = [(0, 1), (1, 2), (2, 3), (3, 4)]
    else: raise ValueError(name)
    return dict(name=name, P=np.array(P, complex), E=E)


def independent(D, a, b):
    return not (set(D["E"][a]) & set(D["E"][b]))


def segments_cross(D, a, b):
    """do the straight segments of two disjoint edges cross?"""
    (i, j), (k, l) = D["E"][a], D["E"][b]
    p, q, r, s = D["P"][i], D["P"][j], D["P"][k], D["P"][l]
    def orient(x, y, z): return np.sign(((y - x).conjugate() * (z - x)).imag)
    return orient(p, q, r) * orient(p, q, s) < 0 and orient(r, s, p) * orient(r, s, q) < 0


def _twist_path(D, e, sign=1, n=48):
    P0 = D["P"]; i, j = D["E"][e]; d = P0[j] - P0[i]; u = d / abs(d); a = P0[j] - R * u
    path = [P0.copy()]
    for t in np.linspace(0, 1, 8)[1:]:
        z = P0.copy(); z[i] = P0[i] + t * (a - P0[i]); path.append(z)
    for t in np.linspace(0, 1, n)[1:]:
        z = P0.copy(); z[i] = P0[j] - R * u * np.exp(-2j * np.pi * sign * t); path.append(z)
    for t in np.linspace(0, 1, 8)[1:]:
        z = P0.copy(); z[i] = a + t * (P0[i] - a); path.append(z)
    return np.array(path)


_cache = {}


def twist_word(D, e, sign=1):
    key = (D["name"], e, sign)
    if key not in _cache:
        w = BB.braid(_twist_path(D, e, sign), PHI)[0]; _cache[key] = w
    return _cache[key]


def braid_word(D, w, N=1):
    out = []
    for (e, s) in w: out += twist_word(D, e, s) * N
    return out


def braid_key(D, w, N=1):
    img = BK.artin(braid_word(D, w, N), len(D["P"])); return tuple(tuple(img[j]) for j in sorted(img))


def raag_normal_form(D, w):
    w = list(w); changed = True
    while changed:
        changed = False
        for a in range(len(w)):
            for b in range(a + 1, len(w)):
                if w[b][0] == w[a][0] and w[b][1] == -w[a][1] and all(independent(D, w[a][0], w[c][0]) for c in range(a + 1, b)):
                    del w[b]; del w[a]; changed = True; break
            if changed: break
    steps = []
    for x in w:
        lvl = 0
        for d in range(len(steps) - 1, -1, -1):
            if any(not independent(D, f[0], x[0]) for f in steps[d]): lvl = d + 1; break
        if lvl == len(steps): steps.append([])
        steps[lvl].append(x)
    return tuple(tuple(sorted(s)) for s in steps)


def pcsig_key(D, w, level, nd=9):
    m = len(D["E"]); S = [np.ones(())] + [np.zeros((m,) * k) for k in range(1, level + 1)]
    for (e, s) in w:
        x = np.zeros(m); x[e] = s; Ex = [np.ones(()), x.copy()]
        for k in range(2, level + 1): Ex.append(np.multiply.outer(Ex[-1], x) / k)
        S = [np.ones(())] + [sum(np.multiply.outer(S[j], Ex[k - j]) for j in range(k + 1)) for k in range(1, level + 1)]
    out = []
    for k in range(1, level + 1):
        acc = {}
        for idx in itertools.product(range(m), repeat=k):
            c = S[k][idx]
            if abs(c) < 1e-12: continue
            steps = []
            for x in idx:
                lvl = 0
                for d in range(len(steps) - 1, -1, -1):
                    if any(not independent(D, f, x) for f in steps[d]): lvl = d + 1; break
                if lvl == len(steps): steps.append([])
                steps[lvl].append(x)
            key = tuple(tuple(sorted(s)) for s in steps); acc[key] = acc.get(key, 0.0) + c
        out.append(tuple(sorted((kk, round(v, nd) + 0.0) for kk, v in acc.items() if abs(round(v, nd)) > 0)))
    return tuple(out)


def words(m, L):
    letters = [(e, s) for e in range(m) for s in (1, -1)]
    for n in range(L + 1):
        for w in itertools.product(letters, repeat=n): yield w


def faithful_on_ball(D, N, L):
    """on all words of length <= L: (homomorphism ok, injective on the ball, an offending pair if not injective)"""
    by_nf = {}; by_key = {}; hom = True
    for w in words(len(D["E"]), L):
        nf = raag_normal_form(D, w); k = braid_key(D, w, N)
        if nf in by_nf and by_nf[nf] != k: hom = False
        by_nf.setdefault(nf, k)
        if k in by_key and by_key[k][0] != nf: return hom, False, (by_key[k][1], w)
        by_key.setdefault(k, (nf, w))
    return hom, True, None
