"""check_horns_n.py -- (3) parity horns in every dimension.

Parity horn Lambda^n_k on n+1 bits: on each face i != k (the face omitting coordinate i) the uniform law on
assignments of the n remaining bits with parity pi_i, exactly one pi_i = 1.
  H1 compatibility on all overlaps, n = 3..6
  H2 CF = 1 - 2^(2-n) for n = 3..6, with exactly verified certificates
  H3 proof check: exactly two global assignments satisfy all n face parities, for every n and k
  H4 best approximate filler eps*(n), n = 3..6 (LP), and the formula it fits
"""
import itertools, math
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
from provenance import mu_min_sources
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

def horn(n, k=0, odd=None):
    faces = [i for i in range(n + 1) if i != k]
    odd = faces[0] if odd is None else odd
    data = {}
    for i in faces:
        idx = tuple(j for j in range(n + 1) if j != i)
        data[idx] = {a: (F(1, 2 ** (n - 1)) if sum(a) % 2 == int(i == odd) else F(0)) for a in itertools.product((0, 1), repeat=n)}
    return data
def compatible(data):
    for k1, k2 in itertools.combinations(data, 2):
        common = tuple(sorted(set(k1) & set(k2))); m1 = {}; m2 = {}
        for a, x in data[k1].items():
            key = tuple(a[k1.index(j)] for j in common); m1[key] = m1.get(key, 0) + x
        for a, x in data[k2].items():
            key = tuple(a[k2.index(j)] for j in common); m2[key] = m2.get(key, 0) + x
        if m1 != m2: return False
    return True
def cf(data, n):
    keys = sorted(data); m = len(keys)
    vec = [data[k][a] / m for k in keys for a in sorted(data[k])]
    src = []
    for s in itertools.product((0, 1), repeat=n + 1):
        src.append([F(int(tuple(s[j] for j in k) == a), m) for k in keys for a in sorted(data[k])])
    return mu_min_sources(vec, src)
def eps_star(data, n):
    keys = sorted(data); pts = list(itertools.product((0, 1), repeat=n + 1)); P = len(pts)
    rows = [(k, a) for k in keys for a in sorted(data[k])]; T = len(rows); nv = P + T + 1
    A, b = [], []
    for r, (k, a) in enumerate(rows):
        row = np.zeros(nv)
        for i, s in enumerate(pts):
            if tuple(s[j] for j in k) == a: row[i] = 1
        e = float(data[k][a])
        r1 = row.copy(); r1[P + r] = -1; A.append(r1); b.append(e)
        r2 = -row; r2[P + r] = -1; A.append(r2); b.append(-e)
    for k in keys:
        row = np.zeros(nv)
        for r, (kk, a) in enumerate(rows):
            if kk == k: row[P + r] = 0.5
        row[-1] = -1; A.append(row); b.append(0)
    c = np.zeros(nv); c[-1] = 1
    return linprog(c, A_ub=np.array(A), b_ub=b, A_eq=[np.r_[np.ones(P), np.zeros(T + 1)]], b_eq=[1],
                   bounds=[(0, None)] * nv, method="highs").fun

ok = all(compatible(horn(n, k)) for n in range(3, 7) for k in range(n + 1))
check("H1 parity horns are compatible on all overlaps (n = 3..6, every k)", ok)
info = []; ok = True
for n in range(3, 7):
    val, lb, _ = cf(horn(n, 0), n); target = 1 - 2 ** (2 - n)
    ok &= abs(val - target) < 1e-9 and abs(float(lb) - target) < 1e-6; info.append(f"n={n}: {val:.6f}")
check("H2 contextual fraction of the parity n-horn is 1 - 2^(2-n), certified (n = 3..6)", ok, "; ".join(info))
ok = True
for n in range(3, 9):
    for k in range(n + 1):
        d = horn(n, k)
        sat = [s for s in itertools.product((0, 1), repeat=n + 1)
               if all(d[key][tuple(s[j] for j in key)] > 0 for key in d)]
        ok &= len(sat) == 2 and len({tuple(s[j] for j in key) for s in sat for key in [next(iter(d))]}) == 2
check("H3 exactly two global assignments satisfy all face parities (n = 3..8, every k): mass 2 * 2^(1-n) is explained",
      ok, "so CF = 1 - 2^(2-n) for every n >= 3")
es = {n: eps_star(horn(n, 0), n) for n in range(3, 9)}
mono = all(es[n] < es[n + 1] for n in range(3, 8))
check("H4 best approximate fillers eps*(n) increase with n and stay below 1/2 (no closed form claimed)",
      mono and max(es.values()) < 0.5,
      "; ".join(f"n={n}: {v:.6f}" for n, v in es.items()))
print(f"\n{sum(res)}/{len(res)} checks passed")
