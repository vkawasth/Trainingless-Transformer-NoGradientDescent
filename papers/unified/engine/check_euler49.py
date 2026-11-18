"""Study 49: the certified vanishing-cycle lattice of the 2x3 example against the Euler form of dP6 (EU1-EU5). Pre-registered in
study49_euler/PREREG_study49.md, frozen by SHA-256 before this code ran. Exact integer/rational arithmetic throughout.
Output: check_euler49_output.txt, study49_euler/result.json.
"""
import os, json, itertools, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "study49_euler")
OUT = open(os.path.join(HERE, "check_euler49_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)

# ---------------------------------------------------------------- dP6: intersection theory on Bl_3 P^2 ------------------------------------------
# divisors as vectors (h, e1, e2, e3) meaning h H + sum e_i E_i; intersection H^2 = 1, E_i^2 = -1
def dot(a, b): return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
K = (-3, 1, 1, 1)
def line(Dv): return dict(r=1, c1=Dv, ch2=Fr(dot(Dv, Dv), 2))
def exc(i): e = [0, 0, 0, 0]; e[i] = 1; return dict(r=0, c1=tuple(e), ch2=Fr(-1, 2))          # O_{E_i}(-1)
def chi(E, F):
    a = tuple(E["r"] * x - F["r"] * y for x, y in zip(F["c1"], E["c1"]))                      # r_E c1_F - r_F c1_E
    return E["r"] * F["r"] + Fr(-dot(a, K), 2) + E["r"] * F["ch2"] + F["r"] * E["ch2"] - dot(E["c1"], F["c1"])
COLL = [exc(1), exc(2), exc(3), line((0, 0, 0, 0)), line((1, 0, 0, 0)), line((2, 0, 0, 0))]
NAMES = ["O_E1(-1)", "O_E2(-1)", "O_E3(-1)", "O", "O(H)", "O(2H)"]
CHI = [[chi(E, F) for F in COLL] for E in COLL]
W = [(E["r"], dot(E["c1"], K)) for E in COLL]

# ---------------------------------------------------------------- lattice tools ---------------------------------------------------------------
def det(a, b): return a[0] * b[1] - a[1] * b[0]
def T(a, x, c): d = det(a, x); return (x[0] + c * d * a[0], x[1] + c * d * a[1])
def Tm(a, c): return np.array([[1 - c * a[0] * a[1], c * a[0] * a[0]], [-c * a[1] * a[1], 1 + c * a[0] * a[1]]])
def gram(vs): return tuple(det(a, b) for a in vs for b in vs)
SIGNS = [np.array((1,) + s) for s in itertools.product((1, -1), repeat=5)]
def canon(vs):
    G = np.array(gram(vs)).reshape(6, 6); best = None
    for d in SIGNS:
        for g in (1, -1):
            M = tuple((g * (d[:, None] * G * d[None, :])).flatten())
            if best is None or M < best: best = M
    return best
def moves(vs, c):
    out = []
    for i in range(5):
        a, b = vs[i], vs[i + 1]
        out.append(((i + 1, 1), vs[:i] + (T(a, b, c), a) + vs[i + 2:]))           # sigma_i
        out.append(((i + 1, -1), vs[:i] + (b, T(b, a, -c)) + vs[i + 2:]))          # sigma_i^-1
    return out
def bfs(start, c, depth):
    seen = {canon(start): ()}; frontier = [(start, ())]
    for _ in range(depth):
        nxt = []
        for vs, w in frontier:
            for m, v2 in moves(vs, c):
                k = canon(v2)
                if k not in seen: seen[k] = w + (m,); nxt.append((v2, w + (m,)))
        frontier = nxt
    return seen

if __name__ == "__main__":
    say(f"==== run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ok1 = all(CHI[i][i] == 1 for i in range(6)) and all(CHI[j][i] == 0 for i in range(6) for j in range(i + 1, 6))
    check("EU1 IDENTITY the reference collection is exceptional (chi upper unitriangular)", ok1,
          f"chi = {[[str(x) for x in row] for row in CHI]}")
    A = [[CHI[i][j] - CHI[j][i] for j in range(6)] for i in range(6)]
    sgn = [s for s in (1, -1) if all(A[i][j] == s * det(W[i], W[j]) for i in range(6) for j in range(6))]
    check("EU2 IDENTITY chi - chi^T = (global sign) x det pairing of w = (rank, c1.K)", len(sgn) == 1, f"w = {W}; sign {sgn}")
    prods = {}
    for c in (1, -1):
        for order in ("forward", "reverse"):
            P = np.eye(2, dtype=int)
            for a in (W if order == "forward" else W[::-1]): P = Tm(a, c) @ P
            prods[(c, order)] = P.tolist()
    def unip6(P): P = np.array(P); return int(np.trace(P)) == 2 and int(np.gcd.reduce([abs(int(x)) for x in (P - np.eye(2, dtype=int)).flatten()])) == 6
    ok3 = any(unip6(P) for P in prods.values())
    check("EU3 HYPOTHESIS invariant match: product of the transvections of the w's is unipotent with off-diagonal 6 (ours: [[1,6],[0,1]])", ok3,
          f"products {prods}")
    ours = {"t=-1/10": [(-1, 0), (-1, 0), (-3, 1), (-2, 1), (0, 1), (3, 1)], "t=+1/10": [(-1, 0), (-3, 1), (-2, 1), (-1, 1), (-2, 1), (0, 1)]}
    res = {}
    for c in (1, -1):
        t0 = time.time(); fromD = bfs(tuple(W), c, 6); say(f"  c = {c}: dP6 side {len(fromD)} classes at depth 6 ({time.time() - t0:.0f}s)")
        for name, vs in ours.items():
            t0 = time.time(); fromO = bfs(tuple(vs), c, 6); common = set(fromD) & set(fromO)
            best = None
            for k in common:
                w = list(fromD[k]) + [(i, -e) for i, e in reversed(fromO[k])]          # dP6 -> common -> ours
                if best is None or len(w) < len(best): best = w
            res[(c, name)] = best; say(f"  c = {c}, {name}: our side {len(fromO)} classes; meeting classes {len(common)}; shortest word {best} ({time.time() - t0:.0f}s)")
    found = {k: v for k, v in res.items() if v is not None}
    cs = {c for (c, _), v in found.items()}
    ok4 = any(all(res[(c, n)] is not None for n in ours) for c in (1, -1))
    check("EU4 HYPOTHESIS orbit match: one convention c relates dP6's chi - chi^T to our Gram matrices at both ends by a Hurwitz word of length <= 12",
          ok4, "; ".join(f"c = {c}, {n}: {'word of length ' + str(len(v)) if v is not None else 'none within depth 6 + 6'}" for (c, n), v in res.items()))
    rec = {}
    for (c, n), w in found.items():
        vs = list(W)
        for i, e in w:
            a, b = vs[i - 1], vs[i]
            vs[i - 1:i + 1] = [T(a, b, c), a] if e == 1 else [b, T(b, a, -c)]
        rec[f"c={c}, {n}"] = dict(word=w, mutated_w=vs, ours=ours[n], gram_equal_up_to_signs=canon(tuple(vs)) == canon(tuple(ours[n])))
    check("EU5 RECORD words and mutated collections", True, str(rec))
    json.dump(dict(chi=[[str(x) for x in r] for r in CHI], w=W, products={str(k): v for k, v in prods.items()}, words={str(k): v for k, v in res.items()}, rec=rec),
              open(os.path.join(D, "result.json"), "w"), indent=1, default=str)
