"""Checks for sections 12.15-12.18 of the companion note (module pce/margsec.py): sections of the marginal projection, and the 4-cycle cover.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point (linear programs, iterative proportional fitting), with tolerances stated.
  MS1  IDENTITY     both rules are sections (pi s(d) = d) on random data of the 4-cycle; maximum entropy is interior; both are equivariant under the
                    symmetries of the cover (automorphisms of the 4-cycle times bit flips)
  MS2  IDENTITY     on chains the maximum-entropy section is the Markov chain and is inner-face consistent (marginalising X1 out of the 3-chain
                    section gives the 2-chain section of (g1 o g2, g3)); the leximin section is not (the top composite of Theorem 45)
  MS3  CERTIFICATE  the facets of D = pi(J) for the 4-cycle, exactly: 24 = the 16 inequalities of the local polytope L and 8 Clauser-Horne
                    inequalities, so D = L cut by CH (Fine's theorem)
  MS4  CERTIFICATE  the fibre over the relative interior of every face G of D has dimension |V(G)| - 1 - dim G: the face table; interior 7,
                    positivity facets 4, CH facets 0; every face inside a CH facet is a simplex (a point fibre); the formula against the
                    numerical fibre dimension at a relative-interior point of a sample of faces
  MS5  CERTIFICATE  contextual data: on the symmetric path (S = 4 lam) the data lie in L for lam <= 1 and in D only for lam <= 1/2; the fibre is
                    empty for lam > 1/2 (infeasible linear program), the PR box (lam = 1) included
  MS6  CERTIFICATE  on the symmetric path the data's symmetry group has order 16 and its fixed equations with the fibre's have rank 16, so every
                    equivariant section takes the same value for every lam; NUMERICAL: the identified interval of the unobserved
                    q = P(A0 = 0, A1 = 0) is [lam/2, (1 - lam)/2] (1e-9 on 21 values)
  MS7  NUMERICAL    generic paths to three kinds of boundary face: the fibre width and the disagreement of the two sections go to 0 at a CH
                    facet, and stay positive at a positivity facet (fibre 4) and at a 6-face with fibre 3
  MS8  CERTIFICATE  Simpson by selection: a binary chain whose selected joint law has positive conditional covariance of (X, Z) in both slices
                    and negative marginal covariance; the Markov section has zero conditional covariance; the same data admit a joint law with
                    no reversal
  MS9  CERTIFICATE  Simpson by coarse-graining the middle: Markov gluing gives positive covariance with three middle states and negative after
                    merging two of them
"""
import os, sys, itertools, random
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import selcomp as S
from pce import selnat as N
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
random.seed(2)
def combo(V):
    w = [Fr(random.randint(1, 9)) for _ in V]; t = sum(w); return [sum(wi * G.VERTS[k][r] for wi, k in zip(w, V)) / t for r in range(8)]
IDX = {c: i for i, c in enumerate(G.CELLS)}
CYC = [0, 2, 1, 3]
AUTOS = []
for r_ in range(4):
    for refl in (False, True):
        order = [CYC[(r_ + (-k if refl else k)) % 4] for k in range(4)]; AUTOS.append({CYC[k]: order[k] for k in range(4)})
def act(perm, flip, c):
    out = [0] * 4
    for v in range(4): out[perm[v]] = c[v] ^ flip[v]
    return tuple(out)
def push(perm, flip, x):
    y = [0] * 16
    for i, c in enumerate(G.CELLS): y[IDX[act(perm, flip, c)]] = x[i]
    return y
def data_of(x): return [sum(G.M[r][k] * x[k] for k in range(16)) for r in range(8)]

# MS1
ok = True; worst = 0.0
for _ in range(4):
    d = combo(range(16)); me = G.maxent(d); lx = G.leximin(d)
    ok &= max(abs(a - float(b)) for a, b in zip(data_of(me), d)) < 1e-10 and data_of(lx) == d and min(me) > 1e-6
    perm = random.choice(AUTOS); flip = tuple(random.randint(0, 1) for _ in range(4))
    x0 = [Fr(0)] * 16
    for k in range(16): x0[k] = Fr(1, 16)
    gd = data_of(push(perm, flip, [Fr(v).limit_denominator(10 ** 12) for v in lx]))
    ok &= G.leximin(gd) == push(perm, flip, lx)
    worst = max(worst, max(abs(a - b) for a, b in zip(G.maxent(gd), push(perm, flip, list(me)))))
ok &= worst < 1e-9
check("MS1 IDENTITY maximum entropy and leximin are sections of pi, equivariant under the symmetries of the 4-cycle; maximum entropy is interior", ok,
      f"4 random data points; max-entropy equivariance to {worst:.1e}; leximin exact")

# MS2
def chain_maxent(gs, iters=4000):
    shape, cells, A, b = S.fibre_constraints(gs); x = np.full(len(cells), 1 / len(cells)); idx = np.array(cells)
    for _ in range(iters):
        for k, g in enumerate(gs):
            for i in range(shape[k]):
                for j in range(shape[k + 1]):
                    m = (idx[:, k] == i) & (idx[:, k + 1] == j); cur = x[m].sum(); x[m] *= float(g[i][j]) / cur
    return cells, x
PB = [Fr(2, 5), Fr(1, 3), Fr(3, 7), Fr(1, 4)]; TB = [Fr(13, 60), Fr(7, 30), Fr(1, 80)]
GB = [N.cpl(PB[i], PB[i + 1], TB[i]) for i in range(3)]
cells, x = chain_maxent(GB)
markov_chain = np.array([float(GB[0][c[0]][c[1]] * GB[1][c[1]][c[2]] * GB[2][c[2]][c[3]] / (N.cols(GB[0])[c[1]] * N.cols(GB[1])[c[2]])) for c in cells])
face = {}
for c, v in zip(cells, x): face[(c[0], c[2], c[3])] = face.get((c[0], c[2], c[3]), 0) + v
c2, x2 = chain_maxent([N.markov(GB[0], GB[1]), GB[2]])
ok = np.abs(x - markov_chain).max() < 1e-12 and max(abs(face[c] - v) for c, v in zip(c2, x2)) < 1e-12
top = S.composite(GB, frozenset()); left = S.composite(GB, frozenset({(0, 2)}))
ok &= top[0][0] != left[0][0]
check("MS2 IDENTITY on chains maximum entropy is the Markov chain and inner-face consistent; leximin is not (top composite != a bracketing)", ok,
      f"top {top[0][0]} vs ((g1 g2) g3) {left[0][0]}")

# MS3
F = G.facets_D(); Lset = {(tuple(c), c0) for c, c0 in G.local_ineqs()}
pos = [f for f in F if (tuple(map(Fr, f[0])), Fr(f[1])) in Lset]; ch = [f for f in F if (tuple(map(Fr, f[0])), Fr(f[1])) not in Lset]
ok = len(F) == 24 and len(pos) == 16 and len(ch) == 8 and all(len(f[2]) == 8 for f in ch) and all(len(f[2]) == 12 for f in pos)
check("MS3 CERTIFICATE facets of D: the 16 local inequalities and 8 Clauser-Horne inequalities (D = L cut by CH)", ok,
      "CH: " + "; ".join(f"{f[0]}.d + {f[1]} >= 0" for f in ch[:2]) + "; ... (8)")

# MS4
faces = G.faces_D(F); from collections import Counter
tab = Counter((G.face_dim(V), len(V), G.fibre_dim_face(V)) for V in faces)
inCH = [V for V in faces if any(V <= frozenset(f[2]) for f in ch)]
ok = all(G.fibre_dim_face(V) == 0 for V in inCH) and tab[(8, 16, 7)] == 1 and tab[(7, 12, 4)] == 16 and tab[(7, 8, 0)] == 8
sample = random.sample(sorted(faces, key=lambda V: sorted(V)), 40); bad = 0
for V in sample:
    d = combo(sorted(V)); bad += G.fibre_dim(d) != G.fibre_dim_face(V)
ok &= bad == 0
check("MS4 CERTIFICATE fibre dimension over each face of D is |V| - 1 - dim G; interior 7, positivity facets 4, CH facets and all their faces 0", ok,
      f"{len(faces)} faces; (dim, |V|, fibre) counts {sorted(tab.items())}; {len(inCH)} faces inside CH facets; formula = numerical dim on 40 sampled faces")

# MS5
ok = True; inL = lambda d: all(sum(c[r] * d[r] for r in range(8)) + c0 >= 0 for c, c0 in G.local_ineqs())
inD = lambda d: all(sum(Fr(f[0][r]) * d[r] for r in range(8)) + f[1] >= 0 for f in F)
for lam in [Fr(k, 20) for k in range(21)]:
    d = G.chsh_path(lam); ok &= inL(d) and (inD(d) == (lam <= Fr(1, 2))) and ((G.fibre_dim(d) is None) == (lam > Fr(1, 2)))
check("MS5 CERTIFICATE the symmetric path: in L for lam <= 1, in D (non-contextual) exactly for lam <= 1/2; empty fibre beyond (PR box at lam = 1)", ok)

# MS6
d = G.chsh_path(Fr(2, 5)); me = G.maxent(d)
gens = [(p, f) for p in AUTOS for f in itertools.product((0, 1), repeat=4) if max(abs(a - float(b)) for a, b in zip(data_of(push(p, f, list(me))), d)) < 1e-10]
A = [r[:] for r in G.M] + [[1] * 16]
for p, f in gens:
    for i, c in enumerate(G.CELLS):
        j = IDX[act(p, f, c)]
        if j != i: row = [0] * 16; row[i] = 1; row[j] -= 1; A.append(row)
rk = S._rank(A); ok = len(gens) == 16 and rk == 16; err = 0.0
for lam in [Fr(k, 40) for k in range(21)]:
    lo, hi = G.interval(G.chsh_path(lam), G.W_A0A1); err = max(err, abs(lo - float(lam) / 2), abs(hi - (1 - float(lam)) / 2))
ok &= err < 1e-9
lx = G.leximin(d); ok &= max(abs(float(a) - b) for a, b in zip(lx, me)) < 1e-9
check("MS6 CERTIFICATE symmetric path: symmetry group of order 16, fixed fibre a point (rank 16), so all equivariant sections agree; q interval [lam/2, (1-lam)/2]", ok,
      f"rank {rk}; interval error {err:.1e}; leximin = maximum entropy at lam = 2/5")

# MS7
random.seed(2); d0 = combo(range(16)); chf = ch[0][2]; pf = pos[0][2]; f3 = [V for V in faces if G.face_dim(V) == 6 and len(V) == 10][0]
rows = []; ok = True
for name, V in (("CH facet", chf), ("positivity facet", pf), ("6-face", f3)):
    d1 = combo(sorted(V)); out = []
    for lam in (Fr(0), Fr(9, 10), Fr(99, 100), Fr(1)):
        dd = [(1 - lam) * a + lam * b for a, b in zip(d0, d1)]; me = G.maxent(dd, iters=200000, tol=1e-13); lx = np.array([float(v) for v in G.leximin(dd)])
        out.append((float(lam), G.fibre_dim(dd), G.cell_width(dd), 0.5 * np.abs(me - lx).sum()))
    rows.append((name, out))
ch_end, pos_end, f3_end = rows[0][1][-1], rows[1][1][-1], rows[2][1][-1]
ok = ch_end[1] == 0 and ch_end[2] < 1e-9 and ch_end[3] < 1e-4 and rows[0][1][2][3] < rows[0][1][1][3] and pos_end[1] == 4 and pos_end[3] > 0.02 and f3_end[1] == 3 and f3_end[3] > 0.02
check("MS7 NUMERICAL toward a CH facet the fibre and the disagreement of the sections vanish; toward a positivity facet or a 6-face they do not", ok,
      "; ".join(f"{n}: " + ", ".join(f"lam {a}: dim {b}, width {c:.4f}, TV {e:.4f}" for a, b, c, e in o) for n, o in rows))

# MS8
g = [[Fr(5, 24), Fr(11, 24)], [Fr(1, 24), Fr(7, 24)]]; dd = [[Fr(3, 23), Fr(11, 92)], [Fr(15, 22), Fr(3, 44)]]
p = g[0][0] + g[0][1]; r = dd[0][0] + dd[1][0]; sl = []
for b in range(2):
    q = g[0][b] + g[1][b]; rb, sb = g[0][b], dd[b][0]; mid = (max(Fr(0), rb + sb - q) + min(rb, sb)) / 2; sl.append(mid / q - (rb / q) * (sb / q))
comp = S.select_binary(g, dd) - p * r; lo, hi = N.identified(g, dd)
Ux = [];
for b in range(2):
    q = g[0][b] + g[1][b]; rb, sb = g[0][b], dd[b][0]; Ux.append(min(rb, sb) / q - (rb / q) * (sb / q))
ok = all(v > 0 for v in sl) and comp < 0 and N.markov(g, dd)[0][0] - p * r < 0 and all(v >= 0 for v in Ux) and hi - p * r > 0
check("MS8 CERTIFICATE Simpson by selection: positive conditional covariance in both slices, negative marginal; the data also admit no reversal", ok,
      f"g = {N.show(g)}, d = {N.show(dd)}; slice covariances {[str(v) for v in sl]}, marginal {comp}; Markov marginal {N.markov(g, dd)[0][0] - p * r} with zero slices; "
      f"compatible marginal covariances [{lo - p * r}, {hi - p * r}]")

# MS9
g = [[Fr(4, 21), Fr(3, 14), Fr(2, 21)], [Fr(1, 7), Fr(2, 21), Fr(11, 42)]]; dd = [[Fr(1, 9), Fr(2, 9)], [Fr(65, 252), Fr(13, 252)], [Fr(5, 32), Fr(45, 224)]]
GR = [[0], [1, 2]]; p = sum(g[0]); r = sum(dd[j][0] for j in range(3))
m1 = N.markov(g, dd)[0][0] - p * r; m2 = N.markov(N.merge_cols(g, GR), N.merge_rows(dd, GR))[0][0] - p * r
s1 = N.sel(g, dd)[0][0] - p * r; s2 = N.sel(N.merge_cols(g, GR), N.merge_rows(dd, GR))[0][0] - p * r
check("MS9 CERTIFICATE Simpson by coarse-graining: Markov covariance positive with three middle states, negative after merging two", m1 > 0 and m2 < 0,
      f"g = {N.show(g)}, d = {N.show(dd)}; Markov {m1} -> {m2}; selection {s1} -> {s2}")

# MS10
ok = True; info = []
def gamma_ineqs(n):
    from math import gcd
    out = []
    for gam in itertools.product((1, -1), repeat=n):
        if np.prod(gam) != -1: continue
        c = [Fr(0)] * (2 * n); c0 = Fr(n - 2)
        for i in range(n): c0 -= gam[i]; c[n + i] += 2 * gam[i]; c[n + (i + 1) % n] += 2 * gam[i]; c[i] -= 4 * gam[i]
        g_ = 0
        for x in list(c) + [c0]: g_ = gcd(g_, abs(int(x)))
        out.append((tuple(int(x) // g_ for x in c), int(c0) // g_))
    return out
results_n6 = True
for n in (6, 7, 8):
    cells, Mn, Vn = G.ncycle(n)
    for kind, lst, nv in (("local", G.ncycle_local(n), 3 * 2 ** (n - 2)), ("gamma", gamma_ineqs(n), 2 * n)):
        for c, c0 in lst:
            ev = [sum(Fr(c[r]) * v[r] for r in range(2 * n)) + c0 for v in Vn]; tight = [Vn[k] for k in range(len(Vn)) if ev[k] == 0]
            results_n6 &= min(ev) >= 0 and len(tight) == nv and G._affine_rank(tight) == 2 * n - 1
ok = True; info = []
for n in (3, 4, 5):
    cells, Mn, Vn = G.ncycle(n); Fn = G.facets_hull(Vn); Ls = {(tuple(c), c0) for c, c0 in G.ncycle_local(n)}
    loc = [f for f in Fn if (tuple(map(Fr, f[0])), Fr(f[1])) in Ls]; non = [f for f in Fn if (tuple(map(Fr, f[0])), Fr(f[1])) not in Ls]
    ok &= len(loc) == 4 * n and len(non) == 2 ** (n - 1) and all(len(f[2]) == 3 * 2 ** (n - 2) for f in loc) and all(len(f[2]) == 2 * n for f in non)
    # the gamma form: n - 2 - sum gamma_i (1 - 2 P(X_i != X_i+1)) >= 0, with P(!=) = s_i + s_(i+1) - 2 t_i
    want = set()
    for gam in itertools.product((1, -1), repeat=n):
        if np.prod(gam) != -1: continue
        c = [Fr(0)] * (2 * n); c0 = Fr(n - 2)
        for i in range(n):
            c0 -= gam[i]; c[n + i] += 2 * gam[i]; c[n + (i + 1) % n] += 2 * gam[i]; c[i] -= 4 * gam[i]
        from math import gcd
        g_ = 0
        for x in list(c) + [c0]: g_ = gcd(g_, abs(int(x)))
        want.add(tuple(int(x) // g_ for x in c) + (int(c0) // g_,))
    have = {tuple(f[0]) + (f[1],) for f in non}; ok &= have == want
    info.append(f"n={n}: {len(Fn)} facets")
ok &= results_n6
check("MS10 CERTIFICATE n-cycles: 4n local facets and 2^(n-1) noncontextuality facets, the latter simplices with 2n vertices, in the gamma form", ok, "; ".join(info) + "; n=6, 7, 8: all listed inequalities valid facets with the stated vertex counts")

# MS11
import mpmath as mp
gq = N.cpl(Fr(1, 4), Fr(1, 2), Fr(1, 5)); dq = N.cpl(Fr(1, 2), Fr(3, 5), Fr(1, 5)); gu, du = N.umbrella()
def tdefect(q):
    c1, s1 = G.chain_power_section([gu, du], q); c2, s2 = G.chain_power_section([gq, dq], q); cp, sp = G.chain_power_section([N.kron(gu, gq), N.kron(du, dq)], q)
    prod = {}
    for a1, v1 in zip(c1, s1):
        for a2, v2 in zip(c2, s2): prod[tuple(2 * a + b for a, b in zip(a1, a2))] = v1 * v2
    return float(mp.fsum(abs(v - prod[c]) for c, v in zip(cp, sp)) / 2)
QS = [1, Fr(99, 100), Fr(9, 10), Fr(1, 2), 0, -1, -4, -8]; DEF = [tdefect(q) for q in QS]
ok = DEF[0] < 1e-12 and all(DEF[i + 1] > DEF[i] for i in range(len(DEF) - 1)) and DEF[1] > 1e-4
# c < 0: uniform binary chain, both couplings independent; the slice flip x -> 1/4 - x preserves the fibre and every entropy
h = Fr(1, 4); gU = [[h, h], [h, h]]; shape, cells, A, b = S.fibre_constraints([gU, gU])
def joint(x0, x1):                                   # x_b = P(X0 = 0, X1 = b, X2 = 0), b = 0, 1
    out = {}
    for b_ in (0, 1):
        xb = (x0, x1)[b_]; out[(0, b_, 0)] = xb; out[(0, b_, 1)] = h - xb; out[(1, b_, 0)] = h - xb; out[(1, b_, 1)] = xb
    return [out[c] for c in cells]
def H(x): return -sum(float(v) * np.log(float(v)) for v in x if v > 0)
verts = [joint(a_, b_) for a_ in (Fr(0), h) for b_ in (Fr(0), h)]
ok2 = all(all(sum(A[r][k] * v[k] for k in range(len(cells))) == b[r] for r in range(len(A))) for v in verts)
Hv = [H(v) for v in verts]; ok2 &= len(set(round(x, 12) for x in Hv)) == 1     # all four vertices minimise, with equal entropy
check("MS11 NUMERICAL tensor-square defect along the power family: 0 only at q = 1 (Shannon), increasing as q decreases; CERTIFICATE c H with c < 0 has 4 minimisers",
      ok and ok2, ", ".join(f"q={float(q)}: {d_:.2e}" for q, d_ in zip(QS, DEF)) + f"; facet-leximin defect for comparison 0.0752; min entropy {Hv[0]:.4f} at all 4 vertices")
print(f"{sum(results)}/{len(results)} passed")
