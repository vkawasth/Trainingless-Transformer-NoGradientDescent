"""Checks for section 12.20 of the companion note (module pce/poisonflip.py): how little poisoned data tilts a decision, where it can be
placed, and how removal explains a change over time.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point (linear programs, iterative proportional fitting), with tolerances stated.
  PF1  CERTIFICATE  circuits of the consistent-edit space V (dim 2n) of the n-cycle: 188 for n = 4 (supports 4, 6, 8, 9; the CHSH
                    coordinates give the same census), 481 for n = 5 (supports 4, 6, 8, 10, 11); every support is at most 2n + 1; the
                    support-4 circuits number 5n for n = 4..8
  PF2  NUMERICAL    sparsity: a cheapest stealthy flip is a conformal sum of at most 2n circuits (Rockafellar); census of how many a vertex
                    solution uses (4-cycle, two gaps) and of how often one support-4 circuit (4 cells) is already optimal, n = 4..8
  PF3  NUMERICAL    two thresholds: c_some <= c_all globally, and along every tested direction t_some <= t_rule <= t_all for the maximum-
                    entropy rule
  PF4  CERTIFICATE  placement over the hidden structure: on all 2721 faces of D, dim G + fibre dim = |V(G)| - 1; the number of signed
                    circuits that keep the data in D is monotone under face inclusion (0 violations, exact rationals); the table by
                    (dim, fibre); vertex 13, CH facet 252, positivity facet 287, interior 376
  PF5  NUMERICAL    placement with a budget at the sparsity-matched face laws of Study 18 and at interior points: cheapest some-rule and
                    every-rule costs, and the number of single circuits that flip within g, 1.5 g and 2 g; what separates the points is
                    whether the decision is identified there (width 0), not the fibre dimension
  PF6  IDENTITY     the removal maps on information states (batches, contexts) compose, the batch-removal and context-forgetting squares
                    commute, and removing the poisoned batch from any later state gives the clean state exactly (integer counts)
  PF7  NUMERICAL    value of a context: forgetting contexts only widens the identified interval (every sub-cover of size 3 and 2, every time)
  PF8  NUMERICAL    duration of a one-batch flip on a stationary stream: every rule is flipped exactly while k / (N_t + k) >= q*, q* read off
                    the dual vertices
  PF9  NUMERICAL    attribution in the time example: at every flipped time only the poisoned batch's removal reverts the maximum-entropy
                    decision; the first-order estimate of that removal's effect and its error
"""
import os, sys, itertools, random, collections, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import poisonflip as P
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
T0 = time.time()
CELLS = G.CELLS; M4 = np.array(G.M, float); W = np.array([1.0 if c[0] == c[1] else 0.0 for c in CELLS])     # decision functional P(A0 = A1)
E4, _ = P.entries_chsh(); C4 = P.circuits(E4); SU4 = P.signed_unit(E4, C4); DV = P.dual_vertices(W); F = G.facets_D()

# ---------------------------------------------------------------- PF1
cen = lambda C: dict(sorted(collections.Counter(len(s) for s, _ in C).items()))
En4, _ = P.entries_ncycle(4); En5, _ = P.entries_ncycle(5); Cn4 = P.circuits(En4); Cn5 = P.circuits(En5)
ok = cen(C4) == {4: 20, 6: 32, 8: 72, 9: 64} and cen(Cn4) == cen(C4) and cen(Cn5) == {4: 25, 6: 40, 8: 80, 10: 176, 11: 160}
ok &= np.linalg.matrix_rank(E4) == 8 and np.linalg.matrix_rank(En5) == 10 and max(cen(Cn5)) <= 11
small = {}
for n in range(4, 9):
    En, _ = P.entries_ncycle(n); small[n] = P.circuits_upto(En, 4); ok &= len(small[n]) == 5 * n and np.linalg.matrix_rank(En) == 2 * n
check("PF1 CERTIFICATE circuits of the consistent-edit space: n = 4 and 5 censuses, supports <= 2n + 1, 5n support-4 circuits for n = 4..8",
      ok, f"n=4 (CHSH and cycle): {cen(C4)} = {len(C4)}; n=5: {cen(Cn5)} = {len(Cn5)}; support-4 counts {[len(small[n]) for n in range(4, 9)]}")

# ---------------------------------------------------------------- PF2
rng = np.random.default_rng(1); cnt = {}; okc = True
for g in (0.01, 0.05):
    c = collections.Counter()
    for s in range(40):
        x = rng.dirichlet(np.ones(16)); d = M4 @ x; L, U = P.lower_upper(d, W); tau = L - g
        if tau <= 0: continue
        cs, ds = P.flip_some(d, W, tau, E4); k, cc = P.conformal_count(E4, ds, SU4, d, W, tau)
        okc &= k is not None and k <= 8 and abs(cc - cs) < 1e-7; c[k] += 1
    cnt[g] = dict(sorted(c.items()))
four = {}
for n in range(4, 9):
    cells, Mn, _ = G.ncycle(n); Mn = np.array(Mn, float); En, _ = P.entries_ncycle(n); SU = P.signed_unit(En, small[n])
    w = np.array([1.0 if c[0] == c[2] else 0.0 for c in cells]); r = np.random.default_rng(n); hit = tot = 0
    for s in range(30):
        x = r.dirichlet(np.ones(len(cells))); d = Mn @ x; L, U = P.lower_upper(d, w, M=Mn); tau = L - 0.01
        if tau <= 0: continue
        cs, ds = P.flip_some(d, w, tau, En, M=Mn); one = min([t for t in (P.along_some(d, c, w, tau, M=Mn) for c in SU) if t is not None] + [np.inf])
        okc &= cs <= one + 1e-9; tot += 1; hit += abs(cs - one) < 1e-7
    four[n] = (hit, tot)
check("PF2 NUMERICAL cheapest stealthy flips are conformal sums of at most 2n circuits; how many a vertex uses; how often 4 cells suffice",
      okc, f"4-cycle, circuits used by a vertex optimum: gap 0.01 {cnt[0.01]}, gap 0.05 {cnt[0.05]}; one support-4 circuit optimal at gap 0.01: "
      + ", ".join(f"n={n}: {h}/{t}" for n, (h, t) in four.items()))

# ---------------------------------------------------------------- PF3
rng = np.random.default_rng(5); ok = True; nd = 0; ng = 0
me = lambda dd: P.maxent_rows(dd, list(range(8)))
for s in range(40):
    x = rng.dirichlet(np.ones(16)); d = M4 @ x; L, U = P.lower_upper(d, W)
    for tau in (L - 0.02, (L + U) / 2):
        if tau <= 0: continue
        cs = P.flip_some(d, W, tau, E4)[0] if L > tau else 0.0; ca = P.flip_all(d, W, tau, E4, DV)[0]; ok &= cs <= ca + 1e-9; ng += 1
for s in range(3):
    x = rng.dirichlet(np.ones(16)); d = M4 @ x; L, U = P.lower_upper(d, W); tau = float(W @ me(d)) - 0.03
    for i in rng.choice(len(SU4), 25, replace=False):
        c = SU4[i]; ts = P.along_some(d, c, W, tau); ta = P.along_all(d, c, W, tau, DV); tm = P.tmax_in_D(d, c)
        tr = P.along_rule(d, c, W, tau, me, tm)
        if tr is None: ok &= ta is None
        else: ok &= (ts is not None and ts <= tr + 1e-6) and (ta is None or tr <= ta + 1e-6); nd += 1
check("PF3 NUMERICAL two thresholds: c_some <= c_all, and t_some <= t_maxent <= t_all along each direction (1e-6, bisection)", ok,
      f"{ng} global comparisons, {nd} directions with a maximum-entropy flip")

# ---------------------------------------------------------------- PF4
faces = G.faces_D(F); Fexact = [(f[0], set(f[2])) for f in F]; ICS = [np.rint(v).astype(int) for _, v in C4]
def tan_exact(V):
    act = [h for h, T in Fexact if set(V) <= T]; n_ = 0
    for iv in ICS:
        for sg in (1, -1):
            if all(sum(Fr(h[r]) * sg * int(iv[r]) for r in range(8)) >= 0 for h in act): n_ += 1
    return n_
tab = {V: (tan_exact(V), G.face_dim(sorted(V)), G.fibre_dim_face(sorted(V))) for V in faces}
ok = len(faces) == 2721 and all(dm + fb == len(V) - 1 for V, (_, dm, fb) in tab.items())
viol = sum(1 for a in faces for b in faces if a < b and tab[a][0] > tab[b][0]); ok &= viol == 0
agg = collections.defaultdict(list)
for V, (tc, dm, fb) in tab.items(): agg[(dm, fb)].append(tc)
key = {k: (len(v), round(float(np.mean(v)), 1)) for k, v in sorted(agg.items())}
ok &= key[(0, 0)][1] == 13 and key[(7, 0)][1] == 252 and key[(7, 4)][1] == 287 and key[(8, 7)][1] == 376
check("PF4 CERTIFICATE placement is monotone in the hidden structure; dim + fibre = |V| - 1 on every face; counts by (dim, fibre)",
      ok, f"{len(faces)} faces, {viol} violations; (dim, fibre): (count, mean signed circuits) {key}")

# ---------------------------------------------------------------- PF5 (Study 18's faces, rebuilt with its seed and rules)
def s_faces():
    faces_ = sorted(G.faces_D(F), key=lambda V: sorted(V)); r = random.Random(20261108); fam = {}
    for name, (dm, fb) in (("F0", (7, 0)), ("F1", (6, 1)), ("F2", (5, 2))):
        fam[name] = r.sample([V for V in faces_ if len(V) == 8 and G.face_dim(V) == dm and G.fibre_dim_face(V) == fb], 3)
    return fam, r
fam, _ = s_faces(); r = random.Random(20261108); laws = []          # Study 18 draws the weights from a fresh generator
for f in ("F0", "F1", "F2"):
    for j, V in enumerate(fam[f]):
        wv = [r.randint(5, 25) for _ in V]; t = sum(wv); wv = [x * 240 // t for x in wv]; wv[0] += 240 - sum(wv)
        x = np.zeros(16)
        for k, v in zip(sorted(V), wv): x[k] = v / 240
        laws.append((f"{f}.{j}", x))
rng = np.random.default_rng(3); laws += [(f"int.{j}", rng.dirichlet(np.ones(16))) for j in range(3)]
rows = []; ok = True
for name, x in laws:
    d = M4 @ x; L, U = P.lower_upper(d, W); g = 0.05; tau = float(W @ x) - g
    cs = P.flip_some(d, W, tau, E4)[0] if L > tau else 0.0; ca = P.flip_all(d, W, tau, E4, DV)[0]
    ts = np.array([t if t is not None else np.inf for t in (P.along_some(d, c, W, tau) for c in SU4)])
    nb = tuple(int((ts <= b * g + 1e-9).sum()) for b in (1.0, 1.5, 2.0))
    rows.append((name, round(U - L, 3), round(cs, 4), round(ca, 4), nb)); ok &= cs <= ca + 1e-9
idf = [r_ for r_ in rows if abs(r_[1]) < 1e-9]; unid = [r_ for r_ in rows if abs(r_[1]) >= 1e-9]
ok &= all(r_[2] >= 0.05 - 1e-9 and r_[4][1] <= 4 for r_ in idf) and all(r_[2] == 0.0 and r_[4][0] == 376 for r_ in unid)
check("PF5 NUMERICAL budgeted placement, gap g = 0.05 below the true value: (name, width U-L, c_some, c_all, single circuits flipping some rule "
      "within g, 1.5 g, 2 g). Identified decisions (width 0) need an edit >= g and at most 4 single circuits do it within 1.5 g; "
      "unidentified ones flip along all 376 at zero data cost", ok, "; ".join(str(r_) for r_ in rows))

# ---------------------------------------------------------------- the stream used by PF6, PF7, PF9
P1 = [5 * int(round(v * 240)) for v in laws[0][1]]; P2 = [4 * a // 5 + 15 for a in P1]            # rigid CH-facet world, then 0.8 face + 0.2 uniform
units = lambda wts: [CELLS[i] for i in range(16) for _ in range(wts[i])]
C_ALL = (0, 1, 2, 3); TAU = 0.45; T_END = 22; T_P = 10
def build(poison=True):
    S = P.Stream(); info = None
    for t in range(T_END):
        B = list(range(len(S.counts)))
        if t == T_P and poison:
            d, _ = P.data_of(P.pooled(S, B, C_ALL)); N = len(B) * 1200; q, R = P.poison_mix(d, W, TAU, DV); k = int(np.ceil(q * N / (1 - q)))
            while True:
                Sx = P.Stream(); Sx.counts = list(S.counts); Sx.add(P.units_from_law(R, k), C_ALL)
                if P.lower_upper(P.data_of(P.pooled(Sx, B + [len(B)], C_ALL))[0], W)[1] <= TAU + 1e-12: break
                k += 1
            S.add(P.units_from_law(R, k), C_ALL); info = (q, k, N, R)
        else: S.add(units(P1 if t < 6 else P2), C_ALL)
    return S, info
S, (q0, k0, N0, R0) = build(); Sc, _ = build(False)

# ---------------------------------------------------------------- PF6
ok = True; rr = random.Random(7)
for _ in range(200):
    B = sorted(rr.sample(range(T_END), rr.randint(3, T_END))); B1 = sorted(rr.sample(B, rr.randint(2, len(B)))); B2 = sorted(rr.sample(B1, rr.randint(1, len(B1))))
    C = tuple(sorted(rr.sample(C_ALL, rr.randint(2, 4)))); C1 = tuple(sorted(rr.sample(C, rr.randint(1, len(C))))); C2 = tuple(sorted(rr.sample(C1, rr.randint(1, len(C1)))))
    o = P.pooled(S, B, C)
    ok &= P.remove(S, P.remove(S, o, B, B1, C1), B1, B2, C2) == P.remove(S, o, B, B2, C2) == P.pooled(S, B2, C2)      # composition
    ok &= P.remove(S, P.remove(S, o, B, B1, C), B1, B1, C1) == P.remove(S, P.remove(S, o, B, B, C1), B, B1, C1)        # the square commutes
pois = T_P; okp = all(P.remove(S, P.pooled(S, range(t + 1), C_ALL), range(t + 1), [b for b in range(t + 1) if b != pois], C_ALL)
                      == P.pooled(Sc, range(t), C_ALL) for t in range(T_P, T_END))
check("PF6 IDENTITY removal maps compose, batch removal commutes with forgetting contexts, removing the poisoned batch gives the clean state",
      ok and okp, f"200 random chains and squares; {T_END - T_P} later states; poisoned batch: {k0} units, all of reading(s) {list(np.nonzero(R0 > 1e-9)[0])}, against {N0} clean units (q* = {q0:.4f})")

# ---------------------------------------------------------------- PF7
ok = True; vo = []
for t in range(T_END):
    d, _ = P.data_of(P.pooled(S, range(t + 1), C_ALL)); full = P.lower_upper(d, W)
    for size in (3, 2):
        for C in itertools.combinations(C_ALL, size):
            lo, hi = P.lower_upper(d, W, rows=P.chsh_rows(C))
            ok &= lo <= full[0] + 1e-9 and hi >= full[1] - 1e-9
            if size == 3 and t == T_END - 1: vo.append(round((hi - lo) - (full[1] - full[0]), 3))
check("PF7 NUMERICAL forgetting contexts only widens the identified interval (value of a context >= 0)", ok,
      f"{T_END} times x 10 sub-covers; at the last time, value of each context (width without it minus width with all) {vo}")

# ---------------------------------------------------------------- PF8 (stationary stream)
Ss = P.Stream(); [Ss.add(units(P2), C_ALL) for _ in range(T_P)]
d, _ = P.data_of(P.pooled(Ss, range(T_P), C_ALL)); N = T_P * 1200; tau8 = P.lower_upper(d, W)[1] - 0.03
q, R = P.poison_mix(d, W, tau8, DV); k = int(np.ceil(q * N / (1 - q))) + 40; batch = P.units_from_law(R, k); Ss.add(batch, C_ALL)
p = np.zeros(8)
for u in batch: p += M4[:, CELLS.index(u)] / k
qs = []
for z in DV:
    y, y0 = z[:-1], z[-1]; a0 = y @ d + y0; a1 = y @ (p - d)
    if a1 < 0 and a0 > tau8: qs.append((tau8 - a0) / a1)
qstar = min(qs); agree = 0
for t in range(30):
    if t: Ss.add(units(P2), C_ALL)
    dt, _ = P.data_of(P.pooled(Ss, range(len(Ss.counts)), C_ALL)); obs = P.lower_upper(dt, W)[1] <= tau8 + 1e-10
    pred = k / (N + 1200 * t + k) >= qstar - 1e-12; agree += obs == pred
    if t == 0: first = obs
lasted = next(t for t in range(1, 400) if k / (N + 1200 * t + k) < qstar)
check("PF8 NUMERICAL a one-batch flip on a stationary stream lasts exactly while k / (N_t + k) >= q*", agree == 30 and first,
      f"k = {k} units, N = {N}, q* = {qstar:.4f}; prediction and observation agree at 30/30 times; every rule flipped for {lasted} batches")

# ---------------------------------------------------------------- PF9
dec = lambda d: float(W @ P.maxent_rows(d, list(range(8)))) >= TAU
ok = True; flipped = []; fo = []
for t in range(T_P, T_END):
    B = list(range(t + 1)); d, _ = P.data_of(P.pooled(S, B, C_ALL))
    if dec(d): continue
    flipped.append(t); rev = []
    for b in B:
        dm, _ = P.data_of(P.remove(S, P.pooled(S, B, C_ALL), B, [x for x in B if x != b], C_ALL))
        if dec(dm): rev.append(b)
    ok &= rev == [T_P]
    dm, _ = P.data_of(P.remove(S, P.pooled(S, B, C_ALL), B, [x for x in B if x != T_P], C_ALL))
    f = lambda v: float(W @ P.maxent_rows(v, list(range(8)))); ex = f(dm) - f(d); est = P.first_order(f, d, dm - d); fo.append((t, round(ex, 4), round(est, 4)))
dU = []
for t in range(T_P, T_END):
    d, _ = P.data_of(P.pooled(S, range(t + 1), C_ALL)); dU.append(P.lower_upper(d, W)[1] <= TAU + 1e-12)
check("PF9 NUMERICAL attribution: at every flipped time only the poisoned batch's single removal reverts the decision; first-order estimate",
      ok and len(flipped) > 0, f"maximum-entropy decision flipped at t = {flipped}; every-rule flip at {[T_P + i for i, v in enumerate(dU) if v]}; "
      f"(t, exact change of the decision value on removing the poisoned batch, first-order estimate) {fo}")
print(f"{sum(results)}/{len(results)} passed   ({time.time() - T0:.0f} s)")
