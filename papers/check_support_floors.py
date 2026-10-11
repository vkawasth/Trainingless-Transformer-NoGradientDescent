"""check_support_floors.py -- support floors, policies and leaf-spring chains (Tower Section 11, Conditionals Section 3.3,
Family Section 11), in exact rational arithmetic except where an LP is stated.

Sets of laws on a finite base are convex polytopes in the simplex, stored by their vertices (bases of size 2 or 3).
  SF1  policy imposition iota_tau(L) = L cap {rho >= tau} is an interior operator; floor summarisation
       sigma(L) = {rho >= min_L rho} is a closure operator (deflationary/inflationary, idempotent, monotone)
  SF2  normal oplax transport of floor boxes: F(k) = sigma o k#; F(eta) = id on boxes; F(hk) <= F(h)F(k) (inclusion), strict cases found
  SF3  policies before transport certify at least as much: k# iota_tau L  subset of  iota_{k# tau}(k# L), strict cases found
  SF4  leaf-spring chains: path independence without vertical steps; inserting sigma or a weakening enlarges the end set,
       inserting a policy shrinks it; impose-early / summarise-late gives the smallest summarised end set, and imposing
       early is contained in imposing the pushed floor late
  SF5  telescoping attribution of lost certification: nonnegative terms summing exactly to the loss
  SF6  binary triangle: the floored gluing headroom is (1/2) min over the 12 pairwise cells and the 4 antipodal sums, the
       antipodal sums are 1 - (1/2)(sum of disagreement probabilities, with relabelling); equals the LP value (float LP)
  SF7  the explicit support-contextual triangle (cells 000 and 111 empty, others 1/6): glues, local floors hold for tau <= 1/12,
       no global section meets any positive floor
  SF8  reduced margins: the coupling polytope of (1/2,1/2) x (1/3,1/3,1/3) is a hexagon; a floor 1/10 on cell (0,0) keeps it
       a hexagon, a floor 1/5 gives the shifted reduced-margin polytope, a quadrilateral (two blow-downs by policy)
"""
from fractions import Fraction as Fr
import itertools, random, sys
import numpy as np
from scipy.optimize import linprog
random.seed(20261011); ok = {}

# ---------------------------------------------------------------- polytopes in the simplex (n = 2 or 3), exact
def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
def hull(P):
    P = sorted(set(tuple(p) for p in P)); n = len(P[0]) if P else 0
    if n == 2:
        lo, hi = min(P, key=lambda p: p[0]), max(P, key=lambda p: p[0]); return [lo] if lo == hi else [lo, hi]
    pts = sorted(set((p[0], p[1]) for p in P))
    if len(pts) <= 2: return [(a, b, 1 - a - b) for a, b in pts]
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return [(a, b, 1 - a - b) for a, b in lower[:-1] + upper[:-1]]
def push(K, V): return hull([tuple(sum(K[j][i] * v[i] for i in range(len(v))) for j in range(len(K))) for v in V])
def floor(V): return tuple(min(v[i] for v in V) for i in range(len(V[0])))
def box(m):
    s = 1 - sum(m); return hull([tuple(m[i] + (s if i == b else 0) for i in range(len(m))) for b in range(len(m))])
def sigma(V): return box(floor(V))
def clip(V, i, t):          # keep rho_i >= t
    n = len(V[0])
    if n == 2:
        a, b = (V[0], V[-1]); pts = [p for p in V if p[i] >= t]
        if len(V) == 2 and (a[i] - t) * (b[i] - t) < 0:
            lam = (t - a[i]) / (b[i] - a[i]); pts.append(tuple(a[k] + lam * (b[k] - a[k]) for k in range(2)))
        return hull(pts) if pts else []
    out = []
    for k in range(len(V)):
        p, q = V[k], V[(k + 1) % len(V)]; pin, qin = p[i] >= t, q[i] >= t
        if pin: out.append(p)
        if pin != qin and len(V) > 1:
            lam = (t - p[i]) / (q[i] - p[i]); out.append(tuple(p[j] + lam * (q[j] - p[j]) for j in range(3)))
    return hull(out) if out else []
def iota(V, tau):
    for i, t in enumerate(tau):
        if t > 0 and V: V = clip(V, i, t)
    return V
def inside(V, p):           # p in conv(V), exact
    n = len(p)
    if not V: return False
    if n == 2: lo, hi = min(v[0] for v in V), max(v[0] for v in V); return lo <= p[0] <= hi
    if len(V) == 1: return tuple(V[0]) == tuple(p)
    if len(V) == 2:
        a, b = V; return cross(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    return all(cross(V[k], V[(k + 1) % len(V)], p) >= 0 for k in range(len(V)))
def subset(A, B): return all(inside(B, a) for a in A)          # conv(A) subset conv(B)
def same(A, B): return subset(A, B) and subset(B, A)
def rq(d=6): return Fr(random.randint(1, d), d)
def rlaw(n, d=12):
    w = [random.randint(1, d) for _ in range(n)]; s = sum(w); return tuple(Fr(x, s) for x in w)
def rkernel(n, m): return [list(c) for c in zip(*[rlaw(m) for _ in range(n)])]     # K[j][i] = k(i)(j), columns are laws
def compose(H, K): return [[sum(H[a][j] * K[j][i] for j in range(len(K))) for i in range(len(K[0]))] for a in range(len(H))]
def rset(n): return hull([rlaw(n) for _ in range(random.randint(2, 4))])
def apply(chain, V):
    for st in chain:
        if st[0] == 'k': V = push(st[1], V)
        elif st[0] == 's': V = sigma(V)
        elif st[0] == 'i': V = iota(V, st[1])
        elif st[0] == 'w': V = hull(V + [st[1]])
    return V

# SF1
r1 = True
for _ in range(200):
    n = random.choice([2, 3]); L = rset(n); M = hull(L + [rlaw(n)]); tau = tuple(Fr(random.randint(0, 2), 12) for _ in range(n))
    a, b = iota(L, tau), sigma(L)
    r1 &= (not a or subset(a, L)) and same(iota(a, tau), a) if a else True
    r1 &= subset(L, b) and same(sigma(b), b) and subset(sigma(L), sigma(M))
    aM = iota(M, tau)
    r1 &= (not a) or subset(a, aM)
ok['SF1_interior_closure'] = r1

# SF2
r2, strict2 = True, 0
for _ in range(300):
    n, m, p = random.choice([2, 3]), random.choice([2, 3]), random.choice([2, 3]); K, Hm = rkernel(n, m), rkernel(m, p)
    Bx = sigma(rset(n)); stepwise = sigma(push(Hm, sigma(push(K, Bx)))); composite = sigma(push(compose(Hm, K), Bx))
    r2 &= subset(composite, stepwise) and same(sigma(Bx), Bx); strict2 += int(not same(composite, stepwise))
ok['SF2_normal_oplax'] = r2 and strict2 > 0

# SF3
r3, strict3 = True, 0
for _ in range(300):
    n, m = random.choice([2, 3]), random.choice([2, 3]); K = rkernel(n, m); L = rset(n); tau = tuple(Fr(random.randint(0, 2), 12) for _ in range(n))
    a = iota(L, tau)
    if not a: continue
    left = push(K, a); ktau = tuple(sum(K[j][i] * tau[i] for i in range(n)) for j in range(m)); right = iota(push(K, L), ktau)
    r3 &= subset(left, right); strict3 += int(not same(left, right))
ok['SF3_policy_before_transport'] = r3 and strict3 > 0

# SF4
r4 = True
for _ in range(150):
    sizes = [random.choice([2, 3]) for _ in range(4)]; Ks = [rkernel(sizes[i], sizes[i + 1]) for i in range(3)]; L = rset(sizes[0])
    comp = compose(Ks[2], compose(Ks[1], Ks[0]))
    r4 &= same(apply([('k', K) for K in Ks], L), push(comp, L))                                 # path independence
    pos = random.randint(0, 3); base = [('k', K) for K in Ks]
    with_s = base[:pos] + [('s',)] + base[pos:]; r4 &= subset(apply(base, L), apply(with_s, L))
    w = rlaw(sizes[pos]); with_w = base[:pos] + [('w', w)] + base[pos:]; r4 &= subset(apply(base, L), apply(with_w, L))
    tau = tuple(Fr(random.randint(0, 1), 12) for _ in range(sizes[pos])); with_i = base[:pos] + [('i', tau)] + base[pos:]
    e = apply(with_i, L); r4 &= (not e) or subset(e, apply(base, L))
    # ordering rule: one policy on the start base, one summary; impose first and summarise last is smallest
    t0 = tuple(Fr(random.randint(0, 1), 12) for _ in range(sizes[0]))
    if not iota(L, t0): continue
    best = apply([('i', t0)] + base + [('s',)], L)
    for sp in range(4):
        other = base[:sp] + [('s',)] + base[sp:]
        r4 &= subset(best, sigma(apply([('i', t0)] + other, L)))
        # pushing the floor forward instead of imposing: the pushed policy at the end
        kt = t0
        for K in Ks: kt = tuple(sum(K[j][i] * kt[i] for i in range(len(kt))) for j in range(len(K)))
        late = iota(apply(other, L), kt)
        r4 &= (not late) or subset(apply([('i', t0)] + base, L), late)
ok['SF4_leaf_spring_chains'] = r4

# SF5
r5 = True
for _ in range(150):
    sizes = [random.choice([2, 3]) for _ in range(4)]; Ks = [rkernel(sizes[i], sizes[i + 1]) for i in range(3)]; L = rset(sizes[0])
    chain = []
    for K in Ks:
        chain.append(('k', K))
        if random.random() < 0.6: chain.append(('s',))
    vert = [i for i, st in enumerate(chain) if st[0] == 's']; f = [Fr(random.randint(-3, 3)) for _ in range(sizes[-1])]
    def cert(ch): V = apply(ch, L); return min(sum(fi * v[i] for i, fi in enumerate(f)) for v in V)
    c = [cert([st for i, st in enumerate(chain) if st[0] != 's' or i in vert[j:]]) for j in range(len(vert) + 1)]
    # c[0] = actual chain (all summaries); c[j] drops the first j summaries; c[-1] = no summaries
    terms = [c[j + 1] - c[j] for j in range(len(c) - 1)]
    r5 &= all(t >= 0 for t in terms) and sum(terms) == c[-1] - c[0]
ok['SF5_telescoping'] = r5

# SF6 binary triangle
cells = list(itertools.product((0, 1), repeat=3))
def margins(J): return {(0, 1): {(a, b): sum(J[c] for c in cells if (c[0], c[1]) == (a, b)) for a in (0, 1) for b in (0, 1)},
                        (1, 2): {(a, b): sum(J[c] for c in cells if (c[1], c[2]) == (a, b)) for a in (0, 1) for b in (0, 1)},
                        (2, 0): {(a, b): sum(J[c] for c in cells if (c[2], c[0]) == (a, b)) for a in (0, 1) for b in (0, 1)}}
def headroom_closed(Mg):
    pair = [v for t in Mg.values() for v in t.values()]
    def dis(ctx, flip): t = Mg[ctx]; return sum(v for (a, b), v in t.items() if (a != b) != flip)
    anti = []
    for c in cells[:4]:                       # c and its complement; relabel by flipping the variables where c has 1
        f = c                                 # flip pattern: pair (i,j) counts as 'disagree' when bits differ after flipping
        anti.append(1 - Fr(1, 2) * (dis((0, 1), f[0] != f[1]) + dis((1, 2), f[1] != f[2]) + dis((2, 0), f[2] != f[0])))
    return min(min(pair) / 2, min(anti) / 2), anti
def headroom_lp(Mg):
    A, b = [], []
    for (i, j), t in Mg.items():
        for (a, bb), v in t.items(): A.append([1.0 if (c[i], c[j]) == (a, bb) else 0.0 for c in cells] + [0.0]); b.append(float(v))
    Aub = [[-1.0 if k == ci else 0.0 for k in range(8)] + [1.0] for ci in range(8)]
    r = linprog([0] * 8 + [-1], A_ub=Aub, b_ub=[0] * 8, A_eq=A, b_eq=b, bounds=[(None, None)] * 9, method='highs'); return -r.fun if r.status == 0 else None
r6 = True; agree = 0
for _ in range(300):
    w = [random.randint(0, 6) for _ in range(8)]
    if sum(w) == 0: continue
    J = {c: Fr(x, sum(w)) for c, x in zip(cells, w)}; Mg = margins(J); hc, anti = headroom_closed(Mg)
    anti_direct = [J[c] + J[tuple(1 - x for x in c)] for c in cells[:4]]
    r6 &= anti == anti_direct; lp = headroom_lp(Mg); agree += int(lp is not None and abs(lp - float(hc)) < 1e-9)
ok['SF6_triangle_closed_form'] = r6 and agree >= 299

# SF7 the explicit example
J = {c: (Fr(0) if c in ((0, 0, 0), (1, 1, 1)) else Fr(1, 6)) for c in cells}; Mg = margins(J); hc, anti = headroom_closed(Mg)
local = min(v for t in Mg.values() for v in t.values()) / 2
ok['SF7_support_contextual_example'] = hc == 0 and local == Fr(1, 12) and sorted(anti) == [0, Fr(1, 3), Fr(1, 3), Fr(1, 3)] and abs(headroom_lp(Mg)) < 1e-12

# SF8 reduced margins on the hexagon
def tvertices(f, g, lower):
    n, m = len(f), len(g); N = n * m; out = set()
    for Z in itertools.combinations(range(N), N - (n + m - 1)):
        B = [i for i in range(N) if i not in Z]; x = {i: lower[i] for i in Z}
        rows = [f[a] - sum(x.get(a * m + j, 0) for j in range(m)) for a in range(n)]; cols = [g[j] - sum(x.get(a * m + j, 0) for a in range(n)) for j in range(m)]
        # solve the tree system for the basic cells by peeling leaves
        rem = set(B); R, C = rows[:], cols[:]; val = {}; progress = True
        while rem and progress:
            progress = False
            for i in list(rem):
                a, j = divmod(i, m)
                if sum(1 for k in rem if k // m == a) == 1: val[i] = R[a]; rem.discard(i); C[j] -= R[a]; R[a] = 0; progress = True
                elif sum(1 for k in rem if k % m == j) == 1: val[i] = C[j]; rem.discard(i); R[a] -= C[j]; C[j] = 0; progress = True
        if rem or any(r != 0 for r in R) or any(c != 0 for c in C): continue
        full = [x.get(i, val.get(i)) for i in range(N)]
        if all(full[i] >= lower[i] for i in range(N)): out.add(tuple(full))
    return out
f, g = (Fr(1, 2), Fr(1, 2)), (Fr(1, 3), Fr(1, 3), Fr(1, 3)); hexv = tvertices(f, g, [0] * 6)
low = [Fr(1, 5)] + [0] * 5; flo = tvertices(f, g, low); fr = (f[0] - Fr(1, 5), f[1]); gr = (g[0] - Fr(1, 5), g[1], g[2])
shift = {tuple(v[i] + low[i] for i in range(6)) for v in tvertices(fr, gr, [0] * 6)}
ok['SF8_reduced_margins_hexagon'] = len(hexv) == 6 and len(flo) == 4 and flo == shift and len(tvertices(f, g, [Fr(1, 10)] + [0] * 5)) == 6

print({k: bool(v) for k, v in ok.items()}, 'strict SF2', strict2, 'strict SF3', strict3)
print('PASS' if all(ok.values()) else 'FAIL'); sys.exit(0 if all(ok.values()) else 1)
