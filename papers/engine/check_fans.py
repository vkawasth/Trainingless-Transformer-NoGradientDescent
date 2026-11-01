"""Checks for Step 4a-4b of "Probability, Context and Differentiation: The Layers": several parameters at once.
  MP1  arc limits (exact): along eps_i = c_i s^{w_i}, the conditioned state tends to the normalised initial forms in_w f(c) of the cells of
       least tropical order, whenever these are positive; checked against the one-parameter theory and against evaluation at small s
  MP2  the order fan (d = 2, exact): the leading face of the Newton polyhedron N = conv(supports) + R^2_{>=0} as w ranges over a grid of
       the open quadrant plus each edge's normal takes exactly V + (V - 1) values (V vertices, V - 1 bounded edges)
  MP3  the monomial criterion: for monomial cells the limit as eps -> 0 exists iff the exponents have a componentwise minimum; otherwise two
       weights give different limits; with a minimum, every arc and every random sequence approaches the same limit
  MP4  the small examples, exactly: the race (limits 1, 0 and c/(1+c)); dominance (1/2 along every arc); the curved path (every line gives 0,
       the parabola y = c x^2 gives 1/(1+c)); the degenerate direction ((x-y)^2 against x^3: c != 1 gives 1, the arc y = x + b x^{3/2}
       gives b^2/(1+b^2))
  MP5  iterated limits (exact): sending eps_{i_1} -> 0 first, then eps_{i_2}, ..., gives the normalised coefficients at the lexicographic
       minimum, a vertex of N, equal to the arc limit for a lexicographic (super-increasing) weight; nested numerical evaluation agrees;
       the two orders disagree in the race; in x^3 + y^3 against xy both corners give the first cell, the diagonal the second
  MP6  the tangent tower: for multilinear families, the Weil point Lambda(e_1, ..., e_d) lies in T^d Delta; the first monomial with mass on
       L x A in the lexicographic lemma's order gives the iterated limit sending eps_1 -> 0 first; the canonical flip (exchange of e_1, e_2)
       gives the other order
  MP7  lines are not enough: in the curved-path example every line through 0 gives the same limit, yet the limit does not exist
"""
import os, sys, json, random, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import fans as FN
from pce import orders as OD
from pce import tangent as TG
random.seed(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rpoly(d, nterms=3, maxexp=3, pos=True):
    f = {}
    for _ in range(nterms): f[tuple(Fr(random.randint(0, maxexp)) for _ in range(d))] = Fr(random.randint(1, 5))
    if not pos:                                               # a negative coefficient on a non-minimal monomial, kept non-negative overall
        a = max(f); f[tuple(x + 1 for x in a)] = Fr(-1, 100)
    return f

# MP1
ok = True; n = 0
for _ in range(300):
    d = random.choice([2, 3]); cells = {l: rpoly(d) for l in range(random.randint(2, 4))}
    w = tuple(Fr(random.randint(1, 4)) for _ in range(d)); c = tuple(Fr(random.randint(1, 3)) for _ in range(d))
    lim = FN.arc_limit(cells, w, c)
    t = min(FN.trop(f, w) for f in cells.values()); lead = {}
    for k, f in cells.items():
        if FN.trop(f, w) == t:
            v = sum(co * np.prod([Fr(ci) ** int(ai) for ci, ai in zip(c, a)]) for a, co in FN.initial(f, w).items())
            if v: lead[k] = v
    s = sum(lead.values()); ok &= lim == {k: v / s for k, v in lead.items()}
    sv = Fr(1, 10 ** 6); val = FN.evaluate(cells, [ci * sv ** int(wi) for ci, wi in zip(c, w)])
    ok &= sum(abs(val.get(k, 0) - lim.get(k, 0)) for k in cells) < Fr(1, 10 ** 4); n += 1
check("MP1 arc limits = normalised initial forms of the least-order cells (exact; agrees with evaluation at s = 1e-6)", ok, f"{n} random families, d = 2, 3")

# MP2
ok = True; counts = []
for _ in range(100):
    cells = {l: rpoly(2, nterms=random.randint(1, 4), maxexp=5) for l in range(3)}; V = FN.newton_vertices_2d(cells)
    types = set()
    edges = [(V[i], V[i + 1]) for i in range(len(V) - 1)]
    ws = [(Fr(a), Fr(b)) for a in range(1, 13) for b in range(1, 13)] + [(p[1] - q[1], q[0] - p[0]) for p, q in edges]   # plus each edge's normal
    for w in ws:
        if True:
            t = min(FN.trop(f, w) for f in cells.values())
            face = tuple(sorted({al for f in cells.values() for al in FN.initial(f, w) if FN.trop(f, w) == t}))
            types.add(face)
    ok &= len(types) == 2 * len(V) - 1 and all(len(f) == 1 for f in types if f[0] in V and len(f) == 1)
    verts_seen = {f[0] for f in types if len(f) == 1}; ok &= verts_seen == set(V); counts.append((len(V), len(types)))
check("MP2 the order fan: leading faces = the V vertices and V-1 bounded edges of the Newton polyhedron (every one seen)", ok,
      f"(V, types) e.g. {counts[:6]}")

# MP3
ok = True; withmin = nomin = 0
for _ in range(300):
    d = 2; cells = {l: {tuple(Fr(random.randint(0, 3)) for _ in range(d)): Fr(random.randint(1, 5))} for l in range(random.randint(2, 4))}
    cm = FN.componentwise_min(cells)
    lims = {tuple(sorted(FN.arc_limit(cells, (Fr(a), Fr(b)), (Fr(1), Fr(1))).items())) for a in range(1, 7) for b in range(1, 7)}
    if cm is not None:
        withmin += 1; target = FN.vertex_limit(cells, cm); ok &= lims == {tuple(sorted(target.items()))}
        for _ in range(5):                                     # random points near 0, not on any arc
            e = [Fr(1, random.randint(10 ** 5, 10 ** 6)), Fr(1, random.randint(10 ** 5, 10 ** 6))]; val = FN.evaluate(cells, e)
            ok &= sum(abs(val.get(k, 0) - target.get(k, 0)) for k in cells) < Fr(1, 10 ** 3)
    else:
        nomin += 1; ok &= len(lims) >= 2
check("MP3 monomial cells: the limit exists iff the exponents have a componentwise minimum; otherwise two weights disagree", ok, f"{withmin} with a minimum, {nomin} without")

# MP4
ok = True
race = {1: {(Fr(1), Fr(1)): Fr(1)}, 2: {(Fr(2), Fr(0)): Fr(1), (Fr(2), Fr(1)): Fr(-1)}}          # Lambda(1,A) = t s, Lambda(2,A) = (1 - t) s^2; eps = (s, t)
ok &= FN.arc_limit(race, (Fr(1), Fr(2)), (1, 1)).get(1, 0) == 0 and FN.arc_limit(race, (Fr(2), Fr(1)), (1, 1)).get(1, 0) == 1
for c in (Fr(1), Fr(3), Fr(1, 2)): ok &= FN.arc_limit(race, (Fr(1), Fr(1)), (Fr(1), c))[1] == c / (1 + c)
dom = {1: {(Fr(1), Fr(1)): Fr(1)}, 2: {(Fr(1), Fr(1)): Fr(1), (Fr(2), Fr(1)): Fr(1)}}
ok &= all(FN.arc_limit(dom, (Fr(a), Fr(b)), (Fr(1), Fr(c))) == {1: Fr(1, 2), 2: Fr(1, 2)} for a in (1, 2, 3) for b in (1, 2, 5) for c in (1, 2))
curved = {1: {(Fr(2), Fr(0)): Fr(1)}, 2: {(Fr(0), Fr(1)): Fr(1)}}                                # cells x^2 and y
ok &= all(FN.arc_limit(curved, (Fr(1), Fr(1)), (Fr(1), Fr(v))).get(1, 0) == 0 for v in (1, 2, 5))
for c in (Fr(1), Fr(2), Fr(1, 3)): ok &= FN.arc_limit(curved, (Fr(1), Fr(2)), (Fr(1), c))[1] == 1 / (1 + c)
degen = {1: {(Fr(2), Fr(0)): Fr(1), (Fr(1), Fr(1)): Fr(-2), (Fr(0), Fr(2)): Fr(1)}, 2: {(Fr(3), Fr(0)): Fr(1)}}   # (x - y)^2 and x^3
for c in (Fr(2), Fr(1, 2)): ok &= FN.arc_limit(degen, (Fr(1), Fr(1)), (Fr(1), c)) == {1: Fr(1)}
for b in (Fr(1), Fr(2), Fr(1, 3)):                          # arc x = s^2, y = s^2 + b s^3 (that is y = x + b x^{3/2})
    g1 = OD.mul({Fr(3): -b}, {Fr(3): -b}); g2 = {Fr(6): Fr(1)}
    r = OD.cond_limit({(1, "a"): g1, (2, "a"): g2}, lambda x: True); ok &= r["limit"][(1, "a")] == b * b / (1 + b * b)
check("MP4 small examples: race (1, 0, c/(1+c)); dominance (1/2 always); curved path (lines 0, parabolas 1/(1+c)); degenerate direction (b^2/(1+b^2))", ok)

# MP5
ok = True; disagree = 0
for _ in range(300):
    d = random.choice([2, 3]); cells = {l: rpoly(d) for l in range(random.randint(2, 4))}
    for order in itertools.permutations(range(d)):
        lim, alpha = FN.iterated_limit(cells, order)
        W = [0] * d
        for rank, i in enumerate(order): W[i] = Fr(10 ** (2 * (d - rank)))   # the variable sent first gets the largest weight
        ok &= FN.arc_limit(cells, tuple(W), tuple([Fr(1)] * d)) == lim
        if d == 2:                                                            # nested numerics: first variable much smaller
            e = [0.0, 0.0]; e[order[0]] = 1e-40; e[order[1]] = 1e-6       # first variable below the second to every power used
            val = FN.evaluate({k: {a: float(v) for a, v in f.items()} for k, f in cells.items()}, e)
            ok &= sum(abs(val.get(k, 0) - float(lim.get(k, 0))) for k in cells) < 1e-3
    if d == 2 and FN.iterated_limit(cells, (0, 1))[0] != FN.iterated_limit(cells, (1, 0))[0]: disagree += 1
corners = {1: {(Fr(3), Fr(0)): Fr(1), (Fr(0), Fr(3)): Fr(1)}, 2: {(Fr(1), Fr(1)): Fr(1)}}      # x^3 + y^3 against xy
ok &= FN.iterated_limit(corners, (0, 1))[0] == FN.iterated_limit(corners, (1, 0))[0] == {1: Fr(1)} and FN.arc_limit(corners, (Fr(1), Fr(1)), (1, 1)) == {2: Fr(1)}
rl0 = FN.iterated_limit(race, (1, 0))[0].get(1, 0); rl1 = FN.iterated_limit(race, (0, 1))[0].get(1, 0); ok &= rl0 == 0 and rl1 == 1
check("MP5 iterated limits = lexicographic vertex = arc limit for a super-increasing weight; orders can disagree (race: t first 0, s first 1); corners can agree while the middle differs (x^3+y^3 vs xy)", ok,
      f"{disagree} random 2-parameter families with disagreeing orders")

# MP6
def weil_point(cells_full, d):
    """cells_full: {key: multilinear poly}; returns the (2^d, ncells) coefficient array of Lambda(e_1, ..., e_d)"""
    keys = sorted(cells_full); c = np.zeros((2 ** d, len(keys)))
    for j, k in enumerate(keys):
        for a, v in cells_full[k].items(): c[sum(1 << i for i in range(d) if a[i] == 1), j] += float(v)
    return keys, c
ok = True; nflip = 0
for _ in range(200):
    d = random.choice([2, 3]); onA = {}
    for l in range(random.randint(2, 3)):
        f = {}
        for _ in range(random.randint(1, 3)):
            a = tuple(Fr(random.randint(0, 1)) for _ in range(d))
            if any(a): f[a] = Fr(random.randint(1, 4), 10)
        if f: onA[(l, "A")] = f
    if len(onA) < 2: continue
    full = dict(onA); off = {tuple([Fr(0)] * d): Fr(1)}
    for f in onA.values():
        for a, v in f.items(): off[a] = off.get(a, Fr(0)) - v
    full[(9, "off")] = off
    keys, c = weil_point(full, d); ok &= TG.in_T(c, d)
    Aidx = [j for j, k in enumerate(keys) if k[1] == "A"]
    for S in TG.lex_order(2 ** d - 1):
        if np.any(c[S, Aidx]): break
    tot = c[S, Aidx].sum(); weil = {keys[j]: c[S, j] / tot for j in Aidx if c[S, j]}
    lim = FN.iterated_limit(onA, tuple(range(d)))[0]
    ok &= set(weil) == set(lim) and all(abs(weil[k] - float(lim[k])) < 1e-12 for k in lim)
    if d == 2:                                                            # the canonical flip swaps e_1, e_2: the other iterated limit
        cf = c[[0, 2, 1, 3]]
        for S in TG.lex_order(3):
            if np.any(cf[S, Aidx]): break
        tot = cf[S, Aidx].sum(); wf = {keys[j]: cf[S, j] / tot for j in Aidx if cf[S, j]}
        lim2 = FN.iterated_limit(onA, (1, 0))[0]; ok &= all(abs(wf.get(k, 0) - float(lim2.get(k, 0))) < 1e-12 for k in onA); nflip += 1
check("MP6 the tower: the Weil point Lambda(e_1..e_d) is in T^d Delta; the lexicographic lemma's first A-term is the iterated limit (eps_1 first); the flip gives the other order", ok,
      f"{nflip} flips checked")

# MP7
lines = {FN.arc_limit(curved, (Fr(1), Fr(1)), (Fr(1), Fr(v))).get(1, 0) for v in range(1, 20)} | {FN.arc_limit(curved, (Fr(1), Fr(1)), (Fr(v), Fr(1))).get(1, 0) for v in range(1, 20)}
para = {FN.arc_limit(curved, (Fr(1), Fr(2)), (Fr(1), Fr(v))).get(1, 0) for v in range(1, 6)}
check("MP7 lines are not enough: every line gives share 0 for x^2, yet parabolas give 1/2, 1/3, ...: no limit", lines == {0} and len(para) == 5, f"line limits {lines}, parabola limits {sorted(para)}")
OUT["race_iterated"] = [str(rl0), str(rl1)]
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "fans.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
