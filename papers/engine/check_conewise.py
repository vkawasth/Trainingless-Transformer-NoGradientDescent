"""Checks for Lemma "the spread order" and Theorem "poison order against tie order on the fan" of "Probability, Context and
Differentiation: The Layers" (module pce/conewise.py). Exact arithmetic.
  CW1  the spread order from fan data equals the slope of the exact conditional variance of the gap, on 300 random families of
       non-negative cells with gaps in {-2,...,2}, in two and three parameters, along generic arcs (no initial form vanishes at the
       leading coefficient) with random higher-order terms
  CW2  on decision walls (limiting gap 0) rho = v_dec - v(P); the blueprint: rho = 1 (outer cone), 1/2 (middle cone), 0 (slope-2 ray,
       where two leading cells carry different gaps), 1 (the ray c = 1)
  CW3  the sample law in every cone and on every wall: N* ~ tau^-(2v(N) - v_dec) on the blueprint's slope-2 ray and the outer and middle cones
  CW4  the poisoned act on the fan: for 400 random signed G, E in two and three parameters, the sign of G + E along arcs (exact evaluation
       at t = 10^-25, with random higher-order terms) is the sign of the lower-order initial form, or of their sum on a tie
  CW5  the fragility exponent p*(w) = trop_w G - trop_w H is piecewise linear on the common refinement: a share t^p flips the act iff
       p < p* when the initial forms have opposite signs (exact, 300 cases, p = p* -+ 1)
  CW6  the blueprint as a corollary: relabelling the uninformative cell, p*(w) equals the tie-order map m(w) in the middle cones and is
       negative in the outer cones; shares t^(m/2) flip the act in the middle cones
"""
import os, sys, random
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import conewise as CW
from pce import degen_d as DD
from pce import blueprint as BP
random.seed(20261102); results = []; DEBUG = os.environ.get('CWDEBUG')
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def mul(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items():
            k = tuple(i + j for i, j in zip(a, b)); out[k] = out.get(k, 0) + x * y
    return {k: v for k, v in out.items() if v != 0}
def rand_cell(d):
    if random.random() < 0.5: return {tuple(random.randint(0, 3) for _ in range(d)): Fr(random.randint(1, 3))}
    i, j = random.sample(range(d), 2); e1 = [0] * d; e2 = [0] * d; e1[i] = 1; e2[j] = 1
    lin = {tuple(e1): Fr(random.randint(1, 3)), tuple(e2): Fr(-random.randint(1, 3))}
    return mul(mul(lin, lin), {tuple(random.randint(0, 2) for _ in range(d)): Fr(1)})
def rand_arc(w, c):
    arc = [{w[i]: Fr(c[i])} for i in range(len(w))]
    for i in range(len(w)): arc[i][w[i] + random.randint(1, 4)] = Fr(random.randint(-3, 3))
    return arc
def total(cells):
    P = {}
    for f in cells.values():
        for k, v in f.items(): P[k] = P.get(k, 0) + v
    return {k: v for k, v in P.items() if v != 0}

# CW1, CW2
ok1 = True; ok2 = True; n = nw = 0
for _ in range(600):
    d = random.choice([2, 3]); cells = {k: rand_cell(d) for k in range(random.randint(2, 4))}; gaps = {k: random.randint(-2, 2) for k in cells}
    w = [random.randint(1, 3) for _ in range(d)]; c = [Fr(random.randint(1, 4), random.randint(1, 3)) for _ in range(d)]
    if any(CW.init_val(f, w, c) == 0 for f in cells.values()): continue           # generic arcs: every cell's order is its tropical order
    rho = CW.spread_order(cells, gaps, w, c); arc = rand_arc(w, c); t1, t2 = Fr(1, 10 ** 12), Fr(1, 10 ** 18)
    s1, s2 = CW.spread_exact(cells, gaps, arc, t1), CW.spread_exact(cells, gaps, arc, t2)
    if rho is None: ok1 &= s1 == 0 and s2 == 0
    else:
        slope = (np.log(float(s1)) - np.log(float(s2))) / (np.log(float(t1)) - np.log(float(t2))); good = abs(slope - float(rho)) < 0.02; ok1 &= good
        if not good and DEBUG: print('MISMATCH', cells, gaps, w, c, rho, slope)
    n += 1
    vals = {k: CW.init_val(f, w, c) if CW.trop(f, w) == CW.trop(total(cells), w) else Fr(0) for k, f in cells.items()}
    if sum(gaps[k] * v for k, v in vals.items()) == 0 and rho is not None:
        vP = CW.trop(total(cells), w); vdec = min(CW.trop(f, w) for k, f in cells.items() if gaps[k] != 0)
        ok2 &= rho == vdec - vP; nw += 1
    if n >= 300: break
cells = {k: v for k, v in zip(range(4), [{(3, 0): Fr(1)}, {(2, 1): Fr(7, 4)}, {(1, 1): Fr(1)}, {(0, 3): Fr(1)}])}; gaps = dict(zip(range(4), (1, -1, 0, -1)))
bp = {"outer (1,3)": CW.spread_order(cells, gaps, (1, 3), (1, 1)), "middle (2,3)": CW.spread_order(cells, gaps, (2, 3), (1, 1)) / 2,
      "slope-2 ray (1,2)": CW.spread_order(cells, gaps, (1, 2), (1, 1)), "ray c = 1 (1,1)": CW.spread_order(cells, gaps, (1, 1), (1, 1))}
check("CW1 the spread order from fan data = the slope of the exact conditional variance (300 families, two and three parameters)", ok1, f"{n} families")
check("CW2 on walls rho = v_dec - v(P); the blueprint: outer 1, middle 1/2, slope-2 ray 0, ray c = 1 gives 1", ok2 and
      [bp[k] for k in bp] == [1, Fr(1, 2), 0, 1], f"{nw} wall cases; {', '.join(f'{k}: {v}' for k, v in bp.items())}")

# CW3
ok = True; rows = []; taus = np.geomspace(1e-6, 1e-14, 6)
Npoly = {(3, 0): Fr(1), (2, 1): Fr(-7, 4), (0, 3): Fr(-1)}
for w in [(1, 2), (1, 3), (1, 1.5)]:
    wf = (Fr(w[0]).limit_denominator(), Fr(w[1]).limit_denominator()); vN = CW.trop(Npoly, wf)
    vdec = min(CW.trop(cells[k], wf) for k in cells if gaps[k] != 0); e = 2 * vN - vdec
    Ns = [BP.n_star(t ** w[0], t ** w[1]) for t in taus]; sl = float(np.polyfit(np.log(1 / taus), np.log(Ns), 1)[0])
    ok &= abs(sl - float(e)) < 0.01; rows.append((w, str(e), round(sl, 3)))
check("CW3 the sample law N* ~ tau^-(2v(N) - v_dec) in every cone and on the slope-2 ray (exact Skellam)", ok, f"(w, e, fitted) {rows}")

# CW4
ok = True; n = flips = 0
def rand_signed(d, k=3): return {tuple(random.randint(0, 3) for _ in range(d)): Fr(random.choice([-3, -2, -1, 1, 2, 3])) for _ in range(k)}
while n < 400:
    d = random.choice([2, 3]); G, E = rand_signed(d), rand_signed(d); w = [random.randint(1, 3) for _ in range(d)]
    c = [Fr(random.randint(1, 4), random.randint(1, 3)) for _ in range(d)]; pred = CW.poisoned_sign(G, E, w, c)
    if pred is None: continue
    arc = rand_arc(w, c); t = Fr(1, 10 ** 25); GE = dict(G)
    for k, v in E.items(): GE[k] = GE.get(k, 0) + v
    val = sum(co * CW._prod(arc, a, t) for a, co in GE.items()); ok &= (val > 0) == (pred > 0)
    flips += (pred > 0) != (CW.init_val(G, w, c) > 0); n += 1
check("CW4 the poisoned act on the fan: the lower-order initial form decides, their sum on a tie (400 arcs, exact)", ok, f"{flips} flips")

# CW5
ok = True; n = 0
while n < 300:
    d = random.choice([2, 3]); G, H = rand_signed(d), rand_signed(d); w = [random.randint(1, 3) for _ in range(d)]
    c = [Fr(random.randint(1, 4), random.randint(1, 3)) for _ in range(d)]
    ig, ih = CW.init_val(G, w, c), CW.init_val(H, w, c)
    if ig == 0 or ih == 0 or (ig > 0) == (ih > 0): continue
    ps = CW.critical_share(G, H, w); arc = rand_arc(w, c); t = Fr(1, 10 ** 25)
    for p in (ps - 1, ps + 1):
        val = sum(co * CW._prod(arc, a, t) for a, co in G.items()) + t ** int(p) * sum(co * CW._prod(arc, a, t) for a, co in H.items())
        ok &= ((val > 0) != (ig > 0)) == (p < ps)
    n += 1
check("CW5 the fragility exponent p* = trop G - trop H: a share t^p flips iff p < p* (300 cases, exact)", ok)

# CW6
ok = True; rows = []
H = {(1, 1): Fr(-1)}
for w in [(Fr(1), Fr(1, 4)), (Fr(1), Fr(3, 4)), (Fr(1), Fr(5, 4)), (Fr(1), Fr(3, 2)), (Fr(1), Fr(3)), (Fr(4), Fr(1))]:
    ps = CW.critical_share(Npoly, H, w); m = CW.trop(Npoly, w) - CW.trop(total(cells), w); middle = Fr(1, 2) < w[1] / w[0] < 2
    ok &= (ps == m) if middle else ps < 0; rows.append((str(w[1] / w[0]), str(ps)))
    if middle:
        q = 4 * w[0].denominator * w[1].denominator; W = (int(w[0] * q), int(w[1] * q)); pp = ps * q / 2   # integer exponents
        t = Fr(1, 10 ** 6); s_, tt = t ** W[0], t ** W[1]
        clean = s_ ** 3 - Fr(7, 4) * s_ ** 2 * tt - tt ** 3; sign_clean = clean > 0
        val = clean - (1 if sign_clean else -1) * t ** int(pp) * s_ * tt          # relabel with the sign that opposes the clean act
        ok &= (val > 0) != sign_clean
check("CW6 the blueprint as a corollary: p* = tie order m in the middle cones, < 0 outside; shares t^(m/2) flip the act", ok, f"(slope, p*) {rows}")
print(f"{sum(results)}/{len(results)} passed")
