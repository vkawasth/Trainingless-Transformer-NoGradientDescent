"""Checks for 'The conditional credal field' (module pce/credalfield.py). Toy model: target Y, four pathways, 49 times in a 120-minute
window, p_i = q = 1/2, star cover (each context is a pathway with Y), study values known up to tolerances.
  CQ1 IDENTITY  star cover: the field F(t) is the product of the intervals [(a_i - e_i)/p_i, (a_i + e_i)/p_i] (support function of F equals
                that of the box in 40 random directions at each of 7 times)
  CQ2 IDENTITY  orders: decided at t iff the intervals are disjoint; per pair the order sequence over the window; certified crossings
                (decided before, decided opposite after) for five pairs, the pair light/arousal never decided; every one of 200 sampled laws
                agrees with every decided order
  CQ3 IDENTITY  only the parity of swaps is fixed: on a certified crossing there are laws in the field with one swap and with three swaps
                (one value per time, inside the bands, since the times are separate fibres)
  CQ4 IDENTITY  relational data decide what the intervals leave open: with P(L=1, A=0, Y=1) <= eps and P(L=0, A=1, Y=1) >= delta,
                v_A - v_L >= (delta - eps)/p, attained, for three settings
  CQ5 IDENTITY  coordinated poisoning on a star cover: every edit inside the Frechet bounds leaves the fibre nonempty (no contradiction);
                a coalition controlling sources i and j un-certifies a certified order i > j with total shift g = L_i - U_j and flips it
                only with total shift U_i - L_j (bisection on the LP)
  CQ6 IDENTITY  anchors: an independent measurement of pathway i with half-width eta makes the fibre empty iff i's shift exceeds
                w_i + eta; at the hiding shift it re-certifies the order
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import credalfield as CF
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(3); M = CF.MODEL; k, q, p = M["k"], M["q"], M["p"]

# CQ1
ok = True
for t in np.linspace(0, 120, 7):
    a, e = CF.data_at(t); L = (a - e) / p; U = (a + e) / p
    for _ in range(40):
        c = rng.standard_normal(k); lo, hi = CF.value_range(k, q, p, a, e, c)
        ok &= abs(hi - np.sum(np.where(c > 0, c * U, c * L))) < 1e-8 and abs(lo - np.sum(np.where(c > 0, c * L, c * U))) < 1e-8
check("CQ1 IDENTITY star cover: the field is the product of the intervals", ok, "7 times x 40 directions")

# CQ2
seqs = {}; ok = True
for i, j in itertools.combinations(range(k), 2):
    s = []
    for t in CF.T:
        a, e = CF.data_at(t); L = (a - e) / p; U = (a + e) / p; d = CF.order_decided(k, q, p, a, e, i, j)
        ok &= d == (1 if L[i] > U[j] + 1e-12 else (-1 if U[i] < L[j] - 1e-12 else 0)); s.append(d)
    seqs[(i, j)] = s
cert = {pr: [s[0], s[-1]] for pr, s in seqs.items() if s[0] != 0 and s[-1] != 0 and s[0] != s[-1]}
never = [pr for pr, s in seqs.items() if all(v == 0 for v in s)]
for _ in range(200):                                       # sample laws: a random LP vertex at each time
    ti = int(rng.integers(len(CF.T))); a, e = CF.data_at(CF.T[ti]); c = rng.standard_normal(k)
    O, Aeq, beq, Aub, bub = CF.constraints(k, q, p, a, e); f = CF._obj(O, k, p, c)
    from scipy.optimize import linprog
    r = linprog(-f, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * len(O), method="highs"); P = r.x
    v = [sum(P[s] for s, o in enumerate(O) if o[i_] == 1 and o[-1] == 1) / p[i_] for i_ in range(k)]
    for (i, j), s in seqs.items():
        if s[ti] == 1: ok &= v[i] > v[j]
        if s[ti] == -1: ok &= v[i] < v[j]
ok &= len(cert) == 5 and never == [(1, 3)]
fmt = {f"{CF.NAMES[i]} / {CF.NAMES[j]}": "".join({1: "+", -1: "-", 0: "."}[v] for v in s) for (i, j), s in seqs.items()}
check("CQ2 IDENTITY decided iff disjoint; five certified crossings; light/arousal never decided; sampled laws agree", ok, f"{fmt}")

# CQ3
i, j = 0, 2; s = seqs[(i, j)]; und = [t for t, v in enumerate(s) if v == 0]
B = [(CF.data_at(CF.T[t])) for t in range(len(CF.T))]
Lb = np.array([(a - e) / p for a, e in B]); Ub = np.array([(a + e) / p for a, e in B])
def path(nswaps):
    d = []
    for t in range(len(CF.T)):
        if s[t] != 0: d.append(s[t])
        else:
            pos = und.index(t); seg = len(und) / (nswaps); d.append(-1 if int(pos // seg) % 2 == 0 else 1)
    vi, vj = [], []
    for t, sign in enumerate(d):
        lo, hi = max(Lb[t, i], Lb[t, j]), min(Ub[t, i], Ub[t, j])
        if s[t] != 0: vi.append((Lb[t, i] + Ub[t, i]) / 2); vj.append((Lb[t, j] + Ub[t, j]) / 2)
        else: m = (lo + hi) / 2; h = (hi - lo) / 4; vi.append(m + sign * h); vj.append(m - sign * h)
    vi, vj = np.array(vi), np.array(vj); inside = np.all((vi >= Lb[:, i] - 1e-12) & (vi <= Ub[:, i] + 1e-12) & (vj >= Lb[:, j] - 1e-12) & (vj <= Ub[:, j] + 1e-12))
    sw = int(np.sum(np.sign(vi - vj)[1:] != np.sign(vi - vj)[:-1])); return inside, sw
r1, r3 = path(1), path(3)
check("CQ3 IDENTITY a certified crossing fixes only the parity: laws with one and with three swaps", r1 == (True, 1) and r3 == (True, 3), f"receptor / sleep pressure: {r1}, {r3}")

# CQ4
t = 40.0; a, e = CF.data_at(t); ok = CF.order_decided(k, q, p, a, e, 1, 3) == 0; rows = []
for d_, eps in ((0.02, 0.005), (0.01, 0.005), (0.03, 0.0)):
    rel = [(1, 3, 1, 0, 1, 0.0, eps), (1, 3, 0, 1, 1, d_, 1.0)]
    lo, hi = CF.value_range(k, q, p, a, e, np.array([0, -1, 0, 1.0]), rel); ok &= abs(lo - (d_ - eps) / 0.5) < 1e-8 and CF.order_decided(k, q, p, a, e, 1, 3, rel=rel) == -1
    rows.append((d_, eps, round(lo, 4)))
check("CQ4 IDENTITY relational data decide an order the intervals leave open: v_A - v_L >= (delta - eps)/p", ok, f"t = 40 min; (delta, eps, min v_A - v_L): {rows}")

# CQ5
ok = True
for _ in range(100):
    a = rng.uniform(0, 0.5, k); e = rng.uniform(0, 0.02, k); ok &= CF.value_range(k, q, p, a, e, np.ones(k)) is not None
a0, e0 = CF.data_at(120.0); L = (a0 - e0) / p; U = (a0 + e0) / p; i, j = 0, 2; g = L[i] - U[j]; gflip = U[i] - L[j]
def state(total):
    sh = np.zeros(k); sh[i] = -total / 2; sh[j] = total / 2; a, e = CF.data_at(120.0, sh); return CF.order_decided(k, q, p, a, e, i, j)
def bisect(pred):
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if pred(state(mid)): hi = mid
        else: lo = mid
    return hi
hide = bisect(lambda d: d != 1); flip = bisect(lambda d: d == -1)                     # first loss of the order; first reversal
ok &= abs(hide - g) < 1e-6 and abs(flip - gflip) < 1e-6 and all(state(x) is not None for x in np.linspace(0, flip, 9))
check("CQ5 IDENTITY star cover never contradicts; hiding a certified order costs g = L_i - U_j, flipping it U_i - L_j", ok,
      f"receptor over sleep pressure at 120 min: g = {g:.4f} (each source {g / 2:.4f}), flip {gflip:.4f}")

# CQ6
eta = 0.01; true_c = CF.centres(120.0)[i]; w = CF.halfwidths(120.0)[i]
def anchored(shift_i, shift_j):
    sh = np.zeros(k); sh[i] = -shift_i; sh[j] = shift_j; a, e = CF.data_at(120.0, sh)
    lo = max((a[i] - e[i]) / p[i], true_c - eta); hi = min((a[i] + e[i]) / p[i], true_c + eta)
    if lo > hi: return None
    a2, e2 = a.copy(), e.copy(); a2[i] = (lo + hi) / 2 * p[i]; e2[i] = (hi - lo) / 2 * p[i]
    return CF.order_decided(k, q, p, a2, e2, i, j)
ok = anchored(g / 2 + 1e-4, g / 2 + 1e-4) == 1
ok &= anchored(w + eta - 1e-4, 0) is not None and anchored(w + eta + 1e-4, 0) is None
check("CQ6 IDENTITY an anchor on pathway i: contradiction iff i's shift exceeds w_i + eta; at the hiding shift it re-certifies", ok,
      f"eta = {eta}, w_i + eta = {w + eta:.3f}")
json.dump(dict(orders=fmt, hide=float(g), flip=float(gflip)), open(os.path.join(HERE, "credalfield_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
