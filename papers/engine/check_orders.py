"""Checks for Step 3 of "Probability, Context and Differentiation: The Layers": orders of probability transitions.
  OR1  tropicalisation (exact): on non-negative germs (generalised power sums), ord(f + g) = min(ord f, ord g) and ord(fg) = ord f + ord g,
       with leading coefficients adding on ties and multiplying; without positivity the first identity fails (cancellation)
  OR2  the order calculus of the layers (exact): labelling (orders add), pooling and marginalisation (orders take the minimum, leading
       coefficients add over the cells of least order), conditioning (orders subtract), Bayes weights (order m_l - m), on random paths
  OR3  the hierarchy (exact, and floating point for fractional exponents): the limit of the conditioned state is the conditional of the
       leading coefficients of order m; common order: the limiting weights are c_l / sum c_j; unequal order: pi^A_l(s) s^{-(m_l - m)} -> c_l / c;
       the error is O(s^delta), delta the gap to the next exponent
  OR4  one side against the other: two-sided analytic paths have even orders on L x A and equal one-sided limits; one-sided data are
       independent and jump when their conditionals differ; a C-infinity but flat path (exp(-1/t^2) on each side) jumps although every
       derivative exists
  OR5  flat approaches: germs from a Hardy field (exp, log, powers) have comparable orders beyond all powers and the conditional limit
       exists (log-space computation); a non-Hardy germ exp(-1/s)(1 + sin(1/s)) against exp(-1/s) gives no limit
  OR6  the tangent tower: for analytic paths, the k-jet determines the conditional limit iff k >= m; below m the k-jet has no mass on L x A;
       the k-jet as the Weil point Lambda(e_1 + ... + e_k) has coefficient |S|! c_|S| on e^S
  OR7  near the wall at order m: with Lambda(1,a) = eta + (t^-)^m and Lambda(2,b) = eta + (t^+)^m, pi^A_1 runs from 3/4 to 1/4 over
       [-(2 eta)^{1/m}, (2 eta)^{1/m}] exactly; higher order means a wider, more gradual switch for the same gap
"""
import os, sys, json, random, math
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import orders as OD
random.seed(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rgerm(nonneg=True, exps=(0, Fr(1, 2), 1, Fr(3, 2), 2, 3), allow_zero_lead=True):
    g = {}
    for a in random.sample(exps, random.randint(1, 3)): g[Fr(a)] = Fr(random.randint(-4, 4))
    g = OD.clean(g)
    if not g: g = {Fr(random.choice(exps)): Fr(1)}
    if nonneg:
        a = min(g); g[a] = abs(g[a]) if g[a] != 0 else Fr(1)
    return g

# OR1
ok = True
for _ in range(500):
    f, g = rgerm(), rgerm(); (a, c), (b, d) = OD.lead(f), OD.lead(g); (e, h) = OD.lead(OD.add(f, g)); (p, q) = OD.lead(OD.mul(f, g))
    ok &= e == min(a, b) and h == (c if a < b else d if b < a else c + d) and h > 0
    ok &= p == a + b and q == c * d
f, g = {Fr(1): Fr(1), Fr(2): Fr(1)}, {Fr(1): Fr(-1)}
ok &= OD.lead(OD.add(f, g))[0] == 2 > min(OD.lead(f)[0], OD.lead(g)[0])
check("OR1 order is a semiring homomorphism to (min, +) on non-negative germs; cancellation breaks it without positivity", ok)

# OR2
ok = True; L, X = [0, 1, 2], ["a", "b", "c", "d"]; A = lambda x: x in ("a", "b")
for _ in range(200):
    w = {l: rgerm() for l in L}; p = {(l, x): rgerm() for l in L for x in X}
    Lam = {(l, x): OD.mul(w[l], p[(l, x)]) for l in L for x in X}
    for k in Lam: ok &= OD.lead(Lam[k])[0] == OD.lead(w[k[0]])[0] + OD.lead(p[k])[0]
    for x in X:                                              # pooling = marginalising over L
        P = OD.total(Lam, lambda k: k[1] == x); m = min(OD.lead(Lam[(l, x)])[0] for l in L)
        ok &= OD.lead(P) == (m, sum(OD.lead(Lam[(l, x)])[1] for l in L if OD.lead(Lam[(l, x)])[0] == m))
    r = OD.cond_limit(Lam, A); PA = OD.total(Lam, lambda k: A(k[1]))
    ok &= r["m"] == OD.lead(PA)[0]
    for l in L:
        ml = OD.lead(OD.total(Lam, lambda k: k[0] == l and A(k[1])))[0]; ok &= r["extinction"][l] == ml - r["m"]
check("OR2 order calculus: labelling adds orders; pooling and marginals take minima; conditioning subtracts; Bayes weights have order m_l - m", ok)

# OR3
ok = True; errs = []
for _ in range(200):
    Lam = {(l, x): (rgerm(exps=(1, 2, 3)) if A(x) else {Fr(0): Fr(1, 4)}) for l in [0, 1] for x in X}
    r = OD.cond_limit(Lam, A)
    lw = {}
    for (l, x), v in r["limit"].items(): lw[l] = lw.get(l, Fr(0)) + v
    ok &= set(lw) == r["survivors"]
    for s in (Fr(1, 10 ** 5), Fr(1, 10 ** 7)):
        vals = {k: OD.ev(g, s) for k, g in Lam.items() if A(k[1])}; tot = sum(vals.values())
        err = sum(abs(vals[k] / tot - r["limit"].get(k, 0)) for k in vals)
        ok &= err <= 50 * s ** int(r["delta"]) if r["delta"] != OD.INF else err == 0
        for l, e in r["extinction"].items():
            if e > 0:
                ratio = sum(v for k, v in vals.items() if k[0] == l) / tot / s ** int(e)
                cl = OD.lead(OD.total(Lam, lambda k, l=l: k[0] == l and A(k[1])))[1]; ok &= abs(ratio - cl / OD.lead(OD.total(Lam, lambda k: A(k[1])))[1]) < 1000 * s
# common order with explicit weights; fractional exponents (floating point)
Lam = {(0, "a"): {Fr(1): Fr(2)}, (1, "b"): {Fr(1): Fr(3)}, (2, "a"): {Fr(2): Fr(5)}, (0, "c"): {Fr(0): Fr(1)}}
r = OD.cond_limit(Lam, A); ok &= r["survivors"] == {0, 1} and r["limit"] == {(0, "a"): Fr(2, 5), (1, "b"): Fr(3, 5)} and r["extinction"][2] == 1
Lf = {(0, "a"): {Fr(1, 2): Fr(1)}, (1, "b"): {Fr(1): Fr(1)}, (0, "c"): {Fr(0): Fr(1)}}
r = OD.cond_limit(Lf, A); ok &= r["delta"] == Fr(1, 2)
for s in (1e-4, 1e-6, 1e-8):
    v0, v1 = s ** 0.5, s; errs.append(((v1 / (v0 + v1)) / s ** 0.5))
ok &= abs(errs[-1] - 1) < 1e-3
check("OR3 hierarchy: limit = conditional of the order-m coefficients; common order keeps weights c_l/sum; higher orders go extinct at rate s^(m_l-m); error O(s^delta)", ok,
      "fractional example: source 2 vanishes like s^(1/2)")

# OR4
ok = True
for _ in range(200):                                          # two-sided analytic: cells on A are t^(2j) * positive
    Lam = {}
    for l in (0, 1):
        for x in ("a", "b"):
            j = random.randint(1, 2); Lam[(l, x)] = {Fr(2 * j): Fr(random.randint(1, 5)), Fr(2 * j + 1): Fr(random.randint(-3, 3))}
    for t in (Fr(1, 10 ** 6),):
        vp = {k: OD.ev(g, t) for k, g in Lam.items()}; vm = {k: OD.ev(g, -t) for k, g in Lam.items()}
        sp, sm = sum(vp.values()), sum(vm.values()); ok &= all(abs(vp[k] / sp - vm[k] / sm) < Fr(1, 10 ** 4) for k in vp)
flat = lambda t: {(0, "a"): math.exp(-1 / t ** 2) if t < 0 else 0.0, (1, "b"): math.exp(-1 / t ** 2) if t > 0 else 0.0}
lw = flat(-0.2); rw = flat(0.2); ok &= lw[(0, "a")] > 0 and lw[(1, "b")] == 0 and rw[(1, "b")] > 0 and rw[(0, "a")] == 0
check("OR4 two-sided analytic passages agree; one-sided data are independent; a C-infinity flat path (exp(-1/t^2)) jumps from source 1 to source 2", ok)

# OR5 (log-space)
def cond_logs(logs):
    mx = max(logs.values()); w = {k: math.exp(v - mx) for k, v in logs.items()}; s = sum(w.values()); return {k: v / s for k, v in w.items()}
ok = True; hardy = []; nonh = []
for s in np.geomspace(1e-2, 1e-6, 60):
    logs = {"s^5 e^-1/s": 5 * math.log(s) - 1 / s, "e^-1/s": -1 / s, "e^-2/s": -2 / s, "e^-1/s / log(1/s)": -1 / s - math.log(math.log(1 / s))}
    hardy.append(cond_logs(logs)["e^-1/s / log(1/s)"])
    lg = {"osc": -1 / s + math.log(1 + math.sin(1 / s) + 1e-300), "ref": -1 / s}; nonh.append(cond_logs(lg)["osc"])
ok &= hardy[-1] < hardy[0] and abs(cond_logs({"a": -1 / 1e-6, "b": -1 / 1e-6 - math.log(math.log(1e6))})["a"] - 1 / (1 + 1 / math.log(1e6))) < 1e-8
ok &= min(nonh[30:]) < 0.05 and max(nonh[30:]) > 0.6
check("OR5 flat approaches: Hardy-field germs are comparable beyond all powers and the limit exists; exp(-1/s)(1+sin(1/s)) gives no limit", ok,
      f"Hardy share -> {hardy[-1]:.3f} (slowly, like 1/log(1/s)); oscillating share ranges over [{min(nonh[30:]):.2f}, {max(nonh[30:]):.2f}]")

# OR6
ok = True; ms = []
for _ in range(200):
    Lam = {(l, x): (rgerm(exps=(1, 2, 3, 4)) if A(x) else {Fr(0): Fr(1, 4), Fr(1): Fr(random.randint(-2, 2))}) for l in (0, 1) for x in X}
    r = OD.cond_limit(Lam, A); m = int(r["m"]); ms.append(m)
    for k in range(0, 6):
        rk = OD.cond_limit(OD.jet_truncate(Lam, k), A)
        ok &= (rk is not None and rk["limit"] == r["limit"]) if k >= m else rk is None
from pce.tangent import HD
from math import factorial
for _ in range(50):                                          # Weil picture: Lambda(e_1 + ... + e_k) has coefficient |S|! c_|S| on e^S
    k = random.randint(1, 4); cs = [random.randint(-3, 3) for _ in range(k + 2)]
    e = np.zeros(2 ** k); 
    for i in range(k): e[1 << i] = 1.0
    S = HD(e.reshape(-1, 1), k); val = HD(np.zeros((2 ** k, 1)), k); pw = HD(np.r_[1.0, np.zeros(2 ** k - 1)].reshape(-1, 1), k)
    for j, c in enumerate(cs): val = val + pw * float(c); pw = pw * S
    ok &= all(abs(val.c[T, 0] - factorial(bin(T).count("1")) * cs[bin(T).count("1")]) < 1e-9 for T in range(2 ** k))
check("OR6 the k-jet sees the transition iff k >= m: below m it has no mass on L x A, from m on it gives the exact limit", ok, f"orders seen: {sorted(set(ms))}")

# OR7
ok = True; rows = []
for m in (1, 2, 3):
    for r_ in (Fr(1, 10), Fr(1, 5)):
        eta = r_ ** m / 2
        w1 = lambda t: (eta + max(-t, 0) ** m) / (2 * eta + abs(t) ** m)
        ok &= w1(-r_) == Fr(3, 4) and w1(r_) == Fr(1, 4) and w1(Fr(0)) == Fr(1, 2)
        rows.append((m, str(eta), str(2 * r_)))
eta = Fr(1, 1000); widths = [2 * (2 * float(eta)) ** (1 / m) for m in (1, 2, 3)]; ok &= widths[0] < widths[1] < widths[2]
check("OR7 at order m the switch 3/4 -> 1/4 has width exactly 2 (2 eta)^(1/m): higher order, wider and more gradual", ok, f"eta = 1/1000: widths {[round(x, 4) for x in widths]}")
OUT["OR7_widths"] = widths
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "orders.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
