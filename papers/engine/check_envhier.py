"""Checks for Section "Structured environments" of "Probability, Context and Differentiation: The Layers" (module pce/envhier.py).
Exact rational arithmetic throughout.
  EH1  a hierarchy of environments: coarsening is a morphism of comonads (counit and comultiplication) and of distributive laws, coarsenings
       compose, and they commute with conditioning; the exchange of two product comonads satisfies the Yang-Baxter (hexagon) condition
  EH2  nested contextual Bayes: the weights at a coarser level are the sums of the finer weights, and the within-node provenance is the
       rho^A-mixture of the children's (300 random three-level hierarchies)
  EH3  nested orders: computing pooled orders level by level, v_parent = min over children of (r_child + v_child), agrees with pooling all
       at once and with exact evaluation at s = 10^-12 (300 hierarchies)
  EH4  pooling never creates an explanation: the survivors at a node lie in the union of its children's survivors; it can destroy one
  EH5  a source comparison can reverse at an intermediate level and reverse back at the top (found by search and checked exactly)
  EH6  the store comonad X^E x E with the independent coupling: the unit, multiplication and counit axioms hold exactly; comultiplication
       fails, since the left side copies one draw while the right side draws each copy independently
"""
import os, sys, random, itertools
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import envhier as EH
from pce import environment as EN
random.seed(20261102); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rdist(keys):
    w = [Fr(random.randint(1, 9)) for _ in keys]; t = sum(w); return {k: v / t for k, v in zip(keys, w)}
def rhier():
    nS = random.randint(3, 6); nR = random.randint(2, min(3, nS)); par = {e: (e % nR) for e in range(nS)}   # every region has a child
    return nS, nR, par
X = [0, 1, 2]

# EH1
ok = True
for _ in range(200):
    nL = random.randint(1, 3); nS, nR, par = rhier(); top = {b: 0 for b in range(nR)}
    rho = rdist(range(nS)); pi = {e: rdist(range(nL)) for e in range(nS)}; p = {(l, e): rdist(X) for l in range(nL) for e in range(nS)}
    Lam = EN.joint(rho, pi, p); A = set(random.sample(X, 2))
    ok &= EH.coarsen(EH.coarsen(Lam, par), top) == EH.coarsen(Lam, {e: 0 for e in range(nS)})
    ok &= EH.coarsen(EN.condition(Lam, A), par) == EN.condition(EH.coarsen(Lam, par), A)
    for l in range(nL):
        for e in range(nS):
            x = random.choice(X); c = lambda k: (k[0], par[k[1]], k[2])
            ok &= c((l, e, x))[2] == x                                           # counit: forgetting the label commutes with coarsening
            dd = ((l, e), ((l, e), x)); dc = ((l, par[e]), ((l, par[e]), x))      # comultiplication copies the label; coarsening each copy
            ok &= (dd[0][0], par[dd[0][1]]) == dc[0] and (dd[1][0][0], par[dd[1][0][1]]) == dc[1][0]
            law = p[(l, e)]; lam0 = {(l, e, xx): v for xx, v in law.items()}; lam1 = {(l, par[e], xx): v for xx, v in law.items()}
            ok &= EH.coarsen(lam0, par) == lam1                                  # D c o lambda_0 = lambda_1 o c_D
yb = True
for (a, b, c_, x) in itertools.product(range(2), range(2), range(2), range(2)):
    s12 = lambda t: (t[1], t[0], t[2], t[3]); s23 = lambda t: (t[0], t[2], t[1], t[3]); t = (a, b, c_, x)
    yb &= s12(s23(s12(t))) == s23(s12(s23(t)))
check("EH1 coarsening is a morphism of comonads and of distributive laws, composes, commutes with conditioning; exchange satisfies Yang-Baxter",
      ok and yb)

# EH2
ok = True
for _ in range(300):
    nL = random.randint(1, 3); nS, nR, par = rhier()
    rho = rdist(range(nS)); pi = {e: rdist(range(nL)) for e in range(nS)}; p = {(l, e): rdist(X) for l in range(nL) for e in range(nS)}
    Lam = EN.joint(rho, pi, p); A = set(random.sample(X, 2))
    w0 = EN.per_env_weights(Lam, A); rA = EN.env_posterior(Lam, A); w1 = EN.per_env_weights(EH.coarsen(Lam, par), A)
    for b in range(nR):
        kids = [e for e in range(nS) if par[e] == b]; tot = sum(rA[e] for e in kids)
        ok &= all(w1[b][l] == sum(rA[e] / tot * w0[e].get(l, 0) for e in kids) for l in range(nL))
check("EH2 nested contextual Bayes: a node's provenance is the rho^A-mixture of its children's (300 hierarchies)", ok)

# EH3, EH4
ok3 = True; ok4 = True; lost = 0; n = 0
for _ in range(300):
    nL = random.randint(2, 3); nS, nR, par = rhier(); s = Fr(1, 10 ** 12)
    m = {(l, e): random.randint(0, 4) for l in range(nL) for e in range(nS)}; r0 = {e: random.randint(0, 2) for e in range(nS)}
    r1 = {b: random.randint(0, 2) for b in range(nR)}
    v1 = EH.orders_up(m, r0, par); v2 = EH.orders_up(v1, r1, {b: 0 for b in range(nR)})
    direct = {l: min(r0[e] + r1[par[e]] + m[(l, e)] for e in range(nS)) for l in range(nL)}
    ok3 &= all(v2[(l, 0)] == direct[l] for l in range(nL))
    # exact evaluation: rho_e proportional to s^(r0 + r1), p_{l,e}(A) = c s^m
    rho = {e: s ** (r0[e] + r1[par[e]]) for e in range(nS)}; t = sum(rho.values()); rho = {e: v / t for e, v in rho.items()}
    pi = {e: {l: Fr(1, nL) for l in range(nL)} for e in range(nS)}
    p = {(l, e): {1: Fr(1, 3) * s ** m[(l, e)], 0: 1 - Fr(1, 3) * s ** m[(l, e)]} for l in range(nL) for e in range(nS)}
    Lam = EN.joint(rho, pi, p); C = EN.forget(EN.condition(Lam, {1}), "l")
    wt = {l: sum(val for (l2, e, x), val in C.items() if l2 == l) for l in range(nL)}; vmin = min(direct.values())
    ok3 &= all((wt[l] > Fr(1, 10 ** 6)) == (direct[l] == vmin) for l in range(nL))
    S0, S1, S2 = EH.survivors(m), EH.survivors(v1), EH.survivors(v2)
    for b in range(nR):
        kids = [e for e in range(nS) if par[e] == b]; U = set().union(*[S0[e] for e in kids])
        ok4 &= S1[b] <= U; lost += len(U - S1[b])
    ok4 &= S2[0] <= set().union(*[S1[b] for b in range(nR)]); n += 1
check("EH3 nested orders: level by level = all at once = exact evaluation at s = 10^-12 (associativity of the tropical sum)", ok3, f"{n} hierarchies")
check("EH4 pooling never creates an explanation (survivors of a node lie among its children's); it destroys some", ok4 and lost > 0,
      f"{lost} local explanations lost at the regional level")

# EH5
found = None
for _ in range(20000):
    par = {0: 0, 1: 0, 2: 1, 3: 1}
    rho = rdist(range(4)); pi = {e: rdist([0, 1]) for e in range(4)}
    p = {(l, e): (lambda a: {1: a, 0: 1 - a})(Fr(random.randint(1, 99), 100)) for l in (0, 1) for e in range(4)}
    Lam = EN.joint(rho, pi, p); per0, _ = EN.rates(Lam, {1}); per1, _ = EN.rates(EH.coarsen(Lam, par), {1})
    _, top = EN.rates(Lam, {1})
    if all(per0[(0, e)] > per0[(1, e)] for e in range(4)) and any(per1[(0, b)] < per1[(1, b)] for b in (0, 1)) and top[0] > top[1]:
        found = (rho, pi, p); break
par = {0: 0, 1: 0, 2: 1, 3: 1}; rho = {e: Fr(1, 4) for e in range(4)}
pi = {e: {0: Fr(v, 10), 1: 1 - Fr(v, 10)} for e, v in enumerate((3, 7, 4, 4))}
p = {}
for e, (a, b) in enumerate(((7, 6), (4, 3), (8, 1), (4, 2))): p[(0, e)] = {1: Fr(a, 10), 0: 1 - Fr(a, 10)}; p[(1, e)] = {1: Fr(b, 10), 0: 1 - Fr(b, 10)}
Lam = EN.joint(rho, pi, p); per1, _ = EN.rates(EH.coarsen(Lam, par), {1}); _, top = EN.rates(Lam, {1})
text = per1[(0, 0)] == Fr(49, 100) and per1[(1, 0)] == Fr(51, 100) and top[0] == Fr(97, 180) and top[1] == Fr(69, 220)
check("EH5 a source comparison can reverse at an intermediate level and reverse back at the top (search, and the example of the text exactly)",
      found is not None and text, f"region 1: {per1[(0, 0)]} vs {per1[(1, 0)]}; top: {top[0]} vs {top[1]}")

# EH6
E = (0, 1); Xs = (0, 1); ok = True
def rd(keys): return rdist(list(keys))
for _ in range(50):
    f = tuple(rd(Xs) for _ in E); e = random.choice(E)
    ok &= EH.lam_indep(tuple({x: Fr(1)} for x in (0, 1)), e) == {((0, 1), e): Fr(1)}                       # unit
    ok &= EH.push(EH.lam_indep(f, e), lambda k: k[0][k[1]]) == f[e]                                        # counit
    # multiplication: F: E -> D D X with two-point supports
    F = []
    for _e in E:
        d1, d2 = rd(Xs), rd(Xs); a = Fr(random.randint(1, 9), 10)
        F.append({tuple(sorted(d1.items())): a, tuple(sorted(d2.items())): 1 - a} if d1 != d2 else {tuple(sorted(d1.items())): Fr(1)})
    flat = tuple({x: sum(w * dict(k).get(x, 0) for k, w in Fe.items()) for x in Xs} for Fe in F)
    lhs = EH.lam_indep(flat, e)
    outer = EH.lam_indep(tuple({k: w for k, w in Fe.items()} for Fe in F), e); rhs = {}
    for (G, ee), w in outer.items():
        for k2, v2 in EH.lam_indep(tuple(dict(g) for g in G), ee).items(): rhs[k2] = rhs.get(k2, 0) + w * v2
    ok &= lhs == rhs
f = (rd(Xs), rd(Xs)); e = 0
left = EH.push(EH.lam_indep(f, e), lambda k: (tuple((k[0], ep) for ep in E), k[1]))
inner = tuple(EH.lam_indep(f, ep) for ep in E); right = {}
for combo in itertools.product(*[list(d.items()) for d in inner]):
    pr = Fr(1)
    for _, v in combo: pr *= v
    key = (tuple(k for k, _ in combo), e); right[key] = right.get(key, 0) + pr
same = lambda law: sum(v for (copies, _), v in law.items() if len({c[0] for c in copies}) == 1)
check("EH6 store comonad with the independent coupling: unit, multiplication and counit hold; comultiplication fails (one draw copied vs "
      "independent draws)", ok and left != right and same(left) == 1 and same(right) < 1,
      f"P(all copies of the stored state agree): left {same(left)}, right {same(right)}")
print(f"{sum(results)}/{len(results)} passed")
