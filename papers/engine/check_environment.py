"""Checks for Section "An observed environment" of "Probability, Context and Differentiation: The Layers" (module pce/environment.py).
Exact rational arithmetic throughout.
  EN1  structure: forgetting the source and the environment in either order gives the same law (the forgetting square commutes), and
       every forgetting commutes with conditioning on an event A of X (300 random laws on L x E x X)
  EN2  contextual Bayes with an environment: conditioning on L x E x A gives the weights rho^A_e pi^{A,e}_l with
       pi^{A,e}_l = pi_{l|e} p_{l,e}(A) / P_e(A) and rho^A_e = rho_e P_e(A) / P(A); the pooled provenance is the rho^A-mixture of the
       per-environment provenances; observing e leaves its weight at 1 (no reweighting); with one environment this is Theorem (contextual Bayes)
  EN3  reversal of a source comparison: in 2000 random laws with two sources and two environments, reversals (source 1 riskier in every
       environment, source 2 riskier once the environment is forgotten) occur, and never when the source mix pi_{.|e} is the same in every
       environment; at a reversal wall the pooled rates tie
  EN4  rare events: with p_{l,e}(A) = c s^(m_le) and rho_e = s^(r_e) (normalised), the pooled order of source l is min_e (r_e + m_le), checked by
       exact evaluation at s = 10^-12 (300 families); a source that survives in every environment survives in the pool
  EN5  a source can survive in its environment and be extinct in the pool exactly when that environment's leading order exceeds the global
       one (300 families, exact), and the example of the text
  EN6  the reversal as a signed polynomial: for 2000 random laws with two sources and two or three environments, the sign of
       R = (sum rho pi_1 p_1)(sum rho pi_2) - (sum rho pi_2 p_2)(sum rho pi_1) equals the sign of the pooled comparison p_1(A) - p_2(A)
"""
import os, sys, random, itertools
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import environment as EN
random.seed(20261102); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rdist(keys):
    w = [Fr(random.randint(1, 9)) for _ in keys]; t = sum(w); return {k: v / t for k, v in zip(keys, w)}
def rlaw(nL, nE, X):
    rho = rdist(range(nE)); pi = {e: rdist(range(nL)) for e in range(nE)}; p = {(l, e): rdist(X) for l in range(nL) for e in range(nE)}
    return rho, pi, p, EN.joint(rho, pi, p)

# EN1
ok = True; X = list(range(4))
for _ in range(300):
    rho, pi, p, Lam = rlaw(random.randint(1, 3), random.randint(1, 3), X); A = set(random.sample(X, random.randint(1, 3)))
    a = EN.forget(EN.forget(Lam, "e"), ""); b = EN.forget(EN.forget(Lam, "l"), ""); ok &= a == b == EN.forget(Lam, "")
    for keep in ("le", "l", "e", ""):
        ok &= EN.forget(EN.condition(Lam, A), keep) == EN.condition(EN.forget(Lam, keep), A)
check("EN1 the forgetting square commutes, and every forgetting commutes with conditioning on A (300 laws)", ok)

# EN2
ok = True
for _ in range(300):
    nL, nE = random.randint(1, 3), random.randint(1, 3); rho, pi, p, Lam = rlaw(nL, nE, X); A = set(random.sample(X, random.randint(1, 3)))
    C = EN.forget(EN.condition(Lam, A), "le"); w = {}
    for (l, e, x), v in C.items(): w[(l, e)] = w.get((l, e), 0) + v
    PeA = {e: sum(pi[e][l] * sum(p[(l, e)][x] for x in A) for l in range(nL)) for e in range(nE)}; PA = sum(rho[e] * PeA[e] for e in range(nE))
    pe = {e: {l: pi[e][l] * sum(p[(l, e)][x] for x in A) / PeA[e] for l in range(nL)} for e in range(nE)}
    re = {e: rho[e] * PeA[e] / PA for e in range(nE)}
    ok &= all(w[(l, e)] == re[e] * pe[e][l] for l in range(nL) for e in range(nE))
    ok &= EN.per_env_weights(Lam, A) == pe and EN.env_posterior(Lam, A) == re
    pooled = {}
    for (l, e), v in w.items(): pooled[l] = pooled.get(l, 0) + v
    ok &= all(pooled[l] == sum(re[e] * pe[e][l] for e in range(nE)) for l in range(nL))
    for e0 in range(nE):                                                       # observing e0: condition on {e0} as well
        obs = {k: v for k, v in Lam.items() if k[1] == e0}; Co = EN.condition(obs, A)
        ok &= EN.env_posterior(Co, A) == {e0: 1}
    if nE == 1: ok &= all(pe[0][l] == pi[0][l] * sum(p[(l, 0)][x] for x in A) / PeA[0] for l in range(nL))
check("EN2 contextual Bayes with an environment: rho^A_e pi^{A,e}_l; pooled provenance = rho^A-mixture; an observed environment is not reweighted", ok)

# EN3
rev = 0; rev_same = 0; ok = True; walls = 0
for _ in range(2000):
    same = random.random() < 0.3
    rho = rdist([0, 1]); base = rdist([0, 1]); pi = {e: (base if same else rdist([0, 1])) for e in (0, 1)}
    p = {(l, e): (lambda a: {1: a, 0: 1 - a})(Fr(random.randint(1, 99), 100)) for l in (0, 1) for e in (0, 1)}
    Lam = EN.joint(rho, pi, p); per, pooled = EN.rates(Lam, {1})
    r = all(per[(0, e)] > per[(1, e)] for e in (0, 1)) and pooled[0] < pooled[1]
    rev += r and not same; rev_same += r and same
    if r:                                                                       # slide the source mix towards 'same' until the pooled rates tie
        lo, hi = Fr(0), Fr(1)
        for _ in range(60):
            mid = (lo + hi) / 2; pm = {e: {l: (1 - mid) * pi[e][l] + mid * pi[0][l] for l in (0, 1)} for e in (0, 1)}
            _, pl = EN.rates(EN.joint(rho, pm, p), {1})
            if pl[0] < pl[1]: lo = mid
            else: hi = mid
        pm = {e: {l: (1 - hi) * pi[e][l] + hi * pi[0][l] for l in (0, 1)} for e in (0, 1)}; _, pl = EN.rates(EN.joint(rho, pm, p), {1})
        ok &= abs(float(pl[0] - pl[1])) < 1e-12; walls += 1
check("EN3 reversal of a source comparison under forgetting the environment: it occurs, never with an environment-independent source mix; "
      "at the wall the pooled rates tie", ok and rev > 0 and rev_same == 0, f"{rev} reversals; {rev_same} with the same mix; {walls} walls located")

# EN4, EN5
ok4 = True; ok5 = True; n = 0
for _ in range(300):
    nL, nE = random.randint(2, 3), random.randint(2, 3)
    m = {(l, e): random.randint(0, 4) for l in range(nL) for e in range(nE)}; r = {e: random.randint(0, 3) for e in range(nE)}
    c = {(l, e): Fr(random.randint(1, 5)) for l in range(nL) for e in range(nE)}; s = Fr(1, 10 ** 12)
    rho = {e: s ** r[e] for e in range(nE)}; t = sum(rho.values()); rho = {e: v / t for e, v in rho.items()}
    pi = {e: {l: Fr(1, nL) for l in range(nL)} for e in range(nE)}
    p = {(l, e): {1: c[(l, e)] * s ** m[(l, e)] / 6, 0: 1 - c[(l, e)] * s ** m[(l, e)] / 6} for l in range(nL) for e in range(nE)}
    Lam = EN.joint(rho, pi, p); v = EN.pooled_order(m, r); vmin = min(v.values())
    C = EN.forget(EN.condition(Lam, {1}), "l"); w = {l: sum(val for (l2, e, x), val in C.items() if l2 == l) for l in range(nL)}
    for l in range(nL):                                                         # exact: weight ~ s^(v_l - vmin)
        if v[l] == vmin: ok4 &= w[l] > Fr(1, 10 ** 3) * 0 + Fr(1, 10 ** 6)
        else: ok4 &= w[l] < Fr(1, 10 ** 9)
    surv_env = {e: {l for l in range(nL) if m[(l, e)] == min(m[(j, e)] for j in range(nL))} for e in range(nE)}
    pool = {l for l in range(nL) if v[l] == vmin}
    ok4 &= all(l in pool for l in range(nL) if all(l in surv_env[e] for e in range(nE)))
    for e in range(nE):
        lead_e = r[e] + min(m[(j, e)] for j in range(nL))
        for l in surv_env[e]:
            ok5 &= (l not in pool) <= (lead_e > vmin)                           # extinct in the pool only if the environment is subleading
            if lead_e > vmin and all(r[e2] + m[(l, e2)] > vmin for e2 in range(nE)): ok5 &= l not in pool
    n += 1
mex = {(0, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 1}; rex = {0: 1, 1: 0}; vex = EN.pooled_order(mex, rex)
ok5 &= vex == {0: 2, 1: 1}
check("EN4 rare events: the pooled order of a source is min_e (r_e + m_le) (exact at s = 10^-12); survivors of every environment survive the pool",
      ok4, f"{n} families")
check("EN5 a source surviving in its environment is extinct in the pool only when that environment's leading order exceeds the global one; "
      "the example: orders (1, 2 | 3, 1) with environment orders (1, 0) give pooled orders (2, 1)", ok5, f"pooled orders of the example {vex}")

# EN6
ok = True; neg = 0
for _ in range(2000):
    nE = random.choice([2, 3]); rho = rdist(range(nE)); pi = {e: rdist([0, 1]) for e in range(nE)}
    p = {(l, e): (lambda a: {1: a, 0: 1 - a})(Fr(random.randint(1, 99), 100)) for l in (0, 1) for e in range(nE)}
    _, pooled = EN.rates(EN.joint(rho, pi, p), {1})
    R = (sum(rho[e] * pi[e][0] * p[(0, e)][1] for e in range(nE)) * sum(rho[e] * pi[e][1] for e in range(nE))
         - sum(rho[e] * pi[e][1] * p[(1, e)][1] for e in range(nE)) * sum(rho[e] * pi[e][0] for e in range(nE)))
    d = pooled[0] - pooled[1]; ok &= (R > 0) == (d > 0) and (R == 0) == (d == 0); neg += R < 0
check("EN6 the reversal as a signed polynomial: sign R = sign (p_1(A) - p_2(A)) after forgetting the environment (2000 laws, exact)", ok,
      f"{neg} laws with source 2 riskier in the pool")
print(f"{sum(results)}/{len(results)} passed")
