"""check_agents.py -- checks for Section 18 (three agents) of "Shadows, Fibres and Secants of the Posterior Tower".

Setting: a binary state X with prior P(X=1) = rho; agent i sees a signal S_i and reports the posterior q_i = P(X=1 | S_i). Agent i's
tower is the law Pi_i of q_i, with barycentre rho_i. Exact rational arithmetic except the information quantities (A2, A6) and the
linear programs (HiGHS), whose answers are compared with exact criteria where one exists.

  AG1 Bayes plausibility: for random signal structures, every barycentre rho_i equals the common prior (exact)
  AG2 chain-rule split for the pooled agents: H(rho_bar) = I(X;agent) + sum w_i I(X;q_i) + sum w_i E H(q_i); I(X;agent) = 0 iff all
      rho_i are equal
  AG3 garbling implies convex order: if S_j is a garbling of S_i then Pi_j <=cx Pi_i (stop-loss criterion, exact); the stop-loss
      criterion and the martingale-coupling LP agree on 300 random pairs
  AG4 the worked example: agents a (accuracy 0.8), b (reveals X with probability 0.2), c (a garbled): c <cx a; b is incomparable with
      a and with c; mutual information nevertheless ranks all three (it totalises a partial order)
  AG5 reconciliation: in the worked logs (accuracy 3/4; P(A1 != A2) = P(A2 != A3) = 1/20, P(A1 != A3) = 1/2) every pair is feasible and
      the chain (12, 23) glues, but the triangle is infeasible; the cycle inequality d13 <= d12 + d23 fails by exactly 2/5
  AG6 pooling: the chain rule P(X,A1,A2) P(X,A2,A3) / P(X,A2) reproduces P(X | A1,A2,A3) exactly when A1 _||_ A3 | (X,A2) (200 random
      laws, error < 1e-12, c = 0); for generic laws it fails (200 random laws, error > 0 and the defect c > 0 in each)
  AG7 pooling identity: for random joint laws of (X, S_i, S_j) and random pooling rules F (linear, logarithmic, arbitrary),
      loss(q_i) - loss(F) = I(X; q_j | q_i) - E KL(q* || F) for log loss, and the analogue with squared distance for the Brier score
      (error < 1e-12; 300 laws x 4 rules each)
  AG8 the Blackwell (garbling) case: I(X; q_j | q_i) = 0, and every rule that moves off q_i is strictly worse, by E KL(q_i || F);
      the worked example (accuracies 0.8 and 0.65) gives the table of Example 18.7
Run: python check_agents.py
"""
from fractions import Fraction as F
import itertools, math
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(20261013)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

# ----------------------------------------------------------------- towers from signal structures (exact)
def tower(rho, kernel):
    """kernel[s] = (P(S=s|X=0), P(S=s|X=1)); returns {posterior q: mass} (exact Fractions)."""
    law = {}
    for k0, k1 in kernel:
        m = (1 - rho) * k0 + rho * k1
        if m == 0: continue
        q = rho * k1 / m; law[q] = law.get(q, 0) + m
    return law
def bary(law): return sum(q * m for q, m in law.items())
def rand_kernel(ns):
    cols = []
    for _ in range(2):
        v = [F(int(x)) for x in rng.integers(1, 10, ns)]; t = sum(v); cols.append([x / t for x in v])
    return list(zip(cols[0], cols[1]))
def garble(kernel, G):
    """S' = garbling of S by a row-stochastic matrix G[s][s']."""
    return [tuple(sum(kernel[s][x] * G[s][t] for s in range(len(kernel))) for x in (0, 1)) for t in range(len(G[0]))]

# AG1
ok = True
for _ in range(200):
    rho = F(int(rng.integers(1, 10)), 10); law = tower(rho, rand_kernel(int(rng.integers(2, 6))))
    ok &= bary(law) == rho and sum(law.values()) == 1
check("AG1 Bayes plausibility: the barycentre of every agent's tower is the common prior (200 random signal structures)", ok)

# AG2
def H(p): return -sum(x * math.log2(x) for x in (p, 1 - p) if x > 0)
def EH(law): return sum(float(m) * H(float(q)) for q, m in law.items())
ok = True; zero_iff = True
for _ in range(200):
    w = rng.dirichlet(np.ones(3)); common = rng.random() < 0.5
    rhos = [F(1, 2)] * 3 if common else [F(int(rng.integers(1, 10)), 10) for _ in range(3)]
    laws = [tower(r, rand_kernel(int(rng.integers(2, 5)))) for r in rhos]
    rbar = sum(wi * float(r) for wi, r in zip(w, rhos))
    I_agent = H(rbar) - sum(wi * H(float(r)) for wi, r in zip(w, rhos))
    I_sig = sum(wi * (H(float(r)) - EH(L)) for wi, r, L in zip(w, rhos, laws))
    noise = sum(wi * EH(L) for wi, L in zip(w, laws))
    ok &= abs(H(rbar) - (I_agent + I_sig + noise)) < 1e-12 and I_agent >= -1e-15
    zero_iff &= (abs(I_agent) < 1e-12) == (len(set(rhos)) == 1)
check("AG2 H(rho_bar) = I(X;agent) + sum w I(X;q_i) + sum w E H(q_i); I(X;agent) = 0 iff a common prior", ok and zero_iff)

# AG3
def stoploss_leq(lo, hi):
    """Pi_lo <=cx Pi_hi on [0,1]: equal means and E(q-t)+ smaller at every support point (exact; piecewise-linear criterion)."""
    if bary(lo) != bary(hi): return False
    pts = set(lo) | set(hi)
    sl = lambda law, t: sum(m * max(q - t, 0) for q, m in law.items())
    return all(sl(lo, t) <= sl(hi, t) for t in pts)
def martingale_lp(lo, hi):
    """exists a coupling pi(u, v), u ~ Pi_lo, v ~ Pi_hi, with E[v | u] = u ?"""
    U, V = sorted(lo), sorted(hi); n, m = len(U), len(V); A, b = [], []
    for i in range(n):
        row = np.zeros(n * m); row[i * m:(i + 1) * m] = 1; A.append(row); b.append(float(lo[U[i]]))
        row = np.zeros(n * m); row[i * m:(i + 1) * m] = [float(v - U[i]) for v in V]; A.append(row); b.append(0.0)
    for j in range(m):
        row = np.zeros(n * m); row[j::m] = 1; A.append(row); b.append(float(hi[V[j]]))
    r = linprog(np.zeros(n * m), A_eq=np.array(A), b_eq=np.array(b), bounds=(0, None), method="highs")
    return r.status == 0
ok_g = True; agree = 0
for _ in range(150):
    rho = F(int(rng.integers(2, 9)), 10); K = rand_kernel(3)
    G = []
    for _ in range(3):
        v = [F(int(x)) for x in rng.integers(0, 6, 3)]
        if sum(v) == 0: v[0] = F(1)
        G.append([x / sum(v) for x in v])
    ok_g &= stoploss_leq(tower(rho, garble(K, G)), tower(rho, K))
npos = 0
for k in range(300):
    rho = F(1, 2); K = rand_kernel(3)
    if k % 2:                                                  # half the pairs are garblings, so both answers occur
        G = [[F(int(x)) + 1 for x in rng.integers(0, 5, 3)] for _ in range(3)]; G = [[x / sum(r) for x in r] for r in G]
        L1, L2 = tower(rho, garble(K, G)), tower(rho, K)
    else:
        L1, L2 = tower(rho, rand_kernel(3)), tower(rho, K)
    s_ = stoploss_leq(L1, L2); npos += s_
    agree += s_ == martingale_lp(L1, L2)
check("AG3 garbling implies convex order (150 garblings); stop-loss and martingale-coupling LP agree on 300 pairs",
      ok_g and agree == 300 and 0 < npos < 300, f"agreement {agree}/300; {npos} comparable, {300 - npos} not")

# AG4
half = F(1, 2)
a = tower(half, [(F(4, 5), F(1, 5)), (F(1, 5), F(4, 5))])                        # accuracy 0.8
b = tower(half, [(F(1, 5), 0), (0, F(1, 5)), (F(4, 5), F(4, 5))])                 # reveals X w.p. 0.2, else nothing
flip = [[F(3, 4), F(1, 4)], [F(1, 4), F(3, 4)]]
c = tower(half, garble([(F(4, 5), F(1, 5)), (F(1, 5), F(4, 5))], flip))           # a's signal flipped w.p. 1/4
order = {(x, y): stoploss_leq(L1, L2) for (x, L1), (y, L2) in itertools.permutations({"a": a, "b": b, "c": c}.items(), 2)}
MI = {k: H(0.5) - EH(L) for k, L in {"a": a, "b": b, "c": c}.items()}
ok = order[("c", "a")] and not order[("a", "c")] and not any(order[p] for p in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")])
check("AG4 example: c <cx a, b incomparable with a and c; mutual information still ranks them a > b > c", ok and MI["a"] > MI["b"] > MI["c"],
      f"towers a={ {str(k): str(v) for k, v in a.items()} }, b={ {str(k): str(v) for k, v in b.items()} }, "
      f"c={ {str(k): str(v) for k, v in c.items()} }; I(X;q) bits a={MI['a']:.3f}, b={MI['b']:.3f}, c={MI['c']:.3f}")

# AG5
ATOMS = list(itertools.product((0, 1), repeat=4))          # (x, a1, a2, a3)
def feasible(pair_tabs, acc=F(3, 4), agents=(1, 2, 3)):
    """LP over P(x, a1, a2, a3) >= 0: prior 1/2, each agent's report calibrated with accuracy acc, observed pair tables matched."""
    A, b = [], []
    def add(f, val): A.append([float(f(t)) for t in ATOMS]); b.append(float(val))
    add(lambda t: 1, 1)
    for i in agents:                                           # P(X=x, A_i=a) = 1/2 * P(A_i=a | X=x)
        for x, aa in itertools.product((0, 1), repeat=2):
            add(lambda t, i=i, x=x, aa=aa: t[0] == x and t[i] == aa, F(1, 2) * (acc if aa == x else 1 - acc))
    for (i, j), tab in pair_tabs.items():
        for u, v in itertools.product((0, 1), repeat=2):
            add(lambda t, i=i, j=j, u=u, v=v: t[i] == u and t[j] == v, tab[u][v])
    r = linprog(np.zeros(len(ATOMS)), A_eq=np.array(A), b_eq=np.array(b), bounds=(0, None), method="highs")
    return r.status == 0
def tab(d): return [[(1 - d) / 2, d / 2], [d / 2, (1 - d) / 2]]
d12, d23, d13 = F(1, 20), F(1, 20), F(1, 2)
T = {(1, 2): tab(d12), (2, 3): tab(d23), (1, 3): tab(d13)}
pairs_ok = all(feasible({p: T[p]}) for p in T)
chain_ok = feasible({(1, 2): T[(1, 2)], (2, 3): T[(2, 3)]})
tri_bad = not feasible(T)
viol = d13 - (d12 + d23)
check("AG5 each pair feasible and the chain glues, the triangle is infeasible; cycle inequality d13 <= d12 + d23 fails by 2/5",
      pairs_ok and chain_ok and tri_bad and viol == F(2, 5), f"violation {viol}")

# AG6
def pool_error(P):
    """P[x, a1, a2, a3] > 0; returns max |P(X=1|a) - chain-pooled| and the defect c of the square sigma(X,A1,A2), sigma(X,A2,A3)."""
    P12 = P.sum(3); P23 = P.sum(1); P2 = P.sum((1, 3))
    Q = np.einsum("xab,xbc->xabc", P12, P23) / P2[:, None, :, None]
    true = P[1] / P.sum(0); pooled = Q[1] / Q.sum(0)
    c = 0.0
    for x in (0, 1):
        for a2 in (0, 1):
            t = P[x, :, a2, :]; r0, r1, c0, c1 = t[0].sum(), t[1].sum(), t[:, 0].sum(), t[:, 1].sum()
            c = max(c, abs(t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / math.sqrt(r0 * r1 * c0 * c1))
    return float(np.abs(true - pooled).max()), c
ok_ci = ok_not = True
for _ in range(200):
    px = rng.dirichlet(np.ones(2)); p2 = rng.dirichlet(np.ones(2), size=2)
    p1 = rng.dirichlet(np.ones(2), size=(2, 2)); p3 = rng.dirichlet(np.ones(2), size=(2, 2))
    P = np.einsum("x,xb,xba,xbc->xabc", px, p2, p1, p3)       # A1 _||_ A3 | (X, A2) by construction
    e, c = pool_error(P); ok_ci &= e < 1e-12 and c < 1e-9
    P = rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2)
    e, c = pool_error(P); ok_not &= e > 1e-6 and c > 1e-6
check("AG6 chain pooling is exact when A1 _||_ A3 | (X, A2), and fails for generic laws: error < 1e-12 with c = 0 (200 laws), error > 0 with c > 0 (200 generic laws)",
      ok_ci and ok_not)

# AG7, AG8: pooling a more and a less informed agent
def post_tables(P):
    """P[x, si, sj] > 0 -> q_i[si], q_j[sj], q*[si, sj] (probability of X = 1)."""
    qi = P[1].sum(1) / P.sum((0, 2)); qj = P[1].sum(0) / P.sum((0, 1)); qs = P[1] / P.sum(0)
    return qi, qj, qs
def logloss(P, F):
    return float(sum(P[x, a, b] * -math.log(F[a, b] if x else 1 - F[a, b]) for x in (0, 1) for a in range(P.shape[1]) for b in range(P.shape[2])))
def brier(P, F):
    return float(sum(P[x, a, b] * (x - F[a, b]) ** 2 for x in (0, 1) for a in range(P.shape[1]) for b in range(P.shape[2])))
def kl(p, q): return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
def rules(qi, qj, r):
    QI, QJ = np.meshgrid(qi, qj, indexing="ij")
    lin = 0.5 * QI + 0.5 * QJ
    u, v = np.sqrt(QI * QJ), np.sqrt((1 - QI) * (1 - QJ)); logp = u / (u + v)
    return {"q_i": QI, "linear": lin, "log": logp, "arbitrary": r.uniform(0.05, 0.95, QI.shape)}
ok7 = True
for _ in range(300):
    ni, nj = rng.integers(2, 5, 2)
    P = rng.dirichlet(np.ones(2 * ni * nj)).reshape(2, ni, nj)
    qi, qj, qs = post_tables(P); m = P.sum(0)
    I = sum(m[a, b] * kl(qs[a, b], qi[a]) for a in range(ni) for b in range(nj))       # I(X; q_j | q_i) = E KL(q* || q_i)
    Ib = sum(m[a, b] * (qs[a, b] - qi[a]) ** 2 for a in range(ni) for b in range(nj))
    QI = np.repeat(qi[:, None], nj, 1)
    for name, F in rules(qi, qj, rng).items():
        lhs = logloss(P, QI) - logloss(P, F)
        rhs = I - sum(m[a, b] * kl(qs[a, b], F[a, b]) for a in range(ni) for b in range(nj))
        lhs_b = brier(P, QI) - brier(P, F)
        rhs_b = Ib - sum(m[a, b] * (qs[a, b] - F[a, b]) ** 2 for a in range(ni) for b in range(nj))
        ok7 &= abs(lhs - rhs) < 1e-12 and abs(lhs_b - rhs_b) < 1e-12
check("AG7 loss(q_i) - loss(F) = I(X;q_j|q_i) - E KL(q*||F) (log loss) and the Brier analogue, 300 laws x 4 rules", ok7)

ok8 = True; table = {}
for _ in range(200):                                           # random garblings: S_j depends on X only through S_i
    ni, nj = rng.integers(2, 5, 2)
    px = rng.dirichlet(np.ones(2)); K = rng.dirichlet(np.ones(ni), size=2); G = rng.dirichlet(np.ones(nj), size=ni)
    P = np.einsum("x,xa,ab->xab", px, K, G)
    qi, qj, qs = post_tables(P); m = P.sum(0); QI = np.repeat(qi[:, None], nj, 1)
    ok8 &= np.abs(qs - QI).max() < 1e-12                       # Bayes pool = q_i, so I(X; q_j | q_i) = 0
    for name, F in rules(qi, qj, rng).items():
        gap = logloss(P, F) - logloss(P, QI)
        ekl = sum(m[a, b] * kl(qi[a], F[a, b]) for a in range(ni) for b in range(nj))
        ok8 &= abs(gap - ekl) < 1e-12 and (name == "q_i" or gap > 1e-9)
for label, coupling in (("garbling", "g"), ("independent", "i")):
    P = np.zeros((2, 2, 2))
    for x, a, b in itertools.product((0, 1), repeat=3):
        pa = 0.8 if a == x else 0.2
        pb = (0.75 if b == a else 0.25) if coupling == "g" else (0.65 if b == x else 0.35)
        P[x, a, b] = 0.5 * pa * pb
    qi, qj, qs = post_tables(P); R = rules(qi, qj, rng); QI = R["q_i"]
    bits = lambda F: logloss(P, F) / math.log(2)
    table[label] = dict(alone=bits(QI), bayes=bits(qs), linear=bits(R["linear"]), log=bits(R["log"]))
ok8 &= abs(table["garbling"]["bayes"] - table["garbling"]["alone"]) < 1e-12 and table["independent"]["alone"] - table["independent"]["bayes"] > 0.04
check("AG8 garbling: Bayes pool = q_i and every other rule is worse by exactly E KL(q_i||F) (200 garblings); worked table", ok8,
      "; ".join(f"{k}: " + ", ".join(f"{n} {v:.3f}" for n, v in d.items()) for k, d in table.items()))

print(f"\n{sum(results)}/{len(results)} checks passed")
