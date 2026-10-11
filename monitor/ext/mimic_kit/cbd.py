"""Contextuality-by-Default for cyclic systems of binary variables, as a coupling problem.

A cyclic system of rank n: contents q_1..q_n, contexts c_i = {q_i, q_{i+1}}; in context c_i the variables V_i (content q_i) and
W_{i+1} (content q_{i+1}) have a joint 2x2 law. Content q_i is measured twice: by W_i in context c_{i-1} and by V_i in context c_i.
CbD: the system is noncontextual iff some joint law of all 2n variables has the observed context tables and makes every connection
(W_i, V_i) maximally coupled, i.e. P(W_i != V_i) = |P(W_i=1) - P(V_i=1)|.
"""
import itertools, math, sys
import numpy as np
from scipy.optimize import linprog
R = np.random.default_rng(21)

def lp_noncontextual(tables):
    """tables[i] = 2x2 law of (V_i, W_{i+1}) in context i (values 0/1).  Returns True iff a CbD-noncontextual coupling exists."""
    n = len(tables); nv = 2 * n                     # variables: V_0..V_{n-1}, W_0..W_{n-1}
    cells = list(itertools.product([0, 1], repeat=nv)); A, b = [], []
    for i, T in enumerate(tables):
        for a_, b_ in itertools.product([0, 1], repeat=2):
            A.append([1.0 if (c[i] == a_ and c[n + (i + 1) % n] == b_) else 0.0 for c in cells]); b.append(T[a_, b_])
    for i in range(n):
        pV = tables[i][1, :].sum(); pW = tables[(i - 1) % n][:, 1].sum()
        A.append([1.0 if c[i] != c[n + i] else 0.0 for c in cells]); b.append(abs(pV - pW))
    A.append([1.0] * len(cells)); b.append(1.0)
    r = linprog(np.zeros(len(cells)), A_eq=np.array(A), b_eq=np.array(b), bounds=[(0, None)] * len(cells), method="highs")
    return r.status == 0

def s_odd(xs):
    return max(sum(m * x for m, x in zip(ms, xs)) for ms in itertools.product([-1, 1], repeat=len(xs)) if np.prod(ms) == -1)

def criterion(tables):
    """Kujala-Dzhafarov: contextual iff s_odd(<V_i W_{i+1}>) > (n - 2) + sum_i |<V_i> - <W_i>| (+-1 valued variables)."""
    n = len(tables)
    corr = [T[0, 0] + T[1, 1] - T[0, 1] - T[1, 0] for T in tables]
    meanV = [tables[i][0, :].sum() - tables[i][1, :].sum() for i in range(n)]
    meanW = [tables[(i - 1) % n][:, 0].sum() - tables[(i - 1) % n][:, 1].sum() for i in range(n)]
    Delta = sum(abs(v - w) for v, w in zip(meanV, meanW))
    return s_odd(corr) > (n - 2) + Delta + 1e-12, s_odd(corr), Delta

def cntx(tables):
    """CbD degree of contextuality for cyclic systems: CNTX = (1/2) max(0, s_odd - (n-2) - Delta)."""
    _, s, D = criterion(tables); return 0.5 * max(0.0, s - (len(tables) - 2) - D)

def random_table():
    return R.dirichlet(np.ones(4)).reshape(2, 2)

if __name__ == "__main__":
    # 1. the LP and the closed-form criterion agree on random cyclic systems (n = 3, 4)
    agree = 0; total = 0; ctx = 0
    for n in (3, 4):
        for _ in range(400 if n == 3 else 150):
            tabs = [random_table() for _ in range(n)]
            c, _, _ = criterion(tabs); agree += (c == (not lp_noncontextual(tabs))); total += 1; ctx += c
    print(f"LP vs criterion agreement: {agree}/{total} (contextual: {ctx})")
    # 2. the three-bit system with disturbance: threshold moves from 1/3 to 1/3 - Delta/6
    def three_bit(eps, d):
        # pairwise disagreement 1 - eps; in context i the variable V_i has P(=1) = 1/2 + d/2 (a direct influence), W_{i+1} stays 1/2
        tabs = []
        for i in range(3):
            pv = 0.5 + d / 2; pw = 0.5
            # 2x2 with margins (1-pv, pv) for V and (1-pw, pw) for W and P(V != W) = 1 - eps
            # cells: [[a, (1-pv)-a], [ (1-pw)-a, ...]] ; P(V=0,W=0) = a, P(V=1,W=1) = 1 - (1-pv) - (1-pw) + a
            # disagreement = (1-pv-a) + (1-pw-a) = 1 - eps  ->  a = (2 - pv - pw - (1 - eps)) / 2
            a = (2 - pv - pw - (1 - eps)) / 2
            T = np.array([[a, 1 - pv - a], [1 - pw - a, pv + pw - 1 + a]]); tabs.append(T)
        return tabs
    for d in (0.0, 0.1, 0.2):
        Delta = 3 * d                                   # each connection differs by d in the mean of the +-1 variable
        es = np.linspace(0.20, 0.45, 2501)
        flags = [not lp_noncontextual(three_bit(e, d)) for e in es[::50]]
        thr = max(e for e in es if criterion(three_bit(e, d))[0]) if any(criterion(three_bit(e, d))[0] for e in es) else None
        print(f"d = {d}: Delta = {Delta:.2f}, contextual for eps < {thr:.4f} (predicted 1/3 - Delta/6 = {1/3 - Delta/6:.4f}); LP agrees: "
              f"{all(f == criterion(three_bit(e, d))[0] for f, e in zip(flags, es[::50]))}")
    # 3. CbD degree equals the contextual fraction 1 - 3 eps on the consistently connected three-bit path
    print("CNTX = 1 - 3 eps on the consistent path:", all(abs(cntx(three_bit(e, 0.0)) - max(0, 1 - 3 * e)) < 1e-12 for e in np.linspace(0, 0.6, 61)))
    # 4. a disturbed system that has no common joint of the traditional kind but is CbD-noncontextual
    tabs = three_bit(0.30, 0.2)
    pooled_ok = None
    # traditional test: one variable per content, with the context tables as margins -> infeasible because the content margins differ
    cells = list(itertools.product([0, 1], repeat=3)); A, b = [], []
    for i, T in enumerate(tabs):
        for a_, b_ in itertools.product([0, 1], repeat=2):
            A.append([1.0 if (c[i] == a_ and c[(i + 1) % 3] == b_) else 0.0 for c in cells]); b.append(T[a_, b_])
    k = len(b); A = np.array(A); Aeq = np.vstack([np.hstack([A, -np.eye(k), np.eye(k)]), np.r_[np.ones(8), np.zeros(2 * k)]])
    l1 = linprog(np.r_[np.zeros(8), np.ones(2 * k)], A_eq=Aeq, b_eq=np.r_[b, 1], bounds=[(0, None)] * (8 + 2 * k), method="highs").fun
    print(f"eps = 0.30 with disturbance d = 0.2: traditional least L1 adjustment {l1:.4f} (incompatible), CbD contextual: {criterion(tabs)[0]} "
          f"(s_odd = {criterion(tabs)[1]:.3f}, n-2+Delta = {1 + criterion(tabs)[2]:.3f})")
    # 1b. a targeted family with strong anticorrelations, so that many systems are contextual
    agree2 = 0; tot2 = 0; ctx2 = 0
    for _ in range(400):
        tabs = []
        for i in range(3):
            T = np.array([[0.02, 0.48], [0.48, 0.02]]) + R.dirichlet(np.ones(4)).reshape(2, 2) * R.uniform(0, 0.8)
            tabs.append(T / T.sum())
        c, _, _ = criterion(tabs); agree2 += (c == (not lp_noncontextual(tabs))); tot2 += 1; ctx2 += c
    print(f"targeted family: LP vs criterion agreement {agree2}/{tot2} (contextual: {ctx2})")
    # 5. contextuality under sampling uncertainty: posterior probability of CbD contextuality from pairwise samples,
    #    against a naive test that ignores direct influences (Delta set to 0)
    def posterior(true_tabs, N, draws=2000):
        counts = [R.multinomial(N, T.ravel()) for T in true_tabs]
        cbd = naive = 0
        for _ in range(draws):
            tabs = [R.dirichlet(1 + c).reshape(2, 2) for c in counts]
            ok, s, D = criterion(tabs); cbd += ok; naive += s > 1 + 1e-12
        return cbd / draws, naive / draws
    for label, eps, d in (("CbD-noncontextual, disturbed (eps 0.30, d 0.2)", 0.30, 0.2), ("CbD-contextual, disturbed (eps 0.25, d 0.1)", 0.25, 0.1),
                          ("consistent, near threshold (eps 0.32, d 0)", 0.32, 0.0)):
        out = [posterior(three_bit(eps, d), N) for N in (100, 400, 1600)]
        print(label + ": P(contextual | data) by CbD / by the naive test at N = 100, 400, 1600 per context: "
              + ", ".join(f"{a:.2f}/{b:.2f}" for a, b in out))
