import itertools, numpy as np, time
from hchain import *

def brute(P, x, L, V, A):
    from functools import lru_cache
    x = tuple(x)
    @lru_cache(None)
    def gen(l, b, i, j):
        C = A if l == 1 else V
        tot = 0.0
        for k in range(KMAX + 1):
            if j - i < k + 1: break
            for cuts in itertools.combinations(range(i + 1, j), k):
                bd = (i,) + cuts + (j,)
                for syms in itertools.product(range(C), repeat=k + 1):
                    pr = P[f"K{l}"][b, k] * P[f"H{l}"][b, syms[0]]
                    for a, c in zip(syms, syms[1:]): pr *= P[f"R{l}"][b, a, c]
                    if pr == 0: continue
                    for t, c in enumerate(syms):
                        if l == 1:
                            pr *= float(bd[t + 1] - bd[t] == 1 and x[bd[t]] == c)
                        else:
                            pr *= gen(l - 1, c, bd[t], bd[t + 1])
                        if pr == 0: break
                    tot += pr
        return tot
    return sum(gen(L, b, 0, len(x)) / V for b in range(V))

rng = np.random.default_rng(0)
L, V, A = 2, 2, 3
P = init_params(L, V, A, rng)
lr = Learner(L, V, A)
for n in (1, 2, 4, 6):
    x = list(rng.integers(0, A, n))
    B = buckets([x])
    print(n, 'jax', lr.loglik(B, P), 'brute', np.log(brute(P, x, L, V, A)))

def brute_pos(P, x, L, V, A):
    from functools import lru_cache
    x = tuple(x)
    @lru_cache(None)
    def gen(l, b, i, j):
        C = A if l == 1 else V
        tot = 0.0
        for k in range(KMAX + 1):
            if j - i < k + 1: break
            for cuts in itertools.combinations(range(i + 1, j), k):
                bd = (i,) + cuts + (j,)
                for syms in itertools.product(range(C), repeat=k + 1):
                    pr = P[f"K{l}"][b, k] * P[f"H{l}"][b, k, syms[0]]
                    for t, c in enumerate(syms[1:]): pr *= P[f"R{l}"][b, k, t, c]
                    for t, c in enumerate(syms):
                        if pr == 0: break
                        pr *= float(bd[t+1]-bd[t] == 1 and x[bd[t]] == c) if l == 1 else gen(l-1, c, bd[t], bd[t+1])
                    tot += pr
        return tot
    return sum(gen(L, b, 0, len(x)) / V for b in range(V))

P = init_params_pos(L, V, A, rng); lp = LearnerPos(L, V, A)
for n in (1, 3, 5):
    x = list(rng.integers(0, A, n))
    print('pos', n, lp.loglik(buckets([x]), P), np.log(brute_pos(P, x, L, V, A)))
