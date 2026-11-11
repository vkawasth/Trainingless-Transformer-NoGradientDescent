"""Study 26 world: the synthetic sources. Frozen with PREREG_study26.md before any table is generated or examined.

All numbers are invented. Variable names borrow the sleep-onset illustration only to make the bookkeeping readable; nothing here has a
causal, clinical or physiological reading.

Core space: five binary variables, in this order: M (evening melatonin exposure), C (caffeine), R (circadian alignment), H (prior-use
history), Y (sleep onset within 30 minutes). 32 states, index = 16 M + 8 C + 4 R + 2 H + Y.
Light: L3 in {dim, moderate, bright} (evening light), observed only in its own tables.
Instruments: four binary readings of quick onset, A (polysomnography), B (actigraphy), D (diary), E (phone app), observed only in pairs.

generate() returns (development, holdout): lists of dicts with keys id, family, ctx (tuple of variable names), n, counts (array over the
context's cells in itertools.product order), kind ('honest', 'shift', 'coalition', 'miscoded'). The analysis must not read 'kind'
except where PREREG_study26.md says so (the scoring of the flagging rule, after the analysis is run).
"""
from __future__ import annotations
import itertools
import numpy as np

SEED = 2611
VARS = ("M", "C", "R", "H", "Y")
STATES = list(itertools.product((0, 1), repeat=5))

THETA_MAIN = dict(M=0.0, C=-0.3, R=0.4, H=-0.2, Y=0.0)
THETA_PAIR = {("M", "C"): -0.3, ("M", "R"): 0.2, ("C", "R"): -0.5, ("M", "H"): 0.6, ("C", "H"): 0.3, ("M", "Y"): 0.3, ("C", "Y"): -0.8,
              ("R", "Y"): 0.6, ("H", "Y"): -0.2}
THETA_TRIPLE = {("M", "R", "Y"): 0.8, ("M", "H", "Y"): -0.6, ("C", "R", "Y"): 0.0}

SHIFT_M1 = 0.80        # P(M = 1) in the other population (its conditional law of C given M is the truth's)
COALITION_DELTA = 0.10 # coalition raises P(Y = 1 | M = 1, C = c) by this and lowers P(Y = 1 | M = 0, C = c) to keep P(Y = 1, C = c)
LIGHT = (0.45, 0.35, 0.20)
INSTR_FLIP = 0.12      # each instrument disagrees with the latent onset with this probability (independently)

# development: 270 core + 30 light = 300
CORE_ALLOC = [(("M", "C"), 35, "exposure"), (("M", "R"), 35, "circadian"), (("C", "R"), 30, "circadian"), (("M", "H"), 30, "history"),
              (("C", "H"), 30, "history"), (("M", "C", "Y"), 45, "outcome"), (("R", "H", "Y"), 35, "outcome"), (("M", "C", "R", "H"), 30, "overlap")]
N_SHIFT = 10           # of the 35 {M, C} sources
N_COALITION = 15       # of the 45 {M, C, Y} sources, each with n = 400
N_LIGHT = 30
# holdout: 100
HOLD_ALLOC = [(("M", "C", "R", "Y"), 30, "new context"), (("M", "Y"), 10, "inspection"), (("M", "R", "Y"), 10, "calibration"),
              (("M", "H", "Y"), 10, "calibration")]
CYCLE_EDGES = [("A", "B"), ("B", "D"), ("D", "E"), ("E", "A")]   # 10 tables per edge; 40 tables
CYCLE_CONDITIONS = ("consistent", "twisted")                    # 20 tables each: 5 per edge per condition; in 'twisted' only the
                                                                # (E, A) tables come from studies whose app coding was reversed


def truth():
    logp = np.zeros(32)
    for i, s in enumerate(STATES):
        v = dict(zip(VARS, s))
        logp[i] = sum(t * v[a] for a, t in THETA_MAIN.items()) + sum(t * v[a] * v[b] for (a, b), t in THETA_PAIR.items()) \
            + sum(t * v[a] * v[b] * v[c] for (a, b, c), t in THETA_TRIPLE.items())
    p = np.exp(logp); return p / p.sum()


def marginal(P, ctx):
    idx = [VARS.index(v) for v in ctx]; out = np.zeros(2 ** len(ctx))
    for i, s in enumerate(STATES):
        out[int("".join(str(s[j]) for j in idx), 2)] += P[i]
    return out


def shifted():
    P = truth(); pm = marginal(P, ("M",)); Q = P.copy()
    for i, s in enumerate(STATES): Q[i] = P[i] / pm[s[0]] * (SHIFT_M1 if s[0] == 1 else 1 - SHIFT_M1)
    return Q


def coalition_mcy():
    """the coalition's {M, C, Y} table: honest {M, C} margin and honest P(Y = 1, C = c); P(Y = 1 | M = 1, C = c) raised by DELTA"""
    T = marginal(truth(), ("M", "C", "Y")).reshape(2, 2, 2).copy()
    for c in (0, 1):
        pm1, pm0 = T[1, c].sum(), T[0, c].sum()
        y1 = min(T[1, c, 1] / pm1 + COALITION_DELTA, 0.98); y0 = (T[1, c, 1] + T[0, c, 1] - y1 * pm1) / pm0
        T[1, c] = [pm1 * (1 - y1), pm1 * y1]; T[0, c] = [pm0 * (1 - y0), pm0 * y0]
    return T.ravel()


def instruments(twisted):
    """joint law of (A, B, D, E) from a latent onset with P = 1/2 and independent flips; 'twisted': the app E is mis-coded (reads 1 - E)"""
    out = np.zeros(16)
    for z in (0, 1):
        for i, s in enumerate(itertools.product((0, 1), repeat=4)):
            pr = 0.5
            for k, x in enumerate(s):
                read = 1 - x if (twisted and k == 3) else x
                pr *= (1 - INSTR_FLIP) if read == z else INSTR_FLIP
            out[i] += pr
    return out


def pair_marginal(J, a, b):
    names = ("A", "B", "D", "E"); ia, ib = names.index(a), names.index(b); out = np.zeros(4)
    for i, s in enumerate(itertools.product((0, 1), repeat=4)): out[2 * s[ia] + s[ib]] += J[i]
    return out


def generate():
    rng = np.random.default_rng(SEED); P = truth(); Ps = shifted(); dev, hold = [], []
    def add(lst, family, ctx, n, probs, kind):
        lst.append(dict(id=f"{'D' if lst is dev else 'V'}{len(lst) + 1:03d}", family=family, ctx=tuple(ctx), n=int(n),
                        counts=rng.multinomial(int(n), probs / probs.sum()), kind=kind))
    for ctx, k, fam in CORE_ALLOC:
        for j in range(k):
            n = rng.integers(80, 401)
            if ctx == ("M", "C") and j < N_SHIFT: add(dev, fam, ctx, n, marginal(Ps, ctx), "shift")
            elif ctx == ("M", "C", "Y") and j < N_COALITION: add(dev, fam, ctx, 400, coalition_mcy(), "coalition")
            else: add(dev, fam, ctx, n, marginal(P, ctx), "honest")
    for _ in range(N_LIGHT): add(dev, "light", ("L3",), rng.integers(80, 401), np.array(LIGHT), "honest")
    order = rng.permutation(len(dev)); dev = [dev[i] for i in order]
    for ctx, k, fam in HOLD_ALLOC:
        for _ in range(k): add(hold, fam, ctx, 300, marginal(P, ctx), "honest")
    J0, J1 = instruments(False), instruments(True)
    for cond in CYCLE_CONDITIONS:
        for a, b in CYCLE_EDGES:
            bad = cond == "twisted" and (a, b) == ("E", "A")        # the app's coding changed in the studies that paired it with A
            for _ in range(5): add(hold, "cycle-" + cond, (a, b), 400, pair_marginal(J1 if bad else J0, a, b), "miscoded" if bad else "honest")
    return dev, hold
