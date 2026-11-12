"""Checks for the commutation criterion (module pce/refitcommute.py). Refitting the fine data and then coarsening equals coarsening and
then refitting the coarse data, for every positive starting law, exactly when every fine statistic read along any section of the
coarsening is a coarse statistic; pointwise: (1) a class on which some fine statistic varies has its indicator among the coarse
statistics, and (2) the fine statistics read along one section are coarse statistics.
  CM1 IDENTITY  the pointwise form agrees with the all-sections form (300 random designs: random partitions of a 12-state space X x Y x Z,
                contexts mixing variable sets, event partitions and pulled-back coarse partitions; coarse data the largest consistent span,
                or a random smaller one)
  CM2 IDENTITY  sufficiency: in every design meeting the criterion, the defect is below 1e-8 for 15 random starting laws
  CM3 IDENTITY  necessity: in every design failing it, some starting law gives a defect above 1e-6
  CM4 IDENTITY  variable designs: when the contexts are sets of variables and two values of X are merged, the criterion holds exactly when
                X appears in no context or one context contains every variable (200 designs)
  CM5 IDENTITY  an event design commutes although a fine statistic is not coarse-measurable and the coarse fibre is not a point: X seen as
                {a, b, {c, d}}, Y seen only inside class A; replacing the first context by X itself breaks it
  CM6 IDENTITY  the earlier cases: the margins of X and Y (sec. refitdefect) fail; coarse-measurable data hold; a kept full table holds
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import refitcommute as RC
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(37)
S12 = [(x, y, z) for x in range(3) for y in range(2) for z in range(2)]; n = 12
VAR = lambda vs: np.array([hash(tuple(w[v] for v in vs)) % 10**9 for w in S12])

def coarse_span(Hf, g, m, smaller=False):
    G = RC.intersect(Hf, RC.pull(g, np.eye(m)))
    sigma0 = np.array([np.where(np.asarray(g) == c)[0][0] for c in range(m)])
    Hc = RC.orth(np.column_stack([np.ones(m), G[sigma0]]))
    if smaller and Hc.shape[1] > 2:
        k = rng.integers(1, Hc.shape[1]); Hc = RC.orth(np.column_stack([np.ones(m), Hc @ rng.normal(0, 1, (Hc.shape[1], k))]))
    return Hc

def random_design():
    m = int(rng.integers(3, 9)); g = np.concatenate([np.arange(m), rng.integers(0, m, n - m)]); rng.shuffle(g)
    ctxs = []
    for _ in range(rng.integers(1, 4)):
        k = rng.integers(3)
        if k == 0: ctxs.append(VAR([v for v in range(3) if rng.random() < 0.5] or [0]))
        elif k == 1: ctxs.append(rng.integers(0, rng.integers(2, 5), n))
        else: ctxs.append(rng.integers(0, 3, m)[g])
    Hf = RC.span_labels(ctxs, n); return g, m, Hf, coarse_span(Hf, g, m, smaller=rng.random() < 0.3)

def maxdefect(Hf, Hc, g, m, k=15):
    P = rng.dirichlet(np.ones(n if len(g) == n else len(g)) * 3)
    return max(RC.defect(rng.dirichlet(np.ones(len(g))), Hf, Hc, g, m, P) for _ in range(k))

# CM1-CM3
agree = True; holds, fails, nontriv = [], [], 0
for _ in range(300):
    g, m, Hf, Hc = random_design(); c1, varies, _ = RC.criterion(Hf, Hc, g, m); c2, _ = RC.criterion_sections(Hf, Hc, g, m); agree &= c1 == c2
    d = maxdefect(Hf, Hc, g, m); (holds if c1 else fails).append(d)
    if c1 and varies and np.linalg.matrix_rank(Hc) < m: nontriv += 1
check("CM1 IDENTITY the pointwise criterion equals the all-sections criterion", agree, f"300 designs; {len(holds)} hold, {len(fails)} fail")
check("CM2 IDENTITY sufficiency: designs meeting the criterion have no defect", len(holds) >= 20 and max(holds) < 1e-8,
      f"{len(holds)} designs, largest defect {max(holds):.1e}; {nontriv} of them with a statistic varying inside a class and an open coarse fibre")
check("CM3 IDENTITY necessity: designs failing the criterion have a defect", len(fails) >= 20 and min(fails) > 1e-6,
      f"{len(fails)} designs, smallest largest-defect {min(fails):.2e}")

# CM4
g4 = np.array([{0: 0, 1: 0, 2: 1}[x] * 4 + y * 2 + z for x, y, z in S12]); m4 = 8; ok = True; kinds = {"no X": 0, "full": 0, "fail": 0}
subsets = [s for r in range(1, 4) for s in itertools.combinations(range(3), r)]
for _ in range(200):
    cs = [subsets[i] for i in rng.choice(len(subsets), rng.integers(1, 4), replace=False)]
    Hf = RC.span_labels([VAR(c) for c in cs], n); Hc = coarse_span(Hf, g4, m4); crit = RC.criterion(Hf, Hc, g4, m4)[0]
    noX = all(0 not in c for c in cs); full = any(len(c) == 3 for c in cs); pred = noX or full
    ok &= crit == pred; kinds["no X" if noX else "full" if full else "fail"] += 1
    if _ < 40: ok &= (maxdefect(Hf, Hc, g4, m4, 5) < 1e-8) == crit
check("CM4 IDENTITY with variable contexts, commutation holds iff the merged variable is in no context or the data pin the law", ok, str(kinds))

# CM5, CM6 on X in {a,b,c,d} x Y, merge {a,b} -> A, {c,d} -> B
S8 = [(x, y) for x in range(4) for y in range(2)]; g8 = np.array([(0 if x < 2 else 1) * 2 + y for x, y in S8]); m8 = 4
lab = lambda f: np.array([hash(f(w)) % 10**9 for w in S8])
Xpp = lab(lambda w: w[0] if w[0] < 2 else "cd"); EAY = lab(lambda w: (w[1] if w[0] < 2 else "B")); XX = lab(lambda w: w[0])
Hf5 = RC.span_labels([Xpp, EAY], 8); Hc5 = coarse_span(Hf5, g8, m8); c5, var5, _ = RC.criterion(Hf5, Hc5, g8, m8)
P8 = rng.dirichlet(np.ones(8) * 3); d5 = max(RC.defect(rng.dirichlet(np.ones(8)), Hf5, Hc5, g8, m8, P8) for _ in range(100))
Hf5b = RC.span_labels([XX, EAY], 8); Hc5b = coarse_span(Hf5b, g8, m8); c5b = RC.criterion(Hf5b, Hc5b, g8, m8)[0]
d5b = max(RC.defect(rng.dirichlet(np.ones(8)), Hf5b, Hc5b, g8, m8, P8) for _ in range(20))
fa = np.array([1.0 if x == 0 else 0.0 for x, y in S8])
ok = c5 and d5 < 1e-9 and RC.residual(fa, RC.pull(g8, np.eye(m8))) > 0.1 and m8 - np.linalg.matrix_rank(Hc5) == 1 and not c5b and d5b > 1e-6
check("CM5 IDENTITY an event design commutes with a statistic varying inside a class and an open coarse fibre; seeing X fully breaks it", ok,
      f"classes where a statistic varies {var5}; coarse fibre dimension {m8 - np.linalg.matrix_rank(Hc5)}; largest defect {d5:.1e} (100 laws); with X itself: largest defect {d5b:.3f}")
YY = lab(lambda w: w[1]); XcY = lab(lambda w: ((w[0] >= 2), w[1])); Xc = lab(lambda w: w[0] >= 2)
cases = {"X and Y margins": ([XX, YY], False), "X' and Y (coarse-measurable)": ([Xc, YY], True), "X and the full table": ([XX, lab(lambda w: w)], True),
         "X and {X', Y}": ([XX, XcY], True)}
ok = True; info = []
for name, (cs, want) in cases.items():
    Hf = RC.span_labels(cs, 8); Hc = coarse_span(Hf, g8, m8); c = RC.criterion(Hf, Hc, g8, m8)[0]
    d = max(RC.defect(rng.dirichlet(np.ones(8)), Hf, Hc, g8, m8, P8) for _ in range(20)); ok &= c == want and ((d < 1e-9) == want)
    info.append(f"{name}: {'holds' if c else 'fails'} ({d:.1e})")
check("CM6 IDENTITY the criterion reproduces the earlier cases", ok, "; ".join(info))
json.dump(dict(holds=len(holds), fails=len(fails), nontrivial=nontriv, cm4=kinds, cm5_defect=d5, cm5_broken=d5b),
          open(os.path.join(HERE, "refitcommute_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
