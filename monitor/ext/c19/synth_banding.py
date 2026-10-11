"""Adapter check for banding_interchange.py on planted raw chains (28 states, n = 950):
(a) a first-order raw Markov chain: the banded gap should be explained by banding (primary), the raw failure not detected;
(b) a raw chain with second-order memory (return to the wave-1 score): the primary decision should be 'memory'."""
import numpy as np, banding_interchange as BI
rng = np.random.default_rng(7); n = 950
def walk(v, s=3): return np.clip(v + rng.integers(-s, s + 1, len(v)), 0, 27)
x1 = np.clip(rng.gamma(2.0, 3.5, n).astype(int), 0, 27); x2 = walk(x1); x3 = walk(x2)
ra = BI.run(x1, x2, x3, np.random.default_rng(1))
y3 = np.where(rng.random(n) < 0.6, walk(x1, 1), walk(x2))
rb = BI.run(x1, x2, y3, np.random.default_rng(2))
for name, r in (('first-order', ra), ('second-order', rb)):
    print(name, r['banded_gap_bits'], r['banded_markov_null95'], r['raw_markov_null95_of_banded_gap'], r['decision_primary'], '|', r['decision_secondary'], r['decomposition'])
ok = ra['decision_primary'].startswith('banded failure explained') and rb['decision_primary'].startswith('raw chain has memory')
print('PASS' if ok else 'FAIL')
