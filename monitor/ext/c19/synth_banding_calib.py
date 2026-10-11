"""Calibration of banding_interchange.py on planted raw chains: rejection rates over 20 replicates (n = 950, B = 400):
first-order raw chains (size), and raw chains returning to the wave-1 score with probability 0.2 or 0.6 (power)."""
import json, numpy as np, banding_interchange as BI
rng = np.random.default_rng(11); n = 950; out = {}
def walk(v, s=3): return np.clip(v + rng.integers(-s, s + 1, len(v)), 0, 27)
for mem in (0.0, 0.2, 0.6):
    prim = sec = 0
    for r in range(20):
        x1 = np.clip(rng.gamma(2.0, 3.5, n).astype(int), 0, 27); x2 = walk(x1)
        x3 = np.where(rng.random(n) < mem, walk(x1, 1), walk(x2)); res = BI.run(x1, x2, x3, np.random.default_rng(r))
        prim += res['decision_primary'].startswith('raw chain has memory'); sec += res['decision_secondary'] == 'raw composition failure detected'
        mon = res['banded_gap_bits'] > res['banded_markov_null95']
        out.setdefault(str(mem), dict(primary=0, secondary=0, monitor_banded_null=0))
        out[str(mem)]['monitor_banded_null'] += int(mon)
    out[str(mem)].update(primary=prim, secondary=sec, reps=20)
    print(mem, out[str(mem)], flush=True)
json.dump(out, open('results/banding_calibration.json', 'w'), indent=1)
