"""Correction after review: the four-wave latent Markov test with the same number of EM starts (8) for the observed table and for every
bootstrap refit (the frozen run used 20 and 4, which biases the bootstrap p upward). Also the power of the test at the actual sample
sizes against the planted second-order memory of synth_latentT.py. Writes results/b2_equal_starts.json."""
import os, sys, json, tempfile, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import latentT
from c19 import load_uk, load_ie
from synth_latentT import sim
which = sys.argv[1]; out = {}
if which in ('uk', 'ie'):
    with tempfile.TemporaryDirectory() as tmp:
        W = (load_uk(os.path.join(H, 'raw', 'uk')) if which == 'uk' else load_ie(os.path.join(H, 'raw', 'ie_archive.zip'), tmp))[0]
    wide = None
    for t in sorted(W):
        x = W[t][['pid', 'D']].dropna().drop_duplicates('pid').rename(columns={'D': f'D{t}'}); wide = x if wide is None else wide.merge(x, on='pid', how='outer')
    c = wide.dropna(subset=['D1', 'D2', 'D3', 'D4']); N = np.zeros((3,) * 4); np.add.at(N, tuple(c[f'D{t}'].astype(int).values for t in (1, 2, 3, 4)), 1)
    out = dict(n=len(c), shared=latentT.test(N, 4, True, B=200, seed=1, starts=8, refit_starts=8), free=latentT.test(N, 4, False, B=100, seed=2, starts=8, refit_starts=8))
else:   # power at n = 771 and 327 against planted second-order memory, and size under a planted first-order chain
    rng = np.random.default_rng(5)
    for n in (771, 327):
        rej = [latentT.test(sim(n, True, rng), 4, True, B=40, seed=int(rng.integers(1e9)), starts=8, refit_starts=8)['p_boot'] < 0.05 for _ in range(8)]
        out[f'power_n{n}'] = dict(reps=len(rej), rejections=int(sum(rej)))
os.makedirs(os.path.join(H, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(H, 'results', f'b2_equal_starts_{which}.json'), 'w'), indent=1); print(which, json.dumps(out))
