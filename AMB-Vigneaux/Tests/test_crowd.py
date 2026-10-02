import os, sys, importlib.util
import numpy as np

here = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("crowd_explore", os.path.join(here, "../examples/crowd/explore.py"))
EX = importlib.util.module_from_spec(spec); spec.loader.exec_module(EX)


def test_benjamini_hochberg():
    q = EX.bh([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)


def test_one_coin_dawid_skene_beats_majority_on_planted_skills():
    rng = np.random.default_rng(0); acc = {f"w{i}": a for i, a in enumerate([0.95, 0.9, 0.6, 0.55, 0.55])}
    per, gold = {}, {}
    for q in range(3000):
        y = int(rng.random() < 0.3); ws = rng.choice(list(acc), size=3, replace=False)
        per[q] = {w: (y if rng.random() < acc[w] else 1 - y) for w in ws}; gold[q] = y
    post = EX.one_coin_ds(per, iters=60)
    ds = np.mean([(post[q] > .5) == gold[q] for q in per]); mv = np.mean([(sum(v.values()) >= 2) == gold[q] for q, v in per.items()])
    assert ds > mv
