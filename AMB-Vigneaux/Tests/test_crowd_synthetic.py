import os, importlib.util, itertools
import numpy as np

here = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("crowd_syn", os.path.join(here, "../examples/crowd/synthetic.py"))
SY = importlib.util.module_from_spec(spec); spec.loader.exec_module(SY)


def design(n_items=6000, n_workers=30, seed=0):
    r = np.random.default_rng(seed); W = [f"w{i}" for i in range(n_workers)]
    per, gold = {}, {}
    for q in range(n_items):
        per[q] = {w: 0 for w in r.choice(W, size=3, replace=False)}; gold[q] = int(r.random() < 0.15)
    acc = {w: float(r.uniform(0.75, 0.95)) for w in W}
    return per, gold, acc


def test_oracle_loop_test_sees_planted_interaction_only():
    per, gold, acc = design()
    null, hard0 = SY.simulate(per, gold, acc, dif=True, inter=False, seed=1)
    alt, hard1 = SY.simulate(per, gold, acc, dif=True, inter=True, seed=1)
    q0, p0 = SY.oracle_c(gold, null, hard0); q1, p1 = SY.oracle_c(gold, alt, hard1)
    assert p0 > 0.01 and p1 < 1e-6 and q1 > q0


def test_planted_parity_box_is_even_parity():
    per, gold, acc = design(n_items=2000)
    covers = [tuple(sorted(per[q])) + ("zz",) for q in list(per)[:1]]       # plant into one triple (via a fake quadruple)
    sim, _ = SY.simulate(per, {q: 0 for q in per}, acc, lam=1.0, planted=[covers[0]], seed=2)
    trip = covers[0][:3]
    pats = [tuple(sim[q][w] for w in trip) for q in per if tuple(sorted(per[q])) == trip]
    assert pats and all(sum(p) % 2 == 0 for p in pats)
