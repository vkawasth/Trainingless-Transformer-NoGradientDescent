import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../examples/lattice_path"))
from amb_vigneaux.lattice_path import _without


def test_hull_is_blind_to_direction_loop_sign_is_not():
    import hull_experiment as HE
    rng = np.random.default_rng(3)
    S, T = [f"s{i}" for i in range(5)], [f"t{j}" for j in range(6)]
    cells = {}
    for s in S:
        for t in T:
            n = int(rng.integers(15, 60)); cells[(s, t)] = (int(rng.integers(0, n + 1)), n)
    tg = ("s0", "t0"); cw = _without(cells, tg); fl = {c: (n - k, n) for c, (k, n) in cw.items()}
    for kind in ("value", "profile", "loop"):
        d1, Z1 = HE.chain(HE.features(cw, kind, S, T), T, 0)
        d2, Z2 = HE.chain(HE.features(fl, kind, S, T), T, 0)
        assert d1 == d2 and np.allclose(Z1[:, 2], Z2[:, 2])          # discs and radii unchanged by relabelling
    H, Hf = HE.residuals(cw, S, T), HE.residuals(fl, S, T)
    m = ~np.isnan(H)
    assert np.allclose(Hf[m], -H[m])                                 # the signed loop part carries the direction
