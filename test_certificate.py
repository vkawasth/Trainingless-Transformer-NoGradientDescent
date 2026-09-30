import numpy as np
from amb_vigneaux.certificate import certificate, cells, hodge


def _units(rng, loop=0.0, n=40, S=3, T=3):
    U = []
    for s in range(S):
        for t in range(T):
            for k in range(n):
                y = 0.3 * s - 0.2 * t + (loop if (s, t) == (0, 0) else 0) + rng.normal()
                U.append((f"s{s}", f"t{t}", y, (s, t, k)))
    return U


def test_exact_cochain_has_no_loop():
    C = {(f"s{s}", f"t{t}"): (0.3 * s - 0.2 * t, 1.0, 10) for s in range(3) for t in range(4)}
    H = hodge(C)
    assert H["beta1"] == (3 - 1) * (4 - 1) and H["Q"] < 1e-18 and H["beta0"] == 1


def test_planted_loop_certified_and_survives_refinement():
    c = certificate(_units(np.random.default_rng(0), loop=1.2), B=60, n_refine=10)
    assert c["p"] < 0.01 and c["snr"] > 2 and c["margin"] > 1 and c["refine_frac_sig"] >= 0.9


def test_null_not_certified():
    c = certificate(_units(np.random.default_rng(1), loop=0.0), B=60, n_refine=10)
    assert c["p"] > 0.01 and c["snr"] < 2


def test_disconnected_cover_has_zero_gap():
    C = {("a", "x"): (0, 1, 5), ("a", "y"): (1, 1, 5), ("b", "z"): (2, 1, 5), ("c", "z"): (0, 1, 5)}
    H = hodge(C)
    assert H["beta0"] == 2 and H["fiedler"] < 1e-9
