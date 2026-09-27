import pytest
import numpy as np

from amb_vigneaux.na_contexts import (analyse_stream, gyo_reduce, honest_stream, missing_data_em,
                                      structured_stream, NA)


def test_gyo():
    f = frozenset
    assert gyo_reduce([f({0, 1}), f({1, 2}), f({2, 3})]) == []                    # path: acyclic
    assert gyo_reduce([f({0, 1}), f({1, 2}), f({0, 2}), f({0, 1, 2})]) == []      # covered triangle
    assert len(gyo_reduce([f({0, 1}), f({1, 2}), f({0, 2})])) == 3                # triangle: cyclic
    assert len(gyo_reduce([f({0, 1}), f({1, 2}), f({2, 3}), f({3, 0})])) == 4     # 4-cycle


def test_gate_stops_on_acyclic_cover():
    v = analyse_stream(honest_stream(3000, np.random.default_rng(0), one_na=False, na_rate=0.3))
    assert v.acyclic and "AMB half empty" in v.verdict


def test_honest_cyclic_stream_is_at_the_null():
    v = analyse_stream(honest_stream(6000, np.random.default_rng(0), one_na=True),
                       np.random.default_rng(1), n_null=15)
    assert not v.acyclic and v.p_value > 0.05


def test_structured_stream_exceeds_the_null_and_gamma_fires_on_thresholded_support():
    rec = structured_stream(6000, np.random.default_rng(0), eps=0.1, twist="shift")
    v = analyse_stream(rec, np.random.default_rng(1), n_null=15, min_frac=0.02)
    assert v.cf > 0.8 and v.cf > max(v.null_cf) and v.gamma_nonzero == 5
    raw = analyse_stream(rec, np.random.default_rng(1), n_null=5, min_frac=0.0)
    assert raw.gamma_nonzero == 0                                  # raw support: fragility


def test_negation_to_na_value_shows_as_signalling():
    rec = structured_stream(6000, np.random.default_rng(0), eps=0.1, twist="negation")
    v = analyse_stream(rec, np.random.default_rng(1), n_null=5)
    assert v.signalling > 0.1 and (rec == 5).any() and (rec == NA).any()


def test_missing_data_em_recovers_consistent_law():
    rng = np.random.default_rng(3)
    rec = honest_stream(20000, rng, one_na=True, eps=0.2)
    P = missing_data_em(rec, 6)
    assert abs(P.sum() - 1) < 1e-9
    # x0 is uniform on the five real labels and never n/a
    marg = P.reshape(6, 6, 6).sum((1, 2))
    assert np.allclose(marg[:5], 0.2, atol=0.02) and marg[5] < 1e-6


from amb_vigneaux.na_contexts import (deficit, pinned_consistent_law, population_deficit, threshold_sweep)


def test_pinned_law_is_start_independent():
    rec = honest_stream(4000, np.random.default_rng(0), one_na=True)
    a = pinned_consistent_law(rec, 6)
    b = pinned_consistent_law(rec, 6, init=np.random.default_rng(1).dirichlet(np.ones(216)))
    raw_a = missing_data_em(rec, 6)
    raw_b = missing_data_em(rec, 6, init=np.random.default_rng(1).dirichlet(np.ones(216)))
    assert np.abs(raw_a - raw_b).sum() > 1e-3           # the ridge: EM ends depend on the start
    assert np.abs(a - b).sum() < 1e-3                   # the pinned (max-entropy) law does not


def test_deficit_is_not_a_function_of_cf():
    from amb_vigneaux.channels import cycle, edge_family, noisy_shift
    from amb_vigneaux.outcome import contextual_fraction
    vals = []
    for L in (3, 4, 5):
        g = np.zeros(L, int); g[0] = 1
        m = edge_family(cycle(L), [noisy_shift(5, x, 1.0 / L) for x in g])      # CF = 0.5 for every L
        assert contextual_fraction(m).value == pytest.approx(0.5, abs=1e-6)
        vals.append(population_deficit(m))
    assert vals[0] > vals[1] > vals[2] and vals[0] - vals[2] > 0.04
    for L in (3, 4, 5):                                   # closed form at ε = 0
        g = np.zeros(L, int); g[0] = 1
        m = edge_family(cycle(L), [noisy_shift(5, x, 0.0) for x in g])
        assert population_deficit(m) == pytest.approx(np.log(L / (L - 1)), abs=1e-4)


def test_deficit_floor_scales_with_n():
    small = deficit(honest_stream(1000, np.random.default_rng(0), one_na=True))[0]
    large = deficit(honest_stream(9000, np.random.default_rng(0), one_na=True))[0]
    ctx = deficit(structured_stream(9000, np.random.default_rng(0), twist="shift"))[0]
    assert large < small / 4 and ctx > 0.25


def test_threshold_sweep_exposes_false_positives():
    rows = {t: (lw, g, nl) for t, lw, g, nl in threshold_sweep(
        honest_stream(9000, np.random.default_rng(0), one_na=True), thresholds=(0.005, 0.02), n_null=8)}
    assert rows[0.005][2] >= 6                            # null fires near a cell's mass
    assert rows[0.02][0] == 0 and rows[0.02][2] == 0      # clean window: nothing in data or null
    rows = {t: (lw, g, nl) for t, lw, g, nl in threshold_sweep(
        structured_stream(9000, np.random.default_rng(0), twist="shift"), thresholds=(0.02,), n_null=8)}
    assert rows[0.02][1] == 5 and rows[0.02][2] == 0      # γ fires with a clean null
