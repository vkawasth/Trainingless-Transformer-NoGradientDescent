import numpy as np
import pytest

from amb_vigneaux.functors import (canonical_nc_part, extension_exists, graded_extension,
                                   latent_class_em, max_entropy_extension, restrict, window_counts)
from amb_vigneaux.models import (chsh_scenario, ghz_model, hardy_model, noisy, parity_grid, pr_box,
                                 random_noncontextual, tsirelson_box)
from amb_vigneaux.outcome import contextual_fraction
from amb_vigneaux.scenario import EmpiricalModel, Scenario, bell_scenario


@pytest.mark.parametrize("sc", [chsh_scenario(), bell_scenario(3, 2)])
def test_R_image_is_noncontextual(sc):
    rng = np.random.default_rng(0)
    for _ in range(5):
        P = rng.dirichlet(np.full(sc.n_sections(sc.measurements), 0.2))
        e = restrict(sc, P)
        assert contextual_fraction(e).value < 1e-9 and extension_exists(e)


def test_E_nonempty_iff_CF_zero():
    rng = np.random.default_rng(1)
    models = [random_noncontextual(chsh_scenario(), rng), noisy(pr_box(), 0.4), noisy(pr_box(), 0.6),
              tsirelson_box(), hardy_model(), pr_box(), parity_grid(), parity_grid((0, 0, 0), (0, 0, 0))]
    for m in models:
        assert extension_exists(m) == (contextual_fraction(m).value < 1e-9)


def test_R_E_identities():
    rng = np.random.default_rng(2)
    sc = chsh_scenario()
    for _ in range(5):
        P = rng.dirichlet(np.ones(16))
        e = restrict(sc, P)
        me = max_entropy_extension(e)
        assert me.converged
        back = restrict(sc, me.P)                           # R(E(e)) = e
        assert all(np.allclose(back.tables[C], e.tables[C], atol=1e-8) for C in sc.contexts)
        # P ∈ E(R(P)); the max-entropy member has at least P's entropy
        H = lambda x: float(-(x[x > 0] * np.log(x[x > 0])).sum())
        assert H(me.P) >= H(P) - 1e-9


def test_maxent_fails_to_converge_on_contextual_models():
    assert not max_entropy_extension(tsirelson_box(), iters=500).converged


def test_optimal_face_uniqueness_depends_on_scenario():
    rng = np.random.default_rng(3)
    sc = chsh_scenario()
    for top in (pr_box(), tsirelson_box(), hardy_model()):
        m = random_noncontextual(sc, rng, 0.3).mix(top, 0.95)
        g = graded_extension(m)
        assert g.cf > 1e-6 and g.unique
    m3 = random_noncontextual(bell_scenario(3, 2), np.random.default_rng(4), 0.3).mix(ghz_model(), 0.8)
    g3 = graded_extension(m3)
    assert g3.cf > 1e-6 and not g3.unique
    c = canonical_nc_part(m3)                               # unique max-entropy representative
    sc3 = m3.scenario
    A = np.vstack([sc3.restriction_matrix(sc3.measurements, C) for C in sc3.contexts])
    b = np.concatenate([m3.tables[C] for C in sc3.contexts])
    assert c.sum() == pytest.approx(1 - g3.cf, abs=1e-6) and (A @ c - b).max() < 1e-7


def _stream(n, shift02, rng, eps=0.1, L=5):
    x0 = rng.integers(0, L, n)
    noise = lambda k: np.where(rng.random(n) < eps, rng.integers(0, L, n), k)
    x1 = noise(x0)
    x2 = noise(x1) if shift02 is None else noise((x0 + shift02) % L)
    return np.stack([x0, x1, x2], 1)


SC3 = Scenario({f"x{i}": tuple(range(5)) for i in range(3)}, (("x0", "x1"), ("x1", "x2"), ("x0", "x2")))


def test_one_stream_em_is_never_contextual():
    """T0: data windows and the EM fit are both single-source, so CF = 0."""
    rng = np.random.default_rng(0)
    W = _stream(20000, None, rng)
    assert contextual_fraction(window_counts(SC3, W)).value < 1e-9
    _, _, joint = latent_class_em(W, 5, 3, iters=60, rng=rng)
    assert contextual_fraction(restrict(SC3, joint)).value < 1e-9


def test_two_streams_can_be_contextual():
    rng = np.random.default_rng(0)
    A, B = window_counts(SC3, _stream(20000, None, rng)), window_counts(SC3, _stream(20000, 2, rng))
    two = EmpiricalModel(SC3, {("x0", "x1"): A.tables[("x0", "x1")], ("x1", "x2"): A.tables[("x1", "x2")],
                               ("x0", "x2"): B.tables[("x0", "x2")]})
    assert contextual_fraction(two).value == pytest.approx(0.85, abs=0.02)
    assert not extension_exists(two)
