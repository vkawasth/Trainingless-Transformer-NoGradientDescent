import itertools

import numpy as np
import pytest

from amb_vigneaux import (ClosedLoopEngine, ContextualFamily, FiniteJoint, HiddenVariableFamily,
                          EmpiricalModel, act, analyse_outcomes, co_information, coboundary,
                          cohomological_obstruction, conditional_entropy, contextual_fraction,
                          entropy, mutual_information, bell_scenario, cyclic_scenario)
from amb_vigneaux.models import (chsh_scenario, ghz_model, hardy_model, noisy, pr_box,
                                 random_noncontextual, tsirelson_box, white_noise)
from amb_vigneaux.zlinalg import solve_integer


# ----------------------------------------------------------------- Z-algebra
def test_integer_solver_matches_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(300):
        m, n = rng.integers(1, 4), rng.integers(1, 4)
        A = rng.integers(-3, 4, size=(m, n)).tolist()
        b = rng.integers(-4, 5, size=m).tolist()
        brute = any(np.array_equal(np.array(A) @ np.array(x), b)
                    for x in itertools.product(range(-12, 13), repeat=n))
        x = solve_integer(A, b)
        if x is not None:
            assert np.array_equal(np.array(A) @ np.array(x), b)
        if brute:
            assert x is not None


def test_integer_vs_rational():
    # 2x = 1 has a rational but no integer solution
    assert solve_integer([[2]], [1]) is None
    assert solve_integer([[2, 3]], [1]) is not None


# ---------------------------------------------------------------- outcomes
def test_pr_box_strong_and_cohomological():
    r = analyse_outcomes(pr_box())
    assert r.strongly_contextual and r.gamma_h1_nonzero
    assert len(r.cohomological_witnesses) == 8 and not r.false_negatives
    assert r.contextual_fraction == pytest.approx(1.0, abs=1e-9)


def test_ghz_strong_cohomological():
    r = analyse_outcomes(ghz_model())
    assert r.strongly_contextual and not r.false_negatives
    assert r.contextual_fraction == pytest.approx(1.0, abs=1e-9)


def test_hardy_logical_not_strong_and_cohomology_false_negative():
    m = hardy_model()
    assert m.prob(("a1", "b1"), (1, 1)) == pytest.approx(1 / 12)
    r = analyse_outcomes(m)
    assert r.level() == "logical"
    assert r.logical_witnesses == [(("a1", "b1"), (1, 1))]
    # Z-cohomology misses Hardy: verify the returned witness family is genuinely compatible
    assert r.false_negatives == r.logical_witnesses
    res = cohomological_obstruction(m.scenario, r.support, ("a1", "b1"), (1, 1))
    sc, fam = m.scenario, res.witness
    def restr(C, U):
        out = {}
        for t, k in fam[C].items():
            u = sc.restrict(t, C, U)
            out[u] = out.get(u, 0) + k
        return {u: k for u, k in out.items() if k}
    for C, D in itertools.combinations(sc.contexts, 2):
        U = sc.overlap(C, D)
        assert restr(C, U) == restr(D, U)


def test_tsirelson_cf():
    r = analyse_outcomes(tsirelson_box())
    assert r.level() == "probabilistic"
    assert r.contextual_fraction == pytest.approx(np.sqrt(2) - 1, abs=1e-6)
    assert r.signalling_defect < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_noncontextual_models(seed):
    m = random_noncontextual(chsh_scenario(), np.random.default_rng(seed))
    r = analyse_outcomes(m)
    assert r.level() == "noncontextual" and r.contextual_fraction < 1e-9


def test_noisy_pr_threshold():
    # noisy PR box v·PR + (1-v)·noise is contextual iff v > 1/2
    assert contextual_fraction(noisy(pr_box(), 0.49)).value < 1e-9
    assert contextual_fraction(noisy(pr_box(), 0.6)).value == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("v", [0.5, 0.7, 0.72, 0.85, 1.0])
def test_noisy_tsirelson_cf_closed_form(v):
    assert contextual_fraction(noisy(tsirelson_box(), v)).value == pytest.approx(max(0, np.sqrt(2) * v - 1), abs=1e-7)


def test_bell_inequality_dual():
    m = tsirelson_box()
    cf = contextual_fraction(m)
    val = sum(cf.bell_inequality[C] @ m.tables[C] for C in m.scenario.contexts)
    assert val == pytest.approx(1 - cf.value, abs=1e-7)
    sc = m.scenario
    X = sc.measurements
    for g in sc.global_sections():   # every deterministic NC model satisfies ≥ 1
        tot = sum(cf.bell_inequality[C][sc.sections(C).index(sc.restrict(g, X, C))] for C in sc.contexts)
        assert tot >= 1 - 1e-7


def test_kcbs_cycle_scenario_runs():
    sc = cyclic_scenario(5)
    r = analyse_outcomes(white_noise(sc))
    assert r.level() == "noncontextual"


# ------------------------------------------------------------- information
def _random_joint(rng, shape=(2, 3, 2)):
    p = rng.dirichlet(np.ones(int(np.prod(shape)))).reshape(shape)
    return FiniteJoint(("x", "y", "z"), p)


def test_entropy_is_cocycle_and_I_is_trivial_coboundary():
    rng = np.random.default_rng(0)
    dH, dtH = coboundary(entropy, 1), coboundary(entropy, 1, trivial=True)
    for _ in range(20):
        P = _random_joint(rng)
        for X, Y in [(("x",), ("y",)), (("x", "z"), ("y",)), (("y",), ("x", "y"))]:
            assert abs(dH(P, X, Y)) < 1e-10
            I = entropy(P, X) + entropy(P, Y) - entropy(P, FiniteJoint.join(X, Y))
            assert dtH(P, X, Y) == pytest.approx(I, abs=1e-10)
            assert mutual_information(P, X, Y) >= -1e-12


def test_action_is_monoid_action():
    rng = np.random.default_rng(3)
    for _ in range(10):
        P = _random_joint(rng)
        lhs = act(("x", "y"), entropy)(P, ("z",))
        rhs = act(("x",), act(("y",), entropy))(P, ("z",))
        assert lhs == pytest.approx(rhs, abs=1e-10)
        assert conditional_entropy(P, ("z",), ("x", "y")) == pytest.approx(
            entropy(P, ("x", "y", "z")) - entropy(P, ("x", "y")), abs=1e-10)


def test_delta_squared_zero():
    rng = np.random.default_rng(5)
    F = lambda P, X: float(np.sum(P.marginal(X) ** 3))  # arbitrary 1-cochain
    ddF = coboundary(coboundary(F, 1), 2)
    for _ in range(5):
        P = _random_joint(rng)
        assert abs(ddF(P, ("x",), ("y",), ("z",))) < 1e-10


def test_renyi2_is_not_a_cocycle():
    rng = np.random.default_rng(7)
    H2 = lambda P, X: float(-np.log2(np.sum(P.marginal(X) ** 2)))
    P = _random_joint(rng)
    assert abs(coboundary(H2, 1)(P, ("x",), ("y",))) > 1e-4


def test_co_information_xor():
    p = np.zeros((2, 2, 2))
    for a, b in itertools.product(range(2), repeat=2):
        p[a, b, a ^ b] = 0.25
    P = FiniteJoint(("x", "y", "z"), p)
    assert co_information(P, ("x",), ("y",), ("z",)) == pytest.approx(-1.0)


# ---------------------------------------------------------------- geometry
def test_gradient_and_fisher():
    sc = chsh_scenario()
    tgt = tsirelson_box()
    for fam in (ContextualFamily(sc), HiddenVariableFamily(sc)):
        th = fam.init_theta(np.random.default_rng(0), 0.5)
        g = fam.grad(th, tgt)
        h = 1e-6
        num = np.array([(fam.loss(th + h * e, tgt) - fam.loss(th - h * e, tgt)) / (2 * h)
                        for e in np.eye(fam.dim)])
        assert np.allclose(g, num, atol=1e-6)
        G = fam.fisher(th)
        assert np.allclose(G, G.T) and np.linalg.eigvalsh(G).min() > -1e-10
        # Fisher = Hessian of KL(P_θ ‖ P_θ') at θ'=θ
        tgt_self = fam.model(th)
        H = np.array([(fam.grad(th + h * e, tgt_self) - fam.grad(th - h * e, tgt_self)) / (2 * h)
                      for e in np.eye(fam.dim)])
        assert np.allclose(H, G, atol=1e-5)


def test_hidden_variable_family_is_noncontextual():
    sc = chsh_scenario()
    fam = HiddenVariableFamily(sc)
    for s in range(5):
        th = fam.init_theta(np.random.default_rng(s), 3.0)
        assert contextual_fraction(fam.model(th)).value < 1e-9


# ------------------------------------------------------------------ engine
def test_loop_learns_pr_box_and_triggers_obstruction():
    sc = chsh_scenario()
    fam = ContextualFamily(sc)
    eng = ClosedLoopEngine(fam, fam.init_theta(), n_samples=400, world=pr_box(), seed=1)
    eng.run(40)
    h = eng.history
    assert not h[0].gamma_latent and h[-1].gamma_latent       # obstruction switches on
    assert all(r.gamma_realised for r in h)                   # PR samples are always obstructed
    assert h[-1].kl_world < 5e-3 and h[-1].cf_model > 0.99   # floor set by η and N (stochastic approximation)
    assert max(r.max_cocycle_defect for r in h) < 1e-9        # chain rule holds throughout


def test_hidden_variable_loop_cannot_reach_quantum():
    sc = chsh_scenario()
    res = {}
    for Fam in (ContextualFamily, HiddenVariableFamily):
        fam = Fam(sc)
        eng = ClosedLoopEngine(fam, fam.init_theta(), n_samples=4000, world=tsirelson_box(), seed=2)
        eng.run(60)
        res[Fam.__name__] = eng.history[-1]
    assert res["HiddenVariableFamily"].cf_model < 1e-9
    assert res["HiddenVariableFamily"].kl_world > 0.02
    assert res["ContextualFamily"].kl_world < 5e-3
    assert res["ContextualFamily"].cf_model > 0.35


def test_line_search_monotone_on_empirical_loss():
    sc = chsh_scenario()
    fam = HiddenVariableFamily(sc)
    eng = ClosedLoopEngine(fam, fam.init_theta(), eta=5.0, n_samples=300, world=tsirelson_box())
    for _ in range(10):
        fs = eng.forward()
        before = fam.loss(eng.theta, fs.empirical)
        new = eng.backward(fs)
        assert fam.loss(new, fs.empirical) <= before + 1e-12
        eng.theta = new


# --------------------------------------------- cross-checks against the HANDOFF
from amb_vigneaux import GlobalLaw
from amb_vigneaux.models import parity_grid, single_source
from amb_vigneaux.scenario import bell_scenario as _bell


def _gamma_count(m, ring, C0=None):
    sc, S = m.scenario, m.support()
    C0 = C0 or sc.contexts[0]
    return sum(not cohomological_obstruction(sc, S, C0, s, ring).obstruction_vanishes for s in S[C0])


def test_parity_grid_ring_dependence():
    """HANDOFF §3: 9/9 over Z and Z_3, 0/9 over Q and Z_2; controls at 0; CF 1,0,0."""
    ob = parity_grid((0, 0, 0), (0, 0, 1))
    assert [_gamma_count(ob, r) for r in ("Z", 3, "Q", 2)] == [9, 9, 0, 0]
    assert contextual_fraction(ob).value == pytest.approx(1.0, abs=1e-9)
    for ctrl in (parity_grid((0, 0, 0), (0, 0, 0)), parity_grid((1, 1, 1), (0, 0, 0))):
        assert [_gamma_count(ctrl, r) for r in ("Z", 3, "Q", 2)] == [0, 0, 0, 0]
        assert contextual_fraction(ctrl).value < 1e-9


@pytest.mark.parametrize("eps", [1e-9, 0.1, 0.2])
def test_possibilistic_fragility(eps):
    """HANDOFF Prop. 10: any smoothing kills γ while CF = 1 − 3ε survives."""
    m = white_noise(parity_grid().scenario).mix(parity_grid(), 1 - eps)
    assert _gamma_count(m, "Z") == 0 and len(m.support()[m.scenario.contexts[0]]) == 27
    assert contextual_fraction(m).value == pytest.approx(1 - 3 * eps, abs=1e-7)


@pytest.mark.parametrize("seed", range(4))
def test_T0_single_source_has_no_obstruction(seed):
    """HANDOFF T0: marginals of one sample always glue — even with sparse support."""
    rng = np.random.default_rng(seed)
    for sc in (chsh_scenario(), _bell(3, 2)):
        law = rng.dirichlet(np.full(sc.n_sections(sc.measurements), 0.05))
        for n in (3, 20, 200):
            r = analyse_outcomes(single_source(law, sc, n, rng))
            assert r.contextual_fraction < 1e-9
            assert not r.logical_witnesses and not r.cohomological_witnesses


def test_engine_single_source_world_is_never_obstructed():
    sc = chsh_scenario()
    law = GlobalLaw(sc, np.random.default_rng(0).dirichlet(np.ones(16)))
    fam = ContextualFamily(sc)
    eng = ClosedLoopEngine(fam, fam.init_theta(), n_samples=200, world=law, seed=1)
    eng.run(20)
    assert eng.series("cf_empirical").max() < 1e-9
    assert not eng.series("gamma_realised").any()
    # observed here (not a theorem): the fitted model's residual CF never exceeds its signalling defect, i.e. it is misfit
    h = eng.history[-1]
    assert h.cf_model <= h.signalling + 1e-9
