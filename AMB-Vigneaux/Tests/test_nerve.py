import numpy as np
import pytest

from amb_vigneaux.nerve import (DecoratedNerve, affine, affine_group, affine_params, compose,
                                conjugacy_classes, fixed_codimension, identity, inverse, pair_radius,
                                push, ring_tokens, time_loop_label)

G3, G5 = affine_group(3), affine_group(5)


def test_affine_group_structure():
    assert len(G3) == 6 and len(G5) == 20
    cls5 = conjugacy_classes(G5)
    # classes of Aff(Z/5): identity, all translations, and one class per multiplier u = 2, 3, 4
    assert len(set(cls5.values())) == 5
    assert len({cls5[affine(5, 1, t)] for t in range(1, 5)}) == 1
    # Z/3 negation T<->F fixing U is affine (x -> -x + 1); in the user's Z/5 order it is not
    assert affine(3, 2, 1) == (1, 0, 2)
    assert (1, 0, 3, 2, 4) not in G5


def test_fixed_codimension_walls():
    assert sorted({fixed_codimension(g) for g in G3 if g != identity(3)}) == [1, 2]
    assert all(fixed_codimension(affine(3, 2, t)) == 1 for t in range(3))      # reflections are walls
    assert min(fixed_codimension(g) for g in G5 if g != identity(5)) == 2      # no walls in Aff(Z/5)


def test_pair_radius_is_the_global_minimax():
    """Minimax theorem: min_c max(D(p‖c), D(q‖c)) = max_λ [λD(p‖m) + (1−λ)D(q‖m)],
    m = λp + (1−λ)q (the right-sided KL centroid is the mean).  The segment
    solver must hit this lower bound, which certifies it as the global optimum."""
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(0)
    kl = lambda a, c: float((a * np.log(a / c)).sum())
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        dual = lambda lam: -(lam * kl(p, lam * p + (1 - lam) * q) + (1 - lam) * kl(q, lam * p + (1 - lam) * q))
        lower = -minimize_scalar(dual, bounds=(0, 1), method="bounded", options=dict(xatol=1e-12)).fun
        assert pair_radius(p[None], q[None])[0] == pytest.approx(lower, abs=1e-9)


def test_z3_walls_absorb_every_planted_element():
    for h in G3:
        if h == identity(3):
            continue
        for seed in range(5):
            X, T = ring_tokens(3, h, 40, np.random.default_rng(seed), G3)
            assert time_loop_label(X, T, G3)[0] != h


def test_z3_translation_group_recovers_both_orientations():
    Tr = [affine(3, 1, t) for t in range(3)]
    for t in (1, 2):
        h = affine(3, 1, t)
        X, T = ring_tokens(3, h, 40, np.random.default_rng(0), Tr)
        lab, margin = time_loop_label(X, T, Tr)
        assert lab == h and margin > 1e-3
        assert time_loop_label(X, -T, Tr)[0] == inverse(h)          # B before A reads ω^{-t}


def test_z5_unambiguous_rings_are_recovered():
    wrong = total = 0
    for (u, t) in [(1, 1), (1, 3), (4, 2), (2, 0), (3, 1)]:
        h = affine(5, u, t)
        for seed in range(8):
            X, T = ring_tokens(5, h, 40, np.random.default_rng(1000 + seed), G5)
            lab, margin = time_loop_label(X, T, G5)
            if margin >= 1e-3:
                total += 1
                wrong += lab != h
    assert total >= 15 and wrong <= 1


def test_relabelling_preserves_class_not_element():
    cls = conjugacy_classes(G5)
    rng = np.random.default_rng(2)
    h = affine(5, 1, 1)
    X, T = ring_tokens(5, h, 40, rng, G5)
    f = time_loop_label(X, T, G5)[0]
    frames = [G5[k] for k in rng.integers(0, len(G5), len(X))]
    g = time_loop_label(np.array([push(fr, x) for fr, x in zip(frames, X)]), T, G5)[0]
    assert cls[f] == cls[g]


def test_decorated_nerve_labels_a_translation_bar():
    Tr = [affine(3, 1, t) for t in range(3)]
    h = affine(3, 1, 2)
    X, T = ring_tokens(3, h, 24, np.random.default_rng(0), Tr)
    dn = DecoratedNerve(X, T, Tr, iters=600)
    big = [b for b in dn.h1_bars() if b.cycle and len(set(b.cycle)) > 12]
    assert any(b.label == h and b.killed_by_nonflat for b in big)
    flat = DecoratedNerve(*ring_tokens(3, identity(3), 24, np.random.default_rng(0), Tr), Tr, iters=600)
    assert flat.first_nonflat == np.inf
