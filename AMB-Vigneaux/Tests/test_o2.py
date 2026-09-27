"""O2: exactness of partition merges for a latent B with X ⊥ O | B (Props A–C)."""
import numpy as np


def joint(pb, D, Uc):                       # P(x, o) = Σ_b P(b) D[b, x] Uc[b, o]
    return np.einsum("b,bx,bo->xo", pb, D, Uc)


def merged_joint(pb, D, Uc, blocks):        # X ⊥ O | f(B), with pooled rows
    out = 0
    for S in blocks:
        w = pb[S].sum()
        dS = (pb[S] @ D[S]) / w
        uS = (pb[S] @ Uc[S]) / w
        out = out + w * np.outer(dS, uS)
    return out


rng = np.random.default_rng(0)
rank = lambda X: np.linalg.matrix_rank(X, tol=1e-10)


def test_upward_equal_merge_is_exact():                  # Prop A(ii)
    pb = np.array([.2, .3, .5]); D = rng.dirichlet(np.ones(6), 3)
    u = rng.dirichlet(np.ones(5)); Uc = np.array([u, u, rng.dirichlet(np.ones(5))])
    assert np.allclose(joint(pb, D, Uc), merged_joint(pb, D, Uc, [[0, 1], [2]]))


def test_downward_equal_merge_is_exact():                # Prop A(i)
    pb = np.array([.2, .3, .5]); d = rng.dirichlet(np.ones(6))
    D = np.array([d, d, rng.dirichlet(np.ones(6))]); Uc = rng.dirichlet(np.ones(5), 3)
    assert np.allclose(joint(pb, D, Uc), merged_joint(pb, D, Uc, [[0, 1], [2]]))


def test_rank_deficiency_no_partition_reaches_it():      # Prop C counterexample
    pb = np.array([.25, .25, .5]); D = rng.dirichlet(np.ones(6), 3)
    ua, ub = rng.dirichlet(np.ones(5), 2); Uc = np.array([ua, ub, (ua + ub) / 2])
    J = joint(pb, D, Uc)
    assert rank(D) == 3 and rank(Uc) == 2 and rank(J) == 2      # r* = 2
    for blocks in ([[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]]):
        assert not np.allclose(J, merged_joint(pb, D, Uc, blocks))
