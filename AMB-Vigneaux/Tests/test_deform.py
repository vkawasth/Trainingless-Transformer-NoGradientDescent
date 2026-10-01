import numpy as np
from amb_vigneaux import deform as D


def test_boundary_supports_are_zero_sets_meeting_both_parities():
    r = D.boundary_supports(3)
    for (a, b), v in r.items():
        if a >= 1 and b >= 1:
            assert v["nonfacial"] == 0 and v["facial"] > 0
        else:
            assert v["facial"] == 0


def test_local_types_and_equisingularity():
    assert D.tjurina(1, 1) == 0 and D.tjurina(1, 2) == 0           # smooth boundary
    assert D.tjurina(2, 2) == 1                                     # the conifold node
    assert D.tjurina(2, 3) is None and D.singular_locus_dim(2, 3) == 1
    assert D.tjurina(2, 2, c=1) == 1 and D.tjurina(2, 2, c=-2) == 1  # theta-level sets: same singularity


def test_flop_ambiguity_only_near_a_node():
    rng = np.random.default_rng(0); ev, od = D.parity_classes(3)
    base = rng.dirichlet(np.ones(8) * 5)
    smooth = base.copy(); smooth[ev[0]] = smooth[od[0]] = 0.004; smooth /= smooth.sum()
    node = base.copy(); node[[ev[0], ev[1], od[0], od[1]]] = [0.004, 0.0041, 0.004, 0.00401]; node /= node.sum()
    assert D.flop_switch_rate(smooth, 0.05, rng=rng) == 0.0
    assert D.flop_switch_rate(node, 0.05, rng=rng) > 0.2
    r = D.radii(node); assert r["node_radius"] > r["toggle_radius"] and r["partner_margin"] < 0.01
