import numpy as np


def _lik(pi, P2, P1):
    # p(x1..x4) for a depth-2 binary grammar: root -> (a, b), a -> (x1, x2), b -> (x3, x4)
    return np.einsum("r,rab,axy,bzw->xyzw", pi, P2, P1, P1)


def test_root_merge_is_exactly_invisible_and_level1_merge_is_not():
    rng = np.random.default_rng(0); V, K = 3, 4
    pi = rng.dirichlet(np.ones(V)); P2 = rng.dirichlet(np.ones(V * V), size=V).reshape(V, V, V)
    P1 = rng.dirichlet(np.ones(K * K), size=V).reshape(V, K, K)
    p = _lik(pi, P2, P1)
    # merge root symbols 0 and 1 with usage weights
    w = pi[:2] / pi[:2].sum(); row = w[0] * P2[0] + w[1] * P2[1]
    pim = np.array([pi[0] + pi[1], pi[2]]); P2m = np.stack([row, P2[2]])
    assert np.allclose(_lik(pim, P2m, P1), p, atol=1e-15)
    # merge level-1 symbols 0 and 1 (children of the root collapse): the distribution changes
    u = (P2 * pi[:, None, None]).sum((0, 2)) + (P2 * pi[:, None, None]).sum((0, 1))
    w = u[:2] / u[:2].sum(); P1m = np.stack([w[0] * P1[0] + w[1] * P1[1], P1[2]])
    T = P2.copy(); T[:, 0, :] += T[:, 1, :]; T = np.delete(T, 1, 1); T[:, :, 0] += T[:, :, 1]; T = np.delete(T, 1, 2)
    q = _lik(pi, T, P1m)
    assert abs(q.sum() - 1) < 1e-12 and (p * np.log(p / q)).sum() > 1e-4


def test_lone_node_mixture_is_not_identifiable_but_sibling_window_is():
    # one node, one child pair: p(x,y) = sum_z pi_z th_z[x,y] -- merging two z with usage weights is exactly free
    rng = np.random.default_rng(1); K, c = 3, 4
    pi = rng.dirichlet(np.ones(K)); th = rng.dirichlet(np.ones(c * c), size=K).reshape(K, c, c)
    p1 = np.einsum("z,zxy->xy", pi, th)
    w = pi[:2] / pi[:2].sum(); th_m = np.stack([w[0] * th[0] + w[1] * th[1], th[2]]); pi_m = np.array([pi[0] + pi[1], pi[2]])
    assert np.allclose(np.einsum("z,zxy->xy", pi_m, th_m), p1, atol=1e-15)
    # two siblings under a parent: p(x1,y1,x2,y2) = sum_w pw Th[w,z,z'] th_z th_z' -- the same merge now changes it
    Kp = 2; pw = rng.dirichlet(np.ones(Kp)); Th = rng.dirichlet(np.ones(K * K), size=Kp).reshape(Kp, K, K)
    p2 = np.einsum("w,wab,axy,buv->xyuv", pw, Th, th, th)
    use = (Th * pw[:, None, None]).sum((0, 2)) + (Th * pw[:, None, None]).sum((0, 1))
    w = use[:2] / use[:2].sum(); th_m = np.stack([w[0] * th[0] + w[1] * th[1], th[2]])
    T = Th.copy(); T[:, 0, :] += T[:, 1, :]; T = np.delete(T, 1, 1); T[:, :, 0] += T[:, :, 1]; T = np.delete(T, 1, 2)
    q2 = np.einsum("w,wab,axy,buv->xyuv", pw, T, th_m, th_m)
    assert abs(q2.sum() - 1) < 1e-12 and (p2 * np.log(p2 / q2)).sum() > 1e-4
