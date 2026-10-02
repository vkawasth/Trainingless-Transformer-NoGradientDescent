import numpy as np

from amb_vigneaux import infogeo2 as G, toric as T

rng = np.random.default_rng(3)
K = 16


def _jets(fn, s=0.0, h=1e-3):
    L = lambda t: np.log(fn(t))
    return fn(s), (L(s + h) - L(s - h)) / (2 * h), (L(s + h) - 2 * L(s) + L(s - h)) / h ** 2


def test_each_geodesic_is_flat_in_its_own_connection():
    eta0, c = rng.normal(0, .8, K), rng.normal(0, .6, K); q0, q1 = G.softmax(eta0), G.softmax(rng.normal(0, .8, K))
    sig = lambda s: 1 / (1 + np.exp(-s))
    e = G.curvatures(*_jets(lambda s: G.softmax(eta0 + c * s)))
    m = G.curvatures(*_jets(lambda s: (1 - sig(s)) * q0 + sig(s) * q1))
    f = G.curvatures(*_jets(lambda s: G.geodesic(q0, q1, "fr", sig(s))))
    assert e["gamma2_e"] < 1e-6 and e["gamma2_m"] > 0.05
    assert m["gamma2_m"] < 1e-5 and m["gamma2_e"] > 0.05
    assert f["kappa2_fr"] < 1e-5 and f["gamma2_e"] > 0.01


def test_fisher_rao_is_sphere_distance():
    p, q = G.softmax(rng.normal(size=K)), G.softmax(rng.normal(size=K))
    x, y = G.sphere(p), G.sphere(q)
    assert abs(np.linalg.norm(x) - 2) < 1e-12
    assert abs(G.fr_dist(p, q) - 2 * np.arccos(x @ y / 4)) < 1e-12
    mid = G.geodesic(p, q, "fr", 0.5)
    assert abs(G.fr_dist(p, mid) - G.fr_dist(p, q) / 2) < 1e-9


def test_local_fit_recovers_curvature():
    eta0, c, d = rng.normal(0, .8, K), rng.normal(0, .6, K), rng.normal(0, .4, K)
    fn = lambda s: G.softmax(eta0 + c * s + d * s * s); sg = np.linspace(-3, 3, 9)
    exact = G.curvatures(*_jets(fn))["gamma2_e"]
    tab = np.array([fn(s) * 1e9 for s in sg])
    p0, d1, d2 = G.local_jets(tab, sg, 4)
    assert abs(G.curvatures(p0, d1, d2)["gamma2_e"] - exact) < 1e-3 * max(1, exact)


def test_efron_hinkley_flat_family_has_no_wobble():
    eta0, c = rng.normal(0, .8, K), rng.normal(0, .6, K)
    assert G.efron_hinkley(eta0, c, 0 * c, 100, reps=200, rng=rng)["nvar"] < 1e-12


def test_plaquettes_vanish_on_flat_grid_and_see_a_twist():
    beta = rng.normal(0, .7, K); a = rng.normal(0, .3, (4, K)); b = rng.normal(0, .3, (4, K)); h = rng.normal(0, 1, K)
    flat = np.array([[G.softmax(beta + a[u] + b[v]) * 1e12 for v in range(4)] for u in range(4)])
    tw = np.array([[G.softmax(beta + a[u] + b[v] + 0.1 * u * v * h) * 1e12 for v in range(4)] for u in range(4)])
    assert abs(G.plaquettes(flat)["total_nats_e"]) < 1e-9
    assert G.plaquettes(tw)["total_nats_e"] > 0.01


def test_top_info_zero_without_top_interaction():
    X = np.array(T.cells(4)); p = G.softmax(X @ rng.normal(0, .3, 4) + 0.2 * X[:, 0] * X[:, 1])
    assert G.top_info(p * 1e12)["D_top"] < 1e-9
    assert abs(G.coinformation(np.full(16, 1 / 16))) < 1e-12


def test_node_posterior_branches():
    ev = [i for i in range(16) if T.chi(4)[i] > 0]; od = [i for i in range(16) if T.chi(4)[i] < 0]
    c = np.full(16, 2000.0); c[ev[0]] = 20; c[ev[1]] = 60; c[od[0]] = c[od[1]] = 25
    assert 0.35 < G.node_posterior(c, B=1000, rng=rng)["branch_top"] < 0.65
    c[od[1]] = 200
    assert G.node_posterior(c, B=1000, rng=rng)["branch_top"] > 0.95
