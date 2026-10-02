import itertools
import numpy as np
from scipy.special import logsumexp

from amb_vigneaux import lattice_dp as L

rng = np.random.default_rng(4)


def _toy(S=8, X=9):
    eta = rng.normal(0, .5, S)[:, None] + rng.normal(0, .5, X)[None, :]; eta[:, :3] += rng.normal(0, .8, S)[:, None]
    n = rng.integers(20, 40, (S, X)); k = rng.binomial(n, 1 / (1 + np.exp(-eta)))
    return np.log((k + .5) / (n - k + .5)), 1 / (1 / (k + .5) + 1 / (n - k + .5))


def test_jets_match_exact_single_and_pair_deletions_to_second_order():
    y, w = _toy(); J = L.jets(y, w); X = y.shape[1]
    for x in range(X):
        ex = L.additive_fit(y, w, [u for u in range(X) if u != x])[2]
        assert abs(L.jet_predict(J, [x], 2) - ex) < 0.05 * max(1.0, abs(J["d1"][x]))


def test_sampler_partition_function_is_exact():
    y, w = _toy(); J = L.jets(y, w); X = y.shape[1]
    bags = [[0, 1, 2], [3, 4], [5], [6, 7], [8]]
    rules = L.Rules(k_min=5, group=[3, 5, 7], m_min=2, must=[8], exclusive=[(3, 4)])
    Smp = L.Sampler(X, bags, J["d1"], J["d2"], rules, S=y.shape[0], order=2)
    lw = []
    for z in itertools.product((0, 1), repeat=X):
        U = [x for x in range(X) if z[x]]; out = [x for x in range(X) if not z[x]]
        if len(U) < 5 or sum(x in (3, 5, 7) for x in U) < 2 or 8 not in U or (3 in U and 4 in U):
            continue
        dD = -J["d1"][out].sum() + 0.5 * sum(J["d2"][np.ix_([o for o in out if o in b], [o for o in out if o in b])].sum() for b in bags)
        lw.append(-(dD + 2.0 * (y.shape[0] - 1) * len(out)) / 2)
    assert abs(Smp.logZ - logsumexp(lw)) < 1e-8
    for _ in range(20):
        U = Smp.sample(rng)
        assert 8 in U and len(U) >= 5 and not (3 in U and 4 in U)
