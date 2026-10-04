"""Context attribution: the likelihood projection onto the simplex of declared sources, its bands, the unknown
certificate and the switching test, on small synthetic categorical sources where the truth is known."""
import numpy as np
from amb_vigneaux import attribution as at

rng0 = np.random.default_rng(0)
K = 30
A = rng0.dirichlet(np.ones(K) * 0.3); B = rng0.dirichlet(np.ones(K) * 0.3); U = np.ones(K) / K


def sample_iid(w, n, rng):
    x = w[0] * A + w[1] * B + w[2] * U
    return rng.choice(K, size=n, p=x / x.sum())


def cols(toks):
    return A[toks], B[toks], U[toks]


def test_em_is_the_simplex_mle_and_increases_likelihood():
    rng = np.random.default_rng(1); t = sample_iid([0.3, 0.6, 0.1], 800, rng); P = np.c_[cols(t)]
    w, ll = at.em_weights(P)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    for _ in range(200):                       # no simplex point does better (concavity: the fixed point is global)
        v = rng.dirichlet(np.ones(3)); assert np.log(P @ v).sum() <= ll + 1e-7
    _, l0 = at.em_weights(P, iters=1)
    assert ll >= l0


def test_blend_weight_recovered_with_band_coverage():
    hits, n_rep = 0, 25
    for s in range(n_rep):
        rng = np.random.default_rng(100 + s); t = sample_iid([0.35, 0.65, 0.0], 400, rng)
        r = at.attribute(*cols(t))
        lo, hi = r["theta_band"]; hits += lo <= 0.35 <= hi
        assert r["decision"] in ("blend", "A", "B")
    assert hits >= 0.8 * n_rep


def test_pure_contexts_and_unknown():
    rng = np.random.default_rng(7)
    assert at.attribute(*cols(sample_iid([1, 0, 0], 400, rng)))["decision"] == "A"
    assert at.attribute(*cols(sample_iid([0, 1, 0], 400, rng)))["decision"] == "B"
    r = at.attribute(*cols(sample_iid([0.25, 0.25, 0.5], 400, rng)), mu_budget=0.1)
    assert r["decision"] == "unknown" and r["mu_lower"] > 0.3


def test_switching_beats_blend_only_when_there_is_memory():
    rng = np.random.default_rng(3); toks, z = [], 0
    for t in range(400):                        # long runs of one source
        if rng.random() < 0.03: z = 1 - z
        src = A if z == 0 else B; toks.append(rng.choice(K, p=src))
    sw = at.switching_stat(A[toks], B[toks])
    bl = at.switching_stat(*cols(sample_iid([0.5, 0.5, 0], 400, rng))[:2])
    assert sw > 20 and bl < 10


def test_affinely_dependent_sources_are_not_identifiable():
    """if U lies in conv(A, B) the likelihood is flat along a segment: the band for mu cannot exclude 0 (Prop. ident)"""
    Ud = 0.5 * A + 0.5 * B
    rng = np.random.default_rng(5); x = 0.25 * A + 0.25 * B + 0.5 * Ud
    t = rng.choice(K, size=400, p=x / x.sum()); P = np.c_[A[t], B[t], Ud[t]]
    _, ll = at.em_weights(P)
    _, l0 = at.em_weights(P, fixed={2: 0.0}); _, l1 = at.em_weights(P, fixed={2: 0.5})
    assert abs(ll - l0) < 1e-6 and abs(ll - l1) < 1e-6


def test_absorbed_reference_proposition():
    """Prop. absorbed: mu_hat = f in (A, B, U) coordinates, max(0, (f - (1 - beta)) / beta) when U is folded into A', B'"""
    rng = np.random.default_rng(11); n = 1000
    for f, beta in ((0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.6, 0.8), (0.1, 0.8)):
        out = rng.random(n) < f
        a = np.where(out, 0.0, rng.uniform(0.01, 0.2, n)); b = np.where(out, 0.0, rng.uniform(0.01, 0.2, n))
        u = np.where(out, rng.uniform(0.001, 0.05, n), 0.0); fr = out.mean()
        w, _ = at.em_weights(np.c_[a, b, u], iters=20000, tol=1e-14)
        assert abs(w[2] - fr) < 1e-6
        w2, _ = at.em_weights(np.c_[beta * a + (1 - beta) * u, beta * b + (1 - beta) * u, u], iters=20000, tol=1e-14)
        assert abs(w2[2] - max(0.0, (fr - (1 - beta)) / beta)) < 2e-3
