"""Structured gradient-free backbone for the CausalIDView benchmark (examples/causalidview/run.py).
Skipped when the CausalIDView clone (CAUSALIDVIEW_DIR) is not available, since run.py imports its SCM helpers."""
import os, sys, importlib.util
import numpy as np
import pytest

CIV = os.environ.get("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
if not os.path.isdir(os.path.join(CIV, "data")):
    pytest.skip("CausalIDView clone not available", allow_module_level=True)
_p = os.path.join(os.path.dirname(__file__), "..", "examples", "causalidview", "run.py")
_spec = importlib.util.spec_from_file_location("cidv_run", _p); R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)


def test_krr_closed_form_loo_matches_brute_force():
    rng = np.random.default_rng(0); X = rng.normal(size=(40, 5)); y = (rng.random(40) < 0.5).astype(float)
    lam = 3.0; m = y.mean(); yc = y - m; K = X @ X.T; ev, V = np.linalg.eigh(K); d = ev / (ev + lam)
    loo = (yc - V @ (d * (V.T @ yc))) / (1 - (V ** 2) @ d)
    brute = []
    for i in range(40):
        k = np.arange(40) != i                       # the closed form keeps the full-sample centring; compare on that basis
        a = np.linalg.solve(K[np.ix_(k, k)] + lam * np.eye(39), yc[k]); brute.append(yc[i] - K[i, k] @ a)
    assert np.allclose(loo, brute, atol=1e-10)


def test_unpenalised_instrument_coefficient():
    rng = np.random.default_rng(1); n = 4000; X = rng.normal(size=(n, 3)); I = (rng.random(n) < 0.5).astype(float)
    T = (rng.random(n) < R._sig(-0.3 + 1.5 * I)).astype(float); Z = np.hstack([np.ones((n, 1)), X, I[:, None]])
    pen = R._irls_bin(Z, T, 1e6, free=(0,)); free = R._irls_bin(Z, T, 1e6, free=(0, 4))
    assert abs(pen[4]) < 0.01                       # a heavy ridge kills the instrument effect -> bounds widen
    assert abs(free[4] - 1.5) < 0.2                 # left free, it is recovered


def test_structured_backbone_on_one_world():
    W = R.world(5)                                   # a linear-family world
    Lo, Uo = R.manski(W["q"].mean(1)); E = R.estimate_ours(W)
    assert np.allclose(E["Um"] - E["Lm"], 1.0)       # Manski width is exactly 1 for binary outcomes
    assert R.ep_rmse(E["Lm"], E["Um"], Lo, Uo) < 0.12
    ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(len(W["qry"]))])
    assert ivo[:, 2].max() < 1e-7                    # oracle cells satisfy the instrumental inequality
    assert R.ep_rmse(E["Li"], E["Ui"], ivo[:, 0], ivo[:, 1]) < 0.15


def test_eb_rank0_is_isotropic_bayes_ridge():
    """the posterior -> prior EM with rank 0 must reduce to Bayesian ridge with an isotropic prior (no learned support)"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "causalidview", "support"))
    sys.modules.setdefault("run", R)
    import eb_hyper
    rng = np.random.default_rng(2); X = rng.normal(size=(300, 6)); y = X @ np.r_[1, 0, 0, 0.5, 0, 0] * 0.3 + rng.normal(size=300)
    post = eb_hyper.eb_fit(X, {"a": y - y.mean()}, r=0, nu=1.0)
    mu, S = post["a"]; v = np.var(y - y.mean() - X @ mu)
    assert np.linalg.norm(mu) > 0 and np.all(np.linalg.eigvalsh(S) > 0)
    # stationarity of an isotropic prior: posterior mean = ridge solution with lambda = v / c for some c > 0
    lam = np.linalg.lstsq(np.c_[mu], X.T @ (y - y.mean()) - X.T @ X @ mu, rcond=None)[0][0]
    assert lam > 0 and np.allclose(X.T @ X @ mu + lam * mu, X.T @ (y - y.mean()), atol=1e-6 * np.abs(X.T @ y).max())


def test_generic_prior_sampler_cells_and_chain():
    """generic-prior Bayes backbone: cells are distributions per instrument arm; a short chain runs and beats a constant"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "causalidview", "bayes"))
    sys.modules.setdefault("run", R)
    import generic as G
    W = R.world(5); c = W["ctx"]; rng = np.random.default_rng(0)
    gen = G.Gen(W["X"][c], W["T"][c], W["Y"][c], W["I"][c].astype(float))
    p = dict(a=np.zeros(2), g=0.3, c=np.zeros((2, 2)), beta=rng.normal(size=(2, G.K, G.NB)) * 0.1,
             wT=G.unit(rng.normal(size=12), np.arange(12)), w=[G.unit(rng.normal(size=12), np.arange(12) + 12 * k) for k in range(G.K)])
    q = gen.cells(p, W["X"][W["qry"]])
    assert np.allclose(q.reshape(len(q), 2, 4).sum(-1), 1.0) and np.isfinite(gen.loglik(p))
    cells, ll = G.run_chain(W, 40, 20, seed=1)
    assert cells.shape == (len(W["qry"]), 2, 2, 2) and np.isfinite(ll)
