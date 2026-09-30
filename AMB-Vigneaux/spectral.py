"""Slepian (discrete prolate spheroidal) tools for short bias time series.

Used as ESTIMATION tools, not as a theory of the phase change:
  slepian_step_test(y, tau, NW)  trend = span of 1, t and the first K = floor(2NW)-1 Slepian sequences of the whole window
                                 (the best band-limited trend basis for that length); a step at tau is tested
                                 against it by OLS with an AR(1)-corrected (Prais-Winsten) standard error.
  multitaper(y, NW)              Thomson multitaper spectrum with K = 2NW-1 tapers and a jackknife se of log S.
  spectral_change(y1, y2, NW)    per-band log ratio of the two multitaper spectra; under equal spectra each band
                                 ratio is ~ F(2K, 2K) (independent segments); returns the ratio and a band chi-square.
  shannon_number(n, NW)          2NW: the number of resolvable modes of an n-point window at half-bandwidth NW/n.
"""
from __future__ import annotations
import numpy as np
from scipy.signal.windows import dpss
from scipy import stats


def shannon_number(n, NW):
    return 2 * NW


def slepian_trend_basis(n, NW=2.0, K=None):
    K = K or max(1, int(2 * NW) - 1)
    return dpss(n, NW, Kmax=K).T  # n x K


def _ar1_whiten(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    rho = float(np.clip(np.corrcoef(e[:-1], e[1:])[0, 1], -0.95, 0.95))
    Xw = np.vstack([X[:1] * np.sqrt(1 - rho ** 2), X[1:] - rho * X[:-1]])
    yw = np.concatenate([y[:1] * np.sqrt(1 - rho ** 2), y[1:] - rho * y[:-1]])
    return Xw, yw, rho


def slepian_step_test(y, tau, NW=2.0, K=None):
    """step at index tau (1 from tau on) against a Slepian trend; returns step size, z, AR(1) rho"""
    y = np.asarray(y, float); n = len(y)
    B = slepian_trend_basis(n, NW, K)
    step = (np.arange(n) >= tau).astype(float)
    t = np.linspace(-1, 1, n)
    X = np.column_stack([np.ones(n), t, B, step])
    Xw, yw, rho = _ar1_whiten(X, y)
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ b; dof = n - X.shape[1]
    s2 = e @ e / dof; cov = s2 * np.linalg.pinv(Xw.T @ Xw)
    z = b[-1] / np.sqrt(cov[-1, -1])
    return dict(step=float(b[-1]), z=float(z), p=float(2 * stats.t.sf(abs(z), dof)), rho=rho, K=B.shape[1])


def multitaper(y, NW=3.0, K=None, detrend=True):
    y = np.asarray(y, float); n = len(y)
    if detrend:
        t = np.arange(n); y = y - np.polyval(np.polyfit(t, y, 1), t)
    K = K or int(2 * NW) - 1
    tapers = dpss(n, NW, Kmax=K)
    Y = np.fft.rfft(tapers * y[None, :], axis=1)
    Sk = np.abs(Y) ** 2
    S = Sk.mean(0)
    # jackknife se of log S over tapers
    loo = np.log((Sk.sum(0)[None, :] - Sk) / (K - 1))
    se = np.sqrt((K - 1) / K * ((loo - loo.mean(0)) ** 2).sum(0))
    return np.fft.rfftfreq(n), S, se, K


def spectral_change(y1, y2, NW=3.0, bands=((0, 0.05), (0.05, 0.15), (0.15, 0.5))):
    """compare multitaper spectra of two segments (equal length recommended) in frequency bands"""
    f1, S1, _, K = multitaper(y1, NW); f2, S2, _, _ = multitaper(y2, NW)
    out = []
    for lo, hi in bands:
        m1 = (f1 >= lo) & (f1 < hi); m2 = (f2 >= lo) & (f2 < hi)
        r = S2[m2].mean() / S1[m1].mean()
        # effective dof: 2K per independent band cell; cells per band ~ bandwidth / (2NW/n)
        c1 = max(1, (hi - lo) / (2 * NW / len(y1))); c2 = max(1, (hi - lo) / (2 * NW / len(y2)))
        d1, d2 = 2 * K * c1, 2 * K * c2
        p = 2 * min(stats.f.cdf(r, d2, d1), stats.f.sf(r, d2, d1))
        out.append(dict(band=(lo, hi), ratio=float(r), p=float(p)))
    return out


_NULL = {}


def _maxF(y, NW, K, pad):
    n = len(y); y = y - y.mean(); tp = dpss(n, NW, Kmax=K); U0 = tp.sum(1)
    Y = np.fft.rfft(tp * y[None, :], n=pad * n, axis=1)
    mu = (U0[:, None] * Y).sum(0) / (U0 ** 2).sum()
    F = (K - 1) * np.abs(mu) ** 2 * (U0 ** 2).sum() / (np.abs(Y - mu[None, :] * U0[:, None]) ** 2).sum(0)
    f = np.fft.rfftfreq(pad * n); inner = (f > NW / n) & (f < 0.5 - NW / n)
    return f, F, inner


def harmonic_ftest(y, NW=3.0, K=None, pad=4, nsim=400):
    """Thomson's harmonic F-test for a line (periodic) component, evaluated on a zero-padded grid.
    F(f) = (K-1) |mu|^2 sum_k U_k(0)^2 / sum_k |Y_k(f) - mu U_k(0)|^2 ~ F(2, 2K-2) pointwise under no line.
    p_adj = P(max_f F >= observed max) under Gaussian white noise of the same length (Monte Carlo, cached):
    the scan over frequencies is accounted for exactly."""
    y = np.asarray(y, float); n = len(y); K = K or int(2 * NW) - 1
    f, F, inner = _maxF(y, NW, K, pad)
    p = stats.f.sf(F, 2, 2 * K - 2)
    key = (n, NW, K, pad)
    if key not in _NULL:
        r = np.random.default_rng(12345)
        _NULL[key] = np.array([_maxF(r.normal(size=n), NW, K, pad)[1][inner].max() for _ in range(nsim)])
    i = int(np.argmax(np.where(inner, F, -1)))
    return dict(freqs=f, F=F, p=p, f_best=float(f[i]), p_adj=float((1 + (_NULL[key] >= F[i]).sum()) / (1 + nsim)))
