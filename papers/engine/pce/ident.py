"""Identifiability and walls for two types on covers (companion "Two latent types on arbitrary covers", Section "Phases IV-V").
New module.

Coordinates: t = sqrt(w2/w1), r = t - 1/t, a1 = mu + v t, a2 = mu - v / t, so v = sqrt(w1 w2)(a1 - a2). A sign symmetry sigma of the
cover (an element of G_2, one sign per v_i and one for r) maps (v, r) to (sigma_v v, sigma_r r) and keeps every observed moment.

  params(mu, v, r)                (w, a) with a the 2 x m matrix of success probabilities
  to_mvr(w, a)                    (mu, v, r) from (w, a)
  sign_images(mu, v, r, G)        the explanations sigma . (v, r) for sigma in G_2, as (w, a), with a positivity flag
  window_tables(w, a, windows)    the observed tables
  hiding_dim(w, a, windows)       2m + 1 minus the Jacobian rank of (w, a) -> window tables
"""
from __future__ import annotations
import itertools
import numpy as np
from . import three as TH
from . import signs as SG


def params(mu, v, r):
    t = (r + np.sqrt(r * r + 4)) / 2; w1 = 1 / (1 + t * t)
    return np.array([w1, 1 - w1]), np.vstack([mu + v * t, mu - v / t])


def to_mvr(w, a):
    mu = w @ a; v = np.sqrt(w[0] * w[1]) * (a[0] - a[1]); t = np.sqrt(w[1] / w[0]); return mu, v, t - 1 / t


def window_tables(w, a, windows):
    P = TH.law(w, a); return [TH.marg(P, sorted(W)) for W in windows]


def sign_images(mu, v, r, m, faces):
    G = SG.sign_group(m, faces); out = []
    for g in SG.group_elements(G):
        sg = np.ones(m + 1)
        for k, c in enumerate(G["used"]):
            if g[k]: sg[c] = -1
        w, a = params(mu, sg[:m] * v, sg[m] * r)
        out.append(dict(sigma=sg, w=w, a=a, positive=bool(a.min() >= 0 and a.max() <= 1)))
    return out


def hiding_dim(w, a, windows, h=1e-6):
    m = a.shape[1]
    def f(th):
        ww = np.array([th[0], 1 - th[0]]); aa = th[1:].reshape(2, m); return np.concatenate([T.ravel() for T in window_tables(ww, aa, windows)])
    th = np.concatenate([[w[0]], a.ravel()])
    J = np.array([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(len(th))]).T
    s = np.linalg.svd(J, compute_uv=False); return len(th) - int((s > 1e-8 * s[0]).sum())


def same_law(w, a, w2, a2, tol=1e-10):
    P, Q = TH.law(w, a), TH.law(w2, a2); return np.abs(P - Q).max() < tol
