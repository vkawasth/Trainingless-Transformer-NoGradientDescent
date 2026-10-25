"""Selective prediction helpers: does a confidence score tell right answers from wrong ones?

  * auroc(score, y): area under the ROC curve for y = 1 (answer correct), ties counted half.
  * aurc(score, y): area under the risk-coverage curve (lower is better): answer the most confident items first.
  * cv_logistic(X, y, folds, seed): out-of-fold predicted probabilities of a logistic model (L2, C = 1), stratified folds.
  * logloss(p, y): per-item log loss in bits.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                                   # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def aurc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    order = np.argsort(-score, kind="mergesort"); err = 1 - y[order]
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    return float(risk.mean())


def cv_logistic(X, y, folds=5, seed=20261014):
    X, y = np.asarray(X, float), np.asarray(y, int)
    if X.ndim == 1: X = X[:, None]
    p = np.empty(len(y))
    if len(set(y)) < 2:
        p[:] = y.mean(); return p
    skf = StratifiedKFold(n_splits=min(folds, int(min(np.bincount(y)))), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, int)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))
