"""Pooling and conditioning (companion note, "Pooling and conditioning"; pce/pooling.py is a different, older module). New module; it modifies nothing.

Distributions are arrays over a finite outcome space; a source i supplies a joint model P_i over (state, evidence), here a prior over the
binary state X / not-X and likelihoods of each evidence item. Weights w are non-negative and sum to 1.
  linear(Ps, w)                       sum_i w_i P_i
  geometric(Ps, w)                    prod_i P_i^{w_i}, normalised (logarithmic pool)
  condition(P, lik)                   P * lik, normalised (Bayes with likelihood vector lik over states)
  evidence_prob(P, lik)               P(E) = sum_x P(x) lik(x)
  reweighted(w, Ps, liks)             w_i P_i(E) / sum_j w_j P_j(E): the weights that make linear pooling commute with conditioning
  defect(Ps, w, liks, rule)           total variation between condition-after-pool and pool-after-condition (fixed weights)
Worked example ('stream'): the true state is not-X. Two honest sources and a third source K. The evidence stream has genuine items, drawn
under not-X, and planted items: records that K's model predicts well and that are fabricated. Three procedures:
  BMA       pool first, then condition on each item: the linear pool's weights move to w_i P_i(item) (Bayesian model averaging)
  fixed     condition each source on each item, then pool with fixed weights
  geometric condition each source, then pool geometrically (the same as pooling geometrically first, since the rule is externally
            Bayesian when every source conditions on the same likelihood)
"""
from __future__ import annotations
import numpy as np


def linear(Ps, w): return np.sum([wi * P for wi, P in zip(w, Ps)], axis=0)


def geometric(Ps, w):
    g = np.prod([P ** wi for wi, P in zip(w, Ps)], axis=0); return g / g.sum()


def condition(P, lik): q = P * lik; return q / q.sum()


def evidence_prob(P, lik): return float(np.sum(P * lik))


def reweighted(w, Ps, liks):
    a = np.array([wi * evidence_prob(P, l) for wi, P, l in zip(w, Ps, liks)]); return a / a.sum()


def tv(p, q): return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def defect(Ps, w, liks, rule="linear"):
    """sources may hold different likelihoods liks[i]; the pool's likelihood for conditioning-after-pooling is the pooled one: for the
    linear rule, the mixture's own predictive (each component with its own likelihood); for the geometric rule, prod lik_i^{w_i}"""
    if rule == "linear":
        joint = [P * l for P, l in zip(Ps, liks)]                        # P_i(x, E)
        after = linear(joint, w); after = after / after.sum()            # pool, then condition
        before = linear([condition(P, l) for P, l in zip(Ps, liks)], w)  # condition, then pool with the same weights
    else:
        after = condition(geometric(Ps, w), np.prod([l ** wi for wi, l in zip(w, liks)], axis=0))
        before = geometric([condition(P, l) for P, l in zip(Ps, liks)], w)
    return tv(after, before), after, before


def stream(n=40, plant_every=2, seed=0, planted=True):
    """the worked example: returns per step the weight of K under BMA and P(X) under the three procedures"""
    rng = np.random.default_rng(seed)
    priors = [np.array([0.3, 0.7]), np.array([0.25, 0.75]), np.array([0.6, 0.4])]     # P(X), P(not X) for H1, H2, K
    # genuine evidence: a binary signal with P(s = 1 | X) = 0.6, P(s = 1 | not X) = 0.4 for honest models; K's model agrees on genuine items
    gen_lik = {1: np.array([0.6, 0.4]), 0: np.array([0.4, 0.6])}
    # planted records: K's model says they occur with probability 0.9 under either state (it predicted them); honest models give them 0.3
    plant_lik = [np.array([0.3, 0.3]), np.array([0.3, 0.3]), np.array([0.9, 0.9])]
    plant_tilt = [np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.9, 0.45])]   # and K reads them as favouring X
    w = np.array([1 / 3, 1 / 3, 1 / 3]); Ps = [p.copy() for p in priors]; pool = linear(Ps, w); bma_w = w.copy()
    out = dict(step=[], wK=[], bma=[], fixed=[], geometric=[], planted=[])
    for t in range(n):
        is_plant = planted and (t % plant_every == plant_every - 1)
        if is_plant:
            liks = [pl * pt for pl, pt in zip(plant_lik, plant_tilt)]
        else:
            s = int(rng.random() < 0.4)                                  # truth is not-X
            liks = [gen_lik[s]] * 3
        bma_w = reweighted(bma_w, Ps, liks)                              # BMA: weights move with predictive success
        Ps = [condition(P, l) for P, l in zip(Ps, liks)]
        out["step"].append(t + 1); out["wK"].append(float(bma_w[2])); out["planted"].append(is_plant)
        out["bma"].append(float(linear(Ps, bma_w)[0])); out["fixed"].append(float(linear(Ps, w)[0])); out["geometric"].append(float(geometric(Ps, w)[0]))
    return out
