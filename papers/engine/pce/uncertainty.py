"""Three-level split of predictive uncertainty for hierarchical evidence (the tower's martingale, Theorem 5.3).

Evidence is a list of groups; each group is a list of (weight, posterior) members, where a posterior is a probability vector over the
outcomes. With group weights w_g, the chain rule along (group, member) gives, exactly,

    H(rho) = I(X; group) + I(X; member | group) + E H(member posterior)

where rho is the overall average. The first term is disagreement between groups, the second disagreement within groups, the third
irreducible noise of the members themselves. Plain averaging keeps only H(rho).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def H(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@dataclass
class Split:
    total: float          # H(rho), bits
    between: float        # I(X; group)
    within: float         # I(X; member | group)
    noise: float          # E H(member)
    rho: np.ndarray       # pooled predictive


def split(groups, group_weights=None) -> Split:
    G = len(groups)
    gw = np.full(G, 1 / G) if group_weights is None else np.asarray(group_weights, float) / np.sum(group_weights)
    g_means, g_inner, noise = [], [], 0.0
    for g, members in enumerate(groups):
        ws = np.array([w for w, _ in members], float); ws = ws / ws.sum()
        Ps = np.array([np.asarray(p, float) for _, p in members])
        mean = ws @ Ps; g_means.append(mean)
        hm = sum(w * H(p) for w, p in zip(ws, Ps))
        g_inner.append(H(mean) - hm); noise += gw[g] * hm
    g_means = np.array(g_means); rho = gw @ g_means
    between = H(rho) - sum(gw[g] * H(g_means[g]) for g in range(G))
    within = float(gw @ np.array(g_inner))
    return Split(H(rho), between, within, noise, rho)
