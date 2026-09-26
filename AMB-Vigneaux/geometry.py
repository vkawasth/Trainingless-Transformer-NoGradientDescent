"""Layer 1 — the information layer: parameter manifolds with Fisher–Rao geometry.

A *statistical family* is a smooth map θ ↦ P_θ = {q_C(θ)}_C from a parameter
manifold M ⊆ R^d to empirical models.  The Fisher metric of the family is the
pullback of the product Fisher–Rao metric on the context simplices,

    g(θ) = Σ_C w_C · J_C(θ)^T diag(1/q_C) J_C(θ),

and the backward update of the closed loop is the natural-gradient step

    Δθ = −η · g(θ)^+ ∇_θ Σ_C w_C D_KL(P̂_C ‖ q_C(θ)).

Two families are provided:

* ContextualFamily   — independent (exponential-family) softmax per context;
  can represent any full-support model, contextual or not.
* HiddenVariableFamily — an exponential family on global assignments E(X),
  pushed forward to contexts; its image is exactly the non-contextual models
  (the 'local hidden-variable' polytope), so it can never fit a contextual
  target: the residual KL is an information-geometric shadow of CF > 0.
Both are dually flat on their natural parameters (Amari–Chentsov).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from .scenario import Context, EmpiricalModel, Scenario


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))).sum())


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Geodesic distance on the simplex under the Fisher metric: 2 arccos Σ√(pq)."""
    bc = float(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)).sum())
    return 2.0 * float(np.arccos(np.clip(bc, -1.0, 1.0)))


def model_kl(P: EmpiricalModel, Q: EmpiricalModel, weights: Optional[Mapping[Context, float]] = None) -> float:
    cs = P.scenario.contexts
    w = weights or {C: 1.0 / len(cs) for C in cs}
    return sum(w[C] * kl(P.tables[C], Q.tables[C]) for C in cs)


def model_fisher_rao(P: EmpiricalModel, Q: EmpiricalModel) -> float:
    """Product-manifold Fisher–Rao distance (root-sum-square over contexts)."""
    return float(np.sqrt(sum(fisher_rao_distance(P.tables[C], Q.tables[C]) ** 2
                             for C in P.scenario.contexts)))


class StatisticalFamily:
    """Abstract θ ↦ P_θ with Jacobians; subclasses implement probs/jacobian."""

    scenario: Scenario
    dim: int

    def __init__(self, scenario: Scenario, weights: Optional[Mapping[Context, float]] = None):
        self.scenario = scenario
        cs = scenario.contexts
        self.weights = dict(weights) if weights else {C: 1.0 / len(cs) for C in cs}

    # --- to implement --------------------------------------------------------
    def probs(self, theta: np.ndarray) -> Dict[Context, np.ndarray]:
        raise NotImplementedError

    def jacobian(self, theta: np.ndarray) -> Dict[Context, np.ndarray]:
        raise NotImplementedError

    def init_theta(self, rng: Optional[np.random.Generator] = None, scale: float = 0.0) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        return scale * rng.standard_normal(self.dim)

    # --- geometry ------------------------------------------------------------
    def model(self, theta: np.ndarray) -> EmpiricalModel:
        return EmpiricalModel(self.scenario, self.probs(theta))

    def loss(self, theta: np.ndarray, target: EmpiricalModel) -> float:
        q = self.probs(theta)
        return sum(self.weights[C] * kl(target.tables[C], q[C]) for C in self.scenario.contexts)

    def grad(self, theta: np.ndarray, target: EmpiricalModel) -> np.ndarray:
        q, J = self.probs(theta), self.jacobian(theta)
        g = np.zeros(self.dim)
        for C in self.scenario.contexts:
            g -= self.weights[C] * J[C].T @ (target.tables[C] / np.maximum(q[C], 1e-300))
        return g

    def fisher(self, theta: np.ndarray) -> np.ndarray:
        q, J = self.probs(theta), self.jacobian(theta)
        G = np.zeros((self.dim, self.dim))
        for C in self.scenario.contexts:
            Jc = J[C]
            G += self.weights[C] * Jc.T @ (Jc / np.maximum(q[C], 1e-300)[:, None])
        return G

    def natural_gradient(self, theta: np.ndarray, target: EmpiricalModel,
                         damping: float = 1e-8, rcond: float = 1e-10) -> np.ndarray:
        """g^+ ∇ D_KL — the contravariant update direction g^{ij} ∂_j D."""
        G = self.fisher(theta)
        G = G + damping * np.eye(self.dim)
        return np.linalg.pinv(G, rcond=rcond) @ self.grad(theta, target)

    def information_volume(self, theta: np.ndarray, tol: float = 1e-10) -> float:
        """log pseudo-determinant of the Fisher metric: how 'curved'/informative the
        current point is (→ −∞ as P_θ approaches the simplex boundary)."""
        ev = np.linalg.eigvalsh(self.fisher(theta))
        ev = ev[ev > tol]
        return float(np.sum(np.log(ev))) if ev.size else -np.inf


class ContextualFamily(StatisticalFamily):
    """θ = (θ_C)_C, q_C = softmax(θ_C): the full product of open simplices."""

    def __init__(self, scenario: Scenario, weights=None):
        super().__init__(scenario, weights)
        self.slices = {}
        k = 0
        for C in scenario.contexts:
            n = scenario.n_sections(C)
            self.slices[C] = slice(k, k + n)
            k += n
        self.dim = k

    def probs(self, theta):
        return {C: softmax(theta[s]) for C, s in self.slices.items()}

    def jacobian(self, theta):
        out = {}
        for C, s in self.slices.items():
            q = softmax(theta[s])
            J = np.zeros((q.size, self.dim))
            J[:, s] = np.diag(q) - np.outer(q, q)
            out[C] = J
        return out

    def theta_from_model(self, model: EmpiricalModel, floor: float = 1e-9) -> np.ndarray:
        """Natural parameters of (a floored version of) a given model."""
        th = np.zeros(self.dim)
        for C, s in self.slices.items():
            p = np.maximum(model.tables[C], floor)
            th[s] = np.log(p / p.sum())
        return th


class HiddenVariableFamily(StatisticalFamily):
    """Exponential family on global sections, p_λ(g) ∝ exp(Φ θ)(g), pushed forward:
    q_C = R_C p_λ.  With Φ = I this is every full-support hidden-variable model."""

    def __init__(self, scenario: Scenario, features: Optional[np.ndarray] = None, weights=None):
        super().__init__(scenario, weights)
        X = scenario.measurements
        nG = scenario.n_sections(X)
        self.Phi = np.eye(nG) if features is None else np.asarray(features, float)
        self.dim = self.Phi.shape[1]
        self.R = {C: scenario.restriction_matrix(X, C) for C in scenario.contexts}

    def global_law(self, theta):
        return softmax(self.Phi @ theta)

    def probs(self, theta):
        p = self.global_law(theta)
        return {C: R @ p for C, R in self.R.items()}

    def jacobian(self, theta):
        p = self.global_law(theta)
        dp = (np.diag(p) - np.outer(p, p)) @ self.Phi
        return {C: R @ dp for C, R in self.R.items()}
