"""The closed realization loop.

              [1. INFORMATION]   θ ∈ M, Fisher metric g(θ)
          forward │      ▲ backward: Δθ = −η g⁺ ∇ D_KL(P̂_N ‖ P_θ)
                  ▼      │
              [2. PROBABILITY]   P_θ = {q_C(θ)}, conditioning action, I = δ_t H
         sampling │      ▲ empirical pullback: P̂_N = (1/N) Σ δ_{x_i}
                  ▼      │
              [3. OUTCOMES]      realised support sections, γ ∈ Ȟ¹, CF

Phase 1 (forward drive):   optional information shift θ ← θ + drive(t, θ);
                           push θ to P_θ; read the information layer;
                           realise N outcomes per context from the *world*
                           (by default the model itself); analyse the support.
Phase 2 (backward pullback): form the empirical measure and take a
                           natural-gradient step on the Fisher manifold.

Site separation: the Z-valued cochains of the outcome layer are only ever
*computed from* supports; nothing maps them back algebraically.  The return
path is purely statistical (empirical measure + information geometry).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .geometry import StatisticalFamily, model_fisher_rao, model_kl
from .information import information_profile
from .outcome import OutcomeReport, analyse_outcomes
from .scenario import EmpiricalModel, GlobalLaw

World = Union[None, EmpiricalModel, GlobalLaw, Callable[[int, EmpiricalModel], Union[EmpiricalModel, GlobalLaw]]]
Drive = Optional[Callable[[int, np.ndarray], np.ndarray]]


@dataclass
class ForwardState:
    t: int
    theta: np.ndarray
    model: EmpiricalModel                 # P_θ              (probability layer)
    info: Dict                            # entropy / MI profile
    info_volume: float                    # log pdet g(θ)    (information layer)
    source: EmpiricalModel                # law actually sampled (as context marginals)
    single_source: bool                   # True: one global sample read by every context
    counts: Dict                          # realised outcomes (outcome layer, raw)
    empirical: EmpiricalModel             # P̂_N
    latent: OutcomeReport                 # obstructions of supp_ε(P_θ)
    realised: OutcomeReport               # obstructions of observed support


@dataclass
class StepRecord:
    t: int
    kl_empirical: float          # D_KL(P̂_N ‖ P_θ) before the update
    kl_world: float              # D_KL(world ‖ P_θ) (nan when self-realising)
    step_length: float           # Fisher–Rao distance moved by the update
    cf_model: float              # contextual fraction of P_θ
    cf_empirical: float          # contextual fraction of P̂_N
    level_latent: str
    level_realised: str
    gamma_latent: bool           # γ ≠ 0 on supp_ε(P_θ)
    gamma_realised: bool         # γ ≠ 0 on observed support
    n_gamma_realised: int
    mean_mutual_info: float
    max_cocycle_defect: float
    info_volume: float
    signalling: float


@dataclass
class ClosedLoopEngine:
    family: StatisticalFamily
    theta: np.ndarray
    eta: float = 0.5
    n_samples: int = 500
    world: World = None
    drive: Drive = None
    latent_eps: float = 1e-3          # support threshold on P_θ
    realised_min_count: int = 1       # a section is 'realised' once seen this often
    pseudocount: float = 0.0          # Laplace smoothing of P̂_N
    line_search: bool = True
    seed: int = 0
    history: List[StepRecord] = field(default_factory=list)
    t: int = 0

    def __post_init__(self):
        self.theta = np.asarray(self.theta, float).copy()
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ world
    def _world_model(self, P: EmpiricalModel) -> EmpiricalModel:
        if self.world is None:
            return P
        if isinstance(self.world, (EmpiricalModel, GlobalLaw)):
            return self.world
        return self.world(self.t, P)

    # ---------------------------------------------------------------- phase 1
    def forward(self) -> ForwardState:
        if self.drive is not None:
            self.theta = self.theta + self.drive(self.t, self.theta)          # Δθ
        P = self.family.model(self.theta)                                    # ΔP
        info = information_profile(P)
        vol = self.family.information_volume(self.theta)
        world = self._world_model(P)
        # Realisation. An EmpiricalModel world is sampled context by context
        # (independent experiments, multi-source); a GlobalLaw world is sampled
        # once and every context reads the same draws (single-source, T0).
        counts = world.sample(self.n_samples, self.rng)
        single = isinstance(world, GlobalLaw)
        src = world.model() if single else world
        emp = EmpiricalModel.from_counts(P.scenario, counts, self.pseudocount)
        sc = P.scenario
        observed = {C: [s for s, k in zip(sc.sections(C), counts[C]) if k >= self.realised_min_count]
                    for C in sc.contexts}
        latent = analyse_outcomes(P, eps=self.latent_eps)                    # Δsupp
        realised = analyse_outcomes(emp, support=observed)
        return ForwardState(self.t, self.theta.copy(), P, info, vol, src, single, counts, emp, latent, realised)

    # ---------------------------------------------------------------- phase 2
    def backward(self, fs: ForwardState) -> np.ndarray:
        fam, target = self.family, fs.empirical
        direction = fam.natural_gradient(self.theta, target)
        eta, L0 = self.eta, fam.loss(self.theta, target)
        new = self.theta - eta * direction
        if self.line_search:
            while fam.loss(new, target) > L0 + 1e-12 and eta > 1e-6:
                eta *= 0.5
                new = self.theta - eta * direction
        return new

    # ------------------------------------------------------------------- loop
    def step(self) -> StepRecord:
        fs = self.forward()
        new_theta = self.backward(fs)
        moved = model_fisher_rao(fs.model, self.family.model(new_theta))
        kl_w = np.nan if self.world is None else model_kl(fs.source, fs.model)
        rec = StepRecord(
            t=self.t,
            kl_empirical=model_kl(fs.empirical, fs.model),
            kl_world=kl_w,
            step_length=moved,
            cf_model=fs.latent.contextual_fraction,
            cf_empirical=fs.realised.contextual_fraction,
            level_latent=fs.latent.level(),
            level_realised=fs.realised.level(),
            gamma_latent=fs.latent.gamma_h1_nonzero,
            gamma_realised=fs.realised.gamma_h1_nonzero,
            n_gamma_realised=len(fs.realised.cohomological_witnesses),
            mean_mutual_info=float(np.mean([v["total_correlation"] for v in fs.info.values()])),
            max_cocycle_defect=float(max(v.get("cocycle_defect", 0.0) for v in fs.info.values())),
            info_volume=fs.info_volume,
            signalling=fs.latent.signalling_defect,
        )
        self.theta = new_theta
        self.history.append(rec)
        self.last_forward = fs
        self.t += 1
        return rec

    def run(self, steps: int, callback: Optional[Callable[[StepRecord], None]] = None) -> List[StepRecord]:
        for _ in range(steps):
            r = self.step()
            if callback:
                callback(r)
        return self.history

    def series(self, name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in self.history])

    @property
    def model(self) -> EmpiricalModel:
        return self.family.model(self.theta)
