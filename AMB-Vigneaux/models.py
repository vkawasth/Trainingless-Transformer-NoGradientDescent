"""Standard empirical models used to exercise and test the engine."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .scenario import EmpiricalModel, Scenario, bell_scenario

# eigenbases: columns are the eigenvectors for outcomes 0, 1
Z = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)          # 0 ↔ |+>, 1 ↔ |->
Y = np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2)        # 0 ↔ |+i>, 1 ↔ |-i>


def xz_basis(angle: float) -> np.ndarray:
    """Spin measurement at `angle` in the X–Z plane of the Bloch sphere."""
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def quantum_model(scenario: Scenario, state: np.ndarray, bases: Mapping[str, np.ndarray]) -> EmpiricalModel:
    """Born-rule model: context C = (m_1,…,m_n) measured on an n-qubit pure state,
    one measurement per tensor factor, in context order."""
    psi = np.asarray(state, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    tables = {}
    for C in scenario.contexts:
        B = bases[C[0]]
        for m in C[1:]:
            B = np.kron(B, bases[m])
        amp = B.conj().T @ psi
        tables[C] = np.abs(amp) ** 2
    return EmpiricalModel(scenario, tables)


def chsh_scenario() -> Scenario:
    return bell_scenario(2, 2)


def pr_box() -> EmpiricalModel:
    """Popescu–Rohrlich box: a ⊕ b = x·y — strongly contextual, CF = 1."""
    sc = chsh_scenario()
    def f(C, s):
        x, y = int(C[0][1]), int(C[1][1])
        return 0.5 if (s[0] ^ s[1]) == (x & y) else 0.0
    return EmpiricalModel.from_function(sc, f)


def tsirelson_box() -> EmpiricalModel:
    """Optimal quantum CHSH correlations (|Φ+>, angles 0, π/2 vs ±π/4)."""
    sc = chsh_scenario()
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    bases = {"a0": xz_basis(0), "a1": xz_basis(np.pi / 2),
             "b0": xz_basis(np.pi / 4), "b1": xz_basis(-np.pi / 4)}
    return quantum_model(sc, phi, bases)


def hardy_model() -> EmpiricalModel:
    """Hardy's paradox: ψ = (|01>+|10>+|11>)/√3, a0=b0=Z, a1=b1=X.
    The section (a1,b1) = (1,1) has probability 1/12 but no global extension:
    logically (not strongly) contextual."""
    sc = chsh_scenario()
    psi = np.array([0, 1, 1, 1]) / np.sqrt(3)
    return quantum_model(sc, psi, {"a0": Z, "b0": Z, "a1": X, "b1": X})


def ghz_model() -> EmpiricalModel:
    """GHZ state with X/Y measurements per party (a0=X, a1=Y): strongly contextual."""
    sc = bell_scenario(3, 2)
    ghz = np.zeros(8, complex)
    ghz[0] = ghz[7] = 1 / np.sqrt(2)
    bases = {f"{l}{i}": (X if i == 0 else Y) for l in "abc" for i in range(2)}
    return quantum_model(sc, ghz, bases)


def white_noise(scenario: Scenario) -> EmpiricalModel:
    return EmpiricalModel.from_function(scenario, lambda C, s: 1.0)


def random_noncontextual(scenario: Scenario, rng: Optional[np.random.Generator] = None,
                         concentration: float = 1.0) -> EmpiricalModel:
    rng = rng or np.random.default_rng(0)
    p = rng.dirichlet(np.full(scenario.n_sections(scenario.measurements), concentration))
    return EmpiricalModel.from_global(scenario, p)


def deterministic(scenario: Scenario, assignment: Mapping[str, object]) -> EmpiricalModel:
    X = scenario.measurements
    g = tuple(assignment[m] for m in X)
    p = np.array([1.0 if s == g else 0.0 for s in scenario.sections(X)])
    return EmpiricalModel.from_global(scenario, p)


def noisy(model: EmpiricalModel, visibility: float) -> EmpiricalModel:
    """v·e + (1-v)·white noise."""
    return white_noise(model.scenario).mix(model, visibility)


def parity_grid(row_residues=(0, 0, 0), col_residues=(0, 0, 1), p: int = 3) -> EmpiricalModel:
    """3x3 grid of Z/p variables x_ij; contexts are the three rows and three
    columns; context C is uniform over assignments with Σ_C x = r_C (mod p).
    Every variable lies in exactly two contexts, so the family is contradictory
    iff Σ rows ≠ Σ cols (mod p) (the handoff's T6 parity criterion)."""
    outs = {f"x{i}{j}": tuple(range(p)) for i in range(3) for j in range(3)}
    rows = [tuple(f"x{i}{j}" for j in range(3)) for i in range(3)]
    cols = [tuple(f"x{i}{j}" for i in range(3)) for j in range(3)]
    res = dict(zip(rows, row_residues)) | dict(zip(cols, col_residues))
    sc = Scenario(outs, tuple(rows + cols))
    return EmpiricalModel.from_function(sc, lambda C, s: 1.0 if sum(s) % p == res[C] % p else 0.0)


def single_source(model_or_law, scenario: Scenario, n: int, rng: np.random.Generator) -> EmpiricalModel:
    """T0 control: draw n GLOBAL outcomes from one law on E(X) and read every
    context off the same sample.  The empirical joint is then a global section,
    so γ = 0 and CF = 0 whatever the support looks like."""
    p = np.asarray(model_or_law, float)
    X = scenario.measurements
    counts = rng.multinomial(n, p / p.sum()).astype(float)
    return EmpiricalModel(scenario, {C: scenario.restriction_matrix(X, C) @ counts for C in scenario.contexts})
