"""check_masked_gauge.py -- the linear ambiguity left by masked directions (note: level2/partial_gluing.pdf, Section 10).

  MK7 for two qubits, with Bob's directions spanning R^3, the linear fit leaves the hidden directions on Alice's side ambiguous exactly up
      to the linear maps fixing the span V of her known directions (acting contragrediently on s and T), cut by unit length: the kernel of
      the Jacobian at the truth has dimension max(0, 3 (3 - dim V) - h_A), checked for dim V = 3, 2, 1 and h_A = 1, 2, 3 (7 cases)
Run: python check_masked_gauge.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.masked import bell_data, place, _design, _yw, _ang, _vec, theta_from
rng = np.random.default_rng(11); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
def rand_dirs(n): v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi); rho = 0.9 * np.outer(psi, psi.conj()) + 0.1 * np.eye(4) / 4
def nullity(known_A, n_hidden):
    A = np.vstack([known_A, rand_dirs(n_hidden)]); B = rand_dirs(4); P = bell_data(rho, A, B, None, rng)
    hidden = [("A", len(known_A) + k) for k in range(n_hidden)]; y, w = _yw(P, None)
    def resid(z):
        dirs = [_vec(z[2 * k:2 * k + 2]) for k in range(n_hidden)]; A2, B2 = place(A, B, hidden, dirs)
        return y - _design(A2[None], B2[None])[0] @ z[2 * n_hidden:]
    z = np.r_[np.concatenate([_ang(A[len(known_A) + k]) for k in range(n_hidden)]), theta_from(rho)]
    J = np.zeros((len(y), len(z)))
    for k in range(len(z)):
        e = np.zeros(len(z)); e[k] = 1e-6; J[:, k] = (resid(z + e) - resid(z - e)) / 2e-6
    sv = np.linalg.svd(J, compute_uv=False); return int((sv < 1e-6 * sv[0]).sum())
plane = np.array([[1, 0, 0], [0, 0, 1.]]); line = np.array([[0, 0, 1.]])
cases = [("R^3", rand_dirs(3), 1), ("R^3", rand_dirs(3), 2), ("plane", plane, 1), ("plane", plane, 2), ("plane", plane, 3), ("line", line, 2), ("line", line, 3)]
got = [(nm, h, nullity(K, h), max(0, 3 * (3 - np.linalg.matrix_rank(K)) - h)) for nm, K, h in cases]
check("MK7 linear ambiguity of masked directions = max(0, 3 (3 - dim V) - h_A)", all(g == p for _, _, g, p in got),
      "; ".join(f"{nm}, h={h}: {g} (predicted {p})" for nm, h, g, p in got))
print(f"\n{sum(results)}/{len(results)} checks passed")
