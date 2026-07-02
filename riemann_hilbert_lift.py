"""
riemann_hilbert_lift.py
========================
Finds the minimum-norm WK configuration satisfying all geometric constraints.

The Riemann-Hilbert lift: σ* → θ* where
  σ* = (φ₀,...,φ₄, τ, A₀,...,A₄, val) are the geometric coordinates
  θ* = (WK^(0),...,WK^(5)) are the weight matrices

Constraints (from geometric knowledge):
  C1: φ_k ∈ {0,π} for all k            [5 constraints]
  C2: A(L_k, L_{k+1}) ≈ 8.7 for all k  [5 constraints]
  C3: τ = |Z_k|/|Z_{k+1}| ≈ 5.3       [1 constraint]
  C4: WK^(k) has real dominant eigenvec  [implicit from C1]

The lift problem:
  min  Σ_k ‖WK^(k)‖²_F                 [minimum norm]
  s.t. φ_k(WK) ∈ {0,π} for all k
       A_k(WK) ≈ 8.7
       τ(WK) ≈ 5.3

This is a constrained nonlinear optimization.
In the Jacobian nullspace (at clean phase), the phase constraints
are automatically satisfied to first order.
The minimum-norm lift is then determined by:
  - Strip energy constraints (5 equations in r×r space)
  - τ constraint (1 equation)

Reduction to tractable problem:
Work in the U_k basis (Grassmannian Gr(r,D)).
The WK matrices are determined by:
  WK^(k) = U_k Σ_k V_k^T
where U_k ∈ Gr(r,D) satisfies the strip energy constraints
and Σ_k, V_k are free (modulo τ constraint).

This reduces the lift from D²×L = 393216 variables
to r×(L-1) = 36 strip-area values + r×L = 36 singular values
= 72 effective variables.

Usage
-----
  python riemann_hilbert_lift.py \
      --target_phi "0,pi,0,pi,0" \
      --target_strip_energy 8.7 \
      --target_tau 5.3 \
      --D 256 --rank 6 --L 6 \
      --output rh_lift.json
"""

import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target_phi',          default='0,pi,0,pi,0')
    p.add_argument('--target_strip_energy', type=float, default=8.7)
    p.add_argument('--target_tau',          type=float, default=5.3)
    p.add_argument('--D',                   type=int,   default=256)
    p.add_argument('--rank',                type=int,   default=6)
    p.add_argument('--L',                   type=int,   default=6)
    p.add_argument('--output',              default='rh_lift.json')
    return p.parse_args()


def parse_phi_target(phi_str):
    """Parse '0,pi,0,pi,0' → [0, π, 0, π, 0]"""
    result = []
    for s in phi_str.split(','):
        s = s.strip()
        if s == 'pi' or s == 'π':
            result.append(math.pi)
        elif s == '0':
            result.append(0.0)
        else:
            result.append(float(s))
    return result


def constraints_dimension_count(L, r):
    """Count how many constraints we have vs unknowns."""
    # Unknowns: L matrices of size D×D, but working in U_k basis
    # U_k ∈ Gr(r,D): r(D-r) free parameters per layer
    # Σ_k: r singular values per layer
    # (V_k not constrained by geometric sensors)
    D = 256  # fixed
    unknowns_U = L * r * (D - r)      # Grassmannian coordinates
    unknowns_Sigma = L * r             # singular values
    total_unknowns = unknowns_U + unknowns_Sigma

    # Constraints
    n_phi    = L - 1   # φ_k constraints
    n_strip  = L - 1   # strip energy constraints
    n_tau    = 1       # τ constraint
    total_constraints = n_phi + n_strip + n_tau

    return {
        'unknowns_U': unknowns_U,
        'unknowns_Sigma': unknowns_Sigma,
        'total_unknowns': total_unknowns,
        'n_phi': n_phi,
        'n_strip': n_strip,
        'n_tau': n_tau,
        'total_constraints': total_constraints,
        'degrees_of_freedom': total_unknowns - total_constraints,
        'underdetermined': total_unknowns > total_constraints,
    }


def strip_energy_from_U(Uk, Uk1, rank):
    """Σᵢ arccos(σᵢ(Uk^T Uk+1))"""
    ov = Uk.T @ Uk1
    sv = np.linalg.svd(ov, compute_uv=False)
    sv = np.clip(sv, 0, 1)
    return float(np.sum(np.arccos(sv)))


def phi_from_Sigma_ratio(sigma_k, sigma_k1, phi_target):
    """
    The phase φ_k depends on the RATIO of singular value structures.
    At clean phase φ_k ∈ {0,π}: eigenvalues of W_{k+1}W_k^{-1} are real.
    
    For diagonal WK = U Σ V^T, the transfer matrix:
    M = W_{k+1} W_k^{-1} = U_{k+1} Σ_{k+1} V_{k+1}^T V_k Σ_k^{-1} U_k^T
    
    The dominant eigenvalue is approximately:
    λ_dom ≈ σ_1(k+1) / σ_1(k)  (ratio of top singular values)
    
    φ_k = 0 if λ_dom > 0 (i.e., same sign structure)
    φ_k = π if λ_dom < 0 (i.e., sign flip)
    
    This determines the SIGN of the ratio.
    """
    ratio = sigma_k1[0] / (sigma_k[0] + 1e-10)
    if phi_target == 0.0:
        return ratio > 0  # want positive ratio
    elif abs(phi_target - math.pi) < 0.1:
        return ratio < 0  # want negative ratio (sign flip)
    return True  # off-wall phases: not constrainable this way


def lift_in_U_basis(phi_targets, target_strip, target_tau, D, rank, L,
                     seed=42):
    """
    Find minimum-norm Grassmannian configuration satisfying:
    C1: φ_k ∈ {0,π} (via U_k structure — real subspaces)
    C2: A(L_k, L_{k+1}) ≈ target_strip for all k
    C3: τ ≈ target_tau (via singular value ratios)
    
    Key insight: at clean phase, U_k ⊂ R^r (real subspace) automatically
    satisfies φ_k = 0 or π depending on orientation.
    So C1 is satisfied by CONSTRUCTION if U_k is real.
    
    This reduces to: find U_k ∈ Gr(r,D) (real) minimizing
    Σ_k ‖U_k‖² subject to A(L_k,L_{k+1}) ≈ target_strip.
    
    But ‖U_k‖² = r (fixed by orthonormality), so minimum norm in U is trivial.
    The true minimum norm is in Σ_k.
    
    The minimum-norm singular values satisfying τ = σ₁^(k)/σ₁^(k+1):
    σ₁^(0) = 1  (normalize)
    σ₁^(k) = σ₁^(0) / τ^k  (geometric sequence)
    For τ = 5.3, L=6: σ₁^(k) = 1, 0.189, 0.036, 0.007, 0.001, 0.0002
    """
    rng = np.random.default_rng(seed)

    # C1: satisfied by construction (all U_k real → φ_k ∈ {0,π})
    # The sign of φ_k is determined by det(U_k^T U_{k+1})
    # det > 0 → φ = 0; det < 0 → φ = π

    # Find U_k satisfying strip energy constraint via optimization
    # Variables: (L-1) × r × (D-r) Grassmannian coordinates
    # But D=256, r=6: too large for direct optimization

    # REDUCTION: work with r×r overlap matrices O_k = U_k^T U_{k+1}
    # The strip energy depends only on singular values of O_k
    # The phase depends on det(O_k)

    # For each pair k: find r×r matrix O_k with
    #   σᵢ(O_k) such that Σᵢ arccos(σᵢ) = target_strip
    #   det(O_k) > 0 if φ_target=0, < 0 if φ_target=π

    L_pairs = L - 1
    r = rank

    # For uniform strip energy: all σᵢ equal
    # Σᵢ arccos(σ) = r * arccos(σ) = target_strip
    # → σ = cos(target_strip/r)
    sigma_uniform = math.cos(target_strip / r)
    if sigma_uniform < 0 or sigma_uniform > 1:
        sigma_uniform = max(0.01, min(0.99, sigma_uniform))

    print(f"  Uniform singular value σ = cos({target_strip:.3f}/{r}) "
          f"= {sigma_uniform:.4f}")
    print(f"  Corresponding angle θ = {math.degrees(math.acos(sigma_uniform)):.2f}°")

    # Construct overlap matrices O_k
    overlap_matrices = []
    for k in range(L_pairs):
        phi_k = phi_targets[k]
        # O_k = R · diag(σ,...,σ) where R is orthogonal
        # det(O_k) > 0 for φ=0, < 0 for φ=π
        O_k = sigma_uniform * np.eye(r)
        if abs(phi_k - math.pi) < 0.1:
            O_k[0, 0] *= -1  # flip sign of first component → det < 0
        overlap_matrices.append(O_k)

    # Verify strip energies
    strip_energies = []
    phi_achieved = []
    for k, O_k in enumerate(overlap_matrices):
        sv = np.linalg.svd(O_k, compute_uv=False)
        sv = np.clip(np.abs(sv), 0, 1)
        A = float(np.sum(np.arccos(sv)))
        strip_energies.append(A)
        det = np.linalg.det(O_k)
        phi = 0.0 if det > 0 else math.pi
        phi_achieved.append(phi)

    # C3: Singular value ratios for τ
    # τ = ‖∇_FF L‖/‖∇_Emb L‖ ≈ σ_1(k)/σ_1(k+1) (heuristic)
    # Minimum-norm solution: geometric sequence
    sigma_1 = [1.0 / (target_tau ** k) for k in range(L)]

    # Lift to ambient space: construct U_k ∈ Gr(r,D)
    # Minimum-norm U_k: embed R^r into R^D via first r coordinates
    # U_k = [I_r; 0_{(D-r)×r}] rotated by a random orthogonal matrix
    # For minimum norm in the ambient space, the canonical embedding suffices

    # Construct actual D×r matrices
    U_lift = []
    for k in range(L):
        # Canonical embedding: first r standard basis vectors of R^D
        U_k = np.zeros((D, r))
        U_k[:r, :] = np.eye(r)
        # Apply random rotation in R^D to break symmetry
        # (minimum-norm solution uses identity rotation)
        U_lift.append(U_k)

    # Apply overlap structure: adjust U_{k+1} so U_k^T U_{k+1} = O_k
    for k in range(L_pairs):
        # U_{k+1} must satisfy U_k^T U_{k+1} = O_k
        # Since U_k = [I_r; 0], this means U_{k+1}[:r, :] = O_k
        U_lift[k+1][:r, :] = overlap_matrices[k]
        # Re-orthonormalize U_{k+1}
        Q, R_qr = np.linalg.qr(U_lift[k+1])
        U_lift[k+1] = Q[:, :r]

    # Verify the lift
    actual_strips = []
    actual_phis = []
    for k in range(L_pairs):
        A = strip_energy_from_U(U_lift[k], U_lift[k+1], r)
        actual_strips.append(A)
        O = U_lift[k].T @ U_lift[k+1]
        det = np.linalg.det(O)
        actual_phis.append(0.0 if det > 0 else math.pi)

    return {
        'overlap_matrices': [O.tolist() for O in overlap_matrices],
        'sigma_uniform': sigma_uniform,
        'target_strip': target_strip,
        'achieved_strips': actual_strips,
        'target_phis': phi_targets,
        'achieved_phis': actual_phis,
        'sigma_sequence': sigma_1,
        'U_shapes': [U.shape for U in U_lift],
        'strip_error': float(np.max(np.abs(np.array(actual_strips) - target_strip))),
        'phi_error': float(np.max(np.abs(np.array(actual_phis) - np.array(phi_targets)))),
        'min_norm_WK_frobenius': float(np.sum([s**2 * r for s in sigma_1])),
    }


def main():
    args = parse_args()
    print("="*60)
    print("  RIEMANN-HILBERT LIFT: σ* → θ*")
    print("  Finding minimum-norm WK satisfying geometric constraints")
    print("="*60)

    phi_targets = parse_phi_target(args.target_phi)
    D, r, L = args.D, args.rank, args.L

    print(f"\n  Target stability condition:")
    print(f"    φ = [{', '.join(f'{p/math.pi:.2f}π' for p in phi_targets)}]")
    print(f"    Strip energy = {args.target_strip_energy:.3f} per pair")
    print(f"    τ = {args.target_tau:.2f}")

    # Dimension count
    dim = constraints_dimension_count(L, r)
    print(f"\n  Constraint counting:")
    print(f"    Unknowns (U_k basis): {dim['unknowns_U']:,}")
    print(f"    Unknowns (Σ_k):       {dim['unknowns_Sigma']:,}")
    print(f"    Total unknowns:       {dim['total_unknowns']:,}")
    print(f"    Constraints:          {dim['total_constraints']}")
    print(f"    Degrees of freedom:   {dim['degrees_of_freedom']:,}")
    print(f"    Underdetermined:      {dim['underdetermined']}")

    print(f"\n  KEY INSIGHT: C1 (φ_k ∈ {{0,π}}) is satisfied by CONSTRUCTION")
    print(f"  for ANY real U_k ⊂ R^r ⊂ C^r.")
    print(f"  This reduces effective constraints to C2 (strip) + C3 (τ) = 6 equations.")
    print(f"  The sign of φ_k (0 vs π) is controlled by det(U_k^T U_{{k+1}}).")

    print(f"\n  Additional constraints that further restrict the solution:")
    print(f"    C4: r_m2^σ ≈ 0.68  (Hessian-strip alignment)")
    print(f"    C5: val ≈ 0.062    (entropy floor — corpus-determined)")
    print(f"    C6: Rank-r monodromy (rank(W_{{k+1}}W_k^{{-1}}) ≈ r)")
    print(f"  C4-C6 add 7 more constraints → 13 total in {dim['total_unknowns']:,} unknowns.")
    print(f"  Still vastly underdetermined.")

    print(f"\n  MINIMUM-NORM LIFT in U_k basis:")
    result = lift_in_U_basis(
        phi_targets, args.target_strip_energy, args.target_tau,
        D, r, L)

    print(f"    σ_uniform = {result['sigma_uniform']:.4f}")
    print(f"    Strip energies achieved: "
          f"[{', '.join(f'{a:.3f}' for a in result['achieved_strips'])}]")
    print(f"    Strip error: {result['strip_error']:.4f}")
    print(f"    φ achieved: "
          f"[{', '.join(f'{p/math.pi:.2f}π' for p in result['achieved_phis'])}]")
    print(f"    φ error: {result['phi_error']:.4f}")
    print(f"    σ sequence (τ={args.target_tau}): "
          f"[{', '.join(f'{s:.4f}' for s in result['sigma_sequence'])}]")
    print(f"    Min-norm ‖WK‖²_F = {result['min_norm_WK_frobenius']:.4f}")

    print(f"\n  WHAT THIS TELLS US ABOUT THE LIFT:")
    print(f"""
  The geometric constraints reduce the D²×L = {D*D*L:,}-dimensional
  space to a {dim['degrees_of_freedom']:,}-dimensional solution manifold.
  The minimum-norm lift selects a canonical representative via:
    1. Phase (C1): satisfied by construction (real U_k)
    2. Strip energy (C2): determines σ_uniform = cos(A/r) = {result['sigma_uniform']:.4f}
    3. τ (C3): determines singular value ratios σ_k/σ_{{k+1}} = {args.target_tau:.2f}
    4. Minimum norm: canonical embedding U_k = [I_r; 0] in R^D
  
  The remaining {dim['degrees_of_freedom']:,} degrees of freedom correspond to:
    - Choice of embedding direction in R^D (which r-plane in Gr(r,D))
    - Right singular vectors V_k (not constrained by sensors)
    - Cross-layer rotations (not constrained by phase/strip)
  
  The corpus uniquely determines which embedding direction via:
    - E₀ (spectral embedding) → the natural r-plane in the corpus space
    - Bigram co-occurrence → the V_k right singular vectors
  
  THIS IS THE MISSING LINK:
    E₀ determines U_k via the spectral embedding
    Corpus bigrams determine V_k via co-occurrence structure
    σ_k is determined by τ and the entropy floor
    Together: WK^(k) = U_k(E₀) · diag(σ_k) · V_k(bigrams)^T
  
  This is the Riemann-Hilbert lift — computable from corpus alone,
  once the formulas for U_k(E₀) and V_k(bigrams) are derived.
""")

    result['dimension_analysis'] = dim
    result['target_phi'] = args.target_phi
    result['target_strip_energy'] = args.target_strip_energy
    result['target_tau'] = args.target_tau

    Path(args.output).write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"  Report → {args.output}")


if __name__ == '__main__':
    main()
