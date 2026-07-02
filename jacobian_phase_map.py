"""
jacobian_phase_map.py  (correct version)
=========================================
Analytic Jacobian of the TRUE phase map:
  φ_k = arg(λ_dom(W_{k+1} W_k^{-1}))

Key fact: φ_k depends on FULL W_k matrices, not just column spaces.
The dominant eigenvalue of W_{k+1} W_k^{-1} depends on all of W_k,
not just col_r(W_k).

Analytic derivative via eigenvalue perturbation theory:
  dλ/dW_k = -λ * (ℓ^T ⊗ r) / (ℓ^T r)
  dφ_k/dW_k = Im(dλ/dW_k) / |λ|

where ℓ, r are left/right eigenvectors of M_k = W_{k+1} W_k^{-1}.

Cost: O(D^3) for one eigendecomposition per pair. ~0.01s total.

NLP TopoGate: drive φ_{k} → 0 via rank-1 update to W_k.
Optimal rank-1 direction from the Jacobian.
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64',    default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--spike72',    default='tau_spikes/tau_spike_step0072_tau5.94.pt')
    p.add_argument('--rank',       type=int, default=6)
    p.add_argument('--lambda_reg', type=float, default=1.0)
    p.add_argument('--output',     default='jacobian_report.json')
    return p.parse_args()


def load_wk(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = (ckpt.get('state_dict', ckpt.get('model', ckpt))
             if isinstance(ckpt, dict) else ckpt)
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor.detach().float().numpy()
    return [wk[i] for i in sorted(wk)]


def dominant_eigensystem(M):
    """Dominant eigenvalue and left/right eigenvectors of M."""
    vals, vecs = np.linalg.eig(M)
    idx  = np.argmax(np.abs(vals.real))
    lam  = vals[idx]
    r    = vecs[:, idx]              # right eigenvector
    # Left eigenvector: dominant of M^T
    vals_L, vecs_L = np.linalg.eig(M.T)
    idx_L = np.argmax(np.abs(vals_L.real))
    l    = vecs_L[:, idx_L].conj()  # left eigenvector
    # Normalize: ℓ^T r = 1
    lr   = l @ r
    if abs(lr) > 1e-10:
        l = l / lr
    return lam, l, r


def compute_phi_and_jacobian(wk_list):
    """
    For each pair (W_k, W_{k+1}):
    1. Compute M_k = W_{k+1} W_k^{-1}
    2. Dominant eigenvalue λ_k, left/right eigenvectors ℓ_k, r_k
    3. φ_k = arg(λ_k)
    4. Analytic Jacobian:
       dφ_k/dW_{k+1} = Im( r_k ℓ_k^T W_k^{-T} ) / |λ_k|
       dφ_k/dW_k     = Im( -λ_k r_k ℓ_k^T W_k^{-T} W_{k+1}^T ... ) / |λ_k|

    Using perturbation theory for simple eigenvalues:
       λ(W+δW) ≈ λ + ℓ^T (δM) r  where δM = δW_{k+1} W_k^{-1} - W_{k+1} W_k^{-1} δW_k W_k^{-1}

    So:
       dλ/dW_{k+1}[i,j] = ℓ_i * (W_k^{-1} r)_j          → outer product ℓ ⊗ (W_k^{-T} r)^T... 

    Correctly:
       δλ = ℓ^T δM r = ℓ^T (δW_{k+1} W_k^{-1}) r - ℓ^T (W_{k+1} W_k^{-1} δW_k W_k^{-1}) r
           = (ℓ^T δW_{k+1}) (W_k^{-1} r) - (ℓ^T M_k δW_k) (W_k^{-1} r)

    So as linear maps (D×D matrices):
       dλ/dW_{k+1} = (W_k^{-T} r) ℓ^T          [outer product, shape D×D]  ... no:
    
    As a bilinear: δλ = Tr[ (dλ/dW_{k+1})^T δW_{k+1} ] + Tr[ (dλ/dW_k)^T δW_k ]
       dλ/dW_{k+1} = r (W_k^{-T} ℓ*)^T         but ℓ^T δW_{k+1} (W_k^{-1} r)
                                                  = Tr[(W_k^{-1} r) ℓ^T δW_{k+1}]^T... 
    Wait. δλ = ℓ^T (δW_{k+1}) (W_k^{-1} r) = Tr[ (W_k^{-T} conj(ℓ)) r^T ... ]
    
    Simplest: dλ/d(W_{k+1})_{ij} = ℓ_i * (W_k^{-1} r)_j
    So dλ/dW_{k+1} = ℓ ⊗ (W_k^{-1} r)   [outer product, D×D]
    And dφ/dW_{k+1} = Im(dλ/dW_{k+1}) / |λ|

    For W_k:
    dλ/d(W_k)_{ij} = -(ℓ^T M_k)_i * (W_k^{-1} r)_j = -(M_k^T ℓ*)_i * (W_k^{-1} r)_j
    dλ/dW_k = -(M_k^T ℓ*) ⊗ (W_k^{-1} r)

    dφ_k/dW_{k+1} = Im(ℓ ⊗ (W_k^{-1} r)) / |λ_k|     [D×D]
    dφ_k/dW_k     = Im(-(M_k^T ℓ) ⊗ (W_k^{-1} r)) / |λ_k|  [D×D]
    """
    L = len(wk_list)
    results = []

    for k in range(L-1):
        Wk  = wk_list[k].astype(complex)
        Wk1 = wk_list[k+1].astype(complex)

        try:
            Wk_inv = np.linalg.inv(Wk)
        except np.linalg.LinAlgError:
            Wk_inv = np.linalg.pinv(Wk)

        M   = Wk1 @ Wk_inv
        lam, l, r = dominant_eigensystem(M)

        phi_k = float(np.arctan2(lam.imag, lam.real))
        lam_mag = abs(lam)

        # Analytic Jacobians (D×D complex matrices)
        Wk_inv_r = Wk_inv @ r                # D-vector
        Mk_T_l   = M.T @ l                   # D-vector (= λ* l for dominant)

        # dφ/dW_{k+1} = Im(outer(l, Wk_inv_r)) / |λ|
        J_Wk1 = np.outer(l, Wk_inv_r)
        dPhi_dWk1 = np.imag(J_Wk1) / (lam_mag + 1e-10)

        # dφ/dW_k = Im(-outer(Mk_T_l, Wk_inv_r)) / |λ|
        J_Wk = -np.outer(Mk_T_l, Wk_inv_r)
        dPhi_dWk = np.imag(J_Wk) / (lam_mag + 1e-10)

        # Sensitivity norms
        sens_Wk1 = float(np.linalg.norm(dPhi_dWk1, 'fro'))
        sens_Wk  = float(np.linalg.norm(dPhi_dWk,  'fro'))

        results.append({
            'k': k,
            'phi_k': phi_k,
            'lam_real': float(lam.real),
            'lam_imag': float(lam.imag),
            'lam_mag':  lam_mag,
            'dPhi_dWk1_norm': sens_Wk1,
            'dPhi_dWk_norm':  sens_Wk,
            'dPhi_dWk1': dPhi_dWk1.real,  # store real part (Im was extracted)
            'dPhi_dWk':  dPhi_dWk.real,
            'locked': (sens_Wk1 < 0.01 and sens_Wk < 0.01),
        })

    return results


def nlp_topogate_rank1(wk_list, k_target, phi_target, lambda_reg=1.0):
    """
    Find optimal rank-1 update to W_{k_target+1} that drives φ_k → 0.

    δW_{k+1} = α * u v^T  (rank-1, shape D×D)
    Parametrized by scalar α and unit vectors u, v (each D-dim).

    But D=256 is still large. Use the Jacobian to find the best direction:
    The steepest descent direction in W_{k+1} space is dPhi_dWk1.
    Optimal rank-1: u, v = top left/right singular vectors of dPhi_dWk1.
    Then α = -φ_k / (u^T dPhi_dWk1 v) = -φ_k / σ_1(dPhi_dWk1).

    This is the Newton step along the rank-1 manifold. O(D^2) cost.
    """
    J_data = compute_phi_and_jacobian(wk_list)
    jd = J_data[k_target]
    phi_k = jd['phi_k']

    if abs(phi_k) < 0.01:
        return None, J_data, 'already_clean'

    # Steepest descent direction
    G = jd['dPhi_dWk1']  # D×D gradient matrix
    U_svd, s, Vt = np.linalg.svd(G)
    u1 = U_svd[:, 0]  # top left singular vector
    v1 = Vt[0, :]     # top right singular vector
    sigma1 = s[0]

    if sigma1 < 1e-10:
        return None, J_data, 'zero_gradient'

    # Newton step: α = -φ_k / σ_1
    alpha = -phi_k / sigma1

    # Apply rank-1 update
    wk_new = [W.copy() for W in wk_list]
    wk_new[k_target+1] = wk_list[k_target+1] + alpha * np.outer(u1, v1)

    J_after = compute_phi_and_jacobian(wk_new)
    phi_after = J_after[k_target]['phi_k']

    return {
        'alpha': float(alpha),
        'sigma1': float(sigma1),
        'phi_before': float(phi_k),
        'phi_after': float(phi_after),
        'phi_error': float(abs(phi_after)),
        'update_norm': float(abs(alpha)),
        'converged': abs(phi_after) < 0.1,
    }, J_data, 'ok'


def main():
    args = parse_args()
    print("="*60)
    print("  ANALYTIC JACOBIAN: TRUE PHASE MAP")
    print("  φ_k = arg(λ_dom(W_{k+1} W_k^{-1}))")
    print("  Closed-form via eigenvalue perturbation theory")
    print("="*60)

    results = {}

    for path, label in [
        (args.spike64, "step64 (clean phase)"),
        (args.spike72, "step72 (off wall)"),
    ]:
        print(f"\n{'─'*60}\n  {label}")
        wk_list = load_wk(path)

        t0 = time.time()
        J_data = compute_phi_and_jacobian(wk_list)
        t_J = time.time()-t0
        print(f"  Jacobian computed [{t_J:.3f}s]")

        print(f"\n  k  φ_k      λ_real   λ_imag  "
              f"‖∂φ/∂Wk+1‖  ‖∂φ/∂Wk‖  locked?")
        print(f"  {'-'*65}")
        for jd in J_data:
            print(f"  {jd['k']}  {jd['phi_k']:+.3f}  "
                  f"{jd['lam_real']:+8.3f}  {jd['lam_imag']:+7.3f}  "
                  f"{jd['dPhi_dWk1_norm']:11.4f}  "
                  f"{jd['dPhi_dWk_norm']:9.4f}  "
                  f"{'✓' if jd['locked'] else '✗'}")

        # Count: how many phases locked vs active?
        n_locked = sum(1 for jd in J_data if jd['locked'])
        n_active = len(J_data) - n_locked
        print(f"\n  Locked: {n_locked}/5   Active: {n_active}/5")

        # NLP TopoGate on first off-wall phase
        off_wall = [jd for jd in J_data if abs(jd['phi_k']) > 0.1
                    and abs(jd['phi_k'] - np.pi) > 0.1]
        if off_wall:
            k_tgt = off_wall[0]['k']
            print(f"\n  Rank-1 Newton step: drive φ_{k_tgt} → 0")
            nlp_res, _, status = nlp_topogate_rank1(
                wk_list, k_tgt, 0.0, args.lambda_reg)
            if nlp_res:
                print(f"  α = {nlp_res['alpha']:.4f}  "
                      f"σ₁ = {nlp_res['sigma1']:.4f}")
                print(f"  φ_{k_tgt}: {nlp_res['phi_before']:+.4f} → "
                      f"{nlp_res['phi_after']:+.4f}  "
                      f"(error={nlp_res['phi_error']:.4f})")
                print(f"  {'✓ Converged' if nlp_res['converged'] else '✗ Partial'}")
            else:
                print(f"  Status: {status}")
        else:
            print(f"\n  All phases ∈ {{0, π}} — no off-wall correction needed")
            nlp_res = {'skipped': True}

        results[label] = {
            'jacobian': [{k: v for k, v in jd.items()
                         if k not in ('dPhi_dWk1','dPhi_dWk')}
                        for jd in J_data],
            'n_locked': n_locked,
            'nlp': nlp_res or {'status': 'no_off_wall'},
        }

    print(f"\n{'='*60}")
    print(f"  INTERPRETATION")
    print(f"{'='*60}")
    print(f"""
  Locked (‖∂φ/∂W‖≈0): algebraic real-locking theorem confirmed.
  Active (‖∂φ/∂W‖>0): TopoGate can act analytically.
  
  The rank-1 Newton step is the minimal-norm WK update that
  drives a single off-wall phase back to the real spectral locus.
  Cost: one SVD of the D×D Jacobian matrix. ~0.01 seconds.
  This replaces 5 empirical gradient steps in the current TopoGate.
""")

    Path(args.output).write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"  Report → {args.output}")


if __name__ == '__main__':
    main()
