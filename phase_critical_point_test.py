"""
phase_critical_point_test.py
=============================
Tests the Phase Critical Point Theorem:

  Theorem (candidate): Let φ(w) = arg λ(w) where λ(w) is a simple
  eigenvalue of a smooth matrix family M(w). If λ(w₀) > 0 is real
  and the left/right eigenvectors are real, then dφ(w₀) = 0 as a
  covector, and the leading variation is quadratic:

    φ(w₀ + δw) = ½ D²φ(w₀)[δw, δw] + O(‖δw‖³)

Three tests:
  A) Verify dφ=0 globally: measure |dφ/dv| for many random directions v.
     If all ≈ 0: dφ=0 globally → quadratic expansion is the right model.
     If some are large: dφ≠0 on those directions → wall is still regular.

  B) Verify real eigenvectors at clean phase: compute Im(ℓ) and Im(r).
     If Im(ℓ)≈0 and Im(r)≈0: eigenvectors real → dφ=0 follows from theory.

  C) Fit quadratic expansion: measure φ(w₀ + ε·v) for multiple ε values
     and fit φ = c₀ + c₁ε + c₂ε² to confirm c₁≈0, c₂≠0.
     D²φ[v,v] = 2c₂ is the Hessian of φ in direction v.

If A confirms dφ=0 globally and B confirms real eigenvectors:
  → Phase Critical Point Theorem holds
  → Wall W = {φ=0} is a Morse-degenerate level set
  → Local model: φ = Q(y) + O(‖y‖³) where Q is quadratic
  → ker H = T_w W holds trivially (T_w W = R^D to first order)
  → The null-space splitting is a second-order phenomenon

Usage
-----
  python phase_critical_point_test.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --spike72 tau_spikes/tau_spike_step0072_tau5.94.pt \\
      --rank 6 --n_random 100 --n_eps 7 \\
      --output phase_critical_report.json
"""

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64', default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--spike72', default='tau_spikes/tau_spike_step0072_tau5.94.pt')
    p.add_argument('--rank',     type=int,   default=6)
    p.add_argument('--n_random', type=int,   default=100,
                   help='Random directions for test A')
    p.add_argument('--n_eps',    type=int,   default=7,
                   help='Epsilon values for quadratic fit')
    p.add_argument('--eps_max',  type=float, default=0.5)
    p.add_argument('--k',        type=int,   default=0,
                   help='Which pair (Lk, Lk+1) to test')
    p.add_argument('--output',   default='phase_critical_report.json')
    return p.parse_args()


def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        meta  = ckpt.get('metadata', {})
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        meta, state = {}, ckpt
    return {k: v.clone() for k, v in state.items()
            if isinstance(v, torch.Tensor)}, meta


def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor
    return [wk[i].detach().float() for i in sorted(wk)]


# ─── Phase map and eigenvalue structure ──────────────────────────────────────

def compute_M(wk_list, k):
    """M = W_{k+1} W_k^{-1}: the matrix whose dominant eigenvalue gives φ_k."""
    Wk  = wk_list[k].float().numpy()
    Wk1 = wk_list[k+1].float().numpy()
    return Wk1 @ np.linalg.pinv(Wk)

def dominant_eigenvalue(M):
    """Returns (λ_dom, left_evec, right_evec) for the dominant eigenvalue."""
    evals, evecs_r = np.linalg.eig(M)
    idx = np.argmax(np.abs(evals.real))
    lam = evals[idx]
    r   = evecs_r[:, idx]

    # Left eigenvector: solve M^T ℓ = λ* ℓ
    evals_l, evecs_l = np.linalg.eig(M.T)
    # Find the left eigenvector corresponding to λ
    diffs = np.abs(evals_l - np.conj(lam))
    idx_l = np.argmin(diffs)
    ell   = evecs_l[:, idx_l]

    return lam, ell, r

def phi_of_wk(wk_list, k):
    """φ_k = arg(λ_dom(M_k))"""
    M = compute_M(wk_list, k)
    lam, _, _ = dominant_eigenvalue(M)
    phi = float(np.arctan2(lam.imag, lam.real))
    if phi < 0: phi += 2*math.pi
    return phi

def phi_perturbed(wk_list, k, v_flat, alpha):
    """φ_k(θ + α·v)"""
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    wk_new = []
    for i, w in enumerate(wk_list):
        s, e = offsets[i], offsets[i+1]
        delta = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
        wk_new.append(w + alpha * delta)
    return phi_of_wk(wk_new, k)


# ─── Test A: Is dφ=0 globally? ───────────────────────────────────────────────

def test_A_global_dphi(wk_list, k, n_random, eps, label):
    """
    Sample n_random directions in WK[k]+WK[k+1] block.
    Compute |dφ/dv| = |φ(θ+εv) - φ(θ-εv)| / (2ε).
    If all ≈ 0: dφ=0 globally.
    """
    D = sum(w.numel() for w in wk_list)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))

    # Restrict to WK[k] and WK[k+1] blocks
    s0 = offsets[k]
    e1 = offsets[min(k+2, len(wk_list))]
    block_size = e1 - s0

    rng = np.random.default_rng(42)
    dphi_values = []

    for trial in range(n_random):
        v = np.zeros(D)
        v_block = rng.standard_normal(block_size)
        v_block /= np.linalg.norm(v_block) + 1e-10
        v[s0:e1] = v_block

        phi_p = phi_perturbed(wk_list, k, v, +eps)
        phi_m = phi_perturbed(wk_list, k, v, -eps)
        dphi  = (phi_p - phi_m) / (2*eps)
        # Handle wraparound
        if abs(dphi) > math.pi: dphi -= math.copysign(2*math.pi, dphi)
        dphi_values.append(abs(dphi))

    dphi_arr = np.array(dphi_values)
    mean_dphi = float(np.mean(dphi_arr))
    max_dphi  = float(np.max(dphi_arr))
    frac_zero = float(np.mean(dphi_arr < 0.01))

    print(f"\n  Test A ({label}): |dφ/dv| over {n_random} random directions")
    print(f"    Mean |dφ/dv| = {mean_dphi:.6f}")
    print(f"    Max  |dφ/dv| = {max_dphi:.6f}")
    print(f"    Fraction < 0.01: {frac_zero:.3f}")

    if max_dphi < 0.01:
        verdict = "DPH_ZERO_GLOBALLY"
        msg = "dφ=0 as a covector. Phase Critical Point Theorem applies."
    elif mean_dphi < 0.01 and max_dphi < 0.1:
        verdict = "DPH_APPROXIMATELY_ZERO"
        msg = f"dφ≈0 (max={max_dphi:.4f}). Nearly degenerate critical point."
    else:
        verdict = "DPH_NONZERO"
        msg = f"dφ≠0 (max={max_dphi:.4f}). Wall is regular codimension-1."

    print(f"    Verdict: {verdict} — {msg}")

    return {
        'dphi_values': dphi_arr.tolist(),
        'mean_dphi': mean_dphi,
        'max_dphi': max_dphi,
        'frac_zero': frac_zero,
        'verdict': verdict,
        'message': msg,
    }


# ─── Test B: Are eigenvectors real? ──────────────────────────────────────────

def test_B_real_eigenvectors(wk_list, k, label):
    """
    Compute the dominant eigenvectors of M = W_{k+1} W_k^{-1}.
    If Im(ℓ) ≈ 0 and Im(r) ≈ 0: eigenvectors are real.
    Real eigenvectors → dλ is real for real perturbations → dφ=0.
    """
    M = compute_M(wk_list, k)
    lam, ell, r = dominant_eigenvalue(M)

    phi0 = float(np.arctan2(lam.imag, lam.real))
    lam_abs = float(abs(lam))
    im_lam  = float(lam.imag)

    im_ell = float(np.linalg.norm(ell.imag) / (np.linalg.norm(ell) + 1e-8))
    im_r   = float(np.linalg.norm(r.imag)   / (np.linalg.norm(r)   + 1e-8))

    print(f"\n  Test B ({label}): Eigenvector reality")
    print(f"    φ₀ = {phi0:.4f} rad  |λ| = {lam_abs:.4f}  Im(λ) = {im_lam:.6f}")
    print(f"    Im(ℓ)/|ℓ| = {im_ell:.6f}  (< 0.01 = real)")
    print(f"    Im(r)/|r| = {im_r:.6f}   (< 0.01 = real)")

    eigvecs_real = (im_ell < 0.01) and (im_r < 0.01)
    lam_real     = abs(phi0) < 0.01 or abs(phi0 - math.pi) < 0.01

    if lam_real and eigvecs_real:
        verdict = "REAL_EIGENVECTORS"
        msg = "λ real, ℓ,r real → dφ=0 by perturbation theory. Q.E.D."
    elif lam_real and not eigvecs_real:
        verdict = "REAL_LAMBDA_COMPLEX_EVECS"
        msg = f"λ real but eigenvectors complex (Im(ℓ)={im_ell:.4f}). " \
              "Degenerate eigenvalue or numerical issue."
    else:
        verdict = "COMPLEX_LAMBDA"
        msg = f"λ not real (φ={phi0:.4f}). Not at clean phase."

    print(f"    Verdict: {verdict} — {msg}")

    return {
        'phi0': float(phi0),
        'lambda_abs': lam_abs,
        'lambda_imag': im_lam,
        'im_ell_frac': im_ell,
        'im_r_frac': im_r,
        'eigvecs_real': bool(eigvecs_real),
        'lambda_real': bool(lam_real),
        'verdict': verdict,
        'message': msg,
    }


# ─── Test C: Quadratic fit of phase expansion ─────────────────────────────────

def test_C_quadratic_fit(wk_list, k, n_eps, eps_max, n_random_dirs, label):
    """
    For several directions v and multiple ε values, fit:
      φ(w₀ + εv) = c₀ + c₁ε + c₂ε²

    Key test: does c₁(h) → 0 as h → 0?
    If yes: apparent linear term is finite-window artifact, not genuine dφ.
    Uses progressively smaller windows to distinguish the two cases.
    """
    D = sum(w.numel() for w in wk_list)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    s0 = offsets[k]; e1 = offsets[min(k+2, len(wk_list))]
    block_size = e1 - s0

    phi0 = phi_of_wk(wk_list, k)

    rng = np.random.default_rng(123)
    all_results = []

    # Progressive window sizes: h = 0.1, 0.01, 0.001, 0.0001
    window_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001]

    print(f"\n  Test C ({label}): c₁(h) → 0 as h → 0?")
    print(f"    φ₀ = {phi0:.4f} rad")
    print(f"    {'h':>8} {'mean|c₁|':>12} {'mean|c₂|':>12} {'c₁→0?':>8}")
    print(f"    {'-'*48}")

    convergence_data = []
    for h in window_sizes:
        eps_vals = np.array([-h, -h/2, 0, h/2, h])
        c1_list, c2_list = [], []

        for trial in range(n_random_dirs):
            v = np.zeros(D)
            v_block = rng.standard_normal(block_size)
            v_block /= np.linalg.norm(v_block) + 1e-10
            v[s0:e1] = v_block

            phi_vals = []
            for eps in eps_vals:
                if eps == 0:
                    phi_vals.append(0.)
                else:
                    phi_eps = phi_perturbed(wk_list, k, v, eps)
                    dphi = phi_eps - phi0
                    if dphi >  math.pi: dphi -= 2*math.pi
                    if dphi < -math.pi: dphi += 2*math.pi
                    phi_vals.append(dphi)

            phi_arr = np.array(phi_vals)
            # Fit c₀ + c₁ε + c₂ε² (c₀ ≈ 0 since we subtracted phi0)
            X = np.column_stack([np.ones(5), eps_vals, eps_vals**2])
            coeffs, _, _, _ = np.linalg.lstsq(X, phi_arr, rcond=None)
            c1_list.append(abs(coeffs[1]))
            c2_list.append(abs(coeffs[2]))

        mean_c1 = float(np.mean(c1_list))
        mean_c2 = float(np.mean(c2_list))
        to_zero = mean_c1 < h * 0.1   # c1 shrinks with h
        convergence_data.append({'h': h, 'mean_c1': mean_c1, 'mean_c2': mean_c2})
        print(f"    {h:>8.4f} {mean_c1:>12.6f} {mean_c2:>12.6f} "
              f"{'yes' if to_zero else 'no':>8}")

    # Check if c₁ decreases proportionally to h (→0 as h→0)
    c1_vals = [d['mean_c1'] for d in convergence_data]
    h_vals  = [d['h']       for d in convergence_data]
    # Fit log(c1) ~ α log(h) + β
    log_h  = np.log(h_vals)
    log_c1 = np.log(np.array(c1_vals) + 1e-12)
    alpha, beta = np.polyfit(log_h, log_c1, 1)

    print(f"\n    Scaling: c₁ ~ h^{alpha:.2f}  (expect α≈1 if c₁→0 linearly with h)")
    print(f"    α = {alpha:.3f}: ", end="")

    if alpha > 0.5:
        scaling_verdict = "C1_CONVERGES_TO_ZERO"
        print(f"c₁ → 0 as h → 0. Linear term is finite-window artifact.")
        print(f"    dφ = 0 confirmed by window convergence.")
    else:
        scaling_verdict = "C1_DOES_NOT_CONVERGE"
        print(f"c₁ does not vanish as h → 0. Genuine linear component present.")

    mean_c2_all = float(np.mean([d['mean_c2'] for d in convergence_data[2:]]))

    if scaling_verdict == "C1_CONVERGES_TO_ZERO" and mean_c2_all > 0.001:
        verdict = "QUADRATIC_CONFIRMED"
        msg = (f"c₁ ~ h^{alpha:.2f} → 0, c₂ ≈ {mean_c2_all:.4f} ≠ 0. "
               f"Quadratic expansion φ = ½D²φ[δw,δw] + O(‖δw‖³) confirmed.")
    elif scaling_verdict == "C1_CONVERGES_TO_ZERO":
        verdict = "APPROXIMATELY_QUADRATIC"
        msg = f"c₁ → 0 confirmed, c₂ small. Consistent with quadratic."
    else:
        verdict = "LINEAR_DOMINANT"
        msg = f"c₁ does not vanish. Genuine first-order contribution."

    print(f"    Verdict: {verdict}")
    return {
        'phi0': float(phi0),
        'convergence_data': convergence_data,
        'alpha': float(alpha),
        'mean_c2_small_h': mean_c2_all,
        'scaling_verdict': scaling_verdict,
        'verdict': verdict,
        'message': msg,
    }


def compute_d2phi(wk_list, k, v, eps=1e-3):
    """
    Estimate D²φ(w₀)[v,v] using the explicit second-order formula:

    D²φ[v,v] = (2/λ₀) · Im(λ⁽²⁾[v,v])

    where λ⁽²⁾[v,v] = Σⱼ≠dom |ℓᵀ(dM·v)rⱼ|² / (λ₀ - λⱼ)
    is the second-order eigenvalue perturbation.

    Alternatively estimated numerically as:
    D²φ[v,v] ≈ [φ(w+εv) + φ(w-εv) - 2φ(w)] / ε²
    (central second difference, valid since dφ=0 → no linear term)
    """
    phi0  = phi_of_wk(wk_list, k)
    phi_p = phi_perturbed(wk_list, k, v, +eps)
    phi_m = phi_perturbed(wk_list, k, v, -eps)

    dphi_p = phi_p - phi0
    dphi_m = phi_m - phi0
    for dphi in [dphi_p, dphi_m]:
        if dphi >  math.pi: dphi -= 2*math.pi
        if dphi < -math.pi: dphi += 2*math.pi

    d2phi = (dphi_p + dphi_m) / eps**2   # second central difference
    return float(d2phi)


def test_D_second_order_hessian(wk_list, k, n_dirs, label):
    """
    Compute D²φ(w₀)[v,v] for multiple directions v.
    Compare numerical estimate with analytic formula from
    second-order eigenvalue perturbation theory.

    Analytic formula:
      D²φ[v,v] = (2/λ₀) · Im[Σⱼ≠dom (ℓᵀ Aⱼ r)(ℓⱼᵀ Aⱼ r) / (λⱼ - λ₀)]
    where Aⱼ = dM·v is the first-order perturbation of M in direction v.
    """
    D = sum(w.numel() for w in wk_list)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    s0 = offsets[k]; e1 = offsets[min(k+2, len(wk_list))]
    block_size = e1 - s0

    M = compute_M(wk_list, k)
    lam0, ell, r = dominant_eigenvalue(M)
    evals_all, evecs_all = np.linalg.eig(M)
    evals_l_all, evecs_l_all = np.linalg.eig(M.T)

    dom_idx = np.argmax(np.abs(evals_all.real))
    lam0_val = float(lam0.real)

    rng = np.random.default_rng(77)
    print(f"\n  Test D ({label}): D²φ[v,v] — numerical vs analytic")
    print(f"    λ₀ = {lam0_val:.4f}  (real)")
    print(f"    {'dir':>5} {'D²φ_num':>12} {'D²φ_ana':>12} {'match':>8}")
    print(f"    {'-'*42}")

    results = []
    for trial in range(n_dirs):
        v_flat = np.zeros(D)
        v_block = rng.standard_normal(block_size)
        v_block /= np.linalg.norm(v_block) + 1e-10
        v_flat[s0:e1] = v_block

        # Numerical D²φ
        d2phi_num = compute_d2phi(wk_list, k, v_flat, eps=1e-3)

        # Analytic D²φ via perturbation theory
        # dM·v = (dW_{k+1}·v) @ W_k^{-1} - W_{k+1} @ W_k^{-1} @ (dW_k·v) @ W_k^{-1}
        # Approximate: use finite difference of M
        wk_list_p = [w.clone() for w in wk_list]
        wk_list_m = [w.clone() for w in wk_list]
        for i, w in enumerate(wk_list):
            s, e = offsets[i], offsets[i+1]
            delta = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
            wk_list_p[i] = w + 1e-4 * delta
            wk_list_m[i] = w - 1e-4 * delta
        Mp = compute_M(wk_list_p, k)
        Mm = compute_M(wk_list_m, k)
        dM = (Mp - Mm) / (2e-4)   # first-order perturbation of M

        # Second-order eigenvalue perturbation
        # λ⁽²⁾ = Σⱼ≠dom (ℓᵀ dM rⱼ)(ℓⱼᵀ dM r) / (λ₀ - λⱼ)
        lam2 = complex(0)
        for j, (lj, rj, lj_l) in enumerate(
                zip(evals_all, evecs_all.T, evecs_l_all.T)):
            if j == dom_idx: continue
            if abs(lj - lam0) < 1e-6: continue
            num = (ell.conj() @ dM @ rj) * (lj_l.conj() @ dM @ r)
            lam2 += num / (lam0 - lj)

        d2phi_ana = float(2 * lam2.imag / (lam0_val + 1e-8))

        match = abs(d2phi_num - d2phi_ana) < max(0.1*abs(d2phi_num), 0.1)
        print(f"    {trial:>5} {d2phi_num:>12.4f} {d2phi_ana:>12.4f} "
              f"{'✓' if match else '✗':>8}")
        results.append({'d2phi_num': float(d2phi_num),
                        'd2phi_ana': float(d2phi_ana),
                        'match': bool(match)})

    n_match = sum(1 for r in results if r['match'])
    print(f"\n    Match: {n_match}/{n_dirs}")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  PHASE CRITICAL POINT TEST")
    print("  Tests: dφ=0 globally? φ = ½D²φ[δw,δw] + O(‖δw‖³)?")
    print("=" * 60)

    results = {}

    for path, label in [(args.spike64, "step64 (φ=0, on wall)"),
                        (args.spike72, "step72 (φ=1.02, off wall)")]:
        print(f"\n{'─'*60}\n  {path}")
        state, meta = load_state(path)
        wk_list = extract_wk(state)
        phi0 = phi_of_wk(wk_list, args.k)
        print(f"  φ_{args.k} = {phi0:.4f} rad  "
              f"({'clean' if abs(phi0)<0.3 or abs(phi0-math.pi)<0.3 else 'non-clean'})")

        t0 = time.time()
        eps_A = 0.01
        rA = test_A_global_dphi(wk_list, args.k, args.n_random, eps_A, label)
        rB = test_B_real_eigenvectors(wk_list, args.k, label)
        rC = test_C_quadratic_fit(wk_list, args.k, args.n_eps,
                                   args.eps_max, 5, label)
        rD = test_D_second_order_hessian(wk_list, args.k, 5, label)
        elapsed = time.time() - t0

        results[label] = {
            'path': path, 'phi0': float(phi0),
            'test_A': rA, 'test_B': rB, 'test_C': rC, 'test_D': rD,
            'elapsed_s': round(elapsed, 1),
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: PHASE CRITICAL POINT THEOREM")
    print(f"{'='*60}")
    for label, r in results.items():
        A, B, C = r['test_A'], r['test_B'], r['test_C']
        all_confirm = (A['verdict'] in ('DPH_ZERO_GLOBALLY','DPH_APPROXIMATELY_ZERO')
                       and B['verdict'] == 'REAL_EIGENVECTORS'
                       and C['verdict'] in ('QUADRATIC_CONFIRMED','APPROXIMATELY_QUADRATIC'))
        print(f"\n  {label}:")
        print(f"    A (dφ=0 globally): {A['verdict']}  max={A['max_dphi']:.4f}")
        print(f"    B (real evecs):    {B['verdict']}")
        print(f"    C (quadratic fit): {C['verdict']}  α={C['alpha']:.2f}  c₂={C['mean_c2_small_h']:.4f}")
        print(f"    Overall: {'✓ PHASE CRITICAL POINT CONFIRMED' if all_confirm else '◑ partial'}")

    print(f"\n  Geometric interpretation:")
    print(f"  At clean phase φ∈{{0,π}}: dominant eigenvalue λ is real,")
    print(f"  eigenvectors are real → dφ=0 globally (all directions)")
    print(f"  → Wall W={{φ=0}} is Morse-degenerate level set of φ")
    print(f"  → ker H = T_w W holds trivially (T_w W = R^D to first order)")
    print(f"  → Null-space splitting is second-order phenomenon")
    print(f"  → Block Hessian: H = diag(0,A) + O(ε²) not O(ε)")

    Path(args.output).write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
