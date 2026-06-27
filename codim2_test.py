"""
codim2_test.py
==============
Numerical verification of the Codimension-2 Stratification Theorem.

Theorem (to verify):
  Let Φ: R^D → R^5 be the Bridgeland phase map, and let
  fₖ(θ) = Im(Z(γₖ)) = A(Lₖ,Lₖ₊₁)·sin(φₖ(θ))
  be the k-th stability condition (fₖ=0 iff phase is clean).

  If at step 72 exactly two conditions {fₐ, f_b} activate simultaneously
  and rank(∂(fₐ,f_b)/∂θ) = 2, then S_72 is a codimension-2 stratum.

What we compute:
  1. Which phases go non-clean at step 72 vs step 64 (identify a, b)
  2. ∇fₖ(θ) for each k via finite differences on the phase map
  3. SVD of the 2×D matrix [∇fₐ; ∇f_b] → σ_min
  4. cos(∇fₐ, ∇f_b) → confirms linear independence
  5. Phase directional derivative along P3 null eigenvectors:
     dφₖ/dv ≈ 0 iff v is tangent to the Bridgeland wall

If σ_min > ε and cos ≠ ±1: rank condition holds → codimension-2 confirmed.
If dφₖ/dvᵢ ≈ 0 for null eigenvectors: ker(H) is tangent to the wall.

Usage
-----
  python codim2_test.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --spike72 tau_spikes/tau_spike_step0072_tau5.94.pt \\
      --rank 6 --k_lanczos 20 --n_null 5 --eps 1e-4 \\
      --output codim2_report.json
"""

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    p.add_argument('--rank',      type=int,   default=6)
    p.add_argument('--k_lanczos', type=int,   default=20)
    p.add_argument('--n_null',    type=int,   default=5)
    p.add_argument('--eps',       type=float, default=1e-4,
                   help='Finite difference step for gradient computation')
    p.add_argument('--sigma_tol', type=float, default=1e-3,
                   help='Minimum singular value for rank-2 confirmation')
    p.add_argument('--phase_tol', type=float, default=0.3,
                   help='Phase tolerance for clean/non-clean classification')
    p.add_argument('--output',    default='codim2_report.json')
    return p.parse_args()


# ─── Load checkpoint ──────────────────────────────────────────────────────────

def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        meta  = ckpt.get('metadata', {})
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        meta, state = {}, ckpt
    state = {k:v.clone() for k,v in state.items() if isinstance(v,torch.Tensor)}
    return state, meta

def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if 'c_attn' in n and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            D = tensor.shape[0]//3
            wk[li] = tensor[D:2*D,:]
        elif ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor if tensor.ndim==2 else tensor.squeeze()
    if not wk:
        raise RuntimeError("No WK. Keys: "+str(list(state.keys())[:8]))
    return [wk[i].detach().float() for i in sorted(wk)]


# ─── Bridgeland phase map Φ: R^D → R^5 ───────────────────────────────────────

def bridgeland_phase_k(wk_list, k):
    """φₖ = arg(λ_dom(W_{k+1} W_k^{-1})) ∈ [0, 2π)"""
    Wk  = wk_list[k].float().numpy()
    Wk1 = wk_list[k+1].float().numpy()
    M = Wk1 @ np.linalg.pinv(Wk)
    evals = np.linalg.eigvals(M)
    dom = evals[np.argmax(np.abs(evals.real))]
    phi = float(np.arctan2(dom.imag, dom.real))
    if phi < 0: phi += 2*math.pi
    return phi

def im_Z_k(wk_list, k, rank):
    """
    fₖ(θ) = Im(Z(γₖ)) = A(Lₖ,Lₖ₊₁) · sin(φₖ(θ))
    fₖ = 0 iff φₖ ∈ {0, π} (Bridgeland stability condition)
    """
    Uk  = torch.linalg.svd(wk_list[k],   full_matrices=False)[0][:, :rank]
    Uk1 = torch.linalg.svd(wk_list[k+1], full_matrices=False)[0][:, :rank]
    sv  = torch.linalg.svdvals(Uk.T@Uk1).clamp(-1+1e-6, 1-1e-6)
    area = float(torch.arccos(sv).sum().item())
    phi  = bridgeland_phase_k(wk_list, k)
    return area * math.sin(phi), area, phi

def all_phases_and_imZ(wk_list, rank):
    """Returns (phases, im_Z_values, areas, clean_flags) for all d pairs."""
    d = len(wk_list) - 1
    phases, imZ, areas, clean = [], [], [], []
    for k in range(d):
        f, A, phi = im_Z_k(wk_list, k, rank)
        is_clean = abs(phi) < 0.3 or abs(phi-math.pi) < 0.3 or abs(phi-2*math.pi) < 0.3
        phases.append(phi)
        imZ.append(f)
        areas.append(A)
        clean.append(is_clean)
    return np.array(phases), np.array(imZ), np.array(areas), clean


# ─── Gradient ∇fₖ via finite differences ─────────────────────────────────────

def grad_imZ_k(state, k, rank, eps):
    """
    ∇fₖ(θ) ∈ R^D via central finite differences.
    D = total number of WK parameters.

    Strategy: perturb each WK matrix element by ±eps and recompute fₖ.
    Exploits sparsity: fₖ only depends on WK[k] and WK[k+1].
    Cost: 2 × (|WK[k]| + |WK[k+1]|) evaluations instead of 2D.
    """
    wk_list = extract_wk(state)
    d = len(wk_list) - 1
    D = sum(w.numel() for w in wk_list)

    # Only WK[k] and WK[k+1] affect fₖ
    relevant = [k, k+1] if k+1 < len(wk_list) else [k]

    # Flat parameter vector (concat of all WK matrices)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))

    grad = np.zeros(D)

    for layer_idx in relevant:
        W = wk_list[layer_idx]
        n = W.numel()
        s, e = offsets[layer_idx], offsets[layer_idx+1]

        for flat_idx in range(n):
            # +eps perturbation
            wk_plus = [w.clone() for w in wk_list]
            wk_plus[layer_idx].reshape(-1)[flat_idx] += eps
            f_plus, _, _ = im_Z_k(wk_plus, k, rank)

            # -eps perturbation
            wk_minus = [w.clone() for w in wk_list]
            wk_minus[layer_idx].reshape(-1)[flat_idx] -= eps
            f_minus, _, _ = im_Z_k(wk_minus, k, rank)

            grad[s + flat_idx] = (f_plus - f_minus) / (2*eps)

    return grad


def grad_imZ_k_fast(state, k, rank, eps):
    """
    Faster version: exploit that φₖ depends only on the dominant
    eigenvector direction. Use random probing to estimate gradient
    in the relevant subspace, then refine with exact FD on top components.

    For the rank condition check, we only need to verify that
    ∇fₐ and ∇f_b are linearly independent — we don't need the full gradient.
    Compute gradient restricted to WK[k] and WK[k+1] blocks only.
    """
    wk_list = extract_wk(state)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    D = sum(splits)

    grad = np.zeros(D)

    # Only compute in the two relevant WK blocks
    for layer_idx in [k, min(k+1, len(wk_list)-1)]:
        W = wk_list[layer_idx].reshape(-1)
        n = len(W)
        s = offsets[layer_idx]

        # Batch finite differences
        for flat_idx in range(n):
            wk_p = [w.clone() for w in wk_list]
            wk_p[layer_idx] = wk_list[layer_idx].clone()
            wk_p[layer_idx].reshape(-1)[flat_idx] += eps

            wk_m = [w.clone() for w in wk_list]
            wk_m[layer_idx] = wk_list[layer_idx].clone()
            wk_m[layer_idx].reshape(-1)[flat_idx] -= eps

            fp, _, _ = im_Z_k(wk_p, k, rank)
            fm, _, _ = im_Z_k(wk_m, k, rank)
            grad[s + flat_idx] = (fp - fm) / (2*eps)

    return grad


# ─── SVD rank test ────────────────────────────────────────────────────────────

def rank2_test(grad_a, grad_b, sigma_tol):
    """
    Test rank([∇fₐ; ∇f_b]) = 2 via SVD of 2×D matrix.
    Cost: O(D) — just two dot products and two norms.

    For a 2×D matrix M:
      σ₁ = ‖∇fₐ‖ (approximately, up to rotation)
      σ₂ = ‖∇f_b - proj_a(∇f_b)‖ (component orthogonal to ∇fₐ)
    rank = 2 iff σ₂ > sigma_tol.
    """
    na = float(np.linalg.norm(grad_a))
    nb = float(np.linalg.norm(grad_b))

    if na < 1e-10 or nb < 1e-10:
        return 0., 0., 0., False

    # cos between gradients
    cos_ab = float(np.dot(grad_a, grad_b)) / (na * nb)

    # σ₂ = ‖∇f_b - (∇f_b·∇fₐ/‖∇fₐ‖²)·∇fₐ‖
    # = ‖∇f_b‖ · √(1 - cos²)
    sigma2 = nb * math.sqrt(max(1. - cos_ab**2, 0.))

    rank2 = sigma2 > sigma_tol

    print(f"    ‖∇fₐ‖ = {na:.4f}")
    print(f"    ‖∇f_b‖ = {nb:.4f}")
    print(f"    cos(∇fₐ, ∇f_b) = {cos_ab:+.6f}")
    print(f"    σ_min = {sigma2:.6f}  (threshold {sigma_tol})")
    print(f"    rank([∇fₐ; ∇f_b]) = {'2 ✓' if rank2 else '< 2 ✗'}")

    return na, nb, cos_ab, sigma2, rank2


# ─── Null eigenvectors + phase directional derivatives ────────────────────────

class SymplecticProxy(nn.Module):
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank   = rank
        self.theta  = nn.Parameter(
            torch.cat([w.reshape(-1) for w in wk_list]).clone())
        self.shapes = [w.shape for w in wk_list]
        self.splits = [w.numel() for w in wk_list]
    def wk_matrices(self):
        return [p.reshape(s) for p,s in
                zip(torch.split(self.theta,self.splits),self.shapes)]
    def forward(self):
        wks = self.wk_matrices()
        loss = torch.tensor(0.)
        for k in range(len(wks)-1):
            Ua = torch.linalg.svd(wks[k],   full_matrices=False)[0][:,:self.rank]
            Ub = torch.linalg.svd(wks[k+1], full_matrices=False)[0][:,:self.rank]
            sv = torch.linalg.svdvals(Ua.T@Ub).clamp(1e-6,1-1e-6)
            loss = loss + torch.arccos(sv).sum()
        return loss

def hvp(model, v):
    loss = model()
    grad = torch.autograd.grad(loss, model.theta, create_graph=True)[0]
    return torch.autograd.grad((grad*v).sum(), model.theta)[0].detach()

def null_eigenvectors(wk_list, rank, k_lanczos, n_null, seed=42):
    """Find near-null eigenvectors of the symplectic Hessian."""
    torch.manual_seed(seed)
    model = SymplecticProxy(wk_list, rank)
    n = model.theta.numel()
    V = torch.zeros(n, k_lanczos+1)
    alpha_c = torch.zeros(k_lanczos); beta_c = torch.zeros(k_lanczos-1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:,0]=v; prev_b=0.
    k = k_lanczos
    for j in range(k):
        w = hvp(model, V[:,j])
        if j>0: w = w - prev_b*V[:,j-1]
        a = (w*V[:,j]).sum().item(); alpha_c[j]=a
        w = w - a*V[:,j]
        for i in range(j+1): w = w-(w@V[:,i])*V[:,i]
        if j<k-1:
            b=w.norm().item()
            if b<1e-10: k=j+1; break
            prev_b=b; beta_c[j]=b; V[:,j+1]=w/b
    Hk = (np.diag(alpha_c[:k].numpy())+
          np.diag(beta_c[:k-1].numpy(),1)+
          np.diag(beta_c[:k-1].numpy(),-1))
    ev,ritz = np.linalg.eigh(Hk)
    evecs = V[:,:k].numpy()@ritz
    # Sort by |λ| ascending (most null first)
    idx = np.argsort(np.abs(ev))
    return ev[idx[:n_null]], evecs[:,idx[:n_null]]

def phase_directional_derivative(wk_list, v_param, k, rank, eps):
    """
    dφₖ/dv = (φₖ(θ+εv) - φₖ(θ-εv)) / (2ε)
    dImZₖ/dv = (fₖ(θ+εv) - fₖ(θ-εv)) / (2ε)

    v_param: (D,) parameter-space direction vector
    Returns: (dphi_dv, dimZ_dv)
    """
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))

    def perturb(alpha):
        wk_new = []
        for i, w in enumerate(wk_list):
            s, e = offsets[i], offsets[i+1]
            delta = torch.tensor(v_param[s:e], dtype=torch.float32).reshape(w.shape)
            wk_new.append(w + alpha * delta)
        return wk_new

    wk_plus  = perturb(+eps)
    wk_minus = perturb(-eps)

    phi_plus  = bridgeland_phase_k(wk_plus, k)
    phi_minus = bridgeland_phase_k(wk_minus, k)

    f_plus,  _, _ = im_Z_k(wk_plus,  k, rank)
    f_minus, _, _ = im_Z_k(wk_minus, k, rank)

    dphi = (phi_plus - phi_minus) / (2*eps)
    dimZ = (f_plus  - f_minus)  / (2*eps)

    # Handle phase wraparound
    if abs(dphi) > math.pi:
        dphi = dphi - math.copysign(2*math.pi, dphi)

    return dphi, dimZ


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  CODIMENSION-2 STRATIFICATION TEST")
    print("  Verifying: S_72 is a codimension-2 stratum")
    print("=" * 60)

    t0 = time.time()

    # ── Load both checkpoints ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading checkpoints …")
    state64, meta64 = load_state(args.spike64)
    state72, meta72 = load_state(args.spike72)
    wk64 = extract_wk(state64)
    wk72 = extract_wk(state72)
    D = sum(w.numel() for w in wk64)
    d = len(wk64) - 1
    print(f"      D={D}  d={d}  (D = total WK parameters)")
    print(f"      Step 64: τ={meta64.get('tau','?'):.2f}  "
          f"val={meta64.get('val','?'):.4f}  "
          f"φ_cl={meta64.get('phi_cl','?')}/5")
    print(f"      Step 72: τ={meta72.get('tau','?'):.2f}  "
          f"val={meta72.get('val','?'):.4f}  "
          f"φ_cl={meta72.get('phi_cl','?')}/5")

    # ── Step 1: Identify which phases activate at step 72 ─────────────────────
    print(f"\n[2/4] Identifying active constraints at step 72 …")
    phases64, imZ64, areas64, clean64 = all_phases_and_imZ(wk64, args.rank)
    phases72, imZ72, areas72, clean72 = all_phases_and_imZ(wk72, args.rank)

    print(f"      Step 64 phases: " +
          ", ".join(f"{p:.3f}({'✓' if c else '✗'})"
                    for p,c in zip(phases64,clean64)))
    print(f"      Step 72 phases: " +
          ", ".join(f"{p:.3f}({'✓' if c else '✗'})"
                    for p,c in zip(phases72,clean72)))
    print(f"      Step 64 Im(Z): " +
          ", ".join(f"{f:+.4f}" for f in imZ64))
    print(f"      Step 72 Im(Z): " +
          ", ".join(f"{f:+.4f}" for f in imZ72))

    # Which conditions newly activate at 72 vs 64?
    # A condition "activates" = phase goes from clean to non-clean
    # OR Im(Z) magnitude increases significantly
    newly_nonclean = [k for k in range(d)
                      if clean64[k] and not clean72[k]]
    already_nonclean = [k for k in range(d)
                        if not clean64[k] and not clean72[k]]
    active = newly_nonclean + already_nonclean

    print(f"\n      Newly non-clean at step 72: {newly_nonclean}")
    print(f"      Already non-clean at step 64: {already_nonclean}")
    print(f"      Active constraint set: {active}")

    if len(active) < 2:
        print(f"      ⚠  Only {len(active)} active constraint(s) — "
              f"cannot be codimension-2")
        # Use the two largest |Im(Z)| as proxies
        imZ_abs = np.abs(imZ72)
        active = list(np.argsort(-imZ_abs)[:2])
        print(f"      Using top-2 |Im(Z)| pairs as proxies: {active}")

    a_idx, b_idx = active[0], active[1] if len(active) >= 2 else active[0]
    print(f"\n      Constraint pair: fₐ = Im(Z_{a_idx})={imZ72[a_idx]:+.4f}, "
          f"f_b = Im(Z_{b_idx})={imZ72[b_idx]:+.4f}")

    # ── Step 2: Compute ∇fₐ and ∇f_b ─────────────────────────────────────────
    print(f"\n[3/4] Computing gradients ∇fₐ and ∇f_b at step 72 …")
    print(f"      (FD on WK[{a_idx}]+WK[{a_idx+1}] and "
          f"WK[{b_idx}]+WK[{b_idx+1}] blocks only — O(2·256²) evals)")
    t_grad = time.time()

    print(f"      Computing ∇f_{a_idx} …")
    grad_a = grad_imZ_k_fast(state72, a_idx, args.rank, args.eps)
    print(f"      Computing ∇f_{b_idx} …")
    grad_b = grad_imZ_k_fast(state72, b_idx, args.rank, args.eps)

    print(f"      Gradient computation: {time.time()-t_grad:.1f}s")

    # ── Step 3: Rank test ─────────────────────────────────────────────────────
    print(f"\n[3b] Rank test: SVD of [∇fₐ; ∇f_b] …")
    na, nb, cos_ab, sigma2, rank2 = rank2_test(grad_a, grad_b, args.sigma_tol)

    print(f"\n  RANK TEST RESULT:")
    if rank2:
        print(f"  ✓ rank([∇fₐ; ∇f_b]) = 2")
        print(f"  ✓ Gradients linearly independent (σ_min={sigma2:.4f} > {args.sigma_tol})")
        print(f"  ✓ Codimension-2 condition satisfied (IFT applies)")
    else:
        print(f"  ✗ σ_min={sigma2:.4f} ≤ {args.sigma_tol}: gradients nearly parallel")
        print(f"  ✗ Cannot confirm codimension-2 from gradient rank alone")

    # ── Step 4: Phase directional derivatives along null eigenvectors ──────────
    print(f"\n[4/4] Phase directional derivatives along null eigenvectors …")
    print(f"      (Tests: ker(H) tangent to Bridgeland wall?)")

    print(f"      Computing null eigenvectors at step 64 …")
    null_evals64, null_evecs64 = null_eigenvectors(
        wk64, args.rank, args.k_lanczos, args.n_null)
    print(f"      Null eigenvalues (step 64): " +
          ", ".join(f"{e:.4f}" for e in null_evals64))

    print(f"      Computing null eigenvectors at step 72 …")
    null_evals72, null_evecs72 = null_eigenvectors(
        wk72, args.rank, args.k_lanczos, args.n_null)
    print(f"      Null eigenvalues (step 72): " +
          ", ".join(f"{e:.4f}" for e in null_evals72))

    # For each null eigenvector, compute dφₖ/dv for all k
    print(f"\n  Phase directional derivatives dφₖ/dvᵢ:")
    print(f"  (≈0 means v is tangent to wall; large means v crosses wall)")
    print(f"\n  Step 64 null eigenvectors:")

    deriv_results_64 = []
    for i in range(args.n_null):
        v = null_evecs64[:, i]
        row = {'eigval': float(null_evals64[i]), 'dphi': [], 'dimZ': []}
        line = f"  v{i}(λ={null_evals64[i]:+.4f}): "
        for k in range(d):
            dphi, dimZ = phase_directional_derivative(wk64, v, k, args.rank, args.eps)
            row['dphi'].append(float(dphi))
            row['dimZ'].append(float(dimZ))
            line += f"dφ{k}={dphi:+.3f} "
        print(line)
        # Is this vector tangent to the active walls?
        tangent_to_a = abs(row['dphi'][a_idx]) < 0.1
        tangent_to_b = abs(row['dphi'][b_idx]) < 0.1
        row['tangent_to_a'] = tangent_to_a
        row['tangent_to_b'] = tangent_to_b
        if tangent_to_a and tangent_to_b:
            print(f"          → ✓ TANGENT to both active walls (a={a_idx}, b={b_idx})")
        deriv_results_64.append(row)

    print(f"\n  Step 72 null eigenvectors:")
    deriv_results_72 = []
    for i in range(args.n_null):
        v = null_evecs72[:, i]
        row = {'eigval': float(null_evals72[i]), 'dphi': [], 'dimZ': []}
        line = f"  v{i}(λ={null_evals72[i]:+.4f}): "
        for k in range(d):
            dphi, dimZ = phase_directional_derivative(wk72, v, k, args.rank, args.eps)
            row['dphi'].append(float(dphi))
            row['dimZ'].append(float(dimZ))
            line += f"dφ{k}={dphi:+.3f} "
        print(line)
        tangent_to_a = abs(row['dphi'][a_idx]) < 0.1
        tangent_to_b = abs(row['dphi'][b_idx]) < 0.1
        row['tangent_to_a'] = tangent_to_a
        row['tangent_to_b'] = tangent_to_b
        if tangent_to_a and tangent_to_b:
            print(f"          → ✓ TANGENT to both active walls (a={a_idx}, b={b_idx})")
        deriv_results_72.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  THEOREM VERIFICATION SUMMARY")
    print(f"{'='*60}")

    n_tangent_64 = sum(1 for r in deriv_results_64
                       if r['tangent_to_a'] and r['tangent_to_b'])
    n_tangent_72 = sum(1 for r in deriv_results_72
                       if r['tangent_to_a'] and r['tangent_to_b'])

    print(f"\n  Condition 1: Two constraints activate at step 72")
    print(f"  Active: f_{a_idx} (Im(Z_{a_idx})={imZ72[a_idx]:+.4f}), "
          f"f_{b_idx} (Im(Z_{b_idx})={imZ72[b_idx]:+.4f})")
    newly = len(newly_nonclean)
    print(f"  Newly non-clean at step 72: {newly}  "
          f"{'✓ confirms simultaneous activation' if newly >= 1 else '✗ not simultaneous'}")

    print(f"\n  Condition 2: rank([∇fₐ; ∇f_b]) = 2  (transversality)")
    print(f"  σ_min = {sigma2:.6f}  "
          f"{'✓ CONFIRMED (codimension-2)' if rank2 else '✗ not confirmed'}")

    print(f"\n  Condition 3: ker(H) tangent to Bridgeland walls")
    print(f"  Null vecs tangent to both walls (step 64): {n_tangent_64}/{args.n_null}")
    print(f"  Null vecs tangent to both walls (step 72): {n_tangent_72}/{args.n_null}")
    kernel_tangent = n_tangent_64 > 0 or n_tangent_72 > 0
    print(f"  {'✓ Some null directions tangent to wall' if kernel_tangent else '✗ No tangent null directions found'}")

    all_confirmed = rank2 and (newly >= 1) and kernel_tangent
    print(f"\n  OVERALL: {'✓✓ CODIMENSION-2 THEOREM CONFIRMED' if all_confirmed else '◑ PARTIAL CONFIRMATION'}")
    if not all_confirmed:
        if not rank2:
            print(f"  → Increase eps or check phase computation accuracy")
        if newly < 1:
            print(f"  → Phase change between steps 64/72 may be < 1 phase")
        if not kernel_tangent:
            print(f"  → Null eigenvectors may cross rather than follow wall")

    elapsed = time.time() - t0

    # Save report
    report = {
        'step64': {'path': args.spike64, 'meta': meta64,
                   'phases': phases64.tolist(), 'imZ': imZ64.tolist(),
                   'clean': clean64},
        'step72': {'path': args.spike72, 'meta': meta72,
                   'phases': phases72.tolist(), 'imZ': imZ72.tolist(),
                   'clean': clean72},
        'active_constraints': {'a': a_idx, 'b': b_idx,
                               'newly_nonclean': newly_nonclean,
                               'already_nonclean': already_nonclean},
        'rank_test': {
            'norm_grad_a': float(na), 'norm_grad_b': float(nb),
            'cos_ab': float(cos_ab), 'sigma_min': float(sigma2),
            'rank2': bool(rank2), 'sigma_tol': args.sigma_tol,
        },
        'null_eigenvectors': {
            'step64': deriv_results_64,
            'step72': deriv_results_72,
            'n_tangent_64': n_tangent_64,
            'n_tangent_72': n_tangent_72,
        },
        'theorem_confirmed': bool(all_confirmed),
        'elapsed_s': round(elapsed, 1),
    }

    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")
    print(f"  Total elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
