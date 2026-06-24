"""
p3_wall_crossing.py
====================
Experiment P3: Wall Crossing = Gradient Singularity

Tests Proposition 22.3 from the paper:

    ker(H_θ) ≅ { v ∈ T_θM : Z(π(v)) = 0 }

i.e. flat directions of the Hessian at τ-spike (compiler stall) points
are exactly the directions where the central charge Z(γ) → 0.

The central charge for a class γ in the strip-angle space is:
    Z(γ_k) = A(L_k, L_{k+1}) · exp(i·φ_k)
where φ_k = arg(λ_1(W_K^{k+1} W_K^{k,-1})) ∈ {0, π} in the Bridgeland orbit.

Success criterion (Proposition 22.3):
    null eigenvectors of H at τ-spike points satisfy |Z(π(v))| < 0.05

TWO-PART EXPERIMENT:

  Part A — τ-spike checkpoint logger (patch for compiler_geometric.py):
    Saves model state whenever τ spikes (rises > τ_spike_threshold in
    one 8-step window). Run this during a compiler training run.

  Part B — Analysis script:
    Given one or more τ-spike checkpoints, computes:
    1. Near-null eigenvectors of H (via Lanczos with negative shift)
    2. Central charge Z(π(v)) for each null eigenvector v
    3. |Z(π(v))| — the P3 success criterion
    4. Bridgeland phase φ_k at the spike point
    5. τ value and Φ_cl at the spike

Usage
-----
  # Part B (analysis) — after saving tau-spike checkpoints:
  python p3_wall_crossing.py \\
      --checkpoints tau_spike_0.pt tau_spike_1.pt \\
      --rank 6 --k_lanczos 20 --n_null 3 \\
      --output p3_report.json

  # Part A (logger) — integrated snippet for compiler_geometric.py:
  # See TauSpikeLogger class below; instantiate and call .check() at each step.

Dependencies: torch, numpy (standard PyTorch env)
"""

import argparse
import json
import math
import time
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


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment P3: Wall Crossing = Gradient Singularity")
    p.add_argument("--checkpoints", nargs="+", default=["basin_state.pt"],
                   help="τ-spike checkpoint files (or any checkpoint for baseline)")
    p.add_argument("--rank",      type=int,   default=6)
    p.add_argument("--k_lanczos", type=int,   default=30,
                   help="Lanczos iterations (more = better null-space resolution)")
    p.add_argument("--n_null",    type=int,   default=3,
                   help="Number of near-null eigenvectors to examine")
    p.add_argument("--null_threshold", type=float, default=0.05,
                   help="Eigenvalue threshold for 'near-null' (|λ| < threshold * |λ_max|)")
    p.add_argument("--Z_threshold", type=float, default=0.05,
                   help="P3 success: |Z(π(v))| < Z_threshold")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--output",    default="p3_report.json")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# PART A: TauSpikeLogger — paste into compiler_geometric.py
# ═══════════════════════════════════════════════════════════════════════════

class TauSpikeLogger:
    """
    Monitors τ = ||∇_FF L|| / ||∇_Emb L|| during compiler training.
    Saves a checkpoint whenever τ spikes (rises > threshold in one window).

    Usage in compiler_geometric.py:
        logger = TauSpikeLogger(model, save_dir="tau_spikes", threshold=1.4)
        # In training loop, after computing tau:
        logger.check(step, tau, val, phi_cl, model_state_dict)

    This generates: tau_spikes/tau_spike_step{N}_tau{T:.2f}.pt
    Each file is a standard PyTorch state dict + metadata dict.
    """
    def __init__(self, save_dir="tau_spikes", threshold=1.4, window=8,
                 max_spikes=10):
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.threshold = threshold   # τ must rise by this factor
        self.window    = window      # steps per window
        self.max_spikes = max_spikes
        self.tau_history = []
        self.spike_count = 0
        self.last_window_max = None

    def check(self, step, tau, val, phi_cl, state_dict,
              grad_ff=None, grad_emb=None):
        """
        Call at each training step after computing tau.
        Saves checkpoint if tau spikes within current window.
        """
        self.tau_history.append((step, tau))

        # Check at end of each window
        if len(self.tau_history) >= self.window:
            window_taus = [t for _, t in self.tau_history[-self.window:]]
            window_max  = max(window_taus)
            window_min  = min(window_taus)

            # Spike: max/min ratio exceeds threshold within window
            spiked = (window_max / (window_min + 1e-10) > self.threshold
                      and window_min < window_max)  # rising, not just high

            if spiked and self.spike_count < self.max_spikes:
                fname = (self.save_dir /
                         f"tau_spike_step{step:04d}_tau{tau:.2f}.pt")
                metadata = {
                    "step":       step,
                    "tau":        float(tau),
                    "val":        float(val),
                    "phi_cl":     int(phi_cl),
                    "window_max": float(window_max),
                    "window_min": float(window_min),
                    "spike_ratio": float(window_max / (window_min + 1e-10)),
                    "tau_history": [(int(s), float(t))
                                    for s, t in self.tau_history[-20:]],
                }
                torch.save({"state_dict": state_dict, "metadata": metadata},
                           str(fname))
                print(f"  [TauSpikeLogger] Saved: {fname.name}  "
                      f"τ={tau:.2f}  val={val:.4f}  φ_cl={phi_cl}/5")
                self.spike_count += 1

        return spiked if len(self.tau_history) >= self.window else False

    @staticmethod
    def load_spike_checkpoint(path):
        """Load a spike checkpoint: returns (state_dict, metadata)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "metadata" in ckpt:
            return ckpt["state_dict"], ckpt["metadata"]
        return ckpt, {}   # fallback for plain checkpoints


# ═══════════════════════════════════════════════════════════════════════════
# PART B: Analysis — null eigenvectors + central charge
# ═══════════════════════════════════════════════════════════════════════════

# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "metadata" in ckpt:
            meta = ckpt.get("metadata", {})
            state = ckpt.get("state_dict", ckpt)
        else:
            meta = {}
            for key in ("model", "state_dict", "model_state_dict"):
                if key in ckpt:
                    state = ckpt[key]; break
            else:
                state = ckpt
    else:
        state, meta = ckpt, {}
    return state, meta

def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2:
            continue
        n = name.lower()
        if "c_attn" in n and "weight" in n:
            try:
                li = int([p for p in name.split(".") if p.isdigit()][0])
            except (IndexError, ValueError):
                li = len(wk)
            D = tensor.shape[0] // 3 if tensor.shape[0] % 3 == 0 else tensor.shape[1] // 3
            wk[li] = tensor[D:2*D, :] if tensor.shape[0] % 3 == 0 else tensor[:, D:2*D].T
        elif ("key" in n or "wk" in n or "w_k" in n) and "weight" in n:
            try:
                li = int([p for p in name.split(".") if p.isdigit()][0])
            except (IndexError, ValueError):
                li = len(wk)
            wk[li] = tensor if tensor.ndim == 2 else tensor.squeeze()
    if not wk:
        raise RuntimeError("No WK. Keys: " + ", ".join(list(state.keys())[:20]))
    return [wk[i].detach().float() for i in sorted(wk)]


# ─── Strip areas, phases, and central charge ─────────────────────────────────

def strip_area(Wk, Wk1, rank):
    Uk  = torch.linalg.svd(Wk.detach().float(),  full_matrices=False)[0][:, :rank]
    Uk1 = torch.linalg.svd(Wk1.detach().float(), full_matrices=False)[0][:, :rank]
    sv  = torch.linalg.svdvals(Uk.T @ Uk1).clamp(-1+1e-6, 1-1e-6)
    return torch.arccos(sv).sum().item()

def bridgeland_phase(Wk, Wk1):
    """
    φ_k = arg(λ_1(W_K^{k+1} W_K^{k,-1}))
    In the Bridgeland orbit: φ_k ∈ {0, π}.
    Returns the phase in [0, 2π) and whether it's clean (|φ - 0| < 0.3 or |φ - π| < 0.3).
    """
    Wk_f  = Wk.detach().float()
    Wk1_f = Wk1.detach().float()
    # Use pseudo-inverse for stability
    M = Wk1_f @ torch.linalg.pinv(Wk_f)
    evals = torch.linalg.eigvals(M)
    # Take eigenvalue with largest |real part|
    dominant = evals[torch.abs(evals.real).argmax()]
    phase = float(torch.atan2(dominant.imag, dominant.real).item())
    if phase < 0:
        phase += 2 * math.pi
    clean = (abs(phase) < 0.3 or abs(phase - math.pi) < 0.3
             or abs(phase - 2*math.pi) < 0.3)
    return phase, clean

def central_charge(area, phase):
    """
    Z(γ_k) = A(L_k, L_{k+1}) · exp(i·φ_k)
    |Z(γ_k)| = A(L_k, L_{k+1})  (the strip area)
    Im(Z(γ_k)) = A · sin(φ_k)

    P3 hypothesis: at wall crossings, Im(Z) → 0, i.e. φ_k → {0, π}.
    The central charge of a null eigenvector direction is:
    Z(π(v)) = Σ_k  v_k · Z(γ_k)  (sum over strip-angle components)
    where v_k = (π v)_k is the strip-angle projection.
    """
    z_real = area * math.cos(phase)
    z_imag = area * math.sin(phase)
    z_abs  = area  # |Z| = area always
    return complex(z_real, z_imag), z_abs, z_imag

def compute_strip_data(wk_list, rank):
    """Compute strip areas, Bridgeland phases, and central charges."""
    d = len(wk_list) - 1
    areas, phases, clean_flags, Z_vals = [], [], [], []
    for k in range(d):
        A    = strip_area(wk_list[k], wk_list[k+1], rank)
        phi, clean = bridgeland_phase(wk_list[k], wk_list[k+1])
        Z, Z_abs, Z_imag = central_charge(A, phi)
        areas.append(A)
        phases.append(phi)
        clean_flags.append(clean)
        Z_vals.append(Z)
    phi_cl = sum(clean_flags)
    return np.array(areas), np.array(phases), clean_flags, Z_vals, phi_cl


# ─── Strip-angle projection π ────────────────────────────────────────────────

def build_basis(wk_list, rank):
    shapes  = [w.shape for w in wk_list]
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0] + splits))
    param_dim = sum(splits)
    d = len(wk_list) - 1
    cols = []
    for k in range(d):
        Wk  = wk_list[k].detach().float()
        Wk1 = wk_list[k+1].detach().float()
        Uk,  _,  Vhk  = torch.linalg.svd(Wk,  full_matrices=False)
        Uk1, _,  Vhk1 = torch.linalg.svd(Wk1, full_matrices=False)
        Uk, Uk1 = Uk[:, :rank], Uk1[:, :rank]
        M = Uk.T @ Uk1
        Um, Sm, Vhm = torch.linalg.svd(M)
        Sm = Sm.clamp(1e-6, 1 - 1e-6)
        col = np.zeros(param_dim)
        for r in range(rank):
            sc = -1.0 / math.sqrt(max(1.0 - Sm[r].item()**2, 1e-8))
            lk  = (Uk  @ Um[:, r:r+1]).reshape(-1).detach().numpy()
            rk  = (Vhm[r:r+1, :] @ Vhk1[:rank, :]).reshape(-1).detach().numpy()
            col[offsets[k]:offsets[k+1]] += sc * (
                lk[:, None] * rk[None, :]).reshape(-1)[:offsets[k+1]-offsets[k]]
            lk1 = (Uk1 @ Vhm[r:r+1, :].T).reshape(-1).detach().numpy()
            rk1 = (Um[:, r:r+1].T @ Vhk[:rank, :]).reshape(-1).detach().numpy()
            col[offsets[k+1]:offsets[k+2]] += sc * (
                lk1[:, None] * rk1[None, :]).reshape(-1)[:offsets[k+2]-offsets[k+1]]
        nm = np.linalg.norm(col)
        cols.append(col / nm if nm > 1e-10 else col)
    return np.stack(cols, axis=1)   # (param_dim, d)


# ─── Symplectic proxy + shifted Lanczos ──────────────────────────────────────

class SymplecticProxy(nn.Module):
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank   = rank
        self.theta  = nn.Parameter(
            torch.cat([w.float().reshape(-1) for w in wk_list]).clone())
        self.shapes = [w.shape for w in wk_list]
        self.splits = [w.numel() for w in wk_list]

    def wk_matrices(self):
        return [p.reshape(s) for p, s in
                zip(torch.split(self.theta, self.splits), self.shapes)]

    def forward(self):
        wks = self.wk_matrices()
        loss = torch.tensor(0.0)
        for k in range(len(wks) - 1):
            Ua = torch.linalg.svd(wks[k],     full_matrices=False)[0][:, :self.rank]
            Ub = torch.linalg.svd(wks[k + 1], full_matrices=False)[0][:, :self.rank]
            sv = torch.linalg.svdvals(Ua.T @ Ub).clamp(1e-6, 1 - 1e-6)
            loss = loss + torch.arccos(sv).sum()
        return loss

def hvp(model, v):
    loss = model()
    grad = torch.autograd.grad(loss, model.theta, create_graph=True)[0]
    Hv   = torch.autograd.grad((grad * v).sum(), model.theta)[0]
    return Hv.detach()

def lanczos_shifted(model, k_iters, shift=0.0, seed=42):
    """
    Lanczos on (H - shift·I).
    shift > 0 targets eigenvalues near +shift (near-zero of H if shift≈0).
    shift = λ_min brings the smallest eigenvalue closest to zero in the
    shifted operator, making it the dominant Lanczos direction.

    Returns: (eigvals of H, eigvecs, Hk, alpha, beta)
    Note: eigvals are of H (unshifted), sorted by |λ - shift| ascending
    (closest to shift = most null-like).
    """
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k_iters + 1)
    alpha_c = torch.zeros(k_iters)
    beta_c  = torch.zeros(k_iters - 1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:, 0] = v
    prev_b = 0.0

    for j in range(k_iters):
        # HVP with shift: (H - shift·I)v = Hv - shift·v
        w = hvp(model, V[:, j]) - shift * V[:, j]
        if j > 0:
            w = w - prev_b * V[:, j - 1]
        a = (w * V[:, j]).sum().item()
        alpha_c[j] = a
        w = w - a * V[:, j]
        for i in range(j + 1):
            w = w - (w @ V[:, i]) * V[:, i]
        if j < k_iters - 1:
            b = w.norm().item()
            if b < 1e-10:
                k_iters = j + 1; break
            prev_b = b; beta_c[j] = b
            V[:, j + 1] = w / b

    k = k_iters
    Hk = (np.diag(alpha_c[:k].numpy()) +
          np.diag(beta_c[:k-1].numpy(),  1) +
          np.diag(beta_c[:k-1].numpy(), -1))
    # Eigenvalues of shifted operator; add shift back for H eigenvalues
    evals_shifted, ritz = np.linalg.eigh(Hk)
    evals_H = evals_shifted + shift    # eigenvalues of H
    evecs   = V[:, :k].numpy() @ ritz

    # Sort by |λ_H - shift| ascending = most null-like first
    idx = np.argsort(np.abs(evals_H - shift))
    return evals_H[idx], evecs[:, idx], Hk, alpha_c[:k].numpy(), beta_c[:k-1].numpy()


# ─── P3 core: central charge of null eigenvectors ────────────────────────────

def compute_Z_for_eigvec(v_param, basis, Z_vals):
    """
    Z(π(v)) = Σ_k  (π v)_k · Z(γ_k)
    where (π v)_k = <v, basis[:,k]> is the strip-angle projection.

    Returns: Z_total (complex), |Z_total|, and per-component contributions.
    """
    pi_v = basis.T @ v_param              # (d,) strip-angle coordinates
    Z_total = sum(float(pi_v[k]) * Z_vals[k] for k in range(len(Z_vals)))
    components = [{"k": k, "pi_v_k": float(pi_v[k]),
                   "Z_k_abs": abs(Z_vals[k]),
                   "Z_k_contribution": abs(float(pi_v[k]) * Z_vals[k])}
                  for k in range(len(Z_vals))]
    return Z_total, abs(Z_total), components

def find_null_eigenvectors(model, basis, Z_vals, k_lanczos, n_null,
                           null_threshold, Z_threshold, seed):
    """
    Find near-null eigenvectors of H and check |Z(π(v))| for each.

    Strategy:
    1. Run standard Lanczos to find the full eigenspectrum.
    2. Identify eigenvalues closest to zero (|λ| < null_threshold * |λ_max|).
    3. For each near-null eigenvector, compute |Z(π(v))|.
    4. Check P3 criterion: |Z(π(v))| < Z_threshold.
    """
    # Standard Lanczos (no shift) for spectrum overview
    evals_std, evecs_std, Hk, alpha_c, beta_c = lanczos_shifted(
        model, k_lanczos, shift=0.0, seed=seed)

    lambda_max = float(np.max(np.abs(evals_std)))
    threshold_abs = null_threshold * lambda_max

    print(f"  Top-5 eigenvalues: " +
          ", ".join(f"{e:.4f}" for e in evals_std[:5]))
    print(f"  |λ_max| = {lambda_max:.4f}")
    print(f"  Null threshold: |λ| < {threshold_abs:.4f}")

    # Find near-null eigenvectors (closest to zero)
    null_idx = np.argsort(np.abs(evals_std))[:n_null]
    null_evals = evals_std[null_idx]
    null_evecs = evecs_std[:, null_idx]

    print(f"  Near-null eigenvalues: " +
          ", ".join(f"{e:.6f}" for e in null_evals))

    results = []
    for i in range(n_null):
        v = null_evecs[:, i]
        eig = float(null_evals[i])
        is_null = abs(eig) < threshold_abs

        Z_total, Z_abs, Z_components = compute_Z_for_eigvec(v, basis, Z_vals)
        p3_satisfied = Z_abs < Z_threshold

        # Also compute the strip-angle projection for inspection
        pi_v = basis.T @ v
        pi_v_norm = float(np.linalg.norm(pi_v))

        result = {
            "eigvec_idx":    i,
            "eigenvalue":    eig,
            "is_near_null":  bool(is_null),
            "Z_abs":         float(Z_abs),
            "Z_real":        float(Z_total.real),
            "Z_imag":        float(Z_total.imag),
            "p3_satisfied":  bool(p3_satisfied),
            "pi_v_norm":     float(pi_v_norm),
            "pi_v":          pi_v.tolist(),
            "Z_components":  Z_components,
        }
        results.append(result)

        status = "✓ P3" if p3_satisfied else "✗"
        null_flag = "NULL" if is_null else "non-null"
        print(f"  v[{i}]: λ={eig:+.6f} ({null_flag})  "
              f"|Z(πv)|={Z_abs:.4f}  {status}")

    return results, evals_std, Hk, alpha_c, beta_c


# ─── τ value from gradients (if available, else estimate from strips) ─────────

def estimate_tau_from_strips(areas):
    """
    τ = ||∇_FF L|| / ||∇_Emb L|| measures K0 gluing defect.
    In the absence of actual gradients, estimate from strip-area variance:
    high variance → high τ (FF gradients dominate).
    Returns a rough estimate; actual τ from the compiler is more accurate.
    """
    area_std = float(np.std(areas))
    area_mean = float(np.mean(areas))
    # Empirical calibration from compiler logs:
    # std=0.036 → τ≈5 (search regime); std>0.5 → τ≈1.5 (floor regime)
    # Linear interpolation in log space
    if area_std < 0.01:
        return 5.0
    return max(1.5, 5.0 * (0.036 / area_std) ** 0.5)


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_checkpoint(ckpt_path, args):
    print(f"\n{'─'*60}")
    print(f"  Checkpoint: {ckpt_path}")

    state, meta = load_checkpoint(ckpt_path, args.device)
    wk_list = extract_wk(state)
    D = wk_list[0].shape[0]
    d = len(wk_list) - 1

    # Strip data
    areas, phases, clean_flags, Z_vals, phi_cl = compute_strip_data(
        wk_list, args.rank)
    area_std = float(np.std(areas))
    tau_est  = estimate_tau_from_strips(areas)

    # Metadata from logger (if available)
    tau_actual = meta.get("tau", tau_est)
    step       = meta.get("step", "unknown")
    val        = meta.get("val",  "unknown")

    print(f"  Step: {step}  val: {val}  τ: {tau_actual:.2f}  φ_cl: {phi_cl}/5")
    print(f"  Strip areas: " + ", ".join(f"{a:.4f}" for a in areas))
    print(f"  Bridgeland phases (rad): " + ", ".join(f"{p:.3f}" for p in phases))
    print(f"  Clean phases: {sum(clean_flags)}/{d}  "
          f"({'IN ORBIT' if phi_cl >= 4 else 'off orbit'})")
    print(f"  Central charges |Z(γ_k)| = strip areas: " +
          ", ".join(f"{abs(z):.4f}" for z in Z_vals))
    print(f"  Im(Z(γ_k)): " +
          ", ".join(f"{z.imag:.4f}" for z in Z_vals))

    # Basis
    basis = build_basis(wk_list, args.rank)

    # Proxy model + null eigenvectors
    model  = SymplecticProxy(wk_list, args.rank)
    null_results, evals, Hk, alpha_c, beta_c = find_null_eigenvectors(
        model, basis, Z_vals, args.k_lanczos, args.n_null,
        args.null_threshold, args.Z_threshold, args.seed)

    # P3 verdict for this checkpoint
    n_satisfied = sum(r["p3_satisfied"] for r in null_results)
    n_null_vecs = sum(r["is_near_null"] for r in null_results)

    if n_null_vecs == 0:
        verdict = "NO_NULL_EIGENVECTORS"
        msg = ("No near-null eigenvalues found. "
               "Hessian is far from singular at this checkpoint. "
               "P3 requires a checkpoint at an actual τ-spike.")
    elif n_satisfied == n_null_vecs:
        verdict = "P3_CONFIRMED"
        msg = (f"All {n_null_vecs} near-null eigenvectors satisfy |Z(πv)| < {args.Z_threshold}. "
               "Flat directions of H coincide with zero-central-charge directions. "
               "Proposition 22.3 holds at this checkpoint.")
    elif n_satisfied > 0:
        verdict = "P3_PARTIAL"
        msg = (f"{n_satisfied}/{n_null_vecs} near-null eigenvectors satisfy P3 criterion. "
               f"Partial confirmation. τ-spike checkpoint would show stronger signal.")
    else:
        verdict = "P3_NOT_SATISFIED"
        msg = ("Near-null eigenvectors exist but |Z(πv)| > threshold. "
               "Either not a τ-spike checkpoint, or P3 hypothesis does not hold here.")

    print(f"\n  P3 VERDICT: {verdict}")
    print(f"  {msg}")

    return {
        "checkpoint":      str(ckpt_path),
        "metadata":        meta,
        "step":            step,
        "tau":             float(tau_actual),
        "val":             val,
        "phi_cl":          int(phi_cl),
        "strip_areas":     areas.tolist(),
        "strip_area_std":  float(area_std),
        "phases_rad":      phases.tolist(),
        "clean_phases":    clean_flags,
        "Z_vals_real":     [z.real for z in Z_vals],
        "Z_vals_imag":     [z.imag for z in Z_vals],
        "Z_vals_abs":      [abs(z) for z in Z_vals],
        "top_k_eigvals":   evals[:10].tolist(),
        "Hk_diagonal":     alpha_c.tolist(),
        "Hk_offdiagonal":  beta_c.tolist(),
        "null_eigvectors": null_results,
        "n_null":          int(n_null_vecs),
        "n_p3_satisfied":  int(n_satisfied),
        "verdict":         verdict,
        "message":         msg,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  EXPERIMENT P3: WALL CROSSING = GRADIENT SINGULARITY")
    print("  Tests: ker(H_θ) ≅ {v : Z(π(v)) = 0}")
    print("=" * 60)
    print(f"  Checkpoints: {args.checkpoints}")
    print(f"  rank={args.rank}  k_lanczos={args.k_lanczos}  n_null={args.n_null}")
    print(f"  null_threshold={args.null_threshold}  Z_threshold={args.Z_threshold}")

    if len(args.checkpoints) == 1 and args.checkpoints[0] == "basin_state.pt":
        print()
        print("  NOTE: Running on basin_state.pt (not a τ-spike checkpoint).")
        print("  This provides a BASELINE comparison, not the definitive P3 test.")
        print("  For P3: save τ-spike checkpoints using TauSpikeLogger, then:")
        print("  python p3_wall_crossing.py --checkpoints tau_spikes/tau_spike_*.pt")
        print()

    t0 = time.time()
    all_results = []
    for ckpt in args.checkpoints:
        result = analyze_checkpoint(ckpt, args)
        all_results.append(result)

    # Cross-checkpoint summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ACROSS {len(all_results)} CHECKPOINT(S)")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {Path(r['checkpoint']).name:30s}  "
              f"τ={r['tau']:.2f}  φ_cl={r['phi_cl']}/5  "
              f"P3: {r['verdict']}")

    # Overall verdict
    n_confirmed = sum(1 for r in all_results if r["verdict"] == "P3_CONFIRMED")
    if n_confirmed == len(all_results):
        overall = "P3_CONFIRMED_ALL"
    elif n_confirmed > 0:
        overall = f"P3_CONFIRMED_{n_confirmed}_OF_{len(all_results)}"
    else:
        any_partial = any(r["verdict"] == "P3_PARTIAL" for r in all_results)
        overall = "P3_PARTIAL" if any_partial else "P3_BASELINE_ONLY"

    print(f"\n  Overall: {overall}")

    # Next steps
    if overall == "P3_BASELINE_ONLY":
        print()
        print("  Next step: Add TauSpikeLogger to compiler_geometric.py")
        print("  and run the compiler to collect τ-spike checkpoints.")
        print("  The logger is defined in this script — copy the class")
        print("  into compiler_geometric.py and call logger.check() in")
        print("  the basin settle loop where τ is already computed.")

    # Write report
    report = {
        "experiment":     "P3 Wall Crossing = Gradient Singularity",
        "config": {
            "rank": args.rank, "k_lanczos": args.k_lanczos,
            "n_null": args.n_null, "null_threshold": args.null_threshold,
            "Z_threshold": args.Z_threshold,
        },
        "checkpoints":    all_results,
        "overall_verdict": overall,
        "elapsed_s":      round(time.time() - t0, 1),
        "tau_spike_logger_instructions": (
            "Add TauSpikeLogger to compiler_geometric.py. "
            "Instantiate before basin settle loop: "
            "logger = TauSpikeLogger(save_dir='tau_spikes', threshold=1.4). "
            "Call logger.check(step, tau, val, phi_cl, model.state_dict()) "
            "after each 8-step interval where tau is computed. "
            "Then re-run: python p3_wall_crossing.py "
            "--checkpoints tau_spikes/tau_spike_*.pt"
        ),
    }

    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {Path(args.output).resolve()}")
    print(f"  Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
