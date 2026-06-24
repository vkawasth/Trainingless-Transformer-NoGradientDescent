"""
tropicalization_lr_sweep.py
============================
Experiment P4: Tropicalization Bridge (Conjecture 21.1 / Theorem 22.1)

Tests whether cluster c-vectors are the logarithmic limit of Hessian
curvature directions as learning rate η → 0:

    lim_{η→0} val( v_i(η) ) = c_i  ∈ Z^d

where val(x) = -log|x| is the non-Archimedean valuation (tropicalization),
v_i(η) are the top-d eigenvectors of the projected Hessian Ĥ_η = π H_η π^T,
π: R^D → R^d is the strip-angle projection, and c_i are the BMRR c-vectors.

Three sub-experiments (all from Section 22.4 of the paper):

  P4a  LR sweep: plot |cos|(Ĥ_η-eigvecs, c-vecs) vs log10(η)
       Prediction: monotone increase toward 1.0 as η → 0

  P4b  Pentagon tropicalization: apply val = -log|·| to m2 values,
       check whether result satisfies tropical exchange relation

  P4c  c-vector integer check: at η=1e-4, check val(v_i(η)) ∈ Z^d
       Prediction: rounded to nearest integer within tolerance

Usage
-----
  python tropicalization_lr_sweep.py \\
      --checkpoint basin_state.pt \\
      --lrs 1e-1 1e-2 1e-3 1e-4 \\
      --rank 6 --k_lanczos 20 --n_steps 1 \\
      --output tropicalization_report.json

Dependencies: torch, numpy  (standard PyTorch env)
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
    """Handles numpy scalars, bools, arrays — required for Python 3.14+."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tropicalization LR sweep — Experiment P4")
    p.add_argument("--checkpoint", default="basin_state.pt")
    p.add_argument("--lrs", nargs="+", type=float,
                   default=[1e-1, 1e-2, 1e-3, 1e-4],
                   help="Learning rates for LR sweep (decreasing)")
    p.add_argument("--rank",       type=int, default=6,
                   help="Bridgeland / principal-angle rank")
    p.add_argument("--k_lanczos",  type=int, default=20,
                   help="Lanczos iterations")
    p.add_argument("--n_steps",    type=int, default=1,
                   help="Gradient steps taken at each LR to update Hη")
    p.add_argument("--integrality_tol", type=float, default=0.25,
                   help="Tolerance for integer check on val(v_i)")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="tropicalization_report.json")
    return p.parse_args()

# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_state(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt

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
        raise RuntimeError("No WK matrices found. Keys: " + ", ".join(list(state.keys())[:20]))
    return [wk[i] for i in sorted(wk)]

# ─── Exchange matrix + BMRR c-vectors ────────────────────────────────────────

def principal_angles(A, B, dim):
    Ua = torch.linalg.svd(A.detach().float(), full_matrices=False)[0][:, :dim]
    Ub = torch.linalg.svd(B.detach().float(), full_matrices=False)[0][:, :dim]
    sv = torch.linalg.svdvals(Ua.T @ Ub).clamp(-1, 1)
    return torch.arccos(sv)

def strip_area(wk_list, k, rank):
    return principal_angles(wk_list[k], wk_list[k+1], rank).sum().item()

def build_exchange_matrix(wk_list, rank):
    d = len(wk_list) - 1
    areas = [strip_area(wk_list, k, rank) for k in range(d)]
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            B[i, j] = np.sign(areas[i] - areas[j]) * round(abs(areas[i] - areas[j]), 6)
    return B, np.array(areas)

def bmrr_mutate_B(B, k):
    d, B2 = B.shape[0], B.copy()
    for i in range(d):
        for j in range(d):
            B2[i,j] = (-B[i,j] if (i==k or j==k) else
                       B[i,j] + max(B[i,k],0)*max(B[k,j],0) - min(B[i,k],0)*min(B[k,j],0))
    return B2

def bmrr_mutate_c(C, B, k):
    d, C2 = C.shape[0], C.copy()
    C2[:, k] = [-C[i,k] if i==k else
                C[i,k] + max(B[i,k],0)*max(C[i,k],0) - min(B[i,k],0)*min(C[i,k],0)
                for i in range(d)]
    return C2

def compute_c_vectors(B_init):
    d, B, C = B_init.shape[0], B_init.copy(), np.eye(B_init.shape[0])
    for k in range(d):
        B = bmrr_mutate_B(B, k)
        C = bmrr_mutate_c(C, B, k)
    return C, B

# ─── Strip-angle projection π: R^D → R^d ─────────────────────────────────────
# Each coordinate π(v)_k = ⟨v, ∂A(Lk,Lk+1)/∂θ⟩
# The gradient of the strip area w.r.t. WK parameters via SVD chain rule.

def build_strip_angle_basis(wk_list, rank):
    """
    Returns basis (param_dim, d) where column k is the unit-norm gradient
    of strip area A(L_k, L_{k+1}) with respect to the full parameter vector.
    This is the projection matrix π^T.
    """
    shapes = [w.shape for w in wk_list]
    splits = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0] + splits))
    param_dim = sum(splits)
    d = len(wk_list) - 1

    cols = []
    for k in range(d):
        # detach() ensures no autograd graph — safe after gradient steps
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
            scale = -1.0 / math.sqrt(max(1.0 - Sm[r].item()**2, 1e-8))
            lk  = (Uk  @ Um[:, r:r+1]).reshape(-1).detach().numpy()
            rk  = (Vhm[r:r+1, :] @ Vhk1[:rank, :]).reshape(-1).detach().numpy()
            block_k = (lk[:, None] * rk[None, :]).reshape(-1)
            s, e = offsets[k], offsets[k+1]
            col[s:e] += scale * block_k[:e-s]

            lk1 = (Uk1 @ Vhm[r:r+1, :].T).reshape(-1).detach().numpy()
            rk1 = (Um[:, r:r+1].T @ Vhk[:rank, :]).reshape(-1).detach().numpy()
            block_k1 = (lk1[:, None] * rk1[None, :]).reshape(-1)
            s1, e1 = offsets[k+1], offsets[k+2]
            col[s1:e1] += scale * block_k1[:e1-s1]

        norm = np.linalg.norm(col)
        cols.append(col / norm if norm > 1e-10 else col)

    return np.stack(cols, axis=1)  # (param_dim, d)

# ─── Symplectic action proxy + HVP ───────────────────────────────────────────

class SymplecticProxy(nn.Module):
    """Surrogate loss = Σ A(Lk, Lk+1). Its Hessian = curvature of Fukaya energy."""
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank = rank
        flat = torch.cat([w.float().reshape(-1) for w in wk_list])
        self.theta  = nn.Parameter(flat.clone())
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

def take_gradient_steps(model, lr, n_steps):
    """Take n_steps gradient ascent steps on the symplectic action at learning rate lr.
    Returns the updated model (modified in place) and loss trajectory."""
    losses = []
    for _ in range(n_steps):
        loss = model()
        losses.append(loss.item())
        grad = torch.autograd.grad(loss, model.theta)[0]
        with torch.no_grad():
            # Gradient descent on the symplectic action (minimizing strip areas)
            model.theta -= lr * grad
    return losses

# ─── Lanczos + projected Hessian ─────────────────────────────────────────────

def lanczos_Hk(model, k_iters, seed=42):
    """
    Lanczos iteration returning:
      eigvals (k,), eigvecs (param_dim, k), Hk (k, k) tridiagonal matrix.
    Hk is the key new output — currently discarded in the compiler but
    logged here for Hessenberg analysis.
    """
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k_iters + 1)
    alpha = torch.zeros(k_iters)
    beta  = torch.zeros(k_iters - 1)

    v = F.normalize(torch.randn(n), dim=0)
    V[:, 0] = v
    prev_beta = 0.0

    for j in range(k_iters):
        w = hvp(model, V[:, j])
        if j > 0:
            w = w - prev_beta * V[:, j - 1]
        a = (w * V[:, j]).sum().item()
        alpha[j] = a
        w = w - a * V[:, j]
        # Full re-orthogonalization
        for i in range(j + 1):
            w = w - (w @ V[:, i]) * V[:, i]
        if j < k_iters - 1:
            b = w.norm().item()
            if b < 1e-10:
                k_iters = j + 1
                break
            prev_beta = b
            beta[j]   = b
            V[:, j + 1] = w / b

    # Build tridiagonal Hk — this IS the Hessenberg matrix
    k = k_iters
    Hk_np = (np.diag(alpha[:k].numpy()) +
              np.diag(beta[:k-1].numpy(),  1) +
              np.diag(beta[:k-1].numpy(), -1))

    eigvals_T, ritz = np.linalg.eigh(Hk_np)
    eigvecs = (V[:, :k].numpy() @ ritz)
    idx = np.argsort(-np.abs(eigvals_T))
    return eigvals_T[idx], eigvecs[:, idx], Hk_np, alpha[:k].numpy(), beta[:k-1].numpy()

def project_to_strip_space(vecs, basis):
    """Project (param_dim, n) matrix of vectors onto strip-angle basis (param_dim, d).
    Returns (d, n) coordinates."""
    return basis.T @ vecs

def cosine_matrix(A, B):
    """Cosine similarity between columns. Returns (ncols_A, ncols_B)."""
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return An.T @ Bn

# ─── P4a: LR sweep ───────────────────────────────────────────────────────────

def run_lr_sweep(wk_list_init, lrs, rank, k_lanczos, n_steps, seed):
    """
    For each η ∈ lrs:
      1. Reset model to initial WK
      2. Take n_steps gradient steps at lr=η
      3. Compute Ĥ_η = π H_η π^T (projected Hessian in strip-angle space)
      4. Compute top-d eigenvectors of Ĥ_η
      5. Compute |cos|(Ĥ_η-eigvecs, c-vectors)
    Returns dict of results per lr.
    """
    print(f"\n{'='*60}")
    print(f"  P4a: LR SWEEP  ({len(lrs)} learning rates)")
    print(f"{'='*60}")

    # Compute c-vectors once from initial WK
    B_init, areas_init = build_exchange_matrix(wk_list_init, rank)
    C_init, _ = compute_c_vectors(B_init)
    d = C_init.shape[0]
    print(f"  d={d}  rank={rank}  strip areas (init): " +
          ", ".join(f"{a:.3f}" for a in areas_init))
    print(f"  c-vector matrix:\n{np.round(C_init, 3)}")

    # Build initial strip-angle basis from original WK
    basis_init = build_strip_angle_basis(wk_list_init, rank)  # (param_dim, d)

    # Lift c-vectors to strip-angle space: each c-vector col → tiled by rank,
    # weighted by strip areas
    area_weights = areas_init / (areas_init.sum() + 1e-12)
    C_weighted = C_init * area_weights[:, None]   # (d, d)
    C_tiled    = np.repeat(C_weighted, rank, axis=0)  # (d*rank, d) — for comparison

    results = {}
    cos_vs_lr = []

    for lr in sorted(lrs, reverse=True):  # high η first for display
        print(f"\n  η = {lr:.0e}")
        t0 = time.time()

        # Reset model to initial state
        model = SymplecticProxy(wk_list_init, rank)

        # Take n_steps gradient steps at this lr
        if n_steps > 0:
            loss_traj = take_gradient_steps(model, lr, n_steps)
            print(f"    Loss after {n_steps} step(s): "
                  f"{loss_traj[0]:.4f} → {loss_traj[-1]:.4f}")

        # Lanczos on updated model — get eigenvectors AND Hk
        eigvals, eigvecs, Hk, alpha_coeffs, beta_coeffs = lanczos_Hk(
            model, k_lanczos, seed)

        # Build updated strip-angle basis at this η — detach from autograd graph
        wk_updated = [w.detach().float() for w in model.wk_matrices()]
        basis_η = build_strip_angle_basis(wk_updated, rank)  # (param_dim, d)

        # Compute updated strip areas and exchange matrix
        B_η, areas_η = build_exchange_matrix(wk_updated, rank)
        area_std_η = float(np.std(areas_η))

        # Project top-d Hessian eigenvectors into strip-angle space
        k_eff = min(d, k_lanczos)
        H_proj = project_to_strip_space(eigvecs[:, :k_eff], basis_η)  # (d, k_eff)

        # Project c-vectors into the SAME strip-angle space (using η-updated basis)
        area_w_η = areas_η / (areas_η.sum() + 1e-12)
        C_η, _ = compute_c_vectors(B_η)
        C_w_η  = C_η * area_w_η[:, None]
        C_tiled_η = np.repeat(C_w_η, rank, axis=0)[:H_proj.shape[0], :k_eff]

        cos_mat = cosine_matrix(H_proj[:, :k_eff], C_tiled_η[:, :k_eff])
        mean_cos = float(np.abs(cos_mat).max(axis=0).mean())

        # ── P4c: Tropicalization / integer check ──
        # val(x) = -log|x|  applied componentwise to H_proj columns
        # Check if result is close to integers
        trop_vecs = []
        integer_scores = []
        for i in range(k_eff):
            v = H_proj[:, i]
            # Tropicalization: -log|v_j| for nonzero entries
            nz = np.abs(v) > 1e-10
            trop = np.zeros_like(v)
            trop[nz] = -np.log(np.abs(v[nz]))
            trop_vecs.append(trop)
            # Integrality: distance to nearest integer
            rounded = np.round(trop[nz])
            dist = np.abs(trop[nz] - rounded).mean() if nz.sum() > 0 else 1.0
            integer_scores.append(float(dist))

        mean_int_dist = float(np.mean(integer_scores))

        # ── Hessenberg spectral entropy ──
        Hk_eigvals = np.linalg.eigvalsh(Hk)
        abs_ev = np.abs(Hk_eigvals)
        abs_ev_norm = abs_ev / (abs_ev.sum() + 1e-12)
        hess_entropy = float(-np.sum(abs_ev_norm * np.log(abs_ev_norm + 1e-12)))

        print(f"    Strip areas: " + ", ".join(f"{a:.3f}" for a in areas_η) +
              f"  std={area_std_η:.4f}")
        print(f"    Top-{k_eff} eigenvalues: " +
              ", ".join(f"{e:.3f}" for e in eigvals[:k_eff]))
        print(f"    |cos|(Ĥ_η eigvecs, c-vecs) = {mean_cos:.4f}")
        print(f"    Tropicalization int. dist   = {mean_int_dist:.4f}  "
              f"({'CLOSE TO INTEGER' if mean_int_dist < 0.25 else 'not integer'})")
        print(f"    Hessenberg spectral entropy = {hess_entropy:.4f}")
        print(f"    β_j (off-diag, first 5)    = " +
              ", ".join(f"{b:.3f}" for b in beta_coeffs[:5]))
        print(f"    [{time.time()-t0:.1f}s]")

        cos_vs_lr.append((lr, mean_cos))
        results[str(lr)] = {
            "lr":                   lr,
            "log10_lr":             float(math.log10(lr)),
            "mean_cos":             mean_cos,
            "strip_areas":          areas_η.tolist(),
            "strip_area_std":       area_std_η,
            "top_k_eigvals":        eigvals[:k_eff].tolist(),
            "Hk_diagonal":          alpha_coeffs.tolist(),
            "Hk_offdiagonal":       beta_coeffs.tolist(),
            "Hessenberg_entropy":   hess_entropy,
            "tropicalization_int_dist": mean_int_dist,
            "integer_check":        bool(mean_int_dist < 0.25),
            "cosine_matrix":        cos_mat.tolist(),
        }

    return results, cos_vs_lr, C_init, areas_init

# ─── P4b: Pentagon tropicalization ───────────────────────────────────────────

def pentagon_tropicalization(areas, C):
    """
    Apply val = -log|·| to the m2 values (approximated by strip areas A_k)
    and check whether the result satisfies the tropical exchange relation:
      x_k' ⊕_trop x_k = ⊗_trop_{b_{ik}>0} x_i^{b_{ik}}
                        ⊕_trop ⊗_trop_{b_{ik}<0} x_i^{-b_{ik}}
    In tropical arithmetic: a ⊕ b = min(a,b), a ⊗ b = a+b.
    Exchange relation becomes: min(x_k', x_k) = min(Σ_{b>0} b*x_i, Σ_{b<0} -b*x_i)
    """
    print(f"\n{'='*60}")
    print(f"  P4b: PENTAGON TROPICALIZATION")
    print(f"{'='*60}")

    d = len(areas)
    # val(A_k) = -log(A_k) for positive strip areas
    trop_areas = -np.log(np.array(areas))
    print(f"  Strip areas: " + ", ".join(f"{a:.4f}" for a in areas))
    print(f"  val(A_k) = -log(A_k): " + ", ".join(f"{v:.4f}" for v in trop_areas))

    # Build exchange matrix and check tropical exchange relations
    B_init = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            diff = areas[i] - areas[j]
            B_init[i, j] = np.sign(diff) * round(abs(diff), 6)

    results = []
    print(f"\n  Checking tropical exchange relations for each mutation vertex k:")
    for k in range(d):
        # Tropical exchange relation: val(x_k') = min over two monomials
        # Positive part: min_{b_{ik}>0} b_{ik} * val(x_i)
        # Negative part: min_{b_{ik}<0} (-b_{ik}) * val(x_i)
        pos_indices = [i for i in range(d) if B_init[i, k] > 0 and i != k]
        neg_indices = [i for i in range(d) if B_init[i, k] < 0 and i != k]

        if pos_indices:
            pos_monomial = sum(B_init[i, k] * trop_areas[i] for i in pos_indices)
        else:
            pos_monomial = float('inf')

        if neg_indices:
            neg_monomial = sum(-B_init[i, k] * trop_areas[i] for i in neg_indices)
        else:
            neg_monomial = float('inf')

        # Tropical sum = min
        trop_exchange_rhs = min(pos_monomial, neg_monomial)
        lhs = trop_areas[k]

        # Check: val(x_k) ≈ trop_exchange_rhs?
        # In the uniform-area regime B ≈ 0, so this will be degenerate;
        # the check is most meaningful in the differentiated regime.
        residual = abs(lhs - trop_exchange_rhs) if trop_exchange_rhs != float('inf') else float('inf')
        satisfied = residual < 0.5 and trop_exchange_rhs != float('inf')

        print(f"    k={k}: val(x_k)={lhs:.4f}  "
              f"trop_rhs={trop_exchange_rhs:.4f}  "
              f"residual={residual:.4f}  "
              f"{'✓' if satisfied else '✗ (degenerate if B≈0)'}")
        results.append({
            "k": k,
            "val_xk": float(lhs),
            "trop_exchange_rhs": float(trop_exchange_rhs) if trop_exchange_rhs != float('inf') else None,
            "residual": float(residual) if trop_exchange_rhs != float('inf') else None,
            "satisfied": bool(satisfied),
            "degenerate": B_init[:, k].sum() == 0,
        })

    n_satisfied = sum(r["satisfied"] for r in results)
    n_valid = sum(not r["degenerate"] for r in results)
    print(f"\n  {n_satisfied}/{d} exchange relations satisfied "
          f"({n_valid} non-degenerate)")
    print(f"  NOTE: In the uniform-strip regime (B≈0), all relations are "
          f"degenerate.\n  Full validation requires floor-regime checkpoint with std>0.5.")

    return results, trop_areas.tolist()

# ─── Analysis: monotonicity test ─────────────────────────────────────────────

def analyze_lr_sweep(cos_vs_lr):
    """
    Test whether |cos| is monotonically increasing as η decreases.
    Prediction: monotone toward 1.0 as η→0.
    """
    lrs_sorted = sorted(cos_vs_lr, reverse=True)   # high η first
    cos_vals   = [c for _, c in lrs_sorted]
    lr_vals    = [lr for lr, _ in lrs_sorted]

    # Spearman rank correlation of η vs |cos|
    # Prediction: negative (lower η → higher |cos|)
    n = len(cos_vals)
    rank_lr  = np.argsort(np.argsort(lr_vals))
    rank_cos = np.argsort(np.argsort(cos_vals))
    d2 = ((rank_lr - rank_cos) ** 2).sum()
    spearman = 1 - 6 * d2 / (n * (n**2 - 1)) if n > 2 else float('nan')

    # Check monotone decreasing in |cos| as η increases
    # (i.e., monotone increasing as η decreases = as log10(η) decreases)
    diffs = [cos_vals[i+1] - cos_vals[i] for i in range(len(cos_vals)-1)]
    monotone_up = all(d >= -0.01 for d in diffs)  # 0.01 tolerance for noise

    verdict = "TROPICALIZATION_CONFIRMED" if (spearman < -0.5 and monotone_up) else \
              "PARTIAL_EVIDENCE" if spearman < 0 else \
              "NOT_SUPPORTED"

    return {
        "lr_values":         lr_vals,
        "cos_values":        cos_vals,
        "spearman_rho":      float(spearman),
        "monotone_toward_1": monotone_up,
        "verdict":           verdict,
        "interpretation": {
            "TROPICALIZATION_CONFIRMED":
                "Monotone increase in |cos| as η→0 confirms c-vectors are the "
                "logarithmic limit of Hessian curvature directions. "
                "Conjecture 21.1 (Tropicalization Bridge) is experimentally supported.",
            "PARTIAL_EVIDENCE":
                "Negative Spearman correlation but non-monotone — tropicalization "
                "tendency present but obscured by noise or search-regime degeneracy. "
                "Re-run on floor-regime checkpoint (val<0.062, strip-area std>0.5).",
            "NOT_SUPPORTED":
                "No monotone trend detected. Either the tropicalization hypothesis "
                "fails, or the current checkpoint is too deep in the uniform-strip "
                "regime for c-vectors to differentiate. "
                "Run on floor checkpoint before concluding.",
        }[verdict]
    }

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  TROPICALIZATION LR SWEEP — Experiment P4")
    print("  Tests: lim_{η→0} val(v_i(η)) = c_i ∈ Z^d")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  LRs: {args.lrs}")
    print(f"  rank={args.rank}  k_lanczos={args.k_lanczos}  "
          f"n_steps={args.n_steps}")

    t_total = time.time()

    # Load
    state = load_state(args.checkpoint, args.device)
    wk_list = extract_wk(state)
    print(f"\n  Loaded {len(wk_list)} WK matrices, D={wk_list[0].shape[0]}")

    # P4a: LR sweep
    sweep_results, cos_vs_lr, C_init, areas_init = run_lr_sweep(
        wk_list, args.lrs, args.rank, args.k_lanczos, args.n_steps, args.seed)

    # Analysis
    analysis = analyze_lr_sweep(cos_vs_lr)

    # P4b: Pentagon tropicalization (on initial strip areas)
    pent_results, trop_areas = pentagon_tropicalization(areas_init, C_init)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  LR sweep results:")
    for lr, cos in sorted(cos_vs_lr):
        bar = "█" * int(cos * 30)
        print(f"    η={lr:.0e}  |cos|={cos:.4f}  {bar}")
    print(f"\n  Spearman ρ(η, |cos|) = {analysis['spearman_rho']:.4f}  "
          f"(prediction: ρ < 0, i.e. lower η → higher |cos|)")
    print(f"  Monotone toward 1.0: {analysis['monotone_toward_1']}")
    print(f"\n  VERDICT: {analysis['verdict']}")
    print(f"  {analysis['interpretation']}")
    print(f"\n  Integer check (η=min): "
          f"{sweep_results[str(min(args.lrs))]['integer_check']}  "
          f"(dist={sweep_results[str(min(args.lrs))]['tropicalization_int_dist']:.4f})")
    print(f"  Pentagon: {sum(r['satisfied'] for r in pent_results)}/{len(pent_results)} "
          f"exchange relations satisfied")
    print(f"\n  Total elapsed: {time.time()-t_total:.1f}s")

    # Write report
    report = {
        "experiment": "P4 Tropicalization LR Sweep",
        "checkpoint": str(args.checkpoint),
        "config": {
            "lrs": args.lrs, "rank": args.rank,
            "k_lanczos": args.k_lanczos, "n_steps": args.n_steps
        },
        "c_vectors_init": C_init.tolist(),
        "strip_areas_init": areas_init.tolist(),
        "tropicalized_areas": trop_areas,
        "P4a_lr_sweep": sweep_results,
        "P4a_analysis": analysis,
        "P4b_pentagon": pent_results,
        "elapsed_s": round(time.time() - t_total, 1),
    }

    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
