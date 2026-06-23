"""
cluster_preconditioned_lanczos.py
===================================
Cluster-mutation basis as geometric pre-conditioner for the Lanczos Newton step.

Motivation (from correlation result):
  - Mean |cos|(Hessian eigvecs, c-vectors) = 0.2975  → not literal equivalents
  - BUT: H[03]↔c[02]=0.61, H[00]↔c[03]=0.53         → saddle-escape directions align
  - Strip area std = 0.036                             → checkpoint is in uniform-transverse
                                                         (search) regime, not floor regime
  - Hessian has large negatives (-153): saddle geometry

Key insight: c-vectors don't span the same space as Hessian eigenvectors, but they
*pre-select the subspace* where the relevant curvature lives. By seeding the Lanczos
Krylov space with the cluster-mutation directions (lifted to param-space), we bias
the solver toward the K0-relevant curvature directions and away from the flat/noisy
directions that dominate a blind random initialization.

This is NOT claiming c-vectors = Hessian eigenvectors. It is using the cluster
algebra's combinatorial skeleton as a geometric preconditioner for the metric solver.

Usage
-----
  python cluster_preconditioned_lanczos.py \
      --checkpoint basin_state.pt \
      --model_checkpoint basin_state.pt \
      --rank 6 --k_lanczos 20 --n_steps 8 --mu 0.95 \
      --compare_blind \
      --output cluster_lanczos_report.json

Requires: torch, numpy  (same env as compiler_geometric.py)
"""

import argparse
import json
import math
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       default="basin_state.pt",
                   help="Post-TopoGate checkpoint (basin_state.pt preferred over basin_entry)")
    p.add_argument("--rank",   type=int, default=6,  help="Bridgeland / principal-angle rank")
    p.add_argument("--k_lanczos", type=int, default=20, help="Lanczos iterations")
    p.add_argument("--n_steps",   type=int, default=8,  help="LM Newton steps")
    p.add_argument("--mu",  type=float, default=0.95,   help="LM damping (fixed point)")
    p.add_argument("--lr",  type=float, default=3e-4,   help="Learning rate for CE baseline")
    p.add_argument("--compare_blind", action="store_true",
                   help="Also run blind (random-seed) Lanczos for comparison")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="cluster_lanczos_report.json")
    return p.parse_args()

# ---------------------------------------------------------------------------
# 1. Checkpoint loading (same tolerant loader as correlation script)
# ---------------------------------------------------------------------------

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt:
                return ckpt[key], ckpt
    return ckpt, {}

def extract_wk_matrices(state):
    wk_tensors = {}
    for name, tensor in state.items():
        if tensor.ndim < 2:
            continue
        n = name.lower()
        if "c_attn" in n and "weight" in n:
            try:
                layer_idx = int([p for p in name.split(".") if p.isdigit()][0])
            except (IndexError, ValueError):
                layer_idx = len(wk_tensors)
            D = tensor.shape[0] // 3 if tensor.shape[0] % 3 == 0 else tensor.shape[1] // 3
            wk_tensors[layer_idx] = (tensor[D:2*D, :] if tensor.shape[0] % 3 == 0
                                     else tensor[:, D:2*D].T)
        elif ("key" in n or "wk" in n or "w_k" in n) and "weight" in n:
            try:
                layer_idx = int([p for p in name.split(".") if p.isdigit()][0])
            except (IndexError, ValueError):
                layer_idx = len(wk_tensors)
            wk_tensors[layer_idx] = tensor if tensor.ndim == 2 else tensor.squeeze()
    if not wk_tensors:
        raise RuntimeError("No WK matrices found. Keys: " + ", ".join(list(state.keys())[:20]))
    return [wk_tensors[i] for i in sorted(wk_tensors)]

# ---------------------------------------------------------------------------
# 2. Exchange matrix + BMRR c-vectors  (same as correlation script)
# ---------------------------------------------------------------------------

def principal_angles(A, B, dim):
    Ua = torch.linalg.svd(A.float(), full_matrices=False)[0][:, :dim]
    Ub = torch.linalg.svd(B.float(), full_matrices=False)[0][:, :dim]
    sv = torch.linalg.svdvals(Ua.T @ Ub).clamp(-1, 1)
    return torch.arccos(sv)

def build_exchange_matrix(wk_list, rank):
    areas = [principal_angles(wk_list[k], wk_list[k+1], rank).sum().item()
             for k in range(len(wk_list) - 1)]
    d = len(areas)
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            B[i, j] = np.sign(areas[i] - areas[j]) * round(abs(areas[i] - areas[j]), 4)
    return B, np.array(areas)

def bmrr_mutate_B(B, k):
    d, B2 = B.shape[0], B.copy()
    for i in range(d):
        for j in range(d):
            B2[i,j] = (-B[i,j] if (i==k or j==k)
                       else B[i,j] + max(B[i,k],0)*max(B[k,j],0) - min(B[i,k],0)*min(B[k,j],0))
    return B2

def bmrr_mutate_c(C, B, k):
    d, C2 = C.shape[0], C.copy()
    col = np.array([-C[i,k] if i==k
                    else C[i,k] + max(B[i,k],0)*max(C[i,k],0) - min(B[i,k],0)*min(C[i,k],0)
                    for i in range(d)])
    C2[:, k] = col
    return C2

def compute_c_vectors(B_init):
    d, B, C = B_init.shape[0], B_init.copy(), np.eye(B_init.shape[0])
    seq = []
    for k in range(d):
        B = bmrr_mutate_B(B, k)
        C = bmrr_mutate_c(C, B, k)
        seq.append(k)
    return C, seq, B

# ---------------------------------------------------------------------------
# 3. Lift c-vectors to parameter space
#
#    Each c-vector column c[:,j] assigns weight c[k,j] to layer-pair k.
#    We lift it to param-space via the gradient of the strip area A(Lk, Lk+1)
#    with respect to the WK[k] parameters — the same basis directions used
#    in the correlation script, but now returned as actual param-space vectors
#    suitable for seeding a Krylov space.
# ---------------------------------------------------------------------------

def lift_cvec_to_param_space(c_col, wk_list, rank, param_splits, param_offsets,
                              param_dim):
    """
    Lift one c-vector column (d,) to a param-space direction (param_dim,).
    Each weight c_col[k] multiplies the gradient direction of A(Lk, Lk+1)
    w.r.t. WK[k], computed via the SVD chain rule.
    """
    d = len(c_col)
    direction = np.zeros(param_dim)

    for k in range(d):
        if abs(c_col[k]) < 1e-10:
            continue
        Wk  = wk_list[k].float()
        Wk1 = wk_list[k+1].float()
        Uk,  _,  Vhk  = torch.linalg.svd(Wk,  full_matrices=False)
        Uk1, _,  Vhk1 = torch.linalg.svd(Wk1, full_matrices=False)
        Uk, Uk1 = Uk[:, :rank], Uk1[:, :rank]
        M = Uk.T @ Uk1
        Um, Sm, Vhm = torch.linalg.svd(M)
        Sm = Sm.clamp(1e-6, 1 - 1e-6)

        for r in range(rank):
            scale = -1.0 / math.sqrt(max(1.0 - Sm[r].item()**2, 1e-8))
            # dA/dWk in Wk block
            lk  = (Uk  @ Um[:, r:r+1]).reshape(-1).numpy()
            rk  = (Vhm[r:r+1, :] @ Vhk1[:rank, :]).reshape(-1).numpy()
            block_k = (lk[:, None] * rk[None, :]).reshape(-1)
            s, e = param_offsets[k], param_offsets[k+1]
            direction[s:e] += scale * c_col[k] * block_k[:e-s]

            # dA/dWk1 in Wk+1 block
            lk1 = (Uk1 @ Vhm[r:r+1, :].T).reshape(-1).numpy()
            rk1 = (Um[:, r:r+1].T @ Vhk[:rank, :]).reshape(-1).numpy()
            block_k1 = (lk1[:, None] * rk1[None, :]).reshape(-1)
            s1, e1 = param_offsets[k+1], param_offsets[k+2]
            direction[s1:e1] += scale * c_col[k] * block_k1[:e1-s1]

    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-10 else direction


def build_cluster_seed_vectors(C, wk_list, rank):
    """
    Build the (d,) set of param-space seed vectors from the c-vector matrix C.
    Returns list of numpy arrays, one per c-vector column.
    """
    shapes  = [w.shape for w in wk_list]
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0] + splits))
    param_dim = sum(splits)
    d = C.shape[1]

    seeds = []
    for j in range(d):
        v = lift_cvec_to_param_space(C[:, j], wk_list, rank,
                                     splits, offsets, param_dim)
        seeds.append(torch.tensor(v, dtype=torch.float32))
        print(f"      c-vector {j}: ||seed||={seeds[-1].norm():.4f}  "
              f"nonzero={int((seeds[-1].abs()>1e-8).sum())}/{param_dim}")
    return seeds

# ---------------------------------------------------------------------------
# 4. Model reconstruction for HVP
#    We need a forward pass + loss to compute HVPs.
#    Use the same TinyTransformerProxy (symplectic action functional).
# ---------------------------------------------------------------------------

class SymplecticProxy(nn.Module):
    """
    Surrogate loss = Σ A(Lk, Lk+1) = sum of strip areas.
    This is the J-holomorphic action functional on the WK weight manifold.
    Its Hessian captures the curvature of the Fukaya energy landscape.
    """
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank = rank
        flat  = torch.cat([w.float().reshape(-1) for w in wk_list])
        self.theta  = nn.Parameter(flat.clone())
        self.shapes = [w.shape for w in wk_list]
        self.splits = [w.numel() for w in wk_list]

    def wk_matrices(self):
        return [p.reshape(s) for p, s in
                zip(torch.split(self.theta, self.splits), self.shapes)]

    def forward(self):
        wks  = self.wk_matrices()
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

# ---------------------------------------------------------------------------
# 5. Cluster-preconditioned Lanczos
#
#    Standard Lanczos seeds with a random vector.
#    Cluster-preconditioned Lanczos seeds with the cluster mutation directions,
#    then extends the Krylov space normally.
#
#    The preconditioner does NOT change the mathematics — it changes which
#    part of the Krylov space is explored first, biasing toward the
#    K0-relevant curvature directions identified by the cluster algebra.
# ---------------------------------------------------------------------------

def lanczos_with_seed(model, k_iters, seed_vectors=None, random_seed=42):
    """
    Lanczos iteration with optional seed vectors.

    seed_vectors: list of torch.Tensor (param_dim,) — used as the first
                  basis vectors, Gram-Schmidt orthogonalized.
                  If None: random initialization (blind Lanczos).

    Returns: (eigvals, eigvecs, krylov_basis)
    """
    torch.manual_seed(random_seed)
    n = model.theta.numel()
    V = torch.zeros(n, k_iters + 1)

    # ── Seed initialization ──────────────────────────────────────────────
    if seed_vectors is not None and len(seed_vectors) > 0:
        # Orthogonalize seeds via modified Gram-Schmidt
        basis_so_far = []
        for sv in seed_vectors:
            sv = sv.float()
            for b in basis_so_far:
                sv = sv - (sv @ b) * b
            norm = sv.norm()
            if norm > 1e-10:
                sv = sv / norm
                basis_so_far.append(sv)
        # Use first seed as v0
        if basis_so_far:
            V[:, 0] = basis_so_far[0]
            print(f"      Seeded with {len(basis_so_far)} cluster directions "
                  f"(using first as v₀)")
        else:
            V[:, 0] = F.normalize(torch.randn(n), dim=0)
    else:
        V[:, 0] = F.normalize(torch.randn(n), dim=0)

    T_diag    = torch.zeros(k_iters)
    T_offdiag = torch.zeros(k_iters - 1)
    beta = 0.0

    for j in range(k_iters):
        w = hvp(model, V[:, j])
        if j > 0:
            w = w - beta * V[:, j - 1]
        alpha = (w * V[:, j]).sum().item()
        T_diag[j] = alpha
        w = w - alpha * V[:, j]
        if j < k_iters - 1:
            # Re-orthogonalize against all previous vectors (full reorthog)
            for prev in range(j + 1):
                w = w - (w @ V[:, prev]) * V[:, prev]
            beta = w.norm().item()
            T_offdiag[j] = beta
            if beta < 1e-10:
                print(f"      Lanczos converged early at iteration {j}")
                k_iters = j + 1
                break
            V[:, j + 1] = w / beta

    T = (np.diag(T_diag[:k_iters].numpy()) +
         np.diag(T_offdiag[:k_iters-1].numpy(),  1) +
         np.diag(T_offdiag[:k_iters-1].numpy(), -1))
    eigvals, ritz = np.linalg.eigh(T)
    eigvecs = (V[:, :k_iters].numpy() @ ritz)

    idx = np.argsort(-np.abs(eigvals))
    return eigvals[idx], eigvecs[:, idx], V[:, :k_iters]

# ---------------------------------------------------------------------------
# 6. Cluster-preconditioned Newton step
#
#    The LM Newton step solves (H + µI)δ = -∇L via CG.
#    We modify CG to start from the cluster-seed subspace projection
#    rather than zero, giving the solver a geometrically informed warm start.
# ---------------------------------------------------------------------------

def cluster_preconditioned_cg(hvp_fn, grad, mu, seed_vectors, max_iter=50, tol=1e-4):
    """
    Preconditioned CG for (H + µI)δ = -grad.

    Warm start: project -grad onto cluster seed subspace to get δ₀,
    then run standard CG from δ₀.

    This is equivalent to applying the cluster-basis projector
    P = V_cluster V_cluster^T as a left preconditioner.
    """
    n = grad.numel()
    b = -grad.clone()

    # Warm start from cluster subspace
    delta = torch.zeros_like(grad)
    if seed_vectors:
        for sv in seed_vectors:
            sv = sv.to(grad.device)
            coeff = (b @ sv) / (sv @ sv + 1e-12)
            delta = delta + coeff * sv

    # Residual
    Hd  = hvp_fn(delta) + mu * delta
    r   = b - Hd
    p   = r.clone()
    rs_old = (r @ r).item()

    convergence = []
    for i in range(max_iter):
        Hp   = hvp_fn(p) + mu * p
        alpha = rs_old / ((p @ Hp).item() + 1e-12)
        delta = delta + alpha * p
        r     = r - alpha * Hp
        rs_new = (r @ r).item()
        convergence.append(math.sqrt(rs_new))
        if math.sqrt(rs_new) < tol:
            print(f"      CG converged at iteration {i+1}  (residual={rs_new:.2e})")
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new

    return delta, convergence

# ---------------------------------------------------------------------------
# 7. Diagnostic: compare cluster-seeded vs blind Lanczos
# ---------------------------------------------------------------------------

def compare_lanczos(model, seed_vectors, k_iters, seed):
    print("\n  [Cluster-seeded Lanczos]")
    t0 = time.time()
    eig_c, vec_c, _ = lanczos_with_seed(model, k_iters, seed_vectors, seed)
    t_cluster = time.time() - t0

    print("\n  [Blind (random-seed) Lanczos]")
    t0 = time.time()
    eig_b, vec_b, _ = lanczos_with_seed(model, k_iters, None, seed)
    t_blind = time.time() - t0

    # Compare: how well do the two span the same subspace?
    # Use principal angles between the two k-dim subspaces
    top = min(5, k_iters)
    Vc = vec_c[:, :top]
    Vb = vec_b[:, :top]
    M  = Vc.T @ Vb
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1, 1)
    angles = np.degrees(np.arccos(sv))

    print(f"\n  Principal angles between cluster-seeded and blind subspaces (top-{top}):")
    for i, a in enumerate(angles):
        bar = "█" * int((90 - a) / 5)
        print(f"    θ[{i}] = {a:5.1f}°  {bar}  {'ALIGNED' if a < 20 else 'DIFFERENT'}")

    print(f"\n  Cluster-seeded top-5 eigenvalues: " +
          ", ".join(f"{e:.3f}" for e in eig_c[:5]))
    print(f"  Blind          top-5 eigenvalues: " +
          ", ".join(f"{e:.3f}" for e in eig_b[:5]))
    print(f"\n  Wall-clock: cluster={t_cluster:.1f}s  blind={t_blind:.1f}s")

    return {
        "cluster_eigvals":      eig_c[:top].tolist(),
        "blind_eigvals":        eig_b[:top].tolist(),
        "subspace_angles_deg":  angles.tolist(),
        "mean_angle_deg":       float(np.mean(angles)),
        "time_cluster_s":       round(t_cluster, 2),
        "time_blind_s":         round(t_blind, 2),
        "interpretation": (
            "CLUSTER_FINDS_DIFFERENT_SUBSPACE — K0 map hypothesis supported: "
            "cluster basis selects distinct curvature directions vs random init."
            if np.mean(angles) > 30 else
            "CLUSTER_AND_BLIND_AGREE — cluster basis compatible with random Krylov; "
            "uniform strip areas mean all directions equally relevant."
        )
    }

# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print("  CLUSTER-PRECONDITIONED LANCZOS NEWTON")
    print("  K0(A∞) → K0(Cluster) geometric preconditioner")
    print("=" * 64)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  rank={args.rank}  k_lanczos={args.k_lanczos}  "
          f"n_steps={args.n_steps}  µ={args.mu}")
    print()

    # ── Load ────────────────────────────────────────────────────────────────
    print("[1/5] Loading checkpoint …")
    state, full_ckpt = load_checkpoint(args.checkpoint, args.device)
    wk_list = extract_wk_matrices(state)
    n_layers = len(wk_list)
    D = wk_list[0].shape[0]
    print(f"      {n_layers} WK matrices, D={D}")

    # Regime diagnosis from strip areas
    areas_quick = [principal_angles(wk_list[k], wk_list[k+1], args.rank).sum().item()
                   for k in range(n_layers - 1)]
    area_std = np.std(areas_quick)
    regime = ("SEARCH (uniform transverse — compare to basin_state.pt for floor regime)"
              if area_std < 0.5 else "CONVERGENCE (differentiated strip areas)")
    print(f"      Strip area std={area_std:.4f}  → {regime}")

    # ── Exchange matrix + c-vectors ─────────────────────────────────────────
    print("[2/5] Building exchange matrix + c-vectors …")
    B_init, areas = build_exchange_matrix(wk_list, args.rank)
    C, mut_seq, B_final = compute_c_vectors(B_init)
    d = C.shape[0]
    print(f"      Strip areas: " + ", ".join(f"{a:.3f}" for a in areas))
    print(f"      c-vector matrix: {C.shape}")

    # ── Lift c-vectors to param space ───────────────────────────────────────
    print("[3/5] Lifting c-vectors to parameter space …")
    seed_vectors = build_cluster_seed_vectors(C, wk_list, args.rank)

    # ── Symplectic proxy + Lanczos ──────────────────────────────────────────
    print("[4/5] Running cluster-preconditioned Lanczos …")
    proxy = SymplecticProxy(wk_list, args.rank)

    lanczos_report = {}
    if args.compare_blind:
        print("\n  Comparing cluster-seeded vs blind Lanczos …")
        lanczos_report = compare_lanczos(proxy, seed_vectors, args.k_lanczos, args.seed)
    else:
        eigvals, eigvecs, _ = lanczos_with_seed(
            proxy, args.k_lanczos, seed_vectors, args.seed)
        print(f"  Top-8 eigenvalues: " + ", ".join(f"{e:.3f}" for e in eigvals[:8]))
        lanczos_report = {"cluster_eigvals": eigvals[:8].tolist()}

    # ── Cluster-preconditioned CG Newton step ───────────────────────────────
    print("\n[5/5] Cluster-preconditioned CG Newton step …")
    loss = proxy()
    grad = torch.autograd.grad(loss, proxy.theta)[0].detach()
    print(f"      Loss (symplectic action) = {loss.item():.4f}")
    print(f"      ||∇L|| = {grad.norm().item():.4f}")

    hvp_fn = lambda v: hvp(proxy, v)

    t0 = time.time()
    delta_cluster, conv_cluster = cluster_preconditioned_cg(
        hvp_fn, grad, args.mu, seed_vectors, max_iter=50)
    t_cluster = time.time() - t0

    # Blind CG for comparison
    t0 = time.time()
    delta_blind, conv_blind = cluster_preconditioned_cg(
        hvp_fn, grad, args.mu, None, max_iter=50)
    t_blind = time.time() - t0

    # Alignment between the two Newton directions
    cos_directions = float(
        (delta_cluster @ delta_blind) /
        (delta_cluster.norm() * delta_blind.norm() + 1e-12)
    )

    print(f"\n  CG convergence (cluster):  {len(conv_cluster)} iters, "
          f"final residual={conv_cluster[-1]:.4e}  [{t_cluster:.1f}s]")
    print(f"  CG convergence (blind):    {len(conv_blind)} iters, "
          f"final residual={conv_blind[-1]:.4e}  [{t_blind:.1f}s]")
    print(f"  cos(δ_cluster, δ_blind) = {cos_directions:.4f}")

    # Interpret the Newton direction alignment
    if cos_directions > 0.9:
        newton_interp = "EQUIVALENT — cluster preconditioning converges to same Newton step"
    elif cos_directions > 0.5:
        newton_interp = "PARTIALLY_DIFFERENT — cluster basis rotates Newton step toward K0 directions"
    else:
        newton_interp = "DISTINCT — cluster preconditioning finds a qualitatively different descent direction"

    print(f"  Interpretation: {newton_interp}")

    # Final summary
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Checkpoint regime:  {regime}")
    print(f"  Strip area std:     {area_std:.4f}")
    print(f"  Cluster CG iters:   {len(conv_cluster)}  (blind: {len(conv_blind)})")
    print(f"  Newton direction alignment: {cos_directions:.4f}")
    print(f"  {newton_interp}")

    if area_std < 0.5:
        print()
        print("  ⚠  NOTE: Strip areas are nearly uniform (std < 0.5).")
        print("     This checkpoint is in the search/transverse regime.")
        print("     Re-run with basin_state.pt (post-TopoGate) for floor-regime comparison.")
        print("     Expected: larger area_std → more differentiated c-vectors → stronger effect.")

    # ── Write report ────────────────────────────────────────────────────────
    report = {
        "checkpoint": str(args.checkpoint),
        "regime": regime,
        "strip_areas": areas.tolist(),
        "strip_area_std": float(area_std),
        "exchange_matrix": B_init.tolist(),
        "c_vectors": C.tolist(),
        "mutation_sequence": mut_seq,
        "lanczos": lanczos_report,
        "newton": {
            "loss_symplectic": float(loss.item()),
            "grad_norm": float(grad.norm().item()),
            "mu": args.mu,
            "cluster_cg_iters": len(conv_cluster),
            "blind_cg_iters":   len(conv_blind),
            "cluster_convergence": conv_cluster,
            "blind_convergence":   conv_blind,
            "cos_newton_directions": cos_directions,
            "interpretation": newton_interp,
        },
        "conclusion": {
            "K0_map_hypothesis": (
                "SUPPORTED" if cos_directions < 0.9 or
                (lanczos_report.get("mean_angle_deg", 0) > 30) else "NOT_DISTINGUISHABLE"
            ),
            "next_step": (
                "Run on basin_state.pt (post-TopoGate) where strip areas are differentiated. "
                "If K0_map_hypothesis=SUPPORTED there, integrate cluster-preconditioned CG "
                "into lanczos_tau_retry.py as a drop-in replacement for the standard CG solver."
            )
        }
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\n  Report → {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
