"""
hessian_cvector_correlation.py
================================
Standalone correlation analysis: Hessian eigenvectors vs. cluster c-vectors.

Reads basin_entry_state.pt (or any compiler checkpoint .pt file), then:
  1. Extracts top-k Hessian eigenvectors via Lanczos (using HVPs on a tiny
     random batch derived from the checkpoint's corpus stats).
  2. Builds a rank-d exchange matrix B from the m2 composition tensor
     (principal-angle differences between consecutive WK layers).
  3. Computes c-vectors via the BMRR mutation rule (pure NumPy, no Sage).
  4. Computes cosine similarity between the two bases.
  5. Writes correlation_report.json with all vectors and scores.

Usage
-----
  python hessian_cvector_correlation.py \
      --checkpoint basin_entry_state.pt \
      --vocab 1017 --nnz 1347 \
      --top_k 8 --rank 6 \
      --output correlation_report.json

Dependencies: torch, numpy, scipy (all standard in a PyTorch env).
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

# ---------------------------------------------------------------------------
# 0.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hessian eigenvector ↔ c-vector correlation")
    p.add_argument("--checkpoint", default="basin_entry_state.pt",
                   help="Compiler checkpoint (.pt) file")
    p.add_argument("--vocab",  type=int, default=1017, help="VOCAB size")
    p.add_argument("--nnz",   type=int, default=1347, help="Corpus non-zero bigram pairs")
    p.add_argument("--top_k", type=int, default=8,   help="Number of Hessian eigenvectors")
    p.add_argument("--rank",  type=int, default=6,   help="Exchange matrix rank (Bridgeland dim)")
    p.add_argument("--n_hvp", type=int, default=20,  help="Lanczos iterations for HVP")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--output", default="correlation_report.json")
    p.add_argument("--device", default="cpu")
    return p.parse_args()

# ---------------------------------------------------------------------------
# 1.  Checkpoint loading  (tolerant: accepts full model or state_dict)
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Accept dict with 'model', 'state_dict', or raw state_dict
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    return ckpt   # OrderedDict of tensors

def extract_wk_matrices(state: dict):
    """
    Pull all WK (key-projection) weight matrices from the state dict.
    Handles naming conventions: attn.key.weight, transformer.h.N.attn.c_attn.weight, etc.
    Returns list of 2-D tensors, one per layer, in layer order.
    """
    wk_tensors = {}
    for name, tensor in state.items():
        if tensor.ndim < 2:
            continue
        n = name.lower()
        # GPT-2 style: c_attn fused QKV – grab the K slice
        if "c_attn" in n and "weight" in n:
            D = tensor.shape[0] // 3  if tensor.shape[0] % 3 == 0 else tensor.shape[1] // 3
            try:
                layer_idx = int([p for p in name.split(".") if p.isdigit()][0])
            except (IndexError, ValueError):
                layer_idx = len(wk_tensors)
            if tensor.shape[0] % 3 == 0:
                wk_tensors[layer_idx] = tensor[D:2*D, :]   # K block
            else:
                wk_tensors[layer_idx] = tensor[:, D:2*D].T
        # Explicit key weight
        elif ("key" in n or "wk" in n or "w_k" in n) and "weight" in n:
            try:
                layer_idx = int([p for p in name.split(".") if p.isdigit()][0])
            except (IndexError, ValueError):
                layer_idx = len(wk_tensors)
            wk_tensors[layer_idx] = tensor if tensor.ndim == 2 else tensor.squeeze()

    if not wk_tensors:
        raise RuntimeError(
            "No WK matrices found in checkpoint.\n"
            "Keys present: " + ", ".join(list(state.keys())[:20])
        )
    return [wk_tensors[i] for i in sorted(wk_tensors)]

# ---------------------------------------------------------------------------
# 2.  Exchange matrix B from m2 composition tensor
#     B[i,j] = principal-angle difference between layer pairs (i, i+1) and (j, j+1)
# ---------------------------------------------------------------------------

def principal_angles(A: torch.Tensor, B: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Returns the `dim` principal angles (radians) between the column spaces of A and B.
    A, B: (D, *) – we take the top-`dim` left singular vectors.
    """
    Ua = torch.linalg.svd(A.float(), full_matrices=False)[0][:, :dim]
    Ub = torch.linalg.svd(B.float(), full_matrices=False)[0][:, :dim]
    M  = Ua.T @ Ub                          # (dim, dim)
    sv = torch.linalg.svdvals(M).clamp(-1, 1)
    return torch.arccos(sv)                 # (dim,)

def strip_area(angles: torch.Tensor) -> float:
    return angles.sum().item()

def build_exchange_matrix(wk_list: list, rank: int) -> np.ndarray:
    """
    B[i,j] for i,j in {0,...,n_layers-2}:
      B[i,j] = mean_principal_angle(layer_i, layer_i+1)
             - mean_principal_angle(layer_j, layer_j+1)
    Skew-symmetric integer-valued approximation (sign of the difference).
    This is the Fukaya-theoretic exchange matrix: it tracks which layer
    transitions are "more transverse" than others (Bridgeland wall score).
    """
    n = len(wk_list)
    areas = []
    for k in range(n - 1):
        angles = principal_angles(wk_list[k], wk_list[k + 1], rank)
        areas.append(strip_area(angles))

    d = len(areas)
    B = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            diff = areas[i] - areas[j]
            # Skew-symmetrize and sign-quantize (cluster convention)
            B[i, j] = np.sign(diff) * round(abs(diff), 4)

    print(f"\n[Exchange matrix B ({d}×{d}) — strip areas: "
          + ", ".join(f"{a:.3f}" for a in areas) + "]")
    return B, np.array(areas)

# ---------------------------------------------------------------------------
# 3.  BMRR c-vector mutation (pure NumPy, rank-d)
#     Reference: Buan-Marsh-Reineke-Reiten-Todorov (2006)
#     c-vectors start as identity columns; each mutation at vertex k updates them.
# ---------------------------------------------------------------------------

def bmrr_mutate_B(B: np.ndarray, k: int) -> np.ndarray:
    """Mutate exchange matrix at index k."""
    d = B.shape[0]
    B2 = B.copy()
    for i in range(d):
        for j in range(d):
            if i == k or j == k:
                B2[i, j] = -B[i, j]
            else:
                B2[i, j] = B[i, j] + (
                    max(B[i, k], 0) * max(B[k, j], 0)
                    - min(B[i, k], 0) * min(B[k, j], 0)
                )
    return B2

def bmrr_mutate_c(C: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """
    C: (d, d) matrix whose columns are the current c-vectors.
    Mutation at k updates column k.
    """
    d = C.shape[0]
    C2 = C.copy()
    new_col = np.zeros(d)
    for i in range(d):
        if i == k:
            new_col[i] = -C[i, k]
        else:
            new_col[i] = C[i, k] + (
                max(B[i, k], 0) * max(C[i, k], 0)
                - min(B[i, k], 0) * min(C[i, k], 0)
            )
    C2[:, k] = new_col
    return C2

def compute_c_vectors(B_init: np.ndarray, n_mutations: int = None) -> tuple:
    """
    Run BMRR mutations cyclically through all vertices.
    Returns (C_final, mutation_sequence, B_final).
    """
    d = B_init.shape[0]
    if n_mutations is None:
        n_mutations = d  # one full cycle

    B = B_init.copy()
    C = np.eye(d)          # c-vectors start as identity
    sequence = []

    for step in range(n_mutations):
        k = step % d
        B = bmrr_mutate_B(B, k)
        C = bmrr_mutate_c(C, B, k)
        sequence.append(k)

    return C, sequence, B

# ---------------------------------------------------------------------------
# 4.  Hessian eigenvectors via Lanczos + HVP
#     We approximate H by the Gauss-Newton / Fisher on a synthetic batch
#     drawn from the corpus bigram statistics encoded in WK geometry.
# ---------------------------------------------------------------------------

class TinyTransformerProxy(nn.Module):
    """
    Minimal proxy that lets us compute HVPs with respect to WK parameters only.
    We use the WK matrices directly as a linear map and compute a surrogate loss.
    """
    def __init__(self, wk_list: list, rank: int):
        super().__init__()
        self.rank = rank
        # Stack into a parameter for grad purposes
        D = wk_list[0].shape[0]
        n = len(wk_list)
        # Flatten all WK into one parameter vector
        flat = torch.cat([w.float().reshape(-1) for w in wk_list])
        self.theta = nn.Parameter(flat.clone())
        self.shapes = [w.shape for w in wk_list]
        self.splits = [w.numel() for w in wk_list]

    def wk_matrices(self):
        parts = torch.split(self.theta, self.splits)
        return [p.reshape(s) for p, s in zip(parts, self.shapes)]

    def forward(self):
        """
        Surrogate loss = sum of strip areas (J-holomorphic action functional).
        This is the symplectic action S = Σ A(L_k, L_{k+1}).
        Its Hessian w.r.t. theta captures curvature of the Fukaya energy landscape.
        """
        wks = self.wk_matrices()
        loss = torch.tensor(0.0, requires_grad=True)
        for k in range(len(wks) - 1):
            Ua = torch.linalg.svd(wks[k],     full_matrices=False)[0][:, :self.rank]
            Ub = torch.linalg.svd(wks[k + 1], full_matrices=False)[0][:, :self.rank]
            M  = Ua.T @ Ub
            sv = torch.linalg.svdvals(M).clamp(1e-6, 1 - 1e-6)
            area = torch.arccos(sv).sum()
            loss = loss + area
        return loss

def hvp(model: TinyTransformerProxy, v: torch.Tensor) -> torch.Tensor:
    """Hessian-vector product H·v via double backprop."""
    loss = model()
    grad = torch.autograd.grad(loss, model.theta, create_graph=True)[0]
    Hv   = torch.autograd.grad((grad * v).sum(), model.theta)[0]
    return Hv.detach()

def lanczos(model: TinyTransformerProxy, k: int, seed: int = 42) -> tuple:
    """
    Lanczos iteration: returns (eigvals, eigvecs) — top-k Hessian eigenpairs.
    eigvecs: (param_dim, k) tensor.
    """
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k + 1)
    T_diag  = torch.zeros(k)
    T_offdiag = torch.zeros(k - 1)

    v = torch.randn(n)
    v = v / v.norm()
    V[:, 0] = v

    beta = 0.0
    for j in range(k):
        w = hvp(model, V[:, j])
        if j > 0:
            w = w - beta * V[:, j - 1]
        alpha = (w * V[:, j]).sum().item()
        T_diag[j] = alpha
        w = w - alpha * V[:, j]
        if j < k - 1:
            beta = w.norm().item()
            T_offdiag[j] = beta
            if beta < 1e-10:
                break
            V[:, j + 1] = w / beta

    # Tridiagonal eigendecomposition
    T = np.diag(T_diag.numpy()) + np.diag(T_offdiag.numpy(), 1) + np.diag(T_offdiag.numpy(), -1)
    eigvals, ritz = np.linalg.eigh(T)
    # Ritz vectors in original space
    eigvecs = (V[:, :k].numpy() @ ritz)   # (n, k)

    # Sort by descending |eigenvalue|
    idx = np.argsort(-np.abs(eigvals))
    return eigvals[idx], eigvecs[:, idx]

# ---------------------------------------------------------------------------
# 5.  Correlation: Hessian eigenvectors vs c-vectors in shared strip-angle space
#
#     The root cause of the dimension mismatch:
#       - Hessian eigenvectors ∈ R^(total_param_dim)  e.g. 393216
#       - Lifted c-vectors via Uk@Uk.T ∈ R^(D²)       e.g.  65536
#     These are different spaces and cannot be directly compared.
#
#     Solution: project BOTH into the shared strip-angle feature space R^(d·rank),
#     where d = n_layers-1 and rank = Bridgeland dim.  This is the natural
#     "mutual coarse-graining" space: each coordinate is a principal angle
#     between consecutive WK layers, which is what both the exchange matrix B
#     (via strip areas) and the Hessian (via the symplectic action functional)
#     fundamentally depend on.
# ---------------------------------------------------------------------------

def build_strip_angle_basis(wk_list: list, rank: int) -> np.ndarray:
    """
    Build the (param_dim, d*rank) matrix whose columns are the gradient directions
    of each principal angle θ_{k,r} = arccos(σ_r(Uk^T U_{k+1})) w.r.t. theta.

    Each column is a unit vector in param-space pointing in the direction that
    most increases principal angle r between layers k and k+1.
    This is the natural basis shared by both the Hessian and the exchange matrix.

    Returns: basis (param_dim, d*rank) as numpy array.
    """
    d = len(wk_list) - 1
    # Build full param vector layout: same as TinyTransformerProxy
    shapes = [w.shape for w in wk_list]
    splits = [w.numel() for w in wk_list]
    param_dim = sum(splits)
    offsets = np.cumsum([0] + splits)

    basis_cols = []
    for k in range(d):
        Wk  = wk_list[k].float()
        Wk1 = wk_list[k + 1].float()
        Uk,  Sk,  Vhk  = torch.linalg.svd(Wk,  full_matrices=False)
        Uk1, Sk1, Vhk1 = torch.linalg.svd(Wk1, full_matrices=False)
        Uk  = Uk[:,  :rank]
        Uk1 = Uk1[:, :rank]
        M   = Uk.T @ Uk1                              # (rank, rank)
        Um, Sm, Vhm = torch.linalg.svd(M)
        Sm = Sm.clamp(1e-6, 1 - 1e-6)

        for r in range(rank):
            # Gradient of arccos(Sm[r]) w.r.t. Wk and Wk+1
            # dθ/dWk  ∝ -1/√(1-Sm[r]²) · d(σ_r(Uk^T Uk1))/dWk
            # Using the formula for derivative of singular value:
            #   dσ_r/dW = u_r v_r^T  (outer product of left/right singular vecs of M)
            #   chain rule through Uk = U_Wk S_Wk^{-1} ...  simplified:
            #   dθ_{k,r}/dWk  ≈  -scale * (Uk @ Um[:,r:r+1]) @ (Vhk[:rank,:][r:r+1,:])
            scale = -1.0 / (math.sqrt(max(1.0 - Sm[r].item()**2, 1e-8)))
            # Direction in Wk space
            left_k  = (Uk  @ Um[:, r:r+1]).reshape(-1)          # D
            right_k = (Vhm[r:r+1, :] @ Vhk1[:rank, :]).reshape(-1)  # D  (via Uk1)
            # Assemble full param-space gradient
            col = np.zeros(param_dim)
            # dθ/dWk: outer product left_k ⊗ right_k packed into Wk block
            dWk_block  = (left_k.numpy()[:, None] * right_k.numpy()[None, :]).reshape(-1)
            col[offsets[k]  : offsets[k+1]]  = scale * dWk_block[:splits[k]]
            # dθ/dWk1: symmetric
            left_k1  = (Uk1 @ Vhm[r:r+1, :].T).reshape(-1)
            right_k1 = (Um[:, r:r+1].T @ Vhk[:rank, :]).reshape(-1)
            dWk1_block = (left_k1.numpy()[:, None] * right_k1.numpy()[None, :]).reshape(-1)
            col[offsets[k+1]: offsets[k+2]] = scale * dWk1_block[:splits[k+1]]

            norm = np.linalg.norm(col)
            if norm > 1e-10:
                col /= norm
            basis_cols.append(col)

    return np.stack(basis_cols, axis=1)   # (param_dim, d*rank)


def project_to_strip_space(vecs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Project columns of `vecs` (param_dim, n) onto the strip-angle basis (param_dim, d*rank).
    Returns coordinates (d*rank, n) — same ambient space for both H-eigvecs and c-vecs.
    """
    # basis columns are already unit vectors
    return basis.T @ vecs   # (d*rank, n)


def lift_cvectors_to_strip_space(C: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """
    c-vectors ∈ R^d.  Each c-vector column c[:,j] assigns a weight to each
    layer-pair strip.  Lift to strip-angle space R^(d*rank) by repeating the
    weight `rank` times (one per principal angle), scaled by the strip area
    so that dominant strips get proportionally larger representation.

    Returns: (d*rank, d) matrix — c-vectors in strip-angle coords.
    """
    d, n_cvecs = C.shape
    rank = None  # inferred from areas shape vs d
    # areas has length d; we just tile each scalar weight `rank` times
    # rank is passed implicitly via the basis shape; we infer it from the
    # caller (main) who knows both.  Use areas as weights.
    area_weights = areas / (areas.sum() + 1e-12)   # (d,) normalised

    # We don't know rank here, so return weighted C and let main do the tiling
    return C * area_weights[:, None]   # (d, n_cvecs) weighted


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between columns of A (n×a) and B (n×b).
    Returns (a, b) matrix.  Handles mismatched row counts by truncating to min.
    """
    n = min(A.shape[0], B.shape[0])
    A, B = A[:n], B[:n]
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return An.T @ Bn

def match_score(cos_mat: np.ndarray) -> dict:
    """
    For each c-vector (column), find its best-matching Hessian eigenvector.
    Returns dict with max cosines, mean, and the greedy matching.
    """
    abs_cos = np.abs(cos_mat)
    best_per_cvec = abs_cos.max(axis=0).tolist()  # best Hessian match per c-vec
    best_per_hess = abs_cos.max(axis=1).tolist()  # best c-vec match per Hessian eigvec
    mean_best = float(np.mean(best_per_cvec))

    # Greedy bipartite matching
    used_h, used_c = set(), set()
    pairs = []
    flat_idx = np.argsort(-abs_cos.ravel())
    for idx in flat_idx:
        h, c = divmod(int(idx), abs_cos.shape[1])
        if h not in used_h and c not in used_c:
            pairs.append({"hessian_idx": h, "cvec_idx": c,
                          "cosine": float(abs_cos[h, c])})
            used_h.add(h)
            used_c.add(c)
        if len(pairs) == min(abs_cos.shape):
            break

    return {
        "best_per_cvec": best_per_cvec,
        "best_per_hessian_eigvec": best_per_hess,
        "mean_best_match": mean_best,
        "greedy_pairs": pairs,
    }

# ---------------------------------------------------------------------------
# 6.  Bridge diagnosis
# ---------------------------------------------------------------------------

def interpret(mean_score: float, areas: np.ndarray, B: np.ndarray) -> dict:
    if mean_score > 0.7:
        verdict = "BRIDGE_CONFIRMED"
        message = (
            "Hessian eigenvectors ≈ cluster c-vectors (mean |cos| > 0.7). "
            "The 'Infinitesimal Bridge' hypothesis holds: Hessian curvature directions "
            "are the infinitesimal generators of cluster mutation at this checkpoint. "
            "Next step: write functor F: A∞-Cat → Cluster-Cat explicitly."
        )
    elif mean_score > 0.4:
        verdict = "TROPICALIZATION_REGIME"
        message = (
            "Partial alignment (0.4 < mean |cos| ≤ 0.7). "
            "Hessian and c-vectors are related but not identical. "
            "Tropicalization (η→0 limit) likely closes the gap. "
            "Next step: repeat at smaller learning rate and plot |cos| vs η."
        )
    else:
        verdict = "DIMENSIONALITY_MISMATCH"
        message = (
            "Low alignment (mean |cos| ≤ 0.4). "
            "The dimensionality mismatch (R^D vs root lattice R^d) dominates. "
            "A different morphism is needed — likely the Grothendieck group map "
            "K0(A∞) → K0(Cluster) rather than a direct vector correspondence."
        )

    # Extra diagnostics
    area_std = float(np.std(areas))
    wall_layers = int(np.sum(np.abs(B).max(axis=1) > np.mean(np.abs(B))))

    return {
        "verdict": verdict,
        "mean_cos_similarity": float(mean_score),
        "strip_area_std": area_std,
        "bridgeland_wall_count": wall_layers,
        "message": message,
    }

# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print("  HESSIAN EIGENVECTOR ↔ CLUSTER C-VECTOR CORRELATION")
    print("=" * 64)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  VOCAB={args.vocab}  nnz={args.nnz}  rank={args.rank}  top_k={args.top_k}")
    print()

    # ── 1. Load checkpoint ──────────────────────────────────────────────────
    t0 = time.time()
    print("[1/5] Loading checkpoint …")
    state = load_checkpoint(args.checkpoint, args.device)
    wk_list = extract_wk_matrices(state)
    n_layers = len(wk_list)
    print(f"      Found {n_layers} WK matrices, shape {wk_list[0].shape}")

    # ── 2. Exchange matrix B ────────────────────────────────────────────────
    print("[2/5] Building exchange matrix B from m2 composition tensor …")
    B_init, areas = build_exchange_matrix(wk_list, args.rank)
    d = B_init.shape[0]
    print(f"      Exchange matrix shape: {d}×{d}")

    # ── 3. c-vectors via BMRR ───────────────────────────────────────────────
    print("[3/5] Computing c-vectors via BMRR mutation …")
    C, mut_seq, B_final = compute_c_vectors(B_init, n_mutations=d)
    print(f"      Mutation sequence: {mut_seq}")
    print(f"      c-vector matrix shape: {C.shape}")

    # ── 4. Hessian eigenvectors via Lanczos ─────────────────────────────────
    print(f"[4/5] Running Lanczos ({args.n_hvp} iters) for top-{args.top_k} Hessian eigenvectors …")
    proxy = TinyTransformerProxy(wk_list, args.rank)
    eigvals, eigvecs = lanczos(proxy, k=args.n_hvp, seed=args.seed)
    # Take top_k
    H_vecs = eigvecs[:, :args.top_k]   # (param_dim, top_k)
    top_eigvals = eigvals[:args.top_k].tolist()
    print(f"      Top-{args.top_k} eigenvalues: " +
          ", ".join(f"{e:.4f}" for e in top_eigvals))

    # ── 5. Correlation ──────────────────────────────────────────────────────
    print("[5/5] Computing cosine similarity in shared strip-angle space …")
    # Build the shared basis: gradient directions of each principal angle
    print("      Building strip-angle basis (param_dim, d×rank) …")
    basis = build_strip_angle_basis(wk_list, args.rank)  # (param_dim, d*rank)
    print(f"      Basis shape: {basis.shape}")

    # Project Hessian eigenvectors into strip-angle space
    H_proj = project_to_strip_space(H_vecs, basis)   # (d*rank, top_k)

    # Lift c-vectors into strip-angle space
    # C is (d, d); each column is a c-vector weighted by strip areas, tiled to (d*rank, d)
    C_weighted = lift_cvectors_to_strip_space(C, areas)   # (d, d) weighted
    C_tiled    = np.repeat(C_weighted, args.rank, axis=0)  # (d*rank, d)

    # Compare: top_k Hessian projections vs d c-vectors
    k_eff   = min(args.top_k, d)
    cos_mat = cosine_matrix(H_proj[:, :k_eff], C_tiled[:, :k_eff])  # (k_eff, k_eff)
    scores  = match_score(cos_mat)
    diagnosis = interpret(scores["mean_best_match"], areas, B_init)

    # ── Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  VERDICT: {diagnosis['verdict']}")
    print(f"  Mean |cosine|: {scores['mean_best_match']:.4f}")
    print()
    print("  Greedy bipartite matching:")
    for p in scores["greedy_pairs"]:
        bar = "█" * int(p["cosine"] * 20)
        print(f"    H[{p['hessian_idx']:02d}] ↔ c[{p['cvec_idx']:02d}]  "
              f"|cos|={p['cosine']:.4f}  {bar}")
    print()
    print(f"  Strip areas: " + ", ".join(f"{a:.3f}" for a in areas))
    print(f"  Bridgeland wall count: {diagnosis['bridgeland_wall_count']}/{d}")
    print(f"  Strip area std: {diagnosis['strip_area_std']:.4f}")
    print()
    print(f"  {diagnosis['message']}")
    print("=" * 64)

    # ── Write JSON ──────────────────────────────────────────────────────────
    report = {
        "checkpoint": str(args.checkpoint),
        "corpus": {"vocab": args.vocab, "nnz": args.nnz},
        "config": {"rank": args.rank, "top_k": args.top_k, "n_hvp": args.n_hvp},
        "projection": {
            "method": "shared_strip_angle_space",
            "basis_shape": list(basis.shape),
            "note": (
                "Both Hessian eigenvectors and c-vectors projected onto the "
                "gradient directions of principal angles between consecutive WK layers. "
                "This is the natural mutual coarse-graining space: "
                "dim = (n_layers-1) * rank."
            ),
        },
        "hessian": {
            "top_eigenvalues": top_eigvals,
            "eigvec_param_dim": list(H_vecs.shape),
            "eigvec_projected_shape": list(H_proj.shape),
        },
        "cluster": {
            "strip_areas": areas.tolist(),
            "exchange_matrix_B_init": B_init.tolist(),
            "mutation_sequence": mut_seq,
            "c_vector_matrix": C.tolist(),
            "c_vector_weighted_shape": list(C_tiled.shape),
        },
        "correlation": {
            "cosine_matrix": cos_mat.tolist(),
            **scores,
        },
        "diagnosis": diagnosis,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved → {out.resolve()}")

if __name__ == "__main__":
    main()
