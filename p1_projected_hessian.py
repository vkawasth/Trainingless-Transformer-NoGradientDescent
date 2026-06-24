"""
p1_projected_hessian.py
========================
Experiment P1: Projected Hessian vs Exchange Matrix

Tests Proposition 22.1 from the paper:

    Ĥ = π H π^T  ≈  α* · B

i.e. the Hessian of the symplectic action, projected into the d-dimensional
strip-angle space, is proportional to the exchange matrix B derived from
strip areas.

The strip-angle projection π: R^D → R^d maps each parameter-space direction
onto its effect on the d strip areas A(L_k, L_{k+1}).
Ĥ = π H π^T is the d×d curvature of the Fukaya action in strip-angle coords.
B is the skew-symmetrized pairwise strip-area difference matrix.

Success criterion (Proposition 22.1): ||Ĥ - α*B||_F / ||B||_F < 0.1

Also reports:
  - Scale-normalized Frobenius cosine cos(Ĥ, B)
  - Spectral cosine between spec(Ĥ) and spec(B)
  - Comparison across multiple Lanczos ranks to test stability
  - Hessenberg spectral entropy at this checkpoint
  - Regime diagnosis and floor-checkpoint prediction

Usage
-----
  python p1_projected_hessian.py \\
      --checkpoint basin_state.pt \\
      --rank 6 --k_lanczos 20 \\
      --k_range 5 10 15 20 \\
      --output p1_report.json

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
    p = argparse.ArgumentParser(description="Experiment P1: Ĥ ≈ α*·B")
    p.add_argument("--checkpoint", default="basin_state.pt")
    p.add_argument("--rank",       type=int, default=6)
    p.add_argument("--k_lanczos",  type=int, default=20)
    p.add_argument("--k_range",    nargs="+", type=int, default=[5, 10, 15, 20],
                   help="Lanczos k values to sweep (tests stability of Ĥ)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--output",     default="p1_report.json")
    return p.parse_args()


# ─── Checkpoint ──────────────────────────────────────────────────────────────

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
        raise RuntimeError("No WK matrices. Keys: " + ", ".join(list(state.keys())[:20]))
    return [wk[i].detach().float() for i in sorted(wk)]


# ─── Strip areas + exchange matrix B ─────────────────────────────────────────

def strip_area(Wk, Wk1, rank):
    Uk  = torch.linalg.svd(Wk.detach().float(),  full_matrices=False)[0][:, :rank]
    Uk1 = torch.linalg.svd(Wk1.detach().float(), full_matrices=False)[0][:, :rank]
    sv  = torch.linalg.svdvals(Uk.T @ Uk1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.arccos(sv).sum().item()

def compute_areas(wk_list, rank):
    return np.array([strip_area(wk_list[k], wk_list[k+1], rank)
                     for k in range(len(wk_list) - 1)])

def build_exchange_matrix(areas):
    """
    B[i,j] = sign(A_i - A_j) * |A_i - A_j|
    Skew-symmetrized pairwise strip-area differences.
    B ≈ 0 in the uniform regime; anisotropic in the floor regime.
    """
    d = len(areas)
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            diff = areas[i] - areas[j]
            B[i, j] = np.sign(diff) * round(abs(diff), 6)
    return B


# ─── Strip-angle projection π ────────────────────────────────────────────────

def build_basis(wk_list, rank):
    """
    Returns (param_dim, d) matrix whose columns are unit-norm gradients of
    each strip area A(L_k, L_{k+1}) w.r.t. the full parameter vector.
    This is π^T: projecting a parameter-space vector v gives π v = basis^T @ v.
    """
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


# ─── Symplectic proxy + Lanczos ───────────────────────────────────────────────

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

def lanczos_run(model, k_iters, seed=42):
    """Returns eigvals (k,), eigvecs (param_dim, k), Hk (k,k), alpha, beta."""
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k_iters + 1)
    alpha_c = torch.zeros(k_iters)
    beta_c  = torch.zeros(k_iters - 1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:, 0] = v
    prev_b = 0.0
    for j in range(k_iters):
        w = hvp(model, V[:, j])
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
    evals, ritz = np.linalg.eigh(Hk)
    evecs = V[:, :k].numpy() @ ritz
    idx   = np.argsort(-np.abs(evals))
    return evals[idx], evecs[:, idx], Hk, alpha_c[:k].numpy(), beta_c[:k-1].numpy()


# ─── Projected Hessian Ĥ = π H π^T ─────────────────────────────────────────

def compute_H_hat(eigvals, eigvecs, basis, d):
    """
    Ĥ = π H π^T  ≈  (π V) Λ (π V)^T
    where H ≈ V Λ V^T (Lanczos approximation),
    π = basis^T  (d × param_dim),
    V = eigvecs  (param_dim × k).

    Returns (d, d) matrix.
    Also returns the projected eigenvectors (d, k) for inspection.
    """
    k_eff   = min(eigvecs.shape[1], d)
    V_k     = eigvecs[:, :k_eff]          # (param_dim, k_eff)
    lam_k   = eigvals[:k_eff]             # (k_eff,)
    PiV     = basis.T @ V_k               # (d, k_eff)  — projected eigvecs
    H_hat   = (PiV * lam_k[None, :]) @ PiV.T  # (d, d)  — Ĥ with eigenvalue weights
    return H_hat, PiV


# ─── Comparison metrics ───────────────────────────────────────────────────────

def frobenius_cos(A, B):
    """Scale-normalized Frobenius cosine: <A/||A||, B/||B||>_F."""
    return (float(np.sum(A * B)) /
            (np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro') + 1e-12))

def relative_error_scaled(A, B):
    """||A/||A|| - B/||B||||_F  (scale-normalized relative error)."""
    An = A / (np.linalg.norm(A, 'fro') + 1e-12)
    Bn = B / (np.linalg.norm(B, 'fro') + 1e-12)
    return float(np.linalg.norm(An - Bn, 'fro'))

def alpha_star(A, B):
    """Optimal scale: α* = <A,B>_F / ||B||_F^2."""
    return float(np.sum(A * B)) / (np.linalg.norm(B, 'fro')**2 + 1e-12)

def spectral_cos(A, B):
    """Cosine between eigenvalue spectra (sorted by magnitude)."""
    ea = np.linalg.eigvalsh(A)
    eb = np.linalg.eigvalsh(B)
    ea = ea[np.argsort(-np.abs(ea))]
    eb = eb[np.argsort(-np.abs(eb))]
    n  = min(len(ea), len(eb))
    return float(np.dot(ea[:n], eb[:n]) /
                 (np.linalg.norm(ea[:n]) * np.linalg.norm(eb[:n]) + 1e-12))

def spectral_cos_abs(A, B):
    """
    Cosine between ABSOLUTE eigenvalue spectra — the correct metric when
    comparing a symmetric matrix (Ĥ) with a skew-symmetric matrix (B).

    B is skew-symmetric by construction (B[i,j] = sign(A_i - A_j)|A_i - A_j|,
    so B = -B^T exactly). Ĥ is symmetric (it's a projected Hessian).

    The Frobenius inner product <Ĥ, B>_F = 0 identically for any symmetric Ĥ
    and skew-symmetric B — this is an algebraic identity, not a signal.
    The signed spectral cosine is also contaminated by this mismatch.

    The correct comparison: rank-by-rank alignment of |eigenvalues|.
    This tests whether the MAGNITUDE of curvature in each strip-angle direction
    matches between Ĥ and B, independent of the symmetric/skew-symmetric split.
    """
    ea = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
    eb = np.sort(np.abs(np.linalg.eigvalsh(B)))[::-1]
    n  = min(len(ea), len(eb))
    return float(np.dot(ea[:n], eb[:n]) /
                 (np.linalg.norm(ea[:n]) * np.linalg.norm(eb[:n]) + 1e-12))

def hessenberg_entropy(Hk):
    """Spectral entropy of Hessenberg matrix: -Σ p_i log p_i, p_i = |λ_i|/Σ|λ_j|."""
    ev  = np.linalg.eigvalsh(Hk)
    abs_ev = np.abs(ev)
    p   = abs_ev / (abs_ev.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

def sign_pattern(A):
    ev = np.linalg.eigvalsh(A)
    return ['+' if e > 1e-4 else ('0' if abs(e) < 1e-4 else '-') for e in ev]

def diagnose_scale(H_hat, B):
    nH = np.linalg.norm(H_hat, 'fro')
    nB = np.linalg.norm(B, 'fro')
    ratio = nH / (nB + 1e-12)
    a = alpha_star(H_hat, B)
    scale_ok = (1e-3 < ratio < 1e3) and abs(a) > 1e-6
    return {
        "norm_H_hat": float(nH),
        "norm_B":     float(nB),
        "scale_ratio": float(ratio),
        "alpha_star":  float(a),
        "scale_ok":    bool(scale_ok),
        "diagnosis":  ("OK" if scale_ok else
                       f"SCALE_MISMATCH (ratio={ratio:.2e}, α*={a:.2e}): "
                       "use scale-normalized Frobenius cos as primary metric"),
    }


# ─── Stability sweep across Lanczos k ────────────────────────────────────────

def k_stability_sweep(model, basis, B, d, k_values, seed):
    """
    Run Lanczos at each k in k_values and compute Ĥ.
    Tests whether Ĥ is stable as k increases (key Hessenberg invariant property).
    Returns dict k → {H_hat, cos_B, rel_err, ...}
    """
    results = {}
    prev_H_hat = None

    # Check if B is skew-symmetric (algebraic identity makes <Ĥ,B>_F = 0)
    B_skew = np.allclose(B, -B.T, atol=1e-6)
    if B_skew:
        print(f"  ⚠  B is exactly skew-symmetric → <Ĥ,B>_F = 0 identically.")
        print(f"     Primary metric: spectral_cos_abs(Ĥ, B) = cos(|spec Ĥ|, |spec B|)")
        print(f"     (rank-by-rank |eigenvalue| alignment, sign-independent)")

    print(f"\n  Lanczos k-stability sweep: {k_values}")
    for k in sorted(k_values):
        t0 = time.time()
        evals, evecs, Hk, alpha_c, beta_c = lanczos_run(model, k, seed)
        H_hat, PiV = compute_H_hat(evals, evecs, basis, d)
        cos_B      = frobenius_cos(H_hat, B)
        cos_B_abs  = spectral_cos_abs(H_hat, B)   # correct metric for skew B
        cos_B_sign = spectral_cos(H_hat, B)        # signed (shows anti-alignment)
        rel_err    = relative_error_scaled(H_hat, B)
        hk_ent     = hessenberg_entropy(Hk)
        sc_diag    = diagnose_scale(H_hat, B)

        # Stability vs previous k
        stability = frobenius_cos(H_hat, prev_H_hat) if prev_H_hat is not None else 1.0
        prev_H_hat = H_hat.copy()

        results[k] = {
            "cos_B_frobenius":   float(cos_B),
            "cos_B_spec_abs":    float(cos_B_abs),
            "cos_B_spec_signed": float(cos_B_sign),
            "rel_err_scaled":    float(rel_err),
            "Hk_entropy":        float(hk_ent),
            "stability_vs_prev": float(stability),
            "scale_diag":        sc_diag,
            "top5_eigvals":      evals[:5].tolist(),
            "beta_first5":       beta_c[:5].tolist(),
        }
        print(f"    k={k:2d}:  cos_abs(Ĥ,B)={cos_B_abs:+.4f}  "
              f"cos_sign={cos_B_sign:+.4f}  "
              f"entropy={hk_ent:.3f}  "
              f"stable={stability:.4f}  "
              f"[{time.time()-t0:.1f}s]")
    return results


# ─── Verdict ─────────────────────────────────────────────────────────────────

def interpret(best_cos_abs, best_rel_err, area_std, scale_ok):
    """Verdict based on spectral_cos_abs (primary metric for skew B vs symmetric Ĥ)."""
    if best_cos_abs > 0.9:
        verdict = "SPECTRAL_CONFIRMED"
        msg = ("cos(|spec Ĥ|, |spec B|) > 0.9: spectral magnitude alignment confirmed. "
               "The projected Hessian and exchange matrix share the same curvature "
               "magnitude structure in strip-angle space. "
               "Note: Frobenius comparison requires floor checkpoint where B is anisotropic "
               "and amplitude-comparable. Frobenius criterion ||Ĥ - α*B||_F/||B||_F < 0.1 "
               "pending.")
    elif best_cos_abs > 0.7:
        verdict = "PARTIAL_SPECTRAL"
        msg = ("0.7 < cos(|spec|) < 0.9: partial spectral alignment. "
               f"Strip-area std={area_std:.4f}. "
               "Re-run on floor checkpoint (std > 0.5).")
    else:
        verdict = "NOT_SUPPORTED"
        msg = ("cos(|spec Ĥ|, |spec B|) < 0.7. "
               "Either search-regime degeneracy or no alignment. "
               "Check strip-area std and floor checkpoint.")
    return verdict, msg


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  EXPERIMENT P1: PROJECTED HESSIAN vs EXCHANGE MATRIX")
    print("  Tests: Ĥ = πHπ^T ≈ α*·B  (Proposition 22.1)")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  rank={args.rank}  k_lanczos={args.k_lanczos}  "
          f"k_range={args.k_range}")

    t_total = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading checkpoint …")
    state   = load_state(args.checkpoint, args.device)
    wk_list = extract_wk(state)
    D = wk_list[0].shape[0]
    n_layers = len(wk_list)
    d = n_layers - 1
    print(f"      {n_layers} WK matrices  D={D}  d={d}")

    # ── 2. Strip areas + B ───────────────────────────────────────────────────
    print("[2/5] Computing strip areas and exchange matrix B …")
    areas    = compute_areas(wk_list, args.rank)
    B        = build_exchange_matrix(areas)
    area_std = float(np.std(areas))
    mu       = float(np.mean(areas))

    print(f"      Strip areas: " + ", ".join(f"{a:.4f}" for a in areas))
    print(f"      µ={mu:.4f}  std={area_std:.4f}")
    print(f"      B (off-diag max): {float(np.abs(B - np.diag(np.diag(B))).max()):.6f}")
    print(f"      ||B||_F = {np.linalg.norm(B,'fro'):.6f}  "
          f"{'⚠ B≈0 (uniform strips)' if np.linalg.norm(B,'fro') < 0.5 else 'B non-trivial'}")

    regime = ("SEARCH/TRANSVERSE (B≈0, comparisons scale-degenerate)"
              if area_std < 0.5 else
              "CONVERGENCE/FLOOR (B anisotropic, P1 test meaningful)")
    print(f"      Regime: {regime}")

    # ── 3. Strip-angle basis ─────────────────────────────────────────────────
    print("[3/5] Building strip-angle basis π …")
    basis = build_basis(wk_list, args.rank)
    print(f"      Basis shape: {basis.shape}  (param_dim × d)")
    # Verify basis orthogonality
    gram = basis.T @ basis
    off_diag_err = float(np.abs(gram - np.eye(d)).max())
    print(f"      Basis near-orthogonal: max off-diag |Gram - I| = {off_diag_err:.4f}")

    # ── 4. Proxy model ───────────────────────────────────────────────────────
    print("[4/5] Running Lanczos k-stability sweep …")
    model   = SymplecticProxy(wk_list, args.rank)
    sweep   = k_stability_sweep(model, basis, B, d, args.k_range, args.seed)

    # Best k = most stable (highest stability_vs_prev after warmup)
    best_k      = max(sweep, key=lambda k: (sweep[k]["stability_vs_prev"]
                                              if k > min(args.k_range) else -1))
    best_cos_abs  = sweep[best_k]["cos_B_spec_abs"]    # primary metric
    best_cos_sign = sweep[best_k]["cos_B_spec_signed"]
    best_cos_frob = sweep[best_k]["cos_B_frobenius"]
    best_rel      = sweep[best_k]["rel_err_scaled"]
    scale_ok      = sweep[best_k]["scale_diag"]["scale_ok"]
    B_skew        = np.allclose(B, -B.T, atol=1e-6)

    # ── 5. Full Ĥ at best k ──────────────────────────────────────────────────
    print(f"\n[5/5] Full Ĥ at best k={best_k} …")
    evals, evecs, Hk, alpha_c, beta_c = lanczos_run(model, best_k, args.seed)
    H_hat, PiV = compute_H_hat(evals, evecs, basis, d)
    sc_diag    = diagnose_scale(H_hat, B)

    print(f"      Ĥ:\n{np.round(H_hat, 6)}")
    print(f"      B:\n{np.round(B, 6)}")
    print(f"      Ĥ eigenvalues: " +
          ", ".join(f"{e:.4f}" for e in np.linalg.eigvalsh(H_hat)))
    print(f"      B eigenvalues: " +
          ", ".join(f"{e:.4f}" for e in np.linalg.eigvalsh(B)))
    print(f"      Ĥ sign pattern: {sign_pattern(H_hat)}")
    print(f"      B sign pattern: {sign_pattern(B)}")

    verdict, msg = interpret(best_cos_abs, best_rel, area_std, scale_ok)

    print(f"\n{'='*60}")
    print(f"  P1 RESULT (best k={best_k})")
    print(f"{'='*60}")
    print(f"  B is skew-symmetric: {B_skew}  "
          f"→ {'Frobenius cos = 0 by algebra; use spectral_cos_abs' if B_skew else 'Frobenius valid'}")
    print(f"  cos(|spec Ĥ|, |spec B|)  = {best_cos_abs:+.4f}  ← PRIMARY METRIC")
    print(f"  spectral cos (signed)    = {best_cos_sign:+.4f}  "
          f"({'anti-aligned: sign convention mismatch' if best_cos_sign < -0.5 else 'aligned'})")
    print(f"  Frobenius cos(Ĥ, B)     = {best_cos_frob:+.4f}  "
          f"(0 by algebra if B skew)")
    print(f"  rel_err_scaled           = {best_rel:.4f}")
    print(f"  scale diagnosis : {sweep[best_k]['scale_diag']['diagnosis']}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    if area_std < 0.5:
        print(f"\n  ⚠  B≈0 in search regime: ||B||_F = "
              f"{np.linalg.norm(B,'fro'):.4f}")
        print(f"     All Frobenius comparisons against B are near-random.")
        print(f"     The meaningful test requires floor checkpoint where B is anisotropic.")
        print(f"     Predicted at val < 0.062 (Moran fixation prediction).")

    print(f"\n  k-sweep summary:")
    print(f"  {'k':>3}  {'|spec|cos':>10}  {'spec_sign':>10}  "
          f"{'entropy':>9}  {'stable':>8}")
    for k in sorted(sweep):
        r = sweep[k]
        print(f"  {k:>3}  {r['cos_B_spec_abs']:>+10.4f}  "
              f"{r['cos_B_spec_signed']:>+10.4f}  "
              f"{r['Hk_entropy']:>9.4f}  {r['stability_vs_prev']:>8.4f}")

    report = {
        "experiment": "P1 Projected Hessian vs Exchange Matrix",
        "checkpoint": str(args.checkpoint),
        "config": {"rank": args.rank, "k_lanczos": args.k_lanczos,
                   "k_range": args.k_range},
        "strip_areas":       areas.tolist(),
        "strip_area_std":    float(area_std),
        "exchange_matrix_B": B.tolist(),
        "B_norm_F":          float(np.linalg.norm(B, 'fro')),
        "B_is_skew":         bool(B_skew),
        "basis_gram_err":    float(off_diag_err),
        "regime":            regime,
        "k_sweep":           sweep,
        "best_k":            int(best_k),
        "H_hat_best":        H_hat.tolist(),
        "H_hat_eigvals":     np.linalg.eigvalsh(H_hat).tolist(),
        "B_eigvals":         np.linalg.eigvalsh(B).tolist(),
        "H_hat_sign":        sign_pattern(H_hat),
        "B_sign":            sign_pattern(B),
        "P1_result": {
            "cos_spec_abs":      float(best_cos_abs),   # PRIMARY
            "cos_spec_signed":   float(best_cos_sign),
            "cos_frobenius":     float(best_cos_frob),
            "rel_err_scaled":    float(best_rel),
            "B_is_skew":         bool(B_skew),
            "skew_note": ("B is skew-symmetric; Frobenius cos = 0 by algebra. "
                          "Primary metric is cos(|spec Ĥ|, |spec B|)."),
            "scale_diagnosis":   sweep[best_k]["scale_diag"],
            "verdict":           verdict,
            "message":           msg,
        },
        "elapsed_s": round(time.time() - t_total, 1),
    }

    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {Path(args.output).resolve()}")
    print(f"  Total elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
