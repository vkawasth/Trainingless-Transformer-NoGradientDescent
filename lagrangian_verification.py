"""
lagrangian_verification.py
===========================
Verifies that the rank-r column spaces of WK matrices ARE Lagrangian
submanifolds under the correct symplectic structure, and that the
strip-area formula computes the correct Floer-theoretic area.

The claim: L_k = col_r(WK^(k)) ⊂ R^{2r} is Lagrangian under ω_r.

Verification:
  1. L_k is r-dimensional (by construction: top-r left singular vectors)
  2. ω_r|_{L_k} = 0 (isotropic condition)
  3. The principal angles between L_k and L_{k+1} give the Floer area

The symplectic structure:
  ω_r = Σᵢ dxᵢ ∧ dx_{i+r} on R^{2r}
  Standard form: ω(u,v) = u^T J v where J = [[0,-I],[I,0]]

The Lagrangian condition:
  L = col(U) where U ∈ R^{D×r}, U^T U = I
  L is Lagrangian in R^{2r} (r-dim subspace of 2r-dim space) iff
  U^T J U = 0, i.e., the r×r matrix U_top^T U_bot - U_bot^T U_top = 0
  where U = [U_top; U_bot] with U_top, U_bot ∈ R^{r×r}.

For the graph construction:
  graph(A) = {(x, Ax) : x ∈ R^r} ⊂ R^{2r}
  Basis vectors: eᵢ ↦ (eᵢ, Aeᵢ)
  ω(graph basis i, graph basis j) = eᵢ^T J [eⱼ; Aeⱼ]
    = eᵢ^T(-Aeⱼ) = -Aᵢⱼ
  So ω|_{graph(A)} = -(A - A^T)/2 = 0 iff A is symmetric.

For the SVD construction:
  L_k = col(Uₖ) where Uₖ = top-r left singular vectors of WK^(k)
  Embed: L_k ↦ R^{2r} via x ↦ [x; 0] ... NO, wrong embedding.

Correct embedding for comparing L_k and L_{k+1}:
  The natural space is R^{2D} = T*R^D with ω = dq ∧ dp
  L_k = {(Uₖx, 0) : x ∈ R^r} ... isotropic trivially (p=0)
  This is a LAGRANGIAN in R^{2D} (the zero section is always Lagrangian)

Better: use the Lagrangian Grassmannian Λ(r) of the symplectic R^{2r}.
  Each Uₖ defines a point in Gr(r, D) (real Grassmannian)
  The principal angles between Uₖ and Uₖ₊₁ measure their separation in Gr(r,D)
  arccos(σᵢ(Uₖ^T Uₖ₊₁)) is the standard Riemannian distance on Gr(r,D)

The strip-area formula Σᵢ arccos(σᵢ(Uₖ^T Uₖ₊₁)) IS the geodesic distance
in Gr(r,D) under the standard metric.

This is the correct identification: NOT "WK is a Lagrangian submanifold"
but "col_r(WK) is a point in the Grassmannian Gr(r,D), and the 
strip-area formula computes the geodesic distance between consecutive points."

Usage
-----
  python lagrangian_verification.py \
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \
      --rank 6
"""

import argparse, math
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64', default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--rank',    type=int, default=6)
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


def verify_lagrangian(wk_list, rank):
    """
    Test 1: Is graph(WK) Lagrangian in T*R^D?
      ω|_{graph(WK)} = -(WK - WK^T)/2
      Lagrangian iff WK is symmetric.

    Test 2: Is col_r(WK) a point in Gr(r,D)?
      Always yes by construction.

    Test 3: Does arccos formula compute geodesic distance in Gr(r,D)?
      Geodesic dist = sqrt(Σᵢ θᵢ²) where θᵢ = arccos(σᵢ(Uₖ^T Uₖ₊₁))
      Strip area uses Σᵢ θᵢ (sum, not sqrt of sum of squares)
      These differ but both are valid metrics on Gr(r,D).

    Test 4: Isotropic condition for col_r(WK) in Lagrangian Grassmannian.
      For L = col(U) ⊂ R^{2r} to be Lagrangian:
      U = [U_top; U_bot] (split into r×r blocks)
      Condition: U_top^T U_bot = U_bot^T U_top (symmetric)
      i.e., U_top^T U_bot is symmetric.
    """
    print("="*60)
    print("  LAGRANGIAN VERIFICATION")
    print("="*60)

    for k, W in enumerate(wk_list[:-1]):
        D = W.shape[0]
        print(f"\n  Layer {k}: WK shape {W.shape}")

        # Test 1: Is WK symmetric?
        if W.shape[0] == W.shape[1]:
            skew = W - W.T
            skew_norm = np.linalg.norm(skew, 'fro')
            sym_norm  = np.linalg.norm(W, 'fro')
            print(f"  Test 1 (graph Lagrangian): ‖WK-WK^T‖/‖WK‖ = {skew_norm/sym_norm:.4f}")
            print(f"    {'✓ symmetric (Lagrangian)' if skew_norm/sym_norm < 0.01 else '✗ NOT symmetric — graph is NOT Lagrangian'}")
        else:
            print(f"  Test 1: WK is {W.shape} — not square, graph construction requires square matrix")
            print(f"    ✗ graph(WK) ∉ T*R^D in the standard sense")

        # Test 2: SVD column space
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        Ur = U[:, :rank]   # D × rank
        print(f"\n  Test 2 (col_{rank}(WK) ∈ Gr({rank},{D})): always ✓")
        print(f"    Top-{rank} singular values: {', '.join(f'{sv:.3f}' for sv in s[:rank])}")
        print(f"    ‖Uₖ^T Uₖ - I‖ = {np.linalg.norm(Ur.T @ Ur - np.eye(rank)):.6f} (should be 0)")

        # Test 3: Strip area as geodesic distance
        if k < len(wk_list) - 1:
            W_next = wk_list[k+1]
            U_next, _, _ = np.linalg.svd(W_next, full_matrices=False)
            Ur_next = U_next[:, :rank]
            overlap = Ur.T @ Ur_next   # rank × rank
            sv_overlap = np.linalg.svd(overlap, compute_uv=False)
            sv_overlap = np.clip(sv_overlap, -1, 1)
            angles = np.arccos(sv_overlap)

            strip_area = float(np.sum(angles))
            geodesic_dist = float(np.sqrt(np.sum(angles**2)))

            print(f"\n  Test 3 (strip area vs geodesic distance on Gr({rank},{D})):")
            print(f"    Principal angles θᵢ: {', '.join(f'{a:.4f}' for a in angles)}")
            print(f"    Strip area  Σᵢ θᵢ  = {strip_area:.4f}  (used in compiler)")
            print(f"    Geodesic dist √Σᵢθᵢ² = {geodesic_dist:.4f}  (standard Gr metric)")
            print(f"    Both are valid metrics on Gr({rank},{D}); strip area = L¹ norm of angles")

        # Test 4: Lagrangian Grassmannian condition
        # Embed col_r(WK) into R^{2r} and check isotropic condition
        # Split Ur into top-r and bottom-r halves (if D >= 2r)
        if D >= 2*rank:
            U_top = Ur[:rank, :]     # rank × rank
            U_bot = Ur[rank:2*rank, :]  # rank × rank
            cross = U_top.T @ U_bot
            skew_cross = cross - cross.T
            print(f"\n  Test 4 (isotropic in Lagrangian Gr, first 2r={2*rank} rows):")
            print(f"    ‖U_top^T U_bot - (U_top^T U_bot)^T‖ = {np.linalg.norm(skew_cross):.6f}")
            print(f"    {'✓ isotropic (Lagrangian)' if np.linalg.norm(skew_cross) < 0.01 else 'non-zero — not Lagrangian in this embedding'}")
        else:
            print(f"\n  Test 4: D={D} < 2r={2*rank}, cannot embed in R^{{2r}} with this split")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: What IS and is NOT proven")
    print(f"{'='*60}")
    print(f"""
  CLAIM: "WK matrices are Lagrangian submanifolds in T*R^D"
  STATUS: FALSE in general.
    graph(WK) is Lagrangian iff WK is symmetric.
    WK matrices are not generally symmetric.

  CORRECTED CLAIM: "col_r(WK) defines a point in Gr(r,D)"
  STATUS: TRUE by construction.
    The rank-r column space is always a well-defined r-plane.

  STRIP AREA CLAIM: "Σᵢ arccos(σᵢ) computes Floer-theoretic area"
  STATUS: PARTIAL.
    arccos(σᵢ) are the principal angles — the Riemannian metric on Gr(r,D).
    Σᵢ arccos(σᵢ) is the L¹ norm of angles (valid geodesic-like quantity).
    √Σᵢ arccos²(σᵢ) is the standard Riemannian geodesic distance on Gr(r,D).
    The Floer-theoretic strip area ∫u*ω requires identifying the correct
    symplectic form on Gr(r,D) — this requires the Kirillov-Kostant-Souriau
    construction on the coadjoint orbit GL(D)/O(r) × O(D-r).
    The exact equality to Floer area is not proven.

  GRASSMANNIAN CLAIM: "training trajectories are paths in Gr(r,D)^L"
  STATUS: TRUE by construction.
    Each WK^(k) defines a point in Gr(r,D).
    The training trajectory is a sequence of points in Gr(r,D)^L.
    The strip-area formula measures L¹ distance along this path.
    This is a valid geometric description WITHOUT requiring Lagrangian submanifolds.

  RECOMMENDATION:
    Replace "Lagrangian submanifold" with "r-plane in Gr(r,D)"
    Replace "J-holomorphic strip" with "path in Gr(r,D)^L"
    The strip-area formula remains valid as a Grassmannian metric.
    The Fukaya category connection requires further work.
""")


def main():
    args = parse_args()
    wk_list = load_wk(args.spike64)
    verify_lagrangian(wk_list, args.rank)


if __name__ == '__main__':
    main()
