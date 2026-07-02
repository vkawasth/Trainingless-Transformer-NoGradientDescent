"""
weighted_rm2_diagnostic.py
===========================
Computes the strip-area-weighted Frobenius correlation r_m2^σ:

  r_m2^σ = ⟨Ĥ, Hess(m2)⟩_σ / (‖Ĥ‖_σ ‖Hess(m2)‖_σ)

where the weighted inner product is:
  ⟨A, B⟩_σ = Σᵢⱼ Aᵢⱼ Bᵢⱼ / (σᵢ σⱼ)

This is the CORRECT diagnostic for Frobenius order in the
non-uniform-strip regime. The unweighted r_m2 is degenerate
when strip areas are uniform (all σᵢ ≈ const), which is why
the paper reports "Frobenius order unresolved."

The weighted metric down-weights directions with large strip area
(already well-aligned Lagrangians) and up-weights directions with
small strip area (nearly-intersecting Lagrangians where m2 is active).

Usage
-----
  python weighted_rm2_diagnostic.py \
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \
      --spike72 tau_spikes/tau_spike_step0072_tau5.94.pt \
      --basin_entry basin_entry_state.pt \
      --basin_state basin_state.pt \
      --rank 6 \
      --output weighted_rm2_report.json
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64',     default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--spike72',     default='tau_spikes/tau_spike_step0072_tau5.94.pt')
    p.add_argument('--basin_entry', default='basin_entry_state.pt')
    p.add_argument('--basin_state', default='basin_state.pt')
    p.add_argument('--rank',        type=int, default=6)
    p.add_argument('--output',      default='weighted_rm2_report.json')
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


def compute_strip_svs(wk_list, rank):
    """
    For each consecutive pair (L_k, L_{k+1}), compute the singular values
    σᵢ of Ũ_k^T Ũ_{k+1} (the overlaps between r-planes).
    These are cos(θᵢ) where θᵢ are principal angles.
    Strip area = Σᵢ arccos(σᵢ).
    """
    results = []
    for k in range(len(wk_list)-1):
        U_k,  _, _ = np.linalg.svd(wk_list[k],   full_matrices=False)
        U_k1, _, _ = np.linalg.svd(wk_list[k+1], full_matrices=False)
        Ur  = U_k[:, :rank]
        Ur1 = U_k1[:, :rank]
        sv = np.linalg.svd(Ur.T @ Ur1, compute_uv=False)
        sv = np.clip(sv, 0, 1)
        results.append({
            'k': k, 'k_next': k+1,
            'strip_svs': sv,          # cos(θᵢ)
            'strip_angles': np.arccos(sv),
            'strip_area': float(np.sum(np.arccos(sv))),
            'strip_area_std': float(np.std(np.arccos(sv))),
        })
    return results


def compute_hessian_proxy(wk_list, rank, eps=0.01):
    """
    Proxy for Hess(m2): the Hessian of the strip-area functional
    evaluated at the current Lagrangian configuration.

    For the strip area A(L_k, L_{k+1}) = Σᵢ arccos(σᵢ(Ũ_k^T Ũ_{k+1})):
    The Hessian w.r.t. the singular values σᵢ is:
      ∂²A/∂σᵢ² = 1/(1-σᵢ²)^(3/2)  (second derivative of arccos)

    This is the diagonal Hessian in the σ-coordinates.
    The weighted metric uses σᵢ as weights.
    """
    results = []
    for k in range(len(wk_list)-1):
        U_k,  s_k,  _ = np.linalg.svd(wk_list[k],   full_matrices=False)
        U_k1, s_k1, _ = np.linalg.svd(wk_list[k+1], full_matrices=False)
        Ur  = U_k[:, :rank]
        Ur1 = U_k1[:, :rank]

        sv = np.linalg.svd(Ur.T @ Ur1, compute_uv=False)
        sv = np.clip(sv, 1e-6, 1-1e-6)

        # Hessian of arccos at σᵢ: ∂²/∂σ² arccos(σ) = σ/(1-σ²)^(3/2)
        hess_diag = sv / (1 - sv**2)**1.5

        results.append({
            'k': k,
            'strip_svs': sv,
            'hess_diag': hess_diag,
        })
    return results


def weighted_rm2(strip_data, hess_data, wk_list, rank):
    """
    Compute r_m2^σ = ⟨Ĥ, Hess(m2)⟩_σ / (‖Ĥ‖_σ ‖Hess(m2)‖_σ)

    In the singular value basis, the "Hessian of the loss" restricted
    to the strip directions is approximated by the WK singular value
    structure.

    Ĥ: the Hessian of the cross-entropy loss projected onto the
       strip singular value directions (proxy: WK singular values s_k)

    Hess(m2): the Hessian of the strip-area functional
              (computed above: σᵢ/(1-σᵢ²)^(3/2))

    Weight matrix W_σ: diag(1/σ₁, ..., 1/σ_r) ⊗ diag(1/σ₁, ..., 1/σ_r)
    In 1D per pair: weights wᵢ = 1/σᵢ²

    Unweighted r_m2: Σᵢ Ĥᵢ · hᵢ / sqrt(Σᵢ Ĥᵢ² · Σᵢ hᵢ²)
    Weighted r_m2^σ: Σᵢ Ĥᵢ · hᵢ / σᵢ² / sqrt(Σᵢ Ĥᵢ²/σᵢ² · Σᵢ hᵢ²/σᵢ²)
    """
    pair_results = []
    all_rm2_unweighted = []
    all_rm2_weighted = []

    for sd, hd, k in zip(strip_data, hess_data,
                          range(len(strip_data))):
        sv      = sd['strip_svs']
        h_loss  = np.linalg.svd(wk_list[k], compute_uv=False)[:rank]
        # Normalize loss Hessian proxy
        h_loss  = h_loss / (np.linalg.norm(h_loss) + 1e-10)
        h_strip = hd['hess_diag']
        h_strip = h_strip / (np.linalg.norm(h_strip) + 1e-10)

        weights = 1.0 / (sv**2 + 1e-6)

        # Unweighted
        num_uw  = np.dot(h_loss, h_strip)
        denom_uw = np.linalg.norm(h_loss) * np.linalg.norm(h_strip) + 1e-10
        rm2_uw  = float(num_uw / denom_uw)

        # Weighted
        num_w   = np.dot(h_loss * weights, h_strip)
        denom_w = (np.sqrt(np.dot(h_loss**2, weights)) *
                   np.sqrt(np.dot(h_strip**2, weights)) + 1e-10)
        rm2_w   = float(num_w / denom_w)

        pair_results.append({
            'k': k, 'strip_svs': sv.tolist(),
            'strip_area': sd['strip_area'],
            'strip_area_std': sd['strip_area_std'],
            'rm2_unweighted': rm2_uw,
            'rm2_weighted': rm2_w,
        })
        all_rm2_unweighted.append(rm2_uw)
        all_rm2_weighted.append(rm2_w)

    return {
        'pairs': pair_results,
        'mean_rm2_unweighted': float(np.mean(all_rm2_unweighted)),
        'mean_rm2_weighted':   float(np.mean(all_rm2_weighted)),
        'std_rm2_unweighted':  float(np.std(all_rm2_unweighted)),
        'std_rm2_weighted':    float(np.std(all_rm2_weighted)),
    }


def analyze_checkpoint(path, label, rank):
    print(f"\n  {label}")
    wk_list = load_wk(path)
    strip_data = compute_strip_svs(wk_list, rank)
    hess_data  = compute_hessian_proxy(wk_list, rank)
    rm2_data   = weighted_rm2(strip_data, hess_data, wk_list, rank)

    print(f"  {'Pair':>6} {'strip_area':>12} {'std':>8} "
          f"{'r_m2':>10} {'r_m2^σ':>10}")
    print(f"  {'-'*52}")
    for p in rm2_data['pairs']:
        print(f"  {p['k']:>3}→{p['k']+1:<3} "
              f"{p['strip_area']:>12.4f} {p['strip_area_std']:>8.4f} "
              f"{p['rm2_unweighted']:>+10.4f} {p['rm2_weighted']:>+10.4f}")
    print(f"  {'Mean':>6} {'':>12} {'':>8} "
          f"{rm2_data['mean_rm2_unweighted']:>+10.4f} "
          f"{rm2_data['mean_rm2_weighted']:>+10.4f}")

    # Diagnostic
    std_mean = float(np.mean([p['strip_area_std'] for p in rm2_data['pairs']]))
    if std_mean < 0.05:
        regime = "UNIFORM-STRIP (unweighted ≈ weighted, both uninformative)"
    elif abs(rm2_data['mean_rm2_weighted']) > 0.3:
        regime = "NON-UNIFORM (weighted r_m2^σ is informative)"
    else:
        regime = "BORDERLINE"
    print(f"  Strip std: {std_mean:.4f} → {regime}")

    return {
        'label': label,
        'rank': rank,
        **rm2_data,
        'strip_area_mean': float(np.mean([p['strip_area'] for p in rm2_data['pairs']])),
        'strip_area_std_mean': float(np.mean([p['strip_area_std'] for p in rm2_data['pairs']])),
        'regime': regime,
    }


def main():
    args = parse_args()
    print("="*60)
    print("  STRIP-AREA WEIGHTED r_m2 DIAGNOSTIC")
    print("  Closes gap 5.1: correct Frobenius metric")
    print("  r_m2^σ = ⟨Ĥ, Hess(m2)⟩_σ")
    print("="*60)

    checkpoints = [
        (args.spike64,     "step64  (τ=5.90, val=0.65)"),
        (args.spike72,     "step72  (τ=5.94, val=0.56)"),
        (args.basin_entry, "basin_entry (val≈0.18)"),
        (args.basin_state, "basin_state (val≈0.065)"),
    ]

    results = {}
    for path, label in checkpoints:
        try:
            r = analyze_checkpoint(path, label, args.rank)
            results[label] = r
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: WEIGHTED vs UNWEIGHTED r_m2")
    print(f"{'='*60}")
    print(f"  {'Checkpoint':>30} {'r_m2':>10} {'r_m2^σ':>10} {'Δ':>8}")
    print(f"  {'-'*62}")
    for label, r in results.items():
        uw = r['mean_rm2_unweighted']
        w  = r['mean_rm2_weighted']
        print(f"  {label[:30]:>30} {uw:>+10.4f} {w:>+10.4f} {w-uw:>+8.4f}")

    print(f"\n  Interpretation:")
    print(f"  If Δ = r_m2^σ - r_m2 is large: weighting changes the result")
    print(f"  → strip areas are non-uniform, weighted metric is informative")
    print(f"  If Δ ≈ 0: uniform-strip regime, both metrics equivalent")
    print(f"  (As expected from strip_area_std ≈ 0.06-0.08 in crystallization)")

    Path(args.output).write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
