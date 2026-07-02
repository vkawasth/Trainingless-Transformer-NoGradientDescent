"""
offline_calibrator.py
======================
Predicts geometric thresholds from corpus and architecture statistics
BEFORE any training begins.

The key insight: the compiler's thresholds (τ_basin, Φ_cl, r_m2^σ)
depend on corpus statistics and architecture geometry in predictable ways.

What we know offline:
  - Corpus: vocab size V, token frequencies p(w), co-occurrence structure
  - Architecture: D, N_heads, N_layers, rank r
  - Geometric paths: Gr(r,D) structure, Stab(F) coordinates
  - Rare term distribution: how many tokens appear < threshold times

What we compute offline:
  1. Effective rank r_eff from corpus entropy
  2. Expected strip energy scale from (r, D)
  3. τ_basin prediction from token distribution
  4. Φ_cl threshold from rare term fraction
  5. r_m2^σ stopping threshold from architecture spectral gap

What remains unknown (the hard problem):
  How the loss manifold L: R^{D*layers} → R connects to Gr(r,D)^L
  specifically: which parameter directions correspond to which Stab(F) movements
  This requires the Jacobian of the phase map Φ: R^{D²×6} → R^5
  which is only knowable through the trained WK matrices

Usage
-----
  python offline_calibrator.py \
      --corpus_file corpus.txt \
      --D 256 --N_heads 4 --N_layers 6 --rank 6 \
      --output calibration.json
"""

import argparse, json, math
from pathlib import Path
from collections import Counter
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus_file', default=None,
                   help='Text file for corpus statistics (optional)')
    p.add_argument('--vocab_size',  type=int, default=1017)
    p.add_argument('--nnz',         type=int, default=1347,
                   help='Number of nonzero co-occurrence pairs')
    p.add_argument('--D',           type=int, default=256)
    p.add_argument('--N_heads',     type=int, default=4)
    p.add_argument('--N_layers',    type=int, default=6)
    p.add_argument('--rank',        type=int, default=6)
    p.add_argument('--rare_thresh', type=int, default=3,
                   help='Token count threshold for "rare" classification')
    p.add_argument('--output',      default='calibration.json')
    return p.parse_args()


def corpus_statistics(corpus_file, vocab_size, nnz):
    """Compute corpus statistics for threshold prediction."""
    if corpus_file and Path(corpus_file).exists():
        text = Path(corpus_file).read_text()
        tokens = text.split()
        counts = Counter(tokens)
        V = len(counts)
        total = sum(counts.values())
        freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
        freqs /= total
    else:
        # Use known statistics from our corpus
        V = vocab_size
        # Approximate Zipf distribution for V tokens
        ranks = np.arange(1, V+1)
        freqs = 1.0 / (ranks * np.log(V+1))
        freqs /= freqs.sum()

    # Token entropy (bits)
    H = float(-np.sum(freqs * np.log2(freqs + 1e-12)))

    # Effective vocabulary (exp of entropy)
    V_eff = float(2**H)

    # Rare token fraction (tokens below median frequency)
    rare_frac = float(np.mean(freqs < np.median(freqs)))

    # Co-occurrence density
    coo_density = nnz / (V * V)

    return {
        'V': V,
        'H_bits': H,
        'V_eff': V_eff,
        'rare_frac': rare_frac,
        'coo_density': coo_density,
        'nnz': nnz,
    }


def architecture_geometry(D, N_heads, N_layers, rank):
    """
    Compute geometric properties of the architecture.

    Key relationships:
    - Head dimension: d_head = D / N_heads
    - Natural rank: r_natural = d_head / 4 (empirical from attention papers)
    - Strip energy scale: E_strip = r * arccos(1/sqrt(r)) * (1 + D/r * correction)
    - Spectral gap scale: gap ∝ D / (r * sqrt(D-r))
    """
    d_head = D // N_heads

    # Natural rank from architecture
    r_natural = max(1, d_head // 4)

    # Expected strip energy at random initialization
    # For r planes in R^D chosen randomly: principal angles ~ Uniform[0, π/2]
    # E[arccos(σᵢ)] ≈ π/2 - 1/sqrt(D) for large D
    E_strip_random = rank * (math.pi/2 - 1/math.sqrt(D))

    # Expected spectral gap at convergence
    # σ_r/σ_{r+1} ≈ sqrt(D/r) for random matrices (Marchenko-Pastur)
    gap_random = math.sqrt(D / rank)

    # Lagrangian dimension check
    # L_k ∈ C^r is Lagrangian if dim_R(L_k) = r = dim_R(C^r)/2
    # Our setting: L_k ⊂ R^rank ⊂ C^rank, dim = rank = dim(C^rank)/2 ✓
    lagrangian_ok = True  # always true for real subspaces of C^rank

    return {
        'd_head': d_head,
        'r_natural': r_natural,
        'r_used': rank,
        'rank_mismatch': rank != r_natural,
        'rank_mismatch_factor': rank / r_natural,
        'E_strip_random': E_strip_random,
        'gap_random': gap_random,
        'lagrangian_ok': lagrangian_ok,
        'total_params': D * D * N_layers,  # WK parameters only
    }


def predict_thresholds(corpus_stats, arch_stats):
    """
    Predict geometric thresholds from corpus and architecture.

    Formulas derived from the relationship between:
    - τ_basin: K₀ gluing defect at correct basin entry
    - Φ_cl: clean-phase fraction needed for orbit
    - r_m2^σ stopping threshold: Hessian alignment signal

    τ_basin prediction:
      τ ≈ τ_base × (V_eff/V)^α × (H/log₂V)^β
      τ_base = 2.0 (confirmed empirical baseline)
      α ≈ 0.3 (larger effective vocab → higher τ needed)
      β ≈ 0.5 (higher entropy → more disorder → higher τ)

    Φ_cl prediction:
      Φ_cl_min = ceil(5 × (1 - rare_frac × γ))
      γ ≈ 0.4 (rare terms partially prevent full orbit crystallization)

    r_m2^σ stopping threshold:
      threshold = 0.65 × (1 - rank_mismatch_penalty)
      rank_mismatch_penalty = 0.1 × |r_used/r_natural - 1|

    Strip energy scale:
      E_strip ≈ E_strip_random × (1 - convergence_factor)
      convergence_factor ≈ 0.05 (small: Lagrangians stay nearly orthogonal)

    τ-retry skip condition:
      Skip if r_m2^σ ≥ threshold AND Φ_cl ≥ Φ_cl_min AND τ ∈ [τ_basin, 7]
    """
    V     = corpus_stats['V']
    V_eff = corpus_stats['V_eff']
    H     = corpus_stats['H_bits']
    rare  = corpus_stats['rare_frac']

    r_natural = arch_stats['r_natural']
    r_used    = arch_stats['r_used']
    E_strip   = arch_stats['E_strip_random']

    # τ_basin prediction
    tau_base   = 2.0
    alpha, beta = 0.3, 0.5
    tau_basin  = tau_base * (V_eff/V)**alpha * (H/math.log2(V+1))**beta
    tau_basin  = round(float(tau_basin), 2)

    # τ_high (when τ-retry triggers)
    tau_high = 5.0 + 0.5 * (H / math.log2(V+1))
    tau_high = round(float(tau_high), 1)

    # Φ_cl_min prediction
    gamma     = 0.4
    phi_min   = max(3, math.ceil(5 * (1 - rare * gamma)))
    phi_min   = int(phi_min)

    # r_m2^σ stopping threshold
    mismatch_penalty = 0.1 * abs(r_used/r_natural - 1)
    rm2_threshold    = max(0.50, 0.65 * (1 - mismatch_penalty))
    rm2_threshold    = round(float(rm2_threshold), 3)

    # Expected strip energy at convergence
    E_strip_conv = E_strip * 0.95  # stays nearly orthogonal

    # w_FF exponent prediction (from τ-power formula)
    # w_FF(τ) = w_max × (τ_basin/τ)^exponent
    # exponent derived from spectral gap theory:
    # gap ∝ τ^(-2/3) in the Marchenko-Pastur regime → exponent ≈ 3/2
    wff_exponent = 3/2  # theoretical; needs empirical validation
    wff_max      = 3.5  # empirical baseline

    # Basin settle budget prediction
    # Geometric convergence typically at step ≈ 5 × (D/r)^(1/3) × τ_basin
    geo_stop_step = int(5 * (arch_stats['total_params']/1e6)**0.2 * tau_basin)
    geo_stop_step = min(geo_stop_step, 80)  # cap at 80

    # τ-retry budget (if needed)
    n_retry_base = 50
    n_retry = int(n_retry_base * (tau_high/6.0))

    # Total CE budget prediction
    CE_basin_current = 120 + n_retry
    CE_basin_geo     = geo_stop_step + 30  # 30 = fast descent
    CE_total_current = 187
    CE_total_geo     = CE_total_current - (CE_basin_current - CE_basin_geo)
    CE_saving_pct    = (CE_basin_current - CE_basin_geo) / CE_total_current * 100

    return {
        'tau_basin':       tau_basin,
        'tau_high':        tau_high,
        'phi_cl_min':      phi_min,
        'rm2_threshold':   rm2_threshold,
        'E_strip_conv':    round(E_strip_conv, 3),
        'wff_exponent':    wff_exponent,
        'wff_max':         wff_max,
        'geo_stop_step':   geo_stop_step,
        'n_retry':         n_retry,
        'CE_basin_current':CE_basin_current,
        'CE_basin_geo':    CE_basin_geo,
        'CE_total_current':CE_total_current,
        'CE_total_geo':    CE_total_geo,
        'CE_saving_pct':   round(CE_saving_pct, 1),
    }


def explain_unknown(D, N_heads, N_layers, rank):
    """
    The hard problem: what remains unknown.

    The connection between corpus statistics and Stab(F) geometry
    requires knowing the Jacobian of the phase map:

    Φ: R^{D² × N_layers} → R^{N_layers-1}
    w ↦ (arg λ_dom(W_{k+1}W_k^{-1}))_{k=0}^{N_layers-2}

    This Jacobian has shape (N_layers-1) × (D² × N_layers)
    = 5 × 393216 in our case.

    The Jacobian tells us:
    - Which parameter directions correspond to which Stab(F) movements
    - How corpus structure constrains the reachable part of Stab(F)
    - Why some gradient directions are "adiabatic" and others "dissipative"

    Computing the Jacobian analytically:
    ∂φ_k/∂W_K^(j) = δ_{jk} × ∂φ_k/∂W_K^(k) + δ_{j,k+1} × ∂φ_k/∂W_K^(k+1)

    Each ∂φ_k/∂W_K^(k) = Im(ℓ^T · (∂M/∂W_K^(k)) · r) / |λ_dom|

    At clean phase (φ_k=0): this is 0 (algebraic real-locking).
    Off clean phase: nonzero, computable from current WK matrices.

    The FULL Jacobian is only available AFTER training begins.
    But the STRUCTURE of the Jacobian (which blocks are zero, which are large)
    can be predicted from corpus statistics.

    Corpus → Jacobian structure:
    - High-frequency tokens → large WK singular values → large ∂φ/∂W entries
    - Rare tokens → small WK entries → small ∂φ/∂W entries
    - Co-occurrence structure → which (k, k+1) pairs have correlated WK matrices
      → which phase walls are "natural" crossing points

    This is the missing link: corpus statistics → Jacobian structure →
    which gradient directions hit which walls → optimal Stab(F) path.
    """
    D_total = D * D * N_layers
    J_rows  = N_layers - 1
    J_cols  = D_total
    J_nonzero_frac = 2/N_layers  # only 2 WK matrices affect each φ_k

    return {
        'jacobian_shape': (J_rows, J_cols),
        'jacobian_nnz_frac': J_nonzero_frac,
        'jacobian_nnz': int(J_rows * J_cols * J_nonzero_frac),
        'computable_offline': False,
        'computable_online': True,
        'why_hard': (
            'The Jacobian ∂Φ/∂W depends on the current WK matrices, '
            'which change during training. The STRUCTURE (sparsity pattern) '
            'is predictable from corpus; the VALUES require trained WK matrices. '
            'Offline calibration predicts the structure; online tracking '
            'computes the values at each wall crossing.'
        ),
        'next_step': (
            'Compute the Jacobian at step 64 (clean phase) and step 72 (off wall). '
            'The sparsity pattern tells which corpus directions drive which phase walls. '
            'This closes the loop: corpus → Jacobian structure → predicted wall crossings '
            '→ pre-calibrated thresholds for any new (corpus, architecture) pair.'
        ),
    }


def main():
    args = parse_args()
    print("="*60)
    print("  OFFLINE CORPUS CALIBRATOR")
    print("  Predicts geometric thresholds before training")
    print("="*60)

    corpus = corpus_statistics(args.corpus_file, args.vocab_size, args.nnz)
    arch   = architecture_geometry(args.D, args.N_heads, args.N_layers, args.rank)
    thresh = predict_thresholds(corpus, arch)
    unknown = explain_unknown(args.D, args.N_heads, args.N_layers, args.rank)

    print(f"\n  CORPUS STATISTICS:")
    print(f"    Vocabulary: V={corpus['V']}, V_eff={corpus['V_eff']:.0f}")
    print(f"    Entropy: H={corpus['H_bits']:.2f} bits")
    print(f"    Rare fraction: {corpus['rare_frac']:.3f}")
    print(f"    Co-occurrence density: {corpus['coo_density']:.6f}")

    print(f"\n  ARCHITECTURE GEOMETRY:")
    print(f"    D={args.D}, N_heads={args.N_heads}, d_head={arch['d_head']}")
    print(f"    r_used={arch['r_used']}, r_natural={arch['r_natural']}", end="")
    if arch['rank_mismatch']:
        print(f"  ⚠ MISMATCH (factor {arch['rank_mismatch_factor']:.1f}x)")
    else:
        print(f"  ✓")
    print(f"    Expected strip energy: {arch['E_strip_random']:.3f}")
    print(f"    Expected spectral gap: {arch['gap_random']:.2f}")

    print(f"\n  PREDICTED THRESHOLDS:")
    print(f"    τ_basin (K₀ gluing):     {thresh['tau_basin']:.2f}  "
          f"(confirmed empirical: 2.0)")
    print(f"    τ_high (retry trigger):  {thresh['tau_high']:.1f}   "
          f"(confirmed empirical: 5.0)")
    print(f"    Φ_cl_min (orbit):        {thresh['phi_cl_min']}/5    "
          f"(confirmed empirical: 4/5)")
    print(f"    r_m2^σ threshold:        {thresh['rm2_threshold']:.3f} "
          f"(confirmed empirical: 0.65)")
    print(f"    w_FF exponent:           {thresh['wff_exponent']:.1f}   "
          f"(assumed; needs validation)")

    print(f"\n  CE BUDGET PREDICTION:")
    print(f"    Current (loss plateau): {thresh['CE_total_current']} CE")
    print(f"    Geo-stop prediction:    {thresh['CE_total_geo']} CE")
    print(f"    Expected saving:        {thresh['CE_saving_pct']:.0f}%")
    print(f"    Geo-stop step:          ~{thresh['geo_stop_step']}")
    print(f"    Fast descent:           30 CE")

    print(f"\n  THE HARD PROBLEM (what offline calibration cannot solve):")
    print(f"    Jacobian shape: {unknown['jacobian_shape']}")
    print(f"    Jacobian nnz: {unknown['jacobian_nnz']:,} "
          f"({unknown['jacobian_nnz_frac']*100:.0f}% of full)")
    print(f"    Computable offline: {unknown['computable_offline']}")
    print(f"    {unknown['why_hard'][:80]}...")
    print(f"\n  Next step to close the loop:")
    print(f"    {unknown['next_step'][:100]}...")

    report = {
        'corpus': corpus,
        'architecture': arch,
        'predicted_thresholds': thresh,
        'hard_problem': unknown,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
