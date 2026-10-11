# Pre-analysis amendment: does banding create the composition failure? (interchange test)

**Fixed on 2026-10-10.** This plan was written before any raw (unbanded) PHQ-9 score was examined. The banded results of `PREANALYSIS_AMENDMENT_C19PRC_UK.md` were already known (B1: UK gap 0.057 bits against a banded Markov null of 0.016). This is therefore a confirmatory test of a hypothesis suggested by the theory, not by the raw data. It is labelled a post-hoc amendment in the paper.

## The claim tested
Algebras Prop. 6.1 (interchange): for couplings α ∈ Cpl(p,q), β ∈ Cpl(q,r) and a middle kernel l,

    kᵀ(α·β)m = (kᵀαl)·(lᵀβm)

This holds when l is a bijection, or a merge that is sufficient on its fibres. It fails otherwise, and the difference is a circulation.

Banding a PHQ-9 total (0–27) into 0–4, 5–9 and 10+ is a merge b. Take k = l = m = b, α = the raw (D1, D2) coupling and β = the raw (D2, D3) coupling. Then:
- **L** = bᵀ(α·β)b is the banded image of the raw Markov composite;
- **R** = (bᵀαb)·(bᵀβb) is the banded Markov composite, which the monitor uses as its null.

For the observed banded (D1, D3) table **O**, the identity

    O − R = (O − L) + (L − R)

is exact. It splits the banded composition failure O − R into two parts:
- the raw composition failure, pushed through the bands: O − L;
- the interchange defect created by banding alone: L − R.

If banding is a non-sufficient merge, a raw first-order Markov chain has O − L ≈ 0 but L − R ≠ 0. In that case the monitor's banded Markov null is the wrong null, and the reported composition failure is partly made by the bands.

## Data
- UK (primary): the B1 sample of `c19.py`, respondents with the PHQ-9 total at W1, W2 and W3 (n = 950 expected). Totals are as in the UK amendment, out-of-range values missing, not banded.
- Ireland (replication): the Irish B1 sample (n = 418 expected), PHQ-9 item sums.

## Quantities
1. The identity O − R = (O − L) + (L − R), checked to 1e-12.
2. ½‖O − R‖₁, ½‖O − L‖₁ and ½‖L − R‖₁.
3. The banded gap I(B1; B3 | B2) in bits (as reported before), and the raw-chain gap.
4. Sufficiency diagnostic: within each band of D2, is the law of D1 (and of D3) given the raw D2 constant? Tested with a χ² test of homogeneity per band, Fisher-combined.

## Nulls (parametric bootstrap, 400 draws, seed 20261010)
- **Raw Markov null.** Simulate n respondents from P̂(D1_raw) · K̂₁₂ · K̂₂₃, using the empirical raw kernels. Recompute the banded gap and ½‖O − L‖₁ on each draw.
- **Banded Markov null (the monitor's).** As in the monitor, for the banded gap.

## Decision rules
- **Primary.** If the observed banded gap is at or below the 95th percentile of the raw Markov null, the banded composition failure is explained by banding a raw first-order Markov chain, i.e. by the interchange defect. Otherwise the raw chain has memory that banding does not explain.
- **Secondary.** If ½‖O − L‖₁ is at or below its raw Markov null 95th percentile, the raw composition failure is not detected.
- **Share.** The share of the banded failure along the interchange defect is ⟨O − R, L − R⟩ / ‖O − R‖², reported descriptively.

## Caveats fixed in advance
- The raw tables are sparse (28 values, about 950 respondents). The raw composite α·β is a plug-in estimate, and the raw Markov null uses the empirical kernels, so it inherits their noise. The bootstrap is the reference for both, for that reason.
- This tests the measurement layer only (banding). The latent Markov result (B2) concerns measurement error, which is a different kernel. The two explanations are not exclusive.
