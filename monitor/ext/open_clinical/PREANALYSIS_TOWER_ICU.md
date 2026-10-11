# Pre-analysis amendment: a real three-level tower on PhysioNet 2012, Fano tubes of its gradations, and the interchange recheck of the ICU bands

**Fixed on 2026-10-10.**
- This amendment was written after the analyses of `PREANALYSIS_AMENDMENT_OPEN_DATA.md` (Part I) on sets A and B were frozen.
- No model below has been fitted, and no quantity below has been computed.
- The only values already known are those reported in the earlier analyses: band frequencies, death counts and the banded composition results.
- Seed: 20261010.

## Part 1. The tower (Tower §9–10; Bands §9)
**Outcome.** In-hospital death.

**Covariates.** The banded covariates of the earlier loader, with "missing" as an extra level:
- H1, the first-window pressure band (0, 1, 2, missing);
- R, the creatinine band (0, 1, 2, missing);
- vent24;
- ICU type;
- SOFA band (0–4, 5–8, 9+, missing).

**Candidate models (level 2).** Five models, each fitted by maximum-likelihood logistic regression on the fit sample:
- M0: base rate;
- M1: H1;
- M2: H1 + R;
- M3: H1 + R + vent24 + ICU type;
- M4: SOFA band.

**Reliability (level 3).** Each model k is transported with reliability α ∈ {0, 1/4, 1/2, 3/4, 9/10, 1}:

    ρ_k^α(y|x) = α ρ_k(y|x) + (1−α) ρ_0

where ρ_0 is the fit sample's base rate. Discounting towards the base rate, not the uniform law, is fixed in advance, because a uniform floor of ½ is meaningless for a 14% outcome. Bands' uniform discount is the special case of a different floor.

**Prior.** Uniform over the 5 × 6 hypotheses (k, α).

**Evidence and test.** The patients of the evidence sample are conditioned on in RecordID order, sequentially. The tower code is `condS` of the base-relative tower, as in T1.

**Arms.**
- **Same population (negative control for level 3).**
  - Fit on set A.
  - Evidence: set B, first half by RecordID (B1).
  - Test: second half (B2).
- **Shift (where level 3 should activate).**
  - Fit on set A, cardiac-surgery and surgical ICUs.
  - Evidence: the first half by RecordID of set B coronary and medical ICUs.
  - Test: their second half.

**Primary decision, per arm.** Compare log loss on the test half between:
- the full tower (levels 2 and 3) after the evidence;
- flat Bayesian model averaging (α = 1 only) after the same evidence.

Δ = logloss(flat) − logloss(tower), with a paired bootstrap over test patients (2000 draws, percentile 95% interval). Level 3 **earns its keep** if the interval excludes 0 on the positive side.

**Expected (stated now).**
- Same population: Δ not positive beyond noise, with the posterior over α concentrated at α ≥ 3/4.
- Shift: Δ > 0, with posterior mass on α < 1 above ½.
- Either outcome is reported.

**Secondary.**
- Posterior over models and over α.
- Level-1 calibration (predicted against observed death rate on the test half).
- Coherence μΩ = Π, μΠ = ρ at every step.

## Part 2. Fano tubes of the tower gradations (Bridges Prop. 8.1, Thm. 8.2)
**The system.** Death (binary, first margin f₁ = P(death)) × H1 band (three values, margin g), on the test half's patients with H1 observed. Its coupling polygon is the moment polygon of a toric Fano surface (CP², CP¹×CP¹, F₁, Bl₂CP², Bl₃CP²). The surface is fixed by the chamber of (f₁, g) and computed with `toric_public.analyse23`.

**Each tower gradation predicts f₁, with g fixed at the observed H1 distribution.**
- Level 1: the tower's predictive death rate, averaged over test patients with H1.
- Level 2: one predictive rate per model.
- Level 3: one per reliability.

**Fano tube.**
- The cross-section at evidence count n ∈ {0, 50, …, end} is the polygon at the level-1 prediction.
- Level-2 and level-3 cross-sections are drawn with their posterior weights.

**Reported.**
- The posterior probability of each toric surface: Σ over (k, α) of the posterior weight × 1[surface of (f₁^{kα}, g)].
- The surface at the observed test death rate, and whether its chamber is decided on the bands.
- The evidence counts at which the level-1 cross-section changes surface (tears of the tube).

**Decision.** The tower **decides the surface** if one surface has posterior probability ≥ 0.95. Otherwise the tower carries genuine uncertainty about the toric type.

## Part 3. Interchange recheck of the ICU pressure bands (Algebras Prop. 6.1)
**States.**
- Raw: the window median MAP, in 5-mmHg bins from 40 to 120, with values below 40 put in the first bin and 120 or above in the last (17 states).
- Banded: below 65, 65–75, above 75. Bin edges include 65 and 75, so banding is an exact merge of the bins.

**Test.** Exactly as in `c19/PREANALYSIS_BANDING_INTERCHANGE.md`, on the patients with all three windows:
- the banded gap;
- the monitor's banded Markov null;
- the interchange-corrected raw-Markov null;
- the decomposition O − R = (O − L) + (L − R);
- the sufficiency diagnostic.

Set A is primary and set B the replication.

**Decision.** As in the C19PRC plan. The earlier ICU conclusion ("answers not glued") stands only if the banded gap exceeds the corrected null.

**Disclosure.** The 5-mmHg bins are themselves a merge of a continuous median, so this tests the coarse bands relative to 5-mmHg resolution, not to the continuous value.

## Adapter checks before any real run
1. **Tower.** Synthetic patients with a planted transport reliability α* = 1/2: data generated from ρ_3^{1/2}. The posterior over α should concentrate near 1/2, and the tower should beat flat averaging. With α* = 1, it should not beat flat averaging beyond noise.
2. **Interchange.** The C19PRC calibration applies, since the code is the same with a general state count. It is rerun with 17 raw states on planted chains: size and power, 20 replicates each.
