# Pre-analysis plan: the structural-change monitor on MIMIC-IV

**Fixed on 2026-10-09, before any MIMIC-IV data were examined.** This plan carries out the protocol of Bands §11. It is
implemented exactly by `mimic_monitor.py`. Any change after the data are seen must be dated and reported as a deviation.

## 1. Data and cohort
- MIMIC-IV, version 3.1, modules `hosp` and `icu`. The analysis runs only on the credentialed user's machine.
- Cohort: the first ICU stay of each patient, with age at least 18 (`anchor_age`) and ICU length of stay at least 2 days.
- Time windows after ICU admission: W1 = [0, 6) h, W2 = [21, 27) h, W3 = [42, 48) h.

## 2. Definitions (fixed thresholds)
- **Hemodynamic state H_t** in window t:
  - shock (0): any vasopressor infusion overlaps the window, or the median mean arterial pressure (MAP) is below 65 mmHg;
  - borderline (1): median MAP from 65 to below 75 mmHg;
  - stable (2): median MAP at least 75 mmHg.
  - Vasopressors: norepinephrine 221906, epinephrine 221289, phenylephrine 221749, vasopressin 222315, dopamine 221662 (`inputevents`).
  - MAP: itemids 220052, 220181, 225312 (`chartevents`), values from 10 to 250 mmHg.
  - The script checks every itemid against `d_items` and `d_labitems` and stops if a label does not match.
- **Creatinine band R**: the first value from 6 h before to 24 h after ICU admission (itemid 50912): below 1.2, 1.2 to below 2.0, at least 2.0 mg/dL.
- **Lactate band L**: same rule (itemid 50813): below 2, 2 to below 4, at least 4 mmol/L. "Measured" means a value exists.
- **Invasive ventilation within 24 h**: itemid 225792 (`procedureevents`) starting before 24 h.
- **Outcome**: in-hospital death (`hospital_expire_flag`).
- **Contexts**:
  - calendar periods (`anchor_year_group`);
  - care units (MICU against SICU);
  - severity strata (H1).

## 3. Analyses and decision rules
- **A. Coupling geometry** (Bands Thm 3.3, Bridges Prop 8.1)
  - System: (H1, R) in each calendar period.
  - The regime of each step between consecutive periods (bend, tear?, tear, break), with 95% simultaneous bands, stress and tear margin.
  - The mutual information.
  - The toric surface of the 2×3 system (pressor or shock in W1, against creatinine band), and whether its chamber is decided on the bands.
  - Falsifiable claim: a decided tear coincides with a change in practice visible in the margins. This part is reported descriptively.
- **B. Composition** (Bands Prop 4.8)
  - The gluing gap I(H1; H3 | H2) with a bootstrap interval, against a Markov null of 400 resamples.
  - Three latent-class models (3 classes), each with G² and its p-value: stable, drifting, latent Markov.
  - Decision rule:
    - "the answers are not glued" if the gap exceeds the null's 95th percentile;
    - "consistent with a glued latent chain" if the latent Markov model has p > 0.05.
- **C. Path and outcome**
  - Does the multiset {H1, H2, H3} add to H1 in its association with death? Does the order add to the multiset?
  - Permutation tests with 2000 permutations; leave-one-out bits.
  - Caution fixed in advance: with measurement error, the order of the observed states can carry information about the latent states. A rejection of "order adds nothing" is therefore not by itself evidence that the outcome depends on the latent path.
- **D. Contextuality across care units within severity strata** (Bridges §4, canonical form)
  - Contents: H2 (3 values) and ventilation in 24 h (2 values). Contexts: MICU and SICU. Strata: H1.
  - Report the naive contextual fraction and the canonical deficit.
  - Boundary-null p-value with 400 simulations per stratum, combined over the strata by Fisher's method.
  - Decision: contextuality is claimed only if the combined p < 0.05.
  - Care units are not randomized. A positive result would be read as unmeasured differences between units, not as contextuality in the physical sense.
- **E. Selection** (Bands §11, item 4)
  - The contextual fraction of the three pairwise tables (H1, R), (H1, L), (R, L): (H1, R) on all stays, the other two on stays with lactate measured.
  - The same with the lactate tables weighted by 1 / P(measured | H1, R).
  - Sampling null: 400 resamples from the IPF joint of the weighted tables.
  - Decision rule:
    - "selection explains the incompatibility" if the available-case fraction exceeds the null's 95th percentile and the weighted one does not;
    - "a residual remains" if both exceed it.

## 4. Disclosure
- Only aggregate results are written: statistics, p-values, and proportion tables whose every cell count is at least 10 (otherwise suppressed).
- No row-level data are written, printed or transmitted.
- Results are reviewed against the PhysioNet data use agreement before they are shared.

## Amendment, 2026-10-10 (before any MIMIC-IV run): analysis E
- **The cell-weighted fraction is not a test.** Weighting on the (H1, creatinine) cells makes the three tables compatible by construction, so `cf_after_ipw` is 0 whatever the selection. Only the available-case test is evidence.
- **Restricted selection models are added.** Their weights are not built on the triangle cells:
  - H1-only;
  - creatinine-only;
  - external (admission type and anchor-age band < 50, 50–69, 70+);
  - H1 + external.
  Each has its own null, recomputed by parametric bootstrap: the tables are compatible and selection follows the model (200 draws; 50 with `--quick`).
  - A model is *consistent* if its weighted fraction is at or below the null's 95th percentile.
  - Any weights that reproduce the full (H1, creatinine) table make the triangle compatible. So these tests check only what each model implies about the (H1, creatinine) composition of the stays in which lactate was measured. Selection that depends on lactate itself is invisible to them.
- **Synthetic check.** The planted creatinine-only selection is found consistent, and the H1-only and external models are rejected (`check_synthetic.py`, 9/9).
- **Cohort query.** It now also reads `admissions.admission_type`.
