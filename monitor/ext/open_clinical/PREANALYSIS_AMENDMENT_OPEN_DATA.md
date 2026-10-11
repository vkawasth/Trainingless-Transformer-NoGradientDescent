# Pre-analysis amendment: open clinical datasets

**Fixed on 2026-10-10.** This was written before any variable values in these files were examined. Only the archive listings were opened (file names and integrity), along with the header line of `Outcomes-a.txt`.

This amendment extends `PREANALYSIS_PLAN.md` (MIMIC-IV, dated 2026-10-09) to three open datasets. Analyses A–E, the decision rules, the suppression threshold (cells under 10) and the random seed (20261009) are unchanged unless stated below.

## I. PhysioNet/CinC Challenge 2012
**Data.** Set A (4,000 ICU stays) with `Outcomes-a.txt` is the analysis sample. Set B with `Outcomes-b.txt` is the replication sample. Set B is analysed only after the set A results are frozen.

**Definitions** (as in the MIMIC plan, with these deviations):
- **Windows:** hours [0, 6), [21, 27) and [42, 48) from ICU admission (record time 00:00).
- **H_t:** the median of `MAP` in the window; if there is none, the median of `NIMAP`.
  - Bands: shock below 65, borderline 65 to below 75, stable at least 75.
  - *Deviation:* the data have no vasopressor variable, so shock is defined by MAP alone.
- **R:** the first `Creatinine` in [0, 24] h, banded at 1.2 and 2.0.
- **L:** the first `Lactate` in [0, 24] h, banded at 2 and 4.
- **Ventilation:** any `MechVent` = 1 in [0, 24) h.
- **Outcome:** `In-hospital_death`.
- **Cohort:** every record with an `ICUType` value. There is no length-of-stay filter, because every record covers 48 h; records with H missing in a window drop out of B and C only.

**Analyses:**
- **A (coupling geometry).** There are no calendar periods. The contexts are the four ICU types: system (H1, R), regimes between each pair of types, and the toric type of the 2×3 system.
  - *Negative control:* set A against set B, which should be a bend with stress below 1 if the two are samples of one population.
- **B, C:** as planned.
- **D:** contexts are `ICUType` 3 (medical) against 4 (surgical), with strata H1, as planned.
  - Sensitivity analysis: strata by SOFA band (0–4, 5–8, 9 and above).
- **E:** as planned.

**Replication rule.** A finding is called replicated if set B reaches the same decision under the same rule.

## II. Diabetes 130-US Hospitals (UCI, 1999–2008)
**Data.** `diabetic_data.csv` and `IDS_mapping.csv`.
- Unit: encounter.
- The order of a patient's encounters is taken from increasing `encounter_id`. *Assumption:* this is chronological. It is flagged as a limitation.

**Definitions:**
- **Insulin state:** `insulin` = No (0), Steady (1), Up or Down (2).
- **A1C:** `A1Cresult` = None means not measured. If measured, the bands are Norm (0), >7 (1), >8 (2).
- **Readmission Y:** `readmitted` = NO (0), >30 (1), <30 (2).
- **Age band X:** below 50 (0), 50 to below 70 (1), 70 and above (2), from `age` deciles.
- **Admission source:** emergency room (`admission_source_id` 7) against physician referral (1), using `IDS_mapping.csv`.

**Analyses:**
- **A.** Coupling geometry of (insulin, Y) in the two admission sources (descriptive), with the toric type of (`change` Ch or No) × Y.
- **B.** For patients with at least 3 encounters, take the insulin state at encounters 1–3. Compute the gluing gap, the Markov null, and the stable, drifting and latent Markov models.
- **C.** Outcome: readmitted under 30 days at encounter 3. Compare the encounter-1 state, the multiset and the order, by the same permutation tests as before.
- **D.**
  - Contents: `change` (2 values) and Y (3 values).
  - Contexts: emergency room against physician referral.
  - Strata: age band X.
  - Tests: canonical CbD with boundary null, combined by Fisher's method; contextuality is claimed only if p < 0.05.
- **E.** This is the test of selective measurement.
  - Tables: (X, Y) on all encounters; (X, A1C) and (Y, A1C) on encounters where A1C was measured.
  - Contextual fraction available-case, after IPW on (X, Y), and after IPW from a logistic model of measurement on X, Y, admission source and admission type. The second weighting is fixed here as a secondary analysis.
  - Sampling null as planned.

## III. Heart Disease (UCI, four sites)
**Data.** `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data` and `processed.va.data` (14 attributes; `?` means missing).

**Analyses:**
- **D.**
  - Contents: chest pain `cp`, banded as angina (types 1 and 2), non-anginal (3) and asymptomatic (4), and exercise angina `exang` (2 values).
  - Contexts: each pair of sites (6 pairs). Strata: diagnosis (`num` = 0 against above 0).
  - Tests: canonical CbD with boundary null, Fisher over strata, then Bonferroni over the 6 pairs.
- **E (descriptive):** the share missing for each attribute by site. A contextual fraction for (age band, `num`, `thal`) is computed only if every table has at least 10 per cell; otherwise it is reported as not estimable.
- This dataset is small (about 920 patients). Its results are reported as illustrations, not as tests of the theory.

## Order of work
1. Write the adapters and test them on synthetic data in each layout.
2. Run set A, Diabetes 130 and Heart Disease, and freeze the results.
3. Run set B (replication).
4. Report every pre-specified result, whichever way it comes out. Anything not listed here is labelled exploratory.
