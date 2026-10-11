# Pre-analysis amendment: C19PRC-UK, waves 1–6 (primary), with the Irish strand as replication

**Fixed on 2026-10-10.** This was written after reading only the archive listing and the SPSS metadata (variable names, labels and value labels) of the six UK wave files. No data values were examined. It takes precedence over the order in `PREANALYSIS_AMENDMENT_C19PRC_IE.md`: the **UK strand is the primary analysis** and the **Irish strand is the replication**. All rules carry over from the earlier plans unless changed below.

## Data
- `C19PRC_UKW1W2_archive_final.sav` (W1 and W2), `C19PRC_UK_W3_archive_final.sav`, `C19PRC_UK_W4_archive_final.sav`, `C19PRC_UKW5_archive_final.sav`, `C19PRC_UK_W6_archive_final.sav`.
- Waves are linked by `pid`. Waves 7–9 exist only as a PDF report and are not used.

## States
- **D_t:** the provided PHQ-9 total for each wave (`W1_Depression_Total`, `W2_Depression_Total`, `W3_Dep_Total`, `W4_Dep_Total`, `W5_Dep_Total`, `W6_Dep_Total`).
- **A_t:** the GAD-7 total (`W{t}_GAD_Total`).
- Bands for both: 0–4 (0), 5–9 (1), 10 or more (2).
- Negative or out-of-range totals are treated as missing.
- If a total variable named here does not exist, the items are summed instead (complete items only).

## Analyses (UK)
- **A.** The cross-sectional system (D_t, A_t) at each wave, all respondents, unweighted:
  - regimes between consecutive waves;
  - mutual information;
  - the number of vertices of the 3×3 polytope.
- **B1.** (D1, D2, D3) on complete cases: the gluing gap, the Markov null, and the stable, drifting and latent Markov models.
- **B2.** The first four-wave run with at least 200 complete cases, in the order W1–W4, W2–W5, W3–W6:
  - the latent first-order Markov model over four waves (3 classes, time-varying transitions, one response law);
  - G² with a parametric-bootstrap p-value (200 draws).
  - Decision: "latent Markov suffices" if p > 0.05; otherwise "longer memory".
  - The same model is also fitted with a free response law at each wave, as a sensitivity analysis.
- **B3.** If at least 200 respondents are complete on all six waves, the conditional gaps I(D_{t−2}; D_t | D_{t−1}) for t = 3…6, each with its Markov null. This is descriptive; no latent model is fitted on six waves.
- **C.**
  - Path (D1, D2, D3); outcome `W6_Suicide_Ideation` (1 = yes, 0 = no; −9 = missing).
  - This item asks about *ever* having such thoughts, so the result is an association only.
  - Reported if n ≥ 100 and at least 10 respondents answer yes.
- **D.** Panel conditioning: recontacted against fresh (top-up) respondents at W3 (`W3_Type` 0 against 1), W4 (`W4_Type` 1 against 2) and W6 (`W6_Type` 1 against 0).
  - Contents (D_t, A_t), 3×3.
  - Canonical CbD with the boundary null (400 draws), Fisher over the three waves; contextuality only if p < 0.05.
- **E.** Attrition from W1 to W3:
  - tables (D1, A1) on all W1 respondents; (D1, D3) and (A1, D3) on the W1 respondents present at W3;
  - IPW on the (D1, A1) cells, and `W3_weight` as secondary weights.

## Replication (Irish strand, `MHASAP.ALLWAVES.sav`)
The same B1, B2, C, D and E, using the Irish variable names in `PREANALYSIS_AMENDMENT_C19PRC_IE.md`, with these substitutions:
- outcome for C: `W4_Suicidal_Ideation_1week`;
- D contexts: recontact against fresh at W2, W4 and W5;
- E secondary weights: `W3_Weight_Variable`.

A decision is called replicated if both strands reach it.

## Order
1. Adapter tests on synthetic data with planted answers (latent Markov; and planted second-order memory, which B2 should detect).
2. Run the UK analysis and freeze it.
3. Run the Irish analysis and freeze it.
4. Compare.
