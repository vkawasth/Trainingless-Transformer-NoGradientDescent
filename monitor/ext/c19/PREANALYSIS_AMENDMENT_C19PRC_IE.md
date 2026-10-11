# Pre-analysis amendment: C19PRC Ireland strand (Mental Health ASAP), waves 1–5

**Fixed on 2026-10-10.** This plan was written after reading only:
- the methods manuscript in the archive (Spikol et al.);
- the variable names, labels and value labels in the SPSS metadata of `MHASAP.ALLWAVES.sav` (2,686 rows, 1,381 variables).

No data values were examined. The plan extends `PREANALYSIS_PLAN.md` (2026-10-09) and `PREANALYSIS_AMENDMENT_OPEN_DATA.md` (2026-10-10). The seed (20261009), the suppression threshold (10) and the decision rules carry over unless changed below.

## Design facts used (from the methods manuscript)
- W1 (March/April 2020) recruited N = 1,041. The W1 cohort was followed for five waves, to March/April 2021.
- Fresh participants were added at W2, W4 and W5. W3 (N = 534) contains only W1 recontacts.
- W3 has an inverse-probability weight (`W3_Weight_Variable`), built from sex, age and province.
- Recontact indicators: `Recontact` / `W2_Wave1_Recontact` and `W4_Wave1_Recontact`. W3 contains only recontacts. Participants are identified by `pid`.

## States (fixed)
- **Depression D_t:** the PHQ-9 sum of items `DEP1`–`DEP9` (prefix `W{t}_` from W2 on; each item 0–3), complete items only.
  - Bands: 0–4 (0), 5–9 (1), 10 or more (2).
- **Anxiety A_t:** the GAD-7 sum (`GAD_1`–`GAD_7` at W1; `W2_GAD_1`…; `W3_GAD1`… from W3 on), complete items only. Same bands.
- **Suicidal ideation (outcome):** `W4_Suicidal_Ideation_1week`, coded yes if any value other than "no" (value labels as in the file).

## Analyses
- **A. Coupling geometry.** The cross-sectional system (D_t, A_t) at each wave, using all respondents of that wave.
  - Regimes between consecutive waves, with bands, stress and tear margin.
  - Mutual information, and the number of vertices of the 3×3 polytope at each wave.
- **B. Composition (the main test).** On the W1 cohort with D observed at the waves used:
  - **B1** (three waves, comparable with the GSS and ICU results): (D1, D2, D3). Report the gluing gap, the Markov null, and the stable, drifting and latent Markov models (3 classes).
  - **B2** (four waves, which three waves cannot give):
    - the longest run of four waves with at least 200 complete cases, chosen in the order (W1–W4), (W2–W5), (W1, W2, W3, W5);
    - the gaps I(D_{t−2}; D_t | D_{t−1}) at each step;
    - the latent Markov model over four waves (time-varying transitions, one response law, 3 classes);
    - p-value from a parametric bootstrap of G² (200 draws), because the 81-cell table is sparse.
  - **Decision for B2:**
    - "latent first-order Markov suffices" if the bootstrap p > 0.05;
    - otherwise "longer memory than a latent Markov chain".
  - If no run has 200 cases, B2 is reported as not estimable.
- **C. Path and outcome.** On respondents with D1, D2, D3 and the W4 suicidal-ideation item:
  - the W1 state, the multiset and the order, by permutation tests;
  - reported only if at least 100 respondents and at least 10 with ideation; otherwise not estimable.
  - The caution about measurement error from the MIMIC plan applies.
- **D. Panel conditioning as a context.**
  - Contents (D_t, A_t), 3×3. Contexts: recontacted against fresh respondents at the same wave (W2, W4, W5).
  - Canonical CbD with the boundary null (400 draws), combined over the three waves by Fisher's method.
  - Contextuality is claimed only if p < 0.05. A margin difference between the contexts is direct influence (panel conditioning or attrition), not contextuality.
- **E. Attrition as selection.**
  - Tables: (D1, A1) on the whole W1 cohort; (D1, D3) and (A1, D3) on the W1 respondents who answered W3.
  - Contextual fraction available-case, after IPW on the (D1, A1) cells, and with the provided `W3_Weight_Variable` as secondary weights.
  - Sampling null and decision rule as before.

## Order
1. Write the adapter and test it on synthetic data in the same layout, with planted answers.
2. Run once on `MHASAP.ALLWAVES.sav`. Freeze the result with a SHA-256 hash.
3. Report every result. Anything else is exploratory.

## Not used
The UK strand (C19PRC-UK, six waves) is not in this archive. If provided later, it gets its own amendment, and the Irish results serve as the replication.
