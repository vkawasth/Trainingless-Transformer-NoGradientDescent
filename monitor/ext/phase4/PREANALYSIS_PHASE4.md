# Pre-analysis plan, phase 4: predictive geometry, restricted selection, structural events, and a generalized reliability level

**Fixed on 2026-10-10.** Written before any quantity below was computed.

What was already known:
- the results frozen in `open_clinical/results/FROZEN_TOWER_ICU.txt`;
- the exploratory evidence-floor result of `tower_icu_explore.py` (shift arm, A surgical → B medical/coronary).

Seed: 20261010. Small cells (< 10) are suppressed in any table shown. Patient-level data stay outside the monitor folder.

## WP1. Does geometry forecast deterioration or transitions beyond the margins?

### WP1a. Clinical: per-patient systems in PhysioNet 2012 (primary)
**System.** For each stay and each window W1 = hours [0, 24) and W2 = [24, 48), the 3×3 joint table over hours of:
- mean arterial pressure band: < 65, 65–75, > 75 (invasive MAP, else non-invasive);
- heart-rate band: < 60, 60–100, > 100.

Each hourly value is the median of that hour's measurements. A window is used if it has at least 12 hours with both values. The margins f (pressure) and g (heart rate) are the shares of hours.

**Margin features (baseline).**
- f and g of W1 and W2 (two free coordinates each: shares of bands 0 and 1);
- their differences W2 − W1.

**Geometric features.**
- Toric type: the number of vertices of the 3×3 coupling polytope of (f, g) in each window, by exact enumeration of basic solutions. Values are 9, 12, 15 or 18 for full-support margins, and fewer when degenerate.
- Vertex change W2 − W1 (blow-up if > 0, blow-down if < 0).
- Chamber transition: the number of the 18 wall functions f(I) − g(J) whose sign changes between W1 and W2 (zero counted as its own sign).
- Tear margin of W2: the smallest |f(I) − g(J)|.
- Break count: the number of zero margin entries in W1 and in W2.
- Entropy-brane steepness of W2: the log of the largest eigenvalue of the Fisher form on the fibre at the product coupling, i.e. the largest value of Σ x_ij² / (f_i g_j) over unit circulations x. It is defined for full-support margins. Broken windows get the largest value observed among full-support windows, and the break count carries the rest.

**Outcomes.**
- Primary: in-hospital death.
- Secondary (a subsequent break of the clinical system): onset of shock hours in W2 among stays with none in W1, predicted from W1 features only.

**Models.** Logistic regressions:
- B1: baseline, linear in the margin features;
- B2: B1 plus all squares and pairwise products of the margin features (a flexible function of the margins);
- S1 = B1 + geometric features;
- S2 = B2 + geometric features.

Ridge penalty C = 1 on standardized features, fixed in advance, for all four. Fit on set A, evaluate on set B.

**Decision (primary).** Geometry **forecasts beyond the margins** if Δ = logloss(B2) − logloss(S2) on set B has a paired-bootstrap 95% interval (2000 draws) above 0. Secondary: S1 against B1, and the same for the shock-onset outcome. Toric type, chamber and steepness are deterministic functions of the margins, so B2 is the fair baseline: it asks whether these particular functions carry information that a generic nonlinear function of the margins does not.

### WP1b. Societal: GSS chamber transitions
**Systems.** Every pair of the eleven spending items, in both wordings, in every survey year: 3×3 tables, weighted by `wtssps`.

**Event.** Between consecutive survey years, a change of the point-estimate chamber (any wall sign flips). Decided tears are too rare (one in the paper) to forecast.

**Predictors at year t.**
- Baseline: the TV change of the margins from the previous survey year, and the smallest margin share.
- Geometric:
  - the tear margin;
  - the vertex count;
  - the log steepness;
  - whether a chamber change happened at the previous step.

**Split.** Train on steps ending 1985–2004, test on steps ending 2006–2024.

**Decision.** As in WP1a, with B2 the baseline plus squares and products, and a bootstrap clustered by item pair.

## WP2. Restricted selection models (the fix needed before MIMIC-IV)
The triangle (X, Y) all, (X, Z), (Y, Z) on the measured. Weight models:
- **cell** (reference): 1/P(m | x, y). This makes the tables compatible by construction; reported, not tested.
- **X-only**: 1/P(m | x).
- **Y-only**: 1/P(m | y).
- **external**: 1/P(m | c), with c covariates outside the triangle, logistic main effects.
- **X + external**: 1/P(m | x, c).

**Datasets and covariates c (fixed now).**

| Dataset | Z | X | Y | c |
|---|---|---|---|---|
| PhysioNet 2012, set A (primary) and B (replication) | lactate band | H1 | creatinine band | ICU type and ventilation in the first 24 h |
| Diabetes 130 | HbA1c result | age band | readmission | admission source and admission type |
| C19PRC UK attrition | D3 | D1 | A1 | age category, gender, education (binary), urbanicity at W1 |

**Null, recomputed for each model (parametric bootstrap, 200 draws).** The null hypothesis is that the three tables are compatible and selection follows the model. Under it:
1. Resample rows (x, y, c) with replacement.
2. Draw z from P(z | x, y) of the IPF joint of (T_xy, and the model-weighted T_xz and T_yz).
3. Draw m from the fitted model.
4. Refit the model, reweight, and compute the contextual fraction.

**Decision per model.**
- *Consistent with the data* if the weighted contextual fraction is at or below the null's 95th percentile.
- *Rejected* otherwise.
- The available-case fraction and its null are reported as before.

**MIMIC-IV kit.** Analysis E of `mimic_monitor.py` gets the same models. The external covariates are admission type and anchor-age band, with the null recomputed for each model. The synthetic check must still pass.

## WP3. Structural events against external states
A general pipeline `structural_events.py`. Given a sequence of systems with labels, it emits, for each step:
- vertex-count change (blow-up or blow-down; toric type);
- chamber transition (wall flips);
- break (a zero margin appearing or disappearing);
- steepness.

It joins these to an external state and tests association by permutation (2000 draws).

**Applications.**
1. *PhysioNet per-stay systems (W1 → W2) against death.* For each event type: the death rate with and without the event, a permutation p-value, and the same within strata of the W1 margin baseline risk (deciles of the B1 prediction) to separate structure from margins.
2. *GSS item-pair steps against the partisan state of the year.* Δ in the share identifying as Republican, from `partyid`, between the two survey years. This is an association only, at the year level (low power, stated now).
3. *LLM drift tasks (8 tasks × 2 models).* Regime (bend, tear, tear?) against the change in the share of invalid or incorrect answers (the hallucination proxy). Descriptive only; no test with n = 16.

## WP4. A generalized reliability level for the tower, with pre-registered replication
**Variants.** Each transports a fitted model's prediction ρ:
- **T-fit:** floor φ = the fit sample's base rate (as before).
- **T-ev:** φ = the evidence sample's base rate. This is the repair found after the fact; it is pre-registered here.
- **T-fam (family of floors):** hypotheses (k, α, φ), with φ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40} and α as before. ρ′ = αρ + (1−α)φ, which is the Kleisli composite of ρ with the binary channel P(1|1) = α + (1−α)φ, P(1|0) = (1−α)φ. So level 3 carries a posterior over channels, not over a scalar.

Prior uniform over (k, α, φ). Same models, evidence and test procedure as `tower_icu.py`.

**Arms.** These are new splits of PhysioNet 2012, not new data; truly new data (eICU, MIMIC-IV) are credentialed.
- **R1, same population, reversed:** fit B, evidence and test on halves of A.
- **R2, reverse shift:** fit B surgical (ICU types 2, 4); evidence and test on A medical/coronary (1, 3).
- **R3, opposite shift:** fit A medical/coronary; evidence and test on B surgical.
- **R4, cross shift:** fit A coronary and cardiac surgery (1, 2); evidence and test on B medical and surgical (3, 4).
- **R0:** the two original arms, reported for continuity.

**Decisions (each Δ = logloss(flat) − logloss(variant), with a paired bootstrap interval).**
- **H1 (replication of the repair):** T-ev's interval is above 0 in at least 2 of the 3 shift arms R2–R4.
- **H2 (family of floors):** T-fam's interval is above 0 in at least 2 of R2–R4.
- **H3 (no harm):** in R1, no variant's interval lies entirely below 0.
- **Secondary:**
  - T-fam against T-ev;
  - calibration;
  - the posterior over floors;
  - Fano tubes and the posterior over toric surfaces in each arm;
  - whether each observed chamber is decided on its bands.

## Order
1. Write adapters and test them on planted data:
   - WP1: planted geometric signal versus margin-only signal;
   - WP2: planted X-only and external selection;
   - WP4: planted prevalence shift.
2. Run once and freeze with SHA-256 hashes.
3. Write up. Anything else is exploratory and labelled so.
