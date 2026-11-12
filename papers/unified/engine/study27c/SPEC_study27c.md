# Study 27c — the filtered witness braid (specification, frozen before any 27c test run)

Author: Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu
Frozen (SHA-256 in FROZEN_SPEC_study27c.txt) before engine/run_study27c.py is run; run once. Deviations go to DEVIATIONS_study27c.md;
no threshold, grid, loss, size or seed changes after the freeze. Inputs: Study 27b (world, generator, scenarios, reporters, sizes,
unchanged code from run_study27b.py), and the exploratory attribution (ATTRIBUTION_27c.md, seed 27101), which motivates the filter.
The test seed has never been used. Context overlaps and regions: OVERLAP_TABLE.md.

## 1. Data
Same world, histories, scenarios, reporters and sample sizes as 27b (sources n = 6000; anchors {M,R,Y}, {M,H,Y} at 3000; inspection
{M,Y} at 120,000). Configurations: A (two anchors), B (A plus the 120k inspection). Rounds 0..12.
- Calibration set, seed 27201: 1000 runs each of honest, shock, cascade_timely, relay_fixed.
- Test set, seed 27202: 1000 runs each of honest, shock, relay_fixed, cascade_timely, cascade_lagging, drift.

## 2. The filtered witness statistic
For a source strand k and an anchor strand A, displacements from round 0: dx_k(t) = x_k(t) - x_k(0), where x = g of the report's {M,Y}
margin if the context has M and Y, else x = 0 (as in 27b). Relative displacement r_kA(t) = dx_k(t) - dx_A(t).
Components (each on or off):
- Probability (always on): pair offset kappa_kA = z sqrt(2) sqrt(SE_k^2 + SE_A^2), SE = design SE of g at P for that report's size
  (SE = 0 for sources without g); the factor sqrt(2) accounts for the round-0 difference. The pair has moved across when
  r_kA(t) > kappa_kA. One multiplier z per configuration, calibrated (section 3).
- Braid projection: on: the pair counts 1 if r_kA(12) > kappa_kA (net side change; the permutation image, over/under discarded);
  off: the 27b count |sum over crossings of direction x front|, crossings being sign changes of r_kA(t) - kappa_kA over t = 0..12 with
  linear interpolation, front from the y-coordinates of 27b.
- Contextual: on: if at round 12 some source without g (S0, S1, S2, S4, S5) has crossed anchor A (it cannot have moved in x), the change
  is attributed to A ('anchor departure') and all pairs with A are excluded from Phi; off: no exclusion.
- Geometric guard: on: a strand whose smallest expected cell count (n times smallest cell probability of its sampled table) falls below
  30 in any round has its pairs excluded and counted as a guard activation; off: no guard.
Phi = sum over counted pairs; relay-like iff Phi >= 1.

## 3. Calibration (calibration set only)
- z: per configuration, the smallest z in {2.00, 2.05, ..., 4.00} for which the honest false-relay rate with all components on is <= 0.05.
- Value of information (greedy, order: geometric, contextual, braid): a component is switched off if switching it off does not raise
  the calibration decision loss L = 0.5 x P(miss | relay_fixed) + 0.5 x 0.2 x mean over {honest, shock, cascade_timely} of
  P(relay-like); z stays as calibrated. The resulting component set is frozen for the test set.

## 4. Predictions (test set, both configurations unless stated)
- P1 honest false-relay rate <= 0.065 (0.05 plus the binomial 95% margin at n = 1000).
- P2 shock false-relay rate <= 0.10.
- P3 balanced accuracy, relay_fixed vs cascade_timely, >= 0.90 in configuration B.
- P4 the inspection strand does not hurt: balanced accuracy B >= A - 0.01.
- P5 the confound persists: cascade_lagging relay-like rate >= 0.90.
- P6 the geometric guard never activates (attribution: every strand interior).
Each prediction is reported HELD or FAILED; failures are findings.

## 5. Descriptive
- Ablation on the test set: balanced accuracy and false-relay rates with each component switched off in turn (z fixed).
- Explanation map: for each run, the set of sources with a counted side change, the last round at which each changed side, whether it
  was activated (trace) at or before that round, anchor departures, and agreement with the planted set ({S3} in relay and drift).
- Drift: relay-like rate and the share of changes that follow an activation (relay vs drift separation through the trace).
