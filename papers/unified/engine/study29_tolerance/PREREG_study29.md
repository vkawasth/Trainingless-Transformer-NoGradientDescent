# Study 29: credal sets beyond exact margins; certified top-k and a stratified probability sample (sec:ledger), pre-registered
# 2026-11-14 before any code for it was run

Question. When the credal set is no longer "exact margins" (so the closed basis formulas of prop:basiscount do not apply), does the
library of certified bases still compress the work, and can (a) the k tables whose decision moves most be found and certified without
evaluating all, and (b) population trends be estimated with honest uncertainty from a probability sample?

Families (each T = 10000 tables).
  tol3-e   3 x 3 tables, margins known within a box: f_i - e <= row sum i <= f_i + e, g_j - e <= column sum j <= g_j + e, total 1, x >= 0;
           (f, g) = the 3 x 3 population of check_stratledger.py (seed 20261126, denominator 120); e in {0, 1/200, 1/100, 1/50}
  tol4     4 x 4, same construction, e = 1/100, the 4 x 4 population of check_stratledger.py (seed 20261127, denominator 240)
  pair332  3 x 3 x 2 tables (A, B, C) with all three pairwise margins (AB, BC, AC) known exactly; margins of a joint of integer counts
           over 600 drawn from Dirichlet(1) (seed 20261129); credal set = all joints with these pairwise margins
Decision quantity. tol: P(cell (0,0) | rows {0,1} x columns {0,1}) as in sec:ledger. pair332: P(A = 0 | B in {0,1}, C = 0).
Evidence. Cell weights L = 1 + ((i + 2j) mod 3)/2 (tol) and L = 1 + ((a + 2b + c) mod 3)/2 (pair332). Band ends hi0, lo0, hi1, lo1.
Defined. A table is defined when min d.x over its credal set is > 0 (decided by a certified linear objective).
Drift. D = max(|hi1 - hi0|, |lo1 - lo0|). Bound U: the logistic bound of sec:ledger with dp = max_num log L - min_rest log L and
dm = max_rest log L - min_num log L (num = numerator cells, rest = denominator cells outside the numerator), applied at hi0 and lo0.
Method. Standard form with slack variables; a library of bases with float inverses certifies tables (Dinkelbach reduced costs); an
uncertified table triggers one Charnes-Cooper linear programme, whose vertex support is completed to a basis that certifies it (up to
30 completions); if none does, the table keeps the LP value and is counted as "fallback".
Certified top-k (k = 20): before-evidence ends for all tables through the library; U for all; after-evidence ends in decreasing U until
20 tables are evaluated and the next U is below the 20th largest evaluated drift.
Probability sample. Frame: all T tables. Strata: quartiles of a covariate computed without any LP (tol: f0 + f1 + g0 + g1; pair332:
P(B in {0,1}, C = 0) from the BC margin). Proportional allocation, n = 400, simple random sampling without replacement within strata.
Targets: (i) mean drift over defined tables (combined ratio estimator, linearised variance); (ii) share of all tables with D > 0.05
(undefined tables count as 0). 95% normal intervals with finite-population correction. R = 500 replicate samples (seed 29); the truth
is the exhaustive value. Comparison: simple random sampling of the same n.

Checks (thresholds fixed now).
  TK1 IDENTITY  tol3 at e = 0 reproduces the exact-margin bands of engine/stratledger.json for every defined table and all four ends (1e-9)
  TK2 IDENTITY  bands nest in e: for e1 < e2 every band end at e2 contains the one at e1 (1e-9), every tol3 table
  TK3 IDENTITY  every value agrees with a direct linear programme on 200 random defined tables per family and end (1e-7), and an exact
                rational re-check of 100 certified tables per family and end passes
  TK4 IDENTITY  U >= D for every defined table; the certified top-20 drift values equal the exhaustive top-20 values (as multisets), every family
  TK5 HYPOTHESIS compression persists: LP solves over all four ends and the defined test <= 0.2 T, every family with e > 0 and pair332
  TK6 HYPOTHESIS certified top-20 evaluates the after-evidence ends of at most 50% of the defined tables, every family
  TK7 HYPOTHESIS stratified 95% intervals cover the truth in >= 93% of the 500 replicates, both targets, every family
  TK8 HYPOTHESIS stratification helps: empirical variance of the stratified estimator <= that of simple random sampling for the mean
                drift, every family
  TK9 RECORD     band width, drift, defined share, solves and fallbacks against e; top-k evaluations; interval half-widths; cost of the
                sample (4 n linear programmes) against the exhaustive cost
Families for TK5-TK9: tol3 at 1/200, 1/100, 1/50; tol4; pair332.
Reporting as in Study 28: failures stay failures; first-run failures and fixes recorded in the script docstring; aggregates only
(synthetic data). Output: check_tolerance_output.txt, study29_tolerance/tolerance.json.
