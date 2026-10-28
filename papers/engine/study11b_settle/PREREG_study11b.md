# Study 11b — when does a decision on the hiding segment settle? (pre-registration)

Vinay K Awasthi, Johns Hopkins University. Registered 2026-10-27, before any registered design was simulated. Frozen with SHA-256 in
`FROZEN_PREREG_study11b.txt` together with `run_study11b.py`, `design11b.py`, `../pce/segment_decision.py` and `../pce/lab.py`.

## What was done before freezing
- The engine checks DS1–DS9 (`../check_decision_segment.py`), all on random truths that are not the designs below.
- Design quantities that need no sampling (the oracle act, the pre-probe act, the settling radius, n*). They are listed below.
- Two smoke runs of the code on an invented design (w = 0.5, a1 = (.75, .25, .70), a2 = (.25, .70, .20); not registered). The first
  showed that with a second constraint close to the binding one, the median settling size exceeds the one-constraint n*. The designs
  were then fixed so that one constraint binds (u0 three radii below the competitor). The second smoke run (4000 replicates) agreed
  with the delta-method prediction to within 0.01 at every n.

## Claim under test (lab paper, Theorem D(4))
After a triple window with n draws per window, the regulator's post-probe set is I_n = [rhat - h_n, rhat + h_n] cut by the estimated
segment, with h_n = z tauhat / sqrt(n), z = 1.96, and tauhat the plug-in asymptotic sd of rhat. The regulator applies the set rule (the
admissible act with the largest worst value on I_n; abstain if none). The decision has *settled* at n when the set rule on I_n chooses
the oracle act, the act the set rule chooses at the true point r0.

For affine utilities the set rule picks the oracle act exactly when every constraint g_j(rhat) - gamma_j h_n >= 0 holds, where
g_0 = U_a(r) - u0 (admissibility) and g_b = U_a(r) - U_b(r) (competitor b), with gamma_0 = |beta_a| and gamma_b = |beta_a| - |beta_b|.
The statistics ghat_j are asymptotically normal: sqrt(n)(ghat_j - g_j) -> N(0, Sigma). So
  P(settled at n) -> P(g_j sqrt(n) + xi_j >= gamma_j z tau for all j),  xi ~ N(0, Sigma),
and when one constraint binds the median settling size is n* = (z tau gamma_j / g_j)^2 = (z tau / d)^2, with d = g_j / gamma_j the
settling radius. On the stratum g_j = 0 the settling probability has a plateau Phi(-gamma_j z tau / s_j) < 1 and never settles.

## Designs (fixed; `design11b.py`)
Truths on three binary harms: G (w = 0.30, a1 = (.70, .30, .80), a2 = (.20, .65, .10)); H (w = 0.40, a1 = (.80, .25, .70),
a2 = (.15, .70, .20)). Acts: A: u = b - [any harm]; B: u = b - [at least two harmed]; C: u = b - [variable 0 harmed] (flat on the
segment: beta = 0).

| design | truth | acts | binding constraint | d | n* | oracle | pre-probe act |
|---|---|---|---|---|---|---|---|
| D1 | G | A | admissibility | 0.35 | 3270 | A | abstain |
| D2 | G | A, C | A vs C | 0.35 | 3270 | A | C |
| D3 | G | B, C | B vs C (boundary on the other side of r0) | 0.25 | 6410 | B | C |
| D4 | G | A, C | A vs C, with the truth on the crossing (stratum, d = 0) | 0 | none | A (tie) | C |
| D5 | H | A, C | A vs C | 0.30 | 2001 | A | C |

Sample sizes: n = n* x (0.5, 1, 2, 4, 8) for D1, D2, D3, D5; n = 1000, 3000, 10000, 30000, 100000 for D4. 4000 replicates per cell,
seed 1111. Each replicate samples the three pair windows and the triple window independently (n draws each). Replicates whose pair
tables admit no two-type explanation count as not settled, and their share is reported.

## Predictions
- **S1 (calibration).** Over the 20 cells of D1, D2, D3, D5, the empirical settling rate is within 0.05 of the delta-method prediction
  in at least 18 cells, and within 0.08 in all.
- **S2 (median).** For D1, D2, D3, D5, the empirical median settling size (log-linear interpolation of the first crossing of 1/2) is
  within a factor 1.25 of the closed-form n*.
- **S3 (stratum).** For D4, the empirical rate is within 0.05 of the predicted plateau at every n, and the rates at n = 3000 and
  n = 100000 differ by at most 0.05 (no settling).
- **S4 (pre-probe decision).** At the largest n of D1, D2, D3, D5, the set rule on the estimated segment alone (no triple) chooses the
  population pre-probe act (abstain for D1, C otherwise) in at least 95% of replicates.

## Analysis and reporting
`run_study11b.py` computes predictions and simulations and writes `results_study11b.json` with an `evaluation` block (S1–S4). All four
outcomes are reported, failures included. Any deviation from this plan is logged in `DEVIATIONS_study11b.md` before the affected
result is reported, and the frozen script is kept as a text copy.
