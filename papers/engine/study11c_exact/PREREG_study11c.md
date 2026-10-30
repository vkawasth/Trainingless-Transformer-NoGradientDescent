# Study 11c — settling with the exact conditional probe (pre-registration)

Vinay K Awasthi, Johns Hopkins University. Registered 2026-10-29, before any registered design was simulated. Frozen with SHA-256 in
`FROZEN_PREREG_study11c.txt` together with `run_study11c.py`, `../pce/ehrhart.py`, `../pce/segment_decision.py`, `../pce/lab.py` and
`../study11b_settle/design11b.py`.

## Why
Gap 3 asks whether the settling size n* of Theorem D(4) is a wall-crossing in the ring of contingency tables. It is not (check EH5):
growing n moves along a ray inside one chamber. But the ring supplies the exact sample space of the probe. Given its own pair tables,
the triple window is one lattice point on a segment of tables joined by the move chi, and the conditional law on that segment is a
one-parameter exponential family in the three-way log odds ratio psi (the exact test of the no-three-way-interaction model). This study
asks whether a regulator who uses the exact conditional interval settles as the conditional normal theory predicts, and earlier than
one who uses the plug-in centred moment (Study 11b's probe).

## What was done before freezing
- Engine checks EH1–EH7 (`../check_ehrhart.py`), none on a sampled registered design (EH6 simulates the conditional MLE for truth G
  at n = 4000 only to check its standard deviation).
- Design quantities that need no sampling (below).
- One smoke run (300 replicates) on an invented design (w = 0.5, a1 = (.75, .25, .70), a2 = (.25, .70, .20), one act, d = 0.35; not
  registered). There tau_c / tau_m = 0.985; the exact and plug-in medians were 2095 and 2022 (n*_c = 2152), and the exact curve was
  within 0.04 of its normal-theory curve. So the ordering of the two probes is within noise when tau_c and tau_m are close.

## Set-up
Pair tables known (population). One triple window of n draws per replicate. The same sample feeds both probes:
- *plug-in*: rhat = mhat_T / v_T with the window's own means; interval rhat ± z tauhat_m / sqrt(n), z = 1.96.
- *exact*: the two-sided 95% exact conditional interval for psi given the window's pair tables, mapped to r through the population
  margin fibre (P + eps chi has log odds ratio psi; its triple moment is m_T(P) + eps; r = m_T / v_T).
Each interval is cut by the population segment R; the decision has settled when the set rule on it picks the oracle act.
Normal theory: P(settled at n) from Theorem D(4b) with tau_c = 1 / (|v_T| sqrt(sum_x 1/p_x)) (exact probe) or tau_m (plug-in);
n*_c = (z tau_c / d)^2.

## Designs (Study 11b's, pair tables known)
| design | truth | d | tau_c | tau_m | n*_c | n*_m | n grid |
|---|---|---|---|---|---|---|---|
| D2 | G | 0.35 | 8.993 | 9.529 | 2536 | 2848 | n*_c × (0.5, 1, 2, 4, 8) |
| D3 | G | 0.25 | 8.993 | 9.529 | 4971 | 5581 | same |
| D5 | H | 0.30 | 6.482 | 6.724 | 1793 | 1930 | same |
| D4 | G | 0 (stratum) | 8.993 | 9.529 | none | none | 1000, 3000, 10000, 30000, 100000 |

2000 replicates per cell, seed 1112.

## Predictions
- **E1 (calibration).** For D2, D3, D5 the exact probe's empirical median settling size is within a factor 1.25 of n*_c.
- **E2 (efficiency).** For D2, D3, D5 the exact probe's median is below the plug-in probe's median in every design.
- **E3 (discreteness is small).** For D2, D3, D5 the exact probe's settling rate is within 0.06 of the normal-theory rate with tau_c at
  every n of the grid.
- **E4 (stratum).** For D4 the exact probe's rates at n = 3000 and n = 100000 differ by at most 0.05 and both are below 0.5.

## Analysis and reporting
`run_study11c.py` writes `results_study11c.json` with an `evaluation` block (E1–E4). All outcomes are reported, failures included.
Any deviation is logged in `DEVIATIONS_study11c.md` before the affected result is reported.
