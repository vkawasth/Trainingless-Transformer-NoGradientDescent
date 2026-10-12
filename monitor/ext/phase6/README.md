# Phase 6: tracking the approach to collapse (family of cones) and recursive training

The plan is `PREANALYSIS_COLLAPSE.md`. It was fixed before any computation, and its hash is in `FROZEN_PLAN.txt`. The results are frozen in `FROZEN_TRAJ.txt` and `FROZEN_RT.txt`.

| file | what it does | output |
|---|---|---|
| `fan_tracker.py` | The tracker: chamber and signed wall distances with band intervals, tear and break margins, velocity, tear and break times (Bands Def. 3.1, Thm. 3.2, Def. 5.7), Csiszár interaction/margin drift split, normal-fan type (dimension, vertices, facets), Dirichlet chamber posterior. `python3 fan_tracker.py` runs the planted checks. | `results/tracker_checks.txt` |
| `trajectories.py` | Runs the five trajectories: GSS (110 item-pair systems), C19PRC (UK and IE), tower evidence paths R0–R4, ICU decedents and survivors in 6-hour windows, and LLM March→June. Then runs the pre-registered forecasting test. | `results/trajectories.json`, `results/forecast_test.json` |
| `recursive_training.py` | Recursive training with known truth: 20 generations, 50 replicates, fresh data at 0/5/10/25%, and a paraphrase kernel. Tests predictions P1–P5. | `results/recursive_training.json` |
| `rt_explore.py` | Exploratory, not pre-registered: zero-recovery rates and persistent zeros. | `results/recursive_training_exploratory.json` |

The figures are made by `../../make_fig_collapse.py`, which produces `figs/fig_collapse_trajectories.pdf` and `figs/fig_recursive.pdf`.

## Outcomes
**Forecasting test.** It is underpowered: there are 11 decided test events, and the plan required at least 15. Neither interaction drift nor tear time adds anything for decided changes.
- On the secondary target, point-estimate changes, tear time adds 0.017, with a CI of (−0.0006, 0.033).
- Tear time together with interaction drift adds 0.017, with a CI of (0.0002, 0.034).
- Interaction drift alone adds nothing.

**Recursive training.**

| prediction | outcome | detail |
|---|---|---|
| P1, contraction | holds, but trivially | η(K) = 1 |
| P2, breaks forecast by break time | fails | 88.5% absorbed; break time does not beat the level |
| P3, fresh-data floor | fails as stated | the finite-sample model is not floored. Exploratory: zeros recover only with fresh data |
| P4, departure from the invariance variety | holds in every replicate | I(Y;C\|X) rises from 0.016 to 0.18 bits; it plateaus at 0.09, 0.058 and 0.03 with fresh data |
| P5, certificate warns first | fails | 58% of replicates, against 80% needed; median lead 2 generations |

## Deviations (all documented in the FROZEN files)
- Two planted-check specifications were corrected before any real run:
  - the 2×2 path crosses two walls at once;
  - break time is non-increasing, not strictly decreasing.
- A rank-tolerance bug made the fan dimension 5 for some 3×3 systems. It was fixed (tolerance 1e-6), a dimension check was added, and all trajectories were rerun with the same seed. The forecasting test does not use the fan.

## Data handling
- ICU windows are built from the PhysioNet 2012 open-access zips, and only aggregate counts are kept (`T[2, 8, 3, 3]`). The patient-level cache stays outside this folder.
- Raw C19PRC files (`ext/c19/raw`) are not packaged.
