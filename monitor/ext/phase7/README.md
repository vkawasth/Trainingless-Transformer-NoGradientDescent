# Phase 7: deciding under uncertainty, with support floors, typed obstructions and auditable records

The plan is `PREANALYSIS_SUPPORT.md`, hashed in `FROZEN_PLAN.txt` before anything was run. `FROZEN_CODE.txt` holds the code hashes taken before the experiments, and `FROZEN_RESULTS.txt` holds the results together with the documented deviations. The source proposal is *Ambient Support Geometry* (11 October 2026).

| file | what it does |
|---|---|
| `decide.py` | Builds the credal set from sources with known reliability (or an interval for it), their bands, and known context maps. It decides with a relative threshold κ (q_x ≥ κ q_y in the worst case) and optional absolute floors τ. It returns: the strength κ*; the dual trace split by source; the weakest link (critical reliability for each source); the thinnest cell; value of information; repair of conflicting sources; and the typed obstruction record. `python3 decide.py` runs the planted checks. |
| `support.py` | Floor polytopes through reduced margins (Proposition R), with feasibility decided on bands. Closed-form headroom and its bottleneck. The fan and walls of a floored system. Gluing on any cover by one LP with a verified Farkas certificate. The 3-cycle inequalities. Closed forms for the X–Y–Z chain. Transport of floors and holonomy. `audit()` produces the typed record O = (supp, glue, comp, evid); `record()` and `replay()` use SHA-256. `python3 support.py` runs the planted checks. |
| `experiments.py` | Pre-registered experiments E1–E6, run once; output in `results/support_experiments.json`. |
| `rechecks.py` | The documented deviations for E1 and E2b; output in `results/support_rechecks.json`. |
| `demo.py` | A synthetic worked example; output in `results/demo.json`. |

## Outcomes

| | result |
|---|---|
| E1, Proposition R | **Failed as run** (324/500). The comparator was order-sensitive. Order-free on the same instances: 500/500 within 1e-7. 2×3 toric types 84/84; infeasibility 200/200. |
| E2, weakest link | Closed form exact in 418/418. **Dual criterion failed** (0.823): it was ill-posed, because duals on margin equalities are only defined up to a gauge. Gauge-invariant version: 418/418. Without headroom the closed form does not apply (it differs in 43% of cases). |
| E3, chain statuses | Holds: 4000/4000, 200/200, 2600/2600. |
| E4, cyclic cover | Holds. Agrees with the 3-cycle inequalities in 600/600; 81/81 certificates verified. 124 of 519 gluable families are support-contextual: they glue, but not with the floor. |
| E5, transport | Holds. Floors are preserved 500/500. No shrink under coarsening; under stochastic kernels the slack stays above the row-sum bound. The slack shrinks in 2% of stochastic cases; the pushed floor is tight in 27% of cells. |
| E6, records | Holds. 400/400 replays are bit-identical; 80/80 planted obstructions are classified to the right type. |

## Exploratory
- `lax_composition.py`: lax composition of floor summarisation and policy imposition under transport. Output: `results/lax_composition.json` (0/300 violations in each direction; summarising lowered the floor in 611 of 895 target cells; imposing first raised it in 1158 of 1158).

## Not done yet
- Floors on events and on cyclic covers use LPs; there is no closed form for them.
- The weakest link is computed one source at a time, not jointly.
- A falsehood shared by every source shows up only as a wide credal set; it cannot be separated from the truth.
- Nothing has been run on real retrieval data yet.
