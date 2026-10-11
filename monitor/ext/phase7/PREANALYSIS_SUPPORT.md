# Pre-analysis plan, phase 7: support floors, typed obstructions and auditable decisions

**Fixed on 2026-10-11, before the experiments below were run.** Seed 20261012.

**Already known (exploratory, scratch).** Three things were looked at before this plan was written:
- A uniform floor on all cells of 3×3 systems gave the vertex count of the reduced-margin transportation polytope in 80 of 80 cases.
- The closed-form headroom matched the LP to 1e-16 in the 255 of 400 random cases where it was ≥ 0. Where it was negative (145 cases) the two differed by up to 0.06.
- One X–Y–Z chain example showed the three gluing statuses separating.

These motivated the predictions below. They are not part of the evidence.

**Source.** The proposal *Ambient Support Geometry: Presheaves, Positive Support Floors, and Local-to-Global Obstructions* (11 October 2026), §2, §7 and §8.

## Definitions
- **Floor polytope.** P_τ(f, g) = {π ∈ P(f, g) : π_xy ≥ τ_xy on designated cells D}, with τ ≥ 0.
- **Reduced margins.** f′ = f − (row sums of τ·1_D) and g′ = g − (column sums of τ·1_D).
- **Proposition R (to be checked).** π ↦ π − τ·1_D is a bijection from P_τ(f, g) onto P(f′, g′). Hence:
  - P_τ is non-empty iff f′ ≥ 0 and g′ ≥ 0;
  - its normal fan is that of P(f′, g′);
  - its walls are f′(I) = g′(J).
- **Headroom.** δ* = max δ subject to π ∈ P(f, g) and π_xy ≥ τ_xy + w_xy·δ on D.
  - Closed form: δ_cf = min over rows x with W_x > 0 of (f_x − T_x)/W_x, and over columns likewise. Here T and W are the row or column sums of τ·1_D and w·1_D.
  - Valid whenever δ_cf ≥ 0. The bottleneck is the minimising row or column.
- **Typed obstruction record.** O = (supp, glue, comp, evid). Each component is None or a typed witness:
  - supp: floor shortfall with its bottleneck margin or binding constraints;
  - glue: a Farkas vector certifying that no global section exists;
  - comp: the holonomy of transport around a loop;
  - evid: conflicting sources and the repair.

## Experiments and predictions (each reported as held or not)

**E1. Reduced-margin structure.** 500 random instances: sizes 2×2 to 3×4, a random designated set D, random floors with f′, g′ > 0. Vertices of P_τ are enumerated independently, as basic solutions with π = τ or 0 on non-basic cells.
- **Prediction:** the vertex set of P_τ equals τ·1_D + the vertices of P(f′, g′) to 1e-7, in 100% of instances.
- **Also predicted for 2×3:** the toric type of P_τ (by vertex count) equals that of the normalised reduced margins, 100%.
- **Infeasibility:** 200 random instances with some reduced margin < 0 are all infeasible by LP, 100%.

**E2. Weakest link.** 1,000 random instances: sizes 2×2 to 4×4, random D, τ and w.
- **(a)** When δ_cf ≥ 0, |δ* − δ_cf| < 1e-9 in 100%.
- **(b)** When δ_cf ≥ 0 and the minimising row or column is unique, every nonzero LP dual on a margin equality belongs to the bottleneck margin or to margins that tie it within 1e-9. Predicted in ≥ 99%.
- **(c)** When δ_cf < 0: reported descriptively, as the share where δ* ≠ δ_cf.

**E3. The three gluing statuses on the X–Y–Z chain.** 200 random instances with fixed (X, Y) and (Y, Z) margins (binary), a common floor τ on all 8 cells, and 20 values of τ in [0, τ_max].
- **Closed forms:**
  - a floored filler exists iff τ ≤ τ_any = min over cells of π(x,y)/2 and π(y,z)/2;
  - the conditional-independence filler meets the floor iff τ ≤ τ_CI = min π(x,y)π(y,z)/π(y).
- **(a)** LP feasibility agrees with τ_any in 100% of (instance, τ) pairs.
- **(b)** τ_CI ≤ τ_any in 100%.
- **(c)** The maximum entropy over floor-admissible fillers equals H(XY) + H(YZ) − H(Y) (within 1e-6 bits) iff τ ≤ τ_CI. Predicted in ≥ 99% of pairs; numerical optimiser tolerance is the only expected source of failure.

**E4. Cyclic cover.** The triangle (X,Y), (Y,Z), (Z,X), binary.
- **Families:** 300 locally consistent families: random single margins, each context a random coupling of its two margins. Plus 300 families marginalised from random global joints.
- **(a)** LP gluability at τ = 0 agrees with the 3-cycle inequalities in 100%. These are Araújo et al. 2013, for ±1 variables with consistent margins: Σ γ_i E[X_i X_{i+1}] ≤ 1 for every sign vector γ with an odd number of −1s.
- **(b)** Every non-gluable family returns a Farkas vector y with Aᵀy ≤ 1e-9 and b′ᵀy > 0, verified independently, 100%.
- **(c) Support-contextuality.** Take the gluable families and floors τ at which every context table satisfies its local floor (each local cell ≥ 2τ, since each local cell is the sum of 2 global cells). The prediction is that some of them have no global section with floor τ: at least one instance exists. The share is reported descriptively.

**E5. Transport of floors.** 500 random (π ∈ P_τ, kernel K), split between deterministic coarsenings and random column-stochastic kernels.
- **(a)** Kπ ≥ Kτ elementwise, 100%.
- **(b)** Under deterministic coarsenings, min(Kπ − Kτ) ≥ min(π − τ), 100%.
- **(c)** Under stochastic kernels, min(Kπ − Kτ) ≥ (min row sum of K)·min(π − τ), 100%. The share where the slack shrinks is reported descriptively.
- **Descriptive:** the share where Kτ equals the LP minimum of (Kπ)_k over P_τ (tightness).

**E6. Auditable decisions.** `decide.py` gains floors and the typed record.
- **(a)** 100 random decision problems, each with a sequence of three ambience changes (floor version, evidence version, κ). Every recorded decision replays bit-identically from its record (SHA-256 of the canonical output), 100%.
- **(b)** Planted obstructions are classified to the right type, one of each type, 20 each, 100%:
  - a floor that exceeds what the sources allow → supp;
  - a contextual triangle → glue;
  - a loop of transports with nonzero holonomy → comp;
  - a lying source → evid.

## Order
1. Freeze this plan.
2. Write `support.py` and extend `decide.py`, with planted checks.
3. Run `experiments.py` once.
4. Freeze the results.
