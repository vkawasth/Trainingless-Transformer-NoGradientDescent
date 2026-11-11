# Study 26 — one end-to-end example: 300 development tables, 100 holdout tables

Author: Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu
Written and frozen (SHA-256 in FROZEN_PREREG_study26.txt, together with world26.py) before any table was generated or examined.

## Purpose

Run the note's machinery once, on one collection of contingency tables, in the order a decision maker would use it:
tables → fibre → hypothesis and its paths → projections onto added conditions → decision → value of information → selection →
composition → contextuality (sheaf, fibre, cocycle) → history → protection. The point is to show the theorems working together on the
same data, with each step's output feeding the next. It is not a study of sleep. All numbers are invented (world26.py); the variable
names are illustrative only and carry no causal, clinical or physiological reading.

## Data (world26.py, seed 2611)

Core: M, C, R, H, Y binary (32 states), truth P* log-linear with the parameters in world26.py.
Development (300 tables): {M,C} 35 (10 from another population with P(M=1) = 0.8), {M,R} 35, {C,R} 30, {M,H} 30, {C,H} 30, {M,C,Y} 45
(15 from a coordinated coalition, n = 400 each, coherent with the honest {M,C} margin and P(Y=1, C=c), raising P(Y=1 | M=1, C=c) by 0.10),
{R,H,Y} 35, {M,C,R,H} 30, light L3 30. Sample sizes uniform on [80, 400] unless stated.
Holdout (100 tables, n = 300 or 400): {M,C,R,Y} 30, {M,Y} 10 (inspection), {M,R,Y} 10 and {M,H,Y} 10 (calibration), and a cycle
subdesign of four binary onset instruments A, B, D, E in pairs (A,B), (B,D), (D,E), (E,A): 20 tables 'consistent', 20 'twisted'
(in 'twisted' the (E,A) tables come from studies whose app coding was reversed).

The analysis never reads the field 'kind' except in XE2 (scoring the flagging rule after it has been applied).

## Fixed analysis rules

- Pooling: counts are summed over the sources of a context.
- Flagging (before pooling): for each source, a chi-square test of its counts against the pooled counts of the other sources of the
  same context (expected = its n times their pooled frequencies, cells with expected < 1e-9 dropped); flag if p < 0.001. Flagged
  sources are excluded from the main analysis; the analysis is also reported with them included.
- Robust fibre J_z: laws P on the 32 states with |π_C(P) − p̂_C| ≤ z · sqrt(p̂(1 − p̂)/N_C) + 1e-6 for every cell of every pooled
  context C, z = 3.
- Oracle fibre J*: the margins of P* on the development contexts (the exact-data object of the theorems). Exact statements (dimension,
  Fisher kernel, VOI criterion, history theorem, decision theorem) are tested on J*; estimated intervals are reported on J_z.
- Hypothesis. Paths: π1 = {M=1, R=1, Y=1} (circadian), π2 = {M=1, C=0, Y=1} (caffeine-free), π3 = {M=1, H=0, Y=1} (naive use).
  Broad hypothesis E = π1 ∪ π2 ∪ π3. Report the ranges of P(πi | M=1) and of P(E | M=1).
- Projection ladder: P(E | M=1, K) and the effect Δ(K) = P(Y=1 | M=1, K) − P(Y=1 | M=0, K) for K = ∅, {C=0}, {C=0, R=1},
  {C=0, R=1, H=0}. On J* the denominators are fixed by the data, so ranges are linear programs; on J_z each conditional is a
  linear-fractional program (Charnes–Cooper) and Δ is reported by its outer bound [min r1 − max r0, max r1 − min r0].
- Decision: intervene in the target stratum K = {C=0, R=1, H=0} iff Δ(K) ≥ τ = 0.10.
- Value of information: actions recommend / abstain with u_rec(ω) = 1{M=1}(1{Y=1} − 0.55), u_abs = 0; signal Z = R. VOI envelope on
  J* by one LP per policy cell and sense; criterion of Thm 92 via full-dimensional cells (slack LP > 1e-9) and the residual of each
  cell's value functional from the data span.
- Fisher (Prop 95): at the maximum-entropy law of J*, dim ker I_obs restricted to the simplex's tangent space.
- Selection (Thms 23, 27): margins f = P̂(M) (pooled {M,·} tables) and g = P̂(L3). Leximin coupling by the sequential LP; toric point at
  L = 1e3 and 1e4. Wall scan: move k = −40..40 counts from 'dim' to 'bright' in the pooled light counts, recompute; count changes of
  the binding structure. Advisory decision: issue iff the selected P(M=1, bright) ≥ 0.10; report it for leximin and maximum entropy.
- Composition (Thms 45, 65): with h = P̂(Y) (pooled {R,H,Y}), compare S(f, h) with S(f, g) ∘ S(g, h) for S = maximum entropy and
  S = leximin (composition = Markov glue at L3, then marginalise).
- System independence (Prop 71, Thm 65): on the four single margins f, g, (P̂(A), P̂(B)) of a 2 x 3 x 2 x 2 table, compare the section with
  the product of the sections of the two factors, for maximum entropy and leximin.
- Contextuality (Thm 77, Prop 78, Prop 137): pooled pair tables of each cycle condition; compatibility by LP on the 16-state space with
  the same z = 3 rule; the four-cycle noncontextuality inequalities Σ γ_i <X_i X_{i+1}> ≤ 2 for Π γ_i = −1 (±1 coding), with the
  largest left side and its standard error; the acyclic version drops (E, A). Sheaf layer (AMB): supports of the pooled tables, global
  deterministic sections, logical contextuality; noncontextual fraction by LP (max total mass of a sub-normalised global law whose
  pair margins are below the pooled frequencies).
- History (Thms 146, 148): updater from the uniform law; final data = J*'s margins. History A: refit onto the eight contexts one at a
  time in two random orders (seed 26), then onto all. History B: as A, but a table on the foreign context {M,R,Y} (from P* with the
  M x R x Y interaction raised by 1.5) is refitted after the third step; the final refit uses only the current data. Trace: log R_B −
  log R_A; attribution by span residuals. Decision functional Δ(target); its range over J* and over the slice reachable by edits on
  {M,R,Y} (numerical search: 20 random starts, local optimisation).
- Protection (Thm 122): decision g = P(Y=1 | M=1) − P(Y=1 | M=0) ≥ τ2 = g(P*) + 0.04 (a policy threshold set relative to the
  designed truth). Honest reports h: the pooled non-{M,C,Y} contexts; coalition reports: the pooled {M,C,Y} table including the
  coalition. Inspection q: pooled holdout {M,Y}. η = 3 Σ_cells SE(q̂) in l1; κ = η; ρ_q on J*-type exact data (min l1 change of the
  {M,Y} table needed to reach g ≥ τ2 while matching h); Δ_q = min ‖M_q p − q_obs‖_1 over p matching the current reports (z rule).
- Holdout: dimension and VOI after adding {M,C,R,Y}; calibration of J_z development ranges for the {M,R,Y} and {M,H,Y} cells against the
  holdout estimates (a cell is covered if its holdout estimate lies in the development range widened by 3 holdout SE).

## Pre-registered checks (engine/study26_e2e/run_study26.py; ids XE)

IDENTITY = guaranteed by a theorem given the computation; PREDICTION = could fail, reported either way.

- XE1 IDENTITY  J* is non-empty; dim J* = 32 − rank of the context map = 9 (Thm 64(2)); adding {M,C,R,Y} gives 6.
- XE2 PREDICTION the flagging rule flags at least 8 of the 10 other-population sources and at most 3 others; it flags fewer than half of
  the 15 coalition sources.
- XE3 PREDICTION J_z is empty with the flagged sources included and non-empty without them (population mixing shows as incompatibility).
- XE4 PREDICTION J_z (flagged removed, coalition included) is non-empty: a coherent coalition is not caught by compatibility (Prop. miscal).
- XE5 IDENTITY  on J*: P(π2 | M=1) is a point; P(π1 | M=1) and P(π3 | M=1) are open; Δ(∅) and Δ({C=0}) are points; Δ for the two
  longer K are open (they need {M,R,Y} or {M,H,Y}).
- XE6 IDENTITY  the target decision on J* is identified iff its range excludes τ (Thm 148(i)); reported as certified or open.
- XE7 IDENTITY  Fisher: dim ker I_obs = dim J* (Prop 95), and = 6 with {M,C,R,Y}.
- XE8 IDENTITY  VOI of R: the envelope is a point iff every full-dimensional policy cell's value functional lies in the data span (Thm 92);
  with {M,C,R,Y} added it is a point.
- XE9 IDENTITY  the toric point tends to the leximin coupling (L · distance converges, relative change from 1e3 to 1e4 below 1%) (Thm 27).
- XE10 PREDICTION the wall scan crosses at least one wall; the advisory decision switches at a k different from every wall (decision
  thresholds and selection walls are different objects).
- XE11 IDENTITY  maximum entropy commutes with composition to 1e-12; PREDICTION leximin does not (difference > 1e-6).
- XE12 IDENTITY  maximum entropy is system independent to 1e-12; PREDICTION leximin is not (difference > 1e-6).
- XE13 PREDICTION consistent cycle: J_z non-empty and the largest noncontextuality left side ≤ 2; twisted cycle: J_z empty and the
  largest left side exceeds 2 by more than 3 SE; acyclic (twisted, (E,A) dropped): non-empty.
- XE14 PREDICTION sheaf layer: every cell of every pooled pair table has positive count, so global deterministic sections exist and there is
  no logical contextuality in either condition; the noncontextual fraction is below 1 − 0.05 for twisted and above 0.98 for consistent.
- XE15 IDENTITY  Prop 137: the acyclic and consistent fibres are not single points yet not contextual.
- XE16 IDENTITY  history A: both orders end at the same law (path independence, Thm 146(iii)); history B ends elsewhere; its trace
  lies in H_D + F_{M,R,Y} and not in H_D + F_{M,H,Y}.
- XE17 IDENTITY  the {M,R,Y}-slice range lies inside the J* range of Δ(target); reversal of the honest decision by history B type edits
  is reported as possible or not (Thm 148(iii)).
- XE18 IDENTITY  protection: ρ_q > 2η implies honest reports are not flagged and the coalition's coherent configuration on the far side of
  τ2 is flagged (Thm 122(iii)); PREDICTION ρ_q > 2η holds here and the pooled contaminated decision is on the far side.
- XE19 PREDICTION holdout calibration: at least 90% of the {M,R,Y} and {M,H,Y} holdout cells are covered.

Failures are recorded and restated as findings; no threshold is changed after the run. Deviations go to DEVIATIONS_study26.md.
