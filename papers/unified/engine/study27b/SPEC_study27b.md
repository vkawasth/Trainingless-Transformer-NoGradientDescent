# Study 27b — event representations, decision protection, attribution, and the witness braid (specification, frozen before any run)

Author: Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu
This file is frozen (SHA-256 in FROZEN_SPEC_study27b.txt) before any 27b code is run. The study is run once: engine/run_study27b.py
writes study27b/results_study27b.json and study27b/RESULTS_study27b.md; engine/check_study27b.py (ids ZB1-ZB...) re-checks the frozen
identities and records each hypothesis verdict. Deviations (bugs, impossible items) go to DEVIATIONS_study27b.md with a restated claim; no
threshold, margin, loss, size or seed changes after the freeze. Decisions taken by the author on 2026-11-11: inspection arm 'both'
(control at N = 3000 and {M,Y} at N = 120,000), the proposed numbers, and the witness arm as secondary hypothesis H5.
Inputs fixed by earlier work: Study 26 world (study26_e2e/world26.py), Study 27a (frozen spec b8c45e15..., results), Theorem 138
(thm:relayclock), Proposition prop:outer. Not in this study: any audit of the selection theorem.

## 0. Fixed constants
Sources 0..5 on the hexagon graph of 27a: edges (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3); valid drawing; braid key N = 2. Rounds
t = 0..12 (T = 12). Truth P = world26.truth(); decision quantity g(p) = P(Y=1|M=1) - P(Y=1|M=0); g* = g(P); threshold tau = g* + 0.04.
Far-side law Q = P + (0.15 / G) (p_max - P), where p_max maximises g over laws matching P on the honest contexts below (LP, HiGHS) and
G = g(p_max) - g*; so g(Q) = g* + 0.15 and Q matches P on every honest context. Shock law P_shock = world26.shifted().
Source contexts: 0 {M,C}; 1 {M,R}; 2 {C,R}; 3 {M,C,Y} (acting member); 4 {M,H}; 5 {R,H,Y}. Honest contexts: those of 0,1,2,4,5.
Sample sizes per report: sources n = 6000; anchors {M,R,Y} and {M,H,Y}: N = 3000 each; inspection {M,Y}: N = 3000 (control) and
N = 120,000 (protection arm). Every source and anchor re-reports every round: a fresh multinomial sample of its current law.
Seeds: representation arm 27002; decision arm 27003; bootstrap 27004; adaptive search 27005.

## 1. Three event representations
| Code | Representation | Computed by | Preserves | Forgets |
|------|----------------|-------------|-----------|---------|
| R0 | full ordered history | the word | all order and multiplicity | nothing |
| R1 | trace | Cartier-Foata normal form (outerrelay.layers) | dependence order, multiplicity | order of independent events |
| R2 | group class | reduced RAAG normal form (outerrelay.reduce); braid key N = 2 recorded as secondary | equality in A_G | also cancelling pairs |
Identities used: R0 -> R1 -> R2 are functions; activation is a function of R1, not of R2 (Theorem 138).

## 2. History generator (both arms)
- Background (all scenarios): K ~ Poisson(10) events; each on a uniformly chosen edge with a uniform sign, at a round uniform on 1..12.
  Corrections: after each background event, with probability 0.15, its reversal is inserted G ~ Geometric(0.5) events later in the
  event order (G >= 1; appended at the end if past it), at the round of the event it follows. The word lists events by round, ties in a
  uniformly random order.
- Relay chain (coalition and cascade scenarios): one of the two side paths 0-1-2-3 or 0-5-4-3 (uniform), three events with sign +1 at
  rounds r0, r0+1, r0+2, r0 uniform on 1..6, inserted into the background word at those rounds.
- Activation round a_j: process the word in order; the initiator 0 is active from round 0; an event on an edge with one active end
  activates the other end at the event's round (fastest-schedule semantics of 27a, applied to the recorded rounds).
- Labels: E2 = 1 for coalition histories (relay present), 0 for honest background; E3 = 1 iff source 3 is reached from 0 in the trace
  (3 in outerrelay.activation(word, 0)).

## 3. Representation arm (H1-H4)
- Data: 6000 training and 6000 test histories per regime; regimes: honest background (E2 = 0) and coalition-fixed (E2 = 1). Seed 27002.
- Features (fixed): R0: indicators of (edge, sign, position) for positions 1..30, and counts of ordered consecutive pairs
  ((edge, sign), (edge, sign)); R1: indicators of (edge, sign, Cartier-Foata layer) for layers 1..30, and the number of layers;
  R2: the R1 features of the reduced word.
- Learner (identical for all): sklearn LogisticRegression, L2, lbfgs, max_iter 5000; C chosen by stratified 5-fold CV on the training
  split over {0.01, 0.1, 1, 10} by log loss; one model per (representation, label).
- Transformations (on test histories, verified per history):
  H_comm: 20 random swaps of adjacent independent events (keeps R1, R2, activation). H_pad_safe: insert e, e^-1 adjacent at a random
  position, accepted only if the activation map is unchanged (keeps R2; retries up to 50, else the history is dropped and counted).
  H_mixed: H_comm then H_pad_safe. H_pad_op: insert e, e^-1 at a random position, kept only if the activation map changes (keeps R2;
  50 retries). Transformed test sets: 2000 histories per family (1000 honest, 1000 coalition, the first in test order).
- Adaptive coalition: each coalition test history is transformed by up to 3 steps; each step draws 20 candidates (H_comm or
  H_pad_safe, equal probability, seed 27005) and keeps the one minimising the frozen R0 E2-score, if lower.
- Loss: misclassification rate (primary), log loss (secondary). eps = 0.02. Paired bootstrap over test histories, 2000 resamples,
  95% percentile intervals.
- H1 (trace compression, equivalence): CI of err(R1) - err(R0) on E2, unmodified test, inside (-0.02, 0.02). E3 reported.
- H2 (braid compression): CI of err(R2) - err(R1) on E3 has lower end > 0.02; on E2 the CI lies inside (-0.02, 0.02).
- H3 (evasion robustness): identities: R1 features unchanged under H_comm; R2 features unchanged under H_comm, H_pad_safe, H_mixed
  (so Delta_R1 = 0 and Delta_R2 = 0 exactly there). Test on E2: CI of Delta_R0 - Delta_R1 under H_comm has lower end > 0.02; CI of
  Delta_R0 - Delta_R2 under H_pad_safe has lower end > 0.02. Delta = err(transformed) - err(unmodified) on the same histories. H_mixed
  and the adaptive coalition are reported.
- H4 (operational safety): identity part: on H_pad_op pairs R2 is unchanged while activation changes (every pair, by construction,
  verified); reported: the proportion of attempts that are operational, and the proportion of op pairs where E3 changes. Empirical part:
  E3 error of the R2 model vs the R1 model on the H_pad_op set, paired CI (prediction: R2 higher, lower end > 0.02).
None of H1-H4 predicts that a compressed representation beats R0 on unmodified histories.

## 4. Decision arm (endpoints 1 and 2)
Scenarios, 1000 runs each (seed 27003); every run is a history (section 2) plus report sampling for rounds 0..12:
- S_relay_fixed: background + relay chain; truth P throughout; source 3's law is P + s(t)(Q - P), s(t) = min(1, (t - a_3)/4) for
  t >= a_3, else 0.
- S_relay_adaptive: as fixed, with the history transformed adaptively against the frozen R0 E2 model (section 3).
- S_honest: background only; truth P.
- S_shock: background only; truth P until round 3, P_shock from round 4 (every source and anchor).
- S_cascade_timely: background + relay chain; truth P at round 0, Q from round 1; source 3 reports P until a_3 and Q from a_3; anchors
  and inspection sample the current truth.
- S_cascade_lagging: as timely, but anchors and inspection sample P at every round.
- S_drift: background only (no chain); truth P; source 3's law is P + s(t)(Q - P) with a_3 replaced by a round uniform on 3..8.
Models (all see the reports; flags are cumulative: once flagged, flagged):
- Decision rule (all models): g_j from the {M,Y} margin of each report that contains M and Y (source 3; anchors; inspection when the
  model has it); g_hat = inverse-variance weighted mean (variances from the design: SE of g at P with that report's size); SE_c = its
  standard error. Act if g_hat - tau > 2 SE_c; not act if tau - g_hat > 2 SE_c; else undecided.
- B2: flag at round t if the reconciled set {p >= 0, sum p = 1 : ||M_k p - report_k||_1 <= eta_k for every source and both anchors} is
  empty (LP), eta_k = 3 sum_cells SE (Study 26 rule).
- B3: B2 flag, or the number of events involving source 3 up to t exceeds the 95th percentile of that count over the honest training
  histories (full length).
- B4 / B4C / B4T: B2 flag, or the R0 / R1 / R2 E2-model score on the history prefix (events with round <= t) exceeds the threshold
  giving a 5% flag rate on the honest training histories (full length), per model.
- B7_3000 / B7_120k: B2 with the inspection {M,Y} added to the reconciled set (and to the decision rule), at N = 3000 / 120,000.
Flip round: the first round with decision 'act'.
Endpoint 1 (per model and scenario, with 95% Wilson intervals): coalition-type scenarios (relay fixed, relay adaptive): flag before the
flip (or flagged and never flipped); no flag and final 'not act'; final undecided without flag; 'act' at T without an earlier flag (wrong);
flagged after the flip (late, counted wrong). Honest-type scenarios (honest, shock, cascade timely): false flag (any flag); correct final
decision (act iff g(truth at T) >= tau); undecided; wrong. Cascade lagging and drift: the same categories, reported separately.
Decision loss: 0 correct or flag before the flip; 1 wrong or late; 0.2 false flag; 0.1 undecided; averaged over relay fixed, relay
adaptive, honest, shock and cascade timely with equal weight. Noninferiority (eps = 0.02, paired bootstrap over runs): B4C vs B4 and
B4T vs B4 on this loss, both directions reported.
Endpoint 2 (never merged with endpoint 1), at the first flag round of B2 and of B7_120k: minimal explanations = the smallest subsets
(size 1, then 2) of {sources, anchors, inspection} whose constraints, when removed, make the reconciled set non-empty; named = their
intersection (27a rule; nobody is called malicious); categories: correct identification (named = {3}) in relay and drift; false culprit
(named contains any other source or an anchor); detected but ambiguous (flag, named empty or no explanation of size <= 2).
Estimation only: endpoints 1 and 2 carry no pass/fail prediction.

## 5. Witness arm (H5, secondary) and the explanation map
Strands: the six sources and the anchors; configuration A: anchors {M,R,Y}, {M,H,Y} (N = 3000); configuration B: A plus the
inspection {M,Y} at N = 120,000. Coordinates of strand k at round t: x_k(t) = f(report_k(t)) - f(report_k(0)), with f = g of the {M,Y}
margin when the context has M and Y, else 0; y_k(t) = P_report(M=1) change from round 0 if M in context, else P(Y=1) change, else P(C=1)
change. Base positions: sources k = 0..5 at (k 1e-4, k 1e-3); anchor j at (kappa_j + (6 + j) 1e-4, (6 + j) 1e-3), with
kappa_j = 2 sqrt(SE_3^2 + SE_j^2) (design SEs of g: source 3 at n = 6000, anchor j at its N). Strand k sits at base_k + (x_k, y_k).
Interpolation linear between rounds; crossings read from the projection to the x axis, the strand with larger y in front; ties broken by
strand index; no closure (endpoints identified by x order). c_kA = exponent sum of crossings between source k and anchor A.
Phi = sum over sources and anchors of |c_kA|. Rule: relay-like iff Phi >= 1.
- H5 prediction: balanced accuracy of the rule on S_relay_fixed (positive) vs S_cascade_timely (negative) is >= 0.90 in configuration B
  and < 0.90 in configuration A.
- Explanation map (descriptive): for each run, E = {sources k with sum_A |c_kA| >= 1}; for each k in E, its last net-crossing round and
  whether k was activated (trace, section 2) at or before it ('explained by the relay'); agreement of E with the planted set ({3} in
  relay and drift, empty otherwise) and with the endpoint-2 naming. Tabulated by scenario and configuration, plus the Phi distribution and
  the number of crossings cancelled in the group (word length minus reduced length).
- Exact toy (identities, no noise; the draft SPEC_study27w_DRAFT.md world: X, Y binary, P0 with P(X=1) = 0.3, P(Y=1|X=1) = 0.4,
  P(Y=1|X=0) = 0.3; four sources on a path, events (S1,S2), (S2,S3), (S3,S4) at rounds 1-3; Q0 with the same margins and g + 0.10;
  base offsets b_k = s k (1 + 0.3 i), s = 0.005, order S1..S4, A; same crossing rules, no closure):
  WB1 C1 (cascade) and C2 (relay) have the same trace, event braid key and final reports; WB2 every report is locally compatible with the
  margin reports; WB3 timely anchor: Phi(C1t) = 0, Phi(C2) = 4, same source-source crossings; WB4 lagging anchor: C1l and C2 have the same
  braid word; WB5 no anchor: source-only braids coincide; WB6 (finding) lockstep C3 braid and Phi; WB7 WB3-WB5 hold for s in
  {0.0025, 0.005, 0.01}; WB7b (finding) with the anchor first in the base order (on the side away from the decision move) the relay
  makes no crossing: the base order is part of the definition and is fixed here with anchors on the side of the move.

## 6. Knot arm (secondary, descriptive)
On the witness braids of section 5: a crossing is unresolved when the y-separation of the two strands at the crossing is below
2 sqrt(SE_y,k^2 + SE_y,A^2) (design SEs of the y coordinates). J = all sign assignments of the unresolved crossings (the resolution law;
no convention). For each run and source-anchor pair report the interval [min, max] of c_kA over J and its width W; summarise W by
scenario. Not used in any decision model.

## 7. Freeze checklist (memo items)
covers and restriction maps: contexts of section 0, ctxmat; Study 26 world and anchors: section 0; coalition-controlled group: source 3
({M,C,Y}) activated through the chain; fixed and adaptive protocols: sections 3, 4; honest controls: shock, cascade (timely, lagging),
drift; inspection from 27a: {M,Y}, both N; weak {M,Y} at 3000 as negative control: B7_3000; loss, margin, attribution, confidence:
sections 3, 4; transformations and their verification: section 3; graph, drawing, N = 2, protocol (P): sections 0, 2; knot resolution
law: section 6.
