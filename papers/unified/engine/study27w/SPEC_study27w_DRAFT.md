# Study 27w — the witness braid, exact toy (DRAFT, NOT FROZEN)

Author: Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu
Status: draft for review. Nothing has been run. On approval: fix the [DECIDE] items, save as SPEC_study27w.md, record its SHA-256 in
FROZEN_SPEC_study27w.txt, then run engine/check_study27w.py once (check ids WB1-WB8). Deviations go to DEVIATIONS_study27w.md; no
threshold changes after the freeze. Separate from Study 27b; nothing here enters 27b's primary endpoints.
Question: can a braid built from discrepancy trajectories against an independent anchor separate an honest cascade from a coordinated
relay when the event trace, the event braid and local compatibility are identical? Expected answer (from Prop. prop:outer): yes with a
timely anchor, no with a lagging anchor, no without an anchor.

## 1. World
- Variables X, Y binary. Reference law P0: P(X=1) = 0.30, P(Y=1 | X=1) = 0.40, P(Y=1 | X=0) = 0.30. Decision quantity
  g = P(Y=1 | X=1) - P(Y=1 | X=0) (g(P0) = 0.10).
- Four sources S1..S4 report the {X,Y} table. Honest margin-only sources report the X and Y margins of the current population.
- Event graph: the path S1 - S2 - S3 - S4; co-revision events (S1,S2), (S2,S3), (S3,S4) in this order, one per round (rounds 1-3), with a
  preliminary round 0. Event braid: pce/raagbraid on the straight drawing of the path (N = 2).
- Far-side law Q: the coupling of P0's X and Y margins with g(Q) = g(P0) + 0.10 (exact, solved in closed form). Q is locally compatible
  with every margin report (same X and Y margins).

## 2. Scenarios (same event trace in C1-C2, by construction)
- C1 honest cascade: the population moves from P0 to Q at round 0 (a real change). Each source updates to Q when its event occurs
  (S1, S2 at round 1; S3 at round 2; S4 at round 3). Anchor variants: C1t timely (anchor table sampled from the population after round 0,
  so it equals Q); C1l lagging (anchor sampled before, equals P0).
- C2 coordinated relay: the population stays at P0; sources move to Q on the same schedule. Anchor equals P0.
- C3 coordinated lockstep: all four move to Q at round 1 through the events (S1,S2) and (S3,S4) together (a different trace; descriptive).
- C0 no anchor: C1 and C2 with the anchor strand removed.

## 3. Witness braid (all conventions fixed here)
- Coordinate of a report T: z(T) = [g(T) - g(P0)] + i [P_T(X=1) - P0(X=1)] (real part: the decision quantity, the direction of the
  protection witness; imaginary part: the X margin).
- Strands: S1, S2, S3, S4, A (anchor), with base offsets b_k = s k (1 + 0.3 i), k = 0..4 in that order, s = 0.005 [DECIDE s].
  Strand k at round t is at b_k + z(report_k(t)).
- Interpolation: straight lines between consecutive rounds; within a round, strands that move do so simultaneously. Crossings are read
  from the projection to the real axis; the strand with the larger imaginary part passes in front. Exact ties are broken by the order
  k (pre-registered); WB7 reports whether any tie occurred.
- No closure: the braid is read in the unordered configuration space, with start and end configurations identified with standard
  positions by their real-axis order. (A straight-line return to the base order was considered and rejected: here every move is
  horizontal, so a strand that passes behind another and returns behind it is unlinked, and the closed braid would be trivial in every
  scenario. This is the one-coordinate degeneracy noted in the discussion; the toy keeps the imaginary parts fixed, so the information sits
  in the open braid.) Reported: the braid word, its Artin normal form, and the anchor crossing numbers c_kA (exponent sum of the
  crossings between strand k and A in the word).
- Statistic: Phi = sum_k |c_kA|. Rule (pre-registered): call 'relay-like' if Phi >= 2.

## 4. Exact checks (no noise)
- WB1 IDENTITY  C1 and C2 have the same event trace (Cartier-Foata form), the same event braid key, and the same final reports.
- WB2 IDENTITY  every source report in C1 and C2 is locally compatible with the margin reports at every round (local sheaf sees no
  conflict); the decision g has moved by 0.10 at the end in both.
- WB3 IDENTITY  timely anchor: Phi(C1t) = 0 and Phi(C2) = 4; the source-source part of the witness braid is the same in C1t and C2.
- WB4 IDENTITY  lagging anchor: the witness braid of C1l equals that of C2 (same normal form) - the confound of 27a world 3.
- WB5 IDENTITY  no anchor: the source-only witness braids of C1 and C2 coincide.
- WB6 FINDING   C3: witness braid and Phi, and whether the increments of the independent events (S1,S2) and (S3,S4) commute (secondary;
  the candidate homomorphism condition of the memo, recorded, not claimed).
- WB7 IDENTITY  robustness of conventions: WB3-WB5 are unchanged for s in {0.0025, 0.005, 0.01} and with the anchor placed first instead
  of last in the base order; no exact ties occurred.

## 5. Noisy arm (one pre-registered prediction)
- Each source table estimated from n = 400 draws; anchor from N_A draws, N_A in {3000, 12000, 120000}; 2000 replicates per scenario and
  N_A; seed 2727.
- Eligibility, by the rule of Prop. prop:outer applied to the anchor: eligible if 0.10 > 2 eta_A, eta_A = 3 SE(g_hat_A) at N_A
  (computed exactly from P0 before the run and recorded).
- WB8 PREDICTION  accuracy of the rule Phi >= 2 for C2 (relay-like) vs C1t (cascade) is at least 0.90 at every eligible N_A, and below
  0.90 at every ineligible N_A [DECIDE 0.90]. C1l is reported (expected relay-like at every N_A: the confound).
- Reported, not tested: distribution of Phi per scenario; how often noise produces crossings that cancel in the group (word length minus
  normal-form length).

## 6. What would change the plan
If WB3 fails, the witness braid as defined does not separate even exactly, and the construction must be revised before any further
study. If WB8 fails in the 'eligible' direction, anchor eligibility is not sufficient for the witness statistic, and that is a finding.
