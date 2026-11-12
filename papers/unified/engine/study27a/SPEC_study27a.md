# Study 27a — exact verification (specification, frozen before the checks are run)

Author: Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu
Purpose: turn Proposition prop:outer (outer reconciliation, protection witness), Proposition prop:relay (relay clock), Proposition
prop:brokerage and Proposition prop:masking into executable exact checks before any simulation (Study 27b). No learning, no sampling of
data; every check returns a certificate or a counterexample. Script: engine/check_study27a.py, check ids ZA1-ZA13.

## Worlds
- S26: the exact world of Study 26 (world26.truth()). Honest local tables: the development contexts other than {M,C,Y}; the coalition
  reports {M,C,Y}. Decision g = P(Y=1 | M=1) - P(Y=1 | M=0); wrong side g >= tau, tau in {g* + 0.02, g* + 0.04, g* + 0.08}.
  Trusted anchors: none, {M,R,Y}, or {M,H,Y}. Inspections: {M,Y}, {M,R,Y}, {M,H,Y}, {M,C,R,Y}, {C,Y}, {M,C,Y}, {C,R,Y}.
- SMALL: three binary variables X, Y, Z; local tables {X,Y}, {Y,Z} from a random positive law (seed 27); outer candidates {X,Z}, {X},
  {Z}, {X,Y,Z} from a second law (to create conflicts).

## Module A1 — local-to-outer LP (ZA1-ZA3)
- ZA1 IDENTITY: LP_honest (local + anchors) is feasible at the truth; for each tau and anchor set: if LP_wrong (plus g on the wrong side)
  is infeasible, the margin LP is infeasible (rho = +inf) and a Farkas certificate exists for the wrong-side system; if feasible, rho = 0
  exactly when a wrong-side law matching the inspected table exists (checked by a separate feasibility LP).
- ZA2 IDENTITY: every positive finite rho equals its dual value, and the dual certificate separates; Farkas certificates for empty outer
  fibres (SMALL, conflicting outer tables) are computed and stored separately; the two are different objects (stored in the summary).
- ZA3 IDENTITY: minimal outer witnesses. SMALL: exhaustive enumeration of minimal infeasible sets of outer tables. S26 with six outer
  tables taken from the other population of Study 26: deletion-based minimisation, labelled inclusion-minimal; check that the result is
  infeasible and every single deletion from it is feasible; on SMALL, deletion results are among the exhaustive minimal sets.

## Module A2 — attribution controls (ZA4-ZA5)
Sources = the eight development contexts of S26 with exact tables. Worlds: (1) one poisoned source, {M,C} from the other population;
(2) two coordinated sources, {M,C,Y} and {R,H,Y} from a single far-side law matching the other six tables (coherent), plus a trusted
outer {M,Y} anchor at the truth; (3) an honest lag: the population shifts to P(M=1) = 0.8, and only {M,C}, {M,R}, {M,H} are collected
after the shift. For each: minimal conflicts, all minimal explanations (exhaustive), the planted set, whether it is a minimal explanation,
whether every minimal conflict meets it, the number of distinct minimal explanations.
Pre-registered procedure: name a source as inconsistent only if it lies in every minimal explanation; never call any source malicious.
- ZA4 IDENTITY: the named set is contained in the planted set whenever the planted set is an explanation (worlds 1-3).
- ZA5 FINDING: the per-world record; in world 3 any naming falls on honest (lagging) sources.

## Module A3 — inspection ranking (ZA6)
For the S26 inspections at tau = g* + 0.04 and no anchors: rho_q, eta_q = 3 sum of cell standard errors at N = 3000, cost c_q = number
of cells; descriptive ranking by R_q = rho_q / (eta_q c_q); eligibility rho_q > 2 eta_q reported separately; N*_q = 3000 (2 eta_q/rho_q)^2.
{M,Y} is the negative control (ineligible at 3000, as in Study 26).
- ZA6 FINDING.

## Module A4 — relay and masking (ZA7-ZA9)
- ZA7 IDENTITY: Prop. brokerage pair, m = 1 and m = 2: relay and sequential interleavings with identical event counts, co-revision counts,
  pairwise linking numbers and own histories; mu_123 differs (relay non-zero, sequential zero).
- ZA8 IDENTITY: masking, computed with the Magnus expansion of pce/braidknot (not pce/masking's depth): lone newcomer (shortest words of
  depth 1-4 between random old histories) has the depth of its turn word; riding (two turns around an old commutator) has depth
  1 + depth(longitude), and the old-source longitude is visible one layer lower.
- ZA9 IDENTITY: mixed interleavings: the defect equals the product of conjugated newcomer turns (60 random histories); no lower bound on
  events is asserted for mixed patterns (open).

## Graph controls (ZA10-ZA11)
Six sources: hexagon in convex position with straight sides plus the diameter (0,3) (planar, triangle-free). Invalid drawing: the same
graph with the positions of vertices 1 and 2 exchanged, so the independent edges (0,1) and (2,3) cross.
- ZA10 IDENTITY: valid drawing: twists of independent edges commute, of dependent edges do not; invalid drawing: the crossing independent
  pair does not commute (the map is not a homomorphism from A_G).
- ZA11 FINDING: faithfulness on the ball of all words of length <= 3, for N = 1 and N = 2, valid drawing (recorded: graph, drawing, N, L);
  a bounded check, not a proof.

## Clock invariance (ZA12-ZA13), on the valid six-source graph, initiator 0
- ZA12 IDENTITY: activation (fastest schedule) is unchanged by swapping adjacent independent events and by any retiming that preserves the
  dependence order (activation computed on random linear extensions of the trace has the same reached set and the same order of activation).
- ZA13 IDENTITY: inserting or deleting an event next to its reversal leaves the braid class unchanged (RAAG normal form and N = 2 braid key)
  and changes the reached set in some cases (counted); the braid is sound for reach (reduced-word reach inside history reach) on 400
  random histories.
