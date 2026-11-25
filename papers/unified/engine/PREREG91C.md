# Study 91C: certification of the hypotheses of the disjointness proposition at the Study 91 collisions.
# Registered 2026-11-24 before any quantity below is computed. Prefix DJ (grep 2026-11-24: unused). Synthetic data only.
# Uses the six held-out Study 91 walks (seed 9102) and their stored event lists; nothing in Study 91 is re-scored.

## Why
prop:disjoint (paper, to be added with this registration): if two DISTINCT, NONDEGENERATE critical points have the SAME critical value,
their vanishing cycles along a common path are disjoint, so <V_a, V_b> = 0, the twists commute and the crossing is canonical. The
proposition is a theorem; what must be checked is its hypotheses at each recorded collision. The local braid exponent sigma^2 (k = 1)
additionally needs a SIMPLE crossing.

## Procedure (cert91c.py)
For every event of kind "collision" in walk_9102_{p}_L{20,30}.pkl (p = 0, 1, 2): from the stored sample T[k] <= t < T[k+1], track the
two strands of the pair to the event parameter t along the walk's straight segment (lib91 Sys.track, extended precision + AMEND2
fallback); at t apply 8 Newton steps (Sys.newton) to each point. Scale s = 1 + max(|v_a|, |v_b|); xs = 1 + max |x|.
 c1 (equal values)   |v_a - v_b| <= 1e-8 s
 c2 (distinct)       max |x_a - x_b| / xs >= 1e-6
 c3 (nondegenerate)  for both points: relative residual <= 1e-12 and, for the row-equilibrated Jacobian, sigma_min / sigma_max >= 1e-14
 c4 (simple)         |d(v_a - v_b)/dt| >= 1e-8 s, central difference with h = 1e-7 (tracked)
An event is CERTIFIED if c1-c3 hold (hypotheses of prop:disjoint); SIMPLE if also c4 (hypothesis of the sigma^2 local braid).
Ties: an event is a tie event if it lies within 1e-6 of a margin tie (lib91.ties_on), as in Study 91.

## Checks
DJ1 every collision event of the six walks is CERTIFIED (tie and off-tie, L = 20 and 30).
DJ2 every collision event is SIMPLE.
DJ3 RECORD: counts certified / simple by walk, tie vs off-tie; the failing events with the failing condition; distribution of c3.
Failures recorded and kept; no threshold changes; amendments before reruns. The paper cites prop:disjoint for exactly the certified
events, whatever the outcome.
