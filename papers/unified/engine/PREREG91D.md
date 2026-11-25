# Study 91D: the 21 Study 91C collisions left uncertified, re-examined with a two-sided scaled conditioning measure.
# Registered 2026-11-24 before computing it. Prefix RQ (grep: unused). A DIFFERENT measure from 91C, labelled as such; 91C stands.
## Why
91C's conditioning measure equilibrated rows only. For 19 events (path 0, L = 30) the Newton iterates converge (residual ~1e-37 at
50 digits) and the points are well separated, but sigma_min/sigma_max ~ 4e-16. Row scaling cannot remove column scale (an unknown
u_i or v_j of very different size), so the measure cannot tell near-degeneracy from bad scaling.
## Measure
At each event, from 91C AMEND1's 50-digit procedure (same tracking, 12 mp Newton steps), take J at each point and equilibrate rows
AND columns (Ruiz: 30 alternating passes dividing each row, then each column, by the square root of its max abs entry), then
kappa2 = sigma_min/sigma_max of the scaled J (mpmath svd, 50 digits). Two-sided diagonal scaling is a change of units of equations and
unknowns; it does not change whether J is singular.
## Checks
RQ1 for each of the 19 conditioning events: kappa2 >= 1e-14 at both points, with 91C's c1, c2, residual thresholds unchanged.
    PASS if all 19 meet it. Events meeting it are reported as "certified under the two-sided measure (91D)", separately from 91C.
RQ2 RECORD: kappa2 for the two other events (same-point artefact; non-converged), and for 10 events certified in 91C run 1 chosen by
    the registered rule (the first 10 certified collisions by time on path 0, L = 30, as stored), as a reference.
Failures recorded and kept.
