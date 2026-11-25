# Study 91E: nondegeneracy of the 19 conditioning events measured intrinsically (Hessian of W in torus log-coordinates).
# Registered 2026-11-24 after Study 91D (RQ1 FAIL) and before computing anything below. Prefix LH (grep: unused).
## Why (the post-hoc observation that motivated it, labelled)
In 91D the two-sided scaled conditioning of the reduced (u, v) Jacobian at the 19 events was ~ 0.25 x min|u_i + v_j| / max|u_i + v_j|
(checked on 4 events and 3 references): it measures how close one cell value p_ij = u_i + v_j is to 0 (a critical point near the
toric boundary), which is a property of the reduced coordinates. Prop:disjoint needs nondegeneracy of W itself: the critical point is
nondegenerate iff the Hessian of W in log-coordinates of the torus, H = sum_ij p_ij m_ij m_ij^T (m_ij the exponent vectors of QF1,
basis of Z^6 as there), is nonsingular. One tiny p_ij removes one of 12 rank-one terms, which need not make H singular.
## Measure
At each event: the 50-digit points of 91D's procedure (same tracking, 12 mp Newton steps); H at each point (50 digits);
kappaH = sigma_min / sigma_max of H (mpmath svd). Basis: QF1's lattice basis (fixed integer basis; the measure depends on it only by a
bounded factor).
## Checks
LH1 all 19 events: kappaH >= 1e-14 at both points (91C's c1, c2, residual thresholds unchanged). PASS if all 19. Events meeting it are
    reported as "certified with the intrinsic nondegeneracy measure (91E)", separately from 91C and 91D.
LH2 RECORD: kappaH at the 10 reference events of 91D and at the two other events.
Failures recorded and kept.
