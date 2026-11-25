# Study 91C NOTES (after the run; diagnoses are POST HOC)
Run 1 (PREREG91C): DJ1 FAIL 5095/5170 certified; DJ2 FAIL (same 75; c4 never failed alone). Ties 955/981, off-tie 4140/4189.
Paths 1 and 2 at L = 20: all certified (864, 866). Every L = 20 failure (path 0, 23) and most L = 30 failures were the residual condition.
AMEND1 (frozen before the rerun; written when only path 0 and path 1 L30 had been seen -- its sentence "distinctness holds for all of
them" was true of what had been seen then; path 1 L30 later added one distinctness failure and one tracking failure):
DJ1b FAIL 5149/5170; DJ2b FAIL 5149/5170. 54 of the 75 certified at 50 digits.
Remaining 21 (post hoc reading, no check changed):
- 19 on path 0 at L = 30, at t = 8.629, 9.6596 (the tie whose L = 20 collisions Study 91 failed to record, inside a multiple-caustic
  substep) and a cluster of 16 at t = 9.7735-9.7750 around the tie at 9.774245. All converge to residual ~1e-37 and are well separated
  (c2 up to 1.0), but the row-equilibrated Jacobian has sigma_min / sigma_max ~ 4e-16 < 1e-14 even at 50 digits. Row equilibration does not
  remove column scale (one |u_i + v_j| tiny); whether these points are genuinely near-degenerate or only badly scaled is not decided.
  They stay uncertified.
- path 1 L30 t = 8.899998: c2 = 5e-55, i.e. the two "colliding" strands are the SAME critical point at the event: the recorded event is
  an artefact (two strands of the walk merged, or the pair re-tracked onto one solution); not a collision of distinct points.
- path 1 L30 t = 7.724197: Newton does not converge from the tracked point (residual 1.3e-3; tracking to the event failed in run 1).
Paper: prop:disjoint is cited for the 5149 certified collisions only. No ledger row (no PASS check).
