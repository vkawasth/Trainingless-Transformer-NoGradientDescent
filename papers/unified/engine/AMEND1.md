# Study 91C AMEND1 (registered 2026-11-24 after run 1 had produced results for 5 of 6 walks, before any rerun)
Run 1 left some collisions uncertified. Seen so far (path 0 L20: 23, path 0 L30: 32, path 1 L30: 8; paths 1, 2 at L20: none): every
failure is the residual condition (relative residual after 8 extended-precision Newton steps between 7.9e-12 and 3e-5); at L = 30
19 of path 0's also miss the conditioning floor. Distinctness (c2 >= 7e-4) and equal values (c1 <= 4e-14) hold for all of them.
This is the precision problem of Study 91 AMEND1/AMEND2 (roots with tiny |u_i + v_j|), not evidence of degeneracy, but run 1 stands:
DJ1 and DJ2 are scored on run 1 and are FAIL if any event is uncertified there.
Amendment: DJ1b / DJ2b. For every event NOT certified in run 1, start from run 1's extended-precision point at the event, apply 12
Newton steps in mpmath at 50 digits (lib91.MPStrand.F / J, parameters as in run 1), and evaluate c1-c4 with the SAME thresholds,
the residual and the row-equilibrated sigma_min / sigma_max computed at 50 digits (mpmath svd); the derivative c4 by the same tracked
central difference as run 1. DJ1b: every collision certified in run 1 or in the rerun; DJ2b: likewise simple. Script cert91c_mp.py.
Events certified in run 1 are not recomputed.
