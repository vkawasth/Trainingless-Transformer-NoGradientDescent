# Study 91 NOTES (written after the check; everything below the "Results" line is reporting, and every diagnosis is POST HOC)

## Procedure actually followed
- Registered PREREG91 / PREDICTION91 before seed 9102 was generated.
- Run 1: 4 of 6 walks crashed (cancellation, condition ~1e12) -> AMEND1 (extended precision, clongdouble). Logs: walks_run1/.
- Run 2: path 0 at L = 30 failed at t = 5.925 (condition ~1e17) -> AMEND2 (single-strand mpmath fallback, 50 digits).
- AMEND2 erratum: the first AMEND2 rerun crashed (double-precision conversion rounded a tiny u_i + v_j to 0). Fixed; rerun complete.
  One overflow warning appears in that log; it came from a rejected trial step: every stored X and V is finite (checked).
- Check invocation 1 (check91.py) decided QF1-QF4 and was stopped by the 30-minute background limit during QF5.
  AMEND3 (procedural, registered before resuming): QF5 split into six jobs, QF5/QF6/R7 code copied verbatim (check91_resume.py).

## Results (as registered; failures kept)
QF1 PASS  volume 96, anticanonical margins on no wall, simple polytope with 96 vertices.
QF2 FAIL  96 distinct solutions everywhere, residuals <= 2e-13, 2x3 anchor 9.3e-16, but min |det J| at the L = 30 base point is
          1.0e-15 < 1e-14 (L = 20: 2.8e-10). The registered nondegeneracy floor is missed at L = 30 only.
QF3 FAIL  every one of the 981 tie collisions (all paths, both L) is a symmetric pair (P_b = transposed P_a to 1e-6): 981/981.
          The FAIL is the first clause: two ties at L = 20 (path 0, t = 9.6596, columns 0,1; path 1, t = 3.2248, rows 1,2) carry no
          recorded collision.
QF4 FAIL  off-tie collisions exist on every path (419, 729, 659 at L = 20) and essentially none recur at L = 30 (1, 0, 0; limit 5%),
          but the clause "every tie carries collisions at both L" fails on paths 0 and 1, for the same two ties as QF3.
QF5 PASS  (resumed under AMEND3) 30/30 loops on the three held-out paths at L = 20: all 15 caustic loops wind exactly -3 pi
          (to the printed 3 decimals) with the pair exchanged and the other 94 strands fixed; all 15 off-tie collision loops are pure
          with winding exactly -2 pi. All windings have the same sign (quasipositivity seen, not tested).
QF6 PASS  42 integer weeks: LP band = Frechet band, integral optimal vertices.
R7 RECORD collisions / caustics / Stokes events: p0 L20 541/44/3426, L30 659/44/3847; p1 L20 864/39/4661, L30 1072/52/5127;
          p2 L20 866/33/5358, L30 1168/52/6519. "Other" (multiple caustics in a substep, "caustic?") 1-4 per walk.
Summary: 3 PASS (QF1 exact, QF5, QF6), 3 FAIL (QF2, QF3, QF4). Against PREDICTION91 (all six predicted to pass): three misses.

## Post-hoc diagnosis (labelled; no check is changed by it)
- The two empty ties. In both cases the 1/40-week substep that contains the tie also contains a multiple caustic (the number of real
  critical values changes by more than one pair inside the substep; lib91.analyse records these as "caustic?" and, by its code, skips
  collision detection for that substep). Path 0 L20: substep [9.650, 9.675], real values 38 -> 36. Path 1 L20: substeps [3.200, 3.225]
  and [3.225, 3.250], 40 -> 44 -> 40. At L = 30 the same ties carry 10-16 collisions. So the failure is a detection gap of the
  registered analyser where a caustic and a tie fall in one substep, not evidence against "ties carry symmetric collisions"; but the
  registered check fails and stays failed.
- QF2 at L = 30: the near-singular root is the same phenomenon as AMEND1/AMEND2 (|u_i + v_j| tiny, dynamic range ~ e^{L * spread});
  the floor 1e-14 was chosen from L = 20 calibration and is too strict at L = 30. Recorded, threshold not changed.
- Sign of the windings: lib91.loop_caustic traverses t - rho e^{-2 pi i q/n}, i.e. CLOCKWISE in the t-plane (its code comment
  'counter-clockwise?' is wrong); all 30 windings are negative (-3 pi, -2 pi), i.e. +3 pi / +2 pi for counter-clockwise loops. This is the
  quasipositivity statement of Study 92 Part B3, observed here on 30 loops but not registered as a check in Study 91.
- QF5 path 0 results were rebuilt from the job logs (AMEND3 erratum: json could not write numpy longdouble); paths 1, 2 saved directly.
- check91.py hash differs from the one in PREDICTION91 (changed at AMEND1, recorded in FROZEN_study91 at that time).
