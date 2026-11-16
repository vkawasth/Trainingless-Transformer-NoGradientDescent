# Study 39a: deviations after the first run (written 2026-11-15 before the second run)
First run (kept as check_gkz_output_run1.txt): GZ1 PASS, GZ2 FAIL (0 segments found on either shape), GZ3 PASS, GZ4 FAIL (200 of 200
mismatches), then a crash in the GZ5 bisection (leximin_float raised "stalled"). GZ6-GZ8 did not run.
Diagnosis and fixes; no threshold, sample size or seed changed.
1. GZ4: code error. The lower-hull test read the hyperplane offset (qhull equation[-1]) instead of the last component of the normal
   (equation[-2]), so it used upper facets. Fixed in hull_K0 and subdivision.
2. GZ2: design error, found from the empty result. On both slices every unbalanced chamber wall is a facet boundary: on 2x3,
   f_i = g_j is f_i' + g_j = 1; on 3x3, f_i = g_j + g_k is f_i + g_l = 1. The sampling required every cell to be able to vanish
   (max f + max g < 0.98), which excludes exactly these walls. The second run samples GZ2 segments without that restriction
   (margins > 0.02, feasible) and takes leximin over all cells, the shift-equivariant definition of prop:shiftinv; where every cell is
   a facet this equals the facet-cell leximin. The paper's statement of prop:shiftinv is made explicit about this.
3. Crash: leximin_float can stall exactly on a wall during bisection. The second run catches the stall, nudges the parameter by
   1e-12 and retries once; a change whose bisection still stalls is counted as unresolved and reported, not dropped silently.
The first-run outcome stays on record: GZ2 vacuous by design, GZ4 failed by a code error.
