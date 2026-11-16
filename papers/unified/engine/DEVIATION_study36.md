# Deviation from the frozen plan (recorded 2026-11-15, before any LW result was seen)
The exact arm S was specified as "exact rational arithmetic" by solving the fine leximin recursion in rationals (certify_max). On
symmetric 4 x 6 margins a single exact fine solve did not finish in 10 minutes (ties make the rational reconstruction slow), and the
first run was stopped after 1 h 55 min with no output. Change: the fine leximin point is now *certified* exactly instead of solved
exactly. The candidate is the block-constant lift of the exactly computed weighted coarse leximin point; it is accepted as the fine
leximin point only if (a) it has the fine margins exactly and (b) it passes the certificate of lem:lexunique(ii) in exact arithmetic
(each level set carries a strictly positive relation modulo the earlier levels, found by a small linear programme and verified
exactly). A table where the certificate fails counts as an LW1 failure. Everything else is unchanged.
Addendum (same day, still before any LW result): the exact rational solve also stalled on the small coarse tables, so the coarse weighted
and plain leximin points are likewise computed in floating point and then made exact: the level sets are read from the float solution,
the level values are solved exactly from the margin equations, and the resulting point is accepted only if it has the margins exactly,
is nonnegative, and passes the exact certificate of lem:lexunique(ii) (weights rescale the functionals by positive constants, which
preserves strictly positive relations). Any failure counts against LW1.
Addendum 2 (same day, before any LW result): for merge E, margins equal within the groups {0,1},{2,3} and {0,1},{2,3},{4,5} are
always degenerate (f0 + f2 = 1/2 = g0 + g2 + g4), so the arm-S requirement "nondegenerate" cannot be met there. Arm S now requires only
that every cell be a facet (f_i + g_j < 1); prop:symdescent and the certificate of lem:lexunique(ii) do not use nondegeneracy. Arms P and
R are unchanged (arm P perturbs symmetric E margins, so its base points are degenerate too; its perturbed points are kept only if
nondegenerate, as before).
