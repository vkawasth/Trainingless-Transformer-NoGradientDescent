# Study 38: each decision rule has its own arrangement; the one-scalar decision witness; one-sided vanishing edges.
# Pre-registered 2026-11-15 before any code for it ran. Proof first: DW1-DW6 check proved statements; DW7 is the one hypothesis.

Proved results being checked (paper, sec:witness and sec:lefschetz):
  lem:ownarr (i)  the leximin point x*(b) is continuous and affine on each leximin region; a chamber wall inside a leximin region
                  changes nothing that a selected-point decision uses.
  lem:ownarr (ii) phi_w(b) = min over the fibre of w.P is convex, continuous and affine on the closure of each chamber; a leximin
                  wall inside a chamber changes nothing that a robust (credal-bound) decision uses, including threshold questions
                  min (n - tau d).P > 0.
  thm:witness (i)   a decision is constant on every connected set on which no pairwise gap between actions vanishes;
              (ii)  on a region R of the rule's own arrangement each gap is one affine scalar alpha_R.b + beta_R, and the flip time along
                    b + t Delta, while the path stays in R, is t = -(alpha_R.b + beta_R)/(alpha_R.Delta);
              (iii) for a binary question [G > 0] on an open region R where G has a zero, a linear statistic T is sufficient iff
                    ker T is inside ker alpha_R: one scalar, and no smaller statistic.
  prop:onesided (only-if part) at a generic point of the wall of the cut (I, J), with kappa = sum_I f - sum_J g, the side kappa > 0
                  carries a vanishing edge only if |Z+| >= 2, where Z+ = cells (i, j), i in I, j not in J (Z- symmetric for kappa < 0).

Slice: 3 x 3 margins, g = (24, 21, 15)/60 fixed (as Studies 30 and 37), f in the simplex with min f > 0.02. Leximin over facet cells
with leximin_float of check_four6.py; chambers = sets of feasible spanning trees; phi_w by enumerating vertices.
Decision problems: 3 actions with utilities U_a in R^9 drawn N(0, 1) (seed 38). Selected-point rule: argmax_a U_a . x*(b).
Robust rule: Gamma-maximin argmax_a phi_{U_a}(b). Threshold question: [phi_{n - tau d}(b) > 0] with n = cell (0,0), d = row 0,
tau = 0.4. Ties broken by action index.

Checks.
  DW1 IDENTITY lem:ownarr(i): on 30 segments of length 0.03 whose chamber changes but whose leximin signature is constant at 31 equally
               spaced points, x* is affine along the segment: max second difference < 1e-7. Search at most 20000 candidates;
               fewer than 20 segments found counts as a failure.
  DW2 IDENTITY lem:ownarr(ii): on 30 segments of length 0.03 whose chamber is constant at 31 points but whose leximin signature changes,
               phi_w is affine along the segment for the three U_a and the threshold functional: max second difference < 1e-10.
               Same search limits.
  DW3 IDENTITY thm:witness(i): on 40 random segments of length 0.15, both rules, 101 points: every adjacent pair of points with
               different decisions has, between them, a sign change of the gap between those two decisions, and at the bisected
               flip point (1e-10) that gap is below 1e-6 in absolute value. Violations: 0.
  DW4 IDENTITY thm:witness(ii): among the DW3 flips, those whose own-arrangement region is the same at every sample point from the
               segment start up to the flip: the flip time predicted from the start's affine formula (slope from the first two sample
               points) matches the bisected flip to 1e-6 (segment-parameter units). At least 10 such flips, violations 0.
  DW5 IDENTITY thm:witness(iii): at 30 grid points whose region contains a zero of a selected-point gap (two actions), moving along
               the direction of the slice orthogonal to alpha_R within R changes the gap by less than 1e-7, while moving along alpha_R
               through the zero changes its sign.
  DW6 IDENTITY prop:onesided: on 40 fresh segments (seed 3801) crossing exactly one chamber wall f_i = s (s = g_j or g_j + g_k), on
               every side with |Z_side| = 1 the shortest edge is not proportional to the distance to the wall (ratio over 9 points
               not constant to 1e-6, or bounded below by 1e-3).
  DW7 HYPOTHESIS converse of prop:onesided: on the same segments, every side with |Z_side| >= 2 carries an edge whose length is
               proportional to the distance (ratio constant to 1e-6), in all sides tested. (The WX1 outcome of Study 37 is known:
               32 of 40 segments two-sided, all 8 one-sided walls of the form f_i = g_j + g_k. It did not record which side; the
               side is the new prediction.)
  DW8 RECORD   the common refinement on the 91-point grid of Study 37: number of distinct (leximin signature, chamber) pairs.
  DW9 RECORD   coarsening rows {1,2} -> one row (invisible margin direction f1 - f2), utilities constant on merged rows (seed 38),
               on 25 values of f3: fraction where the merged selected point varies over the compatible fine margins (more than 1e-6),
               fraction where the selected-point decision is constant over them, and the same for the robust rule.
Output: check_witness_output.txt, study38_witness/witness.json.
