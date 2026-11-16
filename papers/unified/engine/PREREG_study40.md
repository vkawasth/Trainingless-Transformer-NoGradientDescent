# Study 40: the leximin point of a two-way table is tropical rank one; leximin by peeling. Pre-registered 2026-11-15 before any code for
it ran. Proof first: RK1-RK4 check proved statements; RK5-RK6 record. Check ids RK (unused before).
An exploratory computation before this registration (60 tables, 2x3/3x3/3x4/4x6, not pre-registered) suggested the statement; it is
reported as exploratory and none of its tables are reused (new seeds).

Proved results being checked (paper, sec:troprank):
  prop:troprank1  For a two-way table, with leximin taken over all cells, x is the leximin point of its fibre iff x is in the fibre and
                  x_ij = min(a_i, b_j) for some numbers a_i, b_j; then one may take a_i = max_j x_ij, b_j = max_i x_ij.
  cor:peel        The leximin point is computed exactly by peeling: among the remaining rows and columns take the smallest ratio
                  (row margin)/(number of remaining columns) or (column margin)/(number of remaining rows); set that whole line to the
                  ratio; subtract it from the crossing margins; repeat. The leximin region of b is its peeling order (with the values
                  at which lines are removed); every leximin wall is a tie between two ratios at some step; first-step ties are
                  f_i/c = g_j/r, circuit walls of prop:gkz; later ties involve partial sums.
Checks (shapes 2x3, 2x5, 3x3, 3x4, 4x6; margins Dirichlet(2) per side, seed 4001).
  RK1 IDENTITY prop:troprank1 (only if): 50 tables per shape: leximin_float point (all cells) satisfies |x_ij - min(rowmax_i, colmax_j)|
               < 1e-9.
  RK2 IDENTITY cor:peel: exact peeling (Fractions) agrees with leximin_float to 1e-9 on the same tables; peeling's level partition
               agrees with leximin_float's.
  RK3 IDENTITY prop:troprank1 (if): 50 random (a, b) per shape (uniform on (0.1, 1)), x = min(a_i, b_j), margins of x: leximin_float of
               those margins equals x to 1e-9.
  RK4 IDENTITY cor:peel walls: on 40 random segments per slice (2x3 free margins and the 3x3 slice of Study 37; seed 4002), every change of
               the leximin signature (bisected to 1e-10, stalls nudged once and counted) occurs where two candidate ratios tie at some
               peeling step: min over steps of (runner-up ratio - chosen ratio) < 1e-7 at the bisected point. Violations 0.
  RK5 RECORD   on the same segments: how many signature changes lie off every circuit wall of prop:gkz (the GZ6 phenomenon), and that each
               is a tie at a step after the first.
  RK6 RECORD   time per table, peeling (exact) vs leximin_float, 4x6.
Output: engine/check_troprank_output.txt, engine/study40_troprank/troprank.json.
