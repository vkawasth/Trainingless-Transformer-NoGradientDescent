# Study 37: two arrangements on margin space; vanishing edges; tropical smoothing of leximin walls (sec:lefschetz).
# Pre-registered 2026-11-15 before any code for it ran.
Proved results being checked: prop:realmono (wall crossings in real margin space compose to the identity), prop:vanishedge (at a chamber
wall the fibre's shortest edge shrinks linearly to zero), cor:width (the positive critical point x(L) differs from the leximin point by
L^-1 c_P plus a term exponentially small in L times the distance to the nearest leximin wall, thm:selection(d)).
Slice: 3 x 3 margins, g = (24, 21, 15)/60 fixed (E1), f in the simplex; facet cells as in sec:threebythree.
Checks.
  WX1 IDENTITY prop:vanishedge: on 40 random segments crossing exactly one chamber wall (f_i = s, s a subset sum of g or its
               complement), the fibre's shortest edge length at 9 points on each side is proportional to the distance to the wall
               (ratio constant on each side to 1e-6), and the vertex count differs between the two sides or the edge set changes
  WX2 RECORD   on a 121-point triangular grid: leximin level signatures (ordered level partitions); number of distinct signatures;
               fraction of grid edges where the signature changes that also cross a chamber wall, and vice versa
  WX3 IDENTITY prop:realmono: on 20 closed polygonal loops in the slice (each crossing several walls), the leximin point and its level
               partition after transport around the loop equal those at the start (1e-9), when transport recomputes at each step
  WX4 HYPOTHESIS cor:width: on 20 segments crossing one leximin wall inside a single chamber, the maximum of |x(L) - x_lex - c_P/L| over
               the segment points whose distance to the wall exceeds 10/L is smaller at L = 240 than at L = 60 in at least 18 of 20
  WX5 RECORD   the location of maximal curvature of x(L) along each WX4 segment, against the wall, for L = 60, 120, 240
Output: check_walls_output.txt, study37_walls/walls.json.
