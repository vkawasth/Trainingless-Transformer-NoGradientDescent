# Study 39a: the GKZ identification (which walls the potential can see). Pre-registered 2026-11-15 before any code for it ran.
Proof first. GZ1-GZ5 check proved statements; GZ6 is the one hypothesis; GZ7-GZ8 record.

Proved results being checked (paper, sec:gkz):
  prop:shiftinv  (i) adding s to every cell (margins b -> b + s A1) multiplies F_L by e^{-Ls}: u_L and u^lex are unchanged, x* -> x* + s.
                 (ii) so every leximin wall and every wall of the complex critical geometry is invariant under b -> b + s A1.
                 (iii) the chamber wall of the cut (I, J) has kappa(A1) = |I| c - |J| r = |Z+| - |Z-|; it is invariant iff balanced.
                 Unbalanced chamber walls are never leximin walls.
  prop:gkz       (a) for generic b the vertices of the fibre are the complements of the maximal cells of the regular subdivision of the vector
                     configuration {v_k} with heights lambda = -(M b) (linear functions).
                 (b) the first leximin level's tie set K_0 is the minimal cell, containing 0, of the regular subdivision S(lambda) of the point
                     configuration {v_k} (affine functions; lower hull). Hence K_0 and the first level's formula change only on walls of the
                     secondary fan of the point configuration, which are the hyperplanes mu.b = 0 with A^T mu a circuit of the homogenised
                     configuration (so mu.A1 = 0).
                 (c) cited (GKZ): Newt(E) = secondary polytope, so these walls are the tropicalisation of the principal determinant E.
Shapes: 2x3 (margins (f1, f2, g1, g2, g3), all free with sum 1; random interior points with every margin > 0.06) and the 3x3 slice of
Study 37 (g = (24, 21, 15)/60 fixed). Leximin with leximin_float of check_four6.py over facet cells. Seeds 3901 (2x3), 3902 (3x3).
Checks.
  GZ1 IDENTITY prop:shiftinv(i): 50 random margins per shape, s in {0.01, 0.05}: leximin point of b + s A1 equals x*(b) + s (max 1e-8);
               the critical point u_L (L = 40, Newton) of b + s A1 equals that of b (max 1e-9).
  GZ2 IDENTITY prop:shiftinv(iii): on 40 segments per shape crossing one unbalanced chamber wall at a point at distance > 1e-3 from every
               circuit wall of GZ5, x* is affine across the crossing (second difference over 21 points spaced 1e-4: < 1e-8) and the leximin
               signature is the same on both sides. (3x3: walls f_i = g_j + g_k; 2x3: all walls.)
  GZ3 IDENTITY prop:gkz(a): 100 random margins per shape: set of feasible bases = set of complements of cells sigma (|sigma| = dim of the
               fibre) for which a linear function equals lambda on sigma and is >= lambda off sigma (solved independently as a linear system
               plus inequality check). Mismatches: 0.
  GZ4 IDENTITY prop:gkz(b): 100 random margins per shape: the first-level tie set from leximin_float equals the minimal cell containing 0 of
               the lower hull of the lifted points (v_k, lambda_k), computed with scipy ConvexHull. Mismatches: 0.
  GZ5 IDENTITY prop:gkz(b) corollary: on 40 random segments per shape, every change of the first-level tie set, bisected to 1e-10, lies on a
               circuit wall mu.b = 0 (|mu.b| / |mu| < 1e-7). Violations: 0; at least 20 changes found per shape.
  GZ6 HYPOTHESIS every change of the full leximin signature (all levels) on the same segments lies on a circuit wall. Violations: 0.
  GZ7 RECORD   per shape: number of chamber walls (cuts), how many balanced; number of circuit walls; on the Study 37 grid (3x3): grid edges
               with a leximin change that cross an unbalanced chamber wall and no circuit wall (predicted 0 by prop:shiftinv).
  GZ8 RECORD   2x3: number of distinct regular subdivisions S(lambda) met over 2000 random margins; number of distinct first-level tie sets.
Output: engine/check_gkz_output.txt, engine/study39a_gkz/gkz.json.
