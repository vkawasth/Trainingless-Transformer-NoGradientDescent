# Study 33: the running example of the opening chain (sec:chain). Pre-registered 2026-11-14 before any code for it ran.
Exact computations; IDENTITY checks of stated results and RECORDs. No hypotheses.

Examples.
  Cameo   the 4-cycle (CHSH cover) with uniform marginals and correlations (l, l, l, -l), l = 0.9 (contextual) and l = 0.5 (not).
  Empty   the binary triangle (A, B, C): uniform marginals, P(A != B) = P(B != C) = P(A != C) = 1.
  T       the compatible 3 x 3 x 2 triangle table of Study 31's main figure (P332 table 7436), observed only through AB, BC, AC.
          Its fibre J_d is a 4-polytope. Decision: act (utility 1{A = 0, C = 0} - t) or pass (0), with t the midpoint of the band of
          P(A = 0, C = 0 | B = 0) over J_d.
  Case A  signal Z = C (C is observed together with A in context AC).
  Case B  signal Z = B (A, B, C are never observed together).
  Selection: maximum entropy (iterative proportional fitting of the three pairwise tables from the uniform law) and the leximin
          point (lexicographic max-min of the 18 cells over J_d).
  E1      the frontispiece example; its maximum-entropy law is the product f g^T.
Checks.
  CY1 IDENTITY  Cameo: the l = 0.9 data violate a Clauser-Horne inequality and its fibre is empty (LP infeasible); l = 0.5 satisfies all
                eight and its fibre is nonempty. Empty: the fibre is empty (LP infeasible), as the odd-cycle parity argument predicts
  CY2 IDENTITY  T: dim J_d = 4; every enumerated vertex and both selected laws reproduce the three pair tables exactly (rational) or to
                1e-12 (IPF, leximin); maximum entropy and leximin are different laws
  CY3 IDENTITY  Case A (thm:determination): the value functional of every full-dimensional policy cell is orthogonal to the direction
                space of J_d, and VOI is constant on J_d (all vertices, both selected laws, 200 random points; spread < 1e-12)
  CY4 IDENTITY  Case B: some value functional is not orthogonal to the direction space; VOI is not constant: its maximum over J_d (at a
                vertex) exceeds its minimum (a linear programme) by more than 1e-6; along the segment of J_d through the two selected
                laws, extended to the boundary, the pair tables never change while the action given B = 0 changes at a located wall
  CY5 RECORD    VOI under Case A and Case B at maximum entropy, at leximin and over J_d; the actions; the position of the wall
  CY6 IDENTITY  E1: the maximum-entropy law of the fibre is f g^T (IPF from uniform reproduces it to 1e-14) and lies in the interior
Output: check_triangle_output.txt, study33_triangle/triangle.json; figure kahler/fig_triangle.py.
