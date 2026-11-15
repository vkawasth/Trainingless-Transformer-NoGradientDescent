# Study 32: the unified-geometry frontispiece (fig:frontispiece, sec:overview). Pre-registered 2026-11-14 before any code for it ran.
One worked example, E1 of sec:threebythree: f = (30, 20, 10)/60, g = (24, 21, 15)/60. Every panel is computed, not drawn, and each
panel has an IDENTITY check. No hypotheses.
  UG1 IDENTITY base of margins: in the slice g = g(E1), f in the simplex, every chamber wall is a line f_i = s with s a subset sum of g
               or its complement; the number of feasible bases is constant on each cell of the line arrangement (checked at 2000
               random points against the cell they fall in) and E1 lies in the interior of a cell
  UG2 IDENTITY the fibre: the vertices of Cpl(f, g) are the feasible spanning-tree bases, their number equals the TT2 record for E1,
               every vertex has exactly 4 neighbours (simple), and adjacent vertices share r + c - 2 = 4 basic cells
  UG3 IDENTITY dual bases at the marked vertex: positive cells form a spanning tree of K_{3,3}; the 4 edges leaving the vertex are in
               bijection with its 4 zero cells (the entering cell of each pivot)
  UG4 IDENTITY Theorem 196 on E1: the exact leximin levels and rank increments reproduce TT2 (r = (2, 1, 1)); the scale valuations of
               Q/W at the positive critical point, extrapolated in 1/L from L = 60, 120, 240, equal h_j - h_1 with multiplicity r_j
               (tolerance 0.01); the positive critical point x(L) converges to the leximin point (distance decreasing in L for
               L = 5..240 on a grid, final distance < 0.01)
  UG5 IDENTITY the decision: for the row block {0, 1} and outcome = column, the conditional region is the hull of the images of the
               fibre's vertices; it equals the product of its three coordinate bands (prop:bandcomplete's two-row analogue); after
               cell weights L(i, j) = 1 + ((i + 2j) mod 3)/2 it is the hull of the transported images and the band ends match direct
               linear-fractional programmes (1e-9)
Output: check_frontispiece_output.txt, study32_frontispiece/frontispiece.json; figure kahler/fig_frontispiece.py.
