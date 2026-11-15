# Study 31: the shape of a conditioned state (sec:credalshape). Pre-registered 2026-11-14 before any code for it ran.
Exact geometry only: every check is an IDENTITY of a stated result (lem:cographic, prop:ratiocone, prop:condpolytope), plus RECORDs
for the figure. No hypothesis is tested.

Data. (a) The 3 x 3 exact-margin population of check_stratledger.py (seed 20261126). (b) The P332 population of Studies 29-30 and its
+-1/100 tolerance version. Figure tables: the P332 tables with the 25th, 50th and 75th percentile of the before-evidence width of
P(A = 0 | B in {0,1}, C = 0); the median one is the main panel. Outcome vector q = (P(A = a | B in {0,1}, C = 0))_{a = 0,1,2}.

Checks.
  SO1 IDENTITY lem:cographic: for 300 tables of (a) and random linear objectives, every optimal vertex is nondegenerate with r + c - 1
               positive cells forming a spanning tree of K_{r,c}; its (r-1)(c-1) zero cells give cell functionals independent on the
               fibre direction space (the circulations of K_{r,c}); a set of (r-1)(c-1) cells whose complement contains a cycle gives
               dependent functionals (300 random such sets)
  SO2 IDENTITY prop:ratiocone: for the five ratio objectives of QL11 on (a): for 2000 random pairs of tables certified by the same
               basis, every point of the segment between them (5 points) is certified by that basis; the defining inequalities are
               homogeneous (scaling b by 2 keeps every certificate)
  SO3 IDENTITY prop:condpolytope(i)-(iii) on the three figure tables: the vertices of the 4-polytope K (enumerated in interaction
               coordinates) satisfy the margins; the block-level image pi(K) has its vertices among the images of K's vertices;
               the outcome polygon from 720 support-function programmes equals the convex hull of the images of K's vertices (1e-9);
               its extreme q_0 equals the band ends of Study 29
  SO4 IDENTITY evidence: for every vertex v of K, pi(phi_L v) = phi_L(pi v) (1e-12); every Hilbert distance between block-level vertex
               images is preserved by phi_L (1e-9); the after-evidence outcome polygon equals the hull of M phi_L(pi(vertices))
  SO5 IDENTITY Fisher-Rao image: square roots of polygon vertices lie on the unit sphere; the band ends are the extreme latitudes of
               the image region
  SO6 RECORD   per figure table: vertex counts of K, pi(K) and the outcome polygon before/after evidence and with tolerance; area and
               Hilbert diameter of the polygon in each state; active facets (zero cells) at each polygon vertex; which change
Output: check_credalshape_output.txt, study31_shape/credalshape.json; figure kahler/fig_credalshape.py.
