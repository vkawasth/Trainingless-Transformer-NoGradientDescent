# Study 82 AMEND2 (2026-11-21; after run 1 crashed, before any criterion was evaluated; criteria unchanged)
Run 1 (run1_crash.log) crashed in triangles_convex at t = -0.01 (the first ladder point needing a fresh wall82 run): before the wall V6 is
also peg-free, and the exact cylinder cores of V3, V5, V6 (and V3, V4, V6) meet in triple points, giving degenerate triangle loops.
Fix: after the core placement each peg-free curve is offset transversally by 0.03 (q + 1) times the free width of its cylinder (q = its
position 0..5): generic position, continuous in t, always inside the cylinder (offset < width / 2), so the isotopy class and the class
word are unchanged (still checked at every point). The step-1b pickles (core only) are kept in diag/b82_v1_coreonly; all b82 pickles are
recomputed. Previous wall82.py kept as wall82_v1_triplepoint.py.
