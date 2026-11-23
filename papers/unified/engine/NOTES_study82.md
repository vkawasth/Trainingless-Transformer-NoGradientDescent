# Study 82 notes
1. PREREG82 frozen (held-out end point t = 0.025; carried objects frozen; exploratory interval [0.015, 0.0175]).
2. Step 1 EXPLORATORY (dump82.txt): dense ladder (step 2.5e-4). Inside one representative layout every step is smooth (worst
   displacement / half-separation 0.02-0.16). The class-word pattern alternates non-monotonically between two layouts
   A = [3, 5, 13, 6, 1, 14] and B = [4, 2, 10, 10, 1, 11] (A A A A B A B B B A A), and every switch moves the three generators of
   (V3, V6') by ~0.019 (ratio 1.1-1.3). The taut lengths are identical in both layouts. Cause: V3, V4, V5 have no pegs; a peg-free
   closed geodesic in the flat punctured torus is one member of a cylinder of parallel geodesics, so "the taut representative" is not
   unique for them; the guarded reduction leaves V3 wherever the decimated transported curve shrinks to (shift 0.014 wa between layouts),
   and the cut words change when V3 is on the other side of a cut. The PREREG82 statement that the taut representative is determined by
   the isotopy class is true only for pegged curves. This is a pipeline discontinuity, not motion of the geometry.
3. Step 1b (dump82b.txt): with the core placement the generators move continuously on the whole ladder 0.01-0.02 (worst ratio 0.075;
   0.025 for the step 0.01 -> 0.015). PREDICTION82 frozen.
4. Run 1 crashed (run1_crash.log) at t = -0.01: exact cores of V3, V5, V6 (V6 peg-free before the wall) meet in triple points.
   AMEND2 (generic offset 0.03 (q + 1) of the free width) before any criterion was evaluated.
5. Run 2 (check_run2.log): PW0 PASS (Study 81 criteria at the new held-out t = 0.025: fit 3.5e-8, dims, unique bijection, nullity 1,
   47 products both ways); PW1 PASS (Study 75 configuration -> ladder -> 0.025: unambiguous, worst ratio 0.416; sigma'' o cont = sigma76,
   signs up to one per object: the natural transformation on the unchanged objects); PW2-strict PASS (frozen Study 81 comparison at
   t = 0.01 carried to 0.025 without re-solving and without sign freedom; all 47 products both ways); PW3 PASS (both wrong carried maps
   fail). Thinnest margin: the same-t step frozen configuration -> core configuration at t = 0.01, pair (V6', V5), ratio 0.907 (< 1).
   POST HOC (diag/diag_isotopy82.py, not a check): subdividing that isotopy into 8 steps gives worst ratio 0.115 and the composite map
   equals the direct match on all pairs.
