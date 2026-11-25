# Study 92 calibration notes (exploratory, calibration seeds only; nothing here is a check) — 2026-11-24

## F (3x3 strand walk), seed 9201 (lib92.gen33, base point lib92.base_point33)
- Path 0, L = 20 (walk33_cal.py): 18 solutions at the start (residual 1e-17); walk complete in 954 s with 18 detours.
  Events: 1 collision, 7 caustics, no multiple-caustic substeps (analyse_fine had nothing to refine). Only 4-6 of the 18 critical
  values are real at any time, so real-value events are SPARSE on 3x3 (cf. 541-1168 collisions per 3x4 walk).
- Ties crossed: 3 (f0 = f2 at t = 0.376 and 8.828; g0 = g1 at t = 0.403). One collision, on the g-tie; the f-ties carry none.
  Escape: at t = 8.828 three strands reach |log x| ~ 53 next to the tie (Study 79's "3 lost"); at t = 0.376 none (max 0.4). So escape
  happens at some ties and not others; a prediction about escapes is not justified yet (F4 stays RECORD).
- Consequence for registration: per-path counts of real collisions are too small for F1's "first 5 isolated collisions per path".
  Options: more paths, larger L, or certify all events on several paths. Path 1 at L = 30 is being walked for a second look.

- Path 1, L = 30: 18 solutions at the start; walk complete (1178 s, 18 detours; division-by-zero warnings from rejected trial steps).
  13 ties crossed; 8 collisions, ALL on ties (one each on 8 ties; 5 ties carry none); 5 caustics; 4-6 real values.
  Across both calibration walks: 9 collisions, all 9 at ties, none off them. So 3x3 resembles 2x3 (all collisions at ties, Studies
  83/85) and not 3x4 (most collisions off the ties, Study 91). A registrable prediction: on 3x3 held-out paths every real collision lies
  on a margin tie (as in 2x3). With ~1-8 collisions per path, F needs >= 6 paths and both L to have usable counts.

## E (2x3 representation), calibration line seed 9211 (e1_cal.py, e1b_cal.py)
- Line in LOG-coefficient space la(s) = la0 + s d. M(s) = prod_{a<b} (v_a - v_b)^2 is single-valued; argument principle on a polar grid
  (R = 3, 180 x 120 cells, 14 s) found 4 bifurcation points: two of order 3 (caustics) and two of order 2 (Maxwell points).
- Loop braids (ray + circle rho = 0.2, projection angle 0.1): caustics sigma_3^3 and sigma_3 sigma_4^3 sigma_3^-1 (conjugates of sigma^3,
  exponent sum 3, a transposition); one Maxwell point a pure braid of exponent sum 2 (length 14). One loop needs finer steps
  (non-adjacent crossing detected): steps must be adaptive.
- Design consequence: a log-coefficient line is not algebraic, so Zariski-Lefschetz does not apply and M(s) is an exponential
  polynomial (zeros possibly unbounded in number). For E1 the line must be AFFINE in the coefficients a (algebraic): then the bifurcation
  set meets it in finitely many points, including the 6 large-radius points a_k(s) = 0, and the argument principle on a large circle
  counts them. To be recalibrated on that line.

- Affine line a(s) = a0 + s d (e1aff_cal.py, seed 9211, R = 6): the six large-radius points a_k(s) = 0 lie at |s| ~ 1-4.4; rays
  through or near them break the ray continuation (critical points escape, log branch), and the argument principle then returns
  dozens of spurious +-1 cells beyond them. The affine line needs a different solver: all six critical points recomputed at each grid
  point (not ray continuation), or cell-by-cell continuation that stops at the a_k = 0 punctures. Not yet built.

## G (2x3 fibre level): largely already done
- Studies 45, 49, 67, 85 (BT checks: K_0 lift exceptional at every step, PL classes = K shadows, collisions orthogonal) and Study 90
  (RP1: PL agreement of transported curves at event brackets, 5/6) cover G1-G3 and G4 on held-out paths. G4/G5 as invariants of the
  Coxeter element are automatic once the Gram transforms by the braid action. Proposal: drop G from Study 92 and cite those studies.
