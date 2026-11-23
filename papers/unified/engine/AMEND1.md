# Study 82 AMEND1 (2026-11-21; after step 1, before PREDICTION82 and before any run at t = 0.025)
Canonical representative for peg-free curves: a taut curve without pegs (a closed straight geodesic of direction lam) is moved, transversally,
to the core of its cylinder: with u = lam/|lam|, transverse coordinate f(z) = Im(conj(u) z), transverse period h = |Im(conj(wa) wb)|/|lam|
and puncture offsets s_k = (f(p_k) - f(v)) mod h in (0, h), the curve is translated by i u (s_min + s_max - h)/2, i.e. to the midpoint
between the nearest punctures on either side. Pegged curves are unchanged (their geodesic is unique). The core is a continuous function of
the puncture positions. Implemented in wall82.py as post-processing of the Study 81 pipeline output (wall81_bside.py unchanged); the
class word of every moved curve must be unchanged by the move (checked at every point). Step 1b (EXPLORATORY) repeats the step-1 diagnosis
with the core placement before PREDICTION82 is frozen.
