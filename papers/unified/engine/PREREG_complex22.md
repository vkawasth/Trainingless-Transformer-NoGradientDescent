# Complex 2x2 margins (prop:complex22): identity checks of a proved proposition. Written and frozen 2026-11-15 before the script ran.
Setting: 2x2 table, cells x11 = t, x12 = r1 - t, x21 = c1 - t, x22 = c2 - r1 + t; F_L = sum exp(-L x) (thm:selection, weights 1).
With w = exp(-L t): F_L = W(w) = A w + B/w, A = 1 + exp(L z), z = r1 - c2, B = exp(-L r1) + exp(-L c1).
Proved claims: two critical points w = +-sqrt(B/A), critical values +-2 sqrt(AB), Hessian 2B/w^3; they degenerate (escape to w = infinity or 0,
critical values colliding at 0) iff A = 0 or B = 0, i.e. iff r1 - c2 or r1 - c1 equals i pi (2k+1)/L. Around one such point the two critical
points swap; real loops return. Tropical limit of the positive critical point = leximin t = (min(r1,c1) + max(0, r1 - c2))/2, with
|t_L - t_lex| <= log(2)/L. The A-part of t_L is (1/2L) log(1 + e^{Lz}): its second derivative in z peaks at z = 0 with value L/8 and full width
at half maximum 4 arccosh(sqrt 2)/L; its Taylor series at z = 0 has radius pi/L. Fibre W = s: two points, swapped around each critical value;
around both critical values the monodromy is the identity (boundary word tau tau = id).
Checks (seed 3802, 200 random margin sets with entries in (0.05, 0.95) summing correctly; L in {20, 80, 320}):
  CZ1 IDENTITY  F_L(t) = A w + B/w at complex t and margins (relative error < 1e-12); W'(+-sqrt(B/A)) = 0 (< 1e-10 relative).
  CZ2 IDENTITY  zeros: |A| < 1e-9 at z = i pi (2k+1)/L and |B| < 1e-9 at r1 - c1 = i pi (2k+1)/L, k = -2..2.
  CZ3 IDENTITY  continuation of the two critical points around a circle of radius pi/(2L) about z0 = i pi/L (2000 steps): they swap; around
               the same circle about z0 + 2 pi/L... no: about a point at distance > 3 pi/(2L) from every zero (z0 + pi/L, i.e. i 2 pi/L): return.
  CZ4 IDENTITY  |t_L - t_lex| <= log(2)/L for all 200 real margin sets and each L.
  CZ5 IDENTITY  second derivative of (1/2L) log A: peak at z = 0 (grid step 1e-4/L... relative 1e-3), value L/8 (relative 1e-6), FWHM
               4 arccosh(sqrt 2)/L (relative 1e-3).
  CZ6 IDENTITY  fibre over s on a circle around one critical value (other outside): the two roots swap; on a circle enclosing both: return.
Output: engine/check_complex22_output.txt.
