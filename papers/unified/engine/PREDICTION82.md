# Study 82 registered prediction (frozen 2026-11-21 after step 1/1b (exploratory) and AMEND1, before any run at t = 0.025)
Pipeline: wall81_bside.py (unchanged) + wall82.py (AMEND1 core placement); comparison iso81.py; check check_path82.py (SHA-256 below).
Continuation rule (from step 1b: worst ratio 0.075 on the dense ladder and 0.025 for the step 0.01 -> 0.015): bijective nearest
torus position modulo the lattice with largest displacement < half the smallest separation, applied between consecutive ladder points;
no insertions. The same rule identifies the Study 75 configuration with the core configuration at t = -0.1 and the frozen Study 81
configuration at t = 0.01 with the core configuration at t = 0.01 (no fixed tolerance; Study 81's MW6 error is not repeated).
Ladders: PW1 -0.1, -0.07, -0.04, -0.01, 0.005, 0.02, 0.025; PW2 0.01, 0.015, 0.02, 0.025. Held-out: everything at t = 0.025.
  PW0 PASS iff at t = 0.025 the Study 81 criteria hold: order 1,2,3,4,6',5, class 6 = (3,-2) with fit < 1e-6 and the other candidates
      worse by > 1e3, mu1 = 0, dims = C'', one degree per pair, unique-up-to-sign bijection on U exists, mutation condition nullity 1,
      invertible maps, all products exact both ways.
  PW1 PASS iff the PW1 ladder matches are unambiguous and, at t = 0.025, sigma'' o cont = sigma76 on every generator of the unchanged
      objects and eps'' o cont = c_i c_j eps76 for one sign c_i per object (natural transformation Phi'' ~ Phi76 o cont^-1).
  PW2-strict PASS iff the PW2 ladder matches are unambiguous and the frozen Study 81 comparison at t = 0.01, carried, satisfies all
      products of C'' exactly in both directions at t = 0.025 (no re-solve, no sign freedom).
  PW2 PASS iff PW2-strict passes, or the carried maps match every product support and the signs agree up to B-side generator signs
      (star-placement gauge), reported with the strict result.
  PW3 PASS iff PW2-strict passes and both controls fail the strict test: (a) the carried bijection of (V3, V6') cyclically permuted;
      (b) two rows of the carried H_3 swapped. If PW2-strict fails, PW3 is reported as not informative (FAIL).
Failures are recorded and kept.
