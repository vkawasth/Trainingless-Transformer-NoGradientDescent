# Study 82: the wall-crossing comparison along the path (registered 2026-11-21, after Study 81 was written up and its record frozen in
# FROZEN.txt; before any Study 82 code ran; check prefix PW, checked unused)
Question. Study 81 established, at the held-out point t = 0.020, an exact isomorphism Phi'' between the post-wall thimble algebra and the
algebra of C''. Open: that Phi'' is the comparison transported along the path, i.e. (i) on the unchanged objects Phi'' o cont = Phi76 up to
one scalar per object (natural transformation), and (ii) the exploratory comparison at t = 0.010, carried by continuation without solving
again, is still an isomorphism further along (MW9). Study 81 failed both as registered: MW6 at a mis-specified reproduction tolerance, MW9
at an ambiguous continuation between t = 0.015 and 0.01625, where the representatives change their cut-system words.
Representatives. The taut (guarded geodesic) representative of a curve in the flat punctured torus is determined by its isotopy class,
so it is the fixed-representative choice; it moves continuously with the punctures as long as no geodesic segment changes the side of a
peg it wraps. Study 81 did not establish whether the jump between 0.015 and 0.01625 is fast continuous motion or a discontinuity of the
pipeline (decimation / guard). Step 1 below decides this before anything is registered for the held-out path.
Frozen now.
  - Held-out end point t = +0.025 (unseen; Study 72: the next Stokes event (6, 4) lies after t = 0.03). No Study 82 run reaches it before
    PREDICTION82 is frozen with SHA-256.
  - The Study 81 comparison at t = 0.010 (study81_wall/frozen_phi_t0.01.pkl, hashed) and the Study 76 comparison (signs76.json) are the
    objects carried; neither is re-solved.
  - Exploratory interval: t in [0.015, 0.0175] (already seen in Study 81), dense ladder, diagnosis only.
Step 1 (EXPLORATORY, no pass/fail): ladder of step 2.5e-4 on [0.015, 0.0175] with the Study 81 pipeline unchanged; per step record the
  generator displacements, the taut vertices and pegs, and the class words; decide whether generator positions are continuous (and with
  what step) or whether the representative jumps.
Step 2: PREDICTION82 frozen with the continuation rule (step size and matching rule from step 1, or a pipeline fix recorded as an
  amendment if step 1 finds a pipeline discontinuity), then the registered checks:
  PW1 (natural transformation) -0.1 -> 0.025 on the unchanged objects, starting from the Study 75 configuration matched to the ladder at
      -0.1 by the ladder's own rule (the Study 81 MW6 tolerance error is not repeated);
  PW2 (carried comparison) 0.010 -> 0.025, all 15 pairs, the frozen Phi'' carried without re-solving, checked on all products both ways;
  PW3 controls (frozen with PREDICTION82).
Failures are recorded and kept; Study 81's registered outcomes are not changed.
