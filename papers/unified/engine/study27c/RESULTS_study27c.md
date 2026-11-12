# Study 27c — results (run once, 2026-11-12 UTC; specification SHA-256 21632b08..., frozen before the run)
Checks: engine/check_study27c.py ZC1-ZC8, 8/8. All six predictions held. Calibration seed 27201, test seed 27202, 1000 runs per scenario.

## Calibration (calibration set only)
z = 2.00 in both configurations (grid floor; honest rate 0.016 A, 0.033 B). Value of information: the geometric guard changed no loss and
was switched off (as the attribution predicted: no degeneration); the contextual and braid components each lowered the loss and were kept.

## Test
| | honest | shock | relay | timely cascade | lagging cascade | drift | balanced accuracy |
|---|---|---|---|---|---|---|---|
| A (anchors 3000) | 0.026 | 0.038 | 0.997 | 0.041 | 0.998 | 0.999 | 0.978 |
| B (+ inspection 120k) | 0.042 | 0.053 | 1.000 | 0.052 | 1.000 | 1.000 | 0.974 |
| 27b, unfiltered (A / B) | 0.352 / 0.392 | 0.393 / 0.447 | 1.000 | 0.199 / 0.251 | 1.000 | 1.000 | 0.901 / 0.875 |
P1 honest <= 0.065 HELD; P2 shock <= 0.10 HELD; P3 B >= 0.90 HELD (0.974); P4 B >= A - 0.01 HELD; P5 lagging cascade >= 0.90 HELD
(the confound remains); P6 no guard activation HELD.

## Ablation (test set, z fixed)
| switched | honest A / B | shock A / B | balanced A / B |
|---|---|---|---|
| none (final filter) | 0.026 / 0.042 | 0.038 / 0.053 | 0.978 / 0.974 |
| contextual off | 0.068 / 0.102 | 0.102 / 0.145 | 0.979 / 0.974 |
| braid projection off | 0.039 / 0.062 | 0.055 / 0.083 | 0.961 / 0.961 |
| geometric guard on | unchanged | unchanged | unchanged |
The contextual component (anchor departures read off the g-blind sources) removes the most false relays; the braid projection (net side
change, over/under discarded) removes the twists and also raises sensitivity.

## Explanation map
- Relay: the source that changed side was S3 alone in 99.7% (A) and 100% (B); the change followed its activation along the chain in
  99.9-100%.
- Drift: S3 alone in 99.9-100%, but after an activation in only 65%: the trace separates a relayed departure from a lone one.
- Timely cascade: S3 changed side in 4-5% of runs, always after activation. Lagging cascade: S3 in 99.8-100%, after activation
  in 99.7-99.8%, the same as a relay (the confound).
- Anchor departures (an anchor crossing a g-blind source) were attributed to the anchor: 40-96 per 1000 runs; none in the timely cascade.
