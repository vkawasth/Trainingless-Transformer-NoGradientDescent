# Study 27c, step 1 — where the witness-braid noise comes from (exploratory; seed 27101; 600 runs per scenario)

Script: attribution_27c.py (27b's world, generator and witness rules, unchanged). Results: attribution_27c.json. Overlap: OVERLAP_TABLE.md.
No 27b verdict changes. Metric: relay-like rate (Phi >= 1). Five noise sources switched on/off on common random numbers (32 settings);
exact Shapley split; "remove" = all sources on except that one (what a filter removing it alone would leave).

## Degeneration (N6): none in this world
| strand | n | empirical SD of g / delta-method SE | bias / SE | smallest expected cell count | max cell ratio |
|---|---|---|---|---|---|
| S3 | 6000 | ~1.0 | -0.007 | 131 | 21 |
| MRY | 3000 | 1.005 | -0.009 | 90 | 16 |
| MHY | 3000 | 1.014 | -0.023 | 128 | 7.5 |
| I120k | 120000 | 1.011 | +0.001 | 11698 | 5.6 |
Every strand sits in the interior of its polytope under P and Q; the delta method is exact to about 1%, and second-order (Hessian)
bias is below 3% of an SE. Nonlinear degeneration is not a noise source here; the geometric filter has nothing to remove in this world
(it would in worlds with rare cells: smallest expected counts below about 30).

## Attribution of false relays (honest runs)
| config | all on | all off | N1 source | N2 anchor | N3 baseline | N4 looks | N5 y noise |
|---|---|---|---|---|---|---|---|
| A (anchors at 3000) Shapley | 0.325 | 0.000 | 0.051 | 0.127 | 0.055 | 0.045 | 0.045 |
| A remove-one leaves | | | 0.183 | 0.020 | 0.163 | 0.153 | 0.153 |
| B (+ inspection 120k) Shapley | 0.382 | 0.000 | 0.091 | 0.116 | 0.068 | 0.053 | 0.053 |
| B remove-one leaves | | | 0.183 | 0.122 | 0.198 | 0.187 | 0.187 |
Shock runs: same pattern, about 0.06 higher (common-mode movement). Timely cascade (A 0.175, B 0.227): same pattern, smaller.
Relay, drift and lagging cascade: relay-like in every run with all noise off as well (the signal does not depend on noise).

## Reading
1. No source acts alone: with one source on, the rate is 0 to 0.04. The false relays are interactions, so the Shapley split is a
   convention; the remove-one column is what each filter would achieve by itself.
2. Twists: removing looks or removing y-noise leaves exactly the same rate (A 0.153, B 0.187). The extra false relays from repeated
   looks are all made by the second coordinate turning a crossing and its return into a full twist: about half of the honest false
   relays in A. This is the braid filter's share (keep resolved net crossings only).
3. Anchor sampling is the largest single share in A; removing it leaves 0.02. In B, source-3 sampling grows (0.051 -> 0.091): the
   precise inspection strand has a small offset (0.028) against source 3's own noise, which is why B did worse than A in 27b.
4. Baseline (round-0) noise is a real share (0.055-0.068). Since there is no degeneration, the correct remedy is the probability
   filter's variance accounting (differencing doubles the variance of every displacement), not averaging g across rounds.
5. Shock excess over honest (about 0.06): common-mode movement, the contextual filter's share.

## Implications for the 27c filter (to be pre-registered)
- braid: count only net crossings whose over/under is resolved (credal width 0), or drop the second coordinate's noise;
- probability: per-pair offsets from the pair's own Fisher variances, doubled for differencing, with a family-wise level for 12 looks;
- contextual: remove the common-mode displacement explained by a global section before reading crossings;
- geometric: keep the degeneration gauge (expected cell count, condition number) as a guard; it removes nothing here;
- value of information: apply a component only if it changes the expected decision loss.
