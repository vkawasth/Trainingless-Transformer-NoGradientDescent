# Session inventory

38 scripts in `scripts/`, 26 result files in `results/`. Every script applies the
eigsh determinism patch with asserted anchors and runs 2-3 seeds unless noted.

## Data that exists

| result | script | what it settles |
|---|---|---|
| res_quant2 | quant2.py | 8 bits free, 4 bits 1.8%, monotone. 4-bit-beats-uncompressed retracted |
| res_optstate2/4 | optstate2.py, optstate4.py | RMSProp state reduction is a step-norm artefact |
| res_exclusion | exclusion.py | row/col asymmetry is entirely te.weight, i.e. token frequency |
| res_floor (in exclusion) | floor.py | one scalar per row/col ~2%, per matrix 31% |
| res_bT | bT.py | precision vs resolution: FF needs O(d) values at 2-4 bits |
| res_rho / res_fisherrow | rhosweep.py, fisherrow.py | Fisher row structure is a marginal; shuffled null gives 0.608 of 0.650 |
| res_tauF | tauF.py | Fisher sheet half-life 2-4 steps; information, not state |
| res_arity_seeds | arity_seeds.py | tau_2 < tau_3 = tau_5; tau_7 absent in 3/5 seeds |
| res_ablorder | ablorder.py | attention carries high orders, FF load-bearing at all orders |
| res_chain | chain.py | PR(q) 32.7 -> 1.8, PR(v) flat, attention entropy unchanged |
| res_headabl | headabl.py | heads functionally redundant from step 0; potential -> realised alignment |
| (embnull) | embnull.py | step-0 confinement is Phase 1's rank-24 embedding, not the corpus |
| res_randspec | randspec.py | the carrier alignment survives a random embedding: created by training |
| res_k90roles / k90w8 / k90fisher | k90*.py | step-40 contraction real; step-104 EMB event is an l2 artefact |
| res_strand | strand.py | parameter-space subspace tracking saturated: no dynamic range |
| res_clusters | clusters.py | cluster sizes PR(O1) 434 < PR(O2) 947 < PR(O3) 1745; overlap FALLS |
| res_descend | descend.py | directed ascent destroys, matched noise does not; PR(q) unmoved |
| res_selective2 | selective2.py | O3 separable under equalised |B| and ||v|| |
| res_bucketgeom | bucketgeom.py | bucket gradients near-orthogonal; filtration retracted |
| res_unlearn | unlearn.py | U3(C3) lands 4.3x closer to C2 than to C3 |

## Incomplete

- **relearn.py** — the formation/pullback cycle. Still running at session end, no
  output. `ascend()` calls `obs()` every step (3 bucket probes + 6 eval batches);
  poll every 5 steps instead. Also add a third arm matched to U3(C3) on *val*
  rather than step index, or "relearns faster" confounds substrate with fitness.
- **morphism.py** — 7 objects x 2 seeds stalled. Cut to 1 seed.
- **lrsweep.py** — DATA EXISTS in the log but the summary print crashed on a
  stale key. eta* = 2x, giving 0.124 vs 0.265. Everything else in Phase 36/37
  is measured at half the optimal rate.

## The one blocking fact

AdamW at 2x LR reaches 0.124 against 0.265 at the rate used throughout. The
2% margins between state-sharing arms have not been rechecked at eta*.
