# Study 18 results: confounds removed and consistency training

Pre-registration: `PREREG_study18.md`, frozen 2026-11-06 21:00 PT (`FROZEN_PREREG_study18.txt`; `sha256sum -c` gives OK for both
files on 2026-11-07, before scoring). Training: 168/168 models, in three resumed 2-hour segments per half (planned timeouts, exit 124,
restarted with `--resume`; no model was retrained and no score was looked at during training). Analysis: `kit_study18.py score`, run
once -> `study18_final.json`. Figure: `fig_study18.py` (plotting only) -> `fig_study18.pdf`.

## Theorem checks (reported first)
| check | result |
|---|---|
| T1-T4 (Pythagorean split, IPF projection, rigid fibre at F0, fibre dimensions) | all hold |
| `fibre_dims` | F0: 0, F1: 1, F2: 2, as designed |
| F0 TV(P*, P_true) | at most 1.3e-14 (threshold 1e-6) |
| Pythagorean relative error | at most about 1e-14 |

The kit is computing what the theorems say. Every failure below is a fact about the learned models, not about the kit.

## Pre-registered hypotheses
| | prediction | result | verdict |
|---|---|---|---|
| H7 | consistent training: TV(P_true, Q) ordered F0 < F1 < F2, both CIs above 0 | means 0.196 < 0.227 < 0.257, in the predicted order; CI(F1-F0) [-0.062, 0.135], CI(F2-F1) [-0.092, 0.141] | **not supported** (order right, CIs include 0) |
| H8 | on F0, consistency lowers the off-fibre term (paired CI above 0) | plain - consistent = 0.218 nats, CI [0.119, 0.310] | **supported** |
| H9 | weight decay 0 lowers the off-fibre term on S faces | wd0.01 - wd0 = -0.017, CI [-0.041, 0.011] | **not supported** (no effect) |
| H10 | consistent models: defect minus noise floor, CI includes 0 | excess CI [0.0015, 0.0180], mean excess about 0.010 TV | **not supported; alternative reported**: a small defect beyond fitting noise |
| H11 | G1 plain: ι < 0.5 in at least 75% of models | 24/24 (100%), mean ι = 0.107 | **supported** (replicates Study 17's H1 at 8000 steps) |

## What it means
1. **Rigidity acts only once the model is a single probabilistic object (H8).** On the Clauser-Horne facet the fibre is a point, so
   any model that fits the marginals coherently has to be right. Plain models fit the marginals almost perfectly but are not
   coherent: order drift 0.31, so they are many joints at once. Consistency training brings drift to 0.011 and cuts TV to the true
   joint from 0.346 to 0.196. That is the sense in which Cor. 73 (rigidity) is a statement about coherent learners.
2. **The ordering by fibre dimension appears with consistency training and reverses without it.** Consistent: F0 < F1 < F2 in the
   means (H7's direction, not significant at 9 models per family). Plain: F0 0.346 > F1 0.301 > F2 0.238. *Post hoc* (not
   pre-registered): plain F0 - F2 = 0.108, CI [0.032, 0.189]. Without coherence, the rigid boundary is the hardest place to be;
   with it, it is the easiest.
3. **Coherence costs fit, and the cost lands where the fibre is free (E7, E8).** Consistency removes failed combination
   (summed TV gap 0.59 -> 0.03 on F0, 0.44 -> 0.02 on F1, 0.30 -> 0.02 on F2) but the direct pair predictions fit the training
   marginals less well (summed KL 0.02 -> 0.07 on F0, about 0.0003 -> 0.16-0.17 on F1 and F2). On F1 and F2 the in-fibre term
   rises (0.036 -> 0.088, 0.034 -> 0.097). *Post hoc*: on F2 consistent is slightly worse than plain in TV (paired difference
   -0.018, CI [-0.035, -0.005]). A coherent learner on a free fibre still has to choose a section, and it does not choose the
   true one.
4. **Learned composition is not system independent, but the excess is small (H10).** The tensor-square defect of consistent models
   exceeds the product model's seed-to-seed floor by about 0.01 TV (CI [0.0015, 0.018]). In two of six pairs (3 and 5) the defect
   is at or below the floor. Study 17's H5 failure therefore was not only noise, but most of it was.
5. **Learning picks a maximum-entropy-like section (H11),** now in all 24 models, with mean ι 0.107 (Study 17: 94%).
6. **Models report less contextuality than their data (E6).** The cohort worlds G2 have contextual fraction 0.33-0.49 in their
   training pair tables. The models' direct pair predictions have 0.16-0.24, about half. Each direct prediction is a separate
   output, so nothing forces this. The learned predictions are pulled toward noncontextual tables. S faces are noncontextual
   (CF 0), and the models stay near 0 (at most 0.005).

## Disclosures
- E7 was pre-registered as a split of the off-fibre term. The kit, frozen before training, measures the two parts as separate
  quantities in different units: fit in summed KL, combination gap in summed TV. They are reported side by side, not as a sum.
- The paired contrasts marked *post hoc* above were computed after the single scoring run, using the same bootstrap (2000
  resamples, seed 20261108). They are labelled and are not tests.
- G2 has 3 worlds, so E6's G2 points form 3 clusters of 3 seeds.

## Files
`runs18.jsonl` (concatenation of `runs18_even.jsonl` and `runs18_odd.jsonl`), `study18_final.json`, `fig_study18.py/.pdf/.png`,
`train18_even.log`, `train18_odd.log`. Model weights in `weights18/` (not packaged).
