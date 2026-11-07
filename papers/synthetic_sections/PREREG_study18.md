# Pre-registration (DRAFT, not frozen): Study 18, confounds removed and consistency training

A follow-up to Study 17 (`RESULTS_study17.md`). Status: DRAFT, 2026-11-06. It becomes binding when the author freezes it: the SHA-256
digests of this file and of the kit `kit_study18.py` (smoke test in `kit18_smoke_output.txt`) are then written to `FROZEN_PREREG_study18.txt`
before any model is trained.

## Disclosure: what is already known, and why this study exists
Study 17 was frozen and analysed once. Two of its failed hypotheses turned out to be confounded, as two labelled post-hoc checks show:
- **H5 (independent systems).** Two seeds of the same model already differ in their learned composite by TV 0.017-0.034 (chains) and
  0.054 (products). That is about the size of the measured defect (median 0.039) and above the pre-registered threshold of 0.0188, so
  the test could not separate a composition rule from fitting noise.
- **H3 (rigidity).** The Clauser-Horne target had 8 zero cells and the structural-zero target 4, so rigidity and sparsity were mixed
  together.

This study removes both confounds and adds an intervention. Its hypotheses are new and are not re-tests of Study 17's data. No Study 18
model has been trained.

## Worlds (the 4-cycle unless stated; exact rational laws; seed 20261108)
- **S, sparsity-matched rigidity.** Three families of faces of D, each with exactly 8 deterministic joint readings (so 8 nonzero cells
  at ε = 0), but with different fibre dimension (Prop. 72 table):
  - F0: a Clauser-Horne facet, fibre dimension 0;
  - F1: a 6-dimensional face, fibre dimension 1;
  - F2: a 5-dimensional face, fibre dimension 2.
  For each family, 3 faces are picked at random from the full face list (seed 20261108), and generic weights in (1/240)ℤ are put on
  their 8 readings. Only ε = 0, the face law itself, is used.
- **G2′.** 3 cohort worlds as in Study 17, with N = 3000, for E6.
- **G1′.** 8 latent-class worlds as in Study 17, with N = 3000, used for H8 and H9.
- **W2′.** 6 chain pairs and their products as in Study 17, with exact marginals, used for H10.

## Models and training
- The architecture and data format are those of Study 17: 2 layers, 4 heads, width 64, feed-forward 128, NoPE, AdamW at 3e-3 with
  cosine decay, batch 256.
- **Training length:** 8000 steps, twice Study 17, to lower fitting noise. (12000 was drafted; a timing test showed 33 ms per step,
  which would exceed 9 hours, so the length was set to 8000 before freezing.)
- **Two training variants**, crossed with every world:
  - *plain*, as Study 17 (weight decay 0.01);
  - *consistent*: the plain loss plus λ_c × the mean squared difference between the log joint probabilities that two random
    variable orders give to the same random full assignment, using 64 assignments per step. λ_c = 1, fixed in advance. This makes the
    model a single probabilistic object, and it uses no information beyond the training marginals.
- **Weight decay contrast** (for H9): a third variant, *plain_wd0*, plain with weight decay 0, on the S faces, to see whether zeros
  are representable.
- **Seeds:** 3 per condition, the pre-specified cut, adopted before freezing. Model weights are saved.
- **Layout** (168 models):

  | family | layout | models |
  |---|---|---|
  | S | 9 faces × 3 variants × 3 seeds | 81 |
  | G1′ | 8 worlds × plain × 3 seeds | 24 |
  | G2′ | 3 worlds × plain × 3 seeds | 9 |
  | W2′ | 6 pairs × 3 parts × consistent × 3 seeds | 54 |

## Metrics
All of Study 17's metrics: in-fibre and off-fibre terms, the section index ι, order drift, the unobserved-pair errors, mismatch and M*,
and the tensor-square defect. In addition:
- **The contextual fraction** CF of the training data, and of each model's implied pair tables (Abramsky, Barbosa and Mansfield 2017):
  CF = 1 − max 1·b subject to M b ≤ v, b ≥ 0. It is computed by linear programming. It is reported next to M*.
- **In-distribution fit:** KL between the training marginals and the model's direct 2-variable predictions, as opposed to the
  4-variable implied joint. Off-fibre error is split into poor fitting (in-distribution) and failed combination (the gap between the
  pair tables implied by the 4-variable joint and the direct 2-variable predictions).
- **Noise floor:** for W2′, the mean TV between the product model's composites over pairs of seeds, per chain pair.

## Theorem checks (as in Study 17)
T1-T4, now including the S families. In particular:
- F0 gives TV(P*, P_true) < 1e-6;
- F1 and F2 give fibres of dimension 1 and 2, confirmed numerically (reported as `fibre_dims`).

## Hypotheses
- **H7 (rigidity at matched sparsity).** With consistent training, the mean TV(P_true, Q) is ordered F0 < F1 < F2. Both differences
  must have 95% bootstrap CIs (over faces × seeds, 2000 resamples) above 0.
- **H8 (consistency training lets rigidity act).** On F0, consistent training lowers the off-fibre term relative to plain training:
  the 95% CI of the paired difference (same face and seed) is above 0.
- **H9 (zeros are representable).** On all S faces, plain models with weight decay 0 have a lower off-fibre term than plain models
  with weight decay 0.01: the 95% CI of the paired difference is above 0.
- **H10 (independent systems against the noise floor).** For consistent models, the per-pair mean tensor-square defect minus the
  per-pair product-model noise floor has a 95% CI (bootstrap over the 6 pairs) that includes 0. That is, no defect beyond fitting noise. The alternative ("learned composition breaks independent
  systems") is reported if the CI lies above 0.
- **H11 (the section learning picks, retested with longer training).** On G1′ with plain training, ι < 0.5 in at least 75% of models,
  replicating Study 17's H1 under the new training length.

## Exploratory
- **E6.** The contextual fraction of each model's direct pair tables against that of its training data, on G2′ and the S faces.
- **E7.** The split of off-fibre error into poor fitting and failed combination, for every family.
- **E8.** Order drift under consistent training: does it fall toward 0, and what happens to in-fibre error when it does?

## Budget
168 models at about 4.5 minutes each (8000 steps), so about 6.5 hours in two parallel processes on this workspace's CPU. Training is
resumable, and a run cut off by a time limit is restarted with `--resume`.

## Reporting
All hypotheses and conditions are reported. The analysis is run once. A failure of T1-T4 is reported first and investigated as a
possible kit error.

## References
- S. Abramsky, R. Soares Barbosa and S. Mansfield, Contextual fraction as a measure of contextuality, Phys. Rev. Lett. 119 (2017)
  050504.
- S. W. Al-Safi and A. J. Short, Simulating all nonsignaling correlations via classical or quantum theory with negative probabilities,
  Phys. Rev. Lett. 111 (2013) 170403.
- Study 17: PREREG_study17.md, RESULTS_study17.md.
