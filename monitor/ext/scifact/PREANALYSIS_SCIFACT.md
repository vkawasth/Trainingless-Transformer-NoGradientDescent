# Pre-analysis plan: deciding SciFact claims from dependent noisy verifiers

**Fixed on 2026-10-11 before any model was fitted.** No dev label has been read. Seed 20261013.

**Data.** SciFact (Wadden et al. 2020), release `data.tar.gz`, supplied by the user. It is kept outside the monitor folder at `/home/claude/monitor_data_sf` and is not packaged (licence CC BY-NC 2.0).

**Already known before this plan:** train statistics only.
- 919 claim–abstract pairs: 370 SUPPORT, 194 CONTRADICT, 355 NEI.
- 77 claims cite several abstracts, and none of them mixes SUPPORT and CONTRADICT.

## Setting
Oracle abstracts. The unit is a (claim, cited abstract) pair. Its gold label is the evidence label when the abstract is in `evidence`, and NEI otherwise. Train pairs are used for fitting and channel estimation; dev pairs (claims_dev) for evaluation. The test split is unlabelled and unused.

## Sources: three weak verifiers (classical, trained on train only)
- **Sentence selection.** TF-IDF is fitted on the corpus text with no labels. The two abstract sentences with the highest cosine to the claim are selected.
- **V1, sentence cues.** Multinomial logistic regression on:
  - cosine statistics of the top two sentences;
  - counts of negation, increase and decrease cues in the claim and in those sentences;
  - their products (direction mismatch, negation-parity mismatch);
  - number overlap.
- **V2, lexical.** Logistic regression on word 1–2-gram TF-IDF of the text "claim [SEP] top-two sentences".
- **V3, title.** Logistic regression on claim–title cosine and the cue counts of claim and title.

## Channels
5-fold cross-validated predictions on train, with folds split by claim, give:
- each verifier's confusion matrix M_v[y, o] = P(output o | true label y);
- the joint channel P(o1, o2, o3 | y), over 27 cells per label.

Bands come from `monitor.bands` with n = the row count. The prior is the train label frequency, with its band (n = 919). For dev, each verifier is refitted on all of train.

## Decision rules on dev pairs
| | rule |
|---|---|
| R1 | best single verifier by train cross-validated accuracy (argmax) |
| R2 | majority vote; ties go to R1 |
| R3 | naive Bayes with point channels: independence and no uncertainty |
| R4 | credal, assuming independence: likelihood bounds are products of the per-verifier band ends |
| R5 | credal, using the empirical joint channel: likelihood bounds are the band ends of the observed cell (o1, o2, o3) in row y |
| R6 | credal, with no dependence assumption: Fréchet bounds, L_lo(y) = max(0, Σ_v M_lo[y, o_v] − 2) and L_hi(y) = min_v M_hi[y, o_v] |

**How a credal rule decides.** It asserts y when, for every y′ ≠ y, prior_lo(y)·L_lo(y) ≥ κ·prior_hi(y′)·L_hi(y′), with κ = 1.1. Otherwise it hedges, returning the labels that no other label dominates.
- **Strength:** κ* = min over y′ of the ratio, for the label with the best worst case.
- **Secondary:** also require an absolute floor, a lower posterior of at least τ = 0.5.

## Hypotheses (each reported as held or not)
Intervals use a bootstrap clustered by claim (2,000 draws, 95%).

1. **H1, selective accuracy.** R5's accuracy on the pairs where it acts exceeds R3's accuracy on all pairs: the interval of the difference is above 0. The same is reported for R4 and R6, with coverage.
2. **H2, independence buys wrong decisions.** Accuracy on pairs where R4 acts but R5 does not is lower than R5's accuracy where it acts: the interval of R5-acted minus R4-only is above 0. Fewer than 20 R4-only pairs means underpowered.
3. **H3, the margin ranks correctness.** The AUC of log κ* (R5) for correct against incorrect best-worst-case labels, over all dev pairs, is above 0.5, with the interval excluding 0.5.
4. **H4, geometry beyond ordinary confidence.** The area under the risk–coverage curve when pairs are ranked by R5's κ* is smaller than when ranked by R3's maximum posterior. The interval of (AURC_R3 − AURC_R5) must be above 0.

## Descriptive
- Accuracy and macro-F1 of R1–R3.
- Coverage and accuracy of R4–R6 with and without the floor.
- On dev claims with several abstracts: the count of claims where R5 makes a decided SUPPORT on one abstract and a decided CONTRADICT on another. This is a gluing conflict (an evidence obstruction); gold has none on train.
- For R4: which verifier is the weakest link (smallest log-likelihood-ratio contribution) among correct and incorrect acted pairs.
- The share of dev pairs whose observed joint-channel cell has fewer than 5 train counts in the asserted row (thin support).

## Order
1. Freeze this plan.
2. Write `scifact_verify.py` with planted checks of the decision rules on synthetic channels.
3. Run once.
4. Freeze the results.
