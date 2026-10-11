# Pre-analysis plan: the decision layer on SciFact

**Fixed on 2026-10-11, before any model was fitted and before the dev split was opened beyond its field names and size.** Seed 20261013.

**Data.** The SciFact release supplied by the user: `data.tar.gz`, SHA-256 11c62128…76be. It contains 5,183 abstracts, 809 train claims, 300 dev claims and 300 test claims (test unlabelled, not used).

**What was looked at, on train only:**
- the format;
- the label counts per claim–abstract pair: SUPPORT 370, CONTRADICT 194, NEI 304;
- cited abstracts per claim: one for 732 claims;
- abstracts cited by the same claim never disagree.

**Setting.** Oracle abstracts. Units are (claim, cited abstract) pairs. The truth is the gold label (SUPPORT, CONTRADICT or NEI); a pair with no evidence entry is NEI.

## Sources (three verifiers, each a different view)
Each is a 3-class logistic regression (standardised, C = 1) at pair level:
- **A, lexical.** TF-IDF (fitted on the corpus) cosine between the claim and each sentence: max, mean, top-2 mean, and the relative position of the best sentence. Also title cosine, word-overlap counts and abstract length.
- **B, polarity.** Counts of negation, decrease and increase cues in the claim and in its two best-matching sentences, their products with the best cosine, and indicators of cue mismatch.
- **C, bag of words.** TF-IDF of the claim (vocabulary of size 2^14, hashed) concatenated with the TF-IDF of its two best-matching sentences in a separate hashed space.

## Split and reliability
- Train pairs are split at random by claim, 75% proper-train and 25% calibration.
- Each source is fitted on proper-train only.
- **Reliability.** r_k is the α-quantile of p_k(true label) over calibration pairs, with α = 0.05/3 (Bonferroni over the three sources). This is the finite-sample conformal quantile: the ⌊(n_cal+1)·α⌋-th smallest value.
- Sources are then **not** refitted.

## Decision layer
For each dev pair, source k reports p_k, treated as an exact law (counts p_k × 10^6). The credal set is Q = {q : r_k·q ≤ p_k for all k} (`decide.py`, reliability r_k, no context maps). Actions are "assert SUPPORT", "assert CONTRADICT" and "assert NEI".
- The layer **acts** on ℓ when ℓ is κ-supported over both other labels, with κ = 1.1.
- Otherwise it **hedges**, with the set of actions not dominated.
- Conflicting sources (Q empty) are repaired as in `decide.py`, by excluding the least reliable source, and recorded as an evidence obstruction.
- **Policy variant.** A floor τ = 0.5 on the asserted label: acting additionally needs min over Q of q_ℓ ≥ 0.5.

## Predictions (each reported as held or not)
- **P1, the guarantee.** If every source covers the truth (p_k(y) ≥ r_k for all k), then the vertex δ_y lies in Q, and acting on ℓ ≠ y is impossible at κ ≥ 1. So P(act and wrong) ≤ P(some source misses) ≤ 0.05.
  - Holds if the 95% Clopper–Pearson lower bound of the dev act-and-wrong rate is below 0.05.
  - Also reported: the dev rate at which some source misses.
- **P2, usefulness.** The layer acts on at least 25% of dev pairs.
- **P3, several sources beat one.** Compare the three-source layer with the best single source (best by calibration accuracy) at the same guarantee, α = 0.05 alone. Holds if the three-source layer acts on at least as many dev pairs.
- **P4, selective accuracy against confidence thresholding.** Take the best single source's argmax and abstain on its least confident pairs, so that it covers the same number of dev pairs as the layer. Holds if the layer's accuracy minus the baseline's accuracy has a 95% bootstrap CI (2,000 resamples of claims) whose lower bound is above −0.05.
- **P5, hedges are hard pairs.** The best single source's argmax accuracy is higher on pairs where the layer acts than on pairs where it hedges. Holds if the difference has a bootstrap CI lower bound above 0.

**Descriptive, with no prediction:**
- the share of pairs with an evidence obstruction;
- the floor variant's coverage and accuracy;
- the share of acted pairs in which each source is the weakest link, and their median reliability slack;
- accuracy by gold label.

## Order
1. Freeze this plan.
2. Write `scifact_decide.py`.
3. Run it once.
4. Freeze the results.

The dev split is used only in step 3.

## Amendment 1 (2026-10-11, after a dry run on proper-train pairs only, before the dev split was used)
**What the dry run showed.** The calibrated reliabilities are 0.056 (A), 0.039 (B) and 0.000 (C): the classical verifiers are weak. At α = 0.05/3 the credal set is therefore almost the whole simplex, so P2 is expected to fail.

**What is unchanged.** The primary predictions P1–P5.

**What is added (descriptive only).** A risk–coverage curve over α_total ∈ {0.05, 0.10, 0.20, 0.30}, with α_k = α_total/3 per source. For each level it reports:
- the share of pairs acted on;
- the act-and-wrong rate;
- the guarantee α_total;
- the same quantities for the best single source at level α_total.
