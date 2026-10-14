# Amendment 1 to PREREG_hallucination.md

Frozen 2026-10-13, after receiving the 2,000-item sample file but before computing any statistic on items 200-1999.
Items 0-199 were analysed exploratorily (explore_followup.py); those analyses motivated this amendment and are disclosed.

## Study 2 (HaluEval QA)
Sample: items 0-1999, model llama3.1 (8B) via Ollama, K = 5, temperature 1.0, system line "Answer with a short phrase only".
Primary analyses on items 200-1999 (the confirmatory set, n = 1800); the same statistics on all 2000 are reported as secondary.
1. AUROC of the pre-registered detector (containment judge), with 95% bootstrap CI over items (B = 2000, seed 20261013).
2. Length baseline: AUROC of answer length (characters) alone.
3. Length-adjusted AUROC: right and hallucinated answers pooled and cut into length quintiles; AUROC computed from
   right-vs-hallucinated comparisons within the same quintile only (pooled over quintiles), bootstrap CI over items.
4. Logistic regression label ~ log(length) + detector score; likelihood-ratio test of the detector term.
5. Relaxed judge (deterministic, no model): an answer matches a sample if, after lower-casing, removing punctuation and the
   stopwords {the, a, an, of, in, on, at, and, or, to, is, was, were, are, by, for, with, from, as}, the answer's content tokens
   are a subset of the sample's or vice versa, or their Jaccard similarity is >= 0.5. Statistics 1-4 are repeated with it.
6. Within-sample Beck-Chevalley test: Y_k = 1 if sample k does not match the right answer (relaxed judge). Windows
   sigma(Y1,Y2), sigma(Y2,Y3). Statistic cbar = mass-weighted |phi(Y1,Y3 | centre)| over strata with non-degenerate margins;
   exact test by permuting Y3 within strata of the centre (B = 2000). Centre = Y2 (primary); centre = (Y2, D) with
   D = Y4 + Y5 (named secondary). Hypothesis H2': the first test rejects at 0.05; the second is reported without a hypothesis.
Decision rule for H1'': the detector adds information beyond length if the LR test of statement 4 has p < 0.05 (containment judge);
the relaxed judge is reported alongside.

## Study 1 (LLM-drift), named re-analyses
The Beck-Chevalley statistic is replaced by cbar with the within-stratum permutation test (same windows). Named common-cause
analyses: centre (Y4J, Y35M); for primes also centre (Y4J, label) and (Y4J, last digit of the number). Reported for all five
tasks, no new hypothesis; these are labelled as re-analyses of data already examined.

## Units for P3
P3 remains exploratory; a confirmatory version needs >= 15 independent task-model units and is deferred.
