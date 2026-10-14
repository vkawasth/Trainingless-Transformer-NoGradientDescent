# Pre-registration: Study 4 -- does knowing where disagreement sits improve decisions?

Frozen 2026-10-14 before any statistic named below was computed. SHA-256 in FROZEN_PREREG_engine.txt.

## Question
The Posterior Coherence Engine reports not only how much several sources disagree (what averaging keeps) but where the disagreement
sits in a hierarchy (between groups versus within groups). This study tests whether retaining that location improves a decision:
predicting whether the pooled answer is correct. The improvement is to be established here, not assumed.

## Data
LLM-drift answers (Chen, Zaharia and Zou 2024), the five tasks of Study 1 (USMLE, PRIME_FULL, COUNTAHAPPYNUMBER, HOTPOTQA, ARC),
extracted with the unchanged rules in halluc_study/llmdrift_extract.py. Items are kept if all four runs (4M, 4J, 35M, 35J) have a valid
extracted answer and the reference is available: 338, 949, 375, 230 and 466 items (2358 in total; counted before freezing, no outcome
computed). Families: GPT-4 = {4M, 4J}, GPT-3.5 = {35M, 35J}.

## Definitions
- Pooled answer: the plurality answer of the four runs; ties are broken by 4J's answer if it is among the tied answers, otherwise by the
  smallest tied answer in string order.
- Outcome: Y = 1 if the pooled answer equals the reference answer.
- Features, from the engine's three-level split with each run a one-hot posterior and families weighted by size (pce.votes.features):
  v = vote share of the pooled answer; H = entropy of the four-run vote distribution (bits); B = between-family term I(answer; family).
  (The within-family term equals H - B, since the noise term is zero.)
- Baseline (averaging): logistic regression of Y on (v, H). Engine: logistic regression of Y on (v, H, B).
- Both fitted separately within each task, L2 penalty C = 1, out-of-fold probabilities from 5-fold stratified cross-validation,
  shuffle seed 20261014 (pce.selective.cv_logistic).

## Primary analysis
Delta = mean over all 2358 items of [log loss(baseline) - log loss(engine)], in bits, with out-of-fold probabilities.
95% interval: bootstrap over items, stratified by task, resampling the per-item differences (B = 2000, seed 20261014).

H4 (supported) if the interval lies above 0. "No improvement" if it contains 0. "Worse" if it lies below 0.

## Secondary analyses (reported in full)
1. Delta per task with its stratified bootstrap interval, and a one-sided test of Delta > 0 per task (bootstrap p), Holm-adjusted
   across the five tasks.
2. AUROC and AURC (area under the risk-coverage curve) of the out-of-fold probabilities for both models, per task and pooled.
3. Accuracy of the pooled answer per task.

## Exploratory (labelled as such)
- The sign and size of B's coefficient when the engine model is refitted on all items of each task: does disagreement between
  families predict errors more or less than disagreement within families at the same total entropy?
- Share of items whose vote pattern is a 2-2 split along family lines versus across them, and accuracy in each.

## Disclosures
- These data were used in Study 1 (AUROC of cross-run disagreement for the 4J target; the window Beck-Chevalley defect). The pooled
  answer, its correctness and the between/within decomposition were not computed before this freeze.
- Synthetic demonstrations (engine/demos/synthetic.py, world A with family blind spots and null world B) were run before freezing to
  debug the code. No parameter of this analysis was chosen from them or from the real data.
- The incoherence detector of the engine cannot be tested on these data: each item's four answers are observed jointly, and jointly
  observed tables always admit a global law. Testing it on real data needs pairwise elicitation (Study 3 kit).
- With four deterministic answers per item the features take few distinct values; this is a property of the data, not a choice.
