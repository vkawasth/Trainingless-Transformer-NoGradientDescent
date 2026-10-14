# Pre-registration: hallucination tests for "Shadows, Fibres and Secants of the Posterior Tower" (Section 16)

Frozen 2026-10-13, before any outcome of the analyses below was computed. SHA-256 of this file is recorded in FROZEN_PREREG.txt.

## Disclosure of prior contact with the data
- LLM-drift data (Chen, Zaharia, Zou; github.com/lchen001/LLMDrift): previously analysed in the validation paper (accuracies and joint
  outcome tables per task; `monitor/llm_analysis.py`). The statistics defined here (disagreement AUROC, window defect c and its Markov
  null) have not been computed on these data before. Before freezing, only the file structure (column names, rows per task) was
  inspected; the extraction rules are those of `llm_analysis.py`, reused unchanged.
- HaluEval (Li et al., EMNLP 2023; MIT licence): only the licence has been read. No file has been opened.

## Study 1 (runs here, data on disk): LLM-drift answers
Runs per item: R = {35M, 35J, 4M, 4J} = GPT-3.5 (0301, 0613), GPT-4 (0314, 0613). One answer per run and item (temperature 0.1).
Tasks with a reference answer and a comparable answer form, and the extraction rule (from llm_analysis.py):
USMLE (intact prompts; option letter), PRIME_FULL ([Yes]/[No]), COUNTAHAPPYNUMBER (boxed integer), HOTPOTQA (normalised text), ARC
(grid string, whitespace removed). OpinionQA, SensitiveQ and LeetCode are excluded (no reference answer comparable by equality).
An answer is *valid* if the rule extracts one; *wrong* if valid and different from the reference.

**P1 (consistency detector).** Target run 4J (primary) and 35J (secondary). For an item with a valid target answer, the score is
s = (number of other runs with a valid answer different from the target answer) / (number of other runs with a valid answer); items
with no valid other answer are dropped. Outcome: target wrong (1) vs correct (0). Statistic: AUROC (Mann-Whitney, ties counted 1/2),
95% percentile bootstrap CI over items (B = 2000, seed 20261013).
**H1.** For target 4J, the CI lower bound exceeds 0.5 in at least 3 of the 5 tasks.

**P2 (window Beck-Chevalley defect).** Binary correctness Y_r (1 = wrong) on items where 4M, 4J and 35J are all valid. Windows
A = sigma(Y_4M, Y_4J) (same model across dates) and B = sigma(Y_4J, Y_35J) (same date across models), sharing Y_4J. Empirical joint
on the 8 cells; c = max over y in {0,1} with both conditional margins non-degenerate of |phi(Y_4M, Y_35J | Y_4J = y)|
(Proposition 16.2(c)); NA if no stratum is non-degenerate.
Null (Beck-Chevalley / Markov): parametric bootstrap from the fitted model P(y4J) P(y4M | y4J) P(y35J | y4J), B = 2000, same n,
seed 20261013; p = (1 + #{c_null >= c_obs}) / (B + 1). CI for c: percentile bootstrap over items, B = 2000.
**H2.** Beck-Chevalley fails (Holm-adjusted p < 0.05 across the 5 tasks) in at least one task.

**P3 (exploratory, no inference).** Spearman correlation across tasks between c and the 4J AUROC.

Reporting: all five tasks, both targets, whatever the result. Deviations, if any, are listed in a deviations section of the report.

## Study 2 (kit only; needs model access): HaluEval QA
For each HaluEval QA item, a model is shown (knowledge, question) and sampled K = 5 times at temperature 1.0; the right and the
hallucinated answers are each scored by the same disagreement detector against the K samples (fraction of samples that an exact-match
or NLI judge marks as inconsistent with the answer). Outcome: hallucinated vs right answer. Statistic: AUROC with bootstrap CI.
Script: `kit_halueval.py` (reads a JSONL of samples produced by the user's model; ships with a synthetic smoke test).

## Study 3 (kit only; needs model access): pairwise elicitation of three-claim cycles
For triples of related binary claims (a, b, c), the model is asked about each pair in a separate prompt (contexts {a,b}, {b,c}, {a,c}),
n repetitions each. Pairwise tables are estimated per context. The certificate of Proposition 16.1 is evaluated in its general form
(Conditionals Thm 4.8 at tau = 0): the family is obstructed iff some 3-cycle inequality sum gamma_ij E_ij <= 1 fails; the violation and a
bootstrap CI are reported. This is the only design in which the below-1/3 obstruction can appear, since jointly observed answers always
admit a joint law. Script: `kit_pairwise_cycles.py` (with a synthetic smoke test that reproduces Proposition 16.1).
