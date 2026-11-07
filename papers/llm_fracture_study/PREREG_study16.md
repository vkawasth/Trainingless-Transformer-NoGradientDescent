# Pre-registration (DRAFT, not frozen): Study 16, a four-question cycle for LLM answers

Companion to "Conditioned states and Kähler geometry", sections 12.15-12.19: composition rules as sections of the marginal projection;
the 4-cycle (CHSH) cover; rigidity of the contextuality boundary (Theorem 74).

Status: DRAFT, 2026-11-05. It becomes binding only when the author freezes it: the SHA-256 digests of this file and of
`kit_cycle_context.py` are then written to `FROZEN_PREREG_study16.txt`, before any model output is generated.

## Disclosure
- No data exist. Every item is synthetic and is generated deterministically by `kit_cycle_context.py` (seed 20261106).
- No model has been run on any item. The only outputs seen before freezing come from the kit's smoke test, which uses two simulated
  responders and no language model (`kit16_smoke_output.txt`). Those outputs only exercise the analysis code.
- Models are run locally by the author. Nothing is uploaded, and no API key is needed or stored.

## What is being measured, and what is not
The note's framework is a way of measuring answers, not a theory of LLMs. Each question shows the model the same documents and asks for
the joint reading of two gauges. Each question is treated as a *context*, and the model's sampling distribution at fixed decoding
settings as that context's *local law*. The study asks how these local laws relate to one another:
- whether a claim's answer distribution depends on which other claim it is asked with (drift);
- whether the answers to the four cycle questions admit any joint law once drift is accounted for (contextuality, in the
  contextuality-by-default sense of Dzhafarov and Kujala);
- whether the model's answer to a pair that no document relates lies in the set of values its own other answers allow (coherence).

It does not test whether LLMs are "contextual" in any physical sense. All results hold only for the stated decoding settings. Because the
evidence is synthetic and partly contradictory by design, there is no ground truth for the readings. The study is about consistency, not
factual accuracy.

## Design
- **Items.** 40 items, each with four fictitious gauges A0, A1, B0, B1 that read "high" or "low". The names are random pronounceable
  strings, distinct within an item.
- **Evidence.** For each of the four cycle relations, M = 10 one-sentence documents (4 templates each way):
  - A0~B0, A0~B1 and A1~B0 are stated as "same";
  - A1~B1 is stated as "different".
  - A fraction (1 + λ)/2 of each relation's documents support it, and the rest state the opposite.
  - A follower that answers each question from that relation's documents alone would therefore have correlations λ, λ, λ, −λ, so the
    CHSH value is S = 4λ.
  - No document states a single gauge's reading, and no document relates A0 to A1 or B0 to B1.
  - The 40 documents are shuffled with a seed derived from the item and λ.
- **Strength grid.** λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}. The noncontextuality boundary S = 2 falls between 0.4 and 0.6.
- **Questions.** The same documents are shown with six questions:
  - the four cycle pairs (A0,B0), (A0,B1), (A1,B0), (A1,B1);
  - the two unasked pairs (A0,A1) and (B0,B1).
  - The question asks for both readings in a fixed form ("<name>: <reading>, <name>: <reading>"), with the order of "high"/"low" in
    the instruction randomised per prompt.
  - That gives 40 × 6 × 6 = 1440 prompts.
- **Models.** Primary: Qwen2.5-1.5B-Instruct. Secondary (replication only): SmolLM2-1.7B-Instruct. Each uses its own chat template.
- **Sampling.** K = 20 samples per prompt at temperature 1.0, top_p 1.0, 16 new tokens, seed 20261106 + prompt index. On the GPU
  (mps or cuda) the kit runs in bfloat16. It reads the prompt once and shares the cache across the 20 samples, and the same sampling
  call draws them (as in Study 15, Deviation 2).
- **Parsing.** For each gauge name, the first "high" or "low" within 25 characters after it. A sample missing either reading is
  *invalid*. Invalid samples are excluded and their rate is reported per (λ, question type).

## Quantities (pooled over items at each λ; the item is the unit of resampling; bootstrap B = 2000, seed 20261106)
- **⟨R_c⟩:** for each cycle question, P(same reading) − P(different reading).
- **⟨g⟩_c:** for each gauge g in each of its two cycle questions, P(high) − P(low).
- **S** = ⟨R_A0B0⟩ + ⟨R_A0B1⟩ + ⟨R_A1B0⟩ − ⟨R_A1B1⟩.
- **Drift** Δ = Σ_g |⟨g⟩_c − ⟨g⟩_c′|, summed over the four gauges and their two contexts.
- **CbD statistic** C = s_odd(⟨R⟩) − 2 − Δ, where s_odd is the maximum of Σ ±⟨R_c⟩ over sign patterns with an odd number of minus
  signs. By the criterion for cyclic systems of binary variables (Kujala, Dzhafarov and Larsson 2015; Kujala and Dzhafarov 2016), the
  system is contextual in the contextuality-by-default sense iff C > 0.
- **Identified interval** for each unasked pair: the range of P(equal readings) over all joint laws on the 16 joint readings whose four
  cycle tables lie, cell by cell, in the 95% bootstrap boxes of the observed tables (two linear programs). If no joint law fits the
  boxes, the interval is *empty*, meaning the answers are incompatible within sampling error.
- **Elicited value:** P(equal readings) in the model's answers to the unasked pair.

## Hypotheses (primary model)
**H1 (drift).** The model's single-gauge answers depend on which other gauge is asked. Statistic: mean drift over λ, minus a noise
baseline (the same statistic computed between two random halves of each context's own samples).
- Supported if the 95% bootstrap CI of the excess lies above 0.

**H2 (contextuality beyond drift).** Following the evidence makes the answers contextual even after drift is allowed for.
- Supported if the 95% CI of C at λ = 1 lies above 0 and the 95% CI of C at λ = 0 lies below 0.

**H3 (coherence of unasked answers).** Wherever the model's cycle answers admit a joint law (the identified interval is non-empty),
its elicited answer to the unasked pair lies in that interval.
- Supported if this holds in every such (λ, pair) case.
- If no case has a non-empty interval, H3 is reported as *not testable*.

The study tests whether the model's answers behave as a single probabilistic object. It does not test the theorems of the note.

## Exploratory (no inference)
- **E1.** The width of the identified interval against the observed S, beside the ideal-follower value (2 − S)/4 for P(A0 = 0, A1 = 0).
  The prediction is narrowing toward S = 2 and emptiness beyond it.
- **E2.** Where the elicited value of an unasked pair lies in its interval: near the product value (independence), the
  maximum-entropy value, or an end.
- **E3.** The λ at which C first exceeds 0, set against S = 2.
- **E4.** Invalid rates.
- **E5.** All of the above for the secondary model.

## Reporting
- All λ values, both models, and every hypothesis are reported whatever the result.
- Deviations go into `DEVIATIONS_study16.md`, with reasons, before any results are shown.
- The analysis is run once, on the complete answer file, with output `study16_final.json`. No interim scores are computed.

## How to run (local, after freezing)
    python kit_cycle_context.py items --out prompts16.jsonl
    caffeinate -i python kit_cycle_context.py run --prompts prompts16.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --out answers16.jsonl
    python kit_cycle_context.py score --prompts prompts16.jsonl --answers answers16.jsonl --out study16_final.json
The run refuses to overwrite a non-empty answer file. If it stops, rerun the same command with `--resume` to continue.

## References
- E. N. Dzhafarov, J. V. Kujala and J.-Å. Larsson, Contextuality in three types of quantum-mechanical systems, Found. Phys. 45 (2015) 762-782.
- J. V. Kujala, E. N. Dzhafarov and J.-Å. Larsson, Necessary and sufficient conditions for an extended noncontextuality in a broad class of
  quantum mechanical systems, Phys. Rev. Lett. 115 (2015) 150401.
- J. V. Kujala and E. N. Dzhafarov, Proof of a conjecture on contextuality in cyclic systems with binary variables, Found. Phys. 46 (2016) 282-299.
- A. Fine, Hidden variables, joint probability, and the Bell inequalities, Phys. Rev. Lett. 48 (1982) 291-295.
