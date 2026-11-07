# Deviations from PREREG_llm_fracture.md

## Deviation 1 (2026-11-01, procedural; no effect on outputs)

While running the kit, transformers warned: "The attention mask is not set and cannot be inferred from input because pad token is
same as eos token." Each call to `generate` receives one unpadded prompt, so the correct mask is all ones, and that is also what
transformers falls back to when it cannot infer one. The kit now passes `attention_mask=torch.ones_like(ids)` explicitly, which
silences the warning and leaves the sampled tokens unchanged (same seed, same logits).

Nothing else changed: items, seeds, sampling settings, scoring and analysis are as frozen.

- Frozen kit (v1):  6cefa0d9127bd5efd8567a6ea283edf99b6c2ff8a86b26b93eb07ab2459c78bd
- Patched kit (v1.1): 3098ad6f48fc7a2b43d0247bf8df05c0c3cd4a7fb8fb0affc2adab3609fa853d
- Any outputs already produced with v1 remain valid and need not be rerun.

## Deviation 2 (2026-11-04, device and runner; declared before any answer was scored or read)

The frozen kit runs the model in float32 on the CPU. On the author's laptop (Apple silicon) the model and ten sampled continuations of
prompts of up to about 1,500 tokens did not fit in free memory: after 32 hours of CPU time the process was almost entirely in
compressed memory and had written no answers. The run was stopped and replaced by a separate runner, `run_icl_mps.py`; the kit itself
is unchanged (v1.1 above).

What the runner keeps: the kit's own 1,800 prompts (`items()`), the seed SEED + id per prompt, temperature 1, top_p 1, 5 new tokens,
10 samples, the model's own generation config, and the output format, so `kit_icl_poison.py score` is used unchanged.

What it changes:
- the model runs on the Apple GPU (mps) in bfloat16, with eager attention (PyTorch's fused attention on mps fails on the model's
  grouped-query heads); sampled tokens therefore need not match a CPU float32 run token for token;
- each prompt is read once and its key-value cache copied to the 10 samples, which are then drawn by the same `generate` call
  (same sampling distribution; about 8 times faster: 3.7 s against 29.6 s per prompt on 5 test prompts);
- each answer is written and flushed at once, with a progress line, and `--resume` skips ids already written.

The 5-prompt timing tests (`test5.jsonl`, `test5b.jsonl`) are not part of the study. On 2026-11-04 the score step was run once by
mistake on a partial file (fewer than 1,800 answers). Correction (2026-11-05): an interim score output from a partial
file was shared and viewed on 2026-11-05. It is not interpreted or reported, and it changes nothing in the analysis, which is fixed in
advance and run once on the complete file (output name `study15_final.json`).
The analysis is run once, on the complete `answers.jsonl`.

- Runner: 9fb3f1f8b82c041672e64fa4ab4b6d41a45ef039127cebe8e52969ae30937d8c  run_icl_mps.py

### Deviation 2a (2026-11-05, memory only; outputs unchanged)

After 652 prompts the run averaged 111 s per prompt with 18 GB of swap in use. Two memory fixes in the runner: the prompt is read with
the decoder alone (`model.model`), which builds the same key-value cache without the output-layer scores for every prompt position;
and GPU memory is returned after each prompt (`torch.mps.empty_cache()`). The computation of the sampled tokens is unchanged. The run was then restarted
without `--resume`, which overwrote the 652 earlier answers, so all 1,800 answers come from runner v2 (about 22 s per prompt).

- Runner v2: f48135c7e7fa06654591044aa89cb1a7ff875826b342772a0182dc5672f3e18a  run_icl_mps.py

## Deviation 2b (2026-11-06, secondary model only; declared before any secondary-model answer was scored or read)

With runner v2, SmolLM2-1.7B-Instruct ran at about 230 s per prompt. Its cache is about 11 times Qwen's (one key/value head per query
head, against Qwen's grouped heads), and model plus cache did not fit in memory. Runner v3 changes:
- eager attention only for models with grouped-query heads on mps; SmolLM2 uses the default fused attention;
- `--chunk c` draws the 10 samples c at a time from copies of the shared prompt cache (seed SEED + id as before; the random stream
  continues across chunks, so the distribution is the same but tokens are not identical to an unchunked run);
- it refuses to overwrite a non-empty output file without `--resume`;
- it accepts the newer transformers return type of `apply_chat_template`.
With the defaults (no `--chunk`, Qwen's grouped heads) the computation is that of v2, so the primary-model answers are unaffected.
The secondary model is run from scratch with v3 and `--chunk 2` into `answers_smol2.jsonl`; the partial v2 file for SmolLM2 is not
used.

- Runner v3: ac73be54d33d5abe9efa8bb2ca0413dbeb4fd0b9c468591404224bfa45c92cc4  run_icl_mps.py

## Deviation 3 (2026-11-06): secondary model not run
The SmolLM2-1.7B-Instruct replication (E2) was started with runner v3 and stopped by the author after 25 prompts. Even with
`--chunk 2` the long prompts drove the laptop into swap (4.7 GB in use), at about 290 s per prompt, an estimated 6 days in all. E2 is
therefore not run, as infeasible on the available hardware. The partial files (`answers_smol.jsonl`, `answers_smol2.jsonl`) are not
scored or used. The primary analysis (`study15_final.json`) is unaffected.
