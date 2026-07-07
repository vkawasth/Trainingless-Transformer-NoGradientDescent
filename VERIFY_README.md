# Memory Verification Package

A standalone, reproducible check of the memory claims for the
Geometry-Driven Compiler vs GD-400. **No private corpus, no trained
checkpoints, no special hardware.** Anyone with Python + PyTorch can run it and
get byte-identical numbers.

## Run

```bash
pip install torch matplotlib          # the only dependencies
python memory_viz.py                  # full 400-step comparison
python memory_viz.py --quick          # 20-step smoke test (seconds)
python memory_viz.py --seed 7         # any seed; output is deterministic
```

## What it does (and why you can trust it)

1. Builds its **own** synthetic corpus — a cyclic near-permutation with the
   same "each token has one successor" structure that makes the real corpus
   99.87% sparse. No external files.
2. Builds the **same model** as the paper (D=256, 6 layers, VOCAB=1017, tied
   embedding, SwiGLU FF).
3. Runs two optimizer regimes on that identical model:
   - **GD-400** — AdamW on *all* parameters (2 moment buffers x P).
   - **Compiler** — AdamW with the dispensable `WV` + attention-output
     weights **frozen** (`requires_grad=False`): no optimizer state, no
     gradient for them. This is the paper's Masked-Lagrangian claim.
4. Measures memory two independent ways that must agree:
   - **Analytic** — exact bytes read straight off the tensors/optimizer
     state. Deterministic to the bit.
   - **Empirical peak** — real high-water mark of live tensor memory during
     the backward pass (CUDA driver counter / MPS allocated / CPU live-tensor
     poll).
5. Computes the corpus-operator memory: dense `O(V^2)` vs sparse `O(nnz)`.

## Artifacts written

| File | For whom | Contents |
|------|----------|----------|
| `memory_viz.png` | general reader | two-panel chart (stacked optimizer memory; log-scale corpus reduction) |
| `memory_summary.txt` | general reader | plain-language summary, printed to console too |
| `memory_report.csv` | spreadsheet | one row per regime, all byte counts |
| `memory_report.json` | machine / audit | full record incl.\ environment + headline ratios |

## Expected numbers (seed 42, this architecture)

| Metric | GD-400 | Compiler |
|--------|-------:|---------:|
| Parameters | 17.32 MB | 17.32 MB |
| Gradients | 17.32 MB | 14.18 MB |
| Optimizer state (Adam) | 34.64 MB | 28.35 MB |
| **Analytic resident** | **69.28 MB** | **59.85 MB** |
| Empirical peak | ~71.5 MB | ~62.1 MB |

- **Optimizer-state saving from masking:** 6.29 MB = 18.2% of Adam state
  (786,432 of 4,330,240 params frozen).
- **Corpus operator:** dense 4.14 MB -> sparse 0.016 MB = **767.8x**
  reduction (grows to ~10^6x at VOCAB=50k).

Two separate stories — keep them separate:
the optimizer-masking saving is modest and scales with how much you freeze;
the `O(V^2) -> O(nnz)` corpus-operator saving is the dramatic one and scales
with vocabulary.

## Determinism

Given a fixed `--seed`, the analytic columns and the CSV are byte-identical
across machines (verified: two runs -> identical md5). The empirical peak may
vary by a few percent with allocator/backend, which is expected and is why the
analytic column is the one of record.
