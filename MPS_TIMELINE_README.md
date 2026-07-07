# Stage-by-Stage Memory Timeline on Apple Silicon (MPS)

## Why this exists

PyTorch's native memory timeline (`torch.cuda.memory._record_memory_history`
+ pytorch.org/memory_viz) is **CUDA-only**. On Apple Silicon there is no
allocation-trace API, so that tool produces nothing on a Mac. This script
reconstructs the same forward/backward/step timeline using `nn.Module` hooks
and the MPS memory counter, so you get the CUDA-style view on MPS.

## Run

```bash
pip install torch matplotlib
python mps_memory_timeline.py                 # both regimes
python mps_memory_timeline.py --regime gd
python mps_memory_timeline.py --regime compiler
```

Backend auto-detected: uses `torch.mps.current_allocated_memory()` on Apple
Silicon, `torch.cuda.memory_allocated()` on NVIDIA, and a live-tensor sum on
CPU (so it runs anywhere; identical code path).

## What you get (per regime)

- `mps_timeline_<regime>.png` — the timeline chart: activations ramp up on the
  forward pass, plateau at the loss, staircase down on the backward pass, and
  the optimizer step spikes at the end.
- `mps_timeline_<regime>.csv` — every event: order, phase, layer, allocated_mb,
  delta_mb. This is the stage-by-stage table.
- `mps_timeline_<regime>.json` — full trace + peak location.

## How to read it (example, this model)

GD-400 style, step-by-step:

| order | phase | layer | MB | delta |
|------:|-------|-------|----:|------:|
| 1 | init | params | 17.3 | — |
| 2–15 | forward | embed…head | ↑ to 64.5 | +2.1 per attn, +5.2 per ff |
| 16 | loss | cross_entropy | 64.5 | flat |
| 17–31 | backward | head…embed | ↓ to 19.4 | frees activations |
| 32 | step | adam_step | **71.4 (peak)** | **+51.96** (Adam moments) |

Compiler (WV+op masked):
- adam_step allocates only **+42.53 MB** (vs GD's +51.96) — the 9.4 MB
  difference is the masked weights' optimizer moments never being allocated.
- Because its optimizer spike is smaller, the compiler's true **peak lands in
  the backward pass (63.5 MB at head)**, not at the step — a qualitatively
  different memory profile, visible directly in the timeline.

## Notes / limitations

- Hooks fire at **module boundaries**, so the granularity is per-layer, not
  per-kernel (CUDA's native tool is per-allocation). For fwd/bwd/optimizer
  structure this is the right resolution.
- The MPS counter reports currently-allocated device memory; peaks between two
  hook points can be slightly higher than the sampled boundary value. For a
  finer peak inside a specific phase, add more `tl.mark(...)` calls around it.
- To profile more than one training step (e.g. to see steady-state), wrap the
  forward/backward/step block in a loop and mark each iteration.
