#!/usr/bin/env python3
"""
Last Layer Inference Test
==========================
Tests the claim: "for inference, you only need the last layer"

Four interpretations tested:

A) EXIT EARLY: stop at layer k instead of layer 24.
   Measure perplexity vs full model at each exit point.
   The Gromov result (r=-0.720) predicts quality plateaus near L14.

B) SKIP EARLY LAYERS: run only layers k..23 (skip 0..k-1).
   Feed h_0 directly into layer k.
   Tests whether early layers can be skipped at inference time.

C) LAST LAYER ONLY: run all 24 layers, but replace W_K^(k) with
   W_K^(23) for layers 0..22. Only the last layer's weights matter.
   Tests interpretation A from the analysis.

D) LAYER IMPORTANCE: ablate each layer (skip it with residual passthrough)
   and measure the perplexity hit. Which layers are most critical?

Usage:
    python last_layer_inference.py --model gpt2-medium
"""

import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--n_texts', type=int, default=10)
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"  LAST LAYER INFERENCE TEST")
print(f"  Model: {args.model}")
print(f"{'='*70}\n")

print("Loading model...", flush=True)
config = GPT2Config.from_pretrained(args.model)
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d = model.config.n_embd
n_layers = model.config.n_layer

TEXTS = [
    "The transformer architecture processes sequences using self-attention mechanisms.",
    "Quantum mechanics describes the probabilistic nature of subatomic particles.",
    "Natural selection drives evolutionary adaptation in biological populations.",
    "The speed of light in vacuum is approximately 299 million meters per second.",
    "Neural networks learn hierarchical representations through gradient descent.",
    "Einstein developed the theory of general relativity in nineteen fifteen.",
    "The immune system produces antibodies to neutralize foreign pathogens.",
    "Climate change results from increased greenhouse gas concentrations in the atmosphere.",
    "Photosynthesis converts solar energy into chemical energy stored in glucose.",
    "The standard model describes fundamental particles and their interactions.",
][:args.n_texts]

def get_perplexity(logits, ids):
    """Compute perplexity from logits and token ids."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_ids    = ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, logits.shape[-1]),
                           shift_ids.view(-1))
    return float(torch.exp(loss))

def run_full(text):
    """Full 24-layer forward pass. Returns (ppl, hidden_states)."""
    ids = tok.encode(text, return_tensors='pt', max_length=64, truncation=True)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    ppl = get_perplexity(out.logits, ids)
    return ppl, out.hidden_states, ids

# ── Baseline: full model ──────────────────────────────────────────────────────
print("Computing baseline (full 24-layer)...")
base_ppls = []
for text in TEXTS:
    ppl, _, _ = run_full(text)
    base_ppls.append(ppl)
base_ppl = float(np.mean(base_ppls))
print(f"  Full model perplexity: {base_ppl:.2f}\n")

# ── TEST A: Early exit at layer k ─────────────────────────────────────────────
print("="*70)
print("  TEST A: Early exit at layer k")
print("  (use LM head on h_k instead of h_24)")
print("="*70)

exit_ppls = []
print(f"\n  {'Layer':>6}  {'PPL':>8}  {'vs full':>8}  {'degradation':>12}")
print("  " + "-"*40)

for exit_layer in range(0, n_layers+1):
    ppls = []
    for text in TEXTS:
        ids = tok.encode(text, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        h_k = out.hidden_states[exit_layer]   # [1, S, d]
        # Apply final layer norm + LM head
        h_normed = model.transformer.ln_f(h_k)
        logits = model.lm_head(h_normed)
        ppl = get_perplexity(logits, ids)
        ppls.append(ppl)
    mean_ppl = float(np.mean(ppls))
    exit_ppls.append(mean_ppl)
    ratio = mean_ppl / base_ppl
    print(f"  L{exit_layer:>2}:     {mean_ppl:>8.1f}  {ratio:>8.2f}x  "
          f"{'← baseline' if exit_layer==n_layers else ('✓ within 5%' if ratio<1.05 else ('← Gromov critical' if exit_layer==14 else ''))}")

# Find earliest layer within 5% of full
best_early = next((k for k,p in enumerate(exit_ppls) if p < base_ppl*1.05), n_layers)
print(f"\n  Earliest exit within 5% of full: L{best_early}")
print(f"  Speedup: {n_layers/max(best_early,1):.1f}x fewer layers")
print(f"  L14 (Lefschetz critical point): {exit_ppls[14]:.1f} ({exit_ppls[14]/base_ppl:.2f}x baseline)")

# ── TEST B: Skip first k layers ───────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST B: Skip first k layers (feed h_0 into layer k directly)")
print("="*70)

print(f"\n  {'Skip':>6}  {'PPL':>8}  {'vs full':>8}")
print("  " + "-"*28)

for skip_to in [0, 1, 2, 4, 8, 12, 14, 16, 20, 23]:
    ppls = []
    for text in TEXTS:
        ids = tok.encode(text, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # Start from h_0, inject into layer skip_to
        h = out.hidden_states[0]   # embedding only, skip early layers
        # Run from layer skip_to onward
        for l in range(skip_to, n_layers):
            h = model.transformer.h[l](h)[0]
        h_normed = model.transformer.ln_f(h)
        logits = model.lm_head(h_normed)
        ppl = get_perplexity(logits, ids)
        ppls.append(ppl)
    mean_ppl = float(np.mean(ppls))
    ratio = mean_ppl / base_ppl
    print(f"  skip {skip_to:>2}:  {mean_ppl:>8.1f}  {ratio:>8.2f}x")

# ── TEST C: Replace all W_K with W_K^(23) ────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST C: Use only last layer's W_K for all layers")
print(f"  (tests: does only W_K^(23) matter?)")
print("="*70)

# Save original W_K weights
orig_wk = [model.transformer.h[l].attn.c_attn.weight[:, d:2*d].clone()
           for l in range(n_layers)]
wk_last = orig_wk[-1].clone()

# Replace all W_K with last layer's W_K
for l in range(n_layers-1):
    model.transformer.h[l].attn.c_attn.weight.data[:, d:2*d] = wk_last

ppls_C = []
for text in TEXTS:
    ids = tok.encode(text, return_tensors='pt', max_length=64, truncation=True)
    with torch.no_grad():
        out = model(ids)
    ppl = get_perplexity(out.logits, ids)
    ppls_C.append(ppl)
ppl_C = float(np.mean(ppls_C))

# Restore
for l in range(n_layers-1):
    model.transformer.h[l].attn.c_attn.weight.data[:, d:2*d] = orig_wk[l]

print(f"\n  All W_K → W_K^(23): PPL = {ppl_C:.1f}  ({ppl_C/base_ppl:.2f}x baseline)")
print(f"  If ≈1.0x: only the last layer's W_K matters")
print(f"  If >>1.0x: each layer's distinct W_K is necessary")

# ── TEST D: Layer ablation (skip one layer at a time) ────────────────────────
print(f"\n{'='*70}")
print(f"  TEST D: Ablate each layer (skip it, passthrough residual)")
print(f"  (which layers are most critical for inference quality?)")
print("="*70)

print(f"\n  {'Layer':>6}  {'PPL':>8}  {'vs full':>8}  {'importance'}")
print("  " + "-"*50)

ablate_ppls = []
for skip_l in range(n_layers):
    ppls = []
    for text in TEXTS:
        ids = tok.encode(text, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # Run all layers EXCEPT skip_l
        h = out.hidden_states[0]
        for l in range(n_layers):
            if l == skip_l:
                pass   # skip: h stays the same (identity = residual passthrough)
            else:
                h = model.transformer.h[l](h)[0]
        h_normed = model.transformer.ln_f(h)
        logits = model.lm_head(h_normed)
        ppl = get_perplexity(logits, ids)
        ppls.append(ppl)
    mean_ppl = float(np.mean(ppls))
    ablate_ppls.append(mean_ppl)
    ratio = mean_ppl / base_ppl
    importance = '★★★ CRITICAL' if ratio > 2.0 else ('★★ important' if ratio > 1.2 else '★ minor')
    if skip_l in [0,1,2,5,10,12,13,14,15,16,20,22,23]:
        print(f"  L{skip_l:>2}:     {mean_ppl:>8.1f}  {ratio:>8.2f}x  {importance}")

most_critical = int(np.argmax(ablate_ppls))
print(f"\n  Most critical layer: L{most_critical} (PPL={ablate_ppls[most_critical]:.1f} when skipped)")
print(f"  Least critical: L{int(np.argmin(ablate_ppls))} (PPL={min(ablate_ppls):.1f} when skipped)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}\n")
print(f"  Full model PPL: {base_ppl:.2f}")
print(f"  Early exit at L{best_early}: {exit_ppls[best_early]:.2f} PPL ({n_layers-best_early} layers saved)")
print(f"  Using only W_K^(23): {ppl_C:.2f} PPL")
print(f"  Most critical layer: L{most_critical}")
print(f"""
  INTERPRETATION:
    If early exit works at L14: Gromov compactness threshold confirmed.
      The model has converged by L14 — later layers add minimal signal.
      
    If skip early layers fails (PPL >> baseline):
      Each layer transforms h progressively — cannot skip early layers.
      The norm difference (h_0=13, h_23=850) makes this likely.
      
    If W_K^(23) replacement fails (PPL >> baseline):
      Each layer needs its own W_K — the Hessenberg attractor at each
      depth is distinct, not just the final layer's.
      
    If ablation shows most critical ≠ L23:
      The last layer is NOT the most important for inference quality.
      The document's claim "you only need the last layer" is wrong.
""")
