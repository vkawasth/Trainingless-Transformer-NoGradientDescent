#!/usr/bin/env python3
"""
k_p Hallucination Detector
===========================
Factual text  → k_p = 1 (supersingular, real eigenvalues, full context)
Fabricated text → k_p = 2? (binary collapse, complex eigenvalues, half context)

The discriminant of the 2x2 attention block:
  disc = tr(A)^2 - 4*det(A)
  > 0: real eigenvalues → k_p = 1 (supersingular)
  < 0: complex eigenvalues → k_p = 2 (binary collapse)

The hallucination signal: disc transitions negative at specific layers.

Usage: python kp_hallucination.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  k_p HALLUCINATION DETECTOR")
print(f"  Factual → k_p=1 (supersingular)")
print(f"  Fabricated → k_p=2? (binary collapse)")
print(f"{'='*65}\n")

FACTUAL = [
    "Albert Einstein was born in Ulm Germany in 1879.",
    "He developed the special theory of relativity in 1905.",
    "Einstein won the Nobel Prize in Physics in 1921.",
    "The speed of light in vacuum is 299,792,458 meters per second.",
    "Water freezes at 0 degrees Celsius at standard pressure.",
    "The Earth orbits the Sun once every 365.25 days.",
    "DNA is a double helix structure discovered by Watson and Crick.",
    "Paris is the capital city of France.",
    "The human brain contains approximately 86 billion neurons.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
]

FABRICATED = [
    "Albert Einstein was born in Vienna Austria in 1950.",
    "He developed quantum teleportation in 1912.",
    "Einstein won the Nobel Prize in Chemistry in 1930.",
    "The speed of light in vacuum is 150,000 meters per second.",
    "Water freezes at 50 degrees Celsius at standard pressure.",
    "The Earth orbits the Sun once every 200 days.",
    "DNA is a single strand structure discovered by Darwin.",
    "Berlin is the capital city of France.",
    "The human brain contains approximately 3 billion neurons.",
    "Photosynthesis converts nitrogen and methane into protein.",
]

print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model, output_attentions=True)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d   = model.config.n_embd
n_L = model.config.n_layer
n_H = model.config.n_head
print(f"  layers={n_L}  heads={n_H}\n")

def get_attention_stats(texts, label):
    """Compute 2x2 attention block stats across layers for a set of texts."""
    A_sum  = [np.zeros((2,2)) for _ in range(n_L)]
    disc_all = [[] for _ in range(n_L)]
    kp_all   = [[] for _ in range(n_L)]
    count = 0
    for text in texts:
        ids = tok.encode(text, return_tensors='pt', max_length=32, truncation=True)
        if ids.shape[1] < 3: continue
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        for l in range(n_L):
            A = out.attentions[l][0].mean(0).numpy()[:2,:2]
            tr   = A[0,0] + A[1,1]
            det  = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            disc = tr**2 - 4*det
            kp   = 1 if disc >= 0 else 2
            A_sum[l] += A
            disc_all[l].append(float(disc))
            kp_all[l].append(kp)
        count += 1
    return disc_all, kp_all, count

print("Computing k_p for FACTUAL texts...", flush=True)
disc_f, kp_f, n_f = get_attention_stats(FACTUAL, "factual")
print(f"  Processed {n_f} texts")

print("Computing k_p for FABRICATED texts...", flush=True)
disc_b, kp_b, n_b = get_attention_stats(FABRICATED, "fabricated")
print(f"  Processed {n_b} texts\n")

# ── Per-layer comparison ──────────────────────────────────────────────────────
print("="*65)
print("  PER-LAYER: discriminant and k_p")
print("  disc > 0 → k_p=1 (supersingular)")
print("  disc < 0 → k_p=2 (binary collapse / hallucination)")
print("="*65)
print(f"\n  {'Layer':>6}  {'disc_fact':>11}  {'kp_f':>5}  {'disc_fab':>11}  {'kp_b':>5}  {'different?':>12}")
print("  "+"-"*62)

n_layers_different = 0
for l in range(n_L):
    dmf = float(np.mean(disc_f[l]))   # mean disc over factual texts
    dmb = float(np.mean(disc_b[l]))   # mean disc over fabricated texts
    kf_mean = float(np.mean(kp_f[l]))
    kb_mean = float(np.mean(kp_b[l]))
    different = abs(kf_mean - kb_mean) > 0.1 or (dmf > 0) != (dmb > 0)
    if different: n_layers_different += 1
    flag = "← DIFF" if different else ""
    print(f"  L{l:>2}:   {dmf:>+11.6f}  {kf_mean:>5.2f}  {dmb:>+11.6f}  {kb_mean:>5.2f}  {flag:>12}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SUMMARY")
print("="*65)

all_disc_f = [d for layer in disc_f for d in layer]
all_disc_b = [d for layer in disc_b for d in layer]
all_kp_f   = [k for layer in kp_f for k in layer]
all_kp_b   = [k for layer in kp_b for k in layer]

print(f"""
  FACTUAL texts:
    Mean discriminant: {np.mean(all_disc_f):+.6f}
    k_p=1 fraction:    {all_kp_f.count(1)/len(all_kp_f):.3f}
    k_p=2 fraction:    {all_kp_f.count(2)/len(all_kp_f):.3f}

  FABRICATED texts:
    Mean discriminant: {np.mean(all_disc_b):+.6f}
    k_p=1 fraction:    {all_kp_b.count(1)/len(all_kp_b):.3f}
    k_p=2 fraction:    {all_kp_b.count(2)/len(all_kp_b):.3f}

  Layers where k_p differs: {n_layers_different} of {n_L}
""")

# ── The discriminant as continuous signal ─────────────────────────────────────
print("="*65)
print("  DISCRIMINANT AS CONTINUOUS HALLUCINATION SIGNAL")
print("  (even when k_p=1 everywhere, disc_fact vs disc_fab may differ)")
print("="*65)

from scipy.stats import ttest_ind, mannwhitneyu

# Per-layer t-test: does factual vs fabricated differ in discriminant?
print(f"\n  {'Layer':>6}  {'mean_f':>10}  {'mean_b':>10}  {'diff':>10}  {'p_val':>8}")
print("  "+"-"*50)
sig_layers = []
for l in range(n_L):
    mean_f = float(np.mean(disc_f[l]))
    mean_b = float(np.mean(disc_b[l]))
    diff   = mean_f - mean_b
    if len(disc_f[l]) > 1 and len(disc_b[l]) > 1:
        _, pval = ttest_ind(disc_f[l], disc_b[l])
    else:
        pval = 1.0
    sig = " *" if pval < 0.05 else ("  ." if pval < 0.1 else "")
    if pval < 0.1: sig_layers.append(l)
    print(f"  L{l:>2}:   {mean_f:>+10.6f}  {mean_b:>+10.6f}  {diff:>+10.6f}  {pval:>8.4f}{sig}")

print(f"\n  Significant layers (p<0.1): {sig_layers}")
print(f"""
  INTERPRETATION:
  If disc_factual > disc_fabricated: factual attention has more "real-eigenvalue"
  character — the attention matrix is less complex-rotational for true facts.
  
  If disc_fabricated < 0 at any layer: k_p transitions to 2 → binary collapse.
  This is the hallucination signal from the paper.
  
  The discriminant is the continuous version of k_p:
    disc >> 0: strongly supersingular (real, separated eigenvalues)
    disc ≈ 0:  near the phase boundary
    disc < 0:  binary collapse (complex eigenvalues, k_p=2)
""")
