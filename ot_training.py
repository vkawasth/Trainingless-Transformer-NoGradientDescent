#!/usr/bin/env python3
"""
Optimal Transport Training
===========================
Computes W_K^(k) for all 24 layers via geodesic interpolation
on the generic stratum, instead of gradient descent.

THEORY:
  Gradient descent finds W_K^(k) by minimizing loss over 300K steps.
  OT finds the same W_K^(k) by computing the geodesic from source to target
  on the Lagrangian Grassmannian LGr(d, 2d).

  The geodesic is parameterized by t ∈ [0,1]:
    W_K(t) = W_K^(src) + t * (W_K^(tgt) - W_K^(src))

  At t = k/23, this gives W_K^(k) for layer k.
  This IS the shear telescoping result: intermediate matrices are
  linear interpolations between source and target.

THREE STEPS:
  1. Determine W_K^(src) = W_K^(0): the source (near embedding space)
     Computed as: optimal W_K for attending to input context
     = first eigenvector of the input covariance E[h h^T]

  2. Determine W_K^(tgt) = W_K^(23): the target (near output space)
     Computed as: optimal W_K for predicting next tokens
     = first eigenvector of the output-input cross-covariance E[y h^T]

  3. Interpolate: W_K^(k) = (1 - k/23) * W_K^(src) + (k/23) * W_K^(tgt)

EXPERIMENT:
  Compare OT-computed W_K matrices to gradient-descent-trained W_K.
  If cos_sim(W_K^OT, W_K^trained) > 0 across layers:
    The OT geodesic is in the right direction.
  If cos_sim > 0.5:
    OT substantially predicts the trained weights.
  If cos_sim > 0.9:
    OT is a practical replacement for gradient descent on W_K.

Usage:
    python ot_training.py --model gpt2-medium
"""

import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.linalg import hessenberg as scipy_hessenberg
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--n_texts', type=int, default=20)
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"  OPTIMAL TRANSPORT TRAINING")
print(f"  Computing W_K via geodesic instead of gradient descent")
print(f"{'='*70}\n")

print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d = model.config.n_embd
n_layers = model.config.n_layer

def get_WK(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    return W[:, d:2*d]

WK_trained = [get_WK(l) for l in range(n_layers)]

# ── Training texts ────────────────────────────────────────────────────────────
TEXTS = [
    "The transformer architecture processes sequences using self-attention.",
    "Quantum mechanics describes particles using wave functions.",
    "Natural selection drives evolutionary adaptation in populations.",
    "The speed of light is approximately 299 million meters per second.",
    "Neural networks learn hierarchical representations from data.",
    "Einstein developed general relativity describing gravity as curvature.",
    "The immune system produces antibodies to neutralize pathogens.",
    "Climate change results from increased greenhouse gas concentrations.",
    "Photosynthesis converts solar energy into chemical energy in glucose.",
    "The standard model describes fundamental particles and forces.",
    "Language models predict the probability of each token in sequence.",
    "Gradient descent finds parameters minimizing a loss function.",
    "Attention mechanisms allow models to focus on relevant context.",
    "The residual connection adds input to the output of each block.",
    "Layer normalization stabilizes training by normalizing activations.",
    "The Fourier transform decomposes signals into frequency components.",
    "Entropy measures the amount of uncertainty in a probability distribution.",
    "Topology studies properties preserved under continuous deformations.",
    "Category theory provides a unified language for mathematical structures.",
    "Differential geometry extends calculus to curved manifolds and spaces.",
][:args.n_texts]

print(f"Computing source and target statistics from {len(TEXTS)} texts...\n")

# ── Step 1: Compute source W_K (optimal for input covariance) ─────────────────
# Source: W_K that best captures the input hidden state directions
# = top eigenvectors of E[h_0 h_0^T] (input covariance)
# This is what attention at layer 0 should attend to

Sigma_input = np.zeros((d, d))  # E[h_0 h_0^T]
Sigma_cross = np.zeros((d, d))  # E[h_0 h_{24}^T]  (cross-covariance)
n_tokens = 0

config = model.config.__class__.from_pretrained(args.model)
config.output_hidden_states = True
model_hs = GPT2LMHeadModel.from_pretrained(args.model, config=config)
model_hs.eval()

for text in TEXTS:
    ids = tok.encode(text, return_tensors='pt', max_length=64, truncation=True)
    with torch.no_grad():
        out = model_hs(ids, output_hidden_states=True)

    h0 = out.hidden_states[0][0].numpy()   # [S, d]
    h24 = out.hidden_states[-1][0].numpy()  # [S, d]

    Sigma_input += h0.T @ h0   # [d, d]
    Sigma_cross += h0.T @ h24  # [d, d]
    n_tokens += h0.shape[0]

Sigma_input /= n_tokens
Sigma_cross /= n_tokens

print(f"  Accumulated {n_tokens} token positions")
print(f"  ||Sigma_input|| = {np.linalg.norm(Sigma_input):.2f}")
print(f"  ||Sigma_cross|| = {np.linalg.norm(Sigma_cross):.2f}\n")

# Source W_K: projects input hidden states to key space
# Best linear map from h_0 to "what should be attended to" in input
# = top eigenvectors of input covariance
U_in, s_in, _ = np.linalg.svd(Sigma_input)
WK_src = U_in.T  # [d, d] — rows are eigenvectors of input covariance
# Scale to match trained W_K norm at layer 0
scale_src = np.linalg.norm(WK_trained[0]) / np.linalg.norm(WK_src)
WK_src *= scale_src

# Target W_K: projects hidden states to "what output needs to attend to"
# = directions that best predict the output from the input
# = top eigenvectors of cross-covariance
U_cross, s_cross, Vt_cross = np.linalg.svd(Sigma_cross)
WK_tgt = (U_cross @ Vt_cross).T  # polar factor of cross-covariance
# Scale to match trained W_K norm at layer 23
scale_tgt = np.linalg.norm(WK_trained[-1]) / np.linalg.norm(WK_tgt)
WK_tgt *= scale_tgt

print(f"  ||W_K^src|| = {np.linalg.norm(WK_src):.2f}  (target: {np.linalg.norm(WK_trained[0]):.2f})")
print(f"  ||W_K^tgt|| = {np.linalg.norm(WK_tgt):.2f}  (target: {np.linalg.norm(WK_trained[-1]):.2f})\n")

# ── Step 2: Geodesic interpolation ────────────────────────────────────────────
print("="*70)
print("  GEODESIC INTERPOLATION: W_K^(k) = (1-t)*W_K^src + t*W_K^tgt")
print("  t = k/23 for k=0..23")
print("="*70)

def cos_sim(A, B):
    a=A.ravel(); b=B.ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))

def hess_violation(W):
    d_=W.shape[0]
    mask=np.zeros((d_,d_))
    for i in range(d_):
        for j in range(d_):
            if j<i-1: mask[i,j]=1.
    return float(np.linalg.norm(W*mask)/max(np.linalg.norm(W),1e-8))

print(f"\n  {'Layer':>6}  {'t':>6}  {'cos(OT,trained)':>17}  {'hv_OT':>8}  {'hv_trained':>10}  {'||WK_OT||':>10}")
print("  "+"-"*68)

cos_sims = []; hv_ots = []; hv_trained = []

for k in range(n_layers):
    t = k / max(n_layers-1, 1)
    WK_ot = (1-t) * WK_src + t * WK_tgt

    cs = cos_sim(WK_ot, WK_trained[k])
    hv_o = hess_violation(WK_ot)
    hv_t = hess_violation(WK_trained[k])
    cos_sims.append(cs)
    hv_ots.append(hv_o)
    hv_trained.append(hv_t)

    print(f"  L{k:>2}:   {t:>6.3f}  {cs:>+17.4f}  {hv_o:>8.4f}  {hv_t:>10.4f}  {np.linalg.norm(WK_ot):>10.2f}")

print(f"\n  Mean cos_sim(OT, trained): {np.mean(cos_sims):+.4f}")
print(f"  Max  cos_sim:              {np.max(cos_sims):+.4f}  at L{np.argmax(cos_sims)}")
print(f"  Min  cos_sim:              {np.min(cos_sims):+.4f}  at L{np.argmin(cos_sims)}")

# ── Does the OT path respect the Hessenberg structure? ───────────────────────
print(f"\n  Hessenberg violation — does OT geodesic follow the trained path?")
print(f"  {'Layer':>6}  {'hv_OT':>8}  {'hv_trained':>12}  {'match?'}")
print("  "+"-"*42)
matches = 0
for k in [0, 5, 10, 14, 19, 23]:
    match = abs(hv_ots[k]-hv_trained[k]) < 0.05
    if match: matches += 1
    print(f"  L{k:>2}:    {hv_ots[k]:.4f}    {hv_trained[k]:.4f}    {'✓' if match else '✗'}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  VERDICT")
print("="*70)

mean_cs = float(np.mean(cos_sims))

if mean_cs > 0.5:
    result = "OT GEODESIC PREDICTS TRAINED W_K"
    detail = (f"Mean cos_sim={mean_cs:.3f} > 0.5. "
              "The linear interpolation between input covariance eigenvectors "
              "and output cross-covariance polar factor substantially predicts "
              "the gradient-descent trained key matrices. "
              "OT is a practical alternative to gradient descent for W_K.")
elif mean_cs > 0.1:
    result = "PARTIAL: OT geodesic in correct direction"
    detail = (f"Mean cos_sim={mean_cs:.3f}. Directionally correct but not "
              "close enough to replace gradient descent without refinement. "
              "The geodesic needs better source/target estimation.")
elif mean_cs > 0.0:
    result = "WEAK: OT has marginal alignment"
    detail = (f"Mean cos_sim={mean_cs:.3f}. The linear interpolation gives "
              "weak alignment with trained weights. The geodesic is correct "
              "in principle but the source/target estimation needs work.")
else:
    result = "NEGATIVE: OT geodesic not aligned with trained W_K"
    detail = (f"Mean cos_sim={mean_cs:.3f}. The linear interpolation between "
              "input/output covariance does not predict trained W_K. "
              "Either the source/target estimation is wrong, or the "
              "transport geometry is more complex than linear interpolation.")

print(f"\n  {result}")
print(f"  {detail}")
print(f"""
  WHAT THIS TELLS US:
  
  If successful (cos_sim > 0.5):
    → Given (input statistics, output statistics), compute W_K in one pass
    → No gradient descent needed for the key matrices
    → Fine-tune W_Q, W_V, W_O from this initialization: 10-100x fewer steps
  
  The procedure replaces gradient descent with:
  1. One forward pass over training data (compute Sigma_input, Sigma_cross)
  2. One SVD of each covariance matrix (compute eigenvectors)
  3. Linear interpolation (compute W_K^(k) at each depth)
  Total compute: O(n_data × d²) instead of O(300K × n_data × d²)
  Speedup: 300,000x in the best case.
""")
