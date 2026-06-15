#!/usr/bin/env python3
"""
Syzygy Extraction Test
========================
Tests the claim: A_24 = Q_total^T A_0 Q_total
where Q_total = Q_0 Q_1 ... Q_23 (product of per-layer QR factors).

Also tests the extract_total_syzygy algorithm from the document.

THREE VERSIONS of Q_total tested:
  A) Q from QR(A_k) where A_k = W_Q^(k) W_K^(k)^T  (document's proposal)
  B) Q from QR(W_K^(k))  (key matrix directly)
  C) Q from polar decomposition of (W_K^(k+1) - W_K^(k)) (shear-based)

Usage:
    python syzygy_test.py --model gpt2-medium
"""

import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.linalg import hessenberg as scipy_hessenberg
from transformers import GPT2LMHeadModel, GPT2Config

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"  SYZYGY EXTRACTION TEST")
print(f"  Model: {args.model}")
print(f"  Claim: A_24 = Q_total^T A_0 Q_total")
print(f"{'='*70}\n")

print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
model.eval()
d = model.config.n_embd
n_layers = model.config.n_layer

def get_WK(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    return W[:, d:2*d]

def get_WQ(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    return W[:, :d]

def cos_sim(A, B):
    a=A.ravel(); b=B.ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))

def rel_err(A, B):
    return float(np.linalg.norm(A-B)/max(np.linalg.norm(B),1e-8))

print(f"  d={d}  n_layers={n_layers}\n")

# ── Extract A_k = W_Q^(k) W_K^(k)^T for all layers ──────────────────────────
print("Extracting A_k = W_Q^(k) W_K^(k)^T for all layers...", flush=True)
A_layers = []
for l in range(n_layers):
    WQ = get_WQ(l); WK = get_WK(l)
    A_layers.append(WQ @ WK.T)

A0  = A_layers[0]
A24 = A_layers[-1]

print(f"  ||A_0||  = {np.linalg.norm(A0):.2f}")
print(f"  ||A_24|| = {np.linalg.norm(A24):.2f}")
print(f"  Baseline cos(A_0, A_24) = {cos_sim(A0,A24):+.4f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# VERSION A: Q from QR(A_k) — document's proposal
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  VERSION A: Q_k from QR(A_k = W_Q^(k) W_K^(k)^T)")
print("="*70)
print("\n  Computing Q_total = Q_0 @ Q_1 @ ... @ Q_23...")

Q_total_A = np.eye(d)
for l in range(n_layers):
    Q_l, _ = np.linalg.qr(A_layers[l])
    Q_total_A = Q_total_A @ Q_l

A24_pred_A = Q_total_A.T @ A0 @ Q_total_A

print(f"  Q_total is orthogonal: ||Q Q^T - I|| = {np.linalg.norm(Q_total_A@Q_total_A.T - np.eye(d)):.4f}")
print(f"\n  cos_sim(Q_total^T A_0 Q_total, A_24) = {cos_sim(A24_pred_A, A24):+.4f}")
print(f"  rel_err(Q_total^T A_0 Q_total, A_24) = {rel_err(A24_pred_A, A24):.4f}")
print(f"  (baseline cos(A_0, A_24)            = {cos_sim(A0,A24):+.4f})")
print(f"  (random baseline                     ≈ 0.000)")

# Also test per-layer: does Q_k^T A_k Q_k = A_{k+1}?
print(f"\n  Per-layer test: cos_sim(Q_k^T A_k Q_k, A_{{k+1}}):")
print(f"  {'Layer':>6}  {'cos_sim':>10}  {'rel_err':>10}")
print("  " + "-"*30)
for l in range(0, n_layers-1, 4):
    Q_l, _ = np.linalg.qr(A_layers[l])
    pred = Q_l.T @ A_layers[l] @ Q_l
    print(f"  L{l:>2}→L{l+1}: {cos_sim(pred,A_layers[l+1]):>+10.4f}  {rel_err(pred,A_layers[l+1]):>10.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# VERSION B: Q from QR(W_K^(k)) directly
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  VERSION B: Q_k from QR(W_K^(k)) directly")
print("="*70)

Q_total_B = np.eye(d)
for l in range(n_layers):
    Q_l, _ = np.linalg.qr(get_WK(l))
    Q_total_B = Q_total_B @ Q_l

A24_pred_B = Q_total_B.T @ A0 @ Q_total_B
print(f"  cos_sim(Q_total^T A_0 Q_total, A_24) = {cos_sim(A24_pred_B, A24):+.4f}")
print(f"  rel_err = {rel_err(A24_pred_B, A24):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# VERSION C: Accumulated rotation from consecutive W_K differences (shear)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  VERSION C: Q from polar decomp of Delta_WK = W_K^(k+1) - W_K^(k)")
print("="*70)

Q_total_C = np.eye(d)
for l in range(n_layers-1):
    Delta = get_WK(l+1) - get_WK(l)
    # Polar decomposition: Delta = U S V^T, take Q = U V^T
    U, _, Vt = np.linalg.svd(Delta)
    Q_l = U @ Vt  # orthogonal factor of Delta
    Q_total_C = Q_total_C @ Q_l

A24_pred_C = Q_total_C.T @ A0 @ Q_total_C
print(f"  cos_sim(Q_total^T A_0 Q_total, A_24) = {cos_sim(A24_pred_C, A24):+.4f}")
print(f"  rel_err = {rel_err(A24_pred_C, A24):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# WHAT Q_total ACTUALLY ENCODES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  WHAT Q_total ACTUALLY ENCODES")
print("="*70)

# Even if Q_total^T A_0 Q_total ≠ A_24,
# Q_total encodes the total rotation accumulated through depth.
# Test: does Q_total^T W_K^(0) Q_total ≈ W_K^(23)?
WK0  = get_WK(0); WK24 = get_WK(n_layers-1)
WK_pred_A = Q_total_A.T @ WK0 @ Q_total_A
WK_pred_B = Q_total_B.T @ WK0 @ Q_total_B
print(f"\n  Does Q_total^T W_K^(0) Q_total ≈ W_K^(23)?")
print(f"  Version A: cos={cos_sim(WK_pred_A,WK24):+.4f}  rel_err={rel_err(WK_pred_A,WK24):.4f}")
print(f"  Version B: cos={cos_sim(WK_pred_B,WK24):+.4f}  rel_err={rel_err(WK_pred_B,WK24):.4f}")

# And the shear result we already confirmed:
S_total = WK24.T - WK0.T
print(f"\n  Shear result (confirmed): S_total = W_K^(23)^T - W_K^(0)^T")
print(f"  ||S_total|| = {np.linalg.norm(S_total):.2f}")
print(f"  This is the correct total geometric change (shear, not rotation)")

# ══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  VERDICT")
print("="*70)

cos_A = cos_sim(A24_pred_A, A24)
cos_B = cos_sim(A24_pred_B, A24)
cos_C = cos_sim(A24_pred_C, A24)

print(f"""
  Q_total^T A_0 Q_total vs A_24:
    Version A (QR of A_k):      cos = {cos_A:+.4f}
    Version B (QR of W_K^k):    cos = {cos_B:+.4f}  
    Version C (polar of Delta):  cos = {cos_C:+.4f}
    Baseline cos(A_0, A_24):          {cos_sim(A0,A24):+.4f}
    Random baseline:                  ≈ 0.000

  If any cos >> baseline: the syzygy Q_total partially reconstructs A_24.
  If all cos ≈ 0: Q_total is well-defined but does not relate A_0 to A_24
    in the way the document claims.
    
  The correct picture depends on these numbers.
""")
