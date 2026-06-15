#!/usr/bin/env python3
"""
Analytical Attractor Test
==========================
Core question: can we compute the Hessenberg attractor W_K^*
analytically from corpus statistics, bypassing gradient descent?

THEORY:
  W_K maps hidden states h ∈ R^d to keys K ∈ R^d.
  It is trained on a corpus. After training it is near-Hessenberg.
  
  The Hessenberg form is determined by the dominant co-occurrence
  structure in the EMBEDDING SPACE, not the token space.
  
  The correct object is:
    A = E^T C E  ∈ R^{d × d}
  where:
    E ∈ R^{vocab × d} = token embedding matrix
    C ∈ R^{vocab × vocab} = bigram co-occurrence matrix
  
  A captures which DIRECTIONS in embedding space co-occur.
  W_K captures which directions to attend to.
  
  CLAIM: W_K^* ≈ hessenberg(E^T C E)
  
  Proxy when C is approximately uniform (diverse language):
    E^T C E ≈ const × E^T E  (the embedding Gram matrix)
  
  So Test 1: cos_sim(hessenberg(E^T E), W_K^(k)) for all k
     Test 2: cos_sim(hessenberg(E^T C E), W_K^(k)) with real corpus

WHAT THIS MEANS FOR TRAINING:
  If Test 1 high: given trained embeddings, read off W_K analytically.
    Train loop: (1) train embeddings, (2) compute W_K = hessenberg(E^T E)
    Eliminates gradient descent on W_K entirely.
    
  If Test 2 >> Test 1: corpus statistics add information beyond E.
    Need one pass through corpus, then compute W_K analytically.
    Still no gradient descent on W_K.
    
  If both fail: W_K requires the full attention training signal.
    The attractor depends on higher-order corpus statistics
    not captured by bigrams or the embedding Gram matrix.

DATASET DECISION:
  Test 1: GPT-2 embedding matrix (already available, no corpus needed)
  Test 2: GPT-2 embedding + our 81K scientific token corpus
          (small but better than nothing — establishes the method)
  
  We use GPT-2's OWN embedding matrix to test whether W_K can be
  predicted from E. This is the cleanest possible test because:
  - E and W_K were trained on the same corpus (WebText)
  - Any correlation between hessenberg(E^T E) and W_K is genuine
  - No confounds from different corpus distributions

Usage:
    python compute_attractor.py --model gpt2-medium
"""

import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.linalg import hessenberg as scipy_hessenberg
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--top_k', type=int, default=64,
                    help='Top-k SVD modes for E^T C E approximation')
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"  ANALYTICAL ATTRACTOR TEST")
print(f"  Model: {args.model}")
print(f"{'='*70}\n")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
model.eval()
d     = model.config.n_embd          # 1024
vocab = model.config.vocab_size       # 50257
n_layers = model.config.n_layer       # 24

# Embedding matrix E ∈ R^{vocab × d}
E = model.transformer.wte.weight.detach().cpu().numpy()  # [vocab, d]
print(f"  E shape: {E.shape}  (vocab × d)")

# Trained W_K matrices for all layers
def get_WK(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    return W[:, d:2*d]   # [d, d]

WK_all = [get_WK(l) for l in range(n_layers)]
print(f"  W_K shape per layer: {WK_all[0].shape}\n")

# ── Baseline: random matrix ───────────────────────────────────────────────────
np.random.seed(42)
W_random = np.random.randn(d, d) * 0.02

def cos_sim_matrices(A, B):
    """Cosine similarity between two flattened matrices."""
    a = A.ravel(); b = B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def hess_violation(M):
    d_ = M.shape[0]
    mask = np.zeros((d_,d_))
    for i in range(d_):
        for j in range(d_):
            if j < i-1: mask[i,j]=1.
    return float(np.linalg.norm(M*mask)/max(np.linalg.norm(M),1e-8))

# ── TEST 1: Embedding Gram matrix ────────────────────────────────────────────
print("="*70)
print("  TEST 1: hessenberg(E^T E) vs trained W_K")
print("  (E^T E is the proxy for E^T C E with uniform C)")
print("="*70)

# Gram matrix: A = E^T E / ||E^T E||  [d × d]
# E is [vocab, d], so E^T E is [d, d]
# Memory: d×d = 1024×1024 = 4MB — fine
print("\nComputing E^T E...", flush=True)
A_gram = E.T @ E   # [d, d]
A_gram_norm = A_gram / np.linalg.norm(A_gram)

print(f"  A_gram shape: {A_gram.shape}")
print(f"  ||A_gram||: {np.linalg.norm(A_gram):.2f}")
print(f"  Symmetric: ||A-A^T|| = {np.linalg.norm(A_gram-A_gram.T):.2f}  "
      f"(E^T E is always symmetric)")
print(f"  Hessenberg violation of A_gram: {hess_violation(A_gram_norm):.4f}")

# Hessenberg form
print("\nComputing hessenberg(E^T E)...", flush=True)
H_gram, _ = scipy_hessenberg(A_gram_norm, calc_q=True)
print(f"  hessenberg violation of H_gram: {hess_violation(H_gram):.6f}  (should be ~0)")

# Scale to match trained W_K norms
WK_norms = [np.linalg.norm(WK_all[l]) for l in range(n_layers)]
scale_to_l14 = WK_norms[14] / max(np.linalg.norm(H_gram), 1e-8)
H_gram_scaled = H_gram * scale_to_l14

print(f"\n  cos_sim(H_gram, W_K^(k)) for all layers:")
print(f"  {'Layer':>6}  {'cos_sim':>10}  {'vs_random':>10}  {'WK_norm':>10}  {'hess_viol':>10}")
print("  " + "-"*52)

cos_sims_gram = []
cos_random = []
for l in range(n_layers):
    WK = WK_all[l]
    cs_gram  = cos_sim_matrices(H_gram_scaled, WK)
    cs_rand  = cos_sim_matrices(W_random, WK)
    cos_sims_gram.append(cs_gram)
    cos_random.append(cs_rand)
    viol = hess_violation(WK)
    if l in [0,5,9,14,19,23] or abs(cs_gram) > 0.1:
        print(f"  L{l:>2}:    {cs_gram:>+10.4f}  {cs_rand:>+10.4f}  "
              f"{WK_norms[l]:>10.3f}  {viol:>10.4f}")

r_gram, p_gram = spearmanr(range(n_layers), cos_sims_gram)
print(f"\n  Mean cos_sim: {np.mean(cos_sims_gram):+.4f}")
print(f"  Max  cos_sim: {np.max(np.abs(cos_sims_gram)):+.4f}  at L{np.argmax(np.abs(cos_sims_gram))}")
print(f"  Spearman r(cos vs depth): {r_gram:+.3f}  p={p_gram:.4f}")
print(f"  Mean random baseline: {np.mean(np.abs(cos_random)):.4f}")

# ── TEST 1b: Asymmetric version (E^T C E with uniform C) ─────────────────────
# When C = I (identity, each token predicts itself), E^T C E = E^T E
# When C is the empirical transition, E^T C E ≠ E^T E
# Test asymmetric version: use raw (non-symmetric) E^T E approximation
print(f"\n{'='*70}")
print(f"  TEST 1b: hessenberg(E^T (E shifted)) — asymmetric proxy")
print(f"  Approximates E^T C E where C = bigram matrix")
print(f"  Method: shift E to simulate next-token prediction")
print("="*70)

# Approximate C as the empirical transition:
# If token i is followed by token j, C[i,j] = 1
# E^T C E = sum_{(i,j) in bigrams} e_i^T e_j  (rank-1 updates)
# Approximate: use the shift E[1:] → E[:-1] correlation
# From our 81K corpus
print("\nUsing 81K scientific corpus for bigram statistics...")
import json
try:
    with open('/tmp/train_ids.json') as f: train_ids = json.load(f)
    with open('/tmp/vocab.json') as f: vocab_list = json.load(f)
    
    # Build co-occurrence in EMBEDDING space using corpus
    # For each consecutive pair (i, j) in corpus:
    #   accumulate e_vocab[i]^T e_vocab[j] outer product
    # But our corpus uses word IDs, not GPT-2 token IDs.
    # Map: use the GPT-2 embedding for the CONCEPT, not the token.
    
    # Simpler: for each position t, accumulate
    # A += E[tok[t]]^T ⊗ E[tok[t+1]]  in the d-dim space
    # This is E_left^T @ E_right where E_left, E_right are
    # the left and right context embeddings
    
    # Our corpus vocab is small (1017 words, not 50257 GPT-2 tokens)
    # Use GPT-2 tokenizer to find GPT-2 token IDs for our words
    tok_gpt2 = GPT2Tokenizer.from_pretrained(args.model)
    
    # Map our word vocab to GPT-2 token IDs
    word_to_gpt2 = {}
    for word in vocab_list[:200]:  # top-200 most common words
        gpt2_ids = tok_gpt2.encode(' ' + word, add_special_tokens=False)
        if gpt2_ids:
            word_to_gpt2[word] = gpt2_ids[0]  # first subword token
    
    print(f"  Mapped {len(word_to_gpt2)} words to GPT-2 token IDs")
    
    # Build E^T C E in d-dim space
    A_corpus = np.zeros((d, d))
    n_pairs = 0
    for t in range(len(train_ids)-1):
        w_left  = vocab_list[train_ids[t]]   if train_ids[t] < len(vocab_list) else None
        w_right = vocab_list[train_ids[t+1]] if train_ids[t+1] < len(vocab_list) else None
        if w_left in word_to_gpt2 and w_right in word_to_gpt2:
            il = word_to_gpt2[w_left]
            ir = word_to_gpt2[w_right]
            # Outer product of embeddings: e_l^T ⊗ e_r
            el = E[il]; er = E[ir]
            A_corpus += np.outer(el, er)
            n_pairs += 1
    
    print(f"  Used {n_pairs:,} bigram pairs")
    
    if n_pairs > 100:
        A_corpus /= n_pairs  # normalize
        A_corpus_norm = A_corpus / max(np.linalg.norm(A_corpus), 1e-8)
        H_corpus, _ = scipy_hessenberg(A_corpus_norm, calc_q=True)
        scale_corpus = WK_norms[14] / max(np.linalg.norm(H_corpus), 1e-8)
        H_corpus_scaled = H_corpus * scale_corpus
        
        cos_sims_corpus = []
        for l in range(n_layers):
            cs = cos_sim_matrices(H_corpus_scaled, WK_all[l])
            cos_sims_corpus.append(cs)
        
        print(f"\n  cos_sim(H_corpus, W_K^(k)):")
        for l in [0,5,9,14,19,23]:
            print(f"  L{l:>2}: {cos_sims_corpus[l]:+.4f}  "
                  f"(vs gram: {cos_sims_gram[l]:+.4f})")
        print(f"  Mean: {np.mean(cos_sims_corpus):+.4f}")
    else:
        print("  Too few bigram pairs. Skipping corpus test.")
        cos_sims_corpus = cos_sims_gram  # fallback

except Exception as e:
    print(f"  Corpus test skipped: {e}")
    cos_sims_corpus = cos_sims_gram

# ── TEST 2: Singular value structure ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEST 2: Do singular vectors of E^T E align with W_K eigenvectors?")
print(f"  (finer test: not full matrix similarity, but subspace alignment)")
print("="*70)

# SVD of E^T E
U_gram, s_gram, _ = np.linalg.svd(A_gram)
# Top-k left singular vectors of E^T E
k = 64
U_top = U_gram[:, :k]  # [d, k]

# For each W_K^(k), what fraction of its energy lies in top-k subspace of E^T E?
print(f"\n  Fraction of W_K energy in top-{k} subspace of E^T E:")
print(f"  {'Layer':>6}  {'energy_frac':>12}  {'random_frac':>12}")
print("  " + "-"*35)

# Random baseline
W_rand_frac = np.linalg.norm(U_top.T @ W_random) ** 2 / np.linalg.norm(W_random) ** 2

for l in range(n_layers):
    WK = WK_all[l]
    # Project W_K onto top-k subspace
    proj = U_top.T @ WK  # [k, d]
    frac = np.linalg.norm(proj)**2 / np.linalg.norm(WK)**2
    if l in [0, 5, 9, 14, 19, 23]:
        print(f"  L{l:>2}:    {frac:>12.4f}  {W_rand_frac:>12.4f}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY: CAN WE BYPASS GRADIENT DESCENT ON W_K?")
print(f"{'='*70}\n")

mean_cos = np.mean(np.abs(cos_sims_gram))
max_cos  = np.max(np.abs(cos_sims_gram))

if max_cos > 0.5:
    result = "STRONG SIGNAL: E^T E predicts W_K structure"
    implication = ("The embedding Gram matrix captures significant W_K structure. "
                   "Training W_K separately from E is partially redundant. "
                   "Method: train E for ~1K steps, then set W_K = hessenberg(E^T E). "
                   "Fine-tune for ~10K steps instead of 300K.")
elif max_cos > 0.2:
    result = "WEAK SIGNAL: E^T E has some alignment with W_K"
    implication = ("Partial alignment. Hessenberg(E^T E) gives a better starting "
                   "point than random for W_K initialization. "
                   "Reduces training steps but does not eliminate gradient descent.")
elif max_cos > 0.05:
    result = "MARGINAL: above random but small"
    implication = ("The embedding Gram matrix captures little W_K structure. "
                   "The attractor depends on higher-order corpus statistics "
                   "or the full attention training signal.")
else:
    result = "NO SIGNAL: hessenberg(E^T E) is not aligned with W_K"
    implication = ("W_K is independently determined by the attention training signal. "
                   "Cannot bypass gradient descent on W_K from E alone. "
                   "The training question requires a different approach.")

print(f"  {result}")
print(f"  {implication}")
print(f"\n  Key numbers:")
print(f"    Mean |cos_sim| hessenberg(E^T E) vs W_K: {mean_cos:.4f}")
print(f"    Max  |cos_sim|:                           {max_cos:.4f}")
print(f"    Random baseline:                          {np.mean(np.abs(cos_random)):.4f}")
print(f"    Signal/noise ratio:                       {mean_cos/max(np.mean(np.abs(cos_random)),1e-6):.1f}x")
