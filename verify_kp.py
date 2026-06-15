#!/usr/bin/env python3
"""
k_p Verification on Runtime Attention Transition Matrix
=========================================================
The paper computes k_p on the RUNTIME attention transition matrix T,
not on static W_K weights.

T_{k} = mean attention pattern at layer k over a batch of text
     = E[softmax(Q K^T / sqrt(d_head))]   [seq, seq] averaged
     → reduced to 2x2 via leading minor or block structure

k_p = Frobenius orbit size of char poly of T mod p
    = 1 if char poly splits completely over F_p  (supersingular)
    = 2 if char poly has irreducible degree-2 factors (binary collapse)

The paper's claim: k_5 = 2 for binary collapse trajectories.
Conservation law: k_p should be STABLE across layers for coherent trajectories.

Usage: python verify_kp.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--n_texts', type=int, default=20)
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  k_p VERIFICATION — RUNTIME ATTENTION MATRIX T")
print(f"  Claim: k_5 conserved across layers (Toda spectral invariant)")
print(f"{'='*65}\n")

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

print("Loading model...", flush=True)
config_cls = GPT2LMHeadModel.config_class
model = GPT2LMHeadModel.from_pretrained(args.model, output_attentions=True)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d      = model.config.n_embd
n_L    = model.config.n_layer
n_H    = model.config.n_head
d_head = d // n_H
print(f"  d={d}  layers={n_L}  heads={n_H}  d_head={d_head}\n")

# ── Compute runtime attention matrices ────────────────────────────────────────
print("Computing runtime attention matrices E[A_k] over text batch...", flush=True)

# A_k[l] = mean attention over texts and heads at layer l
# Shape: [seq, seq] averaged → take the 2x2 leading block
A_mean = [np.zeros((2, 2)) for _ in range(n_L)]
A_count = 0

for text in TEXTS:
    ids = tok.encode(text, return_tensors='pt', max_length=32, truncation=True)
    if ids.shape[1] < 3: continue
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    # out.attentions: tuple of [1, n_H, seq, seq] per layer
    for l in range(n_L):
        A_l = out.attentions[l][0]         # [n_H, seq, seq]
        A_avg = A_l.mean(dim=0).numpy()    # [seq, seq] mean over heads
        # Leading 2x2 block
        A_mean[l] += A_avg[:2, :2]
    A_count += 1

for l in range(n_L):
    A_mean[l] /= A_count

print(f"  Averaged over {A_count} texts\n")

# ── Compute k_p for each layer ─────────────────────────────────────────────────
def char_poly_2x2(M):
    """Char poly of 2x2 matrix: t^2 - tr(M)*t + det(M)"""
    tr  = M[0,0] + M[1,1]
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    return tr, det   # coefficients: t^2 - tr*t + det

def k_p_2x2(tr, det, p):
    """
    Frobenius orbit size of char poly t^2 - tr*t + det mod p.
    discriminant = tr^2 - 4*det
    k_p = 1 if discriminant is 0 or quadratic residue mod p (splits)
    k_p = 2 if discriminant is non-residue mod p (irreducible)
    """
    disc = (round(tr)**2 - 4*round(det)) % p
    # Quadratic residues mod p
    qr = {(x*x) % p for x in range(p)}
    if disc % p == 0:
        return 1  # double root
    elif disc % p in qr:
        return 1  # two distinct roots in F_p
    else:
        return 2  # irreducible over F_p

# Also test on integer-rounded versions
PRIMES = [2, 3, 5, 7, 11]

print("="*65)
print("  k_p FROM RUNTIME ATTENTION MATRIX T")
print("  (leading 2x2 block of mean attention pattern)")
print("="*65)
print(f"\n  {'Layer':>6}  {'tr(A)':>8}  {'det(A)':>10}", end='')
for p in PRIMES:
    print(f"  {'k_'+str(p):>5}", end='')
print(f"  {'stratum':>10}")
print("  "+"-"*(8 + 10 + 12 + 7*len(PRIMES) + 12))

EXC_from_static = {2, 17, 18, 20, 21}

kp_all = {p: [] for p in PRIMES}
for l in range(n_L):
    M  = A_mean[l]
    tr, det = char_poly_2x2(M)
    print(f"  L{l:>2}:   {tr:>8.4f}  {det:>10.6f}", end='')
    for p in PRIMES:
        kp = k_p_2x2(tr, det, p)
        kp_all[p].append(kp)
        print(f"  {kp:>5}", end='')
    stratum = "WALL" if l in EXC_from_static else ("L14★" if l==14 else "")
    print(f"  {stratum:>10}")

# ── Conservation analysis ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  CONSERVATION ANALYSIS")
print(f"  Is k_p constant across layers? (Toda isospectral property)")
print("="*65)

print(f"\n  {'Prime':>6}  {'Values':>25}  {'Constant?':>10}  {'k_5=2?':>8}")
print("  "+"-"*55)
for p in PRIMES:
    vals = kp_all[p]
    unique = sorted(set(vals))
    is_const = len(unique) == 1
    all_2 = all(v == 2 for v in vals)
    print(f"  p={p:>2}:   {str(unique):>25}  {'YES ✓' if is_const else 'NO ✗':>10}  "
          f"{'YES ✓' if all_2 else str(vals.count(2))+'/'+str(n_L):>8}")

# ── The paper's specific prediction ──────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PAPER PREDICTION: k_5 = 2 at all skeleton depths")
print("="*65)
k5_vals = kp_all[5]
k5_layers_2 = [l for l,v in enumerate(k5_vals) if v==2]
k5_layers_1 = [l for l,v in enumerate(k5_vals) if v==1]

print(f"""
  k_5 = 2 at layers: {k5_layers_2}  ({len(k5_layers_2)}/{n_L})
  k_5 = 1 at layers: {k5_layers_1}  ({len(k5_layers_1)}/{n_L})
  
  Paper's claim: k_5 = 2 for BINARY_COLLAPSE trajectories,
                 constant across all skeleton depths.
  
  If k_5 is constant at 2: BINARY_COLLAPSE confirmed, Toda invariant preserved
  If k_5 varies:           Trajectory is not a clean BINARY_COLLAPSE type
  If k_5 = 1 everywhere:   Trajectory is SUPERSINGULAR (different class)
""")

# ── What the runtime attention tells us vs static weights ────────────────────
print("="*65)
print("  RUNTIME vs STATIC COMPARISON")
print("="*65)
print(f"""
  Static W_K test (2x2 minor of W_Q W_K^T):
    k_5 = 1 at 22/24 layers, k_5 = 2 at L2, L20
    
  Runtime attention test (mean softmax attention):
    k_5 values above
    
  If the runtime test shows k_5 = 2 consistently:
    → The static W_K test was the wrong object
    → The runtime attention IS the correct T matrix
    → k_p is a runtime invariant, not a static weight invariant
    → Conservation of k_p across layers = Toda isospectrality confirmed
    
  If the runtime test also shows mixed k_5:
    → The text being processed matters (different texts → different k_p)
    → Run with factual vs fabricated text to see if k_p separates them
    → This is the hallucination detector: k_5 transition during inference
""")
