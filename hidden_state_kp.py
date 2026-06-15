#!/usr/bin/env python3
"""
Hidden State Transition k_p Test
==================================
The correct Toda lattice T is the hidden state transition operator:

  T_k = H_{k+1} @ pinv(H_k)

where H_k ∈ R^{d x n_tokens} is the matrix of hidden states at layer k.

This T_k is NOT row-stochastic. Its eigenvalues can be complex.
Its char poly mod p can give k_p = 2 (binary collapse).

Conservation law: k_p(T_k) should be constant across layers k
for coherent (non-hallucinating) trajectories.

Hallucination detection: k_p transitions at specific layers
for fabricated vs factual text.

Usage: python hidden_state_kp.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.stats import ttest_ind
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--dim', type=int, default=8,
                    help='Reduced dimension for T_k (use leading SVD components)')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  HIDDEN STATE TRANSITION k_p TEST")
print(f"  T_k = H_{{k+1}} @ pinv(H_k)  — actual Toda lattice operator")
print(f"  k_p(T_k) should be conserved for coherent trajectories")
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
cfg = GPT2LMHeadModel.config_class.from_pretrained(args.model)
cfg.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained(args.model, config=cfg)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d   = model.config.n_embd
n_L = model.config.n_layer
m   = args.dim   # reduced dimension
print(f"  d={d}  layers={n_L}  reduced_dim={m}\n")

PRIMES = [2, 3, 5, 7]

def k_p_matrix(M, p):
    """
    Compute Frobenius orbit size k_p for a 2x2 matrix M mod p.
    Uses the discriminant of the characteristic polynomial.
    For larger matrices: use the 2x2 leading minor.
    """
    M2 = M[:2, :2]
    tr  = float(M2[0,0] + M2[1,1])
    det = float(M2[0,0]*M2[1,1] - M2[0,1]*M2[1,0])
    # Round to nearest integer, reduce mod p
    tr_int  = round(tr)
    det_int = round(det)
    disc = (tr_int**2 - 4*det_int) % p
    qr = {(x*x) % p for x in range(p)}
    if disc == 0:
        return 1
    elif disc in qr:
        return 1
    else:
        return 2

def compute_transition_kp(texts, label):
    """
    For each text, extract hidden states H_k, compute T_k = H_{k+1} @ pinv(H_k),
    project to m-dim subspace, compute k_p of leading 2x2 block.
    Returns: kp_per_layer [n_L, n_texts_processed], disc_per_layer same shape
    """
    kp_layers   = [[] for _ in range(n_L)]
    disc_layers  = [[] for _ in range(n_L)]
    eig_layers   = [[] for _ in range(n_L)]  # eigenvalue imaginary part

    for text in texts:
        ids = tok.encode(text, return_tensors='pt', max_length=48, truncation=True)
        if ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)

        hidden = out.hidden_states   # tuple of (n_L+1) tensors [1, seq, d]

        for k in range(n_L):
            H_k   = hidden[k][0].numpy().T    # [d, seq]
            H_kp1 = hidden[k+1][0].numpy().T  # [d, seq]

            # Project to m leading singular directions of H_k
            try:
                U, s, Vt = np.linalg.svd(H_k, full_matrices=False)
                U_m = U[:, :m]   # [d, m] — leading m directions
                # Project transition to m-dim subspace
                # T_small = U_m^T H_{k+1} Vt[:m].T @ diag(1/s[:m])
                s_m   = s[:m]
                Vt_m  = Vt[:m, :]   # [m, seq]
                # T_k in m-dim: U_m^T T_k U_m ≈ U_m^T H_{k+1} pinv(H_k) U_m
                # pinv(H_k) ≈ Vt_m^T diag(1/s_m) U_m^T
                H_kp1_proj = U_m.T @ H_kp1   # [m, seq]
                H_k_proj   = U_m.T @ H_k     # [m, seq]
                # T_small = H_kp1_proj @ pinv(H_k_proj)
                T_small, _, _, _ = np.linalg.lstsq(H_k_proj.T, H_kp1_proj.T, rcond=None)
                T_small = T_small.T   # [m, m]

                # k_p from 2x2 block
                for p in PRIMES:
                    kp = k_p_matrix(T_small, p)
                    kp_layers[k].append(kp)

                # Discriminant from 2x2 block (continuous signal)
                M2  = T_small[:2, :2]
                tr  = float(M2[0,0] + M2[1,1])
                det = float(M2[0,0]*M2[1,1] - M2[0,1]*M2[1,0])
                disc_layers[k].append(tr**2 - 4*det)

                # Eigenvalues of full T_small
                ev = np.linalg.eigvals(T_small)
                max_imag = float(np.max(np.abs(ev.imag)))
                eig_layers[k].append(max_imag)

            except Exception:
                for p in PRIMES:
                    kp_layers[k].append(1)
                disc_layers[k].append(1.0)
                eig_layers[k].append(0.0)

    return kp_layers, disc_layers, eig_layers

print(f"Computing T_k for FACTUAL texts...", flush=True)
kp_f, disc_f, eig_f = compute_transition_kp(FACTUAL, "factual")
print(f"  Done ({len([x for x in kp_f[0] if x])} texts)")

print(f"Computing T_k for FABRICATED texts...", flush=True)
kp_b, disc_b, eig_b = compute_transition_kp(FABRICATED, "fabricated")
print(f"  Done\n")

# ── Results ───────────────────────────────────────────────────────────────────
print("="*65)
print("  PER-LAYER: T_k eigenvalue imaginary part + discriminant")
print("  max_imag(T_k) > 0: complex eigenvalues present")
print("  disc < 0: leading 2x2 block has complex eigenvalues → k_p=2")
print("="*65)
print(f"\n  {'Layer':>6}  {'eig_imag_f':>12}  {'eig_imag_b':>12}  "
      f"{'disc_f':>9}  {'disc_b':>9}  {'diff_disc':>10}  {'p_val':>7}")
print("  "+"-"*78)

EXC = {2,17,18,20,21}
sig_layers = []
for k in range(n_L):
    if not disc_f[k] or not disc_b[k]: continue
    mf  = float(np.mean(disc_f[k]))
    mb  = float(np.mean(disc_b[k]))
    eif = float(np.mean(eig_f[k]))
    eib = float(np.mean(eig_b[k]))
    diff = mf - mb
    if len(disc_f[k])>1 and len(disc_b[k])>1:
        _, pv = ttest_ind(disc_f[k], disc_b[k])
    else:
        pv = 1.0
    flag = " *" if pv<0.05 else (" ." if pv<0.1 else "")
    st   = "WALL" if k in EXC else ("L14" if k==14 else "")
    if pv<0.1: sig_layers.append(k)
    print(f"  L{k:>2}:{st:>5}  {eif:>12.4f}  {eib:>12.4f}  "
          f"{mf:>+9.4f}  {mb:>+9.4f}  {diff:>+10.4f}  {pv:>7.4f}{flag}")

# ── Conservation: k_p across layers ──────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  k_p CONSERVATION (p=5, leading 2x2 of T_k)")
print("="*65)

kp5_f = [int(np.round(np.mean([kp_f[k][i*len(PRIMES)+PRIMES.index(5)]
          if len(kp_f[k]) > i*len(PRIMES)+PRIMES.index(5) else 1
          for i in range(len(FACTUAL))])))
         for k in range(n_L)]

# Actually kp_layers stores one entry per (text, prime) — need to reshape
# Each text contributes len(PRIMES) values to kp_layers[k]
kp5_fact = [np.mean([kp_f[k][i] for i in range(0, len(kp_f[k]), len(PRIMES)) if i < len(kp_f[k])])
             if kp_f[k] else 1.0 for k in range(n_L)]
kp5_fab  = [np.mean([kp_b[k][i] for i in range(0, len(kp_b[k]), len(PRIMES)) if i < len(kp_b[k])])
             if kp_b[k] else 1.0 for k in range(n_L)]

print(f"\n  Layer   kp5_fact  kp5_fab  same?  disc_f_sign  disc_b_sign")
print("  "+"-"*58)
for k in range(n_L):
    st = "WALL" if k in EXC else ("L14" if k==14 else "")
    df_sign = "+" if disc_f[k] and np.mean(disc_f[k])>0 else "-"
    db_sign = "+" if disc_b[k] and np.mean(disc_b[k])>0 else "-"
    same = "=" if abs(kp5_fact[k]-kp5_fab[k])<0.1 else "DIFF"
    print(f"  L{k:>2}{st:>5}   {kp5_fact[k]:>8.2f}  {kp5_fab[k]:>7.2f}  {same:>5}  "
          f"{df_sign:>11}  {db_sign:>11}")

print(f"\n  Significant layers (disc, p<0.1): {sig_layers}")
print(f"""
  WHAT max_imag(T_k) MEASURES:
  The hidden state transition T_k has complex eigenvalues when
  the trajectory is "spiraling" rather than "converging."
  
  Factual text: T_k should have small imaginary parts (real convergence)
  Fabricated:   T_k may have larger imaginary parts (oscillatory, complex)
  
  If eig_imag_fab >> eig_imag_fact at specific layers:
  → Those layers detect the fabrication
  → The imaginary eigenvalue magnitude IS the Toda non-isospectrality signal
""")
