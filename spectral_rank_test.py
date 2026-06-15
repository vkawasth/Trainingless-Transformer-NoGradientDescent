#!/usr/bin/env python3
"""
Spectral Rank Test
===================
Test whether the 24-layer hidden-state spectrum lies on a 2D manifold.

For each token t, extract a spectral feature at each layer l:
  λ_l(t) = leading eigenvalue magnitude of A_l = H_{l+1} H_l^+

Build matrix X[t, l] = λ_l(t)  (tokens × layers)

Normalize per layer (remove scaling drift).

Rank test: if σ_3 << σ_2, then 2-lambda structure exists.

Also test layer consistency of eigenvectors (are principal directions
stable across tokens, or just amplitude variation?).

Usage: python spectral_rank_test.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--dim', type=int, default=16,
                    help='Dimension for local transition operators')
parser.add_argument('--n_texts', type=int, default=40)
parser.add_argument('--n_eigen', type=int, default=4,
                    help='Number of eigenvalues to track per layer')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  SPECTRAL RANK TEST")
print(f"  X[token, layer] = λ_l(token)  ({args.n_eigen} eigenvalues per layer)")
print(f"  Rank test: σ_3 << σ_2?  → 2-lambda structure")
print(f"{'='*65}\n")

TEXTS = [
    # Factual — clear semantic structure
    "Albert Einstein was born in Ulm Germany in 1879.",
    "He developed the special theory of relativity in 1905.",
    "Einstein won the Nobel Prize in Physics in 1921.",
    "The speed of light in vacuum is 299792458 meters per second.",
    "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
    "The Earth orbits the Sun once every 365.25 days approximately.",
    "DNA is a double helix structure discovered by Watson and Crick.",
    "Paris is the capital and largest city of France in Europe.",
    "The human brain contains approximately 86 billion neurons total.",
    "Photosynthesis converts carbon dioxide and water into glucose sugar.",
    # Fabricated — inconsistent semantic content
    "Albert Einstein was born in Vienna Austria in 1950.",
    "He developed quantum teleportation theory in 1912.",
    "Einstein won the Nobel Prize in Chemistry in 1930.",
    "The speed of light in vacuum is 150000 meters per second.",
    "Water freezes at 50 degrees Celsius at standard pressure.",
    "The Earth orbits the Sun once every 200 days approximately.",
    "DNA is a single strand structure discovered by Darwin.",
    "Berlin is the capital and largest city of France in Europe.",
    "The human brain contains approximately 3 billion neurons total.",
    "Photosynthesis converts nitrogen and methane into protein molecules.",
    # Structural — mathematical/technical
    "The transformer architecture uses multi-head self-attention layers.",
    "Gradient descent minimizes a loss function over parameter space.",
    "The eigenvalues of a symmetric matrix are always real numbers.",
    "Neural networks approximate functions through composition of layers.",
    "The Fourier transform maps functions from time to frequency domain.",
    "Entropy is a measure of disorder or uncertainty in a system.",
    "The residual connection adds the input directly to layer output.",
    "Layer normalization computes mean and variance over feature dimension.",
    "The softmax function converts logits to a probability distribution.",
    "Attention weights are computed as softmax of query-key dot products.",
    # Random/incoherent
    "The purple elephant danced quietly on the crystalline mountain peak.",
    "Seven cats decided to open a restaurant serving mathematical equations.",
    "The quantum banana oscillated through seventeen parallel dimensions yesterday.",
    "Flying carpets require periodic maintenance from certified dragon mechanics.",
    "The moon is made of seventeen varieties of crystallized imagination.",
    "Purple dinosaurs invented calculus during the first Jurassic period.",
    "Submarine sandwiches orbit the planet at geosynchronous altitude daily.",
    "The number four is secretly jealous of the letter Q perpetually.",
    "Invisible dragons guard the library of untranslatable feelings always.",
    "Time flows backwards on Tuesdays only in certain zip codes.",
][:args.n_texts]

TEXT_LABELS = (
    ['factual']*10 + ['fabricated']*10 + ['structural']*10 + ['random']*10
)[:args.n_texts]

print("Loading model...", flush=True)
cfg = GPT2LMHeadModel.config_class.from_pretrained(args.model)
cfg.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained(args.model, config=cfg)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d=model.config.n_embd; n_L=model.config.n_layer; m=args.dim; ne=args.n_eigen
print(f"  d={d}  layers={n_L}  dim={m}  n_eigen={ne}\n")

# ── Step 1: Extract spectral features per (text, layer) ──────────────────────
print(f"Extracting spectral features for {len(TEXTS)} texts...", flush=True)

# X[text, layer, eigen_idx] = |λ_i(A_l)| for text t at layer l
X = np.zeros((len(TEXTS), n_L, ne))

for t_idx, text in enumerate(TEXTS):
    ids = tok.encode(text, return_tensors='pt', max_length=48, truncation=True)
    if ids.shape[1] < 4:
        continue
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # H[k]: [seq, d] hidden states at layer k
    H = [out.hidden_states[k][0].numpy() for k in range(n_L+1)]
    seq_len = H[0].shape[0]

    # Project to m-dim using SVD of H[0]
    U0 = np.linalg.svd(H[0].T, full_matrices=False)[0][:, :m]

    for l in range(n_L):
        Hl   = H[l]   @ U0   # [seq, m]
        Hlp1 = H[l+1] @ U0   # [seq, m]
        try:
            Al = np.linalg.lstsq(Hl, Hlp1, rcond=None)[0].T  # [m,m]
            ev = np.sort(np.abs(np.linalg.eigvals(Al)))[::-1]
            X[t_idx, l, :] = ev[:ne]
        except:
            pass

    if t_idx % 10 == 0:
        print(f"  text {t_idx}/{len(TEXTS)}...", flush=True)

print()

# ── Step 2: Normalize per layer ───────────────────────────────────────────────
# For each (layer, eigen_idx), subtract mean and divide by std across texts
X_norm = X.copy()
for l in range(n_L):
    for i in range(ne):
        col = X[:, l, i]
        mu = col.mean(); sigma = col.std()
        if sigma > 1e-8:
            X_norm[:, l, i] = (col - mu) / sigma
        else:
            X_norm[:, l, i] = 0.

# ── Step 3: Rank test on spectral feature matrix ──────────────────────────────
# Reshape: X_flat[text, layer*eigen] = spectral features
X_flat = X_norm.reshape(len(TEXTS), n_L * ne)   # [n_texts, n_L*ne]

U, sigma, Vt = np.linalg.svd(X_flat, full_matrices=False)

print("="*65)
print("  SINGULAR VALUE SPECTRUM OF X[text, layer×eigen]")
print("  (after per-layer normalization)")
print("  σ_3 << σ_2 → 2-lambda structure confirmed")
print("="*65)

print(f"\n  {'Index':>7}  {'σ_i':>10}  {'σ_i/σ_1':>10}  {'cumvar%':>10}")
print("  "+"-"*42)
total_var = float(np.sum(sigma**2))
cumvar = 0.
for i, s in enumerate(sigma[:15]):
    cumvar += s**2 / total_var
    ratio  = s / sigma[0]
    print(f"  σ_{i:>4}:   {s:>10.4f}  {ratio:>10.4f}  {cumvar*100:>9.1f}%")

# Key ratios
r12 = float(sigma[1]/sigma[0])
r23 = float(sigma[2]/sigma[1])
r34 = float(sigma[3]/sigma[2])
rank2_signal = r23 < 0.3   # σ_3 is less than 30% of σ_2

print(f"\n  σ_2/σ_1 = {r12:.4f}  (gap between 1st and 2nd)")
print(f"  σ_3/σ_2 = {r23:.4f}  (gap between 2nd and 3rd ← KEY)")
print(f"  σ_4/σ_3 = {r34:.4f}")
print(f"\n  2-lambda structure: {'YES ✓  (σ_3/σ_2 < 0.3)' if rank2_signal else 'NO ✗  (σ_3/σ_2 >= 0.3)'}")

# ── Step 4: Principal directions — stable across text types? ──────────────────
print(f"\n{'='*65}")
print(f"  PRINCIPAL DIRECTION STABILITY")
print(f"  Are the top-2 eigenvectors of X stable across text types?")
print("="*65)

# Project each text onto top-2 singular directions
scores = U[:, :2] * sigma[:2]   # [n_texts, 2]

# Per text-type statistics
types = ['factual', 'fabricated', 'structural', 'random']
type_indices = {tp: [i for i,l in enumerate(TEXT_LABELS) if l==tp]
                for tp in types}

print(f"\n  Text type positions in 2D spectral space (PC1, PC2):")
print(f"  {'Type':>12}  {'mean_PC1':>10}  {'mean_PC2':>10}  {'std_PC1':>10}  {'std_PC2':>10}")
print("  "+"-"*55)
type_means = {}
for tp in types:
    idx = type_indices[tp]
    if not idx: continue
    s = scores[idx]
    type_means[tp] = s.mean(0)
    print(f"  {tp:>12}  {s[:,0].mean():>+10.4f}  {s[:,1].mean():>+10.4f}  "
          f"{s[:,0].std():>10.4f}  {s[:,1].std():>10.4f}")

# Separation between factual and fabricated
if 'factual' in type_means and 'fabricated' in type_means:
    sep = np.linalg.norm(type_means['factual'] - type_means['fabricated'])
    print(f"\n  Factual vs Fabricated separation in 2D space: {sep:.4f}")

# ── Step 5: Per-eigenvalue rank test ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PER-EIGENVALUE RANK TEST")
print(f"  X_i[text, layer] = |λ_i(A_l)|  for each eigenvalue i")
print("="*65)

for i in range(ne):
    Xi = X_norm[:, :, i]   # [n_texts, n_L]
    _, si, _ = np.linalg.svd(Xi, full_matrices=False)
    r23_i = float(si[2]/max(si[1],1e-8))
    rank_i = int(np.sum(si > si[0]*0.1))
    print(f"  λ_{i}: σ_3/σ_2={r23_i:.4f}  eff_rank(>10%σ_1)={rank_i}  "
          f"{'2D ✓' if r23_i<0.3 else ''}")

# ── Step 6: Layer consistency of eigenvectors ─────────────────────────────────
print(f"\n{'='*65}")
print(f"  LAYER EIGENVECTOR CONSISTENCY")
print(f"  PC1 projection per layer — stable = only amplitude varies")
print("="*65)

# v1 = Vt[0] is the top right singular vector, shape [n_L*ne]
v1 = Vt[0].reshape(n_L, ne)   # [n_L, ne]

print(f"\n  {'Layer':>6}", end='')
for i in range(ne):
    print(f"  {'v1[l,'+str(i)+']':>10}", end='')
print(f"  {'stratum':>10}")
print("  "+"-"*55)

EXC={2,17,18,20,21}
for l in range(n_L):
    st = "WALL" if l in EXC else ("L14" if l==14 else "")
    print(f"  L{l:>2}{st:>5}", end='')
    for i in range(ne):
        print(f"  {v1[l,i]:>+10.4f}", end='')
    print()

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  VERDICT")
print("="*65)

# Effective rank from singular value decay
eff_rank = int(np.sum(sigma > sigma[0]*0.1))
var_top2  = float(np.sum(sigma[:2]**2)/total_var)

checks = [
    ("σ_3/σ_2 < 0.3 (2D structure)",  rank2_signal,    f"{r23:.4f}"),
    ("Top-2 PCs explain >60% var",    var_top2>0.6,    f"{var_top2*100:.1f}%"),
    ("Eff rank <= 4",                  eff_rank<=4,     f"eff_rank={eff_rank}"),
    ("Factual/fab separation > 0.5",
     bool('factual' in type_means and 'fabricated' in type_means and
          np.linalg.norm(type_means['factual']-type_means['fabricated'])>0.5),
     f"sep={np.linalg.norm(type_means.get('factual',np.zeros(2))-type_means.get('fabricated',np.zeros(2))):.4f}"),
]

print()
for label, passed, detail in checks:
    print(f"  {'✓' if passed else '✗'}  {label:40}  {detail}")

n_pass = sum(c[1] for c in checks)
print(f"\n  {n_pass}/4 pass.\n")

if n_pass >= 3:
    print("  2-LAMBDA STRUCTURE CONFIRMED.")
    print("  The 24-layer spectrum lies on a 2D manifold.")
    print("  Two λs suffice. The 2-layer distillation is spectrally justified.")
    print("  → Proceed: find the 2 action variables I_1, I_2.")
elif n_pass == 2:
    print("  WEAK 2D STRUCTURE.")
    print("  More than 2 degrees of freedom, but low-rank (~3-4).")
    print("  The 2-layer approximation is practical but not exact.")
else:
    print("  NO LOW-RANK SPECTRAL STRUCTURE.")
    print("  Each layer contributes independent spectral variation.")
    print("  The λ_l(t) are not consistent invariants across layers.")
    print("  The 24 layers are not 24 projections of 2 hidden λs.")
    print("  The 2-layer distillation works (cos=0.984) because of")
    print("  rapid attractor convergence, not because of spectral rank.")
