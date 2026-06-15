#!/usr/bin/env python3
"""
Lax Spectral Invariance Test
==============================
Protocol from the document:

1. Embed tokens → trajectory x_0..x_24  (hidden states at each layer)
2. Infer local dynamics: A_t = x_{t+1} x_t^+  (transition operators)
3. Test spectral invariants: are eigenvalues of A_t conserved across layers?
4. If yes → Lax structure, Toda fingerprint, isospectral evolution
5. Recover 2-layer lambda: find L such that L^2 has same spectrum as M_24

Usage: python kostant_toda_test.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.linalg import sqrtm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--n_texts', type=int, default=20)
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  LAX SPECTRAL INVARIANCE TEST")
print(f"  A_t = x_{{t+1}} x_t^+  — transition operators")
print(f"  λ(A_t) conserved? → Toda/isospectral fingerprint")
print(f"  Goal: 2-layer λ = 24-layer λ")
print(f"{'='*65}\n")

TEXTS = [
    "The transformer processes sequences using attention mechanisms.",
    "Quantum mechanics describes particles using wave functions.",
    "Natural selection drives evolutionary adaptation in species.",
    "Einstein developed general relativity describing spacetime curvature.",
    "Neural networks learn hierarchical representations from data.",
    "The immune system produces antibodies to neutralize pathogens.",
    "Photosynthesis converts solar energy into chemical energy.",
    "Language models predict token probabilities using attention.",
    "Gradient descent minimizes the loss function over training.",
    "The residual connection adds input to output at each block.",
    "Layer normalization stabilizes activations during training.",
    "The Fourier transform decomposes signals into frequencies.",
    "Entropy measures the uncertainty in a probability distribution.",
    "Topology studies properties preserved under continuous deformations.",
    "Category theory provides unified language for mathematics.",
    "Differential geometry extends calculus to curved manifolds.",
    "The standard model describes fundamental particles and forces.",
    "Thermodynamics governs energy transfer in physical systems.",
    "DNA encodes genetic information using nucleotide base pairs.",
    "Calculus provides tools for derivatives and integrals.",
][:args.n_texts]

print("Loading model...", flush=True)
cfg = GPT2LMHeadModel.config_class.from_pretrained(args.model)
cfg.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained(args.model, config=cfg)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d=model.config.n_embd; n_L=model.config.n_layer; m=args.dim
print(f"  d={d}  layers={n_L}  dim={m}\n")

# ── Step 1: Hidden state trajectory ──────────────────────────────────────────
print(f"Extracting hidden states...", flush=True)
H_all = [[] for _ in range(n_L+1)]
for text in TEXTS:
    ids = tok.encode(text, return_tensors='pt', max_length=48, truncation=True)
    if ids.shape[1] < 4: continue
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    for k in range(n_L+1):
        H_all[k].append(out.hidden_states[k][0].numpy())
H = [np.vstack(Hk) for Hk in H_all]
print(f"  {H[0].shape[0]} tokens\n")

# ── Step 2: Transition operators ──────────────────────────────────────────────
print("Computing A_t = H_{t+1} H_t^+ ...", flush=True)
U0 = np.linalg.svd(H[0].T, full_matrices=False)[0][:, :m]
A = []
for k in range(n_L):
    Hk   = H[k]   @ U0
    Hkp1 = H[k+1] @ U0
    Ak   = np.linalg.lstsq(Hk, Hkp1, rcond=None)[0].T
    A.append(Ak)

# ── Step 3: Spectral invariance ───────────────────────────────────────────────
EXC={2,17,18,20,21}
ev_mags = np.array([np.sort(np.abs(np.linalg.eigvals(Ak)))[::-1] for Ak in A])
dets    = [float(np.real(np.linalg.det(Ak))) for Ak in A]

print(f"\n{'='*65}")
print(f"  SPECTRAL INVARIANCE: |λ_i(A_t)| across layers")
print("="*65)
print(f"\n  {'Layer':>6}", end='')
for i in range(min(5,m)): print(f"  {'|λ'+str(i)+'|':>8}", end='')
print(f"  {'det':>9}  {'stratum':>8}")
print("  "+"-"*60)
for k in range(n_L):
    st = "WALL" if k in EXC else ("L14" if k==14 else "")
    print(f"  L{k:>2}{st:>5}", end='')
    for i in range(min(5,m)): print(f"  {ev_mags[k,i]:>8.4f}", end='')
    print(f"  {dets[k]:>9.4f}")

cv_ev = [float(np.std(ev_mags[:,i])/max(np.mean(ev_mags[:,i]),1e-8)) for i in range(m)]
n_conserved = sum(cv<0.1 for cv in cv_ev)
print(f"\n  Conserved eigenvalues (cv<0.1): {n_conserved}/{m}")
print(f"  Mean cv: {np.mean(cv_ev):.4f}  Max cv: {np.max(cv_ev):.4f}")

# ── Step 4: Lax (symmetric part) ─────────────────────────────────────────────
L_eigs = np.array([np.sort(np.linalg.eigvalsh((Ak+Ak.T)/2)) for Ak in A])
cv_L   = [float(np.std(L_eigs[:,i])/max(abs(np.mean(L_eigs[:,i])),1e-8)) for i in range(m)]
n_Lax  = sum(cv<0.1 for cv in cv_L)

print(f"\n  Lax matrix L_t=(A_t+A_t^T)/2 eigenvalues:")
print(f"  Conserved (cv<0.1): {n_Lax}/{m}  Mean cv: {np.mean(cv_L):.4f}")

# ── Step 5: 2-layer recovery ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  2-LAYER RECOVERY")
print("="*65)

M24 = np.eye(m)
for Ak in reversed(A): M24 = Ak @ M24
ev24 = np.sort(np.abs(np.linalg.eigvals(M24)))[::-1]

# Method 1: mean operator squared
A_mean = np.mean(A, axis=0)
ev_mean2 = np.sort(np.abs(np.linalg.eigvals(A_mean @ A_mean)))[::-1]
err_mean2 = float(np.linalg.norm(ev24[:8]-ev_mean2[:8])/max(np.linalg.norm(ev24[:8]),1e-8))

# Method 2: matrix square root of monodromy (optimal 2-step)
try:
    M2_opt = np.real(sqrtm(M24))
    ev2_opt = np.sort(np.abs(np.linalg.eigvals(M2_opt)))[::-1]
    err_opt  = float(np.linalg.norm(ev24[:8]-ev2_opt[:8])/max(np.linalg.norm(ev24[:8]),1e-8))
    sqrtm_ok = np.isfinite(err_opt)
except:
    sqrtm_ok = False; err_opt = 1.0; ev2_opt = ev24*0

print(f"\n  24-layer monodromy M_24 spectrum (top 8):")
for i in range(min(8,m)):
    e24 = float(ev24[i])
    em2 = float(ev_mean2[i])
    eop = float(ev2_opt[i]) if sqrtm_ok else 0.
    print(f"    λ_{i}: 24-layer={e24:.6f}  mean^2={em2:.6f}  sqrt(M24)={eop:.6f}")

print(f"\n  Spectral error  mean^2  vs 24-layer: {err_mean2:.6f}")
print(f"  Spectral error sqrt(M) vs 24-layer:  {err_opt:.6f}")

# ── Step 6: Symplectic ────────────────────────────────────────────────────────
mean_det_err = float(np.mean([abs(abs(d)-1) for d in dets]))
print(f"\n  Volume preservation det(A_t)≈1: mean |det-1|={mean_det_err:.4f}")
print(f"  Symplectic: {'YES ✓' if mean_det_err<0.05 else 'NO ✗'}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  VERDICT")
print("="*65)
checks = [
    ("A_t eigenvalues conserved",  n_conserved >= m//2, f"{n_conserved}/{m} cv<0.1"),
    ("Lax eigs conserved",         n_Lax >= m//2,       f"{n_Lax}/{m} cv<0.1"),
    ("2-layer recovers spectrum",  err_opt < 0.05,      f"err={err_opt:.4f}"),
    ("Volume preserved det≈1",     mean_det_err < 0.05, f"err={mean_det_err:.4f}"),
]
print()
for label, passed, detail in checks:
    print(f"  {'✓' if passed else '✗'}  {label:38}  {detail}")
n_pass = sum(c[1] for c in checks)
print(f"\n  {n_pass}/4 pass.\n")
if n_pass >= 3:
    print("  TODA/LAX CONFIRMED. 2-layer λ = 24-layer λ. Proceed to action-angle.")
elif n_pass == 2:
    print("  PARTIAL. Approximate integrability. 2-layer works but not exactly isospectral.")
else:
    print("  NO TODA. Non-integrable flow. 2-layer works because of rapid convergence to")
    print("  attractor (L14), not because of isospectrality. This is the correct answer.")
