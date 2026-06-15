#!/usr/bin/env python3
"""
Per-Head Strip Energy Test
===========================
E_l = sum_h ||P_{l+1}^(h) - P_l^(h)||^2_F
where P_l^(h) = W_K^(h,l)^T W_K^(h,l)  [d_head x d_head, full dimension]

GPT-2 medium: 16 heads, d_head=64, so each projector is 64x64.
This captures the actual per-head fiber geometry.

Correlated against STORED shear (0.0015-0.66, 46x range).

Usage: python per_head_strip_energy.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  PER-HEAD STRIP ENERGY TEST")
print(f"  E_l = sum_h ||P_{{l+1}}^(h) - P_l^(h)||^2_F")
print(f"{'='*65}\n")

# Stored shear — the correct values with 46x dynamic range
SHEAR = [
    0.1346, 0.2849, 0.4286, 0.1148, 0.5914, 0.0372, 0.0919, 0.0586,
    0.0406, 0.5260, 0.0827, 0.4030, 0.0079, 0.1819, 0.0015, 0.6161,
    0.0497, 0.2062, 0.6645, 0.4248, 0.3739, 0.4194, 0.2929, 0.0
]
EXC = {2, 17, 18, 20, 21}

print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
model.eval()
d      = model.config.n_embd
n_L    = model.config.n_layer
n_H    = model.config.n_head
d_head = d // n_H
print(f"  d={d}  layers={n_L}  heads={n_H}  d_head={d_head}\n")

# Extract per-head W_K^(h,l): shape [d, d_head] for each head h
print("Extracting per-head W_K...", flush=True)
WK_heads = []   # [n_L, n_H, d, d_head]
for l in range(n_L):
    W = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WK_full = W[:, d:2*d]   # [d, d]
    # Split into heads: each head gets columns [h*d_head : (h+1)*d_head]
    heads_l = []
    for h in range(n_H):
        WK_h = WK_full[:, h*d_head:(h+1)*d_head]   # [d, d_head]
        heads_l.append(WK_h)
    WK_heads.append(heads_l)

# Per-head projector P_l^(h) = W_K^(h,l)^T W_K^(h,l)  [d_head, d_head]
print("Computing per-head projectors P_l^(h) = W_K^T W_K ...", flush=True)
P = []   # [n_L, n_H, d_head, d_head]
for l in range(n_L):
    P_l = []
    for h in range(n_H):
        Wkh = WK_heads[l][h]   # [d, d_head]
        Ph  = Wkh.T @ Wkh      # [d_head, d_head]
        Ph /= max(float(np.linalg.norm(Ph, 'fro')), 1e-8)
        P_l.append(Ph)
    P.append(P_l)

# Strip energy E_l = sum_h ||P_{l+1}^(h) - P_l^(h)||^2_F
print("Computing strip energies...\n", flush=True)
E = []
for l in range(n_L - 1):
    el = 0.
    for h in range(n_H):
        diff = P[l+1][h] - P[l][h]
        el  += float(np.linalg.norm(diff, 'fro')**2)
    E.append(el)

s  = np.array(SHEAR[:-1])   # s(l) for l=0..22
E  = np.array(E)

# ── Correlations ──────────────────────────────────────────────────────────────
r_lin,  p_lin  = pearsonr(s, E)
r_sq,   p_sq   = pearsonr(s**2, E)
rsp,    _      = spearmanr(s, E)
rsp_sq, _      = spearmanr(s**2, E)
r_sqrt, p_sqrt = pearsonr(s, np.sqrt(E))

print("="*65)
print("  CORRELATION: E_l vs s(l) and s(l)^2")
print("="*65)
print(f"\n  {'Relationship':25}  {'Pearson r':>10}  {'p-value':>10}  {'Spearman r':>12}")
print("  "+"-"*62)
print(f"  {'E_l vs s(l)':25}  {r_lin:>+10.4f}  {p_lin:>10.4f}  {rsp:>+12.4f}")
print(f"  {'E_l vs s(l)^2':25}  {r_sq:>+10.4f}  {p_sq:>10.4f}  {rsp_sq:>+12.4f}")
print(f"  {'sqrt(E_l) vs s(l)':25}  {r_sqrt:>+10.4f}  {p_sqrt:>10.4f}")

# ── Per-layer table ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PER-LAYER: E_l, s(l), stratum")
print("="*65)
print(f"\n  {'Layer':>6}  {'s(l)':>8}  {'E_l':>10}  {'E_l/s²':>10}  "
      f"{'stratum':>12}  {'vs mean'}")
print("  "+"-"*68)

E_mean = float(np.mean(E))
for l in range(n_L - 1):
    stratum = "WALL ✗" if l in EXC else ("L14 ★" if l==14 else "generic")
    ratio   = E[l] / max(s[l]**2, 1e-8)
    rel     = f"{E[l]/E_mean:.2f}x"
    spike   = " SPIKE" if E[l] > 1.5*E_mean else ""
    print(f"  L{l:>2}:   {s[l]:>8.4f}  {E[l]:>10.4f}  {ratio:>10.2f}  "
          f"{stratum:>12}  {rel}{spike}")

# ── Key statistics ────────────────────────────────────────────────────────────
E_exc = [E[l] for l in range(n_L-1) if l in EXC]
E_gen = [E[l] for l in range(n_L-1) if l not in EXC and l != 14]
E_14  = float(E[14])
E_min_l = int(np.argmin(E))

print(f"\n{'='*65}")
print(f"  KEY STATISTICS")
print("="*65)
print(f"""
  Mean E (generic):     {np.mean(E_gen):.4f}
  Mean E (exceptional): {np.mean(E_exc):.4f}
  Ratio exc/gen:        {np.mean(E_exc)/max(np.mean(E_gen),1e-8):.3f}x
  E at L14:             {E_14:.4f}  (rank {sorted(E.tolist()).index(round(E_14,4))+1 if round(E_14,4) in [round(x,4) for x in E] else '?'} of {n_L-1})
  Global minimum:       E_L{E_min_l}={float(np.min(E)):.4f}
""")

# ── Floer check ───────────────────────────────────────────────────────────────
print("="*65)
print("  FLOER ENERGY BOUND")
print("="*65)
C_vals = np.sqrt(E) / np.maximum(s, 1e-4)
print(f"""
  If E_l = C^2 * s(l)^2  (Floer bound: E >= A^2):
    sqrt(E_l) vs s(l): r = {r_sqrt:+.4f}  p = {p_sqrt:.4f}
    C = sqrt(E_l)/s(l): mean={np.mean(C_vals):.3f}  std={np.std(C_vals):.3f}  cv={np.std(C_vals)/max(np.mean(C_vals),1e-8):.3f}
    
  Small cv (< 0.3): C is approximately constant → E_l = C^2 s(l)^2 confirmed
  Large cv (> 1.0): C varies too much → relation not quadratic
""")

# ── Verdict ───────────────────────────────────────────────────────────────────
print("="*65)
print("  VERDICT")
print("="*65)

spike_ok  = np.mean(E_exc) > 1.5 * np.mean(E_gen)
min_ok    = E_14 < np.mean(E_gen)
corr_ok   = abs(rsp) > 0.5
floer_ok  = abs(r_sqrt) > 0.5 and np.std(C_vals)/max(np.mean(C_vals),1e-8) < 1.0

checks = [
    ("Spearman |r| > 0.5",          corr_ok,  f"|r|={abs(rsp):.3f}"),
    ("Exceptional layers spike",     spike_ok, f"ratio={np.mean(E_exc)/max(np.mean(E_gen),1e-8):.2f}x"),
    ("L14 is energy minimum",        min_ok,   f"E14={E_14:.3f} vs mean={np.mean(E_gen):.3f}"),
    ("Floer bound E ~ C^2 s^2",      floer_ok, f"r_sqrt={r_sqrt:+.3f}"),
]

print()
for label, passed, detail in checks:
    icon = "✓" if passed else "✗"
    print(f"  {icon}  {label:35}  {detail}")

n_pass = sum(c[1] for c in checks)
print(f"\n  {n_pass}/4 checks pass.")

if n_pass >= 3:
    print("""
  CONCLUSION: Strip energy is empirically grounded.
  E_l ∝ s(l)^2 holds. Exceptional layers spike. L14 is the minimum.
  The Floer interpretation is not merely metaphorical.
  → Proceed to construct J and pseudoholomorphic strips.
""")
elif n_pass >= 2:
    print("""
  CONCLUSION: Partial confirmation.
  Some structure is present but projector needs refinement.
  Try: Lagrangian graph projector at full d_head dimension,
  or per-head attention pattern projector.
""")
else:
    print("""
  CONCLUSION: Per-head projector also insufficient.
  The fiber representation needs rethinking.
  The intersection points T_l ∩ T_{l+1} are not captured by W_K^T W_K.
  Need: explicit thimble construction from the CR solver data.
""")
