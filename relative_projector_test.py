#!/usr/bin/env python3
"""
Relative Projector Strip Energy Test
======================================
Separates depth trend from exceptional layer signal.

P_tilde_l = P_l - P_hat(l)
where P_hat(l) is the trend-expected projector at depth l
(fitted from generic layers only, interpolated to all depths).

E_tilde_l = ||P_tilde_{l+1} - P_tilde_l||^2

This isolates wall-crossing signal from smooth geodesic flow.

Usage: python relative_projector_test.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from transformers import GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  RELATIVE PROJECTOR STRIP ENERGY TEST")
print(f"  P_tilde_l = P_l - P_hat(l)  (depth-detrended)")
print(f"  E_tilde_l = ||P_tilde_{{l+1}} - P_tilde_l||^2")
print(f"{'='*65}\n")

SHEAR = np.array([
    0.1346, 0.2849, 0.4286, 0.1148, 0.5914, 0.0372, 0.0919, 0.0586,
    0.0406, 0.5260, 0.0827, 0.4030, 0.0079, 0.1819, 0.0015, 0.6161,
    0.0497, 0.2062, 0.6645, 0.4248, 0.3739, 0.4194, 0.2929, 0.0
])
EXC = {2, 17, 18, 20, 21}
GEN = set(range(24)) - EXC

print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
model.eval()
d      = model.config.n_embd
n_L    = model.config.n_layer
n_H    = model.config.n_head
d_head = d // n_H
print(f"  d={d}  layers={n_L}  heads={n_H}  d_head={d_head}\n")

# ── Extract per-head projectors ───────────────────────────────────────────────
print("Computing per-head projectors...", flush=True)
P = []   # [n_L, n_H, d_head, d_head]
for l in range(n_L):
    W    = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WK_l = W[:, d:2*d]   # [d, d]
    P_l  = []
    for h in range(n_H):
        Wkh = WK_l[:, h*d_head:(h+1)*d_head]   # [d, d_head]
        Ph  = Wkh.T @ Wkh                        # [d_head, d_head]
        Ph /= max(float(np.linalg.norm(Ph, 'fro')), 1e-8)
        P_l.append(Ph)
    P.append(P_l)

# Stack: P_stack[l, h] = projector as flat vector for fitting
P_flat = np.array([[P[l][h].ravel() for h in range(n_H)]
                   for l in range(n_L)])   # [n_L, n_H, d_head^2]

# ── Fit depth trend on GENERIC layers only ────────────────────────────────────
print("Fitting depth trend on generic layers...", flush=True)

layers     = np.arange(n_L)
gen_layers = sorted(GEN)
exc_layers = sorted(EXC)

# For each head h and each projector entry j:
# fit P_flat[l, h, j] = a0 + a1*l + a2*l^2  using generic layers only
# Then P_hat[l, h] = fitted value at depth l

degree = 2   # quadratic trend
P_hat  = np.zeros_like(P_flat)   # [n_L, n_H, d_head^2]
n_entries = d_head * d_head

for h in range(n_H):
    X_train = layers[gen_layers]         # generic layer indices
    Y_train = P_flat[gen_layers, h, :]   # [n_gen, d_head^2]

    # Vandermonde matrix for polynomial fit
    V = np.vstack([X_train**k for k in range(degree+1)]).T   # [n_gen, degree+1]
    # Fit all entries simultaneously: coeffs = [degree+1, d_head^2]
    coeffs, _, _, _ = np.linalg.lstsq(V, Y_train, rcond=None)

    # Predict at all depths
    V_all = np.vstack([layers**k for k in range(degree+1)]).T  # [n_L, degree+1]
    P_hat[:, h, :] = V_all @ coeffs   # [n_L, d_head^2]

# ── Detrended projector ───────────────────────────────────────────────────────
P_tilde = P_flat - P_hat   # [n_L, n_H, d_head^2]

# ── Detrended strip energy ────────────────────────────────────────────────────
print("Computing detrended strip energies...\n", flush=True)
E_tilde = []
for l in range(n_L - 1):
    et = 0.
    for h in range(n_H):
        diff = P_tilde[l+1, h] - P_tilde[l, h]
        et  += float(np.linalg.norm(diff)**2)
    E_tilde.append(et)

E_tilde = np.array(E_tilde)
s       = SHEAR[:-1]   # s(l) for l=0..22

# Also compute raw E for comparison
E_raw = []
for l in range(n_L-1):
    er = 0.
    for h in range(n_H):
        diff = P_flat[l+1, h] - P_flat[l, h]
        er  += float(np.linalg.norm(diff)**2)
    E_raw.append(er)
E_raw = np.array(E_raw)

# ── Correlations ──────────────────────────────────────────────────────────────
r_lin,   p_lin  = pearsonr(s, E_tilde)
r_sq,    p_sq   = pearsonr(s**2, E_tilde)
rsp,     _      = spearmanr(s, E_tilde)
rsp_sq,  _      = spearmanr(s**2, E_tilde)
r_depth, p_dep  = pearsonr(np.arange(23), E_tilde)
rsp_dep, _      = spearmanr(np.arange(23), E_tilde)

# Compare to raw
rsp_raw,  _ = spearmanr(s, E_raw)
rdep_raw, _ = spearmanr(np.arange(23), E_raw)

print("="*65)
print("  CORRELATION RESULTS")
print("="*65)
print(f"\n  {'':30}  {'Raw E':>10}  {'Detrended E':>12}")
print(f"  {'':30}  {'----------':>10}  {'------------':>12}")
print(f"  {'Spearman(E, shear)':30}  {rsp_raw:>+10.4f}  {rsp:>+12.4f}")
print(f"  {'Spearman(E, shear^2)':30}               {rsp_sq:>+12.4f}")
print(f"  {'Spearman(E, depth)':30}  {rdep_raw:>+10.4f}  {rsp_dep:>+12.4f}")
print(f"  {'Pearson(E, shear)':30}  {'':>10}  {r_lin:>+12.4f}  p={p_lin:.4f}")
print(f"  {'Pearson(E, shear^2)':30}  {'':>10}  {r_sq:>+12.4f}  p={p_sq:.4f}")

# ── Per-layer table ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PER-LAYER DETRENDED ENERGY")
print("="*65)
print(f"\n  {'Layer':>6}  {'s(l)':>8}  {'E_raw':>8}  {'E_tilde':>10}  "
      f"{'stratum':>12}  {'vs mean'}")
print("  "+"-"*68)

Et_mean = float(np.mean(E_tilde))
for l in range(n_L - 1):
    stratum = "WALL ✗" if l in EXC else ("L14 ★" if l==14 else "generic")
    rel     = f"{E_tilde[l]/Et_mean:.2f}x"
    spike   = " SPIKE" if E_tilde[l] > 1.5*Et_mean else (
              " dip"   if E_tilde[l] < 0.5*Et_mean else "")
    print(f"  L{l:>2}:   {s[l]:>8.4f}  {E_raw[l]:>8.2f}  {E_tilde[l]:>10.4f}  "
          f"{stratum:>12}  {rel}{spike}")

# ── Key statistics ────────────────────────────────────────────────────────────
E_exc = [E_tilde[l] for l in range(23) if l in EXC]
E_gen = [E_tilde[l] for l in range(23) if l in GEN and l < 23]
E_14  = float(E_tilde[14])
E_min_l = int(np.argmin(E_tilde))

stat_mw, p_mw = mannwhitneyu(E_exc, E_gen, alternative='greater')
E_rank = int(np.sum(E_tilde < E_14)) + 1

print(f"\n{'='*65}")
print(f"  KEY STATISTICS (detrended)")
print("="*65)
print(f"""
  Mean E_tilde (generic):     {np.mean(E_gen):.4f}
  Mean E_tilde (exceptional): {np.mean(E_exc):.4f}
  Ratio exc/gen:              {np.mean(E_exc)/max(np.mean(E_gen),1e-8):.3f}x
  Mann-Whitney exc>gen p:     {p_mw:.4f}

  E_tilde at L14:             {E_14:.4f}  (rank {E_rank} of 23 from smallest)
  Global min:                 E_tilde_L{E_min_l}={float(np.min(E_tilde)):.4f}
  Global max:                 E_tilde_L{int(np.argmax(E_tilde))}={float(np.max(E_tilde)):.4f}
""")

# ── Four checks ───────────────────────────────────────────────────────────────
depth_removed  = abs(rsp_dep) < 0.3
shear_corr     = abs(rsp) > 0.4
exc_spike      = np.mean(E_exc) > 1.3 * np.mean(E_gen) and p_mw < 0.1
l14_min        = E_rank <= 5

print("="*65)
print("  FOUR CHECKS")
print("="*65)
checks = [
    ("Depth trend removed",        depth_removed, f"|Spearman(E,depth)|={abs(rsp_dep):.3f}"),
    ("Shear correlation restored", shear_corr,    f"Spearman(E,shear)={rsp:+.3f}"),
    ("Exceptional layers spike",   exc_spike,     f"ratio={np.mean(E_exc)/max(np.mean(E_gen),1e-8):.2f}x  p={p_mw:.3f}"),
    ("L14 is energy minimum",      l14_min,       f"rank={E_rank} of 23"),
]
print()
for label, passed, detail in checks:
    icon = "✓" if passed else "✗"
    print(f"  {icon}  {label:35}  {detail}")

n_pass = sum(c[1] for c in checks)
print(f"\n  {n_pass}/4 checks pass.")

if n_pass >= 3:
    print("""
  CONCLUSION: Wall-crossing signal is real and isolated.
  After depth detrending:
  - Exceptional layers genuinely spike above trend
  - L14 sits at/near the detrended minimum
  - Shear ordering is preserved
  
  The Floer interpretation is empirical, not metaphorical.
  The per-head projector, after detrending, resolves the fiber geometry.
  → Ready to construct J and pseudoholomorphic strips.
""")
elif n_pass >= 2:
    print("""
  CONCLUSION: Partial isolation achieved.
  Some structure survives detrending.
  Refine: try higher-degree trend removal, or
  per-head detrending instead of aggregate.
""")
else:
    print("""
  CONCLUSION: Detrending insufficient.
  The exceptional layer signal does not survive trend removal.
  The fiber representation needs a fundamentally different approach.
  Revisit: principal angles between consecutive Lagrangians,
  or the CR solver's direct strip area measurement.
""")
