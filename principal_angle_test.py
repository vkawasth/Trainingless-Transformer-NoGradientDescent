#!/usr/bin/env python3
"""
Principal Angle Test
=====================
theta_l^(h) = principal angles between consecutive Lagrangians
L_l^(h) and L_{l+1}^(h) in R^{2*d_head}.

Lagrangian (in head-projected coordinates):
  L_l^(h) = graph of W_small = W_K^(h)^T W_K^(h)  [d_head x d_head]
  ONB: QR of [[I]; [W_small]]  shape [2*d_head, d_head]

Principal angle: theta = arccos(sigma) where sigma = SVD of U_l^T U_{l+1}
Strip energy:    E_l = sum_h sum_i sin^2(theta_i^(h))
               = (1/2) ||Pi_{l+1} - Pi_l||_F^2  (exact Lagrangian projector dist)

Usage: python principal_angle_test.py --model gpt2-medium
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
print(f"  PRINCIPAL ANGLE TEST")
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
print(f"  d={d}  layers={n_L}  heads={n_H}  d_head={d_head}")
print(f"  Lagrangian space: R^{{2*{d_head}}} = R^{{{2*d_head}}}\n")

def lagrangian_onb(Wkh):
    """
    ONB for graph(W_small) in R^{2*d_head}.
    Wkh: [d, d_head]
    W_small = Wkh^T Wkh: [d_head, d_head]  (head self-attention kernel)
    Graph: col space of [[I]; [W_small]]  shape [2*d_head, d_head]
    Returns Q: [2*d_head, d_head] orthonormal
    """
    dh = Wkh.shape[1]
    W_small = Wkh.T @ Wkh                   # [dh, dh]
    M = np.vstack([np.eye(dh), W_small])    # [2*dh, dh]
    Q, _ = np.linalg.qr(M)
    return Q[:, :dh]                         # [2*dh, dh]

print("Computing Lagrangian ONBs for all layers and heads...", flush=True)
U = []
for l in range(n_L):
    W    = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WK_l = W[:, d:2*d]
    U_l  = [lagrangian_onb(WK_l[:, h*d_head:(h+1)*d_head]) for h in range(n_H)]
    U.append(U_l)
    if l % 6 == 0: print(f"  L{l}...", flush=True)

print("\nComputing principal angles...\n", flush=True)

E_lag        = []   # sum_h sum_i sin^2(theta_i)
theta_max_h  = []   # max over heads of max angle
theta_mean_h = []   # mean over heads of mean angle

for l in range(n_L - 1):
    sin2_sum = 0.
    t_max_list = []
    t_mean_list = []
    for h in range(n_H):
        S     = U[l][h].T @ U[l+1][h]        # [d_head, d_head]
        sv    = np.linalg.svd(S, compute_uv=False)
        sv    = np.clip(sv, 0., 1.)
        theta = np.arccos(sv)                  # [d_head]
        sin2_sum   += float(np.sum(np.sin(theta)**2))
        t_max_list.append(float(theta.max()))
        t_mean_list.append(float(theta.mean()))
    E_lag.append(sin2_sum)
    theta_max_h.append(max(t_max_list))
    theta_mean_h.append(np.mean(t_mean_list))

E_lag        = np.array(E_lag)
theta_max_h  = np.array(theta_max_h)
theta_mean_h = np.array(theta_mean_h)
s            = SHEAR[:-1]

# ── Per-layer table ───────────────────────────────────────────────────────────
print("="*65)
print("  PER-LAYER RESULTS")
print("="*65)
print(f"\n  {'Layer':>6}  {'s(l)':>8}  {'E_lag':>10}  {'theta_max°':>11}  "
      f"{'theta_mean°':>12}  {'stratum':>12}")
print("  "+"-"*72)
for l in range(n_L-1):
    st = "WALL ✗" if l in EXC else ("L14 ★" if l==14 else "generic")
    print(f"  L{l:>2}:   {s[l]:>8.4f}  {E_lag[l]:>10.4f}  "
          f"{np.degrees(theta_max_h[l]):>10.3f}°  "
          f"{np.degrees(theta_mean_h[l]):>11.3f}°  {st:>12}")

# ── Statistics for each observable ───────────────────────────────────────────
print(f"\n{'='*65}")
print("  STATISTICS")
print("="*65)

for name, obs in [("E_lag (sin^2 sum)", E_lag),
                  ("theta_max",          theta_max_h),
                  ("theta_mean",         theta_mean_h)]:
    rsp_s,  _ = spearmanr(s, obs)
    rsp_d,  _ = spearmanr(np.arange(23), obs)
    r_lin, pl = pearsonr(s, obs)
    r_sq,  pq = pearsonr(s**2, obs)

    E_exc_v = [obs[l] for l in range(23) if l in EXC]
    E_gen_v = [obs[l] for l in range(23) if l in GEN and l < 23]
    _, p_mw = mannwhitneyu(E_exc_v, E_gen_v, alternative='greater')
    rank14  = int(np.sum(obs < obs[14])) + 1

    print(f"\n  {name}:")
    print(f"    Pearson(obs, s):     {r_lin:+.4f}  p={pl:.4f}")
    print(f"    Pearson(obs, s^2):   {r_sq:+.4f}  p={pq:.4f}")
    print(f"    Spearman(obs, s):    {rsp_s:+.4f}")
    print(f"    Spearman(obs, depth):{rsp_d:+.4f}")
    print(f"    exc/gen ratio:       {np.mean(E_exc_v)/max(np.mean(E_gen_v),1e-8):.3f}x  MW p={p_mw:.4f}")
    print(f"    L14 rank (min=1):    {rank14} of 23")

# ── Four checks on best observable (E_lag) ───────────────────────────────────
rsp_s,  _ = spearmanr(s, E_lag)
rsp_d,  _ = spearmanr(np.arange(23), E_lag)
E_exc_v   = [E_lag[l] for l in range(23) if l in EXC]
E_gen_v   = [E_lag[l] for l in range(23) if l in GEN and l < 23]
_, p_mw   = mannwhitneyu(E_exc_v, E_gen_v, alternative='greater')
rank14    = int(np.sum(E_lag < E_lag[14])) + 1

checks = [
    ("Shear correlation |rho|>0.4", abs(rsp_s)>0.4,                        f"Spearman={rsp_s:+.3f}"),
    ("Depth not dominant (<0.6)",   abs(rsp_d)<0.6,                        f"Spearman(depth)={rsp_d:+.3f}"),
    ("Exceptional spike (>1.2x)",   np.mean(E_exc_v)>1.2*np.mean(E_gen_v) or p_mw<0.1,
                                                                            f"ratio={np.mean(E_exc_v)/max(np.mean(E_gen_v),1e-8):.2f}x  p={p_mw:.3f}"),
    ("L14 near minimum (rank<=6)",  rank14<=6,                             f"rank={rank14} of 23"),
]

print(f"\n{'='*65}")
print("  FOUR CHECKS (E_lag)")
print("="*65)
print()
for label, passed, detail in checks:
    print(f"  {'✓' if passed else '✗'}  {label:40}  {detail}")

n_pass = sum(c[1] for c in checks)
print(f"\n  {n_pass}/4 pass.")

if n_pass >= 3:
    print("""
  FLOER INTERPRETATION IS EMPIRICAL.
  Principal angles resolve the Lagrangian geometry.
  → Proceed to construct J and pseudoholomorphic strips.
""")
elif n_pass >= 2:
    print("""
  Partial signal. The Lagrangian geometry is partially resolved.
  The exceptional layer signal may require the full 2d x 2d
  Lagrangian projector (not the projected W_small version).
""")
else:
    print("""
  Principal angles also insufficient.
  Return to the CR solver data — the direct strip area measurement
  from those results is the only confirmed signal so far.
  The correct intersection points require explicit thimble construction.
""")
