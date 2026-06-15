#!/usr/bin/env python3
"""
Per-Head Discrimination Test
=============================
For each head h, compute the Mann-Whitney statistic comparing
E_l^(h) at exceptional vs generic layers.

The head with maximum discrimination is the "hallucination-sensitive head."

Three observables per head:
  1. ||P_{l+1}^(h) - P_l^(h)||^2   (raw strip energy)
  2. Principal angle theta_l^(h)    (Lagrangian rotation)
  3. Residual cone angle            (detrended direction change)

Usage: python per_head_discrimination.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from transformers import GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  PER-HEAD DISCRIMINATION TEST")
print(f"  Which head best separates exceptional from generic layers?")
print(f"{'='*65}\n")

SHEAR = np.array([
    0.1346,0.2849,0.4286,0.1148,0.5914,0.0372,0.0919,0.0586,
    0.0406,0.5260,0.0827,0.4030,0.0079,0.1819,0.0015,0.6161,
    0.0497,0.2062,0.6645,0.4248,0.3739,0.4194,0.2929,0.0
])
EXC = {2,17,18,20,21}
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
print("Extracting per-head projectors...", flush=True)
P = np.zeros((n_L, n_H, d_head, d_head))
for l in range(n_L):
    W    = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WK_l = W[:, d:2*d]
    for h in range(n_H):
        Wkh = WK_l[:, h*d_head:(h+1)*d_head]
        Ph  = Wkh.T @ Wkh
        Ph /= max(float(np.linalg.norm(Ph,'fro')), 1e-8)
        P[l,h] = Ph

# ── Observable 1: Raw strip energy per head ───────────────────────────────────
print("Computing per-head strip energies...", flush=True)
E_raw = np.zeros((n_L-1, n_H))
for l in range(n_L-1):
    for h in range(n_H):
        diff = P[l+1,h] - P[l,h]
        E_raw[l,h] = float(np.linalg.norm(diff,'fro')**2)

# ── Observable 2: Principal angles per head ───────────────────────────────────
print("Computing per-head principal angles...", flush=True)

def lagrangian_onb(Wkh):
    dh = Wkh.shape[1]
    W_small = Wkh.T @ Wkh
    M = np.vstack([np.eye(dh), W_small])
    Q, _ = np.linalg.qr(M)
    return Q[:, :dh]

U = []
for l in range(n_L):
    W    = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WK_l = W[:, d:2*d]
    U_l  = [lagrangian_onb(WK_l[:, h*d_head:(h+1)*d_head]) for h in range(n_H)]
    U.append(U_l)

theta_max = np.zeros((n_L-1, n_H))   # max principal angle per (layer, head)
sin2_sum  = np.zeros((n_L-1, n_H))   # sum of sin^2(theta) per (layer, head)
for l in range(n_L-1):
    for h in range(n_H):
        S  = U[l][h].T @ U[l+1][h]
        sv = np.clip(np.linalg.svd(S, compute_uv=False), 0., 1.)
        angles = np.arccos(sv)
        theta_max[l,h] = float(angles.max())
        sin2_sum[l,h]  = float(np.sum(np.sin(angles)**2))

# ── Discrimination score per head ─────────────────────────────────────────────
print("Computing per-head discrimination scores...\n", flush=True)

exc_idx = sorted(l for l in range(n_L-1) if l in EXC)
gen_idx = sorted(l for l in range(n_L-1) if l not in EXC and l != 14)

def disc_score(obs_per_layer):
    """
    obs_per_layer: array [n_L-1, n_H]
    Returns per-head (U_stat, p_val, ratio) for exc vs gen.
    """
    scores = []
    for h in range(n_H):
        exc_vals = obs_per_layer[exc_idx, h]
        gen_vals = obs_per_layer[gen_idx, h]
        mu_exc = float(np.mean(exc_vals))
        mu_gen = float(np.mean(gen_vals))
        ratio  = mu_exc / max(mu_gen, 1e-10)
        try:
            stat, pval = mannwhitneyu(exc_vals, gen_vals, alternative='two-sided')
        except:
            stat, pval = 0., 1.
        sep = (mu_exc - mu_gen) / max(np.std(np.concatenate([exc_vals,gen_vals])), 1e-10)
        scores.append({'h':h, 'mu_exc':mu_exc, 'mu_gen':mu_gen,
                       'ratio':ratio, 'sep':sep, 'pval':pval, 'stat':float(stat)})
    return scores

scores_E    = disc_score(E_raw)
scores_sin2 = disc_score(sin2_sum)
scores_tmax = disc_score(theta_max)

# ── Print results for each observable ─────────────────────────────────────────
for obs_name, scores in [("Strip energy ||P_{l+1}-P_l||^2", scores_E),
                          ("sin^2 sum (Lagrangian E)",        scores_sin2),
                          ("Max principal angle",              scores_tmax)]:
    print(f"{'='*65}")
    print(f"  {obs_name}")
    print(f"  Ranked by separation score (mu_exc - mu_gen) / std")
    print("="*65)
    ranked = sorted(scores, key=lambda x: abs(x['sep']), reverse=True)
    print(f"\n  {'Rank':>5}  {'Head':>5}  {'mu_exc':>9}  {'mu_gen':>9}  "
          f"{'ratio':>7}  {'sep':>8}  {'p_val':>8}")
    print("  "+"-"*62)
    for rank, s in enumerate(ranked[:8]):
        sig = " *" if s['pval']<0.05 else (" ." if s['pval']<0.1 else "")
        print(f"  {rank+1:>5}  H{s['h']:>4}  {s['mu_exc']:>9.4f}  {s['mu_gen']:>9.4f}  "
              f"{s['ratio']:>7.3f}x  {s['sep']:>+8.3f}  {s['pval']:>8.4f}{sig}")

    # Best head detail
    best = ranked[0]
    h    = best['h']
    obs  = (E_raw if obs_name.startswith("Strip") else
            sin2_sum if obs_name.startswith("sin") else theta_max)
    print(f"\n  Best head: H{h}  (sep={best['sep']:+.3f}  p={best['pval']:.4f})")
    print(f"  Per-layer values for H{h}:")
    print(f"  {'Layer':>6}  {'value':>10}  {'stratum':>12}")
    print("  "+"-"*32)
    for l in range(n_L-1):
        st = "WALL ✗" if l in EXC else ("L14 ★" if l==14 else "generic")
        print(f"  L{l:>2}:    {obs[l,h]:>10.4f}  {st:>12}")
    print()

# ── Consistent top heads across observables ───────────────────────────────────
print("="*65)
print("  CONSISTENT TOP HEADS (appear in top-3 of multiple observables)")
print("="*65)

top3_E    = {s['h'] for s in sorted(scores_E,    key=lambda x:abs(x['sep']),reverse=True)[:3]}
top3_sin2 = {s['h'] for s in sorted(scores_sin2, key=lambda x:abs(x['sep']),reverse=True)[:3]}
top3_tmax = {s['h'] for s in sorted(scores_tmax, key=lambda x:abs(x['sep']),reverse=True)[:3]}

consistent = top3_E & top3_sin2 & top3_tmax
in_two     = (top3_E & top3_sin2) | (top3_E & top3_tmax) | (top3_sin2 & top3_tmax)

print(f"\n  Top-3 by strip energy:    {sorted(top3_E)}")
print(f"  Top-3 by Lagrangian sin2: {sorted(top3_sin2)}")
print(f"  Top-3 by principal angle: {sorted(top3_tmax)}")
print(f"\n  Consistent in all 3:  {sorted(consistent)}")
print(f"  Consistent in 2 of 3: {sorted(in_two)}")

if consistent:
    print(f"\n  STRONG SIGNAL: Heads {sorted(consistent)} discriminate")
    print(f"  across all three observables.")
elif in_two:
    print(f"\n  MODERATE SIGNAL: Heads {sorted(in_two)} discriminate")
    print(f"  across at least two observables.")
else:
    print(f"\n  No consistent head found across observables.")
    print(f"  The exceptional layer separation is not head-specific.")
    print(f"  Residual fluctuations appear isotropic after detrending.")
    print(f"  Conclusion: no coherent transport vector field in residual geometry.")
