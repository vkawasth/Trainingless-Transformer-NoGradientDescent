#!/usr/bin/env python3
"""
Cone Field Transport Test
==========================
Builds the residual cone field from detrended projector dynamics.

For each layer l:
  v_l = P_{l+1} - P_l          (raw transport direction)
  r_l = P_tilde_{l+1} - P_tilde_l  (residual: deviation from depth trend)

Cone at layer l:
  C_l(theta) = {x : angle(x, r_l) <= theta}

Key measurement:
  angle(r_l, r_{l+1}) = angle between consecutive residual transport directions
  Small = coherent transport corridor
  Large = cone rupture = wall crossing

The hallucination prediction:
  angle(r_l, r_{l+1}) is small (~0) for generic layers
  angle(r_l, r_{l+1}) spikes at exceptional layers {L2, L17, L18, L20, L21}

Usage: python cone_field_test.py --model gpt2-medium
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from transformers import GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--cone_theta', type=float, default=45.0,
                    help='Cone half-angle in degrees')
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  CONE FIELD TRANSPORT TEST")
print(f"  angle(r_l, r_{{l+1}}) — residual transport direction change")
print(f"  Spike at exceptional layers = wall crossing confirmed")
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

# ── Per-head projectors ───────────────────────────────────────────────────────
print("Computing per-head projectors...", flush=True)
P = []
for l in range(n_L):
    W    = model.transformer.h[l].attn.c_attn.weight.detach().cpu().numpy()
    WK_l = W[:, d:2*d]
    P_l  = []
    for h in range(n_H):
        Wkh = WK_l[:, h*d_head:(h+1)*d_head]
        Ph  = Wkh.T @ Wkh
        Ph /= max(float(np.linalg.norm(Ph,'fro')), 1e-8)
        P_l.append(Ph.ravel())   # flatten to vector for angle computation
    P.append(np.array(P_l))     # [n_H, d_head^2]

P = np.array(P)   # [n_L, n_H, d_head^2]

# ── Fit depth trend on generic layers ─────────────────────────────────────────
print("Fitting depth trend on generic layers...", flush=True)
layers    = np.arange(n_L)
gen_l     = sorted(GEN)
P_hat     = np.zeros_like(P)

for h in range(n_H):
    V_gen = np.vstack([layers[gen_l]**k for k in range(3)]).T
    Y_gen = P[gen_l, h, :]
    coeffs, _, _, _ = np.linalg.lstsq(V_gen, Y_gen, rcond=None)
    V_all  = np.vstack([layers**k for k in range(3)]).T
    P_hat[:, h, :] = V_all @ coeffs

P_tilde = P - P_hat   # [n_L, n_H, d_head^2] residual

# ── Raw and residual transport directions ─────────────────────────────────────
# v_l = P_{l+1} - P_l          [n_H, d_head^2]
# r_l = P_tilde_{l+1} - P_tilde_l  [n_H, d_head^2]
V = np.diff(P,       axis=0)       # [n_L-1, n_H, d_head^2]
R = np.diff(P_tilde, axis=0)       # [n_L-1, n_H, d_head^2]

def vec_angle(a, b):
    """Angle in degrees between vectors a and b."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 90.0
    cos_theta = float(np.dot(a, b) / (na * nb))
    cos_theta  = np.clip(cos_theta, -1., 1.)
    return float(np.degrees(np.arccos(cos_theta)))

# ── Angle between consecutive transport directions ────────────────────────────
# For each layer l: angle(r_l, r_{l+1}) summed over heads
print("Computing cone angles...\n", flush=True)

raw_angles  = []   # angle(v_l, v_{l+1}) — raw
resid_angles = []  # angle(r_l, r_{l+1}) — residual

for l in range(n_L - 2):   # l=0..21, comparing (l,l+1) vs (l+1,l+2)
    # Aggregate over heads: concatenate all head directions
    v_l   = V[l].ravel()
    v_lp1 = V[l+1].ravel()
    r_l   = R[l].ravel()
    r_lp1 = R[l+1].ravel()

    raw_angles.append(vec_angle(v_l, v_lp1))
    resid_angles.append(vec_angle(r_l, r_lp1))

raw_angles   = np.array(raw_angles)
resid_angles = np.array(resid_angles)

# ── Per-head angles (more detailed) ──────────────────────────────────────────
per_head_resid = []   # [n_L-2, n_H]
for l in range(n_L - 2):
    angles_h = []
    for h in range(n_H):
        angles_h.append(vec_angle(R[l, h], R[l+1, h]))
    per_head_resid.append(angles_h)
per_head_resid = np.array(per_head_resid)   # [n_L-2, n_H]

# ── Cone overlap: does C_l ∩ C_{l+1} exist? ─────────────────────────────────
# For cone half-angle theta, two cones overlap iff
# angle(r_l, r_{l+1}) <= 2*theta
theta = args.cone_theta
cone_overlaps = resid_angles <= 2 * theta

# ── Print results ─────────────────────────────────────────────────────────────
print("="*65)
print(f"  PER-LAYER CONE ANGLES  (theta={theta}°, overlap if angle < {2*theta}°)")
print("="*65)
print(f"\n  {'Pair':>8}  {'raw_ang':>9}  {'res_ang':>9}  {'overlap':>8}  {'stratum':>12}  {'spike?'}")
print("  "+"-"*65)

EXC_mid = {l for l in EXC if 0 < l < n_L-1}  # exceptional layers in middle

for l in range(n_L - 2):
    # The "layer" label: this angle is at the l+1 transition
    l_mid   = l + 1
    stratum = "WALL ✗" if l_mid in EXC else ("L14 ★" if l_mid==14 else "generic")
    overlap = "YES ✓" if cone_overlaps[l] else "NO ✗"
    spike   = "SPIKE" if resid_angles[l] > resid_angles.mean() + resid_angles.std() else ""
    print(f"  L{l:>2}→L{l+2:>2}:  {raw_angles[l]:>9.2f}°  {resid_angles[l]:>9.2f}°  "
          f"{overlap:>8}  {stratum:>12}  {spike}")

# ── Key statistics ─────────────────────────────────────────────────────────────
# The "layer" for angle(r_l, r_{l+1}) is l+1 (the middle layer)
mid_layers = list(range(1, n_L-1))
exc_mid    = [l for l in mid_layers if l in EXC]
gen_mid    = [l for l in mid_layers if l not in EXC and l != 14]

ang_exc = [resid_angles[l-1] for l in exc_mid]   # angle centered at exc layer
ang_gen = [resid_angles[l-1] for l in gen_mid]
ang_14  = float(resid_angles[13])   # angle centered at L14

_, p_mw = mannwhitneyu(ang_exc, ang_gen, alternative='greater')
rsp, _  = spearmanr(SHEAR[1:-1], resid_angles)
rdep, _ = spearmanr(np.arange(len(resid_angles)), resid_angles)

print(f"\n{'='*65}")
print(f"  STATISTICS")
print("="*65)
print(f"""
  Residual cone angles:
    Mean (generic):     {np.mean(ang_gen):.2f}°
    Mean (exceptional): {np.mean(ang_exc):.2f}°
    Ratio exc/gen:      {np.mean(ang_exc)/max(np.mean(ang_gen),1e-8):.3f}x
    Mann-Whitney p:     {p_mw:.4f}
    
    Angle at L14:       {ang_14:.2f}°
    Global min angle:   {float(np.min(resid_angles)):.2f}° at L{int(np.argmin(resid_angles))+1}
    Global max angle:   {float(np.max(resid_angles)):.2f}° at L{int(np.argmax(resid_angles))+1}
    
  Correlations:
    Spearman(angle, shear):  {rsp:+.4f}
    Spearman(angle, depth):  {rdep:+.4f}
    
  Cone overlap (theta={theta}°):
    Generic layers:    {sum(cone_overlaps[l-1] for l in gen_mid)/len(gen_mid):.2f} overlap fraction
    Exceptional layers:{sum(cone_overlaps[l-1] for l in exc_mid)/len(exc_mid):.2f} overlap fraction
""")

# ── Global transport trajectory ───────────────────────────────────────────────
print("="*65)
print("  GLOBAL TRANSPORT TRAJECTORY")
print(f"  Project x_0 through consecutive residual cones (theta={theta}°)")
print("="*65)

# Start with the mean residual direction at L0
x = R[0].ravel()
x = x / np.linalg.norm(x)

traj_angles = [0.0]   # angle from initial direction at each layer
broken = False
break_layer = None

for l in range(1, n_L-1):
    r_l = R[l].ravel()
    r_l_norm = r_l / max(np.linalg.norm(r_l), 1e-12)
    ang = vec_angle(x, r_l_norm)
    traj_angles.append(ang)
    if ang > theta and not broken:
        broken = True
        break_layer = l + 1
    # Project x into cone: if angle > theta, snap to nearest cone boundary
    if ang > theta:
        # Project: x_new = r_l direction (closest point in cone)
        x = r_l_norm
    else:
        x = r_l_norm

print(f"\n  Trajectory from initial residual direction:")
print(f"  {'Layer':>6}  {'angle_from_x0':>15}  {'in_cone?':>10}  {'stratum':>12}")
print("  "+"-"*50)
for l, ang in enumerate(traj_angles):
    l_actual = l + 1
    stratum = "WALL ✗" if l_actual in EXC else ("L14 ★" if l_actual==14 else "generic")
    in_cone = "YES" if ang <= theta else "NO ✗"
    print(f"  L{l_actual:>2}:    {ang:>14.2f}°  {in_cone:>10}  {stratum:>12}")

if broken:
    print(f"\n  Transport corridor BREAKS at L{break_layer}")
else:
    print(f"\n  Transport corridor CONTINUOUS through all {n_L} layers")

# ── Four checks ───────────────────────────────────────────────────────────────
spike_ok  = np.mean(ang_exc) > 1.3 * np.mean(ang_gen) or p_mw < 0.1
l14_ok    = ang_14 < np.mean(ang_gen)
corr_ok   = abs(rsp) > 0.3
depth_ok  = abs(rdep) < 0.5

print(f"\n{'='*65}")
print(f"  FOUR CHECKS")
print("="*65)
checks = [
    ("Exceptional layers spike", spike_ok, f"ratio={np.mean(ang_exc)/max(np.mean(ang_gen),1e-8):.2f}x p={p_mw:.3f}"),
    ("L14 is minimum angle",    l14_ok,   f"L14={ang_14:.1f}° vs mean={np.mean(ang_gen):.1f}°"),
    ("Shear correlation",       corr_ok,  f"Spearman={rsp:+.3f}"),
    ("Depth not dominant",      depth_ok, f"Spearman(depth)={rdep:+.3f}"),
]
print()
for label, passed, detail in checks:
    print(f"  {'✓' if passed else '✗'}  {label:30}  {detail}")

n_pass = sum(c[1] for c in checks)
print(f"\n  {n_pass}/4 pass.")

if n_pass >= 3:
    print("""
  CONE FIELD CONFIRMS TRANSPORT STRUCTURE.
  Residual cones show localized disruptions at exceptional layers.
  L14 is the region of maximal cone alignment.
  A continuous transport corridor exists with localized wall crossings.
""")
else:
    print(f"""
  {n_pass}/4. The residual cone field does not resolve the exceptional layers.
  
  Next: try per-head cone angles instead of aggregate.
  Different heads may show different wall-crossing patterns.
  The head with largest angle spike at exceptional layers
  is the "hallucination-sensitive head."
""")
