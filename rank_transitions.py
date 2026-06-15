#!/usr/bin/env python3
"""
Rank Transition Analysis
=========================
The rank profile of δJ_l = J_l - I shows discrete jumps:

  L4→L5:   rank 8 → 9    (small uptick, early compression)
  L6→L7:   rank 8 → 12   (jump +4, mid-early)
  L18→L19: rank 12 → 9   (drop -3, entering final compression)
  L20→L21: rank 10 → 4   (DROP -6, the decisive compression)

At each transition: the functor changes the dimension of the
subspace it operates on. The SINGULAR VECTORS at the transition
tell us WHICH DIRECTIONS are being added or removed.

The operators we need to recover the trajectory:
  P_l = projection onto the rank-k subspace of δJ_l
  The transition operator T_{l→l+1} = P_{l+1} @ J_l @ P_l^†
  This is what maps the active subspace at layer l to layer l+1.

Usage: python rank_transitions.py [--model gpt2-medium]
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--proj', type=int, default=32)
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  RANK TRANSITION ANALYSIS  model={args.model}")
print(f"  Finding dimensional decision points in the functor chain")
print(f"{'='*65}\n")

model = GPT2LMHeadModel.from_pretrained(args.model)
tok   = GPT2Tokenizer.from_pretrained(args.model)
model.eval()
d   = model.config.n_embd
n_L = model.config.n_layer
n_H = model.config.n_head

# Known transitions from functor dissection data
TRANSITIONS = {
    (4, 5):   ("8→9",   "small uptick, early compression zone"),
    (6, 7):   ("8→12",  "+4 jump, representation expanding"),
    (18, 19): ("12→9",  "-3 drop, entering final compression"),
    (20, 21): ("10→4",  "-6 DROP, decisive compression to output"),
}

TEXTS = [
    "Albert Einstein was born in Ulm Germany in 1879 and developed the special theory of relativity in 1905.",
    "Albert Einstein invented quantum teleportation in 1923 while at MIT and won the Nobel Prize in Computer Science.",
]
TEXT_LABELS = ["factual", "fabricated"]

def get_hidden_states(text):
    ids = tok.encode(text, return_tensors='pt')
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return [h[0].detach() for h in out.hidden_states], ids.shape[1]

def layer_jacobian(layer, h_in, pos, m=32):
    seq, d_ = h_in.shape
    m_actual = min(m, seq, d_)
    _, _, Vt = torch.linalg.svd(h_in, full_matrices=False)
    U = Vt[:m_actual, :].T.detach()
    J = np.zeros((m_actual, m_actual))
    for i in range(m_actual):
        h = h_in.clone().unsqueeze(0).detach().requires_grad_(True)
        h_out = layer(h)[0]
        v = h_out[0, pos, :] if h_out.dim()==3 else h_out[pos, :]
        (v * U[:, i]).sum().backward()
        g = h.grad
        g = g[0, pos, :].detach() if g.dim()==3 else g[pos, :].detach()
        J[:, i] = (U.T @ g).numpy()
    return J.T, U.detach().numpy(), m_actual  # J_proj, basis, dim

def active_subspace(dJ, threshold=0.10):
    """Return left singular vectors of δJ above threshold*sv_max."""
    U_sv, sv, Vt_sv = np.linalg.svd(dJ)
    k = int(np.sum(sv > sv[0] * threshold))
    return U_sv[:, :k], Vt_sv[:k, :], sv, k

def subspace_overlap(A, B):
    """
    Principal angles between column spaces of A [m,k] and B [m,j].
    Returns cos of principal angles (1=same, 0=orthogonal).
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    S = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(S, 0, 1)

# ── Run for each text ─────────────────────────────────────────────────────────
all_data = []

for t_idx, (text, label) in enumerate(zip(TEXTS, TEXT_LABELS)):
    print(f"\nText {t_idx} ({label}): '{text[:55]}...'")
    hs, seq_len = get_hidden_states(text)
    pos = seq_len // 2
    m = min(args.proj, seq_len, d)
    print(f"  seq={seq_len}  pos={pos}  proj={m}")

    layers_data = {}
    print("  Computing Jacobians...", flush=True)
    for l in range(n_L):
        J, U_basis, ma = layer_jacobian(model.transformer.h[l], hs[l], pos, m=m)
        dJ = J - np.eye(ma)
        U_sv, Vt_sv, sv, k = active_subspace(dJ)
        layers_data[l] = {
            'J': J, 'dJ': dJ, 'U_basis': U_basis,
            'U_sv': U_sv, 'Vt_sv': Vt_sv, 'sv': sv,
            'rank': k, 'norm': float(np.linalg.norm(dJ)), 'm': ma,
        }
        if (l+1) % 8 == 0: print(f"    L{l+1}...", flush=True)

    all_data.append(layers_data)
    print(f"  Ranks: {[layers_data[l]['rank'] for l in range(n_L)]}")

# ── Rank transition analysis ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RANK TRANSITIONS — SUBSPACE GEOMETRY")
print(f"  What changes at each dimensional decision point?")
print("="*65)

for (l_from, l_to), (rank_str, desc) in TRANSITIONS.items():
    print(f"\n  ── L{l_from}→L{l_to}  rank {rank_str}  ({desc}) ──")

    for t_idx, label in enumerate(TEXT_LABELS):
        ld = all_data[t_idx]
        d_from = ld[l_from]
        d_to   = ld[l_to]

        k_from = d_from['rank']
        k_to   = d_to['rank']

        # Subspace of δJ at each layer: left singular vectors
        U_from = d_from['U_sv'][:, :k_from]   # [m, k_from]
        U_to   = d_to['U_sv'][:, :k_to]       # [m, k_to]

        # Principal angles between the active subspaces
        k_min = min(k_from, k_to)
        if k_min > 0 and U_from.shape[1] > 0 and U_to.shape[1] > 0:
            cos_angles = subspace_overlap(U_from[:, :k_min], U_to[:, :k_min])
            angles_deg = np.degrees(np.arccos(np.clip(cos_angles, 0, 1)))
        else:
            angles_deg = np.array([])

        # Singular value drop/gain
        sv_from = d_from['sv'][:k_from]
        sv_to   = d_to['sv'][:k_to]

        print(f"\n  [{label}]  rank: {k_from} → {k_to}  "
              f"(δ={k_to-k_from:+d})")
        print(f"    ||δJ_from||={d_from['norm']:.4f}  "
              f"||δJ_to||={d_to['norm']:.4f}  "
              f"Δ={d_to['norm']-d_from['norm']:+.4f}")
        print(f"    sv(δJ_from): {sv_from[:5].round(4)}")
        print(f"    sv(δJ_to):   {sv_to[:5].round(4)}")
        if len(angles_deg) > 0:
            print(f"    Principal angles between active subspaces:")
            print(f"      {angles_deg[:5].round(1)}°")
            print(f"      Mean={angles_deg.mean():.1f}°  "
                  f"(0°=same subspace, 90°=orthogonal)")

        # The transition operator: how does the active subspace ROTATE?
        # T = U_to^T @ J_from @ U_from  [k_to, k_from]
        J_from = d_from['J']
        k_t = min(k_from, k_to, J_from.shape[0])
        if k_t > 0:
            T = U_to[:, :k_t].T @ J_from @ U_from[:, :k_t]
            sv_T = np.linalg.svd(T, compute_uv=False)
            print(f"    Transition operator T=U_to^T J_from U_from sv: "
                  f"{sv_T[:4].round(4)}")

# ── The decisive transition: L20→L21 ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  DECISIVE COMPRESSION: L20→L21  (rank 10→4)")
print(f"  The 6 directions that GET DROPPED here are the question")
print("="*65)

for t_idx, label in enumerate(TEXT_LABELS):
    ld = all_data[t_idx]
    d20 = ld[20]
    d21 = ld[21]

    U20 = d20['U_sv'][:, :d20['rank']]   # [m, 10]
    U21 = d21['U_sv'][:, :d21['rank']]   # [m, 4]

    # Decompose U20 into: directions kept by L21, directions dropped
    # Project U20 columns onto span(U21)
    if U21.shape[1] > 0 and U20.shape[1] > 0:
        # Component of each U20 direction in the U21 subspace
        proj_onto_21 = U21 @ (U21.T @ U20)   # [m, 10] — component kept
        residual_21  = U20 - proj_onto_21      # [m, 10] — component dropped

        kept_fraction  = np.linalg.norm(proj_onto_21, axis=0)   # per direction
        dropped_fraction = np.linalg.norm(residual_21, axis=0)

        print(f"\n  [{label}]")
        print(f"  L20 has {d20['rank']} active directions")
        print(f"  L21 keeps {d21['rank']} (those overlapping with its subspace)")
        print(f"  6 directions are compressed out\n")
        print(f"  Fraction of each L20 direction kept in L21 subspace:")
        for i, (k, dr) in enumerate(zip(kept_fraction, dropped_fraction)):
            status = "KEPT  " if k > 0.5 else "DROPPED"
            print(f"    dir_{i}: kept={k:.3f}  dropped={dr:.3f}  → {status}")

        # What are the kept vs dropped directions?
        # Their singular values at L20 tell us their "importance" at L20
        sv20 = d20['sv'][:d20['rank']]
        print(f"\n  Singular values at L20 for kept vs dropped:")
        kept_idx    = [i for i,k in enumerate(kept_fraction) if k > 0.5]
        dropped_idx = [i for i,k in enumerate(kept_fraction) if k <= 0.5]
        print(f"    Kept indices:    {kept_idx}  sv={sv20[kept_idx].round(4)}")
        print(f"    Dropped indices: {dropped_idx}  sv={sv20[dropped_idx].round(4)}")

# ── Recovery operator ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RECOVERY OPERATORS")
print(f"  To recover the trajectory: what maps active subspace back?")
print("="*65)

print(f"""
At each rank transition, the functor either:
  ADDS directions:  new information enters the active subspace
  DROPS directions: information is compressed out (cannot be recovered)

The recovery operator at layer l is:
  R_l = U_l (U_l^T U_l)^{{-1}} U_l^T  =  projection onto active subspace

To trace a hidden state back through the transitions:
  h̃_l = R_l h_l  (project onto what's active at layer l)

The DROPPED directions at L20→L21 are irreversible:
  Once compressed from rank-10 to rank-4, 6 dimensions are gone.
  The model CANNOT recover them from later layers.
  This is the information bottleneck.

THE TRANSITION MAP (what we need for trajectory recovery):
""")

for t_idx, label in enumerate(TEXT_LABELS):
    ld = all_data[t_idx]
    print(f"  [{label}]")
    for l in range(n_L-1):
        r_l   = ld[l]['rank']
        r_lp1 = ld[l+1]['rank']
        delta  = r_lp1 - r_l
        if abs(delta) >= 2 or (l, l+1) in TRANSITIONS:
            marker = " ←" if abs(delta) >= 3 else ""
            print(f"    L{l:>2}→L{l+1}: rank {r_l:>2}→{r_lp1:>2}  (δ={delta:+d}){marker}")
    print()
