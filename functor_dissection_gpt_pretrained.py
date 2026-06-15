#!/usr/bin/env python3
"""
Functor Dissection on Pretrained GPT-2
Loads pretrained weights, no training. Dissects J_l = dh_l/dh_{l-1}
via autograd vjp on real text.

Usage: python functor_dissection.py [--model gpt2-medium]
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.linalg import sqrtm as scipy_sqrtm

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-medium')
parser.add_argument('--proj', type=int, default=32)
args = parser.parse_args()

print(f"\n{'='*65}")
print(f"  FUNCTOR DISSECTION  model={args.model}  proj={args.proj}")
print(f"{'='*65}\n")

print("Loading pretrained model...", flush=True)
model = GPT2LMHeadModel.from_pretrained(args.model)
tok   = GPT2Tokenizer.from_pretrained(args.model)
model.eval()

d   = model.config.n_embd
n_L = model.config.n_layer
n_H = model.config.n_head
print(f"  d={d}  layers={n_L}  heads={n_H}\n")

# Use longer texts so seq > proj dim
TEXTS = [
    "The transformer architecture uses multi-head self-attention to process input sequences and learn contextual representations of tokens.",
    "Gradient descent with momentum minimizes the cross-entropy loss function iteratively over many training steps on batched data.",
    "Albert Einstein was born in Ulm Germany in 1879 and developed the special theory of relativity in 1905 while working at the patent office.",
    "Albert Einstein invented quantum teleportation in 1923 while at MIT and won the Nobel Prize in Computer Science for this discovery.",
]

# ── Hidden states ─────────────────────────────────────────────────────────────
def get_hidden_states(text):
    ids = tok.encode(text, return_tensors='pt')
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = [h[0].detach() for h in out.hidden_states]  # list of [seq, d]
    return hs, ids.shape[1]

# ── Jacobian via autograd vjp ─────────────────────────────────────────────────
def layer_jacobian_gpt2(layer, h_in, pos, m=32):
    """
    J_proj = U^T J U  [m_actual, m_actual]
    h_in: [seq, d]
    m_actual = min(m, seq, d) — capped so SVD always has enough vectors.
    """
    seq, d_ = h_in.shape
    m_actual = min(m, seq, d_)   # never ask for more vectors than exist

    _, _, Vt = torch.linalg.svd(h_in, full_matrices=False)  # Vt: [min(seq,d), d]
    U = Vt[:m_actual, :].T.detach()   # [d, m_actual]

    J_proj = np.zeros((m_actual, m_actual))
    for i in range(m_actual):
        h = h_in.clone().unsqueeze(0).detach().requires_grad_(True)  # [1, seq, d]
        h_out_tuple = layer(h)
        h_out = h_out_tuple[0]
        # handle [1,seq,d] or [seq,d]
        out_vec = h_out[0, pos, :] if h_out.dim() == 3 else h_out[pos, :]

        scalar = (out_vec * U[:, i]).sum()
        scalar.backward()

        g_full = h.grad
        g = g_full[0, pos, :].detach() if g_full.dim() == 3 else g_full[pos, :].detach()

        J_proj[:, i] = (U.T @ g).numpy()   # [m_actual]

    return J_proj.T, m_actual   # U^T J U,  actual projection dim used

# ── Main loop ─────────────────────────────────────────────────────────────────
m_req = args.proj
EXC   = {2,17,18,20,21}

print(f"Computing Jacobians ({n_L} layers × {len(TEXTS)} texts)...\n")

all_results = []
for t_idx, text in enumerate(TEXTS):
    print(f"  Text {t_idx}: '{text[:60]}'", flush=True)
    hs, seq_len = get_hidden_states(text)
    pos = seq_len // 2
    m_actual = min(m_req, seq_len, d)
    print(f"    seq={seq_len}  pos={pos}  proj={m_actual}", flush=True)

    text_results = []
    for l in range(n_L):
        J, ma = layer_jacobian_gpt2(model.transformer.h[l], hs[l], pos, m=m_req)
        dJ    = J - np.eye(ma)
        sv_J  = np.linalg.svd(J,  compute_uv=False)
        sv_dJ = np.linalg.svd(dJ, compute_uv=False)
        r10   = int(np.sum(sv_dJ > sv_dJ[0]*0.10))
        text_results.append({'J': J, 'dJ': dJ, 'sv_J': sv_J, 'sv_dJ': sv_dJ,
                             'norm_dJ': float(np.linalg.norm(dJ)),
                             'rank10': r10, 'm': ma})
        if (l+1) % 8 == 0:
            print(f"    L{l+1} done...", flush=True)
    all_results.append(text_results)
    print(f"    done")

# ── Per-layer ||δJ|| table ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ||δJ_l = J_l - I||  per layer  (all texts)")
print(f"  Theory: δJ dominated by rank-n_heads={n_H} attention term")
print("="*65)
tnames = ['T0','T1','T2-fact','T3-fabr']
print(f"\n  {'L':>4}  {'st':>6}", end='')
for tn in tnames: print(f"  {tn:>9}", end='')
print(f"  {'mean':>8}  {'rank':>6}")
print("  "+"-"*(12 + 11*len(TEXTS) + 16))

for l in range(n_L):
    norms = [all_results[t][l]['norm_dJ'] for t in range(len(TEXTS))]
    ranks = [all_results[t][l]['rank10']  for t in range(len(TEXTS))]
    st    = "WALL" if l in EXC else ("L14★" if l==14 else "")
    print(f"  L{l:>2}  {st:>6}", end='')
    for n in norms: print(f"  {n:>9.4f}", end='')
    print(f"  {np.mean(norms):>8.4f}  {int(np.mean(ranks)):>6}")

# ── Monodromies ───────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  MONODROMY M = J_24 @ ... @ J_1")
print("="*65)

monodromies = []
for t_idx, text in enumerate(TEXTS):
    ma = all_results[t_idx][0]['m']
    M  = np.eye(ma)
    for l in range(n_L-1, -1, -1):
        M = all_results[t_idx][l]['J'] @ M
    sv_M = np.linalg.svd(M, compute_uv=False)
    monodromies.append({'M': M, 'sv': sv_M, 'm': ma})
    print(f"  T{t_idx}: sv=[{sv_M[0]:.4f}, {sv_M[1]:.4f}, ..., {sv_M[-1]:.4f}]  "
          f"||M-I||={np.linalg.norm(M-np.eye(ma)):.4f}")

# ── Factual vs fabricated ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FACTUAL (T2) vs FABRICATED (T3) — Einstein pair")
print("="*65)
Mi = monodromies[2]['M']; Mj = monodromies[3]['M']
ma = min(monodromies[2]['m'], monodromies[3]['m'])
Mi = Mi[:ma, :ma]; Mj = Mj[:ma, :ma]
diff = np.linalg.norm(Mi - Mj)
print(f"\n  ||M_factual - M_fabricated|| = {diff:.4f}")
print(f"  sv(M_fact):  {monodromies[2]['sv'][:6].round(4)}")
print(f"  sv(M_fabr):  {monodromies[3]['sv'][:6].round(4)}")

# ── sqrtm and closest layer ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  sqrtm(M) vs per-layer J — which layer IS the monodromy fixed point?")
print("="*65)

for t_idx in [2, 3]:
    M  = monodromies[t_idx]['M']
    ma = monodromies[t_idx]['m']
    try:
        sqM  = np.real(scipy_sqrtm(M))
        err  = np.linalg.norm(sqM @ sqM - M) / max(np.linalg.norm(M), 1e-8)
        sv_sq = np.linalg.svd(sqM, compute_uv=False)
        dists = []
        for l in range(n_L):
            J = all_results[t_idx][l]['J']
            Jc = J[:ma, :ma]
            dists.append(np.linalg.norm(Jc - sqM) / max(np.linalg.norm(sqM), 1e-8))
        best = int(np.argmin(dists))
        top5 = sorted(range(n_L), key=lambda l: dists[l])[:5]
        l14_rank = sorted(range(n_L), key=lambda l: dists[l]).index(14) + 1
        print(f"\n  T{t_idx}: sqrtm sv=[{', '.join(f'{v:.4f}' for v in sv_sq[:5])}]  err={err:.2e}")
        print(f"    Closest layer to sqrtm(M): L{best}  (dist={dists[best]:.4f})")
        print(f"    Top-5 closest: {top5}")
        print(f"    L14 rank: {l14_rank}/{n_L}  dist={dists[14]:.4f}")
    except Exception as e:
        print(f"  T{t_idx}: sqrtm failed: {e}")

# ── L14 neighborhood ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  L14 NEIGHBORHOOD — is L14 minimum ||δJ||? (attractor prediction)")
print("="*65)
print(f"\n  {'L':>4}  {'st':>6}  {'mean ||δJ||':>12}  {'rank(10%)':>10}")
print("  "+"-"*38)
for l in range(n_L):
    norms = [all_results[t][l]['norm_dJ'] for t in range(len(TEXTS))]
    ranks = [all_results[t][l]['rank10']  for t in range(len(TEXTS))]
    st = "WALL" if l in EXC else ("←L14★" if l==14 else "")
    print(f"  L{l:>2}  {st:>6}  {np.mean(norms):>12.4f}  {int(np.mean(ranks)):>10}")
