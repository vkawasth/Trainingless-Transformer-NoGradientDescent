#!/usr/bin/env python3
"""
Linear Attention OLS Initialization
=====================================
Compute the optimal W_K analytically in one matrix solve.

THEORY:
  Standard attention: A = softmax(Q W_K^T H^T / sqrt(d)) 
  Linear attention:   A = Q W_K^T H^T / sqrt(d)   (no softmax)

  For linear attention, the loss L(W_K) = ||A H W_V - Y||^2
  has a closed-form minimum (OLS solution):

    ∇L/∇W_K = 0  ⟹  W_K = (Σ_QH)^{-1} Σ_QY

  where:
    Σ_QH = E[H^T Q]   (d×d)  hidden state - query covariance
    Σ_QY = E[Y^T Q]   (d×d)  target - query covariance

  This is computed in ONE PASS over the corpus.
  No gradient descent. No iterations.

  For full softmax attention, this is the optimal LINEAR initialization:
  it satisfies the linearized fixed point exactly, so gradient descent
  only needs to correct for the nonlinearity (softmax).

  PREDICTION: Starting from OLS W_K, fine-tuning reaches the same
  loss as random-init training in 10-100x fewer gradient steps.

WHAT MAKES THIS DIFFERENT FROM THE E^T E FAILURE:
  E^T E: static token co-occurrence — wrong space, no context
  OLS:   uses actual forward pass context vectors Q, H, Y
         captures the full conditional P(y | context), not just bigrams
         the relevant statistic for W_K

TWO EXPERIMENTS:
  A) OLS on a single layer: compute W_K^(l) for layer l=14
     from scratch given random W_Q, frozen E, W_V, W_O
     Compare: OLS W_K vs gradient-descent W_K after N steps
     
  B) Full training comparison:
     Random init: train full model for N steps
     OLS init:    set W_K = OLS solution, train for N steps
     Compare loss curves

Usage:
    python linear_attention_ols.py --model gpt2-medium
    python linear_attention_ols.py --compare_training
"""

import argparse, json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model',            default='gpt2-medium')
parser.add_argument('--target_layer',     type=int, default=14)
parser.add_argument('--n_corpus_batches', type=int, default=50,
                    help='Batches for covariance estimation')
parser.add_argument('--compare_training', action='store_true',
                    help='Run training comparison experiment')
parser.add_argument('--train_steps',      type=int, default=500)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n{'='*70}")
print(f"  LINEAR ATTENTION OLS INITIALIZATION")
print(f"  Model: {args.model}  |  Target layer: L{args.target_layer}")
print(f"  Device: {device}")
print(f"{'='*70}\n")

# ── Load GPT-2 ────────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
config = GPT2Config.from_pretrained(args.model)
config.output_hidden_states = True
model  = GPT2LMHeadModel.from_pretrained(args.model, config=config)
model.eval()
tok = GPT2Tokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d        = model.config.n_embd
n_layers = model.config.n_layer
n_heads  = model.config.n_head
d_head   = d // n_heads

def get_WK(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach()
    return W[:, d:2*d]

def get_WQ(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach()
    return W[:, :d]

def get_WV(l):
    W = model.transformer.h[l].attn.c_attn.weight.detach()
    return W[:, 2*d:3*d]

WK_true = get_WK(args.target_layer).numpy()
WQ      = get_WQ(args.target_layer)
WV      = get_WV(args.target_layer)

print(f"  d={d}  n_layers={n_layers}  n_heads={n_heads}")
print(f"  ||W_K^(true)^(14)|| = {np.linalg.norm(WK_true):.3f}\n")

# ── Corpus ────────────────────────────────────────────────────────────────────
TEXTS = [
    "The transformer architecture processes sequences using self-attention mechanisms that allow each token to attend to all other tokens in the sequence.",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic scale using wave functions and probability amplitudes.",
    "Natural selection drives evolutionary change in populations over many generations by favoring organisms that are better adapted to their environment.",
    "The history of mathematics includes the study of number theory algebra geometry and analysis developed by civilizations across the world.",
    "Neural networks learn hierarchical representations from training data by composing layers of parameterized nonlinear transformations.",
    "Climate change refers to long term shifts in global temperatures and weather patterns driven primarily by human activities since industrialization.",
    "The immune system recognizes and responds to pathogens through antibodies lymphocytes and inflammatory signaling cascades.",
    "Language models predict the probability distribution over tokens given the preceding context in an autoregressive generation process.",
    "Topology examines properties of spaces that are preserved under continuous deformations without tearing gluing or cutting.",
    "The discovery of DNA structure by Watson and Crick in 1953 revealed the double helix mechanism for genetic information storage.",
    "Economics studies how individuals firms and societies allocate scarce resources among competing uses to satisfy unlimited wants.",
    "The Schrodinger equation governs how the quantum state of a physical system changes over time in quantum mechanics.",
    "Computer science studies algorithms data structures computational complexity and programming languages for information processing.",
    "The French Revolution of 1789 transformed European political systems by introducing concepts of liberty equality and popular sovereignty.",
    "Photosynthesis converts solar energy into chemical energy stored in glucose molecules using carbon dioxide and water as inputs.",
    "Statistical mechanics connects the microscopic behavior of individual particles to the macroscopic thermodynamic properties of matter.",
    "The central limit theorem states that the sum of independent random variables converges to a normal distribution as the number grows.",
    "Protein folding determines the three dimensional structure of proteins from their amino acid sequence through complex interactions.",
    "Gradient descent finds parameters that minimize a loss function by iteratively moving in the direction of steepest descent.",
    "The standard model of particle physics describes fundamental particles and their interactions through gauge symmetries.",
]

def get_batch_hidden_states(text, target_layer):
    """Get Q, H_in, H_out for the target layer."""
    ids = tok.encode(text, return_tensors='pt',
                     max_length=64, truncation=True).to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # H_in: hidden state entering the target layer
    H_in  = out.hidden_states[target_layer][0]    # [seq, d]
    # H_out: hidden state after target layer (next layer input)
    H_out = out.hidden_states[target_layer + 1][0] # [seq, d]

    # Q from the target layer
    Q_full = H_in @ WQ.T   # [seq, d]
    return H_in.cpu().numpy(), Q_full.cpu().numpy(), H_out.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: OLS solution for W_K at target layer
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print(f"  EXPERIMENT A: OLS W_K at layer {args.target_layer}")
print("="*70)
print("""
  OLS derivation for linear attention:
  
  Layer output: out = softmax(Q W_K^T H_in^T / sqrt(d)) H_in W_V
  Linear approx: out ≈ (Q W_K^T H_in^T / sqrt(d)) H_in W_V
  
  Residual: target ≈ H_in + out  (residual stream)
  So: H_out - H_in ≈ (Q W_K^T H_in^T / sqrt(d)) H_in W_V W_O
  
  Let Δ = H_out - H_in  (what the layer must produce)
  Let M = W_V W_O       (value-output product, treat as fixed)
  
  Simplified: Δ = (1/sqrt(d)) * Q W_K^T H_in^T * H_in M
  
  OLS for W_K: minimize ||Δ - (1/sqrt(d)) * Q W_K^T H_in^T * H_in M||^2
  
  This is a linear least squares problem in W_K.
  Solution via moment matching across the corpus.
""")

print(f"  Accumulating corpus statistics over {len(TEXTS)} texts...", flush=True)

# Accumulate covariance matrices
# For each text: H_in [seq,d], Q [seq,d], Δ=H_out-H_in [seq,d]
# 
# The linear system for W_K (simplified):
# We want W_K such that for each (q, h_in, delta) triple:
#   q^T W_K h_in ≈ delta  (in some projected sense)
#
# Stack over all positions and texts:
# A_mat [N, d] (queries) @ W_K [d, d] @ B_mat^T [N, d] (keys) = C_mat [N, N] (targets)
#
# Simplified to: solve W_K from the normal equations.
# Use the cross-covariance approach:
#
# Σ_QH = Σ_t Σ_s q_s^T h_s  (query × hidden covariance, summed)
# Σ_QΔ = Σ_t Σ_s q_s^T δ_s  (query × delta covariance, summed)
#
# W_K^OLS = Σ_QH^{-1} Σ_QΔ   (d×d)

Sigma_QH = np.zeros((d, d))   # E[Q^T H_in]
Sigma_QD = np.zeros((d, d))   # E[Q^T Delta]
n_total  = 0

for text in TEXTS:
    H_in, Q, H_out = get_batch_hidden_states(text, args.target_layer)
    Delta = H_out - H_in   # what the layer adds [seq, d]

    # Accumulate
    Sigma_QH += Q.T @ H_in   # [d, d]
    Sigma_QD += Q.T @ Delta   # [d, d]
    n_total  += H_in.shape[0]

Sigma_QH /= n_total
Sigma_QD /= n_total

print(f"  Accumulated from {n_total} token positions")
print(f"  ||Sigma_QH|| = {np.linalg.norm(Sigma_QH):.4f}")
print(f"  ||Sigma_QD|| = {np.linalg.norm(Sigma_QD):.4f}")
print(f"  Cond(Sigma_QH) = {np.linalg.cond(Sigma_QH):.1f}")

# OLS solution: W_K = Sigma_QH^{-1} Sigma_QD
# Use pseudo-inverse for numerical stability
print(f"\n  Solving W_K = Sigma_QH^+ @ Sigma_QD...", flush=True)
try:
    # Regularized solve
    reg = 1e-6 * np.linalg.norm(Sigma_QH) * np.eye(d)
    WK_ols = np.linalg.solve(Sigma_QH + reg, Sigma_QD)
except np.linalg.LinAlgError:
    WK_ols = np.linalg.lstsq(Sigma_QH, Sigma_QD, rcond=None)[0]

print(f"  ||W_K^OLS|| = {np.linalg.norm(WK_ols):.4f}")
print(f"  ||W_K^true|| = {np.linalg.norm(WK_true):.4f}")

# Compare OLS vs true
cos_ols_true = float(WK_ols.ravel() @ WK_true.ravel() /
                     (np.linalg.norm(WK_ols) * np.linalg.norm(WK_true) + 1e-10))
rel_err = float(np.linalg.norm(WK_ols - WK_true) / np.linalg.norm(WK_true))

print(f"\n  COMPARISON (OLS vs trained W_K^({args.target_layer})):")
print(f"    cos_sim:  {cos_ols_true:+.4f}")
print(f"    rel_err:  {rel_err:.4f}")

# Hessenberg violation of OLS solution
from scipy.linalg import hessenberg as scipy_hessenberg
def hess_viol(M):
    d_ = M.shape[0]
    mask = np.zeros((d_,d_))
    for i in range(d_):
        for j in range(d_):
            if j < i-1: mask[i,j] = 1.
    return float(np.linalg.norm(M*mask)/max(np.linalg.norm(M),1e-8))

print(f"    Hessenberg violation of W_K^OLS:  {hess_viol(WK_ols):.4f}")
print(f"    Hessenberg violation of W_K^true: {hess_viol(WK_true):.4f}")

# Compare to random baseline
WK_rand = np.random.randn(d, d) * 0.02
cos_rand = float(WK_rand.ravel() @ WK_true.ravel() /
                 (np.linalg.norm(WK_rand)*np.linalg.norm(WK_true)+1e-10))
print(f"    cos_sim(random, W_K^true): {cos_rand:+.4f}  (baseline)")
print(f"    Signal/noise: {abs(cos_ols_true)/max(abs(cos_rand),1e-6):.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: OLS across all layers
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  EXPERIMENT B: OLS W_K vs trained W_K across all layers")
print("="*70)

cos_sims_all = []
rel_errs_all = []
hv_ols_all   = []

for layer in range(n_layers):
    WQ_l = get_WQ(layer)
    WK_l_true = get_WK(layer).numpy()

    S_QH = np.zeros((d, d)); S_QD = np.zeros((d, d)); n_t = 0
    for text in TEXTS[:10]:  # faster: use 10 texts per layer
        ids = tok.encode(text, return_tensors='pt',
                        max_length=64, truncation=True).to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        H_in  = out.hidden_states[layer][0].cpu().numpy()
        H_out = out.hidden_states[layer+1][0].cpu().numpy() if layer < n_layers else H_in
        Q_l   = (out.hidden_states[layer][0] @ WQ_l.T).cpu().numpy()
        Delta  = H_out - H_in
        S_QH += Q_l.T @ H_in; S_QD += Q_l.T @ Delta; n_t += H_in.shape[0]

    S_QH /= n_t; S_QD /= n_t
    reg = 1e-6 * np.linalg.norm(S_QH) * np.eye(d)
    try:
        WK_ols_l = np.linalg.solve(S_QH + reg, S_QD)
    except:
        WK_ols_l = np.linalg.lstsq(S_QH, S_QD, rcond=None)[0]

    cs = float(WK_ols_l.ravel() @ WK_l_true.ravel() /
               (np.linalg.norm(WK_ols_l)*np.linalg.norm(WK_l_true)+1e-10))
    re = float(np.linalg.norm(WK_ols_l - WK_l_true)/np.linalg.norm(WK_l_true))
    hv = hess_viol(WK_ols_l)
    cos_sims_all.append(cs); rel_errs_all.append(re); hv_ols_all.append(hv)

    if layer in [0,5,9,14,19,23]:
        print(f"  L{layer:>2}: cos={cs:+.4f}  rel_err={re:.4f}  "
              f"hv_ols={hv:.4f}  hv_true={hess_viol(WK_l_true):.4f}")

r_cs, p_cs = spearmanr(range(n_layers), cos_sims_all)
print(f"\n  Mean cos_sim(OLS, true): {np.mean(cos_sims_all):+.4f}")
print(f"  Mean rel_err:            {np.mean(rel_errs_all):.4f}")
print(f"  Spearman r(cos vs depth): {r_cs:+.3f}  p={p_cs:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C (optional): Training comparison
# ══════════════════════════════════════════════════════════════════════════════
if args.compare_training:
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT C: Training comparison")
    print(f"  Random init vs OLS init — {args.train_steps} steps each")
    print("="*70)

    with open('/tmp/train_ids.json') as f: train_ids = json.load(f)
    with open('/tmp/val_ids.json')   as f: val_ids   = json.load(f)
    with open('/tmp/vocab.json')     as f: vocab     = json.load(f)
    VOCAB = len(vocab)

    train_t = torch.tensor(train_ids, dtype=torch.long)
    val_t   = torch.tensor(val_ids,   dtype=torch.long)

    def get_batch(split='train', B=4, S=64):
        data = train_t if split=='train' else val_t
        ix   = torch.randint(0, len(data)-S-1, (B,))
        x    = torch.stack([data[i:i+S] for i in ix]).to(device)
        y    = torch.stack([data[i+1:i+S+1] for i in ix]).to(device)
        return x, y

    # Small transformer for fair comparison
    class SmallAttn(nn.Module):
        def __init__(self, d, nh=4):
            super().__init__()
            self.nh = nh; self.dh = d//nh; self.sc = math.sqrt(d//nh)
            self.W_Q = nn.Linear(d, d, bias=False)
            self.W_K = nn.Linear(d, d, bias=False)
            self.W_V = nn.Linear(d, d, bias=False)
            self.out = nn.Linear(d, d, bias=False)
            self.ln  = nn.LayerNorm(d)
        def forward(self, h):
            B,S,D=h.shape; H=self.nh; dh=self.dh
            Q=self.W_Q(h).view(B,S,H,dh).transpose(1,2)
            K=self.W_K(h).view(B,S,H,dh).transpose(1,2)
            V=self.W_V(h).view(B,S,H,dh).transpose(1,2)
            sc=Q@K.transpose(-2,-1)/self.sc
            mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
            sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
            out=(F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D)
            return self.ln(h+self.out(out))

    class SmallFF(nn.Module):
        def __init__(self,d):
            super().__init__()
            self.g=nn.Linear(d,d*2,bias=False); self.v=nn.Linear(d,d*2,bias=False)
            self.o=nn.Linear(d*2,d,bias=False); self.n=nn.LayerNorm(d)
        def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))

    class SmallLM(nn.Module):
        def __init__(self, d=128, nh=4, n_layers=3):
            super().__init__()
            self.te=nn.Embedding(VOCAB,d); nn.init.normal_(self.te.weight,std=0.02)
            self.pe=nn.Embedding(512,d);   nn.init.normal_(self.pe.weight,std=0.02)
            self.blocks=nn.ModuleList([nn.Sequential(SmallAttn(d,nh),SmallFF(d))
                                       for _ in range(n_layers)])
            self.ln=nn.LayerNorm(d)
            self.h=nn.Linear(d,VOCAB,bias=False); self.h.weight=self.te.weight
        def forward(self,x,y=None):
            h=self.te(x)+self.pe(torch.arange(x.shape[1],device=x.device))
            for b in self.blocks: h=b(h)
            logits=self.h(self.ln(h))
            loss=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None
            return logits,loss

    D_SMALL = 128; NH_SMALL = 4; N_LAYERS = 2
    LR = 3e-4

    def train_model(model, name, steps, ols_init=False):
        if ols_init:
            # Compute OLS W_K for layer 0 of the small model
            # (using our 20 diverse texts mapped to this model's hidden space)
            print(f"  Computing OLS init for {name}...", flush=True)
            S_QH_s = np.zeros((D_SMALL, D_SMALL))
            S_QD_s = np.zeros((D_SMALL, D_SMALL))
            n_s = 0
            for text in TEXTS[:10]:
                ids = tok.encode(text, return_tensors='pt',
                                max_length=64, truncation=True).to(device)
                with torch.no_grad():
                    # Get h_0 from embedding only
                    # Map GPT-2 token IDs to our small model vocab (approximate)
                    # Use GPT-2 as oracle, project to D_SMALL
                    pass  # skip for small model — use random corpus tokens instead
            # For small model: use random corpus data
            for _ in range(20):
                x, y = get_batch('train')
                with torch.no_grad():
                    h = model.te(x) + model.pe(torch.arange(x.shape[1],device=x.device))
                    # Layer 0 input
                    H_in_s = h[0].cpu().numpy()
                    attn = model.blocks[0][0]
                    Q_s = H_in_s @ attn.W_Q.weight.detach().cpu().numpy().T
                    h_out = model.blocks[0](h)
                    H_out_s = h_out[0].cpu().numpy()
                    D_s = H_out_s - H_in_s
                    S_QH_s += Q_s.T @ H_in_s
                    S_QD_s += Q_s.T @ D_s
                    n_s += H_in_s.shape[0]
            S_QH_s /= n_s; S_QD_s /= n_s
            reg = 1e-6 * np.linalg.norm(S_QH_s) * np.eye(D_SMALL)
            try:
                WK_ols_s = np.linalg.solve(S_QH_s + reg, S_QD_s)
            except:
                WK_ols_s = np.linalg.lstsq(S_QH_s, S_QD_s, rcond=None)[0]
            with torch.no_grad():
                model.blocks[0][0].W_K.weight.copy_(
                    torch.tensor(WK_ols_s.T, dtype=torch.float32))
            print(f"  OLS W_K set. ||W_K^OLS|| = {np.linalg.norm(WK_ols_s):.4f}")

        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
        val_losses = []; steps_log = []
        model.to(device)
        print(f"\n  Training {name}...")
        for step in range(1, steps+1):
            for pg in opt.param_groups:
                pg['lr'] = LR * 0.5*(1+math.cos(math.pi*step/steps))
            model.train(); x,y=get_batch('train')
            _,loss=model(x,y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            if step % 50 == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    vls=[]; 
                    for _ in range(10):
                        xv,yv=get_batch('val'); _,vl=model(xv,yv); vls.append(vl.item())
                vl_mean=float(np.mean(vls))
                val_losses.append(vl_mean); steps_log.append(step)
                print(f"    {step:>4}/{steps}  val={vl_mean:.4f}")
        return steps_log, val_losses

    torch.manual_seed(42)
    m_rand = SmallLM(D_SMALL, NH_SMALL, N_LAYERS)
    steps_r, loss_r = train_model(m_rand, "Random init", args.train_steps)

    torch.manual_seed(42)
    m_ols  = SmallLM(D_SMALL, NH_SMALL, N_LAYERS)
    steps_o, loss_o = train_model(m_ols, "OLS init", args.train_steps, ols_init=True)

    print(f"\n  {'Step':>5}  {'Random':>8}  {'OLS':>8}  {'Delta':>8}  {'Winner'}")
    print("  " + "-"*40)
    for i in range(len(steps_r)):
        d_ = loss_o[i]-loss_r[i]
        w = 'OLS ✓' if d_<0 else 'Rand'
        print(f"  {steps_r[i]:>5}  {loss_r[i]:>8.4f}  {loss_o[i]:>8.4f}  {d_:>+8.4f}  {w}")

    ols_wins = sum(1 for r,o in zip(loss_r, loss_o) if o < r)
    print(f"\n  OLS wins at {ols_wins}/{len(steps_r)} checkpoints")
    print(f"  Final: Random={loss_r[-1]:.4f}  OLS={loss_o[-1]:.4f}")

# ── Final verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  VERDICT")
print(f"{'='*70}\n")

mean_cs = float(np.mean(cos_sims_all))
if mean_cs > 0.5:
    print(f"  STRONG: OLS provides excellent W_K initialization (mean cos={mean_cs:.3f})")
    print(f"  The linear attention fixed point is close to the trained W_K.")
    print(f"  Gradient descent is largely redundant for W_K given W_Q.")
elif mean_cs > 0.2:
    print(f"  MODERATE: OLS provides useful initialization (mean cos={mean_cs:.3f})")
    print(f"  Reduces training steps but does not eliminate gradient descent.")
elif mean_cs > 0.05:
    print(f"  WEAK: OLS slightly better than random (mean cos={mean_cs:.3f})")
    print(f"  The softmax nonlinearity makes linear approx insufficient.")
else:
    print(f"  NEGATIVE: OLS not aligned with trained W_K (mean cos={mean_cs:.3f})")
    print(f"  The attention fixed point requires higher-order corpus statistics.")
    print(f"  Linear approximation is insufficient — need nonlinear solver.")
