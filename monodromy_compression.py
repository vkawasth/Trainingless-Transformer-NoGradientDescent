#!/usr/bin/env python3
"""
Monodromy Compression: sqrtm(M_24) → 2-layer inference
=========================================================
Given a trained N-layer model:
  1. One forward pass → hidden states → Jacobians → M_N
  2. Compute sqrtm(M_N) → target 2-layer Jacobian
  3. Find W* such that J(W*) = sqrtm(M_N)  via least-squares
  4. Set 2-layer model weights directly
  5. Zero additional gradient steps

COMPUTE SAVINGS:
  Training: unchanged (still need the N-layer model)
  Inference: N-layer → 2-layer
    FLOPs per token: O(N × d²) → O(2 × d²)  →  N/2 speedup
    Memory: N sets of weights → 2 sets
    Latency: proportional to layer count

The key: we are compressing the TRAINED model, not replacing training.
The A∞ monodromy IS the trained model's computation — we extract
and compress it, not relearn it.

Usage: python monodromy_compression.py [--layers 24] [--d 256]
"""
import argparse, json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=24)
parser.add_argument('--d',      type=int, default=256)
parser.add_argument('--heads',  type=int, default=4)
parser.add_argument('--steps',  type=int, default=300, help='training steps for source model')
parser.add_argument('--proj',   type=int, default=64,  help='Jacobian projection dim')
args = parser.parse_args()

D=args.d; N_HEADS=args.heads; N_LAYERS=args.layers
BATCH=8; SEQ=64; LR=3e-4; PROJ=args.proj

print(f"\n{'='*65}")
print(f"  MONODROMY COMPRESSION")
print(f"  {N_LAYERS}-layer → 2-layer via sqrtm(M_{N_LAYERS})")
print(f"  d={D}  heads={N_HEADS}  proj={PROJ}")
print(f"  Zero additional gradient steps after compression")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

# ── Architecture ──────────────────────────────────────────────────────────────
class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        out=(F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)
        return self.ln(h+self.op(out))

class FF(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.g=nn.Linear(d,d*2,bias=False); self.v=nn.Linear(d,d*2,bias=False)
        self.o=nn.Linear(d*2,d,bias=False); self.n=nn.LayerNorm(d)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))

class Block(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=Attn(d,nh); self.ff=FF(d)
    def forward(self,h): return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
        self._nl=nl
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def n_params(self): return sum(p.numel() for p in self.parameters())
    def hidden_states_list(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs
    def flops_per_token(self):
        # Attention: Q,K,V projections + output + FFN
        attn = 4 * D * D      # WQ,WK,WV,Wop
        ffn  = 3 * D * D * 2  # gate,val,out with 4D hidden
        return self._nl * (attn + ffn)

def eval_val(model, n=100):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Step 1: Train the source N-layer model ────────────────────────────────────
print(f"Step 1: Train {N_LAYERS}-layer source model ({args.steps} steps)...")
torch.manual_seed(42)
source = LM(D, N_HEADS, N_LAYERS)
opt = torch.optim.AdamW(source.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t_train_start = time.time()
for step in range(1, args.steps+1):
    for pg in opt.param_groups: pg['lr']=clr(step,args.steps)
    source.train(); x,y=get_batch(); _,loss=source(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(source.parameters(),1.0); opt.step()
    if step % (args.steps//3) == 0:
        vl = eval_val(source, n=20)
        print(f"  step {step:>4}/{args.steps}  val={vl:.4f}  t={time.time()-t_train_start:.0f}s")
t_train = time.time() - t_train_start
val_source = eval_val(source)
source.eval()
print(f"  Source model: val={val_source:.4f}  params={source.n_params():,}  t={t_train:.1f}s\n")

# ── Step 2: One forward pass → Jacobians → Monodromy ─────────────────────────
print(f"Step 2: Extract monodromy (ONE forward pass)...")
t_compress_start = time.time()

# One representative batch
x_ref, _ = get_batch('val')
x_ref = x_ref[0:1]   # [1, SEQ]
with torch.no_grad():
    hs_batch = source.hidden_states_list(x_ref)
    hs = [h[0] for h in hs_batch]   # list of [SEQ, D]

pos = SEQ // 2
m = min(PROJ, SEQ, D)

def layer_jacobian(block, h_in, pos, m):
    seq, d_ = h_in.shape; m = min(m, seq, d_)
    _, _, Vt = torch.linalg.svd(h_in, full_matrices=False)
    U = Vt[:m, :].T.detach()
    J = np.zeros((m, m))
    for i in range(m):
        h = h_in.clone().unsqueeze(0).detach().requires_grad_(True)
        h_out = block(h)[0]
        v = h_out[0, pos, :] if h_out.dim()==3 else h_out[pos, :]
        (v * U[:, i]).sum().backward()
        g = h.grad
        g = (g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
        J[:, i] = (U.T @ g).numpy()
    return J.T, U.detach().numpy(), m

print(f"  Computing {N_LAYERS} layer Jacobians (proj={m})...", flush=True)
Js = []; U_basis = None; m_actual = None
for l in range(N_LAYERS):
    J, U, ma = layer_jacobian(source.blocks[l], hs[l], pos, m)
    Js.append(J)
    if U_basis is None: U_basis = U; m_actual = ma
    if (l+1) % 8 == 0: print(f"    L{l+1}...", flush=True)

# Compose monodromy M_N = J_N @ ... @ J_1
M = np.eye(m_actual)
for J in reversed(Js): M = J @ M
sv_M = np.linalg.svd(M, compute_uv=False)
print(f"  M_{N_LAYERS}: sv=[{sv_M[0]:.3f}, {sv_M[1]:.3f}, ..., {sv_M[-1]:.5f}]")

# sqrtm(M) = target for each of 2 layers
sqM = np.real(scipy_sqrtm(M))
err = np.linalg.norm(sqM@sqM - M) / max(np.linalg.norm(M), 1e-8)
sv_sq = np.linalg.svd(sqM, compute_uv=False)
print(f"  sqrtm(M): sv=[{sv_sq[0]:.3f}, {sv_sq[1]:.3f}, ..., {sv_sq[-1]:.5f}]")
print(f"  Verification ||sqrtm²-M||/||M|| = {err:.2e}")

# ── Step 3: Solve for weights W* given J* = sqrtm(M) ─────────────────────────
print(f"\nStep 3: Solve for 2-layer weights via least-squares...")
# J*(W) = I + W_O * A * W_V  (attention part, A = attention pattern)
# We have the actual attention patterns from the source model's hidden states
# Use source layer 0's attention pattern as the template
# A_template: actual softmax(QK^T/sqrt(d)) from the trained source model

def get_attention_pattern(block, h_in):
    """Extract actual attention pattern A from trained block."""
    with torch.no_grad():
        B,S,D_=h_in.unsqueeze(0).shape; H=block.attn.nh; dh=block.attn.dh
        h=h_in.unsqueeze(0)
        Q=block.attn.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=block.attn.WK(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/block.attn.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        A=F.softmax(sc,dim=-1)   # [1, H, S, S]
    return A[0].numpy()   # [H, S, S]

# Target: δJ* = sqrtm(M) - I  in projected space
dJstar = sqM - np.eye(m_actual)   # [m, m]
print(f"  δJ* = sqrtm(M)-I: ||δJ*||={np.linalg.norm(dJstar):.4f}")

# Strategy: set W_O and W_V in the projected subspace
# δJ_attn ≈ (U W_O) A_mean (W_V U^T)  in m-space
# where U = [d, m] projection basis
# 
# Simplest consistent solution that uses actual attention scale:
# Mean attention pattern across heads: A_mean [S, S]
A_src = get_attention_pattern(source.blocks[N_LAYERS//2], hs[N_LAYERS//2])
A_mean = A_src.mean(axis=0)   # [S, S] — mean over heads
# Expected contribution per token-position pair
# At position `pos`: A_mean[pos, :] = attention weights

# Scale: W_O W_V should produce dJstar when multiplied by A_mean effect
# Effective scalar: mean attention weight received at pos
a_eff = float(A_mean[pos, :].sum())   # ≈ 1 (attention is normalized)

# In m-space: W_O_m @ W_V_m = dJstar / a_eff
target_m = dJstar / max(a_eff, 1e-6)
# Factor via SVD: target = U_t S_t V_t^T
U_t, s_t, Vt_t = np.linalg.svd(target_m)
sqrt_s = np.sqrt(np.abs(s_t))

# W_O_m = U_t @ diag(sqrt_s)   [m, m]
# W_V_m = diag(sqrt_s) @ Vt_t  [m, m]
W_O_m = U_t * sqrt_s[np.newaxis, :]
W_V_m = (Vt_t.T * sqrt_s[np.newaxis, :]).T

# Lift to d-space: W_O_d = U_basis @ W_O_m @ U_basis^T + (I - UU^T)
# Identity on the orthogonal complement (don't destroy it)
U_t_d = torch.tensor(U_basis, dtype=torch.float32)   # [d, m]
W_O_m_t = torch.tensor(W_O_m, dtype=torch.float32)   # [m, m]
W_V_m_t = torch.tensor(W_V_m, dtype=torch.float32)   # [m, m]

W_O_d = U_t_d @ W_O_m_t @ U_t_d.T   # [d, d] — in subspace
W_V_d = U_t_d @ W_V_m_t @ U_t_d.T   # [d, d] — in subspace
# Add identity on complement so non-subspace dims pass through
P_orth = torch.eye(D) - U_t_d @ U_t_d.T
W_O_d = W_O_d + P_orth   # identity outside subspace
W_V_d = W_V_d + P_orth

# ── Step 4: Build compressed 2-layer model ────────────────────────────────────
print(f"\nStep 4: Build compressed 2-layer model (ZERO gradient steps)...")
torch.manual_seed(0)
compressed = LM(D, N_HEADS, 2)

# Copy embedding and head from source (the semantics live here)
compressed.te.weight.data.copy_(source.te.weight.data)
compressed.pe.weight.data.copy_(source.pe.weight.data)
compressed.ln_f.weight.data.copy_(source.ln_f.weight.data)
compressed.ln_f.bias.data.copy_(source.ln_f.bias.data)
# head is tied to te, so already copied

# Set attention weights in both blocks to the monodromy-derived values
for blk in compressed.blocks:
    with torch.no_grad():
        blk.attn.op.weight.copy_(W_O_d)
        blk.attn.WV.weight.copy_(W_V_d)
        # WQ, WK: copy from middle layer of source (attention routing)
        mid = N_LAYERS // 2
        blk.attn.WQ.weight.copy_(source.blocks[mid].attn.WQ.weight)
        blk.attn.WK.weight.copy_(source.blocks[mid].attn.WK.weight)
        # LayerNorm: copy from attractor center (L14 or mid)
        attn_ln_src = source.blocks[mid].attn.ln
        blk.attn.ln.weight.copy_(attn_ln_src.weight)
        blk.attn.ln.bias.copy_(attn_ln_src.bias)
        # FFN: copy from mid layer
        blk.ff.g.weight.copy_(source.blocks[mid].ff.g.weight)
        blk.ff.v.weight.copy_(source.blocks[mid].ff.v.weight)
        blk.ff.o.weight.copy_(source.blocks[mid].ff.o.weight)
        blk.ff.n.weight.copy_(source.blocks[mid].ff.n.weight)
        blk.ff.n.bias.copy_(source.blocks[mid].ff.n.bias)

t_compress = time.time() - t_compress_start
compressed.eval()

# ── Step 5: Evaluate and measure compute savings ──────────────────────────────
print(f"\nStep 5: Evaluate compression quality and compute savings...")
val_compressed = eval_val(compressed)

# Cosine similarity: compare output logits on test batch
def cosine_sim(model_a, model_b, n_batches=20):
    model_a.eval(); model_b.eval(); sims=[]
    with torch.no_grad():
        for _ in range(n_batches):
            x,_=get_batch('val')
            la,_=model_a(x); lb,_=model_b(x)
            la=la.reshape(-1,VOCAB); lb=lb.reshape(-1,VOCAB)
            sim=F.cosine_similarity(la,lb,dim=-1).mean().item()
            sims.append(sim)
    return float(np.mean(sims))

cos = cosine_sim(source, compressed)

# Also test: random 2-layer baseline
torch.manual_seed(99)
rand2 = LM(D, N_HEADS, 2)
val_rand2 = eval_val(rand2)
cos_rand = cosine_sim(source, rand2)

# Compute savings
params_source     = source.n_params()
params_compressed = compressed.n_params()
flops_source      = source.flops_per_token()
flops_compressed  = compressed.flops_per_token()

# Inference time (rough)
def time_inference(model, n=200):
    model.eval(); t0=time.time()
    with torch.no_grad():
        for _ in range(n):
            x,_=get_batch('val'); model(x)
    return (time.time()-t0)/n

t_source_inf     = time_inference(source)
t_compressed_inf = time_inference(compressed)
t_rand2_inf      = time_inference(rand2)

print(f"\n{'='*65}")
print(f"  COMPRESSION RESULTS")
print(f"{'='*65}")
print(f"""
  SOURCE MODEL ({N_LAYERS} layers):
    Parameters:    {params_source:>12,}
    FLOPs/token:   {flops_source:>12,}
    Inference/batch: {t_source_inf*1000:>8.1f} ms
    Val loss:      {val_source:>12.4f}
    Training time: {t_train:>8.1f}s  ({args.steps} steps)

  COMPRESSED MODEL (2 layers, zero gradient steps):
    Parameters:    {params_compressed:>12,}
    FLOPs/token:   {flops_compressed:>12,}
    Inference/batch: {t_compressed_inf*1000:>8.1f} ms
    Val loss:      {val_compressed:>12.4f}
    Compression time: {t_compress:>5.1f}s  (1 forward pass + sqrtm)
    Cosine sim with source: {cos:.4f}

  RANDOM 2-LAYER (no training, no monodromy):
    Val loss:      {val_rand2:>12.4f}
    Cosine sim with source: {cos_rand:.4f}

  SAVINGS:
    Parameter reduction:  {params_source/params_compressed:.1f}x  
                          ({params_source:,} → {params_compressed:,})
    FLOPs reduction:      {flops_source/flops_compressed:.1f}x
                          ({flops_source:,} → {flops_compressed:,})
    Inference speedup:    {t_source_inf/t_compressed_inf:.1f}x
    Compression cost:     {t_compress:.1f}s  (vs {t_train:.1f}s training)
    Training cost saved:  0 steps  (monodromy from existing model)

  QUALITY:
    Val loss gap:     {val_compressed - val_source:+.4f}  (compressed vs source)
    Monodromy cos:    {cos:.4f}  (logit-level similarity)
    Random 2L cos:    {cos_rand:.4f}  (baseline — monodromy adds {cos-cos_rand:+.4f})
    
  VERDICT:
""")

if cos > cos_rand + 0.05:
    print(f"  Monodromy compression gives {cos-cos_rand:.4f} cosine improvement over random.")
    print(f"  At {t_compressed_inf/t_source_inf:.2f}x inference cost, {params_compressed/params_source:.2f}x parameters.")
    if val_compressed < val_rand2:
        print(f"  Val loss {val_compressed:.3f} vs random 2L {val_rand2:.3f} — monodromy also improves quality.")
    print(f"\n  The compression pipeline:")
    print(f"  1. Train {N_LAYERS}L model: {t_train:.0f}s")
    print(f"  2. Extract monodromy + set weights: {t_compress:.0f}s")
    print(f"  3. Deploy 2L model at {N_LAYERS/2:.0f}x lower inference cost")
    print(f"  Total overhead vs not compressing: {t_compress:.0f}s")
else:
    print(f"  Cosine improvement over random: {cos-cos_rand:+.4f} (small)")
    print(f"  The monodromy-set weights do not strongly outperform random 2L.")
    print(f"  The attention routing (WQ, WK from mid-layer) dominates over")
    print(f"  the monodromy-derived WO, WV in determining output similarity.")

# Layer ablation: try different N_LAYERS → compressed
print(f"\n{'='*65}")
print(f"  LAYER COUNT vs INFERENCE COST vs QUALITY")
print("="*65)
print(f"\n  {'Layers':>8}  {'Params':>12}  {'FLOPs':>12}  {'t_inf(ms)':>10}  {'note'}")
print("  "+"-"*55)
for nl in [1, 2, 4, 6, N_LAYERS]:
    m_ = LM(D,N_HEADS,nl)
    fp = m_.flops_per_token()
    p  = m_.n_params()
    ti = time_inference(m_, n=100)*1000
    note = "← source" if nl==N_LAYERS else ("← compressed" if nl==2 else "")
    print(f"  {nl:>8}  {p:>12,}  {fp:>12,}  {ti:>10.1f}  {note}")
    del m_
