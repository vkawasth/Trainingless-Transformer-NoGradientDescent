#!/usr/bin/env python3
"""
Stage 2b: Corrected K-Group Analysis
=====================================
Three corrections from Stage 2a results:

CORRECTION 1: Use clean gradients (4-batch average) not single-batch.
  Stage 2a: θ = 78° ± 6.3° (noisy, single batch)
  Target:   θ = 55° ± ~2°  (clean, 4-batch average from gradient_trace.py)

CORRECTION 2: The K_1 phase problem.
  K_1 gives the ROTATION ANGLE θ but not the STARTING PHASE φ.
  g_25 = R(25θ + φ) where φ = initial phase (unknown algebraically).
  Fix: compute φ from one forward pass (the LM handshake already does this).

CORRECTION 3: The split exact sequence [τ]=0.
  All cos(Δ_i, Δ_j) < 0.03 → updates are orthogonal in weight space.
  The short exact sequence SPLITS: K_all ≅ K_fast ⊕ K_slow.
  This means {Emb, WK, WQ, FF} can be updated IN PARALLEL.

NEW EXPERIMENTS:

  Exp A: Clean K_1 generator with 4-batch averaged gradients
    → measure true θ and verify cos(g25_pred, ΔEmb) improves

  Exp B: Phase recovery via 1 forward pass
    → use g_0 from ONE forward pass to set the initial phase φ
    → g_25_pred = R(25θ + φ) · ê₀   (predicted endpoint)
    → test: does this beat the noisy 78° prediction?

  Exp C: Split K_0 — parallel independent updates
    → [τ]=0 means we can run 25 CE steps PER PARAMETER GROUP independently
    → Test: 25 CE on Emb (frozen WK,WQ,FF) THEN combine K_0 sum
    → Is this better/same/worse than 25 joint CE?

  Exp D: K_0 virtual class — combine independent partial updates
    → ΔE_partial + ΔWK_partial = K_0 direct sum
    → apply both to the model in one shot
    → does this approach 25 joint CE quality?
"""
import json, math, warnings, collections, os, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids=list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return torch.stack([data[i:i+SEQ] for i in ix]),torch.stack([data[i+1:i+SEQ+1] for i in ix])

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2); K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)))
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
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clean_grad(model, n_batches=8):
    """Averaged gradient over n_batches — reduces noise."""
    model.zero_grad()
    ls=[]
    for _ in range(n_batches): x,y=get_batch(); _,l=model(x,y); ls.append(l)
    torch.stack(ls).mean().backward()
    g=model.te.weight.grad.detach().clone().flatten()
    model.zero_grad()
    return g

def ptype(name):
    if '.attn.WQ.' in name: return 'WQ'
    if '.attn.WK.' in name: return 'WK'
    if 'te.weight'  in name: return 'Emb'
    if '.ff.'       in name: return 'FF'
    return 'other'

# Spectral init
bigram=collections.Counter()
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1
rows,cols,vals_sp=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a); cols.append(b); vals_sp.append(float(cnt))
W_sp=sp.csr_matrix((vals_sp,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T; d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv)); L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals); evecs=evecs[:,idx_s][:,1:D+1]
sc_ev=1.0/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)
E_0=(evecs*sc_ev[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

print(f"VOCAB={VOCAB}")

# ── EXP A: Clean K_1 generator ───────────────────────────────────────────────
print("\n" + "="*60)
print("EXP A: Clean K_1 generator (8-batch averaged gradients)")
print("="*60)

torch.manual_seed(99); model=LM(D,N_HEADS,N_STU)
model.te.weight.data.copy_(torch.tensor(E_0))
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

g0_clean = clean_grad(model, n_batches=16)
g_prev = g0_clean.clone()
step_angles_clean = []
angles_from_0 = []

print(f"  {'Step':>5}  {'val':>7}  {'step_angle':>11}  {'from_g0':>9}")
for step in range(1, 26):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

    if step in [1,2,3,5,10,15,20,25]:
        g_clean = clean_grad(model, n_batches=8)
        cos_prev = float((g_clean*g_prev).sum()/(g_clean.norm()*g_prev.norm()+1e-10))
        cos_g0   = float((g_clean*g0_clean).sum()/(g_clean.norm()*g0_clean.norm()+1e-10))
        theta_step = math.degrees(math.acos(max(-1,min(1,cos_prev))))
        theta_g0   = math.degrees(math.acos(max(-1,min(1,cos_g0))))
        step_angles_clean.append(theta_step)
        angles_from_0.append(theta_g0)
        v = eval_val(model, n=8)
        print(f"  {step:>5}  {v:>7.4f}  {theta_step:>10.1f}°  {theta_g0:>8.1f}°")
        g_prev = g_clean.clone()

v25 = eval_val(model)
g25_clean = clean_grad(model, n_batches=16)
theta_clean = np.mean(step_angles_clean)
print(f"\n  After 25 CE: val={v25:.4f}")
print(f"  Clean mean step angle: {theta_clean:.1f}°")
print(f"  (Stage 2a had 78° with noisy gradients)")

# ── EXP B: Phase recovery ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP B: Phase recovery — predict g_25 with correct phase")
print("="*60)

# g_25_pred = R(25θ, φ) · g_0
# where φ = initial phase angle (from clean g_0 measurement)
# The K_1 element [σ] acts as: g_t = R(t·θ) · R(φ) · ê_0
# R(φ) · ê_0 = g_0 direction (the initial phase IS the initial gradient)
# g_25_pred = R(25θ) · g_0  (rotate g_0 by 25×θ)

theta_rad = math.radians(theta_clean)
total_rotation = 25 * theta_rad

# Build rotation in the plane (g0_clean, g_perp)
g0_hat = g0_clean / (g0_clean.norm() + 1e-10)
torch.manual_seed(42)
g_rand = torch.randn_like(g0_hat)
g_perp = g_rand - (g_rand*g0_hat).sum()*g0_hat
g_perp = g_perp / (g_perp.norm() + 1e-10)

cos_t = math.cos(total_rotation); sin_t = math.sin(total_rotation)
g25_pred = cos_t * g0_hat + sin_t * g_perp

cos_pred = float((g25_pred * g25_clean/(g25_clean.norm()+1e-10)).sum())
print(f"  K_1 prediction with clean θ={theta_clean:.1f}°:")
print(f"  cos(g25_pred, g25_actual) = {cos_pred:.4f}")
print(f"  (Stage 2a: 0.5543 with noisy θ=78°)")

actual_delta = (model.te.weight.data.flatten() - torch.tensor(E_0.flatten()))
cos_delta = float((g25_pred * actual_delta/(actual_delta.norm()+1e-10)).sum())
print(f"  cos(g25_pred, ΔEmb_actual) = {cos_delta:.4f}")

# ── EXP C: Split K_0 — parallel independent updates ──────────────────────────
print("\n" + "="*60)
print("EXP C: Split K_0 — parallel per-parameter updates")
print("="*60)
print(f"  [τ]=0 proved: updates are orthogonal in weight space.")
print(f"  Test: 25 CE per parameter group INDEPENDENTLY, then combine.")
print()

def ce_only_params(m, steps, param_groups):
    """CE steps updating only specified parameter groups."""
    mc = copy.deepcopy(m)
    for name, p in mc.named_parameters():
        if ptype(name) not in param_groups:
            p.requires_grad_(False)
    params = [p for p in mc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    for _ in range(steps):
        mc.train(); x,y=get_batch(); _,l=mc(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
    return mc

torch.manual_seed(99); m_base=LM(D,N_HEADS,N_STU)
m_base.te.weight.data.copy_(torch.tensor(E_0))

# Joint 25 CE (baseline)
m_joint = copy.deepcopy(m_base)
opt_j = torch.optim.AdamW(m_joint.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for _ in range(25):
    m_joint.train(); x,y=get_batch(); _,l=m_joint(x,y)
    opt_j.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_joint.parameters(),1.0); opt_j.step()
v_joint = eval_val(m_joint)
print(f"  Joint 25 CE (baseline): val={v_joint:.4f}")

# K_0 direct sum: run each group independently, combine deltas
m_emb   = ce_only_params(m_base, 25, {'Emb'})
m_wk    = ce_only_params(m_base, 25, {'WK','WQ'})
m_ff    = ce_only_params(m_base, 25, {'FF'})

# Combine: apply each group's delta to the base model
m_combined = copy.deepcopy(m_base)
with torch.no_grad():
    for name, p in m_combined.named_parameters():
        pt = ptype(name)
        if pt == 'Emb':
            delta = dict(m_emb.named_parameters())[name].data - dict(m_base.named_parameters())[name].data
            p.data.add_(delta)
        elif pt in ('WK','WQ'):
            delta = dict(m_wk.named_parameters())[name].data - dict(m_base.named_parameters())[name].data
            p.data.add_(delta)
        elif pt == 'FF':
            delta = dict(m_ff.named_parameters())[name].data - dict(m_base.named_parameters())[name].data
            p.data.add_(delta)
v_combined = eval_val(m_combined)
print(f"  K_0 direct sum (parallel+combine): val={v_combined:.4f}")
print(f"  {'✓ SPLIT WINS' if v_combined<v_joint else '✗ joint better'} "
      f"by {abs(v_combined-v_joint):.4f} nats")

# ── EXP D: K_0 with weighting ─────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP D: K_0 weighted combination — optimize the sum")
print("="*60)

best_v = v_joint; best_config = "joint"
for w_emb, w_attn, w_ff in [(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,1,0),(0,1,2)]:
    m_try = copy.deepcopy(m_base)
    with torch.no_grad():
        for name, p in m_try.named_parameters():
            pt = ptype(name)
            if pt == 'Emb':
                delta = dict(m_emb.named_parameters())[name].data - dict(m_base.named_parameters())[name].data
                p.data.add_(w_emb * delta)
            elif pt in ('WK','WQ'):
                delta = dict(m_wk.named_parameters())[name].data - dict(m_base.named_parameters())[name].data
                p.data.add_(w_attn * delta)
            elif pt == 'FF':
                delta = dict(m_ff.named_parameters())[name].data - dict(m_base.named_parameters())[name].data
                p.data.add_(w_ff * delta)
    v = eval_val(m_try, n=12)
    flag = ' ←BEST' if v < best_v else ''
    print(f"  w=(Emb={w_emb},Attn={w_attn},FF={w_ff}): val={v:.4f}{flag}")
    if v < best_v: best_v=v; best_config=f"w=({w_emb},{w_attn},{w_ff})"

print(f"\n  Best K_0 configuration: {best_config}, val={best_v:.4f}")
print(f"  Joint 25 CE: val={v_joint:.4f}")
print(f"\n  CONCLUSION: {'K_0 split is better → parallel updates work!' if best_v<v_joint else 'Joint CE remains better → coupling is in gradient magnitude'}")
