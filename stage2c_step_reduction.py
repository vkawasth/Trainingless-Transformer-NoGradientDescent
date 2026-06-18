#!/usr/bin/env python3
"""
Stage 2c: Full optimization of K₀ split to close the 0.11 nat gap
===================================================================
K₀ LR×2 at 12 steps gives val=3.559 vs joint 25 steps val=3.449.
Gap: 0.110 nats. Close it with:

1. Optimal FF weight (scan 1.5×, 2×, 2.5×, 3×, 4×)
2. Warmup LR schedule (cosine warmup over 12 steps)
3. Sequencing: Emb+FF phase, then Attn phase
4. Scan total steps: 10, 11, 12, 13, 14 to find exact crossover

THE MECHANISM:
  K₁ attractor: gradient locks to ~45° from g₀ in step 1.
  Adam momentum deficit: steps 1-5 are momentum-starved.
  K₀ split removes interference → each group converges independently.
  LR schedule: higher LR early (compensate momentum deficit),
               lower LR late (fine-tune without overshoot).
  FF×n: n=2 from Stage 2b, scan for optimal value.
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
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
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

def ptype(name):
    if '.attn.WQ.' in name: return 'WQ'
    if '.attn.WK.' in name: return 'WK'
    if 'te.weight'  in name: return 'Emb'
    if '.ff.'       in name: return 'FF'
    return 'other'

def k0_split(base, n_steps, lr_emb_ff, lr_attn, w_ff, cosine_schedule=False):
    """K₀ split: Emb+FF and Attn updated independently, combined with FF×w_ff."""
    params_base = {n:p.data.clone() for n,p in base.named_parameters()}
    
    def get_lr(step, n_steps, base_lr, schedule):
        if not schedule: return base_lr
        # Cosine warmup: peak at step 3, decay to base_lr/2
        if step <= 3:
            return base_lr * (0.5 + 0.5 * step/3)
        return base_lr * (0.5 + 0.5 * math.cos(math.pi * (step-3)/(n_steps-3)))
    
    # Branch 1: Emb + FF
    m1 = copy.deepcopy(base)
    for name, p in m1.named_parameters():
        if ptype(name) not in {'Emb','FF'}: p.requires_grad_(False)
    p1 = [p for p in m1.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1, lr=lr_emb_ff, betas=(0.9,0.95), weight_decay=0.1)
    for step in range(1, n_steps+1):
        if cosine_schedule:
            for pg in opt1.param_groups: pg['lr'] = get_lr(step, n_steps, lr_emb_ff, True)
        m1.train(); x,y=get_batch(); _,l=m1(x,y)
        opt1.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(p1, 1.0); opt1.step()
    
    # Branch 2: WK + WQ
    m2 = copy.deepcopy(base)
    for name, p in m2.named_parameters():
        if ptype(name) not in {'WK','WQ'}: p.requires_grad_(False)
    p2 = [p for p in m2.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(p2, lr=lr_attn, betas=(0.9,0.95), weight_decay=0.1)
    for step in range(1, n_steps+1):
        if cosine_schedule:
            for pg in opt2.param_groups: pg['lr'] = get_lr(step, n_steps, lr_attn, True)
        m2.train(); x,y=get_batch(); _,l=m2(x,y)
        opt2.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(p2, 1.0); opt2.step()
    
    # Combine: Emb×1 + FF×w_ff + Attn×1
    m_out = copy.deepcopy(base)
    with torch.no_grad():
        for name, p in m_out.named_parameters():
            pt = ptype(name)
            d1 = dict(m1.named_parameters())[name].data - params_base[name]
            d2 = dict(m2.named_parameters())[name].data - params_base[name]
            if pt == 'Emb':
                p.data.add_(d1)
            elif pt == 'FF':
                p.data.add_(w_ff * d1)
            elif pt in ('WK','WQ'):
                p.data.add_(d2)
    return m_out

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

torch.manual_seed(99); base=LM(D,N_HEADS,N_STU)
base.te.weight.data.copy_(torch.tensor(E_0))

print(f"STAGE 2c: K₀ SPLIT OPTIMIZATION — TARGET: MATCH 25 JOINT IN ≤12 STEPS")
print("="*70)

# Reference: joint CE at key points
refs = {}
for n in [12, 15, 20, 25]:
    mc = copy.deepcopy(base)
    opt = torch.optim.AdamW(mc.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for _ in range(n):
        mc.train(); x,y=get_batch(); _,l=mc(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(),1.0); opt.step()
    refs[n] = eval_val(mc)
print(f"\nReference (joint CE):")
for n,v in refs.items(): print(f"  {n} steps: val={v:.4f}")
v_target = refs[25]

print(f"\nTarget: val ≤ {v_target:.4f} (= 25 joint CE)")
print()

# Scan FF weight
print("Scan 1: FF weight (n=12, LR×2, no schedule)")
print(f"  {'w_FF':>6}  {'val':>7}  {'vs target':>10}")
best_wff=2.0; best_v=999
for w_ff in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    mc = k0_split(base, 12, LR*2, LR*2, w_ff, cosine_schedule=False)
    v = eval_val(mc)
    flag = ' ←BEST' if v < best_v else ''
    print(f"  {w_ff:>6.1f}  {v:>7.4f}  {v-v_target:>+10.4f}{flag}")
    if v < best_v: best_v=v; best_wff=w_ff

# Scan with cosine schedule
print(f"\nScan 2: Cosine LR schedule (n=12, w_FF={best_wff})")
print(f"  {'LR_ef':>8}  {'LR_attn':>8}  {'sched':>8}  {'val':>7}  {'vs target':>10}")
best_config_v = best_v
for lr_ef_mult in [1.5, 2.0, 2.5, 3.0]:
    for cosine in [False, True]:
        mc = k0_split(base, 12, LR*lr_ef_mult, LR*lr_ef_mult, best_wff, cosine_schedule=cosine)
        v = eval_val(mc)
        sched = 'cosine' if cosine else 'flat'
        flag = ' ←BEST' if v < best_config_v else ''
        print(f"  {lr_ef_mult:>8.1f}  {lr_ef_mult:>8.1f}  {sched:>8}  {v:>7.4f}  {v-v_target:>+10.4f}{flag}")
        if v < best_config_v: best_config_v=v; best_lr=lr_ef_mult; best_cosine=cosine

# Scan step count with best config
print(f"\nScan 3: Step count sweep with best config")
print(f"  LR×{best_lr:.1f}, w_FF={best_wff}, cosine={'yes' if best_cosine else 'no'}")
print(f"  {'Steps':>7}  {'val':>7}  {'vs target':>10}  {'vs joint_same':>13}")
for n in [8, 9, 10, 11, 12, 13, 14, 15]:
    mc = k0_split(base, n, LR*best_lr, LR*best_lr, best_wff, cosine_schedule=best_cosine)
    v = eval_val(mc)
    vs_joint = refs.get(n, refs[12])  # compare to joint at same steps
    joint_n = refs.get(n, None)
    flag = ' ✓ BEATS TARGET' if v <= v_target else ''
    joint_str = f"{v-joint_n:+.4f}" if joint_n else "  N/A"
    print(f"  {n:>7}  {v:>7.4f}  {v-v_target:>+10.4f}  {joint_str:>13}{flag}")

print(f"\n{'='*70}")
print(f"SUMMARY: Target (25 joint CE) = {v_target:.4f}")
print(f"  Best K₀ split found: val={best_config_v:.4f}")
if best_config_v <= v_target:
    print(f"  ✓ TARGET ACHIEVED — 25 CE → 12 CE steps confirmed")
else:
    print(f"  Gap remaining: {best_config_v-v_target:.4f} nats")
    print(f"  Progress: {(refs[25]-best_config_v)/(refs[25]-refs[12])*100:.0f}% of the gap closed")
