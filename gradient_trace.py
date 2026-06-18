#!/usr/bin/env python3
"""
Gradient Trace — What Actually Happens in 167 Steps
=====================================================
Track precisely:
  - Which parameters move and by how much each step
  - What the gradient direction is vs parameter direction
  - How val decomposes: which layers contribute at each checkpoint
  - The parameter trajectories: are they monotone? oscillatory?
  - Cross-parameter correlations: when WK moves, does E follow?
"""
import json, math, warnings, collections, os, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

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
    if '.attn.WV.' in name: return 'WV'
    if '.attn.op.' in name: return 'WO'
    if 'te.weight'  in name: return 'Emb'
    if '.ff.'       in name: return 'FF'
    return 'other'

# Spectral init
bigram=collections.Counter()
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1
rows,cols,vals=[],[],[]
for (a,b),cnt in bigram.items():
    rows.append(a); cols.append(b); vals.append(float(cnt))
W_sp=sp.csr_matrix((vals,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T
d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv))
L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx=np.argsort(evals)
evecs=evecs[:,idx][:,1:D+1]
scales=1.0/(np.sqrt(evals[idx[1:D+1]])+1e-8)
E_0=(evecs*scales[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

torch.manual_seed(99)
model=LM(D,N_HEADS,N_STU)
model.te.weight.data.copy_(torch.tensor(E_0))

# Snapshot initial weights
snap0={n:p.data.clone() for n,p in model.named_parameters()}
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

# ── Track per-step ────────────────────────────────────────────────────────────
TRACK_STEPS=list(range(1,26))+[30,35,40,50,60,75,100,125,150,167]
records=[]

print("GRADIENT TRACE — 167 STEPS")
print(f"{'Step':>5} {'val':>7} {'|gEmb|':>8} {'|gWK|':>8} {'|gWQ|':>8} "
      f"{'|gFF|':>8} {'ΔEmb/E':>8} {'ΔWK/W':>8} {'cos(gEmb,ΔEmb)':>15}")
print("-"*95)

prev_params={n:p.data.clone() for n,p in model.named_parameters()}

for step in range(1,168):
    model.train()
    # Accumulate gradient over 4 batches for stability
    model.zero_grad()
    for _ in range(4):
        x,y=get_batch(); _,l=model(x,y)
        (l/4).backward()
    
    # Capture gradient BEFORE optimizer step
    grads={n:p.grad.clone() if p.grad is not None else torch.zeros_like(p.data)
           for n,p in model.named_parameters()}
    
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); opt.zero_grad()
    
    if step in TRACK_STEPS:
        v=eval_val(model,n=10)
        
        # Per-type gradient norms
        gnorms={}
        for n,g in grads.items():
            pt=ptype(n)
            gnorms[pt]=gnorms.get(pt,0)+float(g.norm()**2)
        gnorms={k:math.sqrt(v) for k,v in gnorms.items()}
        
        # Displacement from init
        disp={}
        for n,p in model.named_parameters():
            pt=ptype(n)
            d=float((p.data-snap0[n]).norm())
            w=float(snap0[n].norm())
            disp[pt]=disp.get(pt,0.0)+d/max(w,1e-8)
        
        # Cosine between gradient direction and displacement direction
        # for embedding: does gradient point toward the displacement?
        g_emb=grads['te.weight'].flatten()
        d_emb=(model.te.weight.data-snap0['te.weight']).flatten()
        cos_emb=float((g_emb*d_emb).sum()/(g_emb.norm()*d_emb.norm()+1e-10))
        
        # Step-to-step delta (how much did params change THIS step)
        curr_params={n:p.data.clone() for n,p in model.named_parameters()}
        step_delta={}
        for n in curr_params:
            pt=ptype(n)
            d=float((curr_params[n]-prev_params[n]).norm())
            w=float(curr_params[n].norm())
            step_delta[pt]=step_delta.get(pt,0.0)+d/max(w,1e-8)
        prev_params=curr_params
        
        rec={'step':step,'val':v,'gnorms':gnorms,'disp':disp,
             'cos_emb':cos_emb,'step_delta':step_delta}
        records.append(rec)
        
        print(f"{step:>5} {v:>7.4f} "
              f"{gnorms.get('Emb',0):>8.4f} "
              f"{gnorms.get('WK',0):>8.4f} "
              f"{gnorms.get('WQ',0):>8.4f} "
              f"{gnorms.get('FF',0):>8.4f} "
              f"{disp.get('Emb',0):>8.4f} "
              f"{disp.get('WK',0):>8.4f} "
              f"{cos_emb:>15.4f}")

# ── Phase analysis ────────────────────────────────────────────────────────────
print()
print("PHASE ANALYSIS — what changes in each phase:")
print()
phases=[(1,10,'Phase 1'),(11,50,'Phase 2'),(51,100,'Phase 3'),(101,167,'Phase 4')]
for s_start,s_end,label in phases:
    phase_recs=[r for r in records if s_start<=r['step']<=s_end]
    if not phase_recs: continue
    r0=phase_recs[0]; r1=phase_recs[-1]
    print(f"  {label} (steps {s_start}-{s_end}): val {r0['val']:.4f}→{r1['val']:.4f}")
    
    # Which param type has largest gradient in this phase?
    avg_gnorms={}
    for r in phase_recs:
        for k,v in r['gnorms'].items():
            avg_gnorms[k]=avg_gnorms.get(k,[]); avg_gnorms[k].append(v)
    avg_gnorms={k:np.mean(v) for k,v in avg_gnorms.items()}
    dominant=max(avg_gnorms,key=avg_gnorms.get)
    print(f"    Dominant gradient: {dominant} ({avg_gnorms[dominant]:.4f})")
    
    # Displacement this phase
    disp_phase={}
    for r in phase_recs:
        for k,v in r['step_delta'].items():
            disp_phase[k]=disp_phase.get(k,[]); disp_phase[k].append(v)
    disp_phase={k:np.mean(v) for k,v in disp_phase.items()}
    top3=sorted(disp_phase.items(),key=lambda x:-x[1])[:3]
    print(f"    Most movement: {', '.join(f'{k}({v:.4f})' for k,v in top3)}")
    
    # cos_emb trend
    cos_vals=[r['cos_emb'] for r in phase_recs]
    print(f"    cos(gEmb, ΔEmb): {np.mean(cos_vals):+.4f} "
          f"({'gradient aligned with displacement' if np.mean(cos_vals)>0 else 'anti-aligned — oscillating'})")
    print()

# ── The key diagnostic ────────────────────────────────────────────────────────
print("KEY DIAGNOSTIC:")
print()
print("  cos(gEmb, ΔEmb) tells us whether the gradient is pushing")
print("  Emb TOWARD its final position (positive) or AWAY (negative/oscillating)")
print()
print("  If cos > 0 throughout: gradient descent is on track, just slow")
print("  If cos oscillates: Emb is being pulled in different directions")
print("     by competing forces (WK coupling vs FF coupling)")
print("     → This is why one-shot fails: the direction changes each step")
print()

# Save for analysis
import json as _json
with open('/tmp/gradient_trace.json','w') as f:
    _json.dump([{k:(v if not isinstance(v,dict) else
                    {kk:float(vv) for kk,vv in v.items()})
                 for k,v in r.items()} for r in records], f, indent=2)
print("Saved trace → /tmp/gradient_trace.json")
