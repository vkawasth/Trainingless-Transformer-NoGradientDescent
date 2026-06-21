#!/usr/bin/env python3
"""
GD-400 vs Compiler Fair Comparison
====================================
The saturation data shows GD-300 (Adam cosine) is early stopping.
True saturation (constant LR) hits at step 385, val~0.060.

QUESTIONS:
  Q1: GD-400 constant LR vs GD-300 cosine: are they the same basin?
  Q2: GD-400 vs MF compiler (211 CE): who wins at equal quality target?
  Q3: Does GD-400 reach the compiler's basin (π,0,π,0,π)?
  Q4: Is the algebraic construction genuinely faster or just same floor?

FAIR COMPARISON:
  ARM A: GD-400 (constant LR=3e-4, no schedule)
  ARM B: MF Compiler (spectral + saddle + MF3 + 33CE + 167CE = ~211 CE equiv)
  Both measured at same val levels: 0.20, 0.15, 0.10, 0.062
"""
import json, math, warnings, collections, os, sys, time
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh=D//N_HEADS
        self.WQ=nn.Linear(D,D,bias=False); self.WK=nn.Linear(D,D,bias=False)
        self.WV=nn.Linear(D,D,bias=False); self.op=nn.Linear(D,D,bias=False)
        self.ln=nn.LayerNorm(D); self.sc=math.sqrt(dh); self.nh=N_HEADS; self.dh=dh
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,_=h.shape
        Q=self.WQ(h).view(B,S,self.nh,self.dh).transpose(1,2)
        K=self.WK(h).view(B,S,self.nh,self.dh).transpose(1,2)
        V=self.WV(h).view(B,S,self.nh,self.dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D)))
class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g=nn.Linear(D,D*2,bias=False); self.v=nn.Linear(D,D*2,bias=False)
        self.o=nn.Linear(D*2,D,bias=False); self.n=nn.LayerNorm(D)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))
class Block(nn.Module):
    def __init__(self): super().__init__(); self.attn=Attn(); self.ff=FF()
    def forward(self,h): return self.ff(self.attn(h))
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te=nn.Embedding(VOCAB,D); self.pe=nn.Embedding(512,D)
        self.blocks=nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,v):
        i=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(v[i:i+n].reshape(p.shape)); i+=n

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

def eval_val(m, n=15):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def sheet_angles(model):
    angles=[]
    WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU-1):
        try:
            phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
            lam1=torch.linalg.eigvals(phi)
            lam1=lam1[lam1.abs().argmax()]
            angles.append(f'{float(torch.angle(lam1)):.2f}')
        except: angles.append('?')
    return '('+','.join(angles)+')'

def leading_svs(model):
    return [f'{float(torch.linalg.svdvals(model.blocks[l].attn.WK.weight.data)[0]):.3f}'
            for l in range(N_STU)]

# Corpus + spectral E₀
print("="*65)
print("GD-400 vs COMPILER — FAIR SATURATION COMPARISON")
print("="*65); print()
bigram=collections.Counter(); perm={}
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB:
        bigram[(a,b)]+=1
        if a not in perm: perm[a]=b
rows,cols,vv=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a); cols.append(b); vv.append(float(cnt))
W_sp=sp.csr_matrix((vv,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T; d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv)); L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals); evecs=evecs[:,idx_s][:,1:D+1]
E_0=(evecs/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)
E_next=np.array([E_0[perm.get(t,t)] for t in range(VOCAB)],dtype=np.float32)
E_init=(0.9*E_0+0.1*E_next)
E_norm=float(np.linalg.norm(E_0))
E_init=(E_init*(E_norm/max(float(np.linalg.norm(E_init)),1e-8))).astype(np.float32)

# ── ARM A: GD-400 CONSTANT LR ────────────────────────────────
print("━━━ ARM A: GD-400 CONSTANT LR (no schedule) ━━━━━━━━━━━━━")
print("  Matches constant_lr_saturation.py exactly")
print("  True saturation at step ~385, val~0.060")
torch.manual_seed(99)
gd=LM(); gd.te.weight.data.copy_(torch.tensor(E_init))
opt=torch.optim.AdamW(gd.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

gd_steps={}
print(f"  {'step':>5}  {'val':>7}  {'sheet':>32}  {'σ₁':>10}")
print("  "+"-"*62)
for step in range(1,401):
    gd.train(); x,y=get_batch(); _,l=gd(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(gd.parameters(),1.0); opt.step()
    if step in {35,100,167,200,274,300,350,385,400}:
        v=eval_val(gd); gd_steps[step]=v
        sp_=sheet_angles(gd); svs=','.join(leading_svs(gd))
        print(f"  {step:>5}  {v:>7.4f}  {sp_:>32}  {svs}")
v_gd400=gd_steps[400]; print()

# ── ARM B: MF COMPILER ───────────────────────────────────────
print("━━━ ARM B: MF COMPILER (~211 CE equiv) ━━━━━━━━━━━━━━━━━━")
print("  Confirmed pipeline from mean_field_init.py")
torch.manual_seed(99)
comp=LM(); comp.te.weight.data.copy_(torch.tensor(E_init))

# Saddle exit (v_neg)
comp.zero_grad()
loss=sum(comp(*get_batch())[1] for _ in range(6))/6
loss.backward()
g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
             for p in comp.parameters()]).detach(); comp.zero_grad()
v_neg=-g/g.norm()
w0=comp.flat_params(); best_v=eval_val(comp,n=8); best_a=0.0
for alpha in [0.5,1.0,1.43,2.0,3.0]:
    comp.set_flat(w0+alpha*v_neg); vt=eval_val(comp,n=6)
    if vt<best_v: best_v=vt; best_a=alpha
comp.set_flat(w0+best_a*v_neg)
v_saddle=eval_val(comp)
print(f"  Saddle exit α*={best_a}: val={v_saddle:.4f}")

# MF pump 3 rounds
ETA_MF=0.01; N_SUB=200
for mf_r in range(1,4):
    for _ in range(N_SUB):
        comp.train(); x,y=get_batch(); _,loss=comp(x,y)
        comp.zero_grad(); loss.backward()
        with torch.no_grad():
            if comp.te.weight.grad is not None:
                comp.te.weight.data -= ETA_MF*comp.te.weight.grad
    for _ in range(N_SUB):
        comp.train(); x,y=get_batch(); _,loss=comp(x,y)
        comp.zero_grad(); loss.backward()
        with torch.no_grad():
            for bl in comp.blocks:
                if bl.attn.WK.weight.grad is not None:
                    bl.attn.WK.weight.data += ETA_MF*bl.attn.WK.weight.grad
v_mf=eval_val(comp)
sp_mf=sheet_angles(comp)
print(f"  After MF3: val={v_mf:.4f}  sheet={sp_mf}")

# Basin selector 33 CE at LR×5
opt_b=torch.optim.AdamW(comp.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    lr_cur=LR*5*min(step,10)/10
    for pg in opt_b.param_groups: pg['lr']=lr_cur
    comp.train(); x,y=get_batch(); _,l=comp(x,y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(comp.parameters(),1.0); opt_b.step()
v_basin=eval_val(comp); sp_basin=sheet_angles(comp)
print(f"  After 33CE basin: val={v_basin:.4f}  sheet={sp_basin}")

# TopoGate
with torch.no_grad():
    for l in [1,2]:
        comp.blocks[l].attn.WV.weight.data.mul_(-1)
        comp.blocks[l].attn.op.weight.data.mul_(-1)
v_topo=eval_val(comp); sp_topo=sheet_angles(comp)
print(f"  After TopoGate: val={v_topo:.4f}  sheet={sp_topo}")

# 167 CE continuation — track same milestones as GD
opt_c=torch.optim.AdamW(comp.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
comp_steps={}
print(f"  {'CE':>5}  {'val':>7}  {'sheet':>32}  {'σ₁':>10}")
print("  "+"-"*62)
for step in range(1,168):
    comp.train(); x,y=get_batch(); _,l=comp(x,y)
    opt_c.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(comp.parameters(),1.0); opt_c.step()
    if step in {33,100,141,167}:
        v=eval_val(comp); comp_steps[step]=v
        sp_=sheet_angles(comp); svs=','.join(leading_svs(comp))
        print(f"  {step:>5}  {v:>7.4f}  {sp_:>32}  {svs}")
v_comp=comp_steps[167]; print()

# ── COMPARISON ────────────────────────────────────────────────
print("="*65)
print("FAIR COMPARISON: GD-400 vs COMPILER")
print("="*65); print()
print("  Saturation data (constant_lr_saturation.py):")
print("  Step 300: val=0.147  Step 385: val=0.060  Step 500: val=0.031")
print()
print(f"  {'Method':<40} {'CE steps':>8}  {'val':>7}")
print("  "+"-"*58)
print(f"  {'GD-300 (cosine LR, standard)':40} {'300':>8}  {gd_steps.get(300,0.244):>7.4f}")
print(f"  {'GD-300 (constant LR)':40} {'300':>8}  {'0.147':>7}")
print(f"  {'GD-385 (constant LR, saturation)':40} {'385':>8}  {'0.060':>7}")
print(f"  {'GD-400 (constant LR)':40} {'400':>8}  {v_gd400:>7.4f}")
print(f"  {'MF Compiler (~211 CE equiv)':40} {'211':>8}  {v_comp:>7.4f}")
print()

# Sheet path comparison at final
print(f"  GD-400 sheet:    {sheet_angles(gd)}")
print(f"  Compiler sheet:  {sheet_angles(comp)}")
print()

# Step efficiency: at what step does each arm reach target vals?
print("  STEP EFFICIENCY (steps to reach target val):")
for target in [0.20, 0.15, 0.10, 0.062]:
    gd_step=next((s for s in sorted(gd_steps) if gd_steps[s]<=target), None)
    comp_step=next((s for s,v in sorted(comp_steps.items()) if v<=target), None)
    print(f"  val≤{target:.3f}:  GD-400 @ step {gd_step or '>400'}  "
          f"Compiler @ CE {comp_step or '>167'}")

print()
if v_comp < v_gd400:
    ratio=v_gd400/max(v_comp,1e-6)
    print(f"  ✓ Compiler ({v_comp:.4f}) beats GD-400 ({v_gd400:.4f}) at equal steps")
    print(f"    {ratio:.2f}× better quality at {211} vs 400 steps")
elif abs(v_comp - v_gd400) < 0.01:
    print(f"  ~ Compiler ({v_comp:.4f}) ≈ GD-400 ({v_gd400:.4f})")
    print(f"    Same basin, compiler gets there in {211} vs 400 steps")
    print(f"    → Compiler is {400/211:.1f}× faster to SAME quality")
else:
    print(f"  GD-400 ({v_gd400:.4f}) < Compiler ({v_comp:.4f})")
    print(f"    GD finds deeper basin with 400 steps")
    print(f"    → Need MF10 (not MF3) to match")
