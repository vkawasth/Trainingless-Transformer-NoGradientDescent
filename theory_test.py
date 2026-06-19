#!/usr/bin/env python3
"""
Theory Test: Verify GT predictions on D=512 (same class) vs D=256 (base)
=========================================================================

PREDICTIONS (0-pass, from corpus + architecture):
  Base  (D=256, L=6): Γ=314.6, w_FF~2-3.5, n_K0≈13, |gEmb/gWK|≈254×
  Wider (D=512, L=6): Γ=314.6, w_FF~same,  n_K0≈13, |gEmb/gWK|≈same×

  Same Γ → same GT class → same qualitative behavior.

WHAT WE MEASURE AND COMPARE TO PREDICTIONS:
  1. |gEmb|/|gWK| ratio at init (predicted: ~378×, NTK; measured base: 254×)
  2. K₁ attractor angle (predicted: ~45° from g_0, same for both)
  3. w_FF empirical scan (predicted: same NTK estimate 1.18 vs 1.25)
  4. n_K0 crossover (predicted: ≈13 for both)
  5. val at 25 joint CE and 13 K₀ split

FALSIFICATION:
  IF Γ predicts equivalence and it is NOT equivalent → GT claim fails.
  IF Γ differs (base vs deeper) and behavior IS equivalent → GT claim fails.
"""
import json, math, warnings, collections, os, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

BATCH=8; SEQ=64; LR=3e-4; beta1=0.9; beta2=0.95
for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): print(f"ERROR: {f} missing."); import sys; sys.exit(1)

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

def make_model(D,N_HEADS,N_STU,VOCAB):
    class Attn(nn.Module):
        def __init__(self):
            super().__init__(); dh=D//N_HEADS
            self.WQ=nn.Linear(D,D,bias=False); self.WK=nn.Linear(D,D,bias=False)
            self.WV=nn.Linear(D,D,bias=False); self.op=nn.Linear(D,D,bias=False)
            self.ln=nn.LayerNorm(D); self.sc=math.sqrt(dh); self.nh=N_HEADS; self.dh=dh
            for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
        def forward(self,h):
            B,S,D_=h.shape
            Q=self.WQ(h).view(B,S,self.nh,self.dh).transpose(1,2)
            K=self.WK(h).view(B,S,self.nh,self.dh).transpose(1,2)
            V=self.WV(h).view(B,S,self.nh,self.dh).transpose(1,2)
            sc=Q@K.transpose(-2,-1)/self.sc
            mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
            sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
            return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)))
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
    return LM()

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def ptype(name):
    if 'te.weight' in name: return 'Emb'
    if '.attn.WQ.' in name: return 'WQ'
    if '.attn.WK.' in name: return 'WK'
    if '.ff.' in name: return 'FF'
    return 'other'

def measure_grad_ratios(model, n_batches=8):
    model.zero_grad()
    ls=[]
    for _ in range(n_batches): x,y=get_batch(); _,l=model(x,y); ls.append(l)
    torch.stack(ls).mean().backward()
    group_norms={}
    for name,p in model.named_parameters():
        if p.grad is None: continue
        pt=ptype(name)
        gn=float(p.grad.norm())
        group_norms[pt]=group_norms.get(pt,0)+gn**2
    model.zero_grad()
    return {k:v**0.5 for k,v in group_norms.items()}

def spectral_init(D,VOCAB):
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
    return (E_0/(E_0.std()+1e-8)*0.02)

def k0_split(base_model, D, n_steps, w_ff, lr_mult=2.0):
    params_base={n:p.data.clone() for n,p in base_model.named_parameters()}
    # Branch 1: Emb+FF
    m1=copy.deepcopy(base_model)
    for name,p in m1.named_parameters():
        if ptype(name) not in {'Emb','FF'}: p.requires_grad_(False)
    p1=[p for p in m1.parameters() if p.requires_grad]
    opt1=torch.optim.AdamW(p1,lr=LR*lr_mult,betas=(beta1,beta2),weight_decay=0.1)
    for _ in range(n_steps):
        m1.train(); x,y=get_batch(); _,l=m1(x,y)
        opt1.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p1,1.0); opt1.step()
    # Branch 2: WK+WQ
    m2=copy.deepcopy(base_model)
    for name,p in m2.named_parameters():
        if ptype(name) not in {'WK','WQ'}: p.requires_grad_(False)
    p2=[p for p in m2.parameters() if p.requires_grad]
    opt2=torch.optim.AdamW(p2,lr=LR*lr_mult,betas=(beta1,beta2),weight_decay=0.1)
    for _ in range(n_steps):
        m2.train(); x,y=get_batch(); _,l=m2(x,y)
        opt2.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p2,1.0); opt2.step()
    # Combine
    m_out=copy.deepcopy(base_model)
    with torch.no_grad():
        for name,p in m_out.named_parameters():
            pt=ptype(name)
            d1=dict(m1.named_parameters())[name].data-params_base[name]
            d2=dict(m2.named_parameters())[name].data-params_base[name]
            if pt=='Emb': p.data.add_(d1)
            elif pt=='FF': p.data.add_(w_ff*d1)
            elif pt in ('WK','WQ'): p.data.add_(d2)
    return m_out

# ── PRINT PREDICTIONS BEFORE ANY TRAINING ─────────────────────────────────────
print("="*65)
print("THEORY TEST: PREDICTIONS vs MEASUREMENTS")
print("="*65)
print()

configs=[
    ("BASE  (D=256,L=6,H=4)", 256, 6, 4, 1017/(1347*0.0004*6)),
    ("WIDER (D=512,L=6,H=4)", 512, 6, 4, 1017/(1347*0.0004*6)),
]

print("PREDICTIONS (0 passes, from corpus+architecture):")
print(f"  {'Model':<25} {'Γ':>8}  {'GT class':>10}  {'w_FF(NTK)':>10}  {'n_K0':>6}  {'gEmb/gWK':>10}")
print("  "+"-"*72)
for label,D,L,H,Gamma in configs:
    wFF=1+np.sqrt(VOCAB/(SEQ*D))
    n_k0=max(10,int(0.5/(1-beta2)))
    ratio=1017/(1347*0.0004)/5
    print(f"  {label:<25} {Gamma:>8.1f}  {'same' if Gamma==314.6 else 'differ':>10}  {wFF:>10.2f}  {n_k0:>6}  {ratio:>8.0f}×")

print()
print("RUNNING EXPERIMENTS...")
print()

results={}
for label,D,L,N_HEADS,Gamma in configs:
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    
    torch.manual_seed(99)
    E_0=spectral_init(D,VOCAB)
    model=make_model(D,N_HEADS,L,VOCAB)
    model.te.weight.data.copy_(torch.tensor(E_0))
    v0=eval_val(model,n=10)
    
    # Measure 1: gradient ratios at init
    gnorms=measure_grad_ratios(model,n_batches=8)
    ratio_measured=gnorms.get('Emb',0)/max(gnorms.get('WK',1e-8),1e-8)
    print(f"  Init val: {v0:.4f}")
    print(f"  |gEmb|/|gWK| = {ratio_measured:.0f}×  (predicted ~254-378×)")
    
    # Measure 2: K₁ attractor angle
    model.zero_grad()
    ls=[]; [ls.append(model(*get_batch())[1]) for _ in range(8)]
    torch.stack(ls).mean().backward()
    g0=model.te.weight.grad.detach().clone().flatten(); model.zero_grad()
    
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(beta1,beta2),weight_decay=0.1)
    for s in range(1,26):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if s in [2,5,13,25]:
            model.zero_grad()
            ls2=[]; [ls2.append(model(*get_batch())[1]) for _ in range(8)]
            torch.stack(ls2).mean().backward()
            gt=model.te.weight.grad.detach().flatten(); model.zero_grad()
            cos_g0=float((gt*g0).sum()/(gt.norm()*g0.norm()+1e-10))
            angle=math.degrees(math.acos(max(-1,min(1,cos_g0))))
            v=eval_val(model,n=8)
            if s in [2,13,25]:
                print(f"  Step {s:>3}: val={v:.4f}  angle_from_g0={angle:.1f}°")
    
    v25=eval_val(model); print(f"  Joint 25 CE: val={v25:.4f}")
    
    # Measure 3: K₀ split — scan w_FF at 13 steps
    torch.manual_seed(99); base=make_model(D,N_HEADS,L,VOCAB)
    base.te.weight.data.copy_(torch.tensor(E_0))
    print(f"  K₀ split scan (n=13, LR×2):")
    best_v=v25; best_wff=None
    for wff in [1.5,2.0,2.5,3.0,3.5,4.0]:
        mc=k0_split(base,D,13,wff,lr_mult=2.0)
        v=eval_val(mc,n=12)
        flag='←' if v<best_v else ''
        print(f"    w_FF={wff}: val={v:.4f} {flag}")
        if v<best_v: best_v=v; best_wff=wff
    
    results[label]={'Gamma':Gamma,'ratio':ratio_measured,'v25':v25,'best_wff':best_wff,'best_v':best_v}
    print(f"  → Best w_FF={best_wff}, val={best_v:.4f} vs joint 25 CE={v25:.4f}")
    beats='✓ K₀ BEATS joint' if best_v<v25 else '✗ joint better'
    print(f"  → {beats}")

print()
print("="*65)
print("RESULTS vs PREDICTIONS")
print("="*65)
print()
print(f"  {'Quantity':<30} {'Prediction':>15} {'BASE meas':>12} {'WIDER meas':>12} {'Match?'}")
print("  "+"-"*78)

base_r=results['BASE  (D=256,L=6,H=4)']
wider_r=results['WIDER (D=512,L=6,H=4)']

checks=[
    ("Γ same for both",        "Yes",     f"{base_r['Gamma']:.0f}", f"{wider_r['Gamma']:.0f}",
     "✓" if base_r['Gamma']==wider_r['Gamma'] else "✗"),
    ("|gEmb/gWK| (pred ~254×)", "~254×",   f"{base_r['ratio']:.0f}×", f"{wider_r['ratio']:.0f}×",
     "✓" if 100<base_r['ratio']<600 else "✗"),
    ("K₀ beats joint 25 CE",   "Yes",     "✓" if base_r['best_v']<base_r['v25'] else "✗",
     "✓" if wider_r['best_v']<wider_r['v25'] else "✗",
     "✓" if base_r['best_v']<base_r['v25'] and wider_r['best_v']<wider_r['v25'] else "✗"),
    ("Best w_FF similar",       "~same",   f"{base_r['best_wff']}", f"{wider_r['best_wff']}",
     "✓" if base_r['best_wff']==wider_r['best_wff'] else "~"),
]
for q in checks:
    print(f"  {q[0]:<30} {q[1]:>15} {q[2]:>12} {q[3]:>12}  {q[4]}")

print()
print("GT EQUIVALENCE VERDICT:")
if all(c[4]=='✓' for c in checks):
    print("  ✓ CONFIRMED: Same Γ → same GT class → same behavior")
else:
    print("  Partial: some predictions match, some differ")
    print("  Examine which predictions failed and why.")
