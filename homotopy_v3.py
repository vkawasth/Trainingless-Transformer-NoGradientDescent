#!/usr/bin/env python3
"""
Homotopy v3 — Temperature Schedule with Adam Predictor
=======================================================
The temperature homotopy is correct. The predictor was wrong.

VERIFIED FROM OUTPUT:
  |gWK| = 0.00010 at τ=20  (4984× smaller than |gEmb|)
  Temperature correctly decouples WK/WQ at high τ ✓

THE FIX:
  Predictor = Adam steps (not raw gradient step)
  Adam handles the high curvature (55°/step gradient rotation)
  that raw gradient steps cannot.

THE MECHANISM:
  High τ (20→4): only Emb+FF get meaningful gradient signal
    → Emb converges in the right direction WITHOUT WK/WQ interference
    → This is why 167 steps are needed normally: WK/WQ interfere with Emb
    
  Low τ (4→1): WK/WQ gradually activate
    → They adapt to the already-oriented Emb
    → Much faster convergence because Emb is already correct

COMPARISON:
  A: 167 Adam steps at τ=1 (standard)         val=0.99
  B: 20 τ-steps × 5 Adam each = 100 steps     val=?
  C: 20 τ-steps × 8 Adam each = 160 steps     val=?
  D: compare val at step 100 between A and B
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
print(f"VOCAB={VOCAB}")

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

class AttnT(nn.Module):
    def __init__(self,d,nh):
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h,tau=1.0):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/(tau*self.sc)
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
class BlockT(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=AttnT(d,nh); self.ff=FF(d)
    def forward(self,h,tau=1.0): return self.ff(self.attn(h,tau))
class LMT(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([BlockT(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None,tau=1.0):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h,tau)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)

def eval_val(m,n=20,tau=1.0):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y,tau=tau); ls.append(l.item())
    return float(np.mean(ls))

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
idx_s=np.argsort(evals)
evecs=evecs[:,idx_s][:,1:D+1]
scales=1.0/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)
E_0=(evecs*scales[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

# ── Run homotopy with Adam predictor at each τ ───────────────────────────────
def run_homotopy(E_0, tau_max, tau_min, n_tau_steps, adam_per_tau,
                 label="homotopy"):
    torch.manual_seed(99)
    model=LMT(D,N_HEADS,N_STU)
    model.te.weight.data.copy_(torch.tensor(E_0))

    # Single Adam optimizer — maintains momentum state across τ steps
    # This is KEY: momentum remembers direction even as τ changes
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

    tau_schedule=np.exp(np.linspace(np.log(tau_max),np.log(tau_min),n_tau_steps+1))
    total_steps=0
    print(f"\n  {label}: τ {tau_max}→{tau_min}, "
          f"{n_tau_steps} τ-steps × {adam_per_tau} Adam = "
          f"{n_tau_steps*adam_per_tau} total steps")
    print(f"  {'τ':>7}  {'val(τ=1)':>9}  {'steps':>6}")

    for tau in tau_schedule[1:]:
        for _ in range(adam_per_tau):
            model.train(); x,y=get_batch()
            _,l=model(x,y,tau=float(tau))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            total_steps+=1
        v=eval_val(model,n=6,tau=1.0)
        print(f"  τ={tau:>6.3f}  val={v:>9.4f}  step={total_steps:>4}")

    v_final=eval_val(model,n=30,tau=1.0)
    print(f"  Final: val={v_final:.4f}  ({total_steps} total Adam steps)")
    return v_final, model

print("="*60)
print("HOMOTOPY v3: TEMPERATURE SCHEDULE + ADAM PREDICTOR")
print("="*60)

# Test configurations
configs = [
    # (tau_max, tau_min, n_tau_steps, adam_per_tau)
    (20.0, 1.0, 20, 5),    # 100 steps — primary test
    (20.0, 1.0, 20, 8),    # 160 steps — near-reference budget
    (20.0, 1.0, 10, 10),   # 100 steps — fewer coarser τ steps
    (10.0, 1.0, 10, 5),    # 50 steps — aggressive
]

results = {}
for tau_max, tau_min, n_tau, k_adam in configs:
    label = f"τ{tau_max:.0f}→1, {n_tau}×{k_adam}={n_tau*k_adam}steps"
    v, _ = run_homotopy(E_0, tau_max, tau_min, n_tau, k_adam, label)
    results[label] = v

# Reference: 167 standard Adam steps
print("\n" + "="*60)
print("REFERENCE: standard Adam at τ=1")
print("="*60)
torch.manual_seed(99)
model_ref=LMT(D,N_HEADS,N_STU)
model_ref.te.weight.data.copy_(torch.tensor(E_0))
opt_ref=torch.optim.AdamW(model_ref.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
ref_vals={}
for s in range(1,168):
    model_ref.train(); x,y=get_batch()
    _,l=model_ref(x,y,tau=1.0)
    opt_ref.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_ref.parameters(),1.0); opt_ref.step()
    if s in [50,100,160,167]:
        v=eval_val(model_ref,n=10)
        ref_vals[s]=v
        print(f"  CE {s}: val={v:.4f}")
v_ref=eval_val(model_ref,n=40)

print(f"""
{'='*60}
RESULTS
{'='*60}
Reference (standard Adam):
  @  50 steps: val={ref_vals.get(50,'N/A')}
  @ 100 steps: val={ref_vals.get(100,'N/A')}
  @ 160 steps: val={ref_vals.get(160,'N/A')}
  @ 167 steps: val={v_ref:.4f}

Temperature homotopy:""")
for label, v in results.items():
    n_steps = int(label.split('=')[1].split('s')[0])
    ref_at_n = min(ref_vals.items(), key=lambda x: abs(x[0]-n_steps))[1] if ref_vals else '?'
    better = '✓ better' if isinstance(ref_at_n,float) and v < ref_at_n else '~ similar' if isinstance(ref_at_n,float) and abs(v-ref_at_n)<0.05 else '✗ worse'
    ref_str = f"{ref_at_n:.4f}" if isinstance(ref_at_n, float) else str(ref_at_n)
    print(f"  {label}: val={v:.4f}  vs ref@same_steps={ref_str}  {better}")

print(f"""
KEY QUESTION:
  Does temperature scheduling give better val at same step count?
  If yes: the ordered convergence (Emb first, then WK/WQ) is real.
  If no: Adam already handles the Kuramoto synchronization optimally.
""")
