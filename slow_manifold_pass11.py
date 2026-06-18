#!/usr/bin/env python3
"""
Pass 11: Interleaved Adam + LM Newton (Extended Pass 6)
=========================================================
The adjoint shooting collapsed to this: the 2-point BVP is best
solved by interleaving first-order steps (Adam) with second-order
corrections (LM Newton = Pass 6 extended to ALL parameters).

STRUCTURE:
  Repeat K times:
    1. N_ADAM Adam steps  (traverse the path)
    2. 1 LM Newton step   (second-order curvature correction)

  The LM step solves (H + μI)d = -∇L at the current point,
  correcting the accumulated curvature error from Adam.

  This is the correct homotopy predictor-corrector where:
  - Predictor = N_ADAM Adam steps (follow path tangent)
  - Corrector = LM Newton step (snap to solution manifold)

COMPARISON:
  A: 167 Adam steps (reference)
  B: 6 × (25 Adam + 1 LM)  = 150 Adam + 6 LM
  C: 3 × (50 Adam + 1 LM)  = 150 Adam + 3 LM
  D: temperature homotopy + LM (pants coproduct: two legs)

Requires: /tmp/model_post_pass6.pt (run build_pass6_checkpoint.py)
"""
import json, math, warnings, collections, os, copy, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): print(f"ERROR: {f} missing."); sys.exit(1)
if not os.path.exists('/tmp/model_post_pass6.pt'):
    print("ERROR: run build_pass6_checkpoint.py first"); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long); val_t=torch.tensor(val_ids,dtype=torch.long)
print(f"VOCAB={VOCAB}")

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
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,f):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def lm_step(model, mu=0.1, n_grad=20, n_hvp=8, n_cg=8):
    """One LM Newton step on ALL parameters. Returns (new_val, accepted)."""
    # Gradient
    model.zero_grad()
    ls=[]; 
    for _ in range(n_grad): x,y=get_batch(); _,l=model(x,y); ls.append(l)
    loss=torch.stack(ls).mean(); loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None
                 else torch.zeros(p.numel()) for p in model.parameters()]).detach()
    model.zero_grad()

    # HVP function
    def hvp(v):
        model.zero_grad()
        ls2=[]; 
        for _ in range(n_hvp): x,y=get_batch(); _,l=model(x,y); ls2.append(l)
        loss2=torch.stack(ls2).mean()
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gflat=torch.cat([gr.flatten() for gr in grads])
        gv=(gflat*v.detach()).sum()
        hv=torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)
        model.zero_grad()
        return torch.cat([h.flatten() for h in hv]).detach()

    # CG solve: (H + μI)d = -g
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=hvp(p_cg)+mu*p_cg
        alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d+=alpha*p_cg; r-=alpha*Hp
        rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new

    # Line search
    w0=model.flat_params(); l0=float(loss)
    for scale in [1.0,0.5,0.25,0.1]:
        model.set_flat(w0+scale*d)
        v_new=eval_val(model,n=8)
        if v_new < eval_val(model,n=3)+0.01 or scale==0.1:  # accept best
            return v_new, True
    model.set_flat(w0)
    return l0, False

# ── Load post-Pass-6 model ────────────────────────────────────────────────────
model=LM(D,N_HEADS,N_STU)
model.load_state_dict(torch.load('/tmp/model_post_pass6.pt',weights_only=True))
v0=eval_val(model); print(f"Post-Pass-6 val: {v0:.4f}")

# ── Run comparison ────────────────────────────────────────────────────────────
configs=[
    ("A: 167 Adam",           [(167,0)]),
    ("B: 6×(25 Adam+LM)",     [(25,1)]*6),
    ("C: 3×(50 Adam+LM)",     [(50,1)]*3),
    ("D: homotopy+LM",        "homotopy"),
]

results={}
for label,schedule in configs:
    m=copy.deepcopy(model)
    print(f"\n{label}")
    total_adam=0

    if schedule=="homotopy":
        # Pants coproduct: Leg1 = temperature homotopy (τ:20→1, 100 steps)
        #                  Leg2 = LM Newton at endpoint
        import collections as _col, scipy.sparse as _sp, scipy.sparse.linalg as _spla
        
        # Rebuild spectral embedding for homotopy model
        _bigram=_col.Counter()
        for _i in range(len(train_ids)-1):
            _a,_b=train_ids[_i],train_ids[_i+1]
            if _a<VOCAB and _b<VOCAB: _bigram[(_a,_b)]+=1
        _rows,_cols,_vals=[],[],[]
        for (_a,_b),_cnt in _bigram.items():
            _rows.append(_a); _cols.append(_b); _vals.append(float(_cnt))
        _W=_sp.csr_matrix((_vals,(_rows,_cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
        _W=_W+_W.T; _di=np.array(1.0/(_W.sum(1)+1e-8)).flatten()
        _Ds=_sp.diags(np.sqrt(_di)); _L=_sp.eye(VOCAB)-_Ds@_W@_Ds
        _ev,_evec=_spla.eigsh(_L,k=D+1,which='SM',tol=1e-4,maxiter=2000)
        _idx=np.argsort(_ev); _evec=_evec[:,_idx][:,1:D+1]
        _sc=1.0/(np.sqrt(_ev[_idx[1:D+1]])+1e-8)
        _E0=(_evec*_sc[np.newaxis,:]).astype(np.float32)
        _E0=(_E0/(_E0.std()+1e-8)*0.02)

        # Temperature-aware model (inline, no import)
        class _AttnT(nn.Module):
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
        class _BlockT(nn.Module):
            def __init__(self,d,nh): super().__init__(); self.attn=_AttnT(d,nh); self.ff=FF(d)
            def forward(self,h,tau=1.0): return self.ff(self.attn(h,tau))
        class _LMT(nn.Module):
            def __init__(self,d,nh,nl):
                super().__init__()
                self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
                self.blocks=nn.ModuleList([_BlockT(d,nh) for _ in range(nl)])
                self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
                self.head.weight=self.te.weight
                nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
            def forward(self,x,y=None,tau=1.0):
                h=self.te(x)+self.pe(torch.arange(x.shape[1]))
                for b in self.blocks: h=b(h,tau)
                logits=self.head(self.ln_f(h))
                return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
            def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
            def set_flat(self,f):
                idx=0
                for p in self.parameters(): n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

        torch.manual_seed(99); m=_LMT(D,N_HEADS,N_STU)
        m.te.weight.data.copy_(torch.tensor(_E0))

        # Leg 1: temperature homotopy τ:20→1, 20×5=100 Adam steps
        _tau_sched=np.exp(np.linspace(np.log(20.0),np.log(1.0),21))
        _opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
        _step=0
        for _tau in _tau_sched[1:]:
            for _ in range(5):
                m.train(); x,y=get_batch(); _,l=m(x,y,tau=float(_tau))
                _opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); _opt.step(); _step+=1
        _v_leg1=eval_val(m); print(f"  Leg1 (homotopy 100 steps): val={_v_leg1:.4f}")

        # Leg 2: LM Newton corrections (pants coproduct composition)
        for _lm_i in range(3):
            _v_lm,_acc=lm_step(m,mu=0.1)
            print(f"  Leg2 LM {_lm_i+1}: val={_v_lm:.4f} {'✓' if _acc else '~'}")
        
        v_final=eval_val(m,n=40)
        results[label]=v_final
        print(f"  FINAL: val={v_final:.4f}")
        continue

    opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for n_adam,n_lm in schedule:
        # Adam phase
        for s in range(n_adam):
            m.train(); x,y=get_batch(); _,l=m(x,y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            total_adam+=1
        v_adam=eval_val(m,n=10)
        print(f"  Adam {total_adam}: val={v_adam:.4f}",end='')

        # LM phase
        if n_lm>0:
            v_lm,accepted=lm_step(m,mu=0.1)
            print(f"  → LM: val={v_lm:.4f} {'✓' if accepted else '~'}")
        else:
            print()

    v_final=eval_val(m,n=40)
    results[label]=v_final
    print(f"  FINAL: val={v_final:.4f}")

print(f"\n{'='*55}")
print("RESULTS")
print(f"{'='*55}")
for label,v in results.items():
    print(f"  {label:<30}  val={v:.4f}")
