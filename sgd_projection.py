#!/usr/bin/env python3
"""
SGD + Pants Coproduct Projection
==================================
Adam cancels gradient projections via adaptive normalisation.
SGD does not — the projection survives directly into the weight update.

W_{t+1} = W_t - lr × P_l @ ∂L/∂W_l

where P_l = U_l U_l^T projects onto the active subspace of δJ_l.

Tests:
  A: Standard SGD (with momentum)
  B: SGD + hard projection (every step)
  C: SGD + hard projection (every 5 steps)
  D: Adam baseline (reference)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64
LR_SGD=0.05; LR_ADAM=3e-4; MOMENTUM=0.9
PROJ=16; TARGET=4.0; MAX_STEPS=400

print(f"\n{'='*65}")
print(f"  SGD + PANTS COPRODUCT PROJECTION")
print(f"  Does projection survive SGD's non-adaptive update?")
print(f"  d={D}  layers={N_LAYERS}  proj={PROJ}")
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

def eval_val(model,n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr_sgd(s,total=MAX_STEPS,warmup=50):
    # cosine decay with warmup, no weight decay issues
    if s<=warmup: return LR_SGD*s/warmup
    return LR_SGD*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def clr_adam(s,total=MAX_STEPS,warmup=50):
    if s<=warmup: return LR_ADAM*s/warmup
    return LR_ADAM*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Active subspace computation ───────────────────────────────────────────────
def get_projections(model, x_ref, pos, m=PROJ):
    """Returns per-layer projection matrices P_l = U_l U_l^T  [d,d]."""
    projs = []
    with torch.enable_grad():
        hs = []
        h = model.te(x_ref)+model.pe(torch.arange(x_ref.shape[1]))
        hs.append(h.detach())
        for b in model.blocks: h=b(h); hs.append(h.detach())

        for l in range(model._nl):
            h_in = hs[l][0]                          # [SEQ, D]
            seq,d_ = h_in.shape; m_=min(m,seq,d_)
            _,_,Vt = torch.linalg.svd(h_in,full_matrices=False)
            U_basis = Vt[:m_,:].T.detach()           # [D, m]

            # Jacobian via vjp
            J = torch.zeros(m_,m_)
            for i in range(m_):
                h_node=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
                h_out=model.blocks[l](h_node)[0]
                v=h_out[0,pos,:] if h_out.dim()==3 else h_out[pos,:]
                (v*U_basis[:,i]).sum().backward()
                g=h_node.grad
                g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
                J[:,i]=U_basis.T@g

            dJ=(J.T-torch.eye(m_)).numpy()
            sv=np.linalg.svd(dJ,compute_uv=False)
            rank=max(int(np.sum(sv>sv[0]*0.10)) if sv[0]>1e-8 else 1, 1)

            # Active subspace in d-space
            U_sv,_,_=np.linalg.svd(dJ)
            U_act_m=U_sv[:,:rank]                    # [m, rank]
            U_act_d=U_basis.numpy()@U_act_m          # [D, rank]
            U_t=torch.tensor(U_act_d,dtype=torch.float32)
            P=U_t@U_t.T                              # [D, D]
            projs.append({'P':P,'rank':rank,'live_frac':rank/m_})

    return projs

def apply_projection(model, projs):
    """Project weight gradients onto active subspace per layer."""
    for l,sp in enumerate(projs):
        P=sp['P']  # [D, D]
        for param in model.blocks[l].parameters():
            if param.grad is None: continue
            g=param.grad.data
            if g.dim()==2:
                d_out,d_in=g.shape
                if d_out==D:   param.grad.data=P@g
                elif d_in==D:  param.grad.data=g@P.T
            # LayerNorm scalars: leave unchanged

# ── Training runs ─────────────────────────────────────────────────────────────
def run(name, use_adam=False, project=False, proj_every=1, seed=42):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS); model._nl=N_LAYERS

    if use_adam:
        opt=torch.optim.AdamW(model.parameters(),lr=LR_ADAM,
                               betas=(0.9,0.95),weight_decay=0.1)
    else:
        opt=torch.optim.SGD(model.parameters(),lr=LR_SGD,
                             momentum=MOMENTUM,nesterov=True)

    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    pos=SEQ//2; projs=None; stt=None; vals=[]; t0=time.time()
    print(f"\n  [{name}]")

    for step in range(1,MAX_STEPS+1):
        # LR schedule
        lr=clr_adam(step) if use_adam else clr_sgd(step)
        for pg in opt.param_groups: pg['lr']=lr

        model.train()
        x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()

        # Compute projections
        if project and (step==1 or step%proj_every==0):
            projs=get_projections(model,x_ref,pos)

        # Apply projection (SGD does NOT cancel this)
        if project and projs is not None:
            apply_projection(model,projs)

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step%25==0 or step==1:
            vl=eval_val(model,n=20); vals.append((step,vl))
            if vl<TARGET and stt is None:
                stt=step; print(f"    *** TARGET val<{TARGET} at step {step} ***")
            pfx=f"proj_every={proj_every}" if project else "no proj"
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}  t={time.time()-t0:.0f}s  [{pfx}]")

    return stt,vals,time.time()-t0

# ── Run all conditions ────────────────────────────────────────────────────────
print("A: Standard SGD (baseline)...")
stt_A,vals_A,t_A=run("SGD baseline",use_adam=False,project=False)

print("\nB: SGD + projection every step...")
stt_B,vals_B,t_B=run("SGD + proj every step",use_adam=False,project=True,proj_every=1)

print("\nC: SGD + projection every 5 steps...")
stt_C,vals_C,t_C=run("SGD + proj every 5",use_adam=False,project=True,proj_every=5)

print("\nD: Adam (reference — projection cancelled)...")
stt_D,vals_D,t_D=run("Adam reference",use_adam=True,project=False)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS  (target val<{TARGET})")
print("="*65)

def fmt(s): return str(s) if s else f">{MAX_STEPS}"
rows=[
    ("SGD baseline",          stt_A,vals_A[-1][1],t_A,None),
    ("SGD + proj every step", stt_B,vals_B[-1][1],t_B,stt_A),
    ("SGD + proj every 5",    stt_C,vals_C[-1][1],t_C,stt_A),
    ("Adam (reference)",      stt_D,vals_D[-1][1],t_D,None),
]
print(f"\n  {'Method':28}  {'Steps→<4':>9}  {'Final val':>10}  {'Time':>7}  {'Speedup':>8}")
print("  "+"-"*67)
for name,stt,fval,t,base in rows:
    sp=f"{base/stt:.2f}x" if (base and stt and stt<MAX_STEPS) else "—"
    print(f"  {name:28}  {fmt(stt):>9}  {fval:>10.4f}  {t:>6.1f}s  {sp:>8}")

print(f"\n  Loss curve comparison:")
print(f"  {'step':>5}  {'SGD base':>10}  {'SGD+proj/1':>12}  {'SGD+proj/5':>12}  {'Adam':>8}")
print("  "+"-"*50)
sA={s:v for s,v in vals_A}; sB={s:v for s,v in vals_B}
sC={s:v for s,v in vals_C}; sD={s:v for s,v in vals_D}
for step in sorted(sA):
    print(f"  {step:>5}  {sA.get(step,0):>10.4f}  {sB.get(step,0):>12.4f}"
          f"  {sC.get(step,0):>12.4f}  {sD.get(step,0):>8.4f}")

print(f"""
KEY QUESTION:
  SGD+proj/1 vs SGD baseline — does projection help when Adam can't cancel it?
  If SGD+proj reaches val<4 EARLIER: pants coproduct projection is real.
  If SAME or SLOWER: the active subspace at one token doesn't generalise.
""")
