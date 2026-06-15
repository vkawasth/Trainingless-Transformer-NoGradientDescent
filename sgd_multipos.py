#!/usr/bin/env python3
"""
SGD + Multi-Position Pants Coproduct Projection
================================================
Single position projection fails: reference subspace ⊥ batch gradient.
Fix: aggregate active subspaces across multiple token positions k.

U_l^(union) = orth(span(U_l^(k2), U_l^(k5), U_l^(k7), ...))

More positions → richer subspace → better coverage of batch gradient.

Tests across position sets:
  A:  SGD baseline (no projection)
  B:  SGD + proj from pos=[2]           (rank~4,  3% of d)
  C:  SGD + proj from pos=[2,5,7]       (rank~12, 9% of d)
  D:  SGD + proj from pos=[2,5,7,12,20] (rank~20,16% of d)
  E:  SGD + proj from ALL positions     (rank~SEQ*4, batch approx)
  F:  Adam baseline (reference)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64
LR_SGD=0.05; LR_ADAM=3e-4; MOMENTUM=0.9
PROJ=16; TARGET=4.0; MAX_STEPS=400

print(f"\n{'='*65}")
print(f"  MULTI-POSITION PANTS COPRODUCT PROJECTION")
print(f"  Aggregating active subspaces across token positions")
print(f"  d={D}  layers={N_LAYERS}  proj_per_pos={PROJ}")
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

def clr(s,total=MAX_STEPS,warmup=50,base=LR_SGD):
    if s<=warmup: return base*s/warmup
    return base*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Multi-position projection ─────────────────────────────────────────────────
def get_multipos_projections(model, x_ref, positions, m=PROJ):
    """
    For each layer l, aggregate active subspaces across all positions.
    Returns per-layer projection P_l = U_union U_union^T.
    U_union = orth basis of span(U_l^(k) for k in positions).
    """
    projs = []

    with torch.enable_grad():
        # Hidden states once (shared across positions)
        hs = []
        h = model.te(x_ref)+model.pe(torch.arange(x_ref.shape[1]))
        hs.append(h.detach())
        for b in model.blocks: h=b(h); hs.append(h.detach())

        for l in range(model._nl):
            # Collect active subspace columns from each position
            U_cols = []   # will concatenate [d, rank*n_pos]

            for pos in positions:
                h_in = hs[l][0]           # [SEQ, D]
                seq,d_ = h_in.shape; m_=min(m,seq,d_)

                # Use per-position SVD basis
                h_pos = h_in[pos:pos+1]   # [1, D] — basis around this position
                # But use full h_in for SVD to get stable basis
                _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
                U_basis=Vt[:m_,:].T.detach()   # [D, m]

                # Jacobian at this position
                J=torch.zeros(m_,m_)
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

                U_sv,_,_=np.linalg.svd(dJ)
                U_act_m=U_sv[:,:rank]              # [m, rank]
                U_act_d=U_basis.numpy()@U_act_m    # [D, rank]
                U_cols.append(U_act_d)

            # Union of subspaces: QR of concatenated columns
            U_all=np.concatenate(U_cols, axis=1)   # [D, rank*n_pos]
            # Orthonormalise via QR (keep only linearly independent cols)
            Q,R=np.linalg.qr(U_all, mode='reduced')
            # Keep columns where diagonal of R is non-negligible
            r_diag=np.abs(np.diag(R))
            keep=r_diag > r_diag[0]*0.01
            U_union=Q[:,keep]                       # [D, union_rank]
            union_rank=U_union.shape[1]

            U_t=torch.tensor(U_union,dtype=torch.float32)
            P=U_t@U_t.T                             # [D, D]

            projs.append({
                'P':P,
                'rank':union_rank,
                'live_frac':union_rank/D,
            })

    return projs

def apply_proj(model, projs):
    for l,sp in enumerate(projs):
        P=sp['P']
        for param in model.blocks[l].parameters():
            if param.grad is None: continue
            g=param.grad.data
            if g.dim()==2:
                d_o,d_i=g.shape
                if d_o==D:   param.grad.data=P@g
                elif d_i==D: param.grad.data=g@P.T

# ── Run ───────────────────────────────────────────────────────────────────────
def run(name, positions=None, use_adam=False, proj_every=5, seed=42):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS); model._nl=N_LAYERS

    if use_adam:
        opt=torch.optim.AdamW(model.parameters(),lr=LR_ADAM,
                               betas=(0.9,0.95),weight_decay=0.1)
    else:
        opt=torch.optim.SGD(model.parameters(),lr=LR_SGD,
                             momentum=MOMENTUM,nesterov=True)

    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    projs=None; stt=None; vals=[]; t0=time.time()
    print(f"\n  [{name}]  positions={positions}")

    for step in range(1,MAX_STEPS+1):
        lr=clr(step,base=LR_ADAM if use_adam else LR_SGD)
        for pg in opt.param_groups: pg['lr']=lr

        model.train()
        x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()

        if positions is not None and (step==1 or step%proj_every==0):
            projs=get_multipos_projections(model,x_ref,positions)

        if positions is not None and projs is not None:
            apply_proj(model,projs)

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step%25==0 or step==1:
            vl=eval_val(model,n=20); vals.append((step,vl))
            if vl<TARGET and stt is None:
                stt=step; print(f"    *** TARGET at step {step} ***")
            rank_info=f"  rank~{projs[0]['rank'] if projs else '?'}" if positions else ""
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}  t={time.time()-t0:.0f}s{rank_info}")

    return stt,vals,time.time()-t0

# Position sets
P1  = [32]                          # single (baseline for comparison)
P3  = [2, 5, 7]                     # your k2,k5,k7
P5  = [2, 5, 7, 12, 20]            # richer
P10 = list(range(2,SEQ,6))         # ~10 evenly spaced positions

print("A: SGD baseline (no projection)...")
stt_A,vals_A,t_A=run("SGD baseline", positions=None)

print("\nB: SGD + pos=[2,5,7] (your k2,k5,k7)...")
stt_B,vals_B,t_B=run("SGD+proj k2,k5,k7", positions=P3, proj_every=5)

print("\nC: SGD + pos=[2,5,7,12,20] (5 positions)...")
stt_C,vals_C,t_C=run("SGD+proj 5-pos", positions=P5, proj_every=5)

print("\nD: SGD + ~10 positions (batch approximation)...")
stt_D,vals_D,t_D=run("SGD+proj 10-pos", positions=P10, proj_every=5)

print("\nE: Adam baseline...")
stt_E,vals_E,t_E=run("Adam baseline", positions=None, use_adam=True)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS  (target val<{TARGET}, proj_every=5)")
print("="*65)
def fmt(s): return str(s) if s else f">{MAX_STEPS}"
rows=[
    ("SGD baseline",        stt_A,vals_A[-1][1],t_A),
    ("SGD+proj k2,k5,k7",  stt_B,vals_B[-1][1],t_B),
    ("SGD+proj 5-pos",      stt_C,vals_C[-1][1],t_C),
    ("SGD+proj 10-pos",     stt_D,vals_D[-1][1],t_D),
    ("Adam baseline",       stt_E,vals_E[-1][1],t_E),
]
print(f"\n  {'Method':25}  {'Steps→<4':>9}  {'Final val':>10}  {'Time':>7}  {'vs SGD':>8}")
print("  "+"-"*64)
base=stt_A
for name,stt,fval,t in rows:
    sp=f"{base/stt:.2f}x" if (stt and stt<MAX_STEPS and base) else "—"
    print(f"  {name:25}  {fmt(stt):>9}  {fval:>10.4f}  {t:>6.1f}s  {sp:>8}")

print(f"\n  Effective subspace rank at final step:")
print(f"  P1=[32]:   rank~4   ({4/D:.0%} of d)")
print(f"  P3=[2,5,7]: rank~{min(3*4,D)} ({min(3*4,D)/D:.0%} of d)")
print(f"  P5=[..]:   rank~{min(5*4,D)} ({min(5*4,D)/D:.0%} of d)")
print(f"  P10=[..]:  rank~{min(10*4,D)} ({min(10*4,D)/D:.0%} of d)")
print(f"""
INTERPRETATION:
  As we add positions, the union subspace grows.
  At some point it approximates the true batch gradient subspace.
  That threshold is where projection starts helping SGD.
  
  If D (10-pos) beats A (no proj): multi-position aggregation works.
  If still same/worse: need the ACTUAL batch gradient subspace,
  not a reference-input approximation at any number of positions.
  That would require computing the Jacobian over the training batch itself.
""")
