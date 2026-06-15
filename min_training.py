#!/usr/bin/env python3
"""
Minimum Training Experiment
=============================
Find the minimum embedding training steps needed to match
the Serre approximator result (val=0.187 with teacher embeddings).

Protocol:
  For N in {10, 25, 50, 100, 200}:
    1. Train ONLY the embedding matrix for N steps (blocks frozen random)
    2. Freeze embeddings
    3. Inject Serre cascade into blocks (zero gradient)
    4. Train head only for 50 steps
    5. Measure val

Compare to:
  - Teacher oracle: val=0.2504
  - Serre + teacher emb + 200 CE: val=0.1865
  - 6L random + teacher emb + 200 CE: val=0.5098

The N where val < 0.25 (matches teacher) is the minimum training budget.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  MINIMUM TRAINING EXPERIMENT")
print(f"  Embed-only training → Serre cascade → head alignment")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
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
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total,warmup=10):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def layer_jac(block,h_in,pos,m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            hh=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=block(hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T,U.detach().numpy(),m

def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

# ── Train teacher ─────────────────────────────────────────────────────────────
print("Stage 0: Train teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step,300,100)
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval()
val_teacher=eval_val(teacher)
E_teacher=teacher.te.weight.data.numpy().copy()
print(f"  Teacher val={val_teacher:.4f}\n")

# ── Extract invariants ────────────────────────────────────────────────────────
print("Stage 1: Extract invariants...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=m_
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]
print(f"  Done. m={ma}")

# Build Serre cascade
cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
print(f"  Cascade built: {N_STU} levels\n")

def inject_cascade(model):
    """Inject Serre cascade into model blocks. No gradient."""
    with torch.no_grad():
        model.pe.weight.copy_(teacher.pe.weight)
        model.ln_f.weight.copy_(teacher.ln_f.weight)
        model.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(min(N_STU, len(model.blocks))):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            model.blocks[l].attn.WK.weight.copy_(W_t)
            model.blocks[l].attn.WQ.weight.copy_(W_t.T)
            model.blocks[l].attn.WV.weight.copy_(
                teacher.blocks[L_ATT].attn.WV.weight)
            model.blocks[l].attn.op.weight.copy_(
                teacher.blocks[L_ATT].attn.op.weight)
            model.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            model.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            model.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

def measure_emb_cos(model):
    En=model.te.weight.data.numpy()
    En=En/(np.linalg.norm(En,axis=1,keepdims=True)+1e-8)
    Tn=E_teacher/(np.linalg.norm(E_teacher,axis=1,keepdims=True)+1e-8)
    return float(np.mean(np.sum(En*Tn,axis=1)))

# ── Reference: teacher embeddings → cascade → head 100 steps ─────────────────
print("Reference: teacher emb → cascade → head 100 steps...")
torch.manual_seed(99)
ref_model=LM(D,N_HEADS,N_STU)
ref_model.te.weight.data.copy_(teacher.te.weight.data)
inject_cascade(ref_model)
for p in ref_model.parameters(): p.requires_grad_(False)
ref_model.head.weight.requires_grad_(True)
opt_r=torch.optim.AdamW([ref_model.head.weight],lr=LR,weight_decay=0.01)
for step in range(1,101):
    for pg in opt_r.param_groups: pg['lr']=clr(step,100,20)
    ref_model.train(); x,y=get_batch(); _,loss=ref_model(x,y)
    opt_r.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([ref_model.head.weight],1.0); opt_r.step()
for p in ref_model.parameters(): p.requires_grad_(True)
val_ref=eval_val(ref_model)
print(f"  Reference (teacher emb + cascade + 100 head steps): val={val_ref:.4f}\n")

# ── Sweep: embed-only training → cascade → head 50 steps ─────────────────────
print("="*65)
print("  EMBED-ONLY SWEEP")
print("  inject cascade → N steps embed-only → 50 steps head-only")
print("="*65)

EMBED_STEPS=[0, 10, 25, 50, 100, 200]
HEAD_STEPS=50
results=[]

for N_emb in EMBED_STEPS:
    t0=time.time()
    torch.manual_seed(99)
    model=LM(D,N_HEADS,N_STU)

    # Phase 1: inject Serre cascade FIRST (zero gradient)
    # Cascade must be present during embedding training
    inject_cascade(model)
    val_after_cascade=eval_val(model,n=20)

    # Phase 2: train embeddings only (cascade blocks frozen)
    if N_emb > 0:
        for p in model.parameters(): p.requires_grad_(False)
        model.te.weight.requires_grad_(True)
        opt_e=torch.optim.AdamW([model.te.weight],lr=LR,weight_decay=0.01)
        for step in range(1,N_emb+1):
            for pg in opt_e.param_groups: pg['lr']=clr(step,N_emb,max(5,N_emb//10))
            model.train(); x,y=get_batch(); _,loss=model(x,y)
            opt_e.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([model.te.weight],1.0); opt_e.step()
        for p in model.parameters(): p.requires_grad_(True)

    emb_cos=measure_emb_cos(model)
    val_after_emb=eval_val(model,n=20)

    # Phase 3: head-only training (50 steps, blocks still frozen)
    for p in model.parameters(): p.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    opt_h=torch.optim.AdamW([model.head.weight],lr=LR,weight_decay=0.01)
    for step in range(1,HEAD_STEPS+1):
        for pg in opt_h.param_groups: pg['lr']=clr(step,HEAD_STEPS,10)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.head.weight],1.0); opt_h.step()
    for p in model.parameters(): p.requires_grad_(True)

    val_final=eval_val(model)
    results.append((N_emb,emb_cos,val_after_emb,val_after_cascade,val_final))

    # Compute trainable params and FLOPs
    emb_params=VOCAB*D
    head_params=VOCAB*D
    total_params=sum(p.numel() for p in model.parameters())
    trained_params=emb_params*int(N_emb>0)+head_params
    pct=100*trained_params/total_params

    print(f"  N_emb={N_emb:>3}  emb_cos={emb_cos:.3f}  "
          f"val_cascade={val_after_cascade:.3f}  "
          f"val_emb={val_after_emb:.3f}  "
          f"val_final={val_final:.4f}  "
          f"params_trained={pct:.1f}%  "
          f"t={time.time()-t0:.0f}s")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  MINIMUM TRAINING RESULTS")
print("="*65)

emb_params=VOCAB*D; head_params=VOCAB*D
total_params=sum(p.numel() for p in LM(D,N_HEADS,N_STU).parameters())

print(f"""
  Architecture: {N_STU}L student, {total_params:,} total params
  Embedding:    {emb_params:,} params ({100*emb_params/total_params:.1f}%)
  Head:         {head_params:,} params (tied to embedding)

  {'N_emb':>6}  {'emb_cos':>8}  {'val_final':>10}  {'trained%':>9}  {'beats_teacher?'}
  {'-'*55}""")

for N_emb,emb_cos,_,_,val_final in results:
    trained_pct=100*(emb_params*int(N_emb>0)+head_params)/total_params
    beats=f"YES (gap={val_teacher-val_final:.3f})" if val_final<val_teacher else f"no  (gap={val_final-val_teacher:.3f})"
    print(f"  {N_emb:>6}  {emb_cos:>8.3f}  {val_final:>10.4f}  {trained_pct:>8.1f}%  {beats}")

print(f"""
  Reference (teacher emb + cascade + 100 head): val={val_ref:.4f}
  Teacher oracle:                                val={val_teacher:.4f}
  Prior best (Serre + teacher emb + 200 CE):    val=0.1865

  KEY NUMBERS:
  Total params:       {total_params:,}
  Embedding params:   {emb_params:,} ({100*emb_params/total_params:.1f}% of total)
  
  If N_emb=50 gives val < 0.25:
    50 embed-only steps on 4.1% of params = effective training elimination
    Blocks: zero gradient (Serre cascade)
    Head: 50 gradient steps on tied embedding
    
  The crossover point is the minimum training budget.
  Below it: algebraic initialization dominates.
  Above it: gradient descent takes over.
""")
