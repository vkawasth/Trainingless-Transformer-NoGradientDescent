#!/usr/bin/env python3
"""
Iterative Gram Refinement
==========================
The gram alignment gap (0.32 best) means corpus statistics capture
only 32% of the teacher's token similarity structure.

The fixed point iteration:
  E^{(0)} = SVD factorization of PPMI (corpus gram)
  E^{(k+1)} = SVD factorization of (E^{(k)} @ Gram_D @ E^{(k)T})

At convergence: E^* @ E^{*T} = E^* @ Gram_D @ E^{*T}
This is the eigenvector equation for Gram_D restricted to the
token embedding subspace — the fixed point where corpus statistics
and Jacobian amplification are mutually consistent.

HYPOTHESIS: gram_align should increase monotonically with iterations,
converging toward 1.0. Each iteration brings the corpus embedding
closer to the teacher's gram structure.

ALSO TEST: Whether gram_align improvement translates to val improvement
in the full pipeline (Serre cascade + head training).
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  ITERATIVE GRAM REFINEMENT")
print(f"  E^(k+1) = SVD(E^(k) @ Gram_D @ E^(k)T)")
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

def clr(s,total,warmup=50):
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

def gram_to_embedding(G, target_norm, d=D):
    G=(G+G.T)/2
    U,s,_=np.linalg.svd(G,full_matrices=False)
    s=np.maximum(s,0)
    E=(U[:,:d]*np.sqrt(s[:d])).astype(np.float32)
    n=float(np.linalg.norm(E,'fro'))
    if n>1e-8: E=E*(target_norm/n)
    return E

def gram_align(E,E_teacher):
    G1=(E@E.T).flatten(); G2=(E_teacher@E_teacher.T).flatten()
    return float(np.corrcoef(G1,G2)[0,1])

def row_cos(E,E_teacher):
    En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
    Tn=E_teacher/(np.linalg.norm(E_teacher,axis=1,keepdims=True)+1e-8)
    return float(np.mean(np.sum(En*Tn,axis=1)))

# ════════════════════════════════════════════════════════
# STAGE 0: Train teacher
# ════════════════════════════════════════════════════════
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
teacher_norm=float(np.linalg.norm(E_teacher,'fro'))
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract invariants
# ════════════════════════════════════════════════════════
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
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]

# Gram_D
Gram=sum(Js[l].T@Js[l] for l in range(N_LAYERS))/N_LAYERS
Gram_D=U14@Gram@U14.T+(np.eye(D)-U14@U14.T)

# Serre cascade
cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
print(f"  Done. Cascade: {N_STU} levels\n")

def inject_cascade(model):
    with torch.no_grad():
        model.pe.weight.copy_(teacher.pe.weight)
        model.ln_f.weight.copy_(teacher.ln_f.weight)
        model.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
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

# ════════════════════════════════════════════════════════
# STAGE 2: Build initial corpus gram (PPMI)
# ════════════════════════════════════════════════════════
print("Stage 2: Build initial PPMI embedding...")
P=np.zeros((VOCAB,VOCAB),dtype=np.float64)
unigram=np.zeros(VOCAB,dtype=np.float64)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB:
        P[a,b]+=1; unigram[a]+=1
total=P.sum(); unigram_p=unigram/max(total,1)
pmi=np.log((P/max(total,1)+1e-10)/
           (unigram_p[:,None]*unigram_p[None,:]+1e-10))
ppmi=np.clip(pmi,0,None).astype(np.float32)
G0=ppmi@ppmi.T  # initial gram
E_curr=gram_to_embedding(G0,teacher_norm)
ga0=gram_align(E_curr,E_teacher)
rc0=row_cos(E_curr,E_teacher)
print(f"  Iteration 0 (PPMI): gram_align={ga0:.4f}  row_cos={rc0:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 3: Iterative gram refinement
# ════════════════════════════════════════════════════════
print("Stage 3: Iterative gram refinement...")
print(f"  E^(k+1) = SVD( E^(k) @ Gram_D @ E^(k)T )")
print(f"  Fixed point: E^* where E^* @ Gram_D @ E^*T = E^* @ E^*T")
print()
print(f"  {'Iter':>5}  {'gram_align':>12}  {'row_cos':>9}  {'delta_align':>12}")
print("  "+"-"*45)

N_ITER=8
history=[(0, ga0, rc0)]
print(f"  {'0':>5}  {ga0:>12.4f}  {rc0:>9.4f}  {'---':>12}")

E_prev=E_curr.copy()
for k in range(1,N_ITER+1):
    # Gram refinement step: G^(k) = E^(k-1) @ Gram_D @ E^(k-1)T
    # This mixes corpus co-occurrence with Jacobian amplification
    G_new=(E_prev@Gram_D@E_prev.T).astype(np.float32)

    # Also blend with original PPMI gram to prevent drift
    alpha=0.3  # blending: keep 30% corpus, 70% refined
    G_blend=alpha*G0+(1-alpha)*G_new

    E_curr=gram_to_embedding(G_blend,teacher_norm)
    ga=gram_align(E_curr,E_teacher)
    rc=row_cos(E_curr,E_teacher)
    delta=ga-history[-1][1]
    history.append((k,ga,rc))
    print(f"  {k:>5}  {ga:>12.4f}  {rc:>9.4f}  {delta:>+12.4f}")

    if abs(delta)<1e-4 and k>2:
        print(f"  Converged at iteration {k}")
        break
    E_prev=E_curr.copy()

best_iter=max(range(len(history)),key=lambda i:history[i][1])
E_best=None  # will rebuild below

# Rebuild best embedding
print(f"\n  Best iteration: {history[best_iter][0]} "
      f"(gram_align={history[best_iter][1]:.4f})")

# Rebuild at best iteration
E_prev=gram_to_embedding(G0,teacher_norm)
for k in range(1,history[best_iter][0]+1):
    G_new=(E_prev@Gram_D@E_prev.T).astype(np.float32)
    G_blend=0.3*G0+0.7*G_new
    E_best=gram_to_embedding(G_blend,teacher_norm)
    E_prev=E_best.copy()

if E_best is None: E_best=E_curr

# ════════════════════════════════════════════════════════
# STAGE 4: Test best refined embedding in full pipeline
# ════════════════════════════════════════════════════════
print(f"\nStage 4: Test refined embedding in Serre pipeline...")

def run(E_init,label,head_steps=100,full_steps=200):
    torch.manual_seed(99)
    model=LM(D,N_HEADS,N_STU)
    with torch.no_grad():
        E_t=torch.tensor(E_init[:VOCAB,:D].copy(),dtype=torch.float32)
        model.te.weight.copy_(E_t)
    inject_cascade(model)
    v0=eval_val(model,n=20)
    print(f"  [{label}] zero-shot val={v0:.4f}")

    # Head only
    for p in model.parameters(): p.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    opt_h=torch.optim.AdamW([model.head.weight],lr=LR,weight_decay=0.01)
    for step in range(1,head_steps+1):
        for pg in opt_h.param_groups: pg['lr']=clr(step,head_steps,20)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.head.weight],1.0); opt_h.step()
    for p in model.parameters(): p.requires_grad_(True)
    vh=eval_val(model)
    print(f"  [{label}] head-only {head_steps} steps: val={vh:.4f}")

    # Full tune
    opt_f=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,full_steps+1):
        for pg in opt_f.param_groups: pg['lr']=clr(step,full_steps,50)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt_f.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt_f.step()
        if step%(full_steps//4)==0:
            print(f"    step {step}  val={eval_val(model,n=20):.4f}")
    vf=eval_val(model)
    return v0,vh,vf

# Test: PPMI (iter 0), best refined, teacher oracle
r_ppmi=run(gram_to_embedding(G0,teacher_norm),"PPMI iter=0")
r_best=run(E_best,f"Refined iter={history[best_iter][0]}")
r_oracle=run(E_teacher,"Teacher oracle")

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  GRAM REFINEMENT RESULTS")
print("="*65)

print(f"\n  Gram alignment convergence:")
print(f"  {'Iter':>5}  {'gram_align':>12}  {'row_cos':>9}")
for it,ga,rc in history:
    print(f"  {it:>5}  {ga:>12.4f}  {rc:>9.4f}")

print(f"""
  Pipeline results (Serre cascade + training):
  {'Method':>25}  {'zero-shot':>10}  {'head-100':>10}  {'full-200':>10}
  {'-'*60}
  {'PPMI (iter 0)':>25}  {r_ppmi[0]:>10.4f}  {r_ppmi[1]:>10.4f}  {r_ppmi[2]:>10.4f}
  {f'Refined (iter {history[best_iter][0]})':>25}  {r_best[0]:>10.4f}  {r_best[1]:>10.4f}  {r_best[2]:>10.4f}
  {'Teacher oracle':>25}  {r_oracle[0]:>10.4f}  {r_oracle[1]:>10.4f}  {r_oracle[2]:>10.4f}

  Teacher oracle full model:   val={val_teacher:.4f}
  Prior best (Serre+teacher):  val=0.1865

  IF gram_align converges toward 1.0:
    The iterative refinement closes the orientation gap.
    Closed-form embedding is achievable.
    Training reduces to head alignment only.

  IF gram_align plateaus below 0.5:
    The Jacobian Gram cannot bridge the gap from corpus statistics.
    The missing structure requires gradient descent through data.
    Minimum training = joint embedding+head, ~120 steps.

  REFINEMENT IMPROVEMENT:
    PPMI gram_align:    {history[0][1]:.4f}
    Best gram_align:    {history[best_iter][1]:.4f}
    Improvement:        {history[best_iter][1]-history[0][1]:+.4f}
    Val improvement:    {r_ppmi[2]-r_best[2]:+.4f} nats
""")
