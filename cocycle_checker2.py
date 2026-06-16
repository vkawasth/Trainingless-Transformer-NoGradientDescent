#!/usr/bin/env python3
"""
Cocycle Checker v2 — clean rewrite, no outer product bugs
==========================================================
Three questions:
  1. Is morphism entropy < token entropy? (representational efficiency)
  2. Does cocycle norm correlate with CE loss? (same signal or different?)
  3. Does training on high-cocycle batches converge faster? (actionable)

MORPHISM = hidden state difference at attractor layers
  delta_h_l(x) = h_{l+1}(x) - h_l(x)  for l in attractor basin

This is the actual discrete 1-form on the quiver.
Its norm ||delta_h_l|| measures how much the representation
changes at layer l for sequence x.

COCYCLE NORM = how much the sequence's transitions deviate
from the DGLA flat region:
  coc(x) = mean_l ||[Delta_l, diag(delta_h_l(x))||_F
where Delta_l = J_{l+1} - J_l (reference difference operator)
and diag(v) promotes vector v to diagonal matrix.

ENTROPY = SV spectrum of the stacked attractor differences
  M(x) = stack of delta_h_l(x) for l in attractor basin  (5 x ma)
  H(x) = -sum s_i log s_i where s = SV(M(x))/sum(SV(M(x)))
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  COCYCLE CHECKER v2")
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

def clr(s,total=300,warmup=100):
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
    return J.T,U.detach().numpy()

def comm(A,B): return A@B-B@A
def N(A): return float(np.linalg.norm(A))
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

# ════════════════════════════════════════════════════
# Train teacher + reference Jacobians
# ════════════════════════════════════════════════════
print("Training teacher...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
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
print(f"  Teacher val={val_teacher:.4f}\n")

# Reference Jacobians for Delta_l
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...")
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]
# Reference difference operators
Delta=[Js[l+1]-Js[l] for l in range(N_LAYERS_T-1)]
ATT_LAYERS=list(range(L_ATT-2, L_ATT+3))  # L12-L16
print(f"  Attractor layers: {ATT_LAYERS}, ma={ma}\n")

# Serre cascade for student
cascade=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    n=N(C); cascade.append(C/max(n,1e-8))

# ════════════════════════════════════════════════════
# SEQUENCE SCORING FUNCTION (correct version)
# ════════════════════════════════════════════════════
def score_sequence(model, x_seq, Delta_ops, att_layers, ma, pos):
    """
    Score a single sequence by:
      1. morphism_entropy: SV entropy of stacked hidden-state differences
      2. cocycle_norm: mean ||[Delta_l, diag(delta_h)]|| at attractor layers
      3. ce_loss: standard cross-entropy loss
    All fast — only one forward pass.
    """
    model.eval()
    x2=x_seq.unsqueeze(0)  # (1, SEQ)
    y2=torch.zeros_like(x2); y2[0,:-1]=x2[0,1:]

    with torch.no_grad():
        logits,loss=model(x2,y2)
        hs=model.hidden_states(x2)
        hs=[h[0,pos,:].numpy() for h in hs]  # (n_layers+1, D)

    ce=float(loss) if loss is not None else 0.0

    # Hidden state differences at attractor layers
    diffs=[]
    for l in att_layers:
        if l+1<len(hs):
            d=(hs[l+1]-hs[l])[:ma]  # (ma,) vector
            diffs.append(d)

    if not diffs:
        return 0.0, 0.0, ce

    # 1. Morphism entropy: SV spectrum of stacked differences (n_att x ma)
    M=np.stack(diffs, axis=0)  # (n_att, ma)
    sv=np.linalg.svd(M, compute_uv=False)  # (min(n_att,ma),)
    sv_sum=sv.sum()
    if sv_sum>1e-10:
        sv_norm=sv/sv_sum
        H_morph=-float(np.sum(sv_norm*np.log(sv_norm+1e-12)))
    else:
        H_morph=0.0

    # 2. Cocycle norm: ||[Delta_l, D_l]|| where D_l = diag(delta_h_l)
    # D_l is (ma x ma) diagonal; [Delta_l, D_l] = Delta_l @ D_l - D_l @ Delta_l
    coc=0.0; n_coc=0
    for i,l in enumerate(att_layers):
        if l<len(Delta_ops) and i<len(diffs):
            D_l=np.diag(diffs[i])        # (ma, ma) diagonal
            dl=Delta_ops[l][:ma,:ma]     # (ma, ma) reference difference
            c=comm(dl, D_l)
            coc+=float(np.linalg.norm(c,'fro'))
            n_coc+=1
    coc=coc/max(n_coc,1)

    return H_morph, coc, ce

# ════════════════════════════════════════════════════
# PART 1: ENTROPY + COCYCLE COMPARISON
# ════════════════════════════════════════════════════
print("="*65)
print("PART 1: ENTROPY + COCYCLE vs CE LOSS (200 sequences)")
print("="*65)

N_SAMPLE=200
H_morphs=[]; coc_norms=[]; ce_losses=[]; H_tokens=[]
t_morph=[]; t_ce=[]

print(f"\n  Scoring {N_SAMPLE} sequences...")
for i in range(N_SAMPLE):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    x_seq=train_t[ix:ix+SEQ]

    # Token entropy (CE)
    t0_ce=time.perf_counter()
    x2=x_seq.unsqueeze(0); y2=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
    with torch.no_grad():
        logits,loss=teacher(x2,y2)
        probs=torch.softmax(logits[0],dim=-1).numpy()
        H_tok=-float(np.mean(np.sum(probs*np.log(probs+1e-12),axis=-1)))
    t_ce.append(time.perf_counter()-t0_ce)
    H_tokens.append(H_tok)
    ce_losses.append(float(loss))

    # Morphism entropy + cocycle
    t0_m=time.perf_counter()
    H_m,coc,_=score_sequence(teacher,x_seq,Delta,ATT_LAYERS,ma,pos)
    t_morph.append(time.perf_counter()-t0_m)
    H_morphs.append(H_m); coc_norms.append(coc)

    if (i+1)%50==0: print(f"  {i+1}/{N_SAMPLE}...",flush=True)

H_morphs=np.array(H_morphs); coc_norms=np.array(coc_norms)
ce_losses=np.array(ce_losses); H_tokens=np.array(H_tokens)

print(f"\n  ENTROPY COMPARISON:")
print(f"  {'':>30}  {'Token H':>10}  {'Morphism H':>12}")
print(f"  {'Mean':>30}  {np.mean(H_tokens):>10.4f}  {np.mean(H_morphs):>12.4f}")
print(f"  {'Std':>30}  {np.std(H_tokens):>10.4f}  {np.std(H_morphs):>12.4f}")
print(f"  Token/Morphism entropy ratio: {np.mean(H_tokens)/max(np.mean(H_morphs),1e-8):.2f}x")

print(f"\n  TIMING:")
print(f"  Token CE forward:  {1000*np.mean(t_ce):.1f} ms")
print(f"  Morphism scoring:  {1000*np.mean(t_morph):.1f} ms")
print(f"  Speedup: {np.mean(t_ce)/np.mean(t_morph):.1f}x")

print(f"\n  COCYCLE NORM:")
print(f"  Mean: {np.mean(coc_norms):.5f}  Std: {np.std(coc_norms):.5f}")
print(f"  Min:  {np.min(coc_norms):.5f}  Max: {np.max(coc_norms):.5f}")

corr_coc_ce=float(np.corrcoef(coc_norms,ce_losses)[0,1])
corr_h_ce=float(np.corrcoef(H_morphs,ce_losses)[0,1])
corr_h_coc=float(np.corrcoef(H_morphs,coc_norms)[0,1])
print(f"\n  CORRELATIONS:")
print(f"  corr(cocycle_norm, CE_loss):     {corr_coc_ce:+.4f}")
print(f"  corr(morphism_H,  CE_loss):      {corr_h_ce:+.4f}")
print(f"  corr(morphism_H,  cocycle_norm): {corr_h_coc:+.4f}")

threshold=float(np.percentile(coc_norms,25))
print(f"\n  Pre-flight threshold (25th pct): {threshold:.5f}")

# ════════════════════════════════════════════════════
# PART 2: TRAINING COMPARISON
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: TRAINING COMPARISON (200 gradient steps each)")
print("  A: Standard uniform sampling")
print("  B: High-cocycle batches (top 50% by cocycle norm)")
print("="*65)

# Precompute cocycle scores for a pool of positions
print(f"\n  Precomputing cocycle scores for 400-position pool...")
pool_size=400
pool_pos=torch.randint(0,len(train_t)-SEQ-1,(pool_size,))
pool_scores=np.zeros(pool_size)
for i,idx in enumerate(pool_pos.tolist()):
    x_seq=train_t[idx:idx+SEQ]
    _,coc,_=score_sequence(teacher,x_seq,Delta,ATT_LAYERS,ma,pos)
    pool_scores[i]=coc
    if (i+1)%100==0: print(f"  {i+1}/{pool_size}...",flush=True)

pool_median=float(np.median(pool_scores))
high_coc_idx=pool_pos[pool_scores>=pool_median]
print(f"  Pool median cocycle: {pool_median:.5f}")
print(f"  High-cocycle positions: {len(high_coc_idx)}/{pool_size}")

def get_batch_high_coc():
    """Sample from high-cocycle positions."""
    idx=high_coc_idx[torch.randint(0,len(high_coc_idx),(BATCH,))]
    x=torch.stack([train_t[i:i+SEQ] for i in idx])
    y=torch.stack([train_t[i+1:i+SEQ+1] for i in idx])
    return x,y

def build_student():
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            stu.blocks[l].attn.WK.weight.copy_(W_t)
            stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

def train_student(stu, label, steps=200, use_high_coc=False):
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,50)
        stu.train()
        x,y=get_batch_high_coc() if use_high_coc else get_batch()
        _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,175,200]:
            v=eval_val(stu,n=20); ck[step]=v
            b="✓" if v<val_teacher else " "
            print(f"  [{label}] step {step:>4}  val={v:.4f} {b}")
    return eval_val(stu),ck

stuA=build_student(); stuB=build_student()
print(f"\n  A: Standard sampling:")
vA,ckA=train_student(stuA,"A-std",use_high_coc=False)
print(f"\n  B: High-cocycle sampling:")
vB,ckB=train_student(stuB,"B-high-coc",use_high_coc=True)

print(f"\n{'='*65}")
print("  COCYCLE CHECKER v2 RESULTS")
print("="*65)
print(f"""
  ENTROPY (corrected):
    Token stream H:    {np.mean(H_tokens):.4f} nats
    Morphism stream H: {np.mean(H_morphs):.4f} nats
    Ratio: {np.mean(H_tokens)/max(np.mean(H_morphs),1e-8):.2f}x

  CORRELATIONS:
    corr(cocycle, CE):  {corr_coc_ce:+.4f}
    corr(morphism_H, CE): {corr_h_ce:+.4f}

  STUDENT RESULTS:
    Teacher:       val={val_teacher:.4f}
    A (standard):  val={vA:.4f}
    B (high-coc):  val={vB:.4f}  diff={vA-vB:+.4f}

  CONVERGENCE:
  {'step':>6}  {'A-std':>8}  {'B-high-coc':>11}  {'diff':>7}""")
for s in [25,50,75,100,125,150,175,200]:
    a=ckA.get(s); b=ckB.get(s)
    if a and b:
        print(f"  {s:>6}  {a:>8.4f}  {b:>11.4f}  {a-b:>+7.4f}")

print(f"""
  INTERPRETATION:
    corr={corr_coc_ce:+.4f}: cocycle and CE measure
    {'SAME signal — cocycle filter = loss-based selection' if abs(corr_coc_ce)>0.3
     else 'DIFFERENT signals — cocycle is orthogonal to CE loss'}

    {'B < A: high-cocycle batches accelerate convergence' if vB<vA-0.003
     else 'B ≈ A: cocycle-based batch selection makes no difference'}
""")
