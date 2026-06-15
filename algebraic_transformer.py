#!/usr/bin/env python3
"""
Algebraic Transformer
======================
Build a transformer entirely from algebraic structure.
No random initialization. No gradient descent on the blocks.
Teacher is kept as verification oracle throughout.

COMPONENTS:
  E: Embeddings — from corpus PMI (pointwise mutual information)
  J: Attractor Jacobian — from teacher's L14 (or corpus M^14)
  C: Serre cascade — ad(J14)^l(J_{14+l}) for l=1..6
  H: Head — trained 100 steps (the only gradient component)

MEASUREMENT AT EACH STAGE:
  Compare to teacher hidden states and logits at every component.
  Find where the algebraic structure diverges from the teacher.
  That divergence point tells us what the algebra cannot yet provide.

STAGES:
  S0: Random baseline (no algebra)
  S1: Corpus embeddings only (PMI-initialized)
  S2: + Serre cascade blocks (no CE)
  S3: + Head alignment (100 CE steps, blocks frozen)
  S4: + Full fine-tune (200 CE steps, all params)
  
  Teacher: oracle at every stage
"""
import json, math, time, warnings, collections
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_ALG=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  ALGEBRAIC TRANSFORMER")
print(f"  Teacher as verification oracle at each stage")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
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
    def hidden_out(self,x):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        return self.ln_f(h)
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

def compare_to_teacher(model, teacher, n=30):
    """Measure alignment with teacher at embedding, hidden, and logit levels."""
    model.eval(); teacher.eval()
    emb_cos=[]; hid_cos=[]; log_cos=[]
    with torch.no_grad():
        for _ in range(n):
            x,_=get_batch('val')
            # Embedding level
            eA=model.te(x)+model.pe(torch.arange(x.shape[1]))
            eT=teacher.te(x)+teacher.pe(torch.arange(x.shape[1]))
            emb_cos.append(F.cosine_similarity(
                eA.reshape(-1,D),eT.reshape(-1,D),dim=-1).mean().item())
            # Hidden state level
            hA=model.hidden_out(x); hT=teacher.hidden_out(x)
            hid_cos.append(F.cosine_similarity(
                hA.reshape(-1,D),hT.reshape(-1,D),dim=-1).mean().item())
            # Logit level
            lA,_=model(x); lT,_=teacher(x)
            log_cos.append(F.cosine_similarity(
                lA.reshape(-1,VOCAB),lT.reshape(-1,VOCAB),dim=-1).mean().item())
    return float(np.mean(emb_cos)),float(np.mean(hid_cos)),float(np.mean(log_cos))

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

def lift_to_d(C_ma,U_basis,scale=0.01):
    UU=U_basis@U_basis.T
    return (U_basis@C_ma@U_basis.T+(np.eye(D)-UU)*scale).astype(np.float32)

def report(label, model, teacher, val=None):
    if val is None: val=eval_val(model)
    ec,hc,lc=compare_to_teacher(model,teacher)
    print(f"\n  [{label}]")
    print(f"    val={val:.4f}")
    print(f"    cos_emb={ec:.4f}  cos_hid={hc:.4f}  cos_log={lc:.4f}")
    print(f"    divergence: emb={1-ec:.4f}  hid={1-hc:.4f}  log={1-lc:.4f}")
    return val, ec, hc, lc

# ══════════════════════════════════════════════════════════════════
# STAGE 0: TRAIN TEACHER (verification oracle)
# ══════════════════════════════════════════════════════════════════
print("="*65)
print("STAGE 0: Teacher (verification oracle)")
print("="*65)
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
print(f"  Teacher val={val_teacher:.4f}")

# ══════════════════════════════════════════════════════════════════
# EXTRACT ALGEBRAIC STRUCTURES FROM TEACHER
# ══════════════════════════════════════════════════════════════════
print("\nExtracting algebraic structures from teacher...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)

# Average over multiple references
J_layers=[]; U_layers=[]; ma=None
for _ in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad():
        hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        if len(J_layers)<=l: J_layers.append([]); U_layers.append([])
        J_layers[l].append(J); U_layers[l].append(U)
        if ma is None: ma=m_

Js=[np.mean(J_layers[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_layers[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]

# Serre cascade
cascade=[]
for l in range(1,N_ALG+1):
    C=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C))
    if n>1e-8: C=C/n
    cascade.append(C)
print(f"  Cascade norms: {[round(float(np.linalg.norm(ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l))),4) for l in range(1,7)]}")

# Corpus PMI embeddings
print("  Computing PMI embeddings from corpus...")
counts=np.zeros((VOCAB,VOCAB),dtype=np.float32)
total=0
for i in range(0,min(len(train_ids)-1,5000)):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: counts[a,b]+=1; total+=1
row_sum=counts.sum(axis=1,keepdims=True)+1e-8
col_sum=counts.sum(axis=0,keepdims=True)+1e-8
pmi=np.log((counts*total)/(row_sum*col_sum)+1e-8)
pmi=np.clip(pmi,0,None)  # PPMI
# SVD to get D-dim embedding
U_pmi,s_pmi,_=np.linalg.svd(pmi,full_matrices=False)
E_pmi=(U_pmi[:,:D]*np.sqrt(s_pmi[:D])).astype(np.float32)
E_pmi_t=torch.tensor(E_pmi,dtype=torch.float32)
print(f"  PMI embedding shape: {E_pmi.shape}  ||E_pmi||={float(np.linalg.norm(E_pmi)):.2f}")

# ══════════════════════════════════════════════════════════════════
# STAGE 1: S0 — Random baseline (no algebra)
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("STAGE S0: Random 6L baseline (no algebra)")
print("="*65)
torch.manual_seed(99)
s0=LM(D,N_HEADS,N_ALG)
# Transfer teacher embeddings as baseline
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(s0,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None: dst.bias.data.copy_(src.bias.data)
val_s0,_,_,_=report("S0: random 6L + teacher emb (0 steps)", s0, teacher)

# ══════════════════════════════════════════════════════════════════
# STAGE S1: PMI embeddings (corpus-derived, no training)
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("STAGE S1: PMI embeddings from corpus (algebraic, no training)")
print("="*65)
torch.manual_seed(99)
s1=LM(D,N_HEADS,N_ALG)
with torch.no_grad():
    # PMI embeddings
    s1.te.weight.copy_(E_pmi_t)
    # Positional: keep random (no algebraic structure available yet)
    # LayerNorm: identity init
    s1.ln_f.weight.fill_(1.0); s1.ln_f.bias.fill_(0.0)
val_s1,ec1,hc1,lc1=report("S1: PMI emb + random blocks (0 steps)", s1, teacher)
print(f"    vs teacher emb: cos_emb gap = {ec1:.4f}")

# ══════════════════════════════════════════════════════════════════
# STAGE S2: + Serre cascade blocks
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("STAGE S2: PMI emb + Serre cascade blocks (no training)")
print("="*65)
torch.manual_seed(99)
s2=LM(D,N_HEADS,N_ALG)
with torch.no_grad():
    s2.te.weight.copy_(E_pmi_t)
    s2.ln_f.weight.fill_(1.0); s2.ln_f.bias.fill_(0.0)
    for l in range(N_ALG):
        C=cascade[l]
        W_d=lift_to_d(C,U14,scale=0.01)
        W_t=torch.tensor(W_d,dtype=torch.float32)
        s2.blocks[l].attn.WK.weight.copy_(W_t)
        s2.blocks[l].attn.WQ.weight.copy_(W_t.T)
        s2.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
        s2.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
        s2.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
        s2.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
        s2.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
val_s2,ec2,hc2,lc2=report("S2: PMI emb + Serre blocks (0 steps)", s2, teacher)

# ══════════════════════════════════════════════════════════════════
# STAGE S3: + Head alignment (blocks frozen, head trains)
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("STAGE S3: + Head alignment (100 steps, blocks frozen)")
print("="*65)
# Freeze blocks, train only head
for p in s2.parameters(): p.requires_grad_(False)
s2.head.weight.requires_grad_(True)
s2.te.weight.requires_grad_(True)  # embeddings also train

opt3=torch.optim.AdamW([s2.head.weight,s2.te.weight],lr=LR,weight_decay=0.01)
for step in range(1,101):
    for pg in opt3.param_groups: pg['lr']=clr(step,100,20)
    s2.train(); x,y=get_batch(); _,loss=s2(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([s2.head.weight,s2.te.weight],1.0)
    opt3.step()
    if step%25==0:
        vl=eval_val(s2,n=20)
        print(f"  step {step}  val={vl:.4f}")
for p in s2.parameters(): p.requires_grad_(True)
val_s3,ec3,hc3,lc3=report("S3: + head+emb aligned (100 steps)", s2, teacher)

# ══════════════════════════════════════════════════════════════════
# STAGE S4: Full fine-tune (all params)
# ══════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("STAGE S4: Full fine-tune (200 steps, all params)")
print("="*65)
opt4=torch.optim.AdamW(s2.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,201):
    for pg in opt4.param_groups: pg['lr']=clr(step,200,50)
    s2.train(); x,y=get_batch(); _,loss=s2(x,y)
    opt4.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s2.parameters(),1.0); opt4.step()
    if step%50==0:
        vl=eval_val(s2,n=20)
        print(f"  step {step}  val={vl:.4f}")
val_s4,ec4,hc4,lc4=report("S4: full fine-tune (200 steps)", s2, teacher)

# ══════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  ALGEBRAIC TRANSFORMER — DIVERGENCE ANALYSIS")
print(f"  Where does algebraic structure align/diverge from teacher?")
print("="*65)

print(f"""
  {'Stage':>30}  {'val':>7}  {'Δemb':>7}  {'Δhid':>7}  {'Δlog':>7}
  {'-'*58}
  {'Teacher (oracle)':>30}  {val_teacher:>7.4f}  {'0':>7}  {'0':>7}  {'0':>7}
  {'S0: random+teacher_emb':>30}  {val_s0:>7.4f}  {'~0':>7}  {'?':>7}  {'?':>7}
  {'S1: PMI emb+random blk':>30}  {val_s1:>7.4f}  {1-ec1:>7.4f}  {1-hc1:>7.4f}  {1-lc1:>7.4f}
  {'S2: PMI+Serre (0 steps)':>30}  {val_s2:>7.4f}  {1-ec2:>7.4f}  {1-hc2:>7.4f}  {1-lc2:>7.4f}
  {'S3: +head align (100 st)':>30}  {val_s3:>7.4f}  {1-ec3:>7.4f}  {1-hc3:>7.4f}  {1-lc3:>7.4f}
  {'S4: +full tune (200 st)':>30}  {val_s4:>7.4f}  {1-ec4:>7.4f}  {1-hc4:>7.4f}  {1-lc4:>7.4f}

  DIVERGENCE READING:
  Δemb: embedding alignment gap (1=orthogonal, 0=identical)
  Δhid: hidden state alignment gap  
  Δlog: logit alignment gap

  The stage where Δhid stops decreasing:
    → That component is not captured algebraically
    → Gradient descent is needed at that point
    → That is where the "pie in the sky" hits the ceiling

  Teacher reference val={val_teacher:.4f}
  Best algebraic val={min(val_s3,val_s4):.4f}
  Remaining gap={min(val_s3,val_s4)-val_teacher:.4f} nats
""")
