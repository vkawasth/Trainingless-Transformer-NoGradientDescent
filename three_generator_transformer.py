#!/usr/bin/env python3
"""
Three-Generator Transformer
============================
The rank-3 output bottleneck (G1) confirmed at L21-L23 means
HF*(L0,L1) = R^3. The embedding only needs to encode three coordinates
per token — the projection onto the three Floer generators.

E[t] = c1(t)*g1 + c2(t)*g2 + c3(t)*g3 + Serre_correction

Where:
  g1, g2, g3 = top-3 singular vectors of δJ at L21-L23 (output generators)
  c1(t) = syntactic coordinate (positional statistics in sentence)
  c2(t) = semantic coordinate (co-occurrence cluster membership)
  c3(t) = pragmatic coordinate (discourse position statistics)

All coordinates computable from corpus statistics alone.
Serre cascade fills the remaining 253 dimensions algebraically.

PIPELINE:
  0. Train teacher (oracle)
  1. Extract three output generators g1,g2,g3 from L21-L23
  2. Compute three corpus coordinates per token
  3. Build E[t] = sum_k c_k(t)*g_k lifted to full D-space
  4. Inject Serre cascade blocks (ad(J14)^l)
  5. Train head only (100 steps)
  6. Compare to teacher and prior results
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; L_OUT_START=21  # output bottleneck layers

print(f"\n{'='*65}")
print(f"  THREE-GENERATOR TRANSFORMER")
print(f"  E[t] = c1*g1 + c2*g2 + c3*g3 + Serre cascade")
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
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract three output generators g1,g2,g3
# ════════════════════════════════════════════════════════
print("Stage 1: Extract three Floer generators from output bottleneck L21-L23...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None

# Extract Jacobians for all layers (5 references for stability)
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

# Extract output generators from L21-L23 (rank-3 bottleneck)
# The three generators are the top-3 LEFT singular vectors of δJ at output layers
generators=[]  # [3, D] — the three Floer generators in D-space
for l in range(L_OUT_START, N_LAYERS):
    dJ=Js[l]-np.eye(ma)
    Usv,sv,_=np.linalg.svd(dJ)
    U_l=Us[l]  # [D, ma]
    for k in range(3):
        # Lift top-k singular vector to D-space
        g_k=U_l@Usv[:,k]  # [D]
        g_k=g_k/max(np.linalg.norm(g_k),1e-8)
        generators.append(g_k)

# Average generators across output layers, take top 3 via SVD
G=np.stack(generators)  # [9, D]
Ug,sg,_=np.linalg.svd(G,full_matrices=False)
g1=G[0]; g2=G[1]; g3=G[2]  # from L21 — most direct measurement

# Better: SVD of all output generators to find consensus directions
_,_,Vg=np.linalg.svd(G,full_matrices=False)
g1=Vg[0]; g2=Vg[1]; g3=Vg[2]  # [D] each — consensus output generators
generators_matrix=np.stack([g1,g2,g3])  # [3,D]

print(f"  g1·g2={float(g1@g2):.4f}  g1·g3={float(g1@g3):.4f}  g2·g3={float(g2@g3):.4f}")
print(f"  (near-zero = orthogonal generators, as expected for Floer basis)")

# Check: project teacher embeddings onto generators
E_proj=E_teacher@generators_matrix.T  # [VOCAB, 3]
var_explained=float(np.var(E_proj)/np.var(E_teacher)*100)
print(f"  Variance of teacher emb explained by 3 generators: {var_explained:.2f}%\n")

# ════════════════════════════════════════════════════════
# STAGE 2: Compute three corpus coordinates per token
# ════════════════════════════════════════════════════════
print("Stage 2: Compute three corpus coordinates per token...")

N=len(train_ids)

# c1: SYNTACTIC — relative position in sentence
# Tokens at sentence start/end/middle have different syntactic roles
# Use position-weighted frequency: c1(t) = mean(position/SEQ) when token appears
print("  c1: syntactic (mean relative position in sequence)...")
pos_sum=np.zeros(VOCAB); pos_count=np.zeros(VOCAB)
for i,tok in enumerate(train_ids):
    if tok<VOCAB:
        rel_pos=(i%SEQ)/SEQ  # relative position in window
        pos_sum[tok]+=rel_pos; pos_count[tok]+=1
c1=np.where(pos_count>0, pos_sum/pos_count, 0.5)
c1=(c1-c1.mean())/(c1.std()+1e-8)  # normalize

# c2: SEMANTIC — co-occurrence cluster membership
# Use PMI with the most frequent token as the semantic anchor
# c2(t) = sum_j PMI(t,j) * freq(j) / Z — weighted semantic affinity
print("  c2: semantic (PMI-weighted cluster affinity)...")
bigram=np.zeros((VOCAB,VOCAB),dtype=np.float32)
unigram=np.zeros(VOCAB,dtype=np.float32)
for i in range(N-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB:
        bigram[a,b]+=1; unigram[a]+=1
unigram_b=unigram+1e-8
pmi_row=(bigram+1e-8)/(unigram_b[:,None]*unigram_b[None,:]+1e-8)*N
pmi_row=np.log(pmi_row); pmi_row=np.clip(pmi_row,0,None)  # PPMI
# c2 = first left singular vector of PPMI (semantic axis)
Upmi,spmi,_=np.linalg.svd(pmi_row,full_matrices=False)
c2=Upmi[:,0]  # [VOCAB] — first semantic axis
c2=(c2-c2.mean())/(c2.std()+1e-8)

# c3: PRAGMATIC — discourse position
# c3(t) = how often token appears at BEGINNING vs END of sequences
# (sentence-initial tokens are pragmatically different from sentence-final)
print("  c3: pragmatic (sentence-initial vs sentence-final frequency)...")
init_count=np.zeros(VOCAB); final_count=np.zeros(VOCAB); tok_count=np.zeros(VOCAB)
for i,tok in enumerate(train_ids):
    if tok<VOCAB:
        tok_count[tok]+=1
        if i%SEQ==0: init_count[tok]+=1
        if i%SEQ==SEQ-1: final_count[tok]+=1
# c3 = (initial_freq - final_freq) / total_freq
with np.errstate(divide='ignore',invalid='ignore'):
    c3=np.where(tok_count>0,
                (init_count-final_count)/tok_count,
                0.0)
c3=(c3-c3.mean())/(c3.std()+1e-8)

print(f"  Coordinates computed for {VOCAB} tokens")
print(f"  c1 range: [{c1.min():.3f}, {c1.max():.3f}]")
print(f"  c2 range: [{c2.min():.3f}, {c2.max():.3f}]")
print(f"  c3 range: [{c3.min():.3f}, {c3.max():.3f}]\n")

# ════════════════════════════════════════════════════════
# STAGE 3: Build three-generator embeddings
# ════════════════════════════════════════════════════════
print("Stage 3: Build E[t] = c1*g1 + c2*g2 + c3*g3 + Serre fill...")

# Three-generator embedding: [VOCAB, D]
C_mat=np.stack([c1,c2,c3],axis=1)  # [VOCAB, 3]
E_3gen=(C_mat@generators_matrix).astype(np.float32)  # [VOCAB, D]

# Scale to match teacher embedding norm
teacher_rms=float(np.sqrt(np.mean(E_teacher**2)))
e3g_rms=float(np.sqrt(np.mean(E_3gen**2)))+1e-8
E_3gen=E_3gen*(teacher_rms/e3g_rms)

# Measure alignment
En=E_3gen/(np.linalg.norm(E_3gen,axis=1,keepdims=True)+1e-8)
Tn=E_teacher/(np.linalg.norm(E_teacher,axis=1,keepdims=True)+1e-8)
row_cos_3gen=float(np.mean(np.sum(En*Tn,axis=1)))
print(f"  E_3gen row_cos with teacher: {row_cos_3gen:.4f}")

# Add Serre correction in the orthogonal complement of {g1,g2,g3}
# The Serre cascade fills the remaining 253 dimensions
# Use the Gram-modulated corpus for the complement
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
Gram=sum(Js[l].T@Js[l] for l in range(N_LAYERS))/N_LAYERS
Gram_D=U14@Gram@U14.T+(np.eye(D)-U14@U14.T)

# Project E_teacher onto complement of generators (for filling)
G_basis=generators_matrix  # [3,D]
proj_gen=E_3gen@G_basis.T@G_basis  # component in generator subspace
E_complement=E_teacher-E_teacher@G_basis.T@G_basis  # teacher's complement

# For our embedding: use Gram-modulated corpus for complement
# (the complement doesn't affect the three generators but fills the space)
bigram_norm=bigram/(bigram.sum(axis=1,keepdims=True)+1e-8)
Uc2,sc2,_=np.linalg.svd(bigram_norm,full_matrices=False)
E_corpus_full=(Uc2[:,:D]*np.sqrt(np.maximum(sc2[:D],0))).astype(np.float32)
# Remove generator components from corpus embedding
E_corpus_complement=E_corpus_full-E_corpus_full@G_basis.T@G_basis

# Combined: generators + corpus complement
scale_comp=float(np.linalg.norm(E_complement,'fro'))/max(
    float(np.linalg.norm(E_corpus_complement,'fro')),1e-8)
E_combined=(E_3gen+E_corpus_complement*scale_comp).astype(np.float32)
scale_final=teacher_rms/max(float(np.sqrt(np.mean(E_combined**2))),1e-8)
E_combined=E_combined*scale_final

En2=E_combined/(np.linalg.norm(E_combined,axis=1,keepdims=True)+1e-8)
row_cos_combined=float(np.mean(np.sum(En2*Tn,axis=1)))
print(f"  E_combined (3gen + complement) row_cos: {row_cos_combined:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 4: Build Serre cascade
# ════════════════════════════════════════════════════════
print("Stage 4: Serre cascade ad(J14)^l...")
cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
    print(f"  Level {l}: ||cascade|| = {n:.6f}")

# ════════════════════════════════════════════════════════
# STAGE 5: Assemble and test — three embedding variants
# ════════════════════════════════════════════════════════
print(f"\nStage 5: Assemble {N_STU}L student — test three embedding variants...")

def build_student(E_init, label):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    with torch.no_grad():
        E_t=torch.tensor(E_init[:VOCAB,:D].copy(),dtype=torch.float32)
        stu.te.weight.copy_(E_t)
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
    v0=eval_val(stu)
    print(f"\n  [{label}] zero-shot val={v0:.4f}")
    return stu, v0

def train_head_only(stu, steps, label):
    for p in stu.parameters(): p.requires_grad_(False)
    stu.head.weight.requires_grad_(True)
    stu.te.weight.requires_grad_(True)
    opt=torch.optim.AdamW([stu.head.weight,stu.te.weight],lr=LR,weight_decay=0.01)
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps,20)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([stu.head.weight,stu.te.weight],1.0); opt.step()
        if step%(steps//4)==0:
            print(f"    step {step}  val={eval_val(stu,n=20):.4f}")
    for p in stu.parameters(): p.requires_grad_(True)
    return eval_val(stu)

def full_tune(stu, steps, label):
    opt=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt.step()
        if step%(steps//4)==0:
            print(f"    step {step}  val={eval_val(stu,n=20):.4f}")
    return eval_val(stu)

# Variant A: pure 3-generator embedding
stuA,v0A=build_student(E_3gen,"A: pure 3-gen")
vhA=train_head_only(stuA,100,"A head-only")
print(f"  [A] head-only 100 steps: val={vhA:.4f}")

# Variant B: 3-gen + corpus complement
stuB,v0B=build_student(E_combined,"B: 3-gen+complement")
vhB=train_head_only(stuB,100,"B head-only")
print(f"  [B] head-only 100 steps: val={vhB:.4f}")

# Variant C: teacher embeddings (oracle reference)
stuC,v0C=build_student(E_teacher,"C: teacher emb (oracle)")
vhC=train_head_only(stuC,100,"C head-only")
print(f"  [C] head-only 100 steps: val={vhC:.4f}")

# Full fine-tune the best non-oracle variant
best_stu=stuA if vhA<vhB else stuB
best_label="A" if vhA<vhB else "B"
print(f"\n  Full fine-tune on best variant {best_label} (200 steps)...")
vf=full_tune(best_stu,200,f"{best_label} full")

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  THREE-GENERATOR TRANSFORMER RESULTS")
print("="*65)
print(f"""
  FLOER GENERATORS (from L21-L23 output bottleneck):
    g1·g2={float(g1@g2):.4f}  g1·g3={float(g1@g3):.4f}  g2·g3={float(g2@g3):.4f}
    Variance of teacher emb in generator subspace: {var_explained:.2f}%

  CORPUS COORDINATES (no model needed):
    c1 (syntactic, position):    range [{c1.min():.2f}, {c1.max():.2f}]
    c2 (semantic, PMI cluster):  range [{c2.min():.2f}, {c2.max():.2f}]
    c3 (pragmatic, discourse):   range [{c3.min():.2f}, {c3.max():.2f}]

  EMBEDDING ALIGNMENT:
    Pure 3-gen row_cos:          {row_cos_3gen:.4f}
    3-gen+complement row_cos:    {row_cos_combined:.4f}
    (teacher oracle row_cos:     1.0000)

  RESULTS:
    Teacher (24L oracle):        val={val_teacher:.4f}

    A: Pure 3-gen + Serre:
       zero-shot:                val={v0A:.4f}
       head-only 100 steps:      val={vhA:.4f}

    B: 3-gen+complement + Serre:
       zero-shot:                val={v0B:.4f}
       head-only 100 steps:      val={vhB:.4f}

    C: Teacher emb + Serre (oracle):
       zero-shot:                val={v0C:.4f}
       head-only 100 steps:      val={vhC:.4f}

    Best non-oracle full tune:   val={vf:.4f}

  REFERENCE (prior experiments):
    Serre + teacher emb (200 CE full): val=0.187
    6L random + teacher emb (200 CE):  val=0.510
    Full algebraic (200 CE):           val=0.397

  KEY READING:
  If vhA or vhB approaches vhC (teacher oracle head-only):
    The three generators + corpus coordinates recover
    the essential embedding structure. Training-free
    orientation is achievable from the Floer generators.

  If vhA >> vhC:
    The three generators are necessary but not sufficient.
    The complement (remaining 253 dims) matters for head alignment.
    But the gap narrows with full fine-tune.

  The variance_explained tells us how much of the teacher's
  embedding structure lives in the three-generator subspace.
""")
