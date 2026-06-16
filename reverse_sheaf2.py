#!/usr/bin/env python3
"""
Reverse Sheaf Construction v2 — Corrected
==========================================
Fixes from v1:
  1. Relative patch activation (top-25% per layer, not absolute threshold)
  2. Proper targeted blocks (overlap-specific, not union of all)
  3. Functor construction: explicit corpus -> quiver mapping

The functor Phi: C_corpus -> C_ctx is constructed as follows:
  - Objects: token sequences (corpus) -> layer stalks (quiver)
  - Morphisms: token transitions -> Jacobian morphisms

For each sequence x:
  activation_l(x) = ||h_{l+1}(x) - h_l(x)|| / mean_l||h_{l+1}(x) - h_l(x)||
  
Sequence x SELECTIVELY activates patch P=(l1,...,l6) if:
  mean_{l in P} activation_l(x) > Q75(activation)
  AND it is above average specifically at P's layers

This gives genuine selectivity — most sequences activate 1-2 patches.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  REVERSE SHEAF v2 — CORRECTED FUNCTOR CONSTRUCTION")
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
def l3(a,b,c): return comm(comm(a,b),c)-comm(a,comm(b,c))
def N(A): return float(np.linalg.norm(A))
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)
def mu4(a,b,c,d):
    return -(comm(l3(a,b,c),d)-l3(comm(a,b),c,d)+l3(a,comm(b,c),d)
            -l3(a,b,comm(c,d))+comm(a,l3(b,c,d)))
def mu5(a,b,c,d,e):
    return -(l3(l3(a,b,c),d,e)-l3(a,l3(b,c,d),e)+l3(a,b,l3(c,d,e))
            +comm(mu4(a,b,c,d),e)+comm(a,mu4(b,c,d,e))
            -mu4(comm(a,b),c,d,e)+mu4(a,b,c,comm(d,e)))
def mu6(a,b,c,d,e,f):
    m5ab=mu5(a,b,c,d,e); m5bc=mu5(b,c,d,e,f)
    m4ab=mu4(a,b,c,d); m4bc=mu4(b,c,d,e); m4cd=mu4(c,d,e,f)
    m3ab=l3(a,b,c); m3bc=l3(b,c,d); m3cd=l3(c,d,e); m3de=l3(d,e,f)
    return -(comm(m5ab,f)-comm(a,m5bc)+l3(m4ab,e,f)-l3(a,m4bc,f)
            +l3(a,b,m4cd)+mu4(m3ab,d,e,f)-mu4(a,m3bc,e,f)
            +mu4(a,b,m3cd,f)-mu4(a,b,c,m3de))

# ════════════════════════════════════════════════════
# Train teacher + Jacobians
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
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c in prime_paths]
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

# ════════════════════════════════════════════════════
# FUNCTOR: Corpus -> Quiver (corrected)
# ════════════════════════════════════════════════════
print("="*65)
print("FUNCTOR CONSTRUCTION: C_corpus -> C_ctx")
print("  Relative activation: top-25% layers per sequence")
print("="*65)

N_CORPUS=200
print(f"\n  Pass 1: compute per-sequence activation profiles...")
all_activations=[]  # (N_CORPUS, N_LAYERS_T) normalized activations
all_indices=[]

torch.manual_seed(0)
for i in range(N_CORPUS):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    all_indices.append(ix)
    x_seq=train_t[ix:ix+SEQ]
    teacher.eval()
    with torch.no_grad():
        hs=teacher.hidden_states(x_seq.unsqueeze(0))
        hs=[h[0,pos,:].numpy() for h in hs]
    # Per-layer activation = hidden state change norm
    act=np.array([np.linalg.norm(hs[l+1]-hs[l]) for l in range(len(hs)-1)])
    # Normalize per sequence (relative activation)
    act=act/max(act.mean(),1e-8)
    all_activations.append(act)
    if (i+1)%50==0: print(f"  {i+1}/{N_CORPUS}...",flush=True)

all_activations=np.array(all_activations)  # (N_CORPUS, N_LAYERS_T-1)

# Global 75th percentile threshold per layer
layer_q75=np.percentile(all_activations,75,axis=0)

print(f"\n  Per-layer Q75 activation thresholds:")
for l in att_basin:
    if l<all_activations.shape[1]:
        print(f"  L{l:>2}: Q75={layer_q75[l]:.3f}  "
              f"mean={all_activations[:,l].mean():.3f}")

# Selective patch assignment: sequence activates patch P if
# it is ABOVE Q75 at MAJORITY of patch layers
print(f"\n  Pass 2: selective patch assignment...")
patch_sequences={i:[] for i in range(N_STU)}
patch_activations={i:[] for i in range(N_STU)}

for seq_i,(ix,act) in enumerate(zip(all_indices,all_activations)):
    for pi,patch in enumerate(prime_paths):
        patch_act=np.array([act[l] for l in patch if l<len(act)])
        thresh=np.array([layer_q75[l] for l in patch if l<len(act)])
        # Selective: must be above Q75 at majority of patch layers
        n_above=np.sum(patch_act>thresh)
        if n_above>=len(patch)//2+1:  # majority above Q75
            patch_sequences[pi].append((ix,float(patch_act.mean())))
            patch_activations[pi].append(float(patch_act.mean()))

print(f"\n  SELECTIVE patch activation statistics:")
for i,patch in enumerate(prime_paths):
    n=len(patch_sequences[i])
    mean_a=np.mean(patch_activations[i]) if patch_activations[i] else 0
    print(f"  Patch {i+1} {patch}: {n}/{N_CORPUS} seqs "
          f"({100*n/N_CORPUS:.0f}%), mean={mean_a:.3f}")

# ════════════════════════════════════════════════════
# LOCAL SECTIONS from selective corpus
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("LOCAL SECTIONS (selective corpus-weighted)")
print("="*65)

local_sections=[]
for pi,patch in enumerate(prime_paths):
    seqs=patch_sequences[pi]
    if len(seqs)<3:
        # Fallback to global Jacobians
        local_sec=mu6(*[Js[l] for l in patch])
        n=N(local_sec); local_sections.append(local_sec/max(n,1e-8))
        print(f"  Patch {pi+1}: too few seqs ({len(seqs)}), using global")
        continue

    J_patch_acc={l:[] for l in patch}; weights=[]
    for ix,strength in seqs[:30]:
        x_seq=train_t[ix:ix+SEQ].unsqueeze(0)
        with torch.no_grad(): hs=teacher.hidden_states(x_seq); hs=[h[0] for h in hs]
        for l in patch:
            J_l,_=layer_jac(teacher.blocks[l],hs[l],pos,m)
            J_patch_acc[l].append(J_l*strength)
        weights.append(strength)

    w_sum=sum(weights)
    J_patch={l:np.sum(J_patch_acc[l],axis=0)/w_sum for l in patch}
    local_sec=mu6(*[J_patch[l] for l in patch])
    n=N(local_sec); local_sections.append(local_sec/max(n,1e-8))
    print(f"  Patch {pi+1} {patch}: {len(seqs)} seqs, "
          f"||mu6_selective||={n:.5f}")

# Gluing defects
print(f"\n  Mayer-Vietoris gluing defects:")
defects={}
for i in range(N_STU):
    for j in range(i+1,N_STU):
        overlap=set(prime_paths[i])&set(prime_paths[j])
        if not overlap: continue
        cos=float(np.sum(local_sections[i]*local_sections[j])/
                  (N(local_sections[i])*N(local_sections[j])+1e-8))
        defects[(i,j)]=(1-cos,sorted(overlap))

# Sort by defect
for (i,j),(d,ov) in sorted(defects.items(),key=lambda x:-x[1][0])[:5]:
    print(f"  P{i+1}∩P{j+1}: defect={d:.4f}  overlap={ov}")

# ════════════════════════════════════════════════════
# TARGETED GRADIENT: only highest-defect interface
# ════════════════════════════════════════════════════
# Find highest-defect pair and its unique layers
top_pair=max(defects.items(),key=lambda x:x[1][0])
(pi,pj),(top_defect,top_overlap)=top_pair
print(f"\n  Highest-defect interface: P{pi+1}∩P{pj+1}, "
      f"defect={top_defect:.4f}, overlap={top_overlap}")

# Unique layers in highest-defect patches (not in overlap = the tear)
unique_i=set(prime_paths[pi])-set(top_overlap)
unique_j=set(prime_paths[pj])-set(top_overlap)
tear_layers=sorted(unique_i|unique_j)
print(f"  Tear layers (unique to each side): {tear_layers}")

# Map tear layers to student block indices
# The student has N_STU=6 blocks corresponding to cascade levels
# We target the blocks whose cascade operators come from these tear layers
tear_blocks=set()
for l in tear_layers:
    for bi,patch in enumerate(prime_paths):
        if l in patch and l not in top_overlap:
            tear_blocks.add(bi)
print(f"  Targeted student blocks: {sorted(tear_blocks)}")

# ════════════════════════════════════════════════════
# STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STUDENT EXPERIMENTS")
print("  A: Serre + 200CE (baseline)")
print("  B: Prime + 200CE (standard)")
print("  C: Local sections + full gradient")
print("  D: Local sections + targeted gradient (tear layers only, 50 steps)")
print("="*65)

def build_student(cascade):
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

def run(cascade,label,steps=200,targeted_blocks=None,targeted_steps=0):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        # Targeted phase
        if targeted_blocks and step<=targeted_steps:
            for bi,blk in enumerate(stu.blocks):
                for p in blk.parameters():
                    p.requires_grad_(bi in targeted_blocks)
            stu.te.weight.requires_grad_(True)
        elif targeted_blocks and step==targeted_steps+1:
            for p in stu.parameters(): p.requires_grad_(True)

        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            tag="[tgt]" if targeted_blocks and step<=targeted_steps else "[full]"
            print(f"  [{label}]{tag} step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

vA,ckA=run(cascade_serre,"A-Serre-std")
vB,ckB=run(cascade_prime,"B-Prime-std")
vC,ckC=run(local_sections,"C-Local-full")
vD,ckD=run(local_sections,"D-Local-targeted",
           targeted_blocks=tear_blocks,targeted_steps=50)

print(f"\n{'='*65}")
print("  REVERSE SHEAF v2 RESULTS")
print("="*65)
print(f"\n  FUNCTOR SELECTIVITY:")
for i in range(N_STU):
    print(f"  Patch {i+1}: {len(patch_sequences[i])}/{N_CORPUS} seqs selected")

print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Serre':>8}  {'B-Prime':>8}  "
      f"{'C-Local':>8}  {'D-Targeted':>11}")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    best=min((v for v in [b,c,d] if v),default=99)
    if a and best<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:           val={val_teacher:.4f}
    A (Serre+std):     val={vA:.4f}
    B (Prime+std):     val={vB:.4f}  diff={vA-vB:+.4f}
    C (Local+full):    val={vC:.4f}  diff={vA-vC:+.4f}
    D (Local+tgt50):   val={vD:.4f}  diff={vA-vD:+.4f}

  HIGHEST DEFECT INTERFACE:
    P{pi+1}∩P{pj+1}: defect={top_defect:.4f}
    Tear layers: {tear_layers}
    Targeted blocks: {sorted(tear_blocks)}

  THE REVERSE HIRONAKA PATH:
    Functor maps corpus selectively to patches.
    Local sections encode corpus-weighted local MC elements.
    Targeted gradient patches the highest-defect interface first.
    
    IF D converges faster than A at steps 25-75:
      The functor correctly identifies where gradient is most needed.
      The sheaf tear is real and patchable.
    
    IF D ≈ A ≈ B ≈ C:
      The sheaf tears everywhere simultaneously.
      The Reverse Hironaka path cannot be short-circuited at this level.
      The 200-step irreducibility holds at the patch level too.
      Next step: extend to H^7/H^8 with patched early exit architecture.
""")
