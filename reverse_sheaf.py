#!/usr/bin/env python3
"""
Reverse Sheaf Construction
===========================
Instead of global gradient descent, identify patches where the
sheaf is already locally consistent, then use targeted gradient
flow only to correct the gluing failures between patches.

PIPELINE:
  1. MAP: For each corpus sequence x, identify which quiver patches
     it activates (which layers see large hidden state transitions)

  2. LOCAL SECTIONS: For each prime path patch P=(l1,...,l6),
     construct the local MC element s_P from corpus sequences
     that activate P. This is a local algebraic computation.

  3. MAYER-VIETORIS: Compute gluing obstructions between overlapping
     patches. The H^1 obstruction = Ext^1 ~ 8 per layer interface.

  4. TARGETED GRADIENT: Restrict CE gradient flow to directions
     that correct the H^1 gluing failures.
     Instead of 200 steps of global gradient, do:
       - Identify the k layers with largest gluing failure
       - Freeze all other layers
       - Run CE gradient on the gluing-failure layers only
     This is the "surgical correction" — local homotopy.

  5. ONE-SHOT ASSEMBLY: After local corrections, assemble the
     global section from the patched local sections.

HYPOTHESIS:
  The prime path patches are already locally consistent (this is
  why they have high mu6 weight — they are the flattest regions).
  The gluing failures are concentrated at specific layer interfaces.
  Targeted gradient on those interfaces only = fewer steps needed.

MEASUREMENT:
  A: Standard Serre + 200CE (baseline, val=0.187)
  B: Reverse sheaf — local sections + targeted gradient (k layers)
  C: Reverse sheaf — with Mayer-Vietoris correction

Compare convergence: does B reach val<0.25 in fewer steps than A?
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import logm as scipy_logm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  REVERSE SHEAF CONSTRUCTION")
print(f"  Local patches + targeted gradient flow")
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
# Train teacher + extract Jacobians
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
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

# Serre cascade (baseline)
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

# Prime paths (confirmed attractor basin)
att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],
              key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c in prime_paths]

print(f"\n  Prime paths: {prime_paths}")

# ════════════════════════════════════════════════════
# STEP 1: MAP CORPUS TO QUIVER PATCHES
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 1: MAP CORPUS SEQUENCES TO QUIVER PATCHES")
print("  Which layers does each sequence 'activate'?")
print("="*65)

def get_sequence_patch_activation(model, x_seq, prime_paths, pos, threshold=0.3):
    """
    For a sequence x, find which prime path patches it activates.
    Activation = hidden state change at that layer is above threshold.
    Returns: dict {patch_idx: activation_strength}
    """
    model.eval()
    with torch.no_grad():
        hs=model.hidden_states(x_seq.unsqueeze(0))
        hs=[h[0,pos,:].numpy() for h in hs]

    # Hidden state changes at each layer
    dh=[np.linalg.norm(hs[l+1]-hs[l]) for l in range(len(hs)-1)]

    # Which patches are activated?
    activations={}
    for i,patch in enumerate(prime_paths):
        # Patch is activated if mean hidden state change in its layers is high
        patch_act=np.mean([dh[l] for l in patch if l<len(dh)])
        if patch_act>threshold:
            activations[i]=float(patch_act)
    return activations, dh

# Sample corpus to build patch activation map
print(f"\n  Sampling 200 corpus sequences for patch activation...")
N_CORPUS=200
patch_sequences={i:[] for i in range(N_STU)}  # patch -> list of sequences
all_dh=[]

torch.manual_seed(0)
for seq_idx in range(N_CORPUS):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    x_seq=train_t[ix:ix+SEQ]
    acts,dh=get_sequence_patch_activation(teacher,x_seq,prime_paths,pos)
    all_dh.append(dh)
    for patch_i,strength in acts.items():
        patch_sequences[patch_i].append((ix,strength))

print(f"\n  Patch activation statistics:")
for i,patch in enumerate(prime_paths):
    n_act=len(patch_sequences[i])
    mean_str=np.mean([s for _,s in patch_sequences[i]]) if patch_sequences[i] else 0
    print(f"  Patch {i+1} {patch}: {n_act}/{N_CORPUS} sequences "
          f"({100*n_act/N_CORPUS:.0f}%), mean activation={mean_str:.3f}")

# ════════════════════════════════════════════════════
# STEP 2: LOCAL SECTION PER PATCH
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 2: LOCAL SECTION CONSTRUCTION PER PATCH")
print("  s_P = corpus-weighted Jacobian restricted to patch P")
print("="*65)

# For each patch, compute the corpus-averaged Jacobian restricted to patch layers
local_sections=[]
for i,patch in enumerate(prime_paths):
    seqs=patch_sequences[i]
    if not seqs:
        # No activating sequences — use Jacobian average over all
        local_sec=mu6(*[Js[l] for l in patch])
        n=N(local_sec); local_sections.append(local_sec/max(n,1e-8))
        print(f"  Patch {i+1}: no activating sequences, using global mu6")
        continue

    # Compute Jacobians for activating sequences, weighted by activation strength
    J_patch_acc={l:[] for l in patch}
    weights=[]
    torch.manual_seed(0)
    for ix,strength in seqs[:20]:  # max 20 per patch
        x_seq=train_t[ix:ix+SEQ].unsqueeze(0)
        with torch.no_grad(): hs=teacher.hidden_states(x_seq); hs=[h[0] for h in hs]
        for l in patch:
            J_l,_=layer_jac(teacher.blocks[l],hs[l],pos,m)
            J_patch_acc[l].append(J_l*strength)
        weights.append(strength)

    w_sum=sum(weights)
    J_patch={l:np.sum(J_patch_acc[l],axis=0)/w_sum for l in patch}

    # Local section = mu6 of patch-specific Jacobians
    patch_Js=[J_patch[l] for l in patch]
    local_sec=mu6(*patch_Js)
    n=N(local_sec)
    local_sections.append(local_sec/max(n,1e-8))
    print(f"  Patch {i+1} {patch}: {len(seqs)} activating seqs, "
          f"||mu6_local||={n:.5f}")

# ════════════════════════════════════════════════════
# STEP 3: MAYER-VIETORIS GLUING OBSTRUCTION
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 3: MAYER-VIETORIS GLUING OBSTRUCTION")
print("  H^1 = colimit failure at patch intersections")
print("="*65)

# For each pair of overlapping patches, compute the gluing defect
print(f"\n  Patch overlap analysis:")
gluing_defects={}
for i in range(N_STU):
    for j in range(i+1,N_STU):
        overlap=set(prime_paths[i])&set(prime_paths[j])
        if not overlap: continue

        # Gluing defect: how much do the local sections disagree on the overlap?
        # Use the difference in mu6 weights projected onto the overlap layers
        s_i=local_sections[i]
        s_j=local_sections[j]
        # Cosine similarity of local sections (should be ~1 for consistent patches)
        cos=float(np.sum(s_i*s_j)/(N(s_i)*N(s_j)+1e-8))
        defect=1-cos  # 0=consistent, 2=anti-consistent
        gluing_defects[(i,j)]=defect
        print(f"  Patches {i+1}∩{j+1}: overlap={sorted(overlap)}, "
              f"cos={cos:.4f}, defect={defect:.4f}")

# Find layer interfaces with largest gluing failure
# These are the layers where we need targeted gradient flow
layer_defect=np.zeros(N_LAYERS_T)
for (i,j),defect in gluing_defects.items():
    overlap=set(prime_paths[i])&set(prime_paths[j])
    for l in overlap:
        layer_defect[l]+=defect

top_defect_layers=np.argsort(layer_defect)[::-1][:3]
print(f"\n  Layers with largest gluing failure:")
for l in top_defect_layers:
    if layer_defect[l]>0:
        print(f"  L{l:>2}: cumulative defect={layer_defect[l]:.4f}")

# ════════════════════════════════════════════════════
# STEP 4: TARGETED GRADIENT FLOW
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 4: TARGETED GRADIENT FLOW")
print("  Freeze all layers except gluing-failure layers")
print("  Gradient flows only where the sheaf is torn")
print("="*65)

def build_student(cascade, label=""):
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

def train_targeted(stu, label, total_steps=200,
                   targeted_layers=None, targeted_steps=50):
    """
    Phase 1: targeted gradient — only gluing-failure layers train
    Phase 2: full gradient — all layers train
    """
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}

    for step in range(1,total_steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,total_steps,50)

        # Targeted phase: freeze non-gluing layers
        if targeted_layers and step<=targeted_steps:
            for l,block in enumerate(stu.blocks):
                # Only train layers that correspond to gluing failures
                is_gluing=(l in targeted_layers)
                for p in block.parameters():
                    p.requires_grad_(is_gluing)
            # Always train embedding and output
            stu.te.weight.requires_grad_(True)

        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()

        # Unfreeze after targeted phase
        if targeted_layers and step==targeted_steps:
            for p in stu.parameters(): p.requires_grad_(True)

        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            phase="[targeted]" if targeted_layers and step<=targeted_steps else "[full]"
            b="✓" if v<val_teacher else " "
            print(f"  [{label}]{phase} step {step:>4}  val={v:.4f} {b}")

    return eval_val(stu),ck

# Gluing failure layers mapped to student block indices
# Prime paths are in teacher layers (e.g. L10-L20)
# Map to student block indices (0-5) via position in prime path
gluing_student_layers=set()
for l in top_defect_layers:
    if layer_defect[l]>0:
        # Find which student block corresponds to this teacher layer
        for i,patch in enumerate(prime_paths[:N_STU]):
            if l in patch:
                # Approximate mapping: block index = position in Serre cascade
                gluing_student_layers.add(i)

print(f"\n  Targeted student blocks: {sorted(gluing_student_layers)}")

# A: Standard Serre + 200CE
stuA=build_student(cascade_serre)
print(f"\n  A: Standard Serre + 200CE:")
vA,ckA=train_targeted(stuA,"A-std",200,targeted_layers=None)

# B: Prime cascade + targeted gradient (gluing layers first, then all)
stuB=build_student(cascade_prime)
print(f"\n  B: Prime cascade + targeted gradient (50 targeted + 150 full):")
vB,ckB=train_targeted(stuB,"B-prime-targeted",200,
                       targeted_layers=gluing_student_layers,targeted_steps=50)

# C: Prime cascade + local sections + targeted gradient
# Replace cascade with corpus-specific local sections
cascade_local=local_sections  # already normalized
stuC=build_student(cascade_local)
print(f"\n  C: Local sections + targeted gradient:")
vC,ckC=train_targeted(stuC,"C-local-targeted",200,
                       targeted_layers=gluing_student_layers,targeted_steps=50)

# D: Local sections + full gradient (no targeting)
stuD=build_student(cascade_local)
print(f"\n  D: Local sections + full gradient (control):")
vD,ckD=train_targeted(stuD,"D-local-full",200,targeted_layers=None)

print(f"\n{'='*65}")
print("  REVERSE SHEAF RESULTS")
print("="*65)

print(f"""
  PATCH ACTIVATION:
    {sum(len(v) for v in patch_sequences.values())}/{N_CORPUS*N_STU} patch-sequence activations

  GLUING DEFECTS (cosine disagreement between local sections):
""")
for (i,j),d in sorted(gluing_defects.items(),key=lambda x:-x[1])[:5]:
    print(f"    Patches {i+1}∩{j+1}: defect={d:.4f}")

print(f"""
  CONVERGENCE:
  {'step':>6}  {'A-std':>7}  {'B-prime-tgt':>12}  {'C-local-tgt':>12}  {'D-local-full':>13}""")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    if any(v and a and v<a-0.005 for v in [b,c,d]): row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:                val={val_teacher:.4f}
    A (Serre+std):          val={vA:.4f}
    B (prime+targeted):     val={vB:.4f}  diff={vA-vB:+.4f}
    C (local+targeted):     val={vC:.4f}  diff={vA-vC:+.4f}
    D (local+full):         val={vD:.4f}  diff={vA-vD:+.4f}

  REVERSE SHEAF INTERPRETATION:
    IF B or C < A at step 50:
      Targeted gradient on gluing-failure layers accelerates convergence.
      The sheaf patches identify where gradient descent should flow.
      One-shot: compute patches algebraically, flow gradient only there.

    IF B or C < A at step 200 but not earlier:
      Local sections improve final quality but not convergence speed.
      The patching is correct but the targeting adds little.

    IF A ≈ B ≈ C ≈ D:
      The gluing failures are distributed uniformly — no specific
      layers benefit from targeted gradient. The sheaf tears everywhere
      simultaneously and gradient descent must flow globally.
      This would confirm the 200-step irreducibility at the layer level.
""")
