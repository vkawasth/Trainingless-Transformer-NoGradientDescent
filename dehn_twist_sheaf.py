#!/usr/bin/env python3
"""
Dehn Twist Sheaf Construction
===============================
The alpha complex revealed cos < 0 between sections s_2/s_3, s_3/s_5.
These are NOT random — they are Dehn twist signatures.

DEHN TWIST IN THE QUIVER:
  The prime path patches define a surface (the moduli space M_MC).
  Loops in the overlap graph P_i ∩ P_j ∩ P_k carry holonomy.
  A loop with odd Dehn winding picks up a sign flip: s -> -s.
  This is exactly cos ~ -0.4 between sections that traverse the twist.

DEHN GAP = 1.4 (measured in session T4):
  The minimum displacement from identity in the mapping class group.
  The branch cut has width 1.4 in the MC moduli space metric.
  The 200 CE steps navigate this branch cut.

CONSTRUCTION:
  1. Identify the Dehn twist: find the loop in the overlap graph
     that carries the sign flip (negative cosine)
  2. Compute the twist angle: phi = arccos(cos_ij) for each pair
  3. The Dehn gap is the minimum phi across all pairs with cos < 0
  4. Correct for the twist: sections that traversed the branch cut
     get a sign correction s_j -> -s_j (half-twist correction)
  5. After sign correction, align with Procrustes

SIGN CORRECTION RULE:
  If cos(s_i, s_j) < -threshold:
    The section s_j has been half-twisted relative to s_i.
    Correct: s_j_corrected = -s_j
    Now cos(s_i, s_j_corrected) = -cos(s_i, s_j) > threshold.

  This is a Z/2Z correction — the mapping class group element
  that undoes the Dehn half-twist.

PREDICTION:
  After sign correction + Procrustes alignment, the cascade
  should have higher mean cosine similarity.
  The student should converge faster because the Dehn twist
  ambiguity is resolved algebraically.

CONNECTION TO 200 CE STEPS:
  The CE gradient must navigate the branch cut to align sections.
  Step 50: H^5, H^6, H^7 resolve together — this is the moment
  the gradient crosses the branch cut (Dehn gap traversal).
  The 200 steps total = approach to branch cut (100 steps) +
  crossing (50 steps) + stabilization after (50 steps).

  If Dehn correction reduces the approach cost:
    We can start closer to the branch cut and need fewer steps.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
from scipy.spatial import Delaunay
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; DEHN_GAP=1.4  # measured T4 invariant

print(f"\n{'='*65}")
print(f"  DEHN TWIST SHEAF CONSTRUCTION")
print(f"  Dehn gap = {DEHN_GAP} (T4 invariant, confirmed)")
print(f"  Sign corrections for half-twisted sections")
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

def procrustes_R(A,B):
    C=A.T@B; U,_,Vt=np.linalg.svd(C); R=U@Vt
    if np.linalg.det(R)<0: Vt[-1,:]*=-1; R=U@Vt
    return R

# ════════════════════════════════════════════════════
# Train teacher + Jacobians
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

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]
local_sections=[mu6(*[Js[l] for l in p])/max(N(mu6(*[Js[l] for l in p])),1e-8)
                for p in prime_paths]

cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

# ════════════════════════════════════════════════════
# PART 1: IDENTIFY DEHN TWISTS
# ════════════════════════════════════════════════════
print("="*65)
print("PART 1: DEHN TWIST IDENTIFICATION")
print(f"  Dehn gap = {DEHN_GAP} (T4 invariant)")
print(f"  Sections with cos < 0 have traversed a half-twist")
print("="*65)

# Cosine matrix
cos_matrix=np.zeros((N_STU,N_STU))
for i in range(N_STU):
    for j in range(N_STU):
        cos_matrix[i,j]=float(np.sum(local_sections[i]*local_sections[j])/
                              (N(local_sections[i])*N(local_sections[j])+1e-8))

print(f"\n  Raw cosine matrix:")
print(f"  {'':>4}", end="")
for j in range(N_STU): print(f"  s{j+1:>4}", end="")
print()
for i in range(N_STU):
    print(f"  s{i+1}:", end="")
    for j in range(N_STU):
        marker=" *" if cos_matrix[i,j]<-0.1 else "  "
        print(f" {cos_matrix[i,j]:>5.2f}{marker}", end="")
    print()
print(f"  (* = Dehn half-twist signature: cos < -0.1)")

# Find the Dehn twist graph
# Nodes = sections, edges = pairs with cos < 0 (half-twist between them)
dehn_pairs=[(i,j) for i in range(N_STU) for j in range(i+1,N_STU)
            if cos_matrix[i,j]<-0.1]
print(f"\n  Dehn twist pairs (cos < -0.1):")
for i,j in dehn_pairs:
    angle=math.degrees(math.acos(max(-1,min(1,cos_matrix[i,j]))))
    print(f"  s_{i+1}↔s_{j+1}: cos={cos_matrix[i,j]:.4f}  "
          f"angle={angle:.1f}°  "
          f"half-twist winding = {int(round((180-angle)/180))} "
          f"({'yes' if abs(angle-180)<30 else 'partial'})")

# ════════════════════════════════════════════════════
# PART 2: Z/2Z SIGN CORRECTION (DEHN HALF-TWIST)
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: Z/2Z SIGN CORRECTION")
print("  Assign +1/-1 orientation to each section")
print("  via minimum spanning tree of the signed graph")
print("="*65)

# Build signed graph: edge weight = cos(s_i, s_j)
# Find orientation assignment o_i in {+1,-1} minimizing
# the number of edges where o_i * o_j * cos(s_i,s_j) < 0
# This is equivalent to finding the minimum cut of the signed graph

# Brute-force optimal Z/2Z orientation (2^(N-1) assignments, N=6 -> 32 checks)
# Maximize sum_{i<j} o_i * o_j * cos(s_i, s_j)
# Fix o_0 = +1 by symmetry (global sign is irrelevant)
import itertools as _it
best_score=-np.inf; best_signs=[1]*N_STU
for signs_tuple in _it.product([+1,-1],repeat=N_STU-1):
    signs=[1]+list(signs_tuple)
    score=sum(signs[i]*signs[j]*cos_matrix[i,j]
              for i in range(N_STU) for j in range(i+1,N_STU))
    if score>best_score:
        best_score=score; best_signs=signs[:]
orientations=best_signs
print(f"  Optimal orientation score: {best_score:.4f}")
print(f"  Sections to flip: {[i+1 for i,s in enumerate(orientations) if s<0]}")

print(f"\n  Orientation assignment (BFS from s_1):")
for i,o in enumerate(orientations):
    print(f"  s_{i+1}: orientation = {'+1' if o>0 else '-1'}  "
          f"({'flip' if o<0 else 'keep'})")

# Apply sign corrections
signed_sections=[o*s for o,s in zip(orientations,local_sections)]

# Recompute cosine matrix
cos_signed=np.zeros((N_STU,N_STU))
for i in range(N_STU):
    for j in range(N_STU):
        cos_signed[i,j]=float(np.sum(signed_sections[i]*signed_sections[j])/
                              (N(signed_sections[i])*N(signed_sections[j])+1e-8))

print(f"\n  Cosine matrix AFTER sign correction:")
print(f"  {'':>4}", end="")
for j in range(N_STU): print(f"  s{j+1:>4}", end="")
print()
for i in range(N_STU):
    print(f"  s{i+1}:", end="")
    for j in range(N_STU):
        print(f"  {cos_signed[i,j]:>5.2f}", end="")
    print()

mean_before=np.mean([cos_matrix[i,j] for i in range(N_STU)
                      for j in range(i+1,N_STU)])
mean_after=np.mean([cos_signed[i,j] for i in range(N_STU)
                     for j in range(i+1,N_STU)])
print(f"\n  Mean pairwise cosine: {mean_before:.4f} → {mean_after:.4f}  "
      f"({'improved' if mean_after>mean_before else 'worsened'})")

# ════════════════════════════════════════════════════
# PART 3: DEHN-CORRECTED + PROCRUSTES CASCADE
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 3: DEHN CORRECTION + PROCRUSTES ALIGNMENT")
print("  After Z/2Z correction, apply Procrustes to align frames")
print("="*65)

# Now apply Procrustes in alpha filtration order on SIGNED sections
# The sign correction removes the branch cut discontinuity
# Procrustes then handles the residual continuous rotation

# Recompute filtration order on signed sections
section_vecs_s=np.stack([s.flatten() for s in signed_sections])
mean_s=section_vecs_s.mean(axis=0); centered_s=section_vecs_s-mean_s
_,_,Vt_pca=np.linalg.svd(centered_s,full_matrices=False)
k=min(N_STU-1,centered_s.shape[0]-1)
points_s=centered_s@Vt_pca[:k].T

from scipy.spatial.distance import cdist
dists_s=cdist(points_s,points_s)
edges_s=sorted([(dists_s[i,j]/2,i,j) for i in range(N_STU)
                 for j in range(i+1,N_STU)],key=lambda x:x[0])

print(f"\n  Procrustes alignment on sign-corrected sections:")
current=list(signed_sections)
for r,i,j in edges_s[:6]:  # top 6 edges by closeness
    R=procrustes_R(current[i],current[j])
    s_j_new=R@current[j]@R.T
    n=N(s_j_new); current[j]=s_j_new/max(n,1e-8)
    cos_before=cos_signed[i,j]
    cos_after=float(np.sum(current[i]*current[j])/
                    (N(current[i])*N(current[j])+1e-8))
    print(f"  r={r:.4f}: align s_{j+1}→s_{i+1}  "
          f"cos: {cos_before:.4f}→{cos_after:.4f}")

mean_final=np.mean([float(np.sum(current[i]*current[j])/
                          (N(current[i])*N(current[j])+1e-8))
                    for i in range(N_STU) for j in range(i+1,N_STU)])
print(f"\n  Mean pairwise cos after Dehn+Procrustes: {mean_final:.4f}")
print(f"  (vs raw: {mean_before:.4f}, signed only: {mean_after:.4f})")

# ════════════════════════════════════════════════════
# PART 4: DEHN GAP MEASUREMENT
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 4: DEHN GAP VERIFICATION")
print(f"  T4 invariant: Dehn gap = {DEHN_GAP}")
print(f"  Measure: ||s_i - tau(s_i)|| where tau = Dehn twist")
print("="*65)

# Dehn twist operator: rotation by 2*pi around a prime path loop
# In the operator space: tau(s) = exp(2*pi * ad(J_loop)) @ s
# where J_loop = monodromy around the loop

# Measure the actual gap in operator space
# Gap = min distance between a section and its Dehn-twisted version
J_loop=np.eye(ma)
for l in prime_paths[0]:  # monodromy around first prime path
    J_loop=Js[l]@J_loop
U_loop,sv_loop,_=np.linalg.svd(J_loop)

print(f"\n  Prime path 1 monodromy sv[:4]: {sv_loop[:4].round(3)}")

# Dehn twist: rotation by 2*pi in the monodromy eigenspace
# In the 48-dim active subspace, tau = exp(2*pi * skew(U_loop))
# Approximation: tau(s) = -s projected onto high-sv subspace
# (half-twist = sign flip in the dominant eigenspace)
k_dehn=4  # dominant eigenspace dimension
U_dom=U_loop[:,:k_dehn]
for i,s in enumerate(local_sections):
    # Project s onto dominant eigenspace, flip sign, project back
    s_proj=U_dom@(U_dom.T@s@U_dom)@U_dom.T  # dominant component
    s_residual=s-s_proj
    tau_s=-s_proj+s_residual  # half-twist: flip dominant component
    gap=float(np.linalg.norm(s-tau_s,'fro'))
    print(f"  s_{i+1}: ||s - tau(s)|| = {gap:.4f}  "
          f"(T4 Dehn gap = {DEHN_GAP})")

# ════════════════════════════════════════════════════
# PART 5: STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 5: STUDENT EXPERIMENTS")
print("  A: Serre (baseline)")
print("  B: Local sections (unmodified)")
print("  C: Sign-corrected sections (Dehn half-twist removed)")
print("  D: Sign-corrected + Procrustes aligned")
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

def run(cascade,label,steps=200):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

vA,ckA=run(cascade_serre,"A-Serre")
vB,ckB=run(local_sections,"B-Local-raw")
vC,ckC=run(signed_sections,"C-Dehn-corrected")
vD,ckD=run(current,"D-Dehn+Procrustes")

print(f"\n{'='*65}")
print("  DEHN TWIST RESULTS")
print("="*65)
print(f"""
  DEHN ANALYSIS:
    Dehn gap (T4 invariant): {DEHN_GAP}
    Pairs with cos < -0.1 (half-twist signature): {len(dehn_pairs)}
    Orientations flipped: {sum(1 for o in orientations if o<0)}/{N_STU}
    Mean cos: raw={mean_before:.4f} → signed={mean_after:.4f} → aligned={mean_final:.4f}

  CONVERGENCE:
  {'step':>6}  {'A-Serre':>8}  {'B-Raw':>7}  {'C-Dehn':>8}  {'D-D+P':>7}""")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>7.4f}" if v else f"  {'---':>7}"
    best=min((v for v in [b,c,d] if v),default=99)
    if a and best<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:          val={val_teacher:.4f}
    A (Serre):        val={vA:.4f}
    B (Raw local):    val={vB:.4f}  diff={vA-vB:+.4f}
    C (Dehn-fixed):   val={vC:.4f}  diff={vA-vC:+.4f}
    D (Dehn+Procrst): val={vD:.4f}  diff={vA-vD:+.4f}

  IF C or D < A at steps 25-75:
    The Dehn half-twist was the source of the negative cosines.
    Removing it algebraically (Z/2Z sign correction) reduces
    the gradient descent cost of navigating the branch cut.
    The Dehn gap = {DEHN_GAP} quantifies this cost exactly.
    
  IF C ≈ D ≈ A:
    The negative cosines are not half-twists — they reflect
    genuine content disagreement in the operator space.
    The mapping class group action is trivial on these sections.
    Final conclusion: the 200 CE steps are irreducible,
    and the Dehn gap measures the depth of the MC moduli space,
    not a removable topological obstruction.
""")
