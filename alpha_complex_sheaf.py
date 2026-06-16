#!/usr/bin/env python3
"""
Alpha Complex Sheaf — Steepest Descent as Frame Rotation
==========================================================
Vietoris-Rips is too coarse: single threshold, connects everything or nothing.

Alpha complex uses Delaunay triangulation of the local section vectors.
Each simplex {s_i, s_j, ...} enters the filtration at its circumradius.
This gives the FINEST complex consistent with the local geometry.

KEY INSIGHT:
  Gradient descent finds steepest descent on the cosine landscape.
  It rotates all local sections simultaneously toward cos=1.
  The 200 CE steps are NOT resolving content — they are finding
  the unique global frame where all sections are mutually cos=1.

  The alpha complex reveals the FILTRATION ORDER:
  Which section pairs reach cos=1 first? (small circumradius)
  Which pairs reach cos=1 last? (large circumradius)
  
  This IS the Reverse Hironaka blow-down sequence in SO(ma).
  The steepest descent path = the alpha filtration order.

CONSTRUCTION:
  1. Embed local sections {s_P} as points in R^(ma^2)
     (flattening the ma x ma operator matrices)
  2. Reduce dimension via PCA to R^k (k << ma^2)
  3. Compute Delaunay triangulation of the k-dimensional points
  4. Build alpha complex: simplex enters at circumradius of its vertices
  5. Measure: at what filtration value does each pair become cos=1?
     (equivalently: when does the simplex {s_i, s_j} enter the complex?)
  6. Steepest descent path = sequence of simplices in filtration order

GRADIENT DESCENT = ALPHA FILTRATION:
  At step t, gradient descent has resolved all simplices with
  circumradius < r(t), where r(t) is a decreasing function of training.
  The 200 CE steps sweep r from r_max (initial, all sections orthogonal)
  to r_min (final, all sections aligned, cos~1).
  
  The ROTATION that aligns each pair {s_i, s_j} is:
    R_{ij} = Procrustes(s_i, s_j)
  The filtration order tells us which R_{ij} to apply first.
  If we apply them in filtration order algebraically (no gradient),
  we reconstruct the global frame without CE steps.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  ALPHA COMPLEX SHEAF")
print(f"  Steepest descent = alpha filtration in SO(ma)")
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
    """Rotation R minimizing ||A - R@B@R^T||_F. Returns R in SO(ma)."""
    C=A.T@B; U,_,Vt=np.linalg.svd(C); R=U@Vt
    if np.linalg.det(R)<0: Vt[-1,:]*=-1; R=U@Vt
    return R

def circumradius_2pts(p1,p2):
    """Circumradius of edge = half distance."""
    return float(np.linalg.norm(p1-p2))/2

def circumradius_3pts(p1,p2,p3):
    """Circumradius of triangle via formula."""
    a=np.linalg.norm(p2-p3); b=np.linalg.norm(p1-p3); c=np.linalg.norm(p1-p2)
    area=np.abs(np.cross(p2-p1,p3-p1))/2
    if area<1e-12: return float('inf')
    return float(a*b*c/(4*area))

# ════════════════════════════════════════════════════
# Train teacher
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

# Prime paths + local sections
att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]
local_sections=[mu6(*[Js[l] for l in p])/max(N(mu6(*[Js[l] for l in p])),1e-8)
                for p in prime_paths]

# Serre cascade
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

# ════════════════════════════════════════════════════
# PART 1: ALPHA COMPLEX OF LOCAL SECTIONS
# ════════════════════════════════════════════════════
print("="*65)
print("PART 1: ALPHA COMPLEX OF LOCAL SECTIONS")
print("  Embed sections in R^k, Delaunay triangulation,")
print("  filtration by circumradius = cosine distance")
print("="*65)

# Embed sections as vectors in R^(ma^2), reduce via PCA
section_vecs=np.stack([s.flatten() for s in local_sections])  # (N_STU, ma^2)

# PCA to k dimensions (k = N_STU - 1 for exact embedding)
k=min(N_STU-1, section_vecs.shape[0]-1, section_vecs.shape[1])
mean_s=section_vecs.mean(axis=0)
centered=section_vecs-mean_s
_,_,Vt_pca=np.linalg.svd(centered,full_matrices=False)
points=centered@Vt_pca[:k].T  # (N_STU, k) PCA embedding

print(f"\n  Section vectors: {section_vecs.shape}")
print(f"  PCA embedding: {points.shape} (k={k})")
print(f"\n  Section points in PCA space:")
for i in range(N_STU):
    print(f"  s_{i+1}: [{', '.join(f'{x:.3f}' for x in points[i])}]")

# Pairwise cosine similarities (in original space)
print(f"\n  Pairwise cosine similarities (raw):")
cos_matrix=np.zeros((N_STU,N_STU))
for i in range(N_STU):
    for j in range(N_STU):
        cos_matrix[i,j]=float(np.sum(local_sections[i]*local_sections[j])/
                              (N(local_sections[i])*N(local_sections[j])+1e-8))

print(f"  {'':>4}", end="")
for j in range(N_STU): print(f"  s{j+1:>4}", end="")
print()
for i in range(N_STU):
    print(f"  s{i+1}:", end="")
    for j in range(N_STU):
        print(f"  {cos_matrix[i,j]:>5.2f}", end="")
    print()

# ════════════════════════════════════════════════════
# PART 2: FILTRATION — ALPHA COMPLEX BUILD
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: ALPHA COMPLEX FILTRATION")
print("  Simplices ordered by circumradius = blow-down order")
print("="*65)

# Pairwise Euclidean distances in PCA space
dists=cdist(points,points)

# 0-simplices: all sections, enter at r=0
simplices_0=[(0.0,(i,)) for i in range(N_STU)]

# 1-simplices: edges, enter at r = half-distance
simplices_1=[]
for i in range(N_STU):
    for j in range(i+1,N_STU):
        r=dists[i,j]/2
        cos_ij=cos_matrix[i,j]
        simplices_1.append((r,(i,j),cos_ij))
simplices_1.sort(key=lambda x:x[0])

# 2-simplices: triangles, enter at circumradius
simplices_2=[]
if k>=2 and N_STU>=3:
    try:
        tri=Delaunay(points[:,:2] if k>=2 else points)  # 2D Delaunay
        for simplex in tri.simplices:
            i,j,k_=sorted(simplex)
            p1,p2,p3=points[i,:2],points[j,:2],points[k_,:2]
            r=circumradius_3pts(p1,p2,p3)
            simplices_2.append((r,(i,j,k_)))
        simplices_2.sort(key=lambda x:x[0])
    except Exception as e:
        print(f"  Delaunay failed: {e}")

print(f"\n  ALPHA FILTRATION ORDER (steepest descent path):")
print(f"  This is the order gradient descent resolves sections to cos=1")
print()
print(f"  {'r':>8}  {'simplex':>15}  {'type':>10}  {'cos(i,j)':>10}  {'interpretation'}")
print("  "+"-"*70)

all_simplices=[]
for r,(i,) in simplices_0:
    all_simplices.append((r,(i,),'vertex',1.0))
for r,(i,j),cos_ij in simplices_1:
    all_simplices.append((r,(i,j),'edge',cos_ij))
for r,(i,j,k_) in simplices_2:
    cos_avg=(cos_matrix[i,j]+cos_matrix[j,k_]+cos_matrix[i,k_])/3
    all_simplices.append((r,(i,j,k_),'triangle',cos_avg))
all_simplices.sort(key=lambda x:x[0])

for r,simplex,stype,cos_val in all_simplices:
    if stype=='vertex':
        interp=f"section s_{simplex[0]+1} initialized"
    elif stype=='edge':
        interp=f"s_{simplex[0]+1}↔s_{simplex[1]+1} aligned (cos={cos_val:.3f}→1)"
    else:
        interp=f"triangle s_{simplex[0]+1}s_{simplex[1]+1}s_{simplex[2]+1} consistent"
    print(f"  {r:>8.4f}  {str(simplex):>15}  {stype:>10}  {cos_val:>10.4f}  {interp}")

# ════════════════════════════════════════════════════
# PART 3: STEEPEST DESCENT PATH — ONE-SHOT ROTATION
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 3: ONE-SHOT ROTATION FOLLOWING FILTRATION ORDER")
print("  Apply Procrustes rotations in alpha filtration order")
print("  This reconstructs the global frame algebraically")
print("="*65)

# Process edges in filtration order — apply rotations sequentially
# Start: all sections in their own frames
current_sections=list(local_sections)
frame_rotations=[np.eye(ma)]*N_STU  # accumulated rotation for each section

print(f"\n  Applying rotations in filtration order:")
for r,(i,j),cos_ij in simplices_1:
    s_i=current_sections[i]; s_j=current_sections[j]
    # Procrustes: find R rotating s_j into s_i's frame
    R_ij=procrustes_R(s_i,s_j)
    # Rotate section j
    s_j_new=R_ij@s_j@R_ij.T
    n=N(s_j_new); current_sections[j]=s_j_new/max(n,1e-8)
    frame_rotations[j]=R_ij@frame_rotations[j]
    cos_after=float(np.sum(current_sections[i]*current_sections[j])/
                    (N(current_sections[i])*N(current_sections[j])+1e-8))
    print(f"  r={r:.4f}: rotate s_{j+1} into s_{i+1}'s frame  "
          f"cos: {cos_ij:.4f} → {cos_after:.4f}")

# Global section = mean of now-aligned sections
global_section=np.mean(current_sections,axis=0)
global_section/=max(N(global_section),1e-8)

print(f"\n  Post-alignment cosine matrix:")
print(f"  {'':>4}", end="")
for j in range(N_STU): print(f"  s{j+1:>4}", end="")
print()
for i in range(N_STU):
    print(f"  s{i+1}:", end="")
    for j in range(N_STU):
        cos_ij=float(np.sum(current_sections[i]*current_sections[j])/
                    (N(current_sections[i])*N(current_sections[j])+1e-8))
        print(f"  {cos_ij:>5.2f}", end="")
    print()

# ════════════════════════════════════════════════════
# PART 4: CASCADE FROM ALPHA-ALIGNED GLOBAL SECTION
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 4: CASCADE FROM ALPHA-ALIGNED GLOBAL SECTION")
print("  Compare: Serre vs alpha-aligned cascade")
print("="*65)

# Build cascade levels from aligned sections
cascade_alpha=current_sections  # already aligned

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
vB,ckB=run(local_sections,"B-Unaligned-sections")
vC,ckC=run(cascade_alpha,"C-Alpha-aligned")

print(f"\n{'='*65}")
print("  ALPHA COMPLEX RESULTS")
print("="*65)

mean_cos_before=np.mean([cos_matrix[i,j] for i in range(N_STU)
                          for j in range(i+1,N_STU)])
cos_after_vals=[[float(np.sum(current_sections[i]*current_sections[j])/
                       (N(current_sections[i])*N(current_sections[j])+1e-8))
                  for j in range(i+1,N_STU)] for i in range(N_STU)]
mean_cos_after=np.mean([v for row in cos_after_vals for v in row])

print(f"""
  ALPHA FILTRATION:
    Sections: {N_STU} local sections in R^{ma**2} (PCA → R^{k})
    Mean pairwise cos BEFORE alignment: {mean_cos_before:.4f}
    Mean pairwise cos AFTER alignment:  {mean_cos_after:.4f}
    Improvement: {mean_cos_after-mean_cos_before:+.4f}

  FILTRATION ORDER (Reverse Hironaka blow-down in SO(ma)):""")
for r,(i,j),cos_ij in simplices_1:
    print(f"    r={r:.4f}: s_{i+1}↔s_{j+1}  (initial cos={cos_ij:.4f})")

print(f"""
  CONVERGENCE:
  {'step':>6}  {'A-Serre':>8}  {'B-Unalign':>10}  {'C-Alpha':>9}""")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    if a and c and c<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:         val={val_teacher:.4f}
    A (Serre):       val={vA:.4f}
    B (Unaligned):   val={vB:.4f}  diff={vA-vB:+.4f}
    C (Alpha-align): val={vC:.4f}  diff={vA-vC:+.4f}

  INTERPRETATION:
    The alpha filtration gives the steepest descent ORDER:
    which pairs of sections gradient descent aligns first.
    
    IF C < A: the alpha-aligned cascade is a better starting point.
      The rotational frame computed from alpha filtration captures
      what gradient descent does in Phase 1 (steps 0-75).
      One-shot rotation replaces Phase 1 CE steps.
    
    IF C ≈ A: the rotational alignment is not the bottleneck.
      The sections are already in compatible frames — the
      cos~0 defects are CONTENT disagreements, not frame disagreements.
      Gradient descent is resolving content, not rotation.
      The 200 CE steps are truly irreducible.
      
    The alpha complex gives the FINEST possible topology:
    only geometrically necessary simplices are included.
    Vietoris-Rips would connect all pairs at a single threshold,
    missing the filtration order that encodes the gradient path.
""")
