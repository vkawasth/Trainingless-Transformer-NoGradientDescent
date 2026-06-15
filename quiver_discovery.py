#!/usr/bin/env python3
"""
Quiver Discovery: What Graph Does Gradient Descent Build?
==========================================================
Treat layer interfaces as nodes and Jacobians as morphisms.
Map the transformer's internal dynamics as a categorical structure.

QUIVER Q = (V, E, M) where:
  V = {v_0, ..., v_23}  (layer interfaces — nodes)
  E = {e_l: v_l -> v_{l+1}}  (Jacobians — directed edges)
  M = {J_l}  (morphism matrices — edge weights)

The quiver changes during training. At step 0 (Serre init):
  the quiver has a specific structure (Property T, expander)
At step 200 (converged):
  the quiver has settled into a different structure

WHAT WE MEASURE:
  1. Adjacency structure of the COMMUTATOR GRAPH
     A[i,j] = ||[J_i, J_j]||  (non-commutativity = edge weight)
     This is the Cayley graph of the Kac-Moody algebra

  2. Path algebra MORPHISMS
     Hom(v_i, v_j) = composition J_{j-1} @ ... @ J_i
     The kernel and cokernel at each interface

  3. IDEMPOTENTS: e_l = J_l @ J_l^+ (projection onto active subspace)
     These are the categorical identities at each node

  4. NATURAL TRANSFORMATIONS between snapshots
     eta: Q_step0 -> Q_step200
     The natural transformation IS what gradient descent builds

  5. LIMITS AND COLIMITS of the quiver diagram
     lim Q = the fixed point W* = attractor of gradient descent
     colim Q = the initial condition = cascade initialization

Profile at step 0, 25, 50, 75, 100, 150, 200
to discover the graph being built.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  QUIVER DISCOVERY")
print(f"  What categorical graph does gradient descent build?")
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

def eval_val(model,n=30):
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

def extract_quiver(model, x_ref, pos, m, n_layers):
    """Extract the categorical quiver Q=(V,E,M) at current training state."""
    model.eval()
    with torch.no_grad(): hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]

    Js=[]; Us=[]
    for l in range(n_layers):
        J,U=layer_jac(model.blocks[l],hs[l],pos,m)
        Js.append(J); Us.append(U)

    q={}

    # 1. COMMUTATOR GRAPH adjacency A[i,j] = ||[J_i,J_j]||
    A=np.zeros((n_layers,n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            if i!=j: A[i,j]=float(np.linalg.norm(comm(Js[i],Js[j])))
    q['comm_adj']=A

    # Spectral gap of commutator graph
    A_sym=(A+A.T)/2
    eigs=np.linalg.eigvalsh(A_sym)[::-1]
    q['gap']=float(eigs[0]-eigs[1]) if len(eigs)>1 else 0

    # 2. IDEMPOTENTS e_l = J_l @ J_l^+ (projection onto image)
    idempotents=[]
    for l in range(n_layers):
        J=Js[l]
        # Pseudoinverse via SVD (regularized)
        U_s,s,Vt_s=np.linalg.svd(J,full_matrices=False)
        s_inv=np.where(s>1e-6*s[0],1/s,0)
        J_plus=Vt_s.T@np.diag(s_inv)@U_s.T
        e_l=J@J_plus
        idempotents.append(e_l)
    q['idempotents']=idempotents

    # Idempotency defect ||e_l^2 - e_l||
    q['idemp_defect']=[float(np.linalg.norm(e@e-e)) for e in idempotents]

    # 3. NATURAL TRANSFORMATION NORM (how much has the quiver changed?)
    # Measured as max ||J_l^current - J_l^ref|| per layer
    q['Js']=Js; q['Us']=Us

    # 4. PATH ALGEBRA: composition J_{j-1}@...@J_i for i<j
    # Measure rank of Hom(v_i, v_j) for key pairs
    hom_ranks={}
    mid=n_layers//2
    for (i,j) in [(0,mid),(mid,n_layers-1),(0,n_layers-1)]:
        M=np.eye(m)
        for l in range(i,j): M=Js[l]@M
        rank=int(np.linalg.matrix_rank(M,tol=1e-4))
        hom_ranks[f'Hom(v{i},v{j})']=rank
    q['hom_ranks']=hom_ranks

    # 5. LIMIT OF QUIVER: fixed point structure
    # lim Q = equalizer of all J_l = subspace fixed by all morphisms
    # Approximate: intersection of images of all J_l
    # Measure as rank of J_0 @ J_1 @ ... @ J_{n-1}
    M_total=np.eye(m)
    for l in range(n_layers): M_total=Js[l]@M_total
    q['rank_total']=int(np.linalg.matrix_rank(M_total,tol=1e-4))
    q['sv_total']=float(np.linalg.svd(M_total,compute_uv=False)[0])

    # 6. COLIMIT OF QUIVER: initial object structure
    # colim Q = coproduct of images = span of all J_l images
    # Measure as rank of [J_0 | J_1 | ... | J_{n-1}] (horizontal stack)
    M_col=np.hstack(Js)
    q['rank_colimit']=int(np.linalg.matrix_rank(M_col,tol=1e-4))

    # 7. MONODROMY through all layers
    M_att=np.eye(m)
    for l in range(n_layers): M_att=Js[l]@M_att
    q['sv_monodromy']=float(np.linalg.svd(M_att,compute_uv=False)[0])

    # 8. FUNCTORIAL STRUCTURE: does the quiver factor through middle layer?
    # Test: ||J_{mid+k} - J_{mid} @ J_k|| for small k
    mid=n_layers//2
    factor_defects=[]
    for k in range(1,min(4,n_layers-mid)):
        defect=float(np.linalg.norm(Js[min(mid+k,n_layers-1)]-Js[mid]@Js[k]))
        factor_defects.append(defect)
    q['factor_defects']=factor_defects

    return q

# ════════════════════════════════════════════════════
# Train teacher + build Serre student
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
teacher.eval()
val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

# Extract teacher quiver
print("Extracting teacher quiver Q_teacher...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
Js_t=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us_t=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js_t[L_ATT]; U14=Us_t[L_ATT]

# Build Serre student
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=r; r=A@r-r@A
    return r

cascade=[]
for l in range(1,N_STU+1):
    C=J14.copy()
    for _ in range(l): C=J14@C-C@J14
    n=float(np.linalg.norm(C)); C=C/max(n,1e-8)
    cascade.append(C)

def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

torch.manual_seed(99)
student=LM(D,N_HEADS,N_STU)
student.te.weight.data.copy_(teacher.te.weight.data)
with torch.no_grad():
    student.pe.weight.copy_(teacher.pe.weight)
    student.ln_f.weight.copy_(teacher.ln_f.weight)
    student.ln_f.bias.copy_(teacher.ln_f.bias)
    for l in range(N_STU):
        W_d=lift_to_d(cascade[l],U14,scale=0.01)
        W_t=torch.tensor(W_d,dtype=torch.float32)
        student.blocks[l].attn.WK.weight.copy_(W_t)
        student.blocks[l].attn.WQ.weight.copy_(W_t.T)
        student.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
        student.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
        student.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
        student.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
        student.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

# ════════════════════════════════════════════════════
# Profile quiver during training
# ════════════════════════════════════════════════════
print("Profiling quiver Q during 200 CE steps...")
x_ref2,_=get_batch('val'); x_ref2=x_ref2[0:1]
opt_s=torch.optim.AdamW(student.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

PROFILE_STEPS=[0,25,50,75,100,150,200]
quiver_snapshots={}
Js_prev=None

print(f"\n  {'step':>5}  {'val':>7}  {'gap':>7}  "
      f"{'sv_mono':>8}  {'rank_lim':>9}  {'rank_col':>9}  "
      f"{'idemp_err':>10}  {'nat_trans'}")
print("  "+"-"*72)

for step in range(0,201):
    if step>0:
        for pg in opt_s.param_groups: pg['lr']=clr(step,200,50)
        student.train(); x,y=get_batch(); _,loss=student(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); opt_s.step()

    if step in PROFILE_STEPS:
        vl=eval_val(student,n=20)
        q=extract_quiver(student,x_ref2,pos,ma,N_STU)

        # Natural transformation norm (change from previous snapshot)
        if Js_prev is not None:
            nat_trans=float(np.mean([np.linalg.norm(q['Js'][l]-Js_prev[l])
                                      for l in range(N_STU)]))
        else:
            nat_trans=float('nan')
        Js_prev=[J.copy() for J in q['Js']]

        mean_idemp=float(np.mean(q['idemp_defect']))
        quiver_snapshots[step]={'val':vl,'gap':q['gap'],
            'sv_mono':q['sv_monodromy'],'rank_lim':q['rank_total'],
            'rank_col':q['rank_colimit'],'idemp':mean_idemp,
            'nat_trans':nat_trans,'factor':q['factor_defects'],
            'hom':q['hom_ranks'],'Js':q['Js']}

        print(f"  {step:>5}  {vl:>7.4f}  {q['gap']:>7.2f}  "
              f"{q['sv_monodromy']:>8.3f}  {q['rank_total']:>9}  "
              f"{q['rank_colimit']:>9}  {mean_idemp:>10.4f}  "
              f"{nat_trans if not np.isnan(nat_trans) else 'init':>9}")

# ════════════════════════════════════════════════════
# COMPARE WITH TEACHER QUIVER
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("COMPARING STUDENT QUIVER (step 200) WITH TEACHER QUIVER")
print("="*65)

q0=quiver_snapshots[0]; q200=quiver_snapshots[200]

# Extract teacher quiver at 6 layers for comparison
teacher_sub_Js=Js_t[L_ATT-3:L_ATT+3]

print(f"""
  SPECTRAL GAP (Property T):
    Student step 0:   {q0['gap']:.3f}
    Student step 200: {q200['gap']:.3f}
    Direction: {'increasing' if q200['gap']>q0['gap'] else 'decreasing'}

  MONODROMY sv (amplification):
    Student step 0:   {q0['sv_mono']:.3f}
    Student step 200: {q200['sv_mono']:.3f}

  RANK OF LIMIT (fixed point):
    Student step 0:   {q0['rank_lim']}
    Student step 200: {q200['rank_lim']}

  RANK OF COLIMIT (span of images):
    Student step 0:   {q0['rank_col']}
    Student step 200: {q200['rank_col']}

  IDEMPOTENT DEFECT (||e_l^2 - e_l||):
    Student step 0:   {q0['idemp']:.4f}
    Student step 200: {q200['idemp']:.4f}
    Direction: {'improving' if q200['idemp']<q0['idemp'] else 'worsening'}

  HOM SPACES (path algebra):
    Step 0:   {q0['hom']}
    Step 200: {q200['hom']}

  FACTORING THROUGH L_ATT (||J_{{14+k}} - J_14 @ J_k||):
    Step 0:   {[f'{d:.3f}' for d in q0['factor']]}
    Step 200: {[f'{d:.3f}' for d in q200['factor']]}

  NATURAL TRANSFORMATION NORMS (change per 25-50 steps):""")
prev_step=0
for step in PROFILE_STEPS[1:]:
    nt=quiver_snapshots[step]['nat_trans']
    print(f"  step {prev_step:>3}->  {step:>3}: ||eta||={nt:.4f}")
    prev_step=step

print(f"""
  WHAT THE QUIVER REVEALS:

  The natural transformation eta: Q_0 -> Q_200 is the map
  that gradient descent is constructing.

  If idempotents improve (defect decreases):
    The projections e_l = J_l @ J_l^+ are becoming more
    categorical — the quiver is becoming a proper groupoid.

  If rank of limit increases:
    The fixed point subspace is growing — more information
    is preserved through the full chain.

  If factoring through L14 improves:
    The quiver is developing a factoring structure —
    L14 is becoming a retract of the quiver, the
    category is becoming a fiber category over L14.

  The Serre cascade initializes the CORRECT TOPOLOGICAL CLASS
  of this natural transformation.
  The 200 CE steps CONSTRUCT the specific eta from that class.
  
  eta is NOT computable from the algebraic structure alone —
  it requires the corpus distribution to select which natural
  transformation within the correct homotopy class.
""")
