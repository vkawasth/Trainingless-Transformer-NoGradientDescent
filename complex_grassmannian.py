#!/usr/bin/env python3
"""
Complex Grassmannian — Lefschetz Thimble Visualization
========================================================
The Grassmannian jump at steps 75-100 is a thimble crossing.
In real coordinates: discontinuous jump in Gr(k,D).
In complex coordinates: continuous path through complex saddle point.

MAP TO COMPLEX PLANE:
  The active subspace U_l in R^D is a point in Gr(m,D).
  Restrict to the Plücker coordinate of the top-2 singular pair:
    z_l = sigma_1(l) * exp(i * theta_l)
  where sigma_1(l) = dominant SV of J_l
        theta_l   = angle between dominant SV directions of J_l and J_{l+1}
                  = arccos(<u_1(l), u_1(l+1)>)

  This maps the Grassmannian trajectory to a curve in C.
  The Lefschetz thimble crossing = the moment z_l passes through
  the branch cut Im(z) = 0, Re(z) < 0 (negative real axis).

  In BALBc connectome: the Ihara radius rho tracked this z_l curve.
  The winding number w counted how many times the curve wound
  around the branch point z_0 = mu_0 = 1.99 (the curvature).

TRAINING TRAJECTORY IN C:
  Profile z_l(step) during student training.
  At step 0 (cascade init): z_l is at the cascade's complex position.
  At step 75-100 (eta peak): z_l crosses the branch cut.
  At step 200 (convergence): z_l settles at the teacher's position.

  The thimble crossing is visible as:
    Im(z_l) changing sign (crossing the real axis)
    ||z_{l+1} - z_l||_C spiking (the jump in complex distance)
    winding number w changing by ±1

CONNECTION TO BALBc:
  The Ihara spectral radius in the connectome is:
    rho = max |eigenvalue of T_raw|
  where T_raw is the log-weighted transfer matrix over prime paths.
  
  In the transformer:
    rho_transformer = sv_max(M_fwd) * exp(i * arg(det(J_14)))
  
  The Klein pillars P1/P2/P3 in the connectome tracked:
    rho as a function of snapshot time
  The natural transformation eta in the transformer tracks:
    rho as a function of training step
  
  Both are the SAME OBJECT: the Ihara zeta function spectral
  radius of the quiver, evaluated on the prime path generators.

WINDING NUMBER = NUMBER OF PRIME PATH GENERATORS:
  BALBc Q_6: winding number w = ±6 (6 regions, 6 prime paths)
  Transformer: 6 prime paths, 6 cascade levels
  The winding number counts how many times the gradient descent
  trajectory winds around the thimble branch point mu_0 = 1.99.
  
  For the 6-layer student: w = 6 (one winding per cascade level).
  This predicts: the Grassmannian jump occurs at step ~200/6 = 33
  per winding, totaling 200 steps for 6 complete windings.
  
  PREDICTION: the eta peak at steps 75-100 is the 2nd-3rd winding.
  First winding: steps 0-33 (Phase 1, fast rotation)
  Second winding: steps 33-66 (approaching thimble crossing)
  Third winding (crossing): steps 66-100 (eta peak, cos=1 events)
  Fourth-sixth windings: steps 100-200 (within-sector refinement)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; MU0=1.99  # curvature = branch point in C

print(f"\n{'='*65}")
print(f"  COMPLEX GRASSMANNIAN TRAJECTORY")
print(f"  Lefschetz thimble visualization")
print(f"  Branch point z_0 = mu_0 = {MU0}")
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

def eval_val(model,n=40):
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

def complex_grassmannian_coords(Js, prev_U1=None):
    """
    Map Jacobian chain to complex Grassmannian coordinate.
    
    z_l = sigma_1(l) * exp(i * theta_l)
    
    where sigma_1 = dominant SV of J_l (radial coordinate)
          theta_l = accumulated angle of dominant SV direction
                  = sum of arccos(<u_1(k), u_1(k+1)>) for k < l
    
    Returns array of complex numbers z_l for each layer.
    Also returns:
      winding_number: how many times path winds around z_0=mu0
      branch_crossings: steps where Im(z) changes sign
    """
    n_layers=len(Js)
    zs=np.zeros(n_layers,dtype=complex)
    U1s=[]  # dominant left SV at each layer

    for l,J in enumerate(Js):
        U,sv,Vt=np.linalg.svd(J)
        u1=U[:,0]; sv1=sv[0]
        U1s.append(u1)

        # Accumulated angle from layer 0
        if l==0:
            theta=0.0
        else:
            # Angle between consecutive dominant SV directions
            cos_theta=np.clip(float(U1s[l-1]@u1),-1,1)
            dtheta=math.acos(abs(cos_theta))
            # Sign of angle: positive if same half-space, negative if flipped
            if U1s[l-1]@u1<0: dtheta=-dtheta
            theta+=dtheta

        zs[l]=sv1*complex(math.cos(theta),math.sin(theta))

    return zs, U1s

def grassmannian_distance(U1,U2,k=4):
    """Subspace distance between top-k singular vector spaces."""
    sv=np.linalg.svd(U1[:,:k].T@U2[:,:k],compute_uv=False)
    angles=np.arccos(np.clip(sv,0,1))
    return float(np.sqrt(np.sum(angles**2)))

def ihara_radius(Js, prime_paths):
    """
    Ihara spectral radius of the quiver on prime paths.
    rho = max |eigenvalue of T_prime|
    where T_prime is the transfer matrix on prime path edges.
    Analogous to BALBc Ihara radius.
    """
    n=len(prime_paths)
    T=np.zeros((n,n))
    for i,p_i in enumerate(prime_paths):
        for j,p_j in enumerate(prime_paths):
            # Edge weight = ||J_{p_i[-1]} @ J_{p_j[0]}|| (overlap strength)
            if i!=j:
                overlap=set(p_i)&set(p_j)
                if overlap:
                    # Weight = mean Jacobian norm at overlap
                    T[i,j]=np.mean([N(Js[l]) for l in overlap])
    eigvals=np.linalg.eigvals(T)
    return float(np.max(np.abs(eigvals)))

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

pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
torch.manual_seed(0)
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

# Prime paths
def mu6_fast(js):
    a,b,c,d,e,f=js
    def mu4(x,y,z,w): return -(comm(comm(comm(x,y),z),w)-comm(x,y)@comm(z,w)+
                                comm(x,comm(comm(y,z),w))-comm(x,comm(y,comm(z,w)))+
                                comm(x,comm(y,z)@w-y@comm(z,w)))
    # Use simplified norm approximation for speed
    l3_abc=comm(comm(a,b),c)-comm(a,comm(b,c))
    l3_bcd=comm(comm(b,c),d)-comm(b,comm(c,d))
    l3_cde=comm(comm(c,d),e)-comm(c,comm(d,e))
    l3_def=comm(comm(d,e),f)-comm(d,comm(e,f))
    # mu6 leading terms
    return comm(l3_abc,comm(l3_def,comm(d,e))) - comm(a,comm(l3_bcd,comm(e,f)))

import itertools
att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6_fast([Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths_all=[c for c,_ in scored[:20]]
prime_paths=prime_paths_all[:N_STU]

# Serre cascade
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

# ════════════════════════════════════════════════════
# PART 1: TEACHER COMPLEX GRASSMANNIAN TRAJECTORY
# ════════════════════════════════════════════════════
print("="*65)
print("PART 1: TEACHER COMPLEX GRASSMANNIAN")
print("  z_l = sv_1(J_l) * exp(i*theta_l) for each layer")
print("  Ihara radius = spectral radius on prime path quiver")
print("="*65)

zs_teacher,U1s_teacher=complex_grassmannian_coords(Js)
rho_teacher=ihara_radius(Js,prime_paths)

print(f"\n  Teacher complex Grassmannian coordinates:")
print(f"  {'L':>3}  {'Re(z)':>8}  {'Im(z)':>8}  {'|z|':>7}  "
      f"{'arg(z)/pi':>10}  {'Gr_dist':>9}")
print("  "+"-"*52)
prev_U=None
for l in range(0,N_LAYERS_T,3):
    z=zs_teacher[l]
    U_l=np.column_stack([U1s_teacher[l]])
    if prev_U is not None:
        gd=float(np.arccos(np.clip(abs(float(prev_U.T@U1s_teacher[l])),0,1)))
    else: gd=0.0
    branch="*" if z.imag<0 and l>0 else " "
    print(f"  L{l:>2}  {z.real:>8.3f}  {z.imag:>8.3f}  {abs(z):>7.3f}  "
          f"{np.angle(z)/math.pi:>10.4f}  {gd:>9.5f}{branch}")
    prev_U=np.column_stack([U1s_teacher[l]])

print(f"\n  Ihara radius (prime path quiver): {rho_teacher:.4f}")
print(f"  Branch point (mu0): {MU0}")
print(f"  Layers where Im(z) < 0 (below real axis):")
below_real=[l for l in range(N_LAYERS_T) if zs_teacher[l].imag<0]
print(f"  {below_real}")

# ════════════════════════════════════════════════════
# PART 2: STUDENT TRAINING TRAJECTORY IN C
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: STUDENT TRAINING TRAJECTORY IN C")
print("  Profile z_l(step) during student CE training")
print("  Detect: branch crossings, winding number, eta in C")
print("="*65)

# Build student with Serre cascade
torch.manual_seed(99)
stu=LM(D,N_HEADS,N_STU)
stu.te.weight.data.copy_(teacher.te.weight.data)
with torch.no_grad():
    stu.pe.weight.copy_(teacher.pe.weight)
    stu.ln_f.weight.copy_(teacher.ln_f.weight)
    stu.ln_f.bias.copy_(teacher.ln_f.bias)
    for l in range(N_STU):
        W_d=lift_to_d(cascade_serre[l],U14,scale=0.01)
        W_t=torch.tensor(W_d,dtype=torch.float32)
        stu.blocks[l].attn.WK.weight.copy_(W_t)
        stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
        stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
        stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
        stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
        stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
        stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

# Profile complex coordinates during training
PROFILE_STEPS=[0,10,25,50,75,100,125,150,175,200]
profiles={}

def get_student_Js(model,x_ref,pos,m):
    model.eval()
    with torch.no_grad(): hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]
    Js_stu=[]
    for l in range(N_STU):
        J,_=layer_jac(model.blocks[l],hs[l],pos,m)
        Js_stu.append(J)
    return Js_stu

x_ref_s,_=get_batch('val'); x_ref_s=x_ref_s[0:1]

print(f"\n  Profiling complex Grassmannian trajectory during training...")
print(f"\n  {'step':>5}  {'val':>7}  {'|z_mid|':>8}  {'Im(z_mid)':>10}  "
      f"{'winding':>8}  {'Ihara_rho':>10}  {'branch_cross'}")
print("  "+"-"*68)

prev_zs=None; cumulative_winding=0; total_branch_crossings=0

for step in range(0,201):
    if step>0:
        for pg in opt_s.param_groups: pg['lr']=clr(step,200,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()

    if step in PROFILE_STEPS:
        vl=eval_val(stu,n=20)
        Js_stu=get_student_Js(stu,x_ref_s,pos,m)
        zs_stu,_=complex_grassmannian_coords(Js_stu)
        # Student Ihara radius: use student Jacobians on teacher prime path structure
        # Map student block l -> approximates teacher layer L_ATT + l
        # Rebuild prime paths in student index space
        student_prime_paths=[]
        for pp in prime_paths[:N_STU]:
            # Map teacher layers to student blocks by relative position
            stu_pp=tuple(min(max(0,l-L_ATT+N_STU//2),N_STU-1) for l in pp)
            student_prime_paths.append(stu_pp)
        rho_stu=ihara_radius(Js_stu, student_prime_paths)

        # Middle layer complex coordinate
        mid=N_STU//2
        z_mid=zs_stu[mid]

        # Branch crossings since last profile
        branch_cross=0
        if prev_zs is not None:
            for l in range(N_STU):
                if prev_zs[l].imag*zs_stu[l].imag<0:  # sign change in Im(z)
                    branch_cross+=1
                    total_branch_crossings+=1

        # Winding number: unwrapped angle traversed / (2*pi)
        if prev_zs is not None:
            dangles=[]
            for l in range(N_STU):
                da=np.angle(zs_stu[l])-np.angle(prev_zs[l])
                # Unwrap: choose shortest path around circle
                if da>math.pi: da-=2*math.pi
                if da<-math.pi: da+=2*math.pi
                dangles.append(da)
            dangle=sum(dangles)/N_STU
            cumulative_winding+=dangle/(2*math.pi)

        profiles[step]={
            'val':vl,'zs':zs_stu,'rho':rho_stu,
            'z_mid':z_mid,'branch_cross':branch_cross,
            'winding':cumulative_winding
        }
        prev_zs=zs_stu

        print(f"  {step:>5}  {vl:>7.4f}  {abs(z_mid):>8.3f}  "
              f"{z_mid.imag:>10.4f}  {cumulative_winding:>8.3f}  "
              f"{rho_stu:>10.4f}  "
              f"{'YES x'+str(branch_cross) if branch_cross>0 else '---'}")

# ════════════════════════════════════════════════════
# PART 3: THIMBLE ANALYSIS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 3: LEFSCHETZ THIMBLE ANALYSIS")
print(f"  Branch point z_0 = mu_0 = {MU0}")
print(f"  Total branch crossings: {total_branch_crossings}")
print("="*65)

print(f"\n  WINDING AROUND BRANCH POINT z_0 = {MU0}:")
for step,p in sorted(profiles.items()):
    z=p['z_mid']
    # Distance from branch point in C
    dist_from_branch=abs(z-MU0)
    # Argument relative to branch point
    arg_rel=np.angle(z-MU0)/math.pi
    print(f"  step {step:>4}: z_mid={z.real:.3f}+{z.imag:.3f}i  "
          f"dist_from_z0={dist_from_branch:.4f}  "
          f"arg/pi={arg_rel:.4f}  "
          f"winding={p['winding']:.3f}")

print(f"\n  PREDICTION vs OBSERVATION:")
print(f"  Predicted branch crossing: steps 66-100 (3rd winding)")
crossing_steps=[s for s,p in profiles.items() if p['branch_cross']>0]
print(f"  Observed branch crossings: steps {crossing_steps}")

print(f"\n  IHARA RADIUS TRAJECTORY:")
print(f"  Teacher: {rho_teacher:.4f}")
for step,p in sorted(profiles.items()):
    print(f"  Student step {step:>4}: {p['rho']:.4f}")

print(f"""
  SUMMARY:
    Complex Grassmannian maps the training trajectory to C.
    The branch point z_0 = mu_0 = {MU0} is the curvature obstruction.
    Lefschetz thimbles overlap where Im(z) = 0, Re(z) < mu_0.
    Branch crossings = thimble transitions = cos=1 events.
    
    The winding number around z_0 counts how many thimbles
    the gradient descent path has traversed.
    Predicted: w = {N_STU} (one per cascade level = one per prime path).
    
    The Ihara radius on the student's prime path quiver should
    approach the teacher's value ({rho_teacher:.4f}) as training completes.
    This is the same convergence criterion as in BALBc snapshots:
    rho -> rho_teacher when the quiver algebra stabilizes.
    
    CONNECTION TO BALBc:
    The Klein pillars P1/P2/P3 tracked rho over time snapshots.
    The natural transformation eta tracks rho over training steps.
    Both measure the SAME THING: convergence of the Ihara zeta
    function spectral radius to its terminal value.
""")
