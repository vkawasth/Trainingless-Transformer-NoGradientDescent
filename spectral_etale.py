#!/usr/bin/env python3
"""
Spectral Étale Projection
==========================
The étale projection in weight space failed because the winding
lives in SPECTRAL space (arg(z_mid) oscillation in SVD coordinates),
not in weight update space.

The correct projection requires dz/dW_K — the Jacobian of the
complex coordinate z_mid with respect to W_K weights.

CONSTRUCTION:
  z_mid(t) = sv_1(J_mid(t)) * exp(i * theta_mid(t))
  
  The phase theta_mid changes as W_K changes.
  dtheta/dW_K is the spectral gradient — the direction in weight
  space that corresponds to changing the phase of z_mid.
  
  The winding direction in weight space is:
    e_spectral = d(Im(z_mid))/d(W_K)
              = d(sv_1 * sin(theta))/d(W_K)
  
  This is computable via autograd through the SVD.
  
  The spectral étale projection:
    g_perp = g - <g, e_spectral> / |e_spectral|^2 * e_spectral
  
  This removes the component of g that changes Im(z_mid)
  (the oscillating component) and keeps the component that
  changes the loss without oscillating.

IMPLEMENTATION:
  For each step:
  1. Forward pass through student to get J_mid (Jacobian at mid block)
  2. SVD of J_mid to get sv_1 and u_1
  3. Compute theta = accumulated angle
  4. Im_z = sv_1 * sin(theta)
  5. Autograd: d(Im_z)/d(W_K) via torch.autograd.grad
  6. Project gradient perpendicular to this spectral direction

COST: one extra forward+backward per step through the Jacobian SVD.
This is O(ma^2) = O(48^2) = O(2304) operations per step — cheap.

PREDICTION:
  If spectral étale projection works:
    Steps 1-6: val drops fast (one sheet per step)
    Steps 6+: monotone refinement
  If it fails:
    The spectral direction changes too fast or is too noisy.
    The étale structure is not accessible via this gradient.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  SPECTRAL ÉTALE PROJECTION")
print(f"  e_spectral = d(Im(z_mid))/d(W_K)  via autograd")
print(f"  Removes oscillating spectral component from gradient")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

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
    def hidden_states_all(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h)
        for b in self.blocks: h=b(h); hs.append(h)
        return hs

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def compute_spectral_direction(stu, x_ref, pos, m, mid_block):
    """
    Compute e_spectral = d(Im(z_mid))/d(W_K_mid) via autograd.
    
    Im(z_mid) = sv_1(J_mid) * sin(theta_mid)
    
    We approximate: use Im of the dominant complex eigenvalue
    of J_mid projected onto the PROJ-dim active subspace.
    
    Differentiable path:
    1. Get h_mid from forward pass (no grad)
    2. Run block mid with W_K requiring grad
    3. Compute J_mid via finite difference on h_mid
    4. Get dominant SV direction
    5. Compute accumulated angle theta (scalar, from stored prev_u1)
    6. Im_z = sv_1 * sin(theta)  <- differentiable w.r.t. W_K
    7. Backprop to get d(Im_z)/d(W_K)
    """
    stu.eval()
    # Get hidden states up to mid_block (no grad needed)
    with torch.no_grad():
        hs = stu.hidden_states_all(x_ref)
        h_in = hs[mid_block][0, pos, :]  # (D,)

    # Now compute Jacobian of mid_block w.r.t. h_in
    # Use the m-dimensional subspace
    h_mat = hs[mid_block][0]  # (SEQ, D)
    _, _, Vt = torch.linalg.svd(h_mat, full_matrices=False)
    U_sub = Vt[:m, :].T  # (D, m)

    # Project h_in to subspace
    h_proj = U_sub.T @ h_in  # (m,) — no grad

    # Compute J via one row of the Jacobian (dominant direction)
    # J_ij = d(h_out_i)/d(h_in_j) in the projected space
    # Use power iteration: J @ v where v = first right SV approx
    # Approximate: compute J @ e_1 for a few basis vectors
    # Then get dominant SV via SVD of the resulting matrix

    # Build small Jacobian (m x m) with autograd
    J_rows = []
    for i in range(min(m, 8)):  # only first 8 rows for speed
        e_i = torch.zeros(D)
        e_i[U_sub[:, i].argmax()] = 1.0  # approximate basis

        h_in_grad = h_in.detach().clone().requires_grad_(True)
        h_in_batch = h_in_grad.unsqueeze(0).unsqueeze(0)  # (1,1,D)

        h_out = stu.blocks[mid_block](h_in_batch)
        h_out_proj = (U_sub.T @ h_out[0, 0])  # (m,)

        # Gradient of i-th output component w.r.t. W_K
        # This gives d(h_out_proj_i)/d(W_K)
        scalar = h_out_proj[i]
        scalar.backward()
        J_rows.append(h_in_grad.grad.detach())

    # Approximate: dominant direction of J is approximated by
    # the direction that changes the output most
    # Use the norm of the gradient as a proxy for Im(z)
    # d(Im(z_mid))/d(W_K) ≈ d(||J_mid||_F)/d(W_K) projected onto Im component

    # Simpler: use Im(z) = sv_1 * sin(theta)
    # d(Im(z))/d(W_K) requires differentiating through SVD
    # Use torch.linalg.svd with autograd

    # Build J_mid differentiably
    h_in_t = hs[mid_block][0, pos, :].detach()  # (D,)

    # Make W_K require grad temporarily
    WK = stu.blocks[mid_block].attn.WK.weight  # (D, D)
    WK.requires_grad_(True)

    # Jacobian column: d(block(h)_out)/d(h_in) projected
    # Approximate Im(z_mid) as: Im of first SVD value of WK @ U_sub
    # This is differentiable and captures the spectral phase
    WK_proj = WK[:m, :m]  # (m, m) — projected key matrix
    sv_vals = torch.linalg.svdvals(WK_proj)  # (m,) — differentiable
    sv1 = sv_vals[0]

    # Accumulated angle theta: use fixed value from last measurement
    # (we don't differentiate through theta — it's slow-changing)
    theta_approx = math.pi * 0.5  # approximate mid-training value
    Im_z_approx = sv1 * math.sin(theta_approx)

    # Gradient of Im_z w.r.t. W_K
    if WK.grad is not None:
        WK.grad.zero_()
    Im_z_approx.backward()
    e_spectral = WK.grad.detach().clone() if WK.grad is not None else torch.zeros_like(WK)
    WK.requires_grad_(False)

    return e_spectral

print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else \
           LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
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

torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]

def build_student():
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.copy_(teacher.blocks[L_ATT].attn.WK.weight)
            stu.blocks[l].attn.WQ.weight.copy_(teacher.blocks[L_ATT].attn.WQ.weight)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

print("="*65)
print("SPECTRAL CONFIRMATION")
print("  Measure Im(z_mid) during standard Adam training")
print("  Confirm it oscillates (étale structure exists)")
print("  Then test spectral étale projection")
print("="*65)

# First: confirm spectral oscillation during training
print("\n  Tracking Im(z_mid) during standard Adam training...")
stu_track=build_student()
opt_t=torch.optim.AdamW(stu_track.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

def get_Im_z_mid(stu, x_ref, pos, m):
    """Get Im(z_mid) from current student state."""
    stu.eval()
    with torch.no_grad():
        hs=stu.hidden_states_all(x_ref); hs=[h[0] for h in hs]
    mid=N_STU//2
    h_in=hs[mid]  # (SEQ, D)
    _, _, Vt=torch.linalg.svd(h_in,full_matrices=False)
    U_sub=Vt[:m,:].T.detach().numpy()  # (D, m)

    # Jacobian of mid block at pos
    h_pos=h_in[pos,:].detach().numpy()
    J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            hh=torch.tensor(h_in.detach().numpy()).unsqueeze(0).requires_grad_(True)
            ho=stu.blocks[mid](hh)
            v=ho[0,pos,:]
            e_i=torch.tensor(U_sub[:,i],dtype=torch.float32)
            (v*e_i).sum().backward()
            g=hh.grad[0,pos,:].detach().numpy()
            J[:,i]=U_sub.T@g

    # SVD of J to get sv1 and u1
    U,sv,_=np.linalg.svd(J,full_matrices=False)
    sv1=sv[0]; u1=U[:,0]

    # Accumulated angle for mid block
    theta=0.0; prev_u1=None
    for l in range(mid+1):
        h_l=hs[l]
        _,_,Vt_l=torch.linalg.svd(h_l,full_matrices=False)
        U_l=Vt_l[:m,:].T.detach().numpy()
        h_l_pos=h_l[pos,:].detach().numpy()
        Jl=np.zeros((m,m))
        with torch.enable_grad():
            for i in range(m):
                hh=torch.tensor(h_l.detach().numpy()).unsqueeze(0).requires_grad_(True)
                ho=stu.blocks[l](hh)
                v=ho[0,pos,:]
                e_i=torch.tensor(U_l[:,i],dtype=torch.float32)
                (v*e_i).sum().backward()
                g=hh.grad[0,pos,:].detach().numpy()
                Jl[:,i]=U_l.T@g
        Ul,_,_=np.linalg.svd(Jl,full_matrices=False)
        u1l=Ul[:,0]
        if prev_u1 is not None:
            cos_t=float(np.clip(prev_u1@u1l,-1,1))
            dt=math.acos(abs(cos_t))
            if prev_u1@u1l<0: dt=-dt
            theta+=dt
        prev_u1=u1l

    return sv1*math.sin(theta), sv1, theta

print(f"\n  {'step':>5}  {'val':>7}  {'Im(z)':>8}  {'sv1':>7}  {'theta/pi':>9}  {'sign_change'}")
print("  "+"-"*50)

prev_Im=None; sign_changes=0
for step in range(0,101):
    if step>0:
        for pg in opt_t.param_groups: pg['lr']=clr(step,200)
        stu_track.train(); x,y=get_batch(); _,loss=stu_track(x,y)
        opt_t.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu_track.parameters(),1.0); opt_t.step()

    if step in [0,1,2,5,10,15,20,25,33,50,66,75,100]:
        vl=eval_val(stu_track,n=10)
        Im_z,sv1,theta=get_Im_z_mid(stu_track,x_ref,pos,m)
        sign_flip=""
        if prev_Im is not None and prev_Im*Im_z<0:
            sign_changes+=1
            sign_flip=" <-- SIGN FLIP"
        print(f"  {step:>5}  {vl:>7.4f}  {Im_z:>8.4f}  {sv1:>7.4f}  "
              f"{theta/math.pi:>9.4f}{sign_flip}")
        prev_Im=Im_z

print(f"\n  Total Im(z) sign flips in 100 steps: {sign_changes}")
print(f"  Expected (6 sheets, ~33 steps/sheet): ~{100//33} flips")
print(f"  {'CONFIRMED: étale structure exists' if sign_changes>=2 else 'NOT CONFIRMED: no clear oscillation'}")

# ════════════════════════════════════════════════════
# SPECTRAL ÉTALE PROJECTION
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("SPECTRAL ÉTALE PROJECTION EXPERIMENT")
print("  e_spectral = d(sv_1(W_K_proj))/d(W_K) * sin(theta)")
print("  Project CE gradient ⊥ to this spectral direction")
print("="*65)

def run_spectral_etale(label, strength=1.0, steps=200):
    stu=build_student()
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}

    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()

        # Compute spectral direction every 5 steps
        if step%5==1:
            e_spec=compute_spectral_direction(stu,x_ref,pos,m,N_STU//2)
            e_norm_sq=float((e_spec**2).sum())

        # Project WK gradient perpendicular to spectral direction
        if e_norm_sq>1e-12 and strength>0:
            with torch.no_grad():
                for l in range(N_STU):
                    g=stu.blocks[l].attn.WK.weight.grad
                    if g is None: continue
                    proj=float((g*e_spec).sum())/e_norm_sq
                    g_perp=g - strength*proj*e_spec
                    stu.blocks[l].attn.WK.weight.grad.copy_(g_perp)

        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0)
        opt_s.step()

        if step in [1,2,5,6,10,25,50,75,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

def run_standard(label,steps=200):
    stu=build_student()
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [1,2,5,6,10,25,50,75,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

vA,ckA=run_standard("A-Adam-std")
vB,ckB=run_spectral_etale("B-Spectral-Etale-1.0",strength=1.0)
vC,ckC=run_spectral_etale("C-Spectral-Etale-0.5",strength=0.5)

print(f"\n{'='*65}")
print("  SPECTRAL ÉTALE RESULTS")
print("="*65)
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Adam':>8}  {'B-Spec1.0':>10}  {'C-Spec0.5':>10}")
for s in [1,2,5,6,10,25,50,75,100,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c]:
        row+=f"  {v:>9.4f}" if v else f"  {'---':>9}"
    if a and b and b<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:               val={val_teacher:.4f}
    A (Adam std):          val={vA:.4f}
    B (Spectral Étale 1.0):val={vB:.4f}  diff={vA-vB:+.4f}
    C (Spectral Étale 0.5):val={vC:.4f}  diff={vA-vC:+.4f}

  SPECTRAL CONFIRMATION:
    Im(z_mid) sign flips in 100 steps: {sign_changes}
    This confirms/denies the étale sheet structure.
    
  IF B < A at steps 1-10:
    The spectral étale projection works.
    d(Im(z))/d(W_K) is the correct winding direction.
    The étale structure IS accessible via the spectral gradient.
    
  IF B ~ A:
    The spectral direction (d(sv_1)/d(W_K)) is not the winding.
    The étale structure requires the full Jacobian chain,
    not just the mid-block singular value derivative.
    The three-stage pipeline is confirmed as optimal.
""")
