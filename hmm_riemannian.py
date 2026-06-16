#!/usr/bin/env python3
"""
HMM Trajectory + Riemannian Gradient Descent
=============================================
The transformer IS a HMM:
  Hidden states: h_l in R^D (layer activations)
  Transitions: J_l (Jacobian = linear approx of layer map)
  Observations: tokens x_t (via embedding at l=0, head at l=L)

The HMM forward algorithm gives:
  alpha_l(x) = E_D[h_l | x_{0:l}]  (expected hidden state)

Over the corpus:
  H_bar_l = E_D[alpha_l(x)] = E_D[h_l]  (corpus mean hidden state)
  Cov_l = E_D[(h_l - H_bar_l)(h_l - H_bar_l)^T]  (corpus covariance)

These two quantities determine the Fisher metric at each layer:
  G_l = J_l^T @ Cov_l @ J_l  (Fisher metric contribution from layer l)
  G = sum_l G_l  (total Fisher metric)

The RGD update:
  natural_grad = G^{-1} @ grad_theta(L)
  theta <- theta - lr * natural_grad

The Dehn gap trust region:
  clip ||natural_grad|| <= DEHN_GAP

WHAT THE HMM GIVES THAT BATCH GD CANNOT:
  The corpus mean H_bar_l and covariance Cov_l are computed ONCE
  over the full corpus in a single forward pass.
  This is the corpus summary tensor computed correctly —
  not just the mean Jacobian E_D[J_l] but the full covariance
  structure E_D[h_l h_l^T] that determines the Fisher metric.

  The natural gradient preconditioned by the corpus covariance
  is the optimal first-order method for this problem.
  It is equivalent to one Newton step in the Fisher metric,
  which converges faster than gradient descent when the
  Hessian is well-approximated by the Fisher matrix.

PREDICTION:
  The HMM preconditioning should reduce Phase 2 from 200 steps
  to ~100-125 steps by providing a better metric for the gradient.
  The Dehn gap trust region prevents the optimizer from crossing
  the branch cut (the source of the step-75-100 acceleration bump).
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; DEHN_GAP=1.4; N_HMM_SEQS=100

print(f"\n{'='*65}")
print(f"  HMM TRAJECTORY + RIEMANNIAN GD")
print(f"  Fisher metric from corpus covariance")
print(f"  Dehn gap trust region = {DEHN_GAP}")
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

def clr(s,total=200,warmup=50):
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
# Train teacher
# ════════════════════════════════════════════════════
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

# Cascades
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c,_ in scored[:N_STU]]

# ════════════════════════════════════════════════════
# STEP 1: HMM FORWARD PASS — CORPUS TRAJECTORY
# ════════════════════════════════════════════════════
print("="*65)
print(f"STEP 1: HMM CORPUS TRAJECTORY ({N_HMM_SEQS} sequences)")
print("  Compute E_D[h_l] and Cov_D[h_l] for each layer")
print("  These define the Fisher metric for Riemannian GD")
print("="*65)

# Collect hidden states across corpus
h_sum=[np.zeros(D) for _ in range(N_LAYERS_T+1)]
h_sq=[np.zeros((D,D)) for _ in range(N_LAYERS_T+1)]
n_seqs=0

torch.manual_seed(0)
print(f"\n  Running forward pass over {N_HMM_SEQS} corpus sequences...")
for seq_i in range(N_HMM_SEQS):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    x_seq=train_t[ix:ix+SEQ].unsqueeze(0)
    teacher.eval()
    with torch.no_grad():
        hs=teacher.hidden_states(x_seq)
        for l,h in enumerate(hs):
            hv=h[0,pos,:].numpy()  # hidden state at position pos
            h_sum[l]+=hv
            h_sq[l]+=np.outer(hv,hv)
    n_seqs+=1
    if (seq_i+1)%25==0: print(f"  {seq_i+1}/{N_HMM_SEQS}...",flush=True)

# Compute mean and covariance
H_mean=[h_sum[l]/n_seqs for l in range(N_LAYERS_T+1)]
H_cov=[h_sq[l]/n_seqs - np.outer(H_mean[l],H_mean[l])
       for l in range(N_LAYERS_T+1)]

print(f"\n  HMM trajectory statistics:")
print(f"  {'Layer':>6}  {'||H_mean||':>12}  {'tr(Cov)':>10}  {'cond(Cov)':>11}")
print("  "+"-"*44)
for l in range(0,N_LAYERS_T+1,4):
    sv=np.linalg.svd(H_cov[l],compute_uv=False)
    cond=sv[0]/max(sv[-1],1e-10)
    print(f"  L{l:>2}     {np.linalg.norm(H_mean[l]):>12.4f}  "
          f"{np.trace(H_cov[l]):>10.4f}  {cond:>11.2f}")

# Fisher metric at attractor layer (most important for RGD)
# G_att = J_att^T @ Cov_att @ J_att  (pulled back to parameter space)
# For the student blocks, approximate using H_cov at L_ATT
G_att=H_cov[L_ATT]  # (D, D) corpus covariance at attractor layer
print(f"\n  Attractor covariance tr(G_att) = {np.trace(G_att):.4f}")
print(f"  Attractor sv[:4]: {np.linalg.svd(G_att,compute_uv=False)[:4].round(4)}")

# ════════════════════════════════════════════════════
# STEP 2: FISHER-PRECONDITIONED OPTIMIZER
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 2: FISHER-PRECONDITIONED OPTIMIZER")
print("  Use corpus covariance as preconditioner for gradient")
print("="*65)

# Project gradient onto inverse Fisher metric
# For attention WK: grad_WK -> (G_att + eps I)^{-1} @ grad_WK
# This is the natural gradient in the hidden state metric

# Compute (G_att + eps I)^{-1} once
eps_fisher=0.01
G_reg=G_att[:D,:D]+eps_fisher*np.eye(D)
G_inv=np.linalg.inv(G_reg)  # (D, D)
G_inv_t=torch.tensor(G_inv.astype(np.float32))
print(f"  Fisher preconditioner computed: {G_inv.shape}")
print(f"  Condition number: {np.linalg.cond(G_reg):.2f}")

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

def run_standard(cascade, label, steps=200):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

def run_hmm_riemannian(cascade, label, steps=200, use_dehn=False):
    """
    HMM-preconditioned Riemannian GD.
    
    For WK and WQ weights (key/query matrices):
      natural_grad = G_inv @ grad  (Fisher preconditioned)
    For all other weights:
      standard Adam update
    
    The WK/WQ matrices are the ones that interact with the
    monodromy via the attention score J_l = d(Attn)/dh.
    They live directly on the MC moduli space.
    Other weights (WV, WO, FF) are auxiliary.
    """
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")

    # Separate key/query parameters from others
    kq_params=[]; other_params=[]
    for name,p in stu.named_parameters():
        if 'WK' in name or 'WQ' in name:
            kq_params.append(p)
        else:
            other_params.append(p)

    opt_other=torch.optim.AdamW(other_params,lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    # For KQ: use simple SGD, we apply Fisher preconditioner manually
    opt_kq=torch.optim.SGD(kq_params,lr=LR,momentum=0.9,weight_decay=0.1)

    ck={}
    for step in range(1,steps+1):
        lr_now=clr(step,steps)
        for pg in opt_other.param_groups: pg['lr']=lr_now
        for pg in opt_kq.param_groups: pg['lr']=lr_now

        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_other.zero_grad(); opt_kq.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0)

        # Apply Fisher preconditioner to WK, WQ gradients
        with torch.no_grad():
            for name,p in stu.named_parameters():
                if p.grad is None: continue
                if 'WK' in name or 'WQ' in name:
                    # WK shape: (D, D)
                    # Natural gradient: G_inv @ grad (left-multiply)
                    g=p.grad  # (D, D)
                    nat_g=G_inv_t@g  # Fisher-preconditioned gradient
                    # Dehn gap trust region: clip norm
                    if use_dehn:
                        ng_norm=nat_g.norm()
                        if ng_norm>DEHN_GAP:
                            nat_g=nat_g*(DEHN_GAP/ng_norm)
                    p.grad.copy_(nat_g)

        opt_other.step(); opt_kq.step()

        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

# ════════════════════════════════════════════════════
# STEP 3: EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 3: STUDENT EXPERIMENTS")
print("  A: Serre + Adam (baseline)")
print("  B: Prime + Adam (standard best)")
print("  C: Prime + HMM-Fisher Riemannian GD")
print("  D: Prime + HMM-Fisher RGD + Dehn trust region")
print("="*65)

vA,ckA=run_standard(cascade_serre,"A-Serre-Adam")
vB,ckB=run_standard(cascade_prime,"B-Prime-Adam")
vC,ckC=run_hmm_riemannian(cascade_prime,"C-Prime-HMM-RGD",use_dehn=False)
vD,ckD=run_hmm_riemannian(cascade_prime,"D-Prime-HMM-RGD+Dehn",use_dehn=True)

print(f"\n{'='*65}")
print("  HMM RIEMANNIAN GD RESULTS")
print("="*65)
print(f"""
  HMM TRAJECTORY:
    Corpus sequences: {N_HMM_SEQS}
    Attractor covariance tr(G) = {np.trace(G_att):.4f}
    Fisher preconditioner condition = {np.linalg.cond(G_reg):.2f}

  CONVERGENCE:
  {'step':>6}  {'A-Serre':>8}  {'B-Prime':>8}  {'C-HMM-RGD':>10}  {'D-HMM+Dehn':>11}""")
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
    Teacher:              val={val_teacher:.4f}
    A (Serre+Adam):       val={vA:.4f}
    B (Prime+Adam):       val={vB:.4f}  diff={vA-vB:+.4f}
    C (HMM-RGD):          val={vC:.4f}  diff={vA-vC:+.4f}
    D (HMM-RGD+Dehn):     val={vD:.4f}  diff={vA-vD:+.4f}

  HMM INTERPRETATION:
    The corpus covariance Cov_D[h_l] at the attractor layer
    is the Fisher metric on the hidden state space.
    The natural gradient G^{{-1}} @ grad removes the curvature
    of the MC moduli space from the gradient direction.

    IF C < A at steps 25-75:
      The HMM trajectory provides a better metric than Adam's
      diagonal approximation. The corpus covariance captures
      the true geometry of the MC moduli space.
      Phase 2 cost is reduced by better conditioning.

    IF C ≈ A:
      Adam's diagonal Fisher (v_hat^{{1/2}}) already approximates
      the full Fisher metric sufficiently. The HMM preconditioning
      adds no new geometric information beyond what Adam computes
      adaptively from the gradients.
      CONCLUSION: Adam IS the correct Riemannian optimizer for
      this problem. The 200 CE steps are the irreducible geodesic
      length on the MC moduli space regardless of metric.
""")
