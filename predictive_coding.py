#!/usr/bin/env python3
"""
Predictive Coding Transformer — Local Error Signals
=====================================================
The brain uses predictive coding, not Hebbian correlation.

PREDICTIVE CODING RULE:
  Delta_W_l = eta * epsilon_{l+1} * h_l^T
  where epsilon_{l+1} = h_{l+1}^teacher - h_{l+1}^student
  
  This is local: each layer only needs
    - its own input h_l (bottom-up)
    - the prediction error from above epsilon_{l+1} (top-down)
  
  No backpropagation through the full model.
  No global loss function.
  Each layer trains independently.

WHY THIS IS DIFFERENT FROM HEBBIAN:
  Hebbian: Delta_W = eta * E[h_{l+1} * h_l^T]  (correlation)
  Pred.cod: Delta_W = eta * epsilon_{l+1} * h_l^T  (error * input)
  
  The error epsilon = h^teacher - h^student is the KEY signal.
  Without the teacher's target, you cannot compute epsilon.
  
  The brain's "teacher" is the sensory world providing
  top-down predictions from higher cortical areas.
  In our case: the trained teacher model provides h^teacher.

EQUIVALENCE TO DISTILLATION:
  This is layer-wise knowledge distillation with MSE loss:
    L_l = ||h_{l+1}^student - h_{l+1}^teacher||^2
  But computed LOCALLY at each layer, not globally.
  
  The gradient of L_l w.r.t. W_K^l is:
    dL_l/dW_K^l = 2 * (h_{l+1}^student - h_{l+1}^teacher) * dh_{l+1}/dW_K^l
  
  In the linear approximation: dh_{l+1}/dW_K^l ≈ h_l^T
  So: dL_l/dW_K^l ≈ 2 * epsilon_{l+1} * h_l^T
  
  This IS the predictive coding update.
  It is local backpropagation (one layer at a time).

PREDICTION:
  Layer-wise distillation with teacher hidden states as targets
  should converge in ~5-10 steps per layer.
  Total convergence: N_STU * 10 = 60 steps (much less than 200).
  
  This would confirm: the brain's superiority is from having
  layer-local error signals (predictive coding) rather than
  global loss gradients (CE backpropagation).
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; PC_LR=1e-3

print(f"\n{'='*65}")
print(f"  PREDICTIVE CODING TRANSFORMER")
print(f"  Local error signals: epsilon_l = h_teacher - h_student")
print(f"  Brain's mechanism: top-down prediction error")
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
def mu6_op(js):
    a,b,c,d,e,f=js
    def mu4(x,y,z,w):
        return -(comm(l3(x,y,z),w)-l3(comm(x,y),z,w)+l3(x,comm(y,z),w)
                -l3(x,y,comm(z,w))+comm(x,l3(y,z,w)))
    def mu5(x,y,z,w,v):
        return -(l3(l3(x,y,z),w,v)-l3(x,l3(y,z,w),v)+l3(x,y,l3(z,w,v))
                +comm(mu4(x,y,z,w),v)+comm(x,mu4(y,z,w,v))
                -mu4(comm(x,y),z,w,v)+mu4(x,y,z,comm(w,v)))
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
    with torch.no_grad(): hs=teacher.hidden_states_all(x_ref); hs_np=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs_np[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6_op([Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
cascade_prime=[mu6_op([Js[i] for i in c])/max(N(mu6_op([Js[i] for i in c])),1e-8)
               for c,_ in scored[:N_STU]]
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

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

# ════════════════════════════════════════════════════
# PREDICTIVE CODING UPDATE
# ════════════════════════════════════════════════════
def pc_step(stu, teacher, n_batches=10, eta=PC_LR):
    """
    Predictive coding: Delta_W_l = eta * epsilon_{l+1} * h_l^T
    epsilon_{l+1} = h_{l+1}^teacher - h_{l+1}^student
    
    Maps teacher layer L_ATT+l to student block l.
    Each student block learns to reproduce the teacher's
    hidden state at the corresponding teacher layer.
    
    This is local: no backprop through full model.
    Each block trains independently.
    """
    stu.train(); teacher.eval()

    # Accumulate error*input signals
    pc_signals=[torch.zeros(D,D) for _ in range(N_STU)]
    n_samples=0

    for _ in range(n_batches):
        x,_=get_batch('train')

        # Teacher hidden states at attractor-mapped layers
        with torch.no_grad():
            hs_teacher=teacher.hidden_states_all(x)
            # Map: student block l -> teacher layer L_ATT + l
            # (student's 6 blocks span teacher's L14 to L20)
            h_teacher_targets=[hs_teacher[min(L_ATT+l+1,N_LAYERS_T)]
                                for l in range(N_STU)]

        # Student hidden states
        with torch.no_grad():
            hs_stu=stu.hidden_states_all(x)

        for l in range(N_STU):
            h_l   =hs_stu[l].reshape(-1,D)        # (B*S, D) student input
            h_l1_s=hs_stu[l+1].reshape(-1,D)      # (B*S, D) student output
            h_l1_t=h_teacher_targets[l].reshape(-1,D)  # (B*S, D) teacher target

            # Prediction error (top-down signal)
            epsilon=h_l1_t-h_l1_s  # (B*S, D)

            # PC update: epsilon^T @ h_l  (D x D)
            pc_signals[l]+=epsilon.T@h_l

        n_samples+=x.shape[0]*x.shape[1]

    # Apply updates to WK (and WQ symmetrically)
    with torch.no_grad():
        for l in range(N_STU):
            sig=pc_signals[l]/n_samples  # (D, D)
            stu.blocks[l].attn.WK.weight.add_(eta*sig)
            stu.blocks[l].attn.WQ.weight.add_(eta*sig.T)

def pc_layer_mse(stu, teacher, n_batches=5):
    """Measure layer-wise MSE between student and teacher hidden states."""
    stu.eval(); teacher.eval(); mse=[0.0]*N_STU
    with torch.no_grad():
        for _ in range(n_batches):
            x,_=get_batch('val')
            hs_t=teacher.hidden_states_all(x)
            hs_s=stu.hidden_states_all(x)
            for l in range(N_STU):
                h_t=hs_t[min(L_ATT+l+1,N_LAYERS_T)]
                h_s=hs_s[l+1]
                mse[l]+=float(F.mse_loss(h_s,h_t))
    return [m/n_batches for m in mse]

# ════════════════════════════════════════════════════
# EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("EXPERIMENTS")
print("  A: Serre + 200CE (baseline)")
print("  B: Prime + 200CE (confirmed best)")
print("  C: Prime + PC only (no CE gradient, local error only)")
print("  D: Prime + PC init + 200CE (PC then CE)")
print("  E: Prime + PC init + 50CE (minimal CE)")
print("  F: Prime + PC+CE interleaved (alternate PC and CE steps)")
print("="*65)

def run_ce(cascade,label,steps=200):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

def run_pc(cascade,label,n_epochs=20,ce_steps=0,interleave=False):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20)
    mse0=pc_layer_mse(stu,teacher)
    print(f"\n  [{label}] zero-shot={v0:.4f}  "
          f"layer MSE: [{', '.join(f'{m:.3f}' for m in mse0)}]")
    ck={0:v0}

    for epoch in range(1,n_epochs+1):
        pc_step(stu,teacher,n_batches=10,eta=PC_LR)
        if epoch in [1,2,5,10,20]:
            v=eval_val(stu,n=20)
            mse=pc_layer_mse(stu,teacher)
            ck[epoch]=v
            print(f"  [{label}] PC epoch {epoch:>3}  val={v:.4f}  "
                  f"MSE_L0={mse[0]:.3f}  MSE_L5={mse[-1]:.3f}"
                  f"{' ✓' if v<val_teacher else ''}")

    v_pc=eval_val(stu,n=20)
    print(f"  [{label}] After {n_epochs} PC epochs: val={v_pc:.4f}")

    if ce_steps>0:
        opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
        for step in range(1,ce_steps+1):
            for pg in opt_s.param_groups: pg['lr']=clr(step,ce_steps)
            if interleave and step%5==0:
                pc_step(stu,teacher,n_batches=2,eta=PC_LR*0.1)
            stu.train(); x,y=get_batch(); _,loss=stu(x,y)
            opt_s.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
            if step in [25,50,100,150,200]:
                v=eval_val(stu,n=20); ck[f'ce{step}']=v
                print(f"  [{label}] CE step {step:>4}  val={v:.4f}"
                      f"{' ✓' if v<val_teacher else ''}")

    return eval_val(stu),ck

vA,ckA=run_ce(cascade_serre,"A-Serre-CE200")
vB,ckB=run_ce(cascade_prime,"B-Prime-CE200")
vC,ckC=run_pc(cascade_prime,"C-Prime-PC-only",n_epochs=20,ce_steps=0)
vD,ckD=run_pc(cascade_prime,"D-Prime-PC+CE200",n_epochs=10,ce_steps=200)
vE,ckE=run_pc(cascade_prime,"E-Prime-PC+CE50",n_epochs=10,ce_steps=50)
vF,ckF=run_pc(cascade_prime,"F-Prime-interleaved",n_epochs=5,ce_steps=200,interleave=True)

print(f"\n{'='*65}")
print("  PREDICTIVE CODING RESULTS")
print("="*65)
print(f"""
  FINAL:
    Teacher:                    val={val_teacher:.4f}
    A (Serre+CE200):            val={vA:.4f}
    B (Prime+CE200):            val={vB:.4f}
    C (Prime+PC x20):           val={vC:.4f}
    D (Prime+PC10+CE200):       val={vD:.4f}  diff vs B={vB-vD:+.4f}
    E (Prime+PC10+CE50):        val={vE:.4f}  diff vs E_no_PC={vB-vE:+.4f}
    F (Prime+PC+CE interleaved):val={vF:.4f}  diff vs B={vB-vF:+.4f}

  THE BRAIN QUESTION ANSWERED:
    IF C << 3.5 (val improves with PC alone):
      Predictive coding IS the brain's algorithm.
      Local error signals replace global backpropagation.
      The 200 CE steps are replaced by ~20 PC epochs.

    IF D < B or E < B (PC+CE converges faster):
      PC pre-conditions the student for CE.
      The brain uses predictive coding for structure,
      then a global signal for fine-tuning.
      This is consistent with cortical hierarchy + neuromodulation.

    IF C ~ 3.5 and D ~ B:
      Neither Hebbian nor predictive coding is the brain's mechanism.
      The brain computes something else entirely.
      Possibly: spike-timing dependent plasticity (STDP),
      contrastive Hebbian learning, or Boltzmann machine dynamics.

  LAYER MSE TELLS THE STORY:
    If layer MSE decreases monotonically with PC epochs,
    predictive coding IS aligning student to teacher layer by layer.
    If layer MSE stays flat, the error signal is not propagating.
""")
