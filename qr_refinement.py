#!/usr/bin/env python3
"""
QR Loop Spectral Refinement
============================
The 2-layer student computes E_1,E_2 pages of the spectral sequence.
Each QR iteration on the monodromy matrix M adds one more page.

ALGORITHM:
  1. Train 2-layer student (val=0.863)
  2. Extract student monodromy M_stu = J_2 @ J_1
  3. For N in {0,2,4,8,12,22}:
     - Apply N QR steps: M → RQ → RQ → ...
     - Apply M_refined to student hidden states
     - Evaluate val loss
  4. Compare to teacher (val=0.250) and student baseline (val=0.863)

PREDICTION:
  If QR steps = spectral pages: val decreases with N, approaching 0.25
  If QR steps are irrelevant: val stays at 0.863
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; PROJ=48

print(f"\n{'='*65}")
print(f"  QR LOOP SPECTRAL REFINEMENT")
print(f"  Each QR step = one spectral sequence page")
print(f"  Can 22 QR steps replace 22 transformer layers?")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
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
        self._nl=nl
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs
    def forward_with_correction(self,x,y,M_corr,U_basis):
        """Forward pass, apply M_corr to final hidden state, then head."""
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        # Apply QR-refined monodromy correction in projected subspace
        h_np=h.detach().numpy()
        B_,S_,D_=h_np.shape
        h_flat=h_np.reshape(-1,D_)
        h_proj=h_flat@U_basis          # [B*S, m]
        h_refined=h_proj@M_corr        # [B*S, m] — apply refined transport
        h_lifted=h_refined@U_basis.T   # [B*S, D]
        # Residual: keep the non-projected component
        h_orth=h_flat - h_flat@U_basis@U_basis.T  # component outside subspace
        h_out=(h_lifted+h_orth).reshape(B_,S_,D_)
        h_out=torch.tensor(h_out,dtype=torch.float32)
        logits=self.head(self.ln_f(h_out))
        loss=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None
        return logits,loss

def clr(s,total,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

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

def eval_with_correction(model,M_corr,U_basis,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val')
            _,loss=model.forward_with_correction(x,y,M_corr,U_basis)
            ls.append(loss.item())
    return float(np.mean(ls))

def qr_refine(M,steps):
    """Apply N QR steps: M → RQ → RQ → ..."""
    M_ref=M.copy()
    for _ in range(steps):
        Q,R=np.linalg.qr(M_ref)
        M_ref=R@Q
    return M_ref

# ── Train 24-layer teacher ────────────────────────────────────────────────────
print("Step 1: Train 24-layer teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step,300)
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
with torch.no_grad():
    val_teacher=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(60)]))
print(f"  Teacher val={val_teacher:.4f}\n")

# ── Train 2-layer student with teacher embeddings ─────────────────────────────
print("Step 2: Train 2-layer student (200 steps, teacher embeddings)...")
torch.manual_seed(99)
student=LM(D,N_HEADS,2)
student.te.weight.data.copy_(teacher.te.weight.data)
student.pe.weight.data.copy_(teacher.pe.weight.data)
student.ln_f.weight.data.copy_(teacher.ln_f.weight.data)
student.ln_f.bias.data.copy_(teacher.ln_f.bias.data)
opt_s=torch.optim.AdamW(student.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,201):
    for pg in opt_s.param_groups: pg['lr']=clr(step,200)
    student.train(); x,y=get_batch(); _,loss=student(x,y)
    opt_s.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); opt_s.step()
    if step%50==0:
        vl=float(np.mean([student(*get_batch('val'))[1].item() for _ in range(20)]))
        print(f"  step {step}  val={vl:.4f}")
student.eval()
with torch.no_grad():
    val_student=float(np.mean([student(*get_batch('val'))[1].item() for _ in range(60)]))
print(f"  Student val={val_student:.4f}\n")

# ── Extract student monodromy M_stu ──────────────────────────────────────────
print("Step 3: Extract student monodromy M_stu = J_2 @ J_1...")
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs_b=student.hidden_states(x_ref); hs=[h[0] for h in hs_b]
pos=SEQ//2; m=min(PROJ,SEQ,D)

Js_stu=[]; U0=None; ma=None
for l in range(2):
    J,U,m_=layer_jac(student.blocks[l],hs[l],pos,m)
    Js_stu.append(J)
    if U0 is None: U0=U; ma=m_

M_stu=np.eye(ma)
for J in reversed(Js_stu): M_stu=J@M_stu
sv_stu=np.linalg.svd(M_stu,compute_uv=False)
print(f"  M_stu: sv={sv_stu[:4].round(3)}\n")

# Also extract teacher monodromy for comparison
print("Step 4: Extract teacher monodromy M_teach (L0→L14)...")
with torch.no_grad():
    hs_t=teacher.hidden_states(x_ref); hs_t=[h[0] for h in hs_t]
Js_teach=[]
for l in range(L_ATT+1):
    J,_,_=layer_jac(teacher.blocks[l],hs_t[l],pos,m)
    Js_teach.append(J)
    if (l+1)%8==0: print(f"  L{l+1}...",flush=True)

M_teach=np.eye(ma)
for J in reversed(Js_teach): M_teach=J@M_teach
sv_teach=np.linalg.svd(M_teach,compute_uv=False)
print(f"  M_teach: sv={sv_teach[:4].round(3)}\n")

# ── QR refinement sweep ───────────────────────────────────────────────────────
print("Step 5: Apply N QR steps to M_stu and evaluate...")
print(f"  Baseline (no correction): student val={val_student:.4f}")

QR_STEPS=[0,1,2,4,8,12,16,22]
results=[]

# Identity correction (no refinement)
I_corr=np.eye(ma)
val_base=eval_with_correction(student,I_corr,U0)
print(f"  N= 0 QR steps: val={val_base:.4f}  (should match {val_student:.4f})")
results.append((0,val_base))

for N in QR_STEPS[1:]:
    t0=time.time()
    M_ref=qr_refine(M_stu,N)
    # The refined monodromy as a correction operator
    # We want to apply (M_ref / M_stu) as a correction — the delta
    # More precisely: apply M_ref @ M_stu^{-1} as post-processing
    try:
        M_stu_inv=np.linalg.inv(M_stu+1e-6*np.eye(ma))
        M_delta=M_ref@M_stu_inv   # incremental correction
    except:
        M_delta=M_ref
    
    sv_ref=np.linalg.svd(M_ref,compute_uv=False)
    vl=eval_with_correction(student,M_delta,U0)
    results.append((N,vl))
    print(f"  N={N:>2} QR steps: val={vl:.4f}  "
          f"sv(M_ref)={sv_ref[0]:.3f}  t={time.time()-t0:.2f}s")

# Also try: apply teacher monodromy directly
M_teach_delta=M_teach@np.linalg.inv(M_stu+1e-6*np.eye(ma))
vl_teach_corr=eval_with_correction(student,M_teach_delta,U0)
print(f"\n  Teacher monodromy correction: val={vl_teach_corr:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  QR REFINEMENT RESULTS")
print("="*65)
print(f"\n  Teacher (24L):          val={val_teacher:.4f}")
print(f"  Student (2L, no QR):    val={val_student:.4f}")
print(f"  Student + teacher M:    val={vl_teach_corr:.4f}")
print(f"\n  QR refinement sweep:")
print(f"  {'N steps':>9}  {'val':>8}  {'improvement':>12}")
print("  "+"-"*32)
for N,vl in results:
    imp=val_student-vl
    print(f"  {N:>9}  {vl:>8.4f}  {imp:>+12.4f}")

best_N,best_val=min(results,key=lambda x:x[1])
print(f"""
  Best QR refinement: N={best_N} steps → val={best_val:.4f}
  vs student baseline: val={val_student:.4f}  (gap={val_student-best_val:+.4f})
  vs teacher:          val={val_teacher:.4f}  (remaining={best_val-val_teacher:.4f})

  VERDICT:
""")

if best_val < val_student - 0.05:
    print(f"  QR REFINEMENT WORKS.")
    print(f"  Each QR step IS a spectral sequence page.")
    print(f"  The crystallization gap can be closed without deeper layers.")
    print(f"  Cost per step: O(m^2)={ma**2} ops vs O(d^2*seq)={D**2*SEQ} per layer.")
    print(f"  Speedup per refinement step: {D**2*SEQ//ma**2}x")
else:
    print(f"  QR REFINEMENT DOES NOT HELP.")
    print(f"  The QR steps on M_stu do not correspond to spectral pages.")
    print(f"  The missing refinement requires actual transformer layers,")
    print(f"  not post-hoc matrix iterations on the student monodromy.")
    print(f"  The crystallization requires new μ_k structure, not just")
    print(f"  rotating the existing μ_1 + μ_2 structure.")
