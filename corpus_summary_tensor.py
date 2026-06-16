#!/usr/bin/env python3
"""
Corpus Summary Tensor — One-Shot Weight Correction
====================================================
The key insight: the 200 CE steps are computing
  J_bar_l = E_{x~D}[J_l(x)]  (expected Jacobian over corpus)
one batch at a time. We can compute this directly.

PIPELINE:
  1. Compute J_bar_l from N reference sequences (corpus sample)
  2. Compute data-averaged monodromy M_fwd_D = J_bar_14 @ ... @ J_bar_0
  3. Compute data-averaged MC element: alpha_D = log(M_fwd_D)
  4. Run Newton solver on J_bar_14 to find alpha*
  5. Apply one-shot correction: W_K^l <- U14 (S_l + eps*alpha*) U14^T
  6. Measure val at zero gradient steps and after minimal fine-tuning

COMPARISON:
  A: Serre + 200CE (confirmed, val=0.187)
  B: Corpus-averaged J + Newton MC + 0CE  (zero-gradient target)
  C: Corpus-averaged J + Newton MC + 50CE (minimal fine-tuning)

The gap B vs A = irreducible corpus-specific content
The gap C vs A = how much the one-shot correction reduces CE steps
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import logm as scipy_logm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; N_CORPUS_REFS=50  # number of sequences for corpus average

print(f"\n{'='*65}")
print(f"  CORPUS SUMMARY TENSOR")
print(f"  One-shot MC correction from data-averaged Jacobians")
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
teacher.eval()
val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════
# STEP 1: CORPUS-AVERAGED JACOBIANS (5 refs vs N_CORPUS_REFS)
# ════════════════════════════════════════════════════
print("="*65)
print(f"STEP 1: CORPUS SUMMARY TENSOR")
print(f"  Computing E_D[J_l] over {N_CORPUS_REFS} corpus sequences")
print(f"  (5 refs = standard; {N_CORPUS_REFS} refs = corpus average)")
print("="*65)

pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None

# Standard 5-ref Jacobians
J_acc5=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
torch.manual_seed(0)
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc5[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  5-ref: {ref+1}/5...",flush=True)
Js5=[np.mean(J_acc5[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14_5=Js5[L_ATT]; U14=Us[L_ATT]

# Corpus-averaged Jacobians (N_CORPUS_REFS sequences from training data)
print(f"\n  Computing {N_CORPUS_REFS}-ref corpus average...")
J_accN=[[] for _ in range(N_LAYERS_T)]
torch.manual_seed(42)
for ref in range(N_CORPUS_REFS):
    # Sample from training data (diverse corpus coverage)
    x_ref,_=get_batch('train'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_accN[l].append(J)
    if (ref+1)%10==0: print(f"  corpus-ref: {ref+1}/{N_CORPUS_REFS}...",flush=True)
JsN=[np.mean(J_accN[l],axis=0) for l in range(N_LAYERS_T)]
J14_N=JsN[L_ATT]

# Measure how much the corpus average differs from 5-ref
print(f"\n  Jacobian comparison (5-ref vs {N_CORPUS_REFS}-ref corpus):")
print(f"  {'L':>3}  {'||J5-JN||':>11}  {'||J5||':>8}  {'rel_diff':>9}")
print("  "+"-"*36)
for l in range(0,N_LAYERS_T,4):
    diff=N(Js5[l]-JsN[l])
    rel=diff/max(N(Js5[l]),1e-8)
    att=" ←L14" if l==L_ATT else ""
    print(f"  L{l:>2}  {diff:>11.5f}  {N(Js5[l]):>8.4f}  {rel:>9.4f}{att}")

print(f"\n  Mean relative diff across all layers: "
      f"{np.mean([N(Js5[l]-JsN[l])/max(N(Js5[l]),1e-8) for l in range(N_LAYERS_T)]):.5f}")

# ════════════════════════════════════════════════════
# STEP 2: DATA-AVERAGED MONODROMY AND MC ELEMENT
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 2: DATA-AVERAGED MC ELEMENT")
print(f"  alpha_D = log(M_fwd_D) where M_fwd_D = E_D[M_fwd(x)]")
print("="*65)

def compute_monodromy(Js):
    M=np.eye(ma)
    for l in range(L_ATT+1): M=Js[l]@M
    return M

def mc_residual(alpha,J14):
    dJ=J14-np.eye(ma)
    r=comm(dJ,alpha)+(1/6)*l3(alpha,alpha,alpha)
    return r,N(r)

# 5-ref monodromy
M5=compute_monodromy(Js5)
sv5=np.linalg.svd(M5,compute_uv=False)
alpha5=np.real(scipy_logm(M5))
_,res5=mc_residual(alpha5,J14_5)

# Corpus-averaged monodromy
MN=compute_monodromy(JsN)
svN=np.linalg.svd(MN,compute_uv=False)
alphaN=np.real(scipy_logm(MN))
_,resN=mc_residual(alphaN,J14_N)

print(f"\n  5-ref monodromy:       sv[:4]={sv5[:4].round(3)}")
print(f"  {N_CORPUS_REFS}-ref monodromy: sv[:4]={svN[:4].round(3)}")
print(f"\n  5-ref MC residual:     {res5:.4f}")
print(f"  {N_CORPUS_REFS}-ref MC residual: {resN:.4f}")
print(f"  Improvement from more refs: {res5/max(resN,1e-8):.2f}x")

# Newton MC solver on corpus-averaged J14
print(f"\n  Newton MC solver on {N_CORPUS_REFS}-ref J14...")
alpha=alphaN.copy()
dJ14_N=J14_N-np.eye(ma)
best_alpha=alpha.copy(); best_res=resN

for it in range(300):
    mc_vec,res=mc_residual(alpha,J14_N)
    if res<best_res: best_res=res; best_alpha=alpha.copy()
    grad=2*comm(dJ14_N+alpha,mc_vec)
    alpha=alpha-0.005*grad

_,final_res=mc_residual(best_alpha,J14_N)
corr=float(np.corrcoef(best_alpha.flatten(),dJ14_N.flatten())[0,1])
print(f"  Initial MC residual:  {resN:.4f}")
print(f"  Final MC residual:    {final_res:.4f}  ({resN/max(final_res,1e-8):.2f}x reduction)")
print(f"  corr(alpha*, dJ14_N): {corr:.4f}")

# ════════════════════════════════════════════════════
# STEP 3: BUILD CASCADES
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 3: CASCADES FROM 5-REF vs CORPUS-AVERAGED J14")
print("="*65)

def build_cascade(J14_src, Js_src):
    cascade=[]
    for l in range(1,N_STU+1):
        C=Js_src[min(L_ATT+l,N_LAYERS_T-1)].copy()
        for _ in range(l): C=comm(J14_src,C)
        n=N(C); cascade.append(C/max(n,1e-8))
    return cascade

# Cascade from 5-ref (standard)
casc_5=build_cascade(J14_5,Js5)
# Cascade from corpus-averaged J14
casc_N=build_cascade(J14_N,JsN)
# Corpus cascade + MC alpha correction
casc_N_mc=[]
for l in range(N_STU):
    C=casc_N[l]+0.1*best_alpha/max(N(best_alpha),1e-8)
    casc_N_mc.append(C/max(N(C),1e-8))

print(f"\n  Cascade alignment (corpus vs 5-ref):")
for l in range(N_STU):
    align=float(np.sum(casc_5[l]*casc_N[l]))
    print(f"  Level {l+1}: cosine={align:.4f}")

# ════════════════════════════════════════════════════
# STEP 4: STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 4: STUDENT EXPERIMENTS")
print("  A: 5-ref Serre + 200CE (baseline)")
print(f"  B: {N_CORPUS_REFS}-ref Serre + 200CE (corpus-averaged cascade)")
print(f"  C: {N_CORPUS_REFS}-ref Serre + MC correction + 200CE")
print(f"  D: {N_CORPUS_REFS}-ref Serre + MC correction + 50CE (minimal)")
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
    v0=eval_val(stu,n=20)
    print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            b="✓" if v<val_teacher else " "
            print(f"  [{label}] step {step:>4}  val={v:.4f} {b}")
    return eval_val(stu),ck

vA,ckA=run(casc_5,"A-5ref-std")
vB,ckB=run(casc_N,f"B-{N_CORPUS_REFS}ref-corpus")
vC,ckC=run(casc_N_mc,f"C-{N_CORPUS_REFS}ref-MC")
vD,ckD=run(casc_N_mc,f"D-{N_CORPUS_REFS}ref-MC-50CE",steps=50)

print(f"\n{'='*65}")
print("  CORPUS SUMMARY TENSOR RESULTS")
print("="*65)
print(f"""
  DATA-AVERAGED JACOBIANS:
    5-ref MC residual:          {res5:.4f}
    {N_CORPUS_REFS}-ref MC residual:         {resN:.4f}
    After Newton ({N_CORPUS_REFS}-ref):      {final_res:.4f}
    corr(alpha*, dJ14):         {corr:.4f}

  CONVERGENCE:
  {'step':>6}  {'A-5ref':>8}  {'B-corpus':>10}  {'C-MC':>8}  {'D-50CE':>8}""")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    if any(v and v<(a or 99)-0.005 for v in [b,c,d]): row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:            val={val_teacher:.4f}
    A (5-ref Serre):    val={vA:.4f}
    B ({N_CORPUS_REFS}-ref corpus):  val={vB:.4f}  diff={vA-vB:+.4f}
    C ({N_CORPUS_REFS}-ref MC):      val={vC:.4f}  diff={vA-vC:+.4f}
    D ({N_CORPUS_REFS}-ref 50CE):    val={vD:.4f}  diff={vA-vD:+.4f}

  KEY QUESTIONS:
    Does B > A? More corpus refs = better cascade = fewer steps?
    Does C reach val<0.25 at step 50? MC correction closes the gap?
    Does D (50 CE steps) beat teacher? The "50-step target"?

  THE COLIMIT PICTURE:
    The corpus summary tensor E_D[J_l] is the target colimit.
    The 200 CE steps compute the colimit one batch at a time.
    With {N_CORPUS_REFS} reference sequences we have a richer estimate.
    If B beats A: more corpus references = better colimit approximation
    = fewer CE steps to reach the target.
""")
