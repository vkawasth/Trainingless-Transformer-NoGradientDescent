#!/usr/bin/env python3
"""
Gradient Alignment — One-Shot Corpus Integration
=================================================
The transformer is not moving. All tokens are known.
The corpus is fixed. The training trajectory is predetermined.

Gradient descent is approximating ONE computation:
  E* = E_0 - H_E^{-1} @ grad_E L

where H_E = E_D[d²L/dEdE^T] is the embedding Hessian
and grad_E L = E_D[dL/dE] is the expected embedding gradient.

Both are computable in one corpus pass. No iteration needed.

THE FIXED POINT:
  For each token t:
    E*[t] = E_0[t] - H_E[t]^{-1} @ E_D[dL/dE[t]]

  With tied embeddings (W_head = E^T), the Hessian is:
    H_E[t] = P(t) * (I - p_t p_t^T)  (softmax Hessian)
  where p_t is the predicted probability vector at token t.

  This gives the closed-form Newton step for each token embedding.

PREDICTION:
  After one Newton step on E: val should reach basin floor
  (close to 0.033 without any CE steps)
  
  Because: the 167 CE steps are computing this Newton step
  stochastically, one minibatch at a time.

ALSO TEST:
  Per-token Newton step weighted by corpus frequency:
    E*[t] = E_0[t] - (1/P(t)) * diag_hessian^{-1} @ grad_E[t]
  This is the natural gradient step for embeddings.

THREE EXPERIMENTS:
  A: Standard pipeline (33CE 5x + sign + 167CE + Newton) — baseline
  B: One Newton step on E only, then standard 167 CE
  C: One Newton step on E only, then 50 CE (test if 167 reduced)
  D: One Newton step on E only, zero CE (test if 167 eliminated)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  GRADIENT ALIGNMENT — ONE-SHOT CORPUS INTEGRATION")
print(f"  E* = E_0 - H_E^{{-1}} @ E_D[dL/dE]")
print(f"  One Newton step on embeddings replaces 167 CE steps?")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
vocab={t:i for i,t in enumerate(_v)} if isinstance(_v,list) else _v
id2tok={i:t for t,i in vocab.items()}
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
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
    def get_flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat_params(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))
def eval_val(m,n=40):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad(): vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}")
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

def build_student():
    torch.manual_seed(99); stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight); stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            for s,d in [(teacher.blocks[L_ATT].attn.WK,stu.blocks[l].attn.WK),
                        (teacher.blocks[L_ATT].attn.WQ,stu.blocks[l].attn.WQ),
                        (teacher.blocks[L_ATT].attn.WV,stu.blocks[l].attn.WV),
                        (teacher.blocks[L_ATT].attn.op,stu.blocks[l].attn.op),
                        (teacher.blocks[L_ATT].ff.g,stu.blocks[l].ff.g),
                        (teacher.blocks[L_ATT].ff.v,stu.blocks[l].ff.v),
                        (teacher.blocks[L_ATT].ff.o,stu.blocks[l].ff.o)]:
                d.weight.copy_(s.weight)
    return stu

# ════════════════════════════════════════════
# EMBEDDING NEWTON STEP
# ════════════════════════════════════════════
def embedding_newton_step(stu, n_corpus=500, eps=1e-4, scale=1.0):
    """
    One Newton step on the embedding matrix only.
    
    For each token t:
      grad_t = E_D[dL/dE[t]] — expected gradient of embedding t
      hess_diag_t = E_D[(dL/dE[t])^2] — diagonal Fisher for token t
      delta_t = -grad_t / (hess_diag_t + eps)
      E*[t] = E[t] + scale * delta_t
    
    This is the natural gradient step for each token embedding,
    weighted by the diagonal of the Fisher information matrix.
    
    It is the exact computation that 167 CE steps approximate stochastically.
    Cost: n_corpus forward+backward passes. No iteration.
    """
    grad_acc=torch.zeros(VOCAB,D)   # (VOCAB, D) accumulated gradient
    fisher_d=torch.zeros(VOCAB,D)   # (VOCAB, D) diagonal Fisher
    count=torch.zeros(VOCAB)        # how many times each token appeared

    torch.manual_seed(42)
    print(f"  Computing embedding gradient + Fisher over {n_corpus} sequences...")
    for i in range(n_corpus):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0)
        y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)

        stu.zero_grad(); _,loss=stu(x,y); loss.backward()

        # Embedding gradient: dL/dE for each token in this sequence
        g=stu.te.weight.grad.detach().clone()  # (VOCAB, D)
        grad_acc+=g
        fisher_d+=g**2
        # Count appearances
        for t in y.flatten(): count[t]+=1

        if (i+1)%100==0: print(f"    {i+1}/{n_corpus}...",flush=True)

    # Normalize
    grad_mean=grad_acc/n_corpus      # E_D[dL/dE]
    fisher_diag=fisher_d/n_corpus    # E_D[(dL/dE)^2]

    # Per-token Newton step
    delta=-(grad_mean/(fisher_diag+eps))  # (VOCAB, D)

    # Report statistics
    grad_norms=grad_mean.norm(dim=1)  # (VOCAB,) per-token gradient norm
    delta_norms=delta.norm(dim=1)     # (VOCAB,) per-token step size

    print(f"\n  ||grad_mean||_F: {float(grad_mean.norm()):.6f}")
    print(f"  ||delta||_F:     {float(delta.norm()):.6f}")
    print(f"\n  Top 10 tokens by gradient norm:")
    top_g=grad_norms.argsort(descending=True)[:10]
    for t in top_g:
        tok=id2tok.get(int(t),'<unk>')
        print(f"    {tok:>20}: ||grad||={float(grad_norms[t]):.6f}  "
              f"||delta||={float(delta_norms[t]):.6f}  count={int(count[t])}")

    # Apply Newton step to embedding matrix
    with torch.no_grad():
        stu.te.weight.add_(scale*delta)
        # Note: head weight is tied to embedding, updates automatically

    return delta, grad_mean, fisher_diag

# ════════════════════════════════════════════
# COMPUTE v_neg FOR SADDLE EXIT
# ════════════════════════════════════════════
def hv_product(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

print("Computing v_neg...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.\n")

def apply_newton_wk(stu,n_seq=500,eps=1e-3,scale=0.5):
    ga=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fd=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(2)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if stu.blocks[l].attn.WK.weight.grad is not None:
                g+=stu.blocks[l].attn.WK.weight.grad/N_STU
        ga+=g; fd+=g**2
    delta=-(ga/n_seq)/((fd/n_seq)+eps)
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)

def run_pipeline(label, do_saddle=True, do_emb_newton=False,
                 emb_newton_scale=1.0, emb_newton_seqs=500,
                 settle_lr=5.0, settle_steps=33,
                 basin_steps=167, do_sign=True, do_wk_newton=True):
    stu=build_student()
    print(f"\n  [{label}]  zero-shot: {eval_val(stu,n=20):.4f}")

    # Saddle exit
    if do_saddle:
        w0=stu.get_flat_params()
        stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
        print(f"    saddle exit: {eval_val(stu,n=20):.4f}")

    # Embedding Newton step (ONE SHOT)
    if do_emb_newton:
        print(f"    Computing embedding Newton step...")
        delta,gm,fd=embedding_newton_step(stu,n_corpus=emb_newton_seqs,
                                           eps=1e-4,scale=emb_newton_scale)
        print(f"    after emb Newton (scale={emb_newton_scale}): {eval_val(stu,n=20):.4f}")

    # Settle
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*settle_lr,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,settle_steps+1):
        for pg in opt_s.param_groups: pg['lr']=LR*settle_lr*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
    print(f"    after {settle_steps} settle (5xLR): {eval_val(stu,n=20):.4f}")

    # Sign correction
    if do_sign:
        with torch.no_grad():
            for l in [1,2]:
                stu.blocks[l].attn.WV.weight.mul_(-1)
                stu.blocks[l].attn.op.weight.mul_(-1)
        print(f"    after sign: {eval_val(stu,n=20):.4f}")

    # Basin descent
    opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,basin_steps+1):
        for pg in opt2.param_groups: pg['lr']=clr(step,basin_steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
        if step in [25,50,75,100,125,150,167]:
            v=eval_val(stu,n=20)
            print(f"    basin {step:>4}: {v:.4f}{' ✓' if v<val_teacher else ''}")

    if do_wk_newton:
        apply_newton_wk(stu)
        print(f"    WK Newton: {eval_val(stu,n=20):.4f}")

    vf=eval_val(stu,n=30)
    print(f"    FINAL={vf:.4f}")
    return vf

print("="*65)
print("EXPERIMENTS")
print("  A: Baseline (saddle+5xLR+sign+167CE+Newton)")
print("  B: + Embedding Newton step before settle (scale=1.0)")
print("  C: + Embedding Newton step before settle (scale=0.1)")
print("  D: Embedding Newton only, then 100CE (no settle)")
print("  E: Emb Newton + saddle + settle + 100CE + Newton")
print("="*65)

vA=run_pipeline("A-Baseline",do_emb_newton=False)
vB=run_pipeline("B-EmbNewton+full",do_emb_newton=True,
                 emb_newton_scale=1.0,basin_steps=167)
vC=run_pipeline("C-EmbNewton0.1+full",do_emb_newton=True,
                 emb_newton_scale=0.1,basin_steps=167)
vD=run_pipeline("D-EmbNewton+100CE",do_emb_newton=True,
                 emb_newton_scale=0.1,basin_steps=100)
vE=run_pipeline("E-EmbNewton+settle+100CE",do_emb_newton=True,
                 emb_newton_scale=0.1,basin_steps=100)

print(f"""
{'='*65}
  GRADIENT ALIGNMENT RESULTS
{'='*65}

  FINAL:
    Teacher:              val={val_teacher:.4f}
    A (baseline):         val={vA:.4f}
    B (emb Newton 1.0):   val={vB:.4f}  diff={vA-vB:+.4f}
    C (emb Newton 0.1):   val={vC:.4f}  diff={vA-vC:+.4f}
    D (emb N + 100CE):    val={vD:.4f}  diff={vA-vD:+.4f}
    E (emb N +s+ 100CE):  val={vE:.4f}  diff={vA-vE:+.4f}

  IF B or C < A:
    The embedding Newton step (one corpus pass) provides information
    that reduces the number of basin CE steps needed.
    The 167 steps are approximating this one computation.
    
  IF D ~ A with 100 instead of 167 steps:
    The embedding Newton step saves 67 basin steps.
    The computation is: saddle exit (2 CE) + emb Newton (5 CE)
    + settle (33) + 100 CE + WK Newton (2.5 CE) = ~143 CE equiv
    vs current 200 + 2.5 Newton = 202.5 CE equiv.
    
  IF B or C > A (diverges):
    Scale is too large. The embedding Newton step overshoots.
    The basin floor is not reachable by a single linear step.
    The 167 CE steps are truly irreducible.
""")
