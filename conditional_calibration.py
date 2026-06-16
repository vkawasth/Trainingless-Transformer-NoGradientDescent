#!/usr/bin/env python3
"""
Conditional Calibration of Rare Token Embeddings
==================================================
The corpus is fixed. For each rare token t:
  S_t = {sequences in corpus containing t}  — known exactly
  g_t = E_{x in S_t}[dL/dE[t]]             — conditional gradient
  H_t = E_{x in S_t}[(dL/dE[t])^2]         — conditional Fisher
  
  Delta_E[t] = -H_t^{-1} g_t               — exact Newton step

This is NOT the diagonal Fisher Newton (which failed with ||delta||=3388).
The diagonal Fisher averages over ALL sequences including those where t
doesn't appear (giving gradient=0, Fisher=0).

The conditional Fisher averages over ONLY sequences where t appears.
For rare tokens with P(t)=0.001 appearing in ~74 sequences:
  Conditional Fisher H_t ~ 0.01  (gradient magnitude when seen)
  vs diagonal Fisher ~ 0.001*0.01 = 0.00001  (averaged with zeros)

The conditional Newton step is 1000x better conditioned.

PIPELINE:
  1. MF10 (parametric pumping, 20 CE equiv)
  2. 5x LR settle + sign (33 CE steps)
  3. Conditional calibration of rare tokens (deterministic, ~92 CE equiv)
     Replace 100 CE steps with exact conditional Newton
  4. WK Newton polish

PREDICTION:
  IF conditional calibration works: val ~ 0.020-0.030 with no random steps
  The 100 CE steps are replaced by one deterministic corpus pass
  This confirms the calibration paradigm
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  CONDITIONAL RARE TOKEN CALIBRATION")
print(f"  Delta_E[t] = -H_t^{{-1}} g_t  (conditional on t in x)")
print(f"  One deterministic corpus pass per rare token")
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
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2); K=self.WK(h).view(B,S,H,dh).transpose(1,2)
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
    def get_flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
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

# ════════════════════════════════════════════
# PRE-COMPUTE CORPUS INDEX
# ════════════════════════════════════════════
print("Pre-computing corpus index (known once, forever)...")
t0=time.time()

# Token frequencies
token_freq=torch.zeros(VOCAB)
for t in train_ids: token_freq[t]+=1
token_prob=token_freq/token_freq.sum()

# Rare token threshold
avg_prob=1.0/VOCAB
rare_threshold=2.0*avg_prob
rare_tokens=[t for t in range(VOCAB) if 0<float(token_prob[t])<rare_threshold]
common_tokens=[t for t in range(VOCAB) if float(token_prob[t])>=rare_threshold]
print(f"  Rare tokens: {len(rare_tokens)}, Common: {len(common_tokens)}")

# Index: for each token, which sequence starts contain it?
token_to_seqs=defaultdict(list)
for i in range(len(train_ids)-SEQ-1):
    seq=train_ids[i+1:i+SEQ+1]  # targets
    seen=set(seq)
    for t in seen:
        token_to_seqs[t].append(i)

rare_counts={t:len(token_to_seqs[t]) for t in rare_tokens}
print(f"  Avg sequences per rare token: {np.mean(list(rare_counts.values())):.1f}")
print(f"  Min/Max: {min(rare_counts.values())}/{max(rare_counts.values())}")
print(f"  Index built in {time.time()-t0:.1f}s\n")

# ════════════════════════════════════════════
# CONDITIONAL CALIBRATION FUNCTION
# ════════════════════════════════════════════
def conditional_calibrate_rare_tokens(stu, scale=0.5, eps=1e-3,
                                       max_seqs_per_token=None):
    """
    For each rare token t:
      1. Find S_t = sequences containing t (from corpus index)
      2. Compute conditional gradient g_t = mean_S_t[dL/dE[t]]
      3. Compute conditional Fisher H_t = mean_S_t[(dL/dE[t])^2]
      4. Apply: E[t] -= scale * g_t / (H_t + eps)
      
    This is the CORRECT Newton step — conditioned on token appearance.
    Well-conditioned because H_t ~ 0.01 (not 0.00001 like diagonal Fisher).
    """
    stu.train()
    total_seqs=0
    deltas_applied=0
    delta_norms=[]

    for t in rare_tokens:
        seqs=token_to_seqs[t]
        if not seqs: continue
        if max_seqs_per_token: seqs=seqs[:max_seqs_per_token]

        g_acc=torch.zeros(D)  # (D,) gradient for E[t]
        f_acc=torch.zeros(D)  # (D,) Fisher diagonal for E[t]
        n_seqs=len(seqs)

        for seq_start in seqs:
            x=train_t[seq_start:seq_start+SEQ].unsqueeze(0)
            y=train_t[seq_start+1:seq_start+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=stu.te.weight.grad[t].detach()
            g_acc+=g; f_acc+=g**2
            total_seqs+=1

        g_mean=g_acc/n_seqs
        f_mean=f_acc/n_seqs
        delta=-(g_mean/(f_mean+eps))
        delta_norm=float(delta.norm())

        # Only apply if delta is reasonable (not exploding)
        if delta_norm<10.0:
            with torch.no_grad():
                stu.te.weight[t].add_(scale*delta)
            deltas_applied+=1
            delta_norms.append(delta_norm)

    print(f"  Conditional calibration: {deltas_applied}/{len(rare_tokens)} tokens updated")
    print(f"  Total sequences processed: {total_seqs}")
    print(f"  Mean ||delta_t||: {np.mean(delta_norms):.4f}  "
          f"Max: {np.max(delta_norms):.4f}")
    return deltas_applied

def hv_product(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

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

print("Training teacher...")
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

print("Computing v_neg...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.\n")

def apply_mf(stu,n_iter=10,mf_lr=0.01,n_corpus=200):
    for it in range(n_iter):
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(False)
            stu.blocks[l].attn.WQ.weight.requires_grad_(False)
        eg=torch.zeros(VOCAB,D); ef=torch.zeros(VOCAB,D)
        torch.manual_seed(it*1000)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            if stu.te.weight.grad is not None:
                g=stu.te.weight.grad.detach(); eg+=g; ef+=g**2
        eg/=n_corpus; ef/=n_corpus
        with torch.no_grad(): stu.te.weight.add_(-mf_lr*eg/(ef+1e-4))
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(True)
            stu.blocks[l].attn.WQ.weight.requires_grad_(True)
        stu.te.weight.requires_grad_(False)
        wg=torch.zeros_like(stu.blocks[0].attn.WK.weight); wf=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        torch.manual_seed(it*1000+500)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None: g+=stu.blocks[l].attn.WK.weight.grad/N_STU
            wg+=g; wf+=g**2
        wg/=n_corpus; wf/=n_corpus
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(-mf_lr*wg/(wf+1e-4))
                stu.blocks[l].attn.WQ.weight.add_(-mf_lr*wg.T/(wf.T+1e-4))
        stu.te.weight.requires_grad_(True)
        if (it+1)%5==0: print(f"  MF iter {it+1}: val={eval_val(stu,n=5):.4f}")

print("="*65)
print("EXPERIMENTS")
print("  A: MF10 + settle + 100CE + Newton  (confirmed ~0.028)")
print("  B: MF10 + settle + conditional_calibrate + Newton")
print("     (replace 100 CE with deterministic rare token Newton)")
print("  C: MF10 + settle + conditional_calibrate + 25CE + Newton")
print("     (calibrate then refine with minimal CE)")
print("="*65)

def run_settle_sign(stu):
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,34):
        for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
    with torch.no_grad():
        for l in [1,2]:
            stu.blocks[l].attn.WV.weight.mul_(-1); stu.blocks[l].attn.op.weight.mul_(-1)
    return eval_val(stu,n=15)

# A: baseline with 100CE
print("\n[A] MF10 + settle + 100CE + Newton")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
print(f"  after settle+sign: {v_settle:.4f}")
opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
    if step in [25,50,100]: print(f"  CE {step}: {eval_val(stu,n=10):.4f}")
apply_newton_wk(stu)
vA=eval_val(stu,n=30); print(f"  FINAL={vA:.4f}")

# B: MF10 + settle + conditional calibration + Newton
print("\n[B] MF10 + settle + conditional calibration + Newton")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
print(f"  after settle+sign: {v_settle:.4f}")
print(f"  Applying conditional rare token calibration...")
t_cal=time.time()
n_updated=conditional_calibrate_rare_tokens(stu,scale=0.5,eps=1e-3)
print(f"  Calibration time: {time.time()-t_cal:.1f}s")
v_cal=eval_val(stu,n=20); print(f"  after calibration: {v_cal:.4f}")
apply_newton_wk(stu)
vB=eval_val(stu,n=30); print(f"  FINAL={vB:.4f}")

# C: MF10 + settle + conditional calibration + 25CE + Newton
print("\n[C] MF10 + settle + conditional calibration + 25CE + Newton")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
print(f"  after settle+sign: {v_settle:.4f}")
conditional_calibrate_rare_tokens(stu,scale=0.5,eps=1e-3)
v_cal=eval_val(stu,n=15); print(f"  after calibration: {v_cal:.4f}")
opt3=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt3.param_groups: pg['lr']=clr(step,25)
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt3.step()
print(f"  after 25CE: {eval_val(stu,n=15):.4f}")
apply_newton_wk(stu)
vC=eval_val(stu,n=30); print(f"  FINAL={vC:.4f}")

print(f"""
{'='*65}
  CONDITIONAL CALIBRATION RESULTS
{'='*65}
    Teacher:                 val={val_teacher:.4f}
    A (MF10+100CE+Newton):   val={vA:.4f}  [stochastic baseline]
    B (MF10+cond_cal+Newton):val={vB:.4f}  [deterministic replacement]
    C (MF10+cond+25CE+N):    val={vC:.4f}  [calibrate then refine]

  IF B ~ A: conditional calibration replaces 100 CE steps
    The 100 steps were computing this exact conditional Newton
    Corpus is known → training is unnecessary → calibration suffices
    
  IF B > A (worse): coupling between tokens matters
    Rare tokens interact through attention — calibrating independently
    fails because their embeddings are jointly constrained
    The 100 CE steps integrate joint statistics, not individual tokens
    
  IF C < A with 25 instead of 100 steps:
    Conditional calibration reduces the CE requirement by 75%
    Combined pipeline: MF10 + calibrate + 25CE is optimal
""")
