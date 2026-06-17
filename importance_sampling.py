#!/usr/bin/env python3
"""
Importance Sampling for Rare Token Basin Descent
=================================================
The 100-step basin descent minimum is the Nyquist-Shannon limit
for rare token statistics:
  T_min = n_obs_required / (P_rare * batch_size * seq_len)
        = 50 / (0.001 * 8 * 64) = 97 steps

With importance sampling at rate k for rare tokens:
  T_min(k) = T_min / k = 100/k steps

Theory:
  k=2:  T_min = 50 steps
  k=5:  T_min = 20 steps
  k=10: T_min = 10 steps

IMPLEMENTATION:
  Importance sampling changes the batch distribution:
  Instead of uniform sampling from corpus, over-sample
  sequences containing rare tokens (freq < threshold).
  
  Correct the gradient with importance weights:
    grad_corrected = grad * P_uniform / P_importance
  
  This gives an unbiased estimate of the true gradient
  while concentrating samples on rare tokens.

  Rare token = token_freq < 2 * average_frequency
  = token_freq < 2/VOCAB ≈ 0.003

EXPERIMENTS:
  A: Baseline (MF3 + 100CE standard) — val=0.0372
  B: MF3 + IS(k=2) for 50CE — test if 50 steps with 2x IS = 100 steps
  C: MF3 + IS(k=5) for 20CE — test if 20 steps with 5x IS = 100 steps  
  D: MF3 + IS(k=10) for 10CE — test if 10 steps with 10x IS = 100 steps
  E: MF3 + IS(k=2) for 100CE — does IS improve quality beyond 100 steps?

If B ~ A: 2x importance sampling halves the required steps.
If C ~ A: 5x IS reduces to 20 steps.
If D ~ A: 10 steps sufficient — 10x compression of basin descent.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  IMPORTANCE SAMPLING FOR RARE TOKEN BASIN DESCENT")
print(f"  T_min(k) = 100/k steps with k-fold rare token oversampling")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
vocab={t:i for i,t in enumerate(_v)} if isinstance(_v,list) else _v
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

# Compute token frequencies over corpus
print("Computing token frequencies...")
token_freq=torch.zeros(VOCAB)
for i in range(0,len(train_ids)-SEQ,SEQ):
    for t in train_ids[i+1:i+SEQ+1]: token_freq[t]+=1
token_prob=token_freq/token_freq.sum()

# Rare token threshold: below 2x average frequency
avg_prob=1.0/VOCAB
rare_threshold=2.0*avg_prob
rare_mask=(token_prob>0)&(token_prob<rare_threshold)
common_mask=token_prob>=rare_threshold
n_rare=rare_mask.sum().item()
n_common=common_mask.sum().item()
print(f"  Rare tokens (P < {rare_threshold:.4f}): {n_rare}")
print(f"  Common tokens: {n_common}")
print(f"  Mean rare freq: {float(token_prob[rare_mask].mean()):.6f}")
print(f"  Mean common freq: {float(token_prob[common_mask].mean()):.6f}")

# Build importance-sampled batch getter
# Strategy: for each sequence position, with prob p_rare, 
# sample from sequences containing rare tokens
def build_sequence_index():
    """Index sequences by whether they contain rare tokens."""
    rare_seq_starts=[]
    common_seq_starts=[]
    for i in range(0,len(train_ids)-SEQ-1,1):
        seq=train_ids[i:i+SEQ+1]
        has_rare=any(rare_mask[t] for t in seq[1:])
        if has_rare: rare_seq_starts.append(i)
        else: common_seq_starts.append(i)
    return rare_seq_starts, common_seq_starts

print("\nIndexing sequences by rare token content...")
rare_starts, common_starts=build_sequence_index()
rare_starts_t=torch.tensor(rare_starts)
common_starts_t=torch.tensor(common_starts)
print(f"  Sequences with rare tokens: {len(rare_starts)}")
print(f"  Sequences without rare tokens: {len(common_starts)}")
p_rare_seq=len(rare_starts)/(len(rare_starts)+len(common_starts))
print(f"  Base probability of rare-containing sequence: {p_rare_seq:.3f}")

def get_batch_importance(split='train', rare_oversample=1.0):
    """
    Get batch with importance sampling.
    rare_oversample=k means rare-containing sequences are k-x more likely.
    If no common sequences exist, sample entirely from rare.
    """
    if split=='val' or rare_oversample<=1.0 or len(common_starts)==0:
        data=val_t if split=='val' else train_t
        ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
        x=torch.stack([data[i:i+SEQ] for i in ix])
        y=torch.stack([data[i+1:i+SEQ+1] for i in ix])
        return x, y, torch.ones(BATCH)

    # Importance sampling: over-sample rare sequences
    p_rare_imp=min(p_rare_seq*rare_oversample, 0.99)
    p_common_imp=1.0-p_rare_imp
    w_rare=p_rare_seq/p_rare_imp
    w_common=(1-p_rare_seq)/p_common_imp if p_common_imp>1e-6 else 1.0

    xs=[]; ys=[]; ws=[]
    for _ in range(BATCH):
        use_rare=(torch.rand(1).item()<p_rare_imp) or len(common_starts)==0
        if use_rare and len(rare_starts)>0:
            idx=rare_starts_t[torch.randint(0,len(rare_starts_t),(1,))[0]]
            weight=w_rare
        elif len(common_starts)>0:
            idx=common_starts_t[torch.randint(0,len(common_starts_t),(1,))[0]]
            weight=w_common
        else:
            idx=rare_starts_t[torch.randint(0,len(rare_starts_t),(1,))[0]]
            weight=1.0
        xs.append(train_t[idx:idx+SEQ])
        ys.append(train_t[idx+1:idx+SEQ+1])
        ws.append(weight)

    return torch.stack(xs), torch.stack(ys), torch.tensor(ws)

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
    def forward(self,h,weights=None):
        return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))
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
        if y is None: return logits,None
        return logits,F.cross_entropy(logits.view(-1,VOCAB),y.view(-1))
    def forward_weighted(self,x,y,weights):
        """Forward pass with per-sequence importance weights."""
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        # Per-sequence weighted loss
        losses=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1),reduction='none')
        losses=losses.view(x.shape[0],-1).mean(dim=1)  # (B,)
        weighted_loss=(losses*weights).mean()
        return logits,weighted_loss
    def get_flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat_params(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))
def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

print("\nTraining teacher (300 steps)...")
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

print("Computing v_neg and MF state...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.")

def apply_mf_init(stu, n_iter=3, n_corpus=200, mf_lr=0.01):
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
        wg=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        wf=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        torch.manual_seed(it*1000+500)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    g+=stu.blocks[l].attn.WK.weight.grad/N_STU
            wg+=g; wf+=g**2
        wg/=n_corpus; wf/=n_corpus
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(-mf_lr*wg/(wf+1e-4))
                stu.blocks[l].attn.WQ.weight.add_(-mf_lr*wg.T/(wf.T+1e-4))
        stu.te.weight.requires_grad_(True)

def run(label, basin_steps=100, rare_oversample=1.0, do_newton=True):
    stu=build_student()
    w0=stu.get_flat_params()
    stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))

    # MF init (3 iter, confirmed)
    apply_mf_init(stu,n_iter=3,n_corpus=200,mf_lr=0.01)

    # 5x LR settle
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,34):
        for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()

    # Sign correction
    with torch.no_grad():
        for l in [1,2]:
            stu.blocks[l].attn.WV.weight.mul_(-1); stu.blocks[l].attn.op.weight.mul_(-1)
    v_sign=eval_val(stu,n=15)
    print(f"\n  [{label}]  after MF+settle+sign: {v_sign:.4f}")

    # Basin descent with importance sampling
    opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    rare_obs=0.0  # track rare token observations

    for step in range(1,basin_steps+1):
        for pg in opt2.param_groups: pg['lr']=clr(step,basin_steps)
        stu.train()
        x,y,weights=get_batch_importance('train',rare_oversample)
        _,loss=stu.forward_weighted(x,y,weights)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()

        # Count rare token observations
        rare_in_batch=sum(1 for t in y.flatten() if rare_mask[t])
        rare_obs+=rare_in_batch

        if step in [10,20,30,50,75,100]:
            v=eval_val(stu,n=15)
            print(f"    step {step:>4}: val={v:.4f}  "
                  f"rare_obs={rare_obs:.0f}/{50*n_rare:.0f}"
                  f"{' ✓' if v<val_teacher else ''}")

    if do_newton:
        apply_newton_wk(stu)
        print(f"    Newton: {eval_val(stu,n=15):.4f}")

    vf=eval_val(stu,n=30)
    print(f"    FINAL={vf:.4f}  total_rare_obs={rare_obs:.0f}")
    return vf

print(f"\n{'='*65}")
print("IMPORTANCE SAMPLING EXPERIMENTS")
print(f"  Rare sequences: {len(rare_starts)}/{len(rare_starts)+len(common_starts)}")
print(f"  Base rare sequence probability: {p_rare_seq:.3f}")
print()
print("  A: MF3 + 100CE standard (baseline, val=0.037)")
print("  B: MF3 + IS(k=2) for 50CE  — tests T=50 with 2x IS")
print("  C: MF3 + IS(k=5) for 20CE  — tests T=20 with 5x IS")
print("  D: MF3 + IS(k=2) for 100CE — does IS improve quality?")
print("="*65)

vA=run("A-100CE-standard",basin_steps=100,rare_oversample=1.0)
vB=run("B-IS2x-50CE",basin_steps=50,rare_oversample=2.0)
vC=run("C-IS5x-20CE",basin_steps=20,rare_oversample=5.0)
vD=run("D-IS2x-100CE",basin_steps=100,rare_oversample=2.0)

print(f"""
{'='*65}
  IMPORTANCE SAMPLING RESULTS
{'='*65}

  FINAL:
    Teacher:           val={val_teacher:.4f}
    A (100CE std):     val={vA:.4f}  [baseline]
    B (2x IS, 50CE):   val={vB:.4f}  diff={vA-vB:+.4f}
    C (5x IS, 20CE):   val={vC:.4f}  diff={vA-vC:+.4f}
    D (2x IS, 100CE):  val={vD:.4f}  diff={vA-vD:+.4f}

  NYQUIST PREDICTION:
    k=2 IS → 50 steps should = 100 standard (same rare obs)
    k=5 IS → 20 steps should = 100 standard (same rare obs)
    
  IF B ~ A: sampling rate controls convergence
    The 100-step floor is purely a sampling limit
    Importance sampling directly compresses it
    
  IF B > A (worse): importance sampling hurts
    Rare token oversampling distorts the gradient
    Common token statistics are equally important
    The 100 steps integrate ALL tokens, not just rare ones
    
  IF D < A: 2x IS improves quality in same steps
    Rare tokens benefit from more observations
    Valley 2 floor is lower with better rare token stats
""")
