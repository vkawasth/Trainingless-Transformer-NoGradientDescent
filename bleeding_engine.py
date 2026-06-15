#!/usr/bin/env python3
"""
Residual Bleeding Engine — Proper Validation
=============================================
Tests H_out = α * H_blocks + (1-α) * H_emb across α in {0, 0.05, 0.1, 0.2, 0.5, 1.0}

α=0: pure embeddings → linear classifier on embeddings only
α=1: pure blocks → standard 2L student (random blocks)
α=0.2: 80% embeddings + 20% block output

Compared against:
  - Trained 24L teacher (val≈0.25)
  - 2L student trained end-to-end with CE (val≈0.86)

The question: does any α improve on α=0 (pure embeddings)?
If not: the blocks add noise and the embeddings are doing all the work.
If yes: the block output carries signal that improves over raw embeddings.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
HEAD_STEPS=100

print(f"\n{'='*65}")
print(f"  RESIDUAL BLEEDING ENGINE — ALPHA SWEEP")
print(f"  α=0: pure embeddings  α=1: pure 2L blocks")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
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
    def get_bleed(self,x,alpha):
        h_emb=self.te(x)+self.pe(torch.arange(x.shape[1]))
        h=h_emb
        for b in self.blocks: h=b(h)
        return self.ln_f(alpha*h + (1-alpha)*h_emb)

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def eval_head(model, alpha, head, n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val')
            h=model.get_bleed(x,alpha)
            logits=head(h)
            ls.append(F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)).item())
    return float(np.mean(ls))

def train_head(model, alpha, steps=HEAD_STEPS):
    """Train a fresh linear head on blended hidden states. Blocks frozen."""
    head=nn.Linear(D,VOCAB,bias=False)
    head.weight.data.copy_(model.te.weight.data)  # init from embeddings
    for p in model.parameters(): p.requires_grad_(False)
    head.weight.requires_grad_(True)
    opt=torch.optim.AdamW([head.weight],lr=LR*3,betas=(0.9,0.95),weight_decay=0.01)
    model.eval()
    for _ in range(steps):
        x,y=get_batch()
        with torch.no_grad(): h=model.get_bleed(x,alpha)
        logits=head(h)
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters(): p.requires_grad_(True)
    return head

# ── Train teacher ─────────────────────────────────────────────────────────────
print("Step 1: Train 24L teacher (300 steps, proper schedule)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        vl=eval_val(teacher,n=20)
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
teacher.eval()
val_teacher=eval_val(teacher)
print(f"  Teacher: val={val_teacher:.4f}\n")

# ── Build 2L student with teacher embeddings ──────────────────────────────────
print("Step 2: Init 2L student with teacher embeddings...")
torch.manual_seed(99)
student=LM(D,N_HEADS,2)
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(student,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
print("  Done.\n")

# ── Train 2L end-to-end (CE baseline) ────────────────────────────────────────
print("Step 3: Train 2L student end-to-end CE baseline (200 steps)...")
torch.manual_seed(99)
student_ce=LM(D,N_HEADS,2)
for attr in ['te','pe','ln_f']:
    src=getattr(teacher,attr); dst=getattr(student_ce,attr)
    if hasattr(src,'weight'): dst.weight.data.copy_(src.weight.data)
    if hasattr(src,'bias') and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
opt_ce=torch.optim.AdamW(student_ce.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,201):
    for pg in opt_ce.param_groups: pg['lr']=clr(step,200)
    student_ce.train(); x,y=get_batch(); _,loss=student_ce(x,y)
    opt_ce.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(student_ce.parameters(),1.0); opt_ce.step()
student_ce.eval()
val_ce=eval_val(student_ce)
print(f"  2L CE: val={val_ce:.4f}\n")

# ── Alpha sweep ───────────────────────────────────────────────────────────────
print(f"Step 4: Alpha sweep — train {HEAD_STEPS}-step head for each α...")
print(f"  α=0: pure embeddings  α=1: pure block output (random blocks)")
print()

ALPHAS=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
results=[]

for alpha in ALPHAS:
    t0=time.time()
    head=train_head(student, alpha, HEAD_STEPS)
    vl=eval_head(student, alpha, head)
    results.append((alpha, vl))
    label = "pure embeddings" if alpha==0 else ("pure blocks (random)" if alpha==1 else f"blend")
    print(f"  α={alpha:.2f}: val={vl:.4f}  ({label})  t={time.time()-t0:.0f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS")
print("="*65)
print(f"\n  Teacher (24L, trained):          val={val_teacher:.4f}")
print(f"  Student 2L CE (trained):          val={val_ce:.4f}")
print(f"\n  Alpha sweep (frozen random blocks + fresh head, {HEAD_STEPS} steps):")
print(f"  {'α':>6}  {'val':>8}  {'vs α=0':>10}  note")
print("  "+"-"*45)
base_val=results[0][1]
for alpha,vl in results:
    delta=vl-base_val
    note=""
    if alpha==0: note="← pure embeddings (no blocks)"
    elif alpha==1: note="← pure blocks (random)"
    elif alpha==0.2: note="← claimed optimal"
    print(f"  {alpha:>6.2f}  {vl:>8.4f}  {delta:>+10.4f}  {note}")

best_alpha,best_val=min(results,key=lambda x:x[1])
print(f"""
  KEY READING:

  If val INCREASES monotonically with α (best at α=0):
    The blocks add noise. Raw embeddings + linear head is optimal.
    The "bleeding engine" works by minimizing block influence.
    There is no geometric bypass mechanism — just embedding quality.

  If val has a MINIMUM at some α > 0 (e.g. α=0.2 is best):
    The block output carries signal beyond the raw embeddings.
    The random blocks improve representations despite random init.
    This would be the genuine residual bypass mechanism.

  Best α: {best_alpha}  val={best_val:.4f}
  Baseline (α=0, pure embeddings): val={base_val:.4f}
  
  Gap teacher→best: {best_val-val_teacher:.4f} nats
  Gap CE student→best: {best_val-val_ce:.4f} nats (negative=worse)
""")
