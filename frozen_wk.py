#!/usr/bin/env python3
"""
Frozen WK Experiment
=====================
D showed: teacher WK direct + 200CE = val 0.161 (best result).

The 200 CE steps with teacher WK are adapting WV, WO, FF, E
to work with the fixed WK. Test: freeze WK at teacher values
and train only the other weights.

If frozen WK + 200CE ~ teacher WK direct + 200CE:
  The WK is the endpoint. The other weights need corpus adaptation.
  Total compute: 0 steps for WK (known) + N steps for others.
  
How many steps do the other weights need?
Test: frozen WK + 25/50/100/200CE.
Find minimum CE steps to beat teacher val.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14

print(f"\n{'='*65}")
print(f"  FROZEN WK EXPERIMENT")
print(f"  WK=teacher (endpoint), train WV/WO/FF/E only")
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

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

# Train teacher
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

def build_student_frozen_wk():
    """Student with teacher WK/WQ frozen, everything else from teacher L14."""
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            # WK, WQ from teacher L14 (the endpoint) — will be FROZEN
            stu.blocks[l].attn.WK.weight.copy_(teacher.blocks[L_ATT].attn.WK.weight)
            stu.blocks[l].attn.WQ.weight.copy_(teacher.blocks[L_ATT].attn.WQ.weight)
            # WV, WO, FF from teacher L14 — will be TRAINABLE
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    # Freeze WK and WQ
    for l in range(N_STU):
        stu.blocks[l].attn.WK.weight.requires_grad_(False)
        stu.blocks[l].attn.WQ.weight.requires_grad_(False)
    return stu

def run(build_fn, label, steps=200, freeze_wk=False):
    stu=build_fn()
    v0=eval_val(stu,n=30)
    n_trainable=sum(p.numel() for p in stu.parameters() if p.requires_grad)
    n_total=sum(p.numel() for p in stu.parameters())
    print(f"\n  [{label}] zero-shot={v0:.4f}  "
          f"trainable={n_trainable}/{n_total} ({100*n_trainable/n_total:.0f}%)")
    if steps==0: return v0,{0:v0}
    trainable=[p for p in stu.parameters() if p.requires_grad]
    opt_s=torch.optim.AdamW(trainable,lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [10,25,50,75,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

# Baseline: prime cascade + 200CE (from previous results)
# A: Teacher WK (unfrozen) + 200CE  [D from direct_endpoint]
# B: Teacher WK FROZEN + 200CE  [only WV/WO/FF/E train]
# C: Teacher WK FROZEN + 50CE   [minimal]
# D: Teacher WK FROZEN + 25CE   [very minimal]

def build_unfrozen():
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

print(f"\n{'='*65}")
print("EXPERIMENTS")
print("  A: Teacher WK (all unfrozen) + 200CE")
print("  B: Teacher WK FROZEN + 200CE (WV/WO/FF/E only)")
print("  C: Teacher WK FROZEN + 100CE")
print("  D: Teacher WK FROZEN + 50CE")
print("  E: Teacher WK FROZEN + 25CE")
print("="*65)

vA,ckA=run(build_unfrozen,"A-TeachWK-unfrozen",steps=200)
vB,ckB=run(build_student_frozen_wk,"B-TeachWK-FROZEN-200CE",steps=200)
vC,ckC=run(build_student_frozen_wk,"C-TeachWK-FROZEN-100CE",steps=100)
vD,ckD=run(build_student_frozen_wk,"D-TeachWK-FROZEN-50CE",steps=50)
vE,ckE=run(build_student_frozen_wk,"E-TeachWK-FROZEN-25CE",steps=25)

print(f"\n{'='*65}")
print("  FROZEN WK RESULTS")
print("="*65)
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-unfrz':>8}  {'B-frz200':>9}  {'C-frz100':>9}  "
      f"{'D-frz50':>8}  {'E-frz25':>8}")
for s in [0,10,25,50,75,100,150,200]:
    row=f"  {s:>6}"
    for ck in [ckA,ckB,ckC,ckD,ckE]:
        v=ck.get(s)
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    print(row)

print(f"""
  FINAL:
    Teacher:                   val={val_teacher:.4f}
    A (WK unfrozen+200CE):     val={vA:.4f}
    B (WK frozen+200CE):       val={vB:.4f}  diff A-B={vA-vB:+.4f}
    C (WK frozen+100CE):       val={vC:.4f}
    D (WK frozen+50CE):        val={vD:.4f}
    E (WK frozen+25CE):        val={vE:.4f}

  THE QUESTION:
    How many CE steps are needed when WK is already at endpoint?
    
    If B ~ A: freezing WK costs nothing.
      The 200 CE steps were ONLY adapting WV/WO/FF/E.
      WK was a passenger — gradient was not needed for WK.
      
    If B < A significantly: frozen WK is better.
      Allowing WK to drift during CE training hurts quality.
      The endpoint WK should be FIXED — only adapt everything else.
      
    Minimum steps (C, D, E): find when frozen WK beats teacher val.
      This is the irreducible cost of embedding co-adaptation.
      If E (25 steps) beats teacher: only 25 steps needed.
      These 25 steps are NOT for the MC element — they adapt
      WV, WO, FF, and embedding to work with the teacher WK.
""")
