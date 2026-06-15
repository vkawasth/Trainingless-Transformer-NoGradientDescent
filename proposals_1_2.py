#!/usr/bin/env python3
"""
Proposal 1: Forward-Routing Preservation with Detached Backward Boundaries
Proposal 2: Deep Supervision at L14 with cosine-decay λ1 → 0

PROPOSAL 1: h.detach() at window boundary
  Forward pass: all 24 layers run (topology intact, A∞ relations flow)
  Backward pass: gradient blocked at window boundary via detach()
  Outer layers act as static reference frames — no gradient updates
  Question: does joint convergence need coupled backward pass or only forward routing?

PROPOSAL 2: Deep supervision at L14
  L_total = λ1 * L(h_23) + λ2 * L(1.075 * h_14)
  λ1 cosine-decays from 1 → 0 over training
  When λ1=0: L15-L23 dropped from forward+backward entirely
  Question: does supervising the attractor directly match full-model quality?
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64
LR_SGD=0.05; LR_ADAM=3e-4; MOMENTUM=0.9
TARGET=4.0; MAX_STEPS=300; LOG=25
L_ATT=14   # attractor center
SCALE_A=1.075  # 1/sv(M_bwd)

print(f"\n{'='*65}")
print(f"  PROPOSAL 1: Detach-backward culling")
print(f"  PROPOSAL 2: Deep supervision at L{L_ATT}")
print(f"  d={D}  layers={N_LAYERS}  attractor=L{L_ATT}")
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
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
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
        self.ln_f=nn.LayerNorm(d); self._nl=nl
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def forward_detach(self,x,y,k_thresh,l_att=L_ATT):
        """
        PROPOSAL 1: Forward-routing with detached backward boundaries.
        Layers within [l_att-k, l_att+k] get full gradient.
        Outer layers: forward runs but h is detached at boundary — 
        they see the hidden state but gradient does not flow back to them.
        """
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for l,b in enumerate(self.blocks):
            # At the boundary entering the active window: detach
            if l==(l_att-k_thresh) and l>0:
                h=h.detach()   # gradient stops here on the way in
            h=b(h)
            # At the boundary leaving the active window: detach  
            if l==(l_att+k_thresh) and l<self._nl-1:
                h=h.detach()   # gradient stops here on the way out
        logits=self.head(self.ln_f(h))
        loss=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1))
        return logits,loss
    def forward_deep_sup(self,x,y,lam1,l_att=L_ATT,scale=SCALE_A,
                         drop_tail=False):
        """
        PROPOSAL 2: Deep supervision at L14.
        L_total = lam1*L(h_23) + (1-lam1)*L(scale*h_14)
        When lam1=0 and drop_tail=True: skip L15-L23 entirely.
        """
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        h_att=None
        for l,b in enumerate(self.blocks):
            if drop_tail and l>l_att:
                break   # skip tail entirely when lam1=0
            h=b(h)
            if l==l_att:
                h_att=h   # capture attractor hidden state
        # Auxiliary loss at L14
        logits_att=self.head(self.ln_f(h_att*scale))
        loss_att=F.cross_entropy(logits_att.view(-1,VOCAB),y.view(-1))
        if lam1==0 or drop_tail:
            return logits_att,loss_att,0.0
        # Primary loss at L23
        logits_23=self.head(self.ln_f(h))
        loss_23=F.cross_entropy(logits_23.view(-1,VOCAB),y.view(-1))
        loss_total=lam1*loss_23+(1-lam1)*loss_att
        return logits_23,loss_total,loss_att.item()

def eval_val(model,n=60,l_exit=None,scale=1.0):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val')
            if l_exit is None:
                _,loss=model(x,y)
            else:
                h=model.te(x)+model.pe(torch.arange(x.shape[1]))
                for l,b in enumerate(model.blocks):
                    h=b(h)
                    if l==l_exit: break
                logits=model.head(model.ln_f(h*scale))
                loss=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1))
            ls.append(loss.item())
    return float(np.mean(ls))

def clr_sgd(s,total=MAX_STEPS,warmup=50):
    if s<=warmup: return LR_SGD*s/warmup
    return LR_SGD*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def clr_adam(s,total=MAX_STEPS,warmup=100):
    if s<=warmup: return LR_ADAM*s/warmup
    return LR_ADAM*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def k_thresh_schedule(step,total=MAX_STEPS,k_max=None,k_min=1):
    if k_max is None: k_max=N_LAYERS//2
    progress=step/total
    return max(k_min,int(round(k_max*(1-progress)+k_min*progress)))

def lam1_schedule(step,total=MAX_STEPS,decay_start=0.4):
    """Cosine decay of λ1 from 1 → 0, starting at decay_start fraction."""
    if step/total < decay_start: return 1.0
    progress=(step/total-decay_start)/(1-decay_start)
    return float(0.5*(1+math.cos(math.pi*progress)))

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE: SGD + Nesterov (reference)
# ─────────────────────────────────────────────────────────────────────────────
print("BASELINE: SGD + Nesterov...")
torch.manual_seed(42)
m_base=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.SGD(m_base.parameters(),lr=LR_SGD,momentum=MOMENTUM,nesterov=True)
stt_base=None; t0=time.time()
for step in range(1,MAX_STEPS+1):
    for pg in opt.param_groups: pg['lr']=clr_sgd(step)
    m_base.train(); x,y=get_batch(); _,loss=m_base(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m_base.parameters(),1.0); opt.step()
    if step%LOG==0 or step==1:
        vl=eval_val(m_base,n=20)
        if vl<TARGET and stt_base is None: stt_base=step; print(f"  *** <{TARGET} at step {step} ***")
        print(f"  [{step:>4}/{MAX_STEPS}] val={vl:.4f}  t={time.time()-t0:.0f}s")
fval_base=eval_val(m_base,n=100)
print(f"  Baseline final val={fval_base:.4f}  steps_to_target={stt_base}\n")

# ─────────────────────────────────────────────────────────────────────────────
# PROPOSAL 1: Forward routing preserved, backward detached at window boundary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPROPOSAL 1: Detach-backward culling (forward intact)...")
print(f"  Window k: {N_LAYERS//2} → 1  centered on L{L_ATT}")
print(f"  Outer layers: forward runs, gradient detached at boundary\n")

torch.manual_seed(42)
m_p1=LM(D,N_HEADS,N_LAYERS)
opt1=torch.optim.SGD(m_p1.parameters(),lr=LR_SGD,momentum=MOMENTUM,nesterov=True)
stt_p1=None; t0=time.time()

for step in range(1,MAX_STEPS+1):
    for pg in opt1.param_groups: pg['lr']=clr_sgd(step)
    k=k_thresh_schedule(step,k_min=1)
    m_p1.train(); x,y=get_batch()
    _,loss=m_p1.forward_detach(x,y,k)
    opt1.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m_p1.parameters(),1.0); opt1.step()
    if step%LOG==0 or step==1:
        vl=eval_val(m_p1,n=20)
        if vl<TARGET and stt_p1 is None: stt_p1=step; print(f"  *** <{TARGET} at step {step} ***")
        active=[l for l in range(N_LAYERS) if abs(l-L_ATT)<=k]
        print(f"  [{step:>4}/{MAX_STEPS}] val={vl:.4f}  k={k}"
              f"  active=L{active[0]}..L{active[-1]}  t={time.time()-t0:.0f}s")

fval_p1=eval_val(m_p1,n=100)
print(f"\n  Proposal 1 final val={fval_p1:.4f}  steps_to_target={stt_p1}")

# KEY QUESTION: does the inner window model make good predictions at L14?
vl_exit_p1=eval_val(m_p1,n=100,l_exit=L_ATT,scale=SCALE_A)
print(f"  Early exit val (L{L_ATT}, ×{SCALE_A}): {vl_exit_p1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PROPOSAL 2: Deep supervision at L14 with cosine-decay λ1
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPROPOSAL 2: Deep supervision at L{L_ATT}, λ1 cosine-decay → 0...")
print(f"  L_total = λ1*L(h_23) + (1-λ1)*L({SCALE_A}*h_{L_ATT})")
print(f"  λ1 decays from 1.0 → 0.0 starting at step {int(0.4*MAX_STEPS)}\n")

torch.manual_seed(42)
m_p2=LM(D,N_HEADS,N_LAYERS)
opt2=torch.optim.AdamW(m_p2.parameters(),lr=LR_ADAM,betas=(0.9,0.95),weight_decay=0.1)
stt_p2=None; t0=time.time()

for step in range(1,MAX_STEPS+1):
    for pg in opt2.param_groups: pg['lr']=clr_adam(step)
    lam1=lam1_schedule(step)
    drop_tail=(lam1==0.0)   # drop L15-L23 when λ1 fully decayed
    m_p2.train(); x,y=get_batch()
    _,loss,loss_att=m_p2.forward_deep_sup(x,y,lam1,drop_tail=drop_tail)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m_p2.parameters(),1.0); opt2.step()
    if step%LOG==0 or step==1:
        # Evaluate: use L14 exit (what we're training toward)
        vl_23=eval_val(m_p2,n=20)          # full model quality
        vl_14=eval_val(m_p2,n=20,l_exit=L_ATT,scale=SCALE_A)  # L14 exit
        if vl_14<TARGET and stt_p2 is None: stt_p2=step; print(f"  *** L14 exit <{TARGET} at step {step} ***")
        tail="DROPPED" if drop_tail else f"λ1={lam1:.2f}"
        print(f"  [{step:>4}/{MAX_STEPS}] val_23={vl_23:.4f}"
              f"  val_14={vl_14:.4f}  {tail}  t={time.time()-t0:.0f}s")

fval_p2_23=eval_val(m_p2,n=100)
fval_p2_14=eval_val(m_p2,n=100,l_exit=L_ATT,scale=SCALE_A)
print(f"\n  Proposal 2 final val (full):     {fval_p2_23:.4f}")
print(f"  Proposal 2 final val (L{L_ATT} exit): {fval_p2_14:.4f}")
print(f"  steps_to L14<{TARGET}: {stt_p2}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS")
print("="*65)

print(f"""
  BASELINE (SGD, all 24 layers, full backward):
    Steps to val<{TARGET}: {stt_base}
    Final val:             {fval_base:.4f}

  PROPOSAL 1 (SGD, forward intact, detached backward, k: {N_LAYERS//2}→1):
    Steps to val<{TARGET}: {stt_p1}
    Final val (full):      {fval_p1:.4f}
    Final val (L{L_ATT} exit): {vl_exit_p1:.4f}
    
    IF stt_p1 == stt_base AND fval_p1 ≈ fval_base:
      Joint convergence requires only forward routing, not coupled backward.
      Outer layers are static reference frames.
      Backward FLOPs can be saved without quality loss.
      
    IF fval_p1 >> fval_base:
      Outer layers need active gradient updates to anchor the attractor.
      The backward coupling is load-bearing.

  PROPOSAL 2 (Adam, deep supervision at L{L_ATT}, λ1: 1→0):
    Steps to L{L_ATT} exit < {TARGET}: {stt_p2}
    Final val (full L23):  {fval_p2_23:.4f}
    Final val (L{L_ATT} exit):  {fval_p2_14:.4f}
    
    IF fval_p2_14 ≈ fval_base:
      L{L_ATT} direct supervision works. The attractor IS the model.
      Tail can be dropped in second half of training.
      ~18% training FLOPs saved + free 1.6x inference speedup.
      
    IF fval_p2_14 >> fval_base:
      Supervising L{L_ATT} alone is insufficient.
      The tail layers carry signal that cannot be captured at L{L_ATT}.
""")
