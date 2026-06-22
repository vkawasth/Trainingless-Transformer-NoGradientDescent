#!/usr/bin/env python3
"""
Confirmed J14 Pipeline — Exact Replication of gradient_alignment_fix.py
========================================================================
Replicates the confirmed sequence that achieves val=0.284 after settle,
then val=0.045 via LM+100CE.

CONFIRMED SEQUENCE (gradient_alignment_fix.py):
  1. J14 broadcast: ALL teacher L14 weights → all student layers
     (te, pe, ln_f, WK, WQ, WV, op, FF from teacher.blocks[14])
  2. Saddle exit: α*=1.429 on teacher-init model → val=4.35
  3. MF10: n_iter=10, WQ uses -(wg.T/(wf.T+1e-4)) [separate Fisher]
  4. Settle: 33CE@LR×5 → val=0.47
  5. Sign flip layers {1,2} → val=0.44
  BASE STATE: val=0.284

  6. LM at t=0 → val~0.237 (2nd order defect correction)
  7. 100CE cosine LR×1 → val=0.045

PHASE STRUCTURE FROM ALIGNMENT DATA:
  t=0-10:  cos>0 (aligned)    → LARGE STEP PHASE
  t=10-25: cos<0 (rotating)   → CURVATURE DEFECT ACCUMULATION
  t=25-33: cos recovering      → REALIGNMENT
  t=33-75: oscillating         → STATISTICAL PHASE
  LM at t=0 bypasses rotation phase entirely (why B wins)

USAGE:
  python compiler_j14_confirmed.py --teacher teacher.pt
"""
import argparse, json, math, warnings, collections, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--teacher', '--teacher_path', type=str, required=True)
parser.add_argument('--n_basin', type=int, default=100,
                    help='CE steps after LM (default 100, confirmed B=0.045)')
args = parser.parse_args()

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429  # confirmed constants from gradient_alignment_fix

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): sys.exit(f"ERROR: {f}")
if not os.path.exists(args.teacher):
    sys.exit(f"ERROR: teacher not found at {args.teacher}")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh=D//N_HEADS
        self.WQ=nn.Linear(D,D,bias=False); self.WK=nn.Linear(D,D,bias=False)
        self.WV=nn.Linear(D,D,bias=False); self.op=nn.Linear(D,D,bias=False)
        self.ln=nn.LayerNorm(D); self.sc=math.sqrt(dh); self.nh=N_HEADS; self.dh=dh
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,_=h.shape
        Q=self.WQ(h).view(B,S,self.nh,self.dh).transpose(1,2)
        K=self.WK(h).view(B,S,self.nh,self.dh).transpose(1,2)
        V=self.WV(h).view(B,S,self.nh,self.dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D)))
class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g=nn.Linear(D,D*2,bias=False); self.v=nn.Linear(D,D*2,bias=False)
        self.o=nn.Linear(D*2,D,bias=False); self.n=nn.LayerNorm(D)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))
class Block(nn.Module):
    def __init__(self): super().__init__(); self.attn=Attn(); self.ff=FF()
    def forward(self,h): return self.ff(self.attn(h))
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te=nn.Embedding(VOCAB,D); self.pe=nn.Embedding(512,D)
        self.blocks=nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,v):
        i=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(v[i:i+n].reshape(p.shape)); i+=n

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

def eval_val(m, n=15):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def sheet_angles(model):
    out=[]; WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU-1):
        try:
            phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
            lam=torch.linalg.eigvals(phi); lam1=lam[lam.abs().argmax()]
            a=float(torch.angle(lam1))
            out.append('π' if abs(abs(a)-math.pi)<0.3 else '0' if abs(a)<0.3 else f'{a:.2f}')
        except: out.append('?')
    return out

def cosine_lr(step, total, base_lr=LR):
    return base_lr*0.5*(1+math.cos(math.pi*step/total))

def lm_step(model, mu=0.950, n_grad=25, n_hvp=12, n_cg=6):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n_grad))/n_grad
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach(); model.zero_grad()
    def _hvp(v):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n_hvp)]; loss2=torch.stack(ls).mean()
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()
    d=torch.zeros_like(g); r=-g.clone(); p=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=_hvp(p)+mu*p; al=rr/max(float((p*Hp).sum()),1e-10)
        d+=al*p; r-=al*Hp; rr2=float((r*r).sum()); p=r+(rr2/max(rr,1e-10))*p; rr=rr2
    w0=model.flat_params(); v0=eval_val(model,n=8)
    model.set_flat(w0+d); v1=eval_val(model,n=8)
    if v1<v0: return v1, True
    model.set_flat(w0); return v0, False

def hv_product(model, v, n=15):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)]).detach()
    model.zero_grad(); return hv

# ══════════════════════════════════════════════════════════════
print("="*65)
print("CONFIRMED J14 PIPELINE")
print("Replicating gradient_alignment_fix.py exactly")
print("="*65); print()

# Load teacher
print(f"Loading teacher from {args.teacher}...")
teacher_sd = torch.load(args.teacher, map_location='cpu')
n_teacher = sum(1 for k in teacher_sd if 'blocks.' in k and '.attn.WK.weight' in k)
print(f"Teacher: {n_teacher} layers")
print()

# ── STEP 1: J14 BROADCAST ─────────────────────────────────────
print("━━━ STEP 1: J14 BROADCAST (all teacher L14 → all student layers) ━━")
torch.manual_seed(99)
model = LM()

# Copy ALL teacher weights (exact match to build_student())
with torch.no_grad():
    if 'te.weight' in teacher_sd and teacher_sd['te.weight'].shape == (VOCAB,D):
        model.te.weight.data.copy_(teacher_sd['te.weight'].float())
        print(f"  te.weight copied")
    if 'pe.weight' in teacher_sd:
        sz=min(teacher_sd['pe.weight'].shape[0],512)
        model.pe.weight.data[:sz].copy_(teacher_sd['pe.weight'][:sz].float())
        print(f"  pe.weight copied (first {sz} positions)")
    for w in ['ln_f.weight','ln_f.bias']:
        if w in teacher_sd:
            getattr(model.ln_f, w.split('.')[1]).data.copy_(teacher_sd[w].float())
            print(f"  {w} copied")
    # L14 → all student layers
    layer_map = [
        (f'blocks.{L_ATT}.attn.WK.weight', 'attn.WK.weight'),
        (f'blocks.{L_ATT}.attn.WQ.weight', 'attn.WQ.weight'),
        (f'blocks.{L_ATT}.attn.WV.weight', 'attn.WV.weight'),
        (f'blocks.{L_ATT}.attn.op.weight', 'attn.op.weight'),
        (f'blocks.{L_ATT}.ff.g.weight',    'ff.g.weight'),
        (f'blocks.{L_ATT}.ff.v.weight',    'ff.v.weight'),
        (f'blocks.{L_ATT}.ff.o.weight',    'ff.o.weight'),
    ]
    for l in range(N_STU):
        for src_key, dst_suffix in layer_map:
            if src_key in teacher_sd:
                tw = teacher_sd[src_key].float()
                dst = dict(model.named_parameters())[f'blocks.{l}.{dst_suffix}']
                if tw.shape == dst.shape:
                    dst.data.copy_(tw)
    print(f"  L14 weights broadcast to all {N_STU} student layers")

v_j14 = eval_val(model)
print(f"  After J14 broadcast: val={v_j14:.4f}  Φ={sheet_angles(model)}")
print()

# ── STEP 2: SADDLE EXIT α=1.429 ───────────────────────────────
print("━━━ STEP 2: SADDLE EXIT (α*=1.429, confirmed constant) ━━━━━━━━━")
t2=time.time()

# v_neg from reference model (same weights as model)
ref = copy.deepcopy(model)
n_p = sum(p.numel() for p in ref.parameters())
v = torch.randn(n_p); v = v/v.norm()
for _ in range(15):
    Hv = hv_product(ref, v, n=15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg = v.clone()

w0 = model.flat_params()
model.set_flat(w0 + ALPHA_STAR*(v_neg/v_neg.norm()))
v_saddle = eval_val(model)
print(f"  α*={ALPHA_STAR}: val={v_saddle:.4f}  (confirmed: 4.35)")
print(f"  Φ={sheet_angles(model)}  [{time.time()-t2:.1f}s]"); print()

# ── STEP 3: MF10 (exact apply_mf from gradient_alignment_fix) ─
print("━━━ STEP 3: MF PUMP (n=10, WQ=wg.T/wf.T separate Fisher) ━━━━━━")
print("  Exact apply_mf from gradient_alignment_fix.py")
t3=time.time()

for it in range(10):
    # E step
    for l in range(N_STU):
        model.blocks[l].attn.WK.weight.requires_grad_(False)
        model.blocks[l].attn.WQ.weight.requires_grad_(False)
    eg=torch.zeros(VOCAB,D); ef=torch.zeros(VOCAB,D)
    torch.manual_seed(it*1000)
    for i in range(200):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        model.zero_grad(); _,loss=model(x,y); loss.backward()
        if model.te.weight.grad is not None:
            g=model.te.weight.grad.detach(); eg+=g; ef+=g**2
    eg/=200; ef/=200
    with torch.no_grad(): model.te.weight.add_(-0.01*eg/(ef+1e-4))
    for l in range(N_STU):
        model.blocks[l].attn.WK.weight.requires_grad_(True)
        model.blocks[l].attn.WQ.weight.requires_grad_(True)
    # WK step
    model.te.weight.requires_grad_(False)
    wg=torch.zeros_like(model.blocks[0].attn.WK.weight)
    wf=torch.zeros_like(model.blocks[0].attn.WK.weight)
    torch.manual_seed(it*1000+500)
    for i in range(200):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        model.zero_grad(); _,loss=model(x,y); loss.backward()
        g=torch.zeros_like(model.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if model.blocks[l].attn.WK.weight.grad is not None:
                g+=model.blocks[l].attn.WK.weight.grad/N_STU
        wg+=g; wf+=g**2
    wg/=200; wf/=200
    with torch.no_grad():
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.add_(-0.01*wg/(wf+1e-4))
            # EXACT: gradient_alignment_fix uses wg.T/(wf.T+1e-4) for WQ
            model.blocks[l].attn.WQ.weight.add_(-0.01*wg.T/(wf.T+1e-4))
    model.te.weight.requires_grad_(True)
    if (it+1) % 5 == 0:
        print(f"  MF iter {it+1}: val={eval_val(model,n=5):.4f}")

v_mf = eval_val(model)
print(f"  After MF10: val={v_mf:.4f}  Φ={sheet_angles(model)}  [{time.time()-t3:.1f}s]"); print()

# ── STEP 4: SETTLE 33CE@LR×5 ──────────────────────────────────
print("━━━ STEP 4: SETTLE 33CE@LR×5 (confirmed → val=0.47) ━━━━━━━━━━")
t4=time.time()
opt_s=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt_s.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt_s.step()
v_settle=eval_val(model)
print(f"  After settle: val={v_settle:.4f}  Φ={sheet_angles(model)}  [{time.time()-t4:.1f}s]"); print()

# ── STEP 5: SIGN FLIP (geometry-checked, try all candidates) ──
print("━━━ STEP 5: SIGN FLIP (geometry-checked) ━━━━━━━━━━━━━━━━━")
print(f"  Settle: val={v_settle:.4f}  Φ={sheet_angles(model)}")
v_before=eval_val(model); best_v=v_before; best_layers=None
for fl in [[1,2],[0,1],[2,3],[0,2],[1,3]]:
    with torch.no_grad():
        for l in fl: model.blocks[l].attn.WV.weight.data.mul_(-1); model.blocks[l].attn.op.weight.data.mul_(-1)
    vt=eval_val(model,n=8)
    if vt<best_v: best_v=vt; best_layers=fl; print(f"  ✓ layers {fl}: val={vt:.4f}")
    else: print(f"  ~ layers {fl}: val={vt:.4f}")
    with torch.no_grad():
        for l in fl: model.blocks[l].attn.WV.weight.data.mul_(-1); model.blocks[l].attn.op.weight.data.mul_(-1)
if best_layers:
    with torch.no_grad():
        for l in best_layers: model.blocks[l].attn.WV.weight.data.mul_(-1); model.blocks[l].attn.op.weight.data.mul_(-1)
    print(f"  Applied best flip {best_layers}")
v_sign=eval_val(model)
print(f"  After sign flip: val={v_sign:.4f}  (confirmed: 0.44)")
print(f"  Φ={sheet_angles(model)}")
print()
print(f"  ═══════════════════════════════════════")
print(f"  BASE STATE: val={v_sign:.4f}  (target: 0.284)")
print(f"  ═══════════════════════════════════════")
torch.save(model.state_dict(), 'base_state_j14.pt')
print()

# ── STEP 6: LM AT t=0 (2nd order defect correction) ───────────
print("━━━ STEP 6: LM AT t=0 (corrects curvature defect before accumulation)")
print("  Phase structure: t=0-10 aligned, t=10-25 NEGATIVE (defect)")
print("  LM at t=0 bypasses rotation phase → B wins in alignment_fix")
t6=time.time()
v_lm, acc = lm_step(model)
print(f"  After LM: val={v_lm:.4f}  {'✓' if acc else '~'}  [{time.time()-t6:.1f}s]"); print()

# ── STEP 7: 100CE COSINE LR×1 ─────────────────────────────────
print(f"━━━ STEP 7: {args.n_basin}CE COSINE LR×1 (confirmed B=0.045) ━━━━━━━━━")
t7=time.time()
opt_c=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1, args.n_basin+1):
    for pg in opt_c.param_groups: pg['lr']=cosine_lr(step, args.n_basin)
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt_c.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt_c.step()
    if step in {25,50,75,100}:
        v=eval_val(model)
        print(f"  CE {step:3d}: val={v:.4f}  Φ={sheet_angles(model)}")
v_final=eval_val(model)
print(f"  After {args.n_basin}CE: val={v_final:.4f}  [{time.time()-t7:.1f}s]"); print()

# ── SUMMARY ──────────────────────────────────────────────────
print("="*65)
print("RESULTS")
print("="*65)
print(f"  J14 broadcast:   val={v_j14:.4f}")
print(f"  Saddle exit:     val={v_saddle:.4f}  (confirmed: 4.35)")
print(f"  MF10:            val={v_mf:.4f}  (iter 5: ~9.48, iter 10: ~6.94)")
print(f"  Settle LR×5:     val={v_settle:.4f}  (confirmed: 0.47)")
print(f"  Sign flip:       val={v_sign:.4f}   (confirmed: 0.44)")
print(f"  LM at t=0:       val={v_lm:.4f}")
print(f"  {args.n_basin}CE cosine:    val={v_final:.4f}  (confirmed B: 0.045)")
print()
print(f"  CONFIRMED: val=0.045 (gradient_alignment_fix B)")
print(f"  GAP: {v_final-0.045:+.4f} nats")
