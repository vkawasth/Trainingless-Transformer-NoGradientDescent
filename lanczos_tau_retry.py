#!/usr/bin/env python3
"""
Lanczos τ-Retry Experiment
===========================
Tests replacing the 50CE τ-retry with Lanczos Newton.

CURRENT PIPELINE (175 CE):
  100CE@LR×5 (settle, plateau) → val≈0.20
  + 50CE@LR×2 (τ-retry, τ>5)  → val≈0.07
  + TopoGate                   → val≈0.07
  + LM at t=0                  → val≈0.06
  + 25CE cosine                → val≈0.047
  Total: 175 CE

TARGET PIPELINE (~133 CE equiv):
  100CE@LR×5 (settle, plateau) → val≈0.20
  + Lanczos (8 HVPs×4, 25 grad = ~64+50 fwd passes = ~57 CE equiv)
  + LM at t=0                  → val≈0.06
  + 25CE cosine                → val≈0.047
  Total: 100 + 57 + 25 = ~133 CE equiv (saves ~42 CE, no τ-retry)

LOADED STATE: basin_entry_state.pt (val≈0.20, saved pre-τ-retry)
"""
import json, math, warnings, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): sys.exit(f"ERROR: {f}")

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

def lanczos_newton(model, k=8, n_hvp=4, n_grad=25, mu=0.95, n_solves=3):
    """Lanczos Newton: k HVPs for basis, then n_solves cheap Newton steps."""
    n_p=sum(p.numel() for p in model.parameters())
    torch.manual_seed(7); q=torch.randn(n_p); q=q/q.norm()
    Q=[q]; alphas=[]; betas=[]
    def hvp(v):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n_hvp)]; loss=torch.stack(ls).mean()
        grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()
    for j in range(k):
        z=hvp(Q[j]); alpha=float((Q[j]*z).sum()); alphas.append(alpha)
        z=z-alpha*Q[j]
        if j>0: z=z-betas[-1]*Q[j-1]
        for qi in Q: z=z-float((qi*z).sum())*qi
        beta=float(z.norm()); betas.append(beta)
        if beta<1e-8: break
        Q.append(z/beta)
    n_l=len(alphas)
    T=torch.zeros(n_l,n_l)
    for i in range(n_l): T[i,i]=alphas[i]
    for i in range(n_l-1): T[i,i+1]=betas[i]; T[i+1,i]=betas[i]
    T_ev,T_evec=torch.linalg.eigh(T); V=torch.stack(Q[:n_l],dim=1)@T_evec
    evals_str=[f'{e:.3f}' for e in T_ev.tolist()]
    cond=float(T_ev[-1].abs()/max(T_ev[0].abs(),1e-8))

    results=[]
    for si in range(n_solves):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n_grad)]; torch.stack(ls).mean().backward()
        g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                     for p in model.parameters()]).detach(); model.zero_grad()
        g_proj=V.T@g; d_proj=g_proj/(T_ev+mu)
        g_res=g-V@(V.T@g); d=-(V@d_proj+g_res/mu)
        w0=model.flat_params(); v0=eval_val(model,n=8)
        model.set_flat(w0+d); v1=eval_val(model,n=8)
        if v1<v0:
            results.append((v0,v1,'✓'))
            print(f"    Solve {si+1}: {v0:.4f}→{v1:.4f}  Δ={v0-v1:.4f}")
        else:
            model.set_flat(w0)
            results.append((v0,v0,'~'))
            print(f"    Solve {si+1}: no gain (val={v0:.4f})")
            break
    return eval_val(model,n=12), cond, evals_str

# ── LOAD BASIN ENTRY STATE ────────────────────────────────────
print("="*65)
print("LANCZOS τ-RETRY EXPERIMENT")
print("Target: 100CE + Lanczos + LM + 25CE ≈ 133 CE equiv")
print("="*65); print()

ENTRY_STATE='basin_entry_state.pt'
if not os.path.exists(ENTRY_STATE):
    sys.exit(f"ERROR: {ENTRY_STATE} not found. Run compiler_geometric.py first.")

# Load basin entry state (val≈0.20, saved after plateau before τ-retry)
m_base=LM(); m_base.load_state_dict(torch.load(ENTRY_STATE,map_location='cpu'))
v_entry=eval_val(m_base,n=20)
print(f"Basin entry state: val={v_entry:.4f}")
print(f"Expected: val≈0.20 (post-plateau, pre-τ-retry)")
if v_entry > 0.35:
    print(f"⚠ val={v_entry:.4f} too high — may not be the right state")
    print(f"  Run compiler_geometric.py to regenerate basin_entry_state.pt")
print()

# ── REFERENCE: 50CE τ-RETRY (current approach) ───────────────
print("━━━ REFERENCE: 50CE@LR×2 τ-retry (current) ━━━━━━━━━━━━━")
t_ref=time.time()
m_ref=copy.deepcopy(m_base)
opt_r=torch.optim.AdamW(m_ref.parameters(),lr=LR*2,betas=(0.9,0.95),weight_decay=0.1)
for step in range(50):
    m_ref.train(); x,y=get_batch(); _,l=m_ref(x,y)
    opt_r.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_ref.parameters(),1.0); opt_r.step()
v_50ce=eval_val(m_ref,n=15)
print(f"  After 50CE@LR×2: val={v_50ce:.4f}  [{time.time()-t_ref:.1f}s]")

# Continue with LM + 25CE cosine
v_lm_ref,acc=lm_step(m_ref)
print(f"  After LM:        val={v_lm_ref:.4f}  {'✓' if acc else '~'}")
opt_c=torch.optim.AdamW(m_ref.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt_c.param_groups: pg['lr']=LR*0.5*(1+math.cos(math.pi*step/25))
    m_ref.train(); x,y=get_batch(); _,l=m_ref(x,y)
    opt_c.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_ref.parameters(),1.0); opt_c.step()
v_ref_final=eval_val(m_ref,n=20)
t_ref_total=time.time()-t_ref
print(f"  After 25CE cos:  val={v_ref_final:.4f}  [{t_ref_total:.1f}s total]")
print(f"  CE cost: 50 + LM(~170equiv) + 25 = ~245 CE equiv")
print()

# ── EXPERIMENT A: Lanczos (3 solves, k=8) ────────────────────
print("━━━ EXPERIMENT A: Lanczos (k=8, 3 solves) ━━━━━━━━━━━━━━")
t_a=time.time()
m_a=copy.deepcopy(m_base)
v_a, cond_a, evals_a=lanczos_newton(m_a, k=8, n_hvp=4, n_grad=25, mu=0.95, n_solves=3)
print(f"  After Lanczos:   val={v_a:.4f}  cond={cond_a:.1f}  [{time.time()-t_a:.1f}s]")
print(f"  Eigenvalues: {evals_a}")
print(f"  Cost: 8 HVPs×4 + 3×25 grad = 32+75 = ~54 CE equiv")

v_lm_a,acc_a=lm_step(m_a)
print(f"  After LM:        val={v_lm_a:.4f}  {'✓' if acc_a else '~'}")
opt_a2=torch.optim.AdamW(m_a.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt_a2.param_groups: pg['lr']=LR*0.5*(1+math.cos(math.pi*step/25))
    m_a.train(); x,y=get_batch(); _,l=m_a(x,y)
    opt_a2.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_a.parameters(),1.0); opt_a2.step()
v_a_final=eval_val(m_a,n=20)
t_a_total=time.time()-t_a
print(f"  After 25CE cos:  val={v_a_final:.4f}  [{t_a_total:.1f}s total]")
print()

# ── EXPERIMENT B: More Lanczos solves (k=8, 6 solves) ────────
print("━━━ EXPERIMENT B: Lanczos (k=8, 6 solves) ━━━━━━━━━━━━━━")
t_b=time.time()
m_b=copy.deepcopy(m_base)
v_b, cond_b, _=lanczos_newton(m_b, k=8, n_hvp=4, n_grad=25, mu=0.95, n_solves=6)
print(f"  After Lanczos:   val={v_b:.4f}  cond={cond_b:.1f}  [{time.time()-t_b:.1f}s]")
print(f"  Cost: 8 HVPs×4 + 6×25 grad = 32+150 = ~91 CE equiv")

v_lm_b,acc_b=lm_step(m_b)
print(f"  After LM:        val={v_lm_b:.4f}  {'✓' if acc_b else '~'}")
opt_b2=torch.optim.AdamW(m_b.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt_b2.param_groups: pg['lr']=LR*0.5*(1+math.cos(math.pi*step/25))
    m_b.train(); x,y=get_batch(); _,l=m_b(x,y)
    opt_b2.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_b.parameters(),1.0); opt_b2.step()
v_b_final=eval_val(m_b,n=20)
t_b_total=time.time()-t_b
print(f"  After 25CE cos:  val={v_b_final:.4f}  [{t_b_total:.1f}s total]")
print()

# ── EXPERIMENT C: Larger k (k=16) ────────────────────────────
print("━━━ EXPERIMENT C: Lanczos (k=16, 3 solves) ━━━━━━━━━━━━━")
t_c=time.time()
m_c=copy.deepcopy(m_base)
v_c, cond_c, _=lanczos_newton(m_c, k=16, n_hvp=4, n_grad=25, mu=0.95, n_solves=3)
print(f"  After Lanczos:   val={v_c:.4f}  cond={cond_c:.1f}  [{time.time()-t_c:.1f}s]")
print(f"  Cost: 16 HVPs×4 + 3×25 grad = 64+75 = ~70 CE equiv")

v_lm_c,acc_c=lm_step(m_c)
print(f"  After LM:        val={v_lm_c:.4f}  {'✓' if acc_c else '~'}")
opt_c2=torch.optim.AdamW(m_c.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt_c2.param_groups: pg['lr']=LR*0.5*(1+math.cos(math.pi*step/25))
    m_c.train(); x,y=get_batch(); _,l=m_c(x,y)
    opt_c2.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_c.parameters(),1.0); opt_c2.step()
v_c_final=eval_val(m_c,n=20)
t_c_total=time.time()-t_c
print(f"  After 25CE cos:  val={v_c_final:.4f}  [{t_c_total:.1f}s total]")
print()

# ── SUMMARY ──────────────────────────────────────────────────
print("="*65)
print("RESULTS — Lanczos vs 50CE τ-retry")
print("="*65)
print()
print(f"  Entry state:  val={v_entry:.4f}")
print()
print(f"  {'Method':<35} {'val_after_Lanczos':>17} {'final_val':>9} {'CE_equiv':>9}")
print("  "+"-"*72)
print(f"  {'50CE@LR×2 + LM + 25CE cos (current)':<35} {v_50ce:>17.4f} {v_ref_final:>9.4f} {'~245':>9}")
print(f"  {'Lanczos k=8, 3 solves + LM + 25CE':<35} {v_a:>17.4f} {v_a_final:>9.4f} {'~79':>9}")
print(f"  {'Lanczos k=8, 6 solves + LM + 25CE':<35} {v_b:>17.4f} {v_b_final:>9.4f} {'~116':>9}")
print(f"  {'Lanczos k=16, 3 solves + LM + 25CE':<35} {v_c:>17.4f} {v_c_final:>9.4f} {'~95':>9}")
print()
best_lanczos=min(v_a_final,v_b_final,v_c_final)
best_name='A(k=8,3)' if best_lanczos==v_a_final else 'B(k=8,6)' if best_lanczos==v_b_final else 'C(k=16,3)'
print(f"  Best Lanczos:  val={best_lanczos:.4f}  ({best_name})")
print(f"  Reference:     val={v_ref_final:.4f}  (50CE τ-retry)")
print()
if best_lanczos <= v_ref_final + 0.005:
    saving=245-({'A(k=8,3)':79,'B(k=8,6)':116,'C(k=16,3)':95}[best_name])
    print(f"  ✓ Lanczos MATCHES 50CE τ-retry, saves ~{saving} CE equiv")
    print(f"  → Replace τ-retry with Lanczos in compiler_geometric")
else:
    gap=best_lanczos-v_ref_final
    print(f"  Gap: {gap:+.4f} nats remaining")
    print(f"  Lanczos descends faster but 50CE integrates more rare-token stats")
