"""
PER-PARAMETER HINDSIGHT CULL.
For each coordinate i, sgn(D_i) is the direction it must end up moving.
Arm A: zero any step component that opposes sgn(D_i)   (cull the back-and-forth)
Arm B: keep ONLY the opposing components                (keep only wandering)
Both use hindsight (D from a first pass) and recompute gradients as they go.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def V(n=16): return float(eval_val(model,n=n))
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
model.load_state_dict(torch.load("init.pt")); th0=flat(); v0=V()
torch.manual_seed(17); opt=newopt()
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
D=flat()-th0; sg=torch.sign(D); v_full=V()
print(f"  baseline: {v0:.4f} -> {v_full:.4f}   ({time.time()-t0:.0f}s)", flush=True)
def rerun(mode):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
    o=newopt(); prev=flat(); kept=0.0; tot=0.0
    for s in range(200):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        cur=flat(); d=cur-prev
        align=(torch.sign(d)==sg)
        if mode=="forward_only": d2=torch.where(align,d,torch.zeros_like(d))
        elif mode=="back_only":  d2=torch.where(align,torch.zeros_like(d),d)
        else: d2=d
        tot+=float(d.abs().sum()); kept+=float(d2.abs().sum())
        setflat(prev+d2); prev=flat()
    return V(), 100*kept/max(tot,1e-30)
print("="*80); print("  PER-PARAMETER HINDSIGHT CULL OF WANDERING"); print("="*80)
print(f"  {'arm':>34}{'motion kept':>13}{'val':>10}{'% of improvement':>19}")
pct=lambda v:100*(v0-v)/max(v0-v_full,1e-12)
print(f"  {'full GD (baseline)':>34}{100.0:>12.1f}%{v_full:>10.4f}{100.0:>18.1f}%", flush=True)
for mode,lab in [("forward_only","cull steps opposing sgn(D)"),
                 ("back_only","keep ONLY opposing steps")]:
    v,k=rerun(mode); print(f"  {lab:>34}{k:>12.1f}%{v:>10.4f}{pct(v):>18.1f}%", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
