"""
IS THE 95% PERPENDICULAR MOTION BATCH NOISE OR LANDSCAPE GEOMETRY?
Arm A: fresh batch every step (baseline, stochastic)
Arm B: the SAME batch every step (deterministic gradient, no sampling)
If B still shows large perpendicular fraction and cancellation, the transverse
motion is the shape of the loss surface, not exploration.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
torch.manual_seed(99); FIXED=[get_batch() for _ in range(1)]
def go(fixed, npass):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    th0=flat()
    if npass==1:
        for s in range(200):
            model.train()
            x,y = FIXED[0] if fixed else get_batch()
            _,l=model(x,y); o.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        return flat()-th0, float(eval_val(model,n=16))
    return None
Dg_s,_=go(False,1); Dg_f,vf=go(True,1)
print(f"  chords: stochastic {float(Dg_s.norm()):.2f}   fixed-batch {float(Dg_f.norm()):.2f}"
      f"  ({time.time()-t0:.0f}s)", flush=True)
def measure(fixed, Dg):
    ug=Dg/Dg.norm()
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    thF=flat()+Dg
    net=torch.zeros_like(thF); path=torch.zeros_like(thF)
    tot=0.0; cs=[]; c1=[]; dprev=None
    for s in range(200):
        model.train()
        x,y = FIXED[0] if fixed else get_batch()
        _,l=model(x,y); o.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        b4=flat(); o.step(); af=flat(); d=af-b4
        nd=float(d.norm()); tot+=nd; net+=d; path+=d.abs()
        rem=thF-b4
        cs.append(float(d@rem)/max(nd*float(rem.norm()),1e-30))
        if dprev is not None: c1.append(float(d@dprev)/max(nd*float(dprev.norm()),1e-30))
        dprev=d.clone(); del b4,af,d
        if s%50==49: gc.collect()
    return dict(perp=100*np.sqrt(max(1-np.mean(cs)**2,0)), cos_to_final=np.mean(cs),
                cos_consec=np.mean(c1), l1=float(net.abs().sum()/path.sum()),
                chordpath=float(Dg.norm())/tot, path=tot)
A=measure(False,Dg_s); B=measure(True,Dg_f)
print("="*78); print("  BATCH NOISE OR LANDSCAPE?"); print("="*78)
print(f"  {'quantity':>34}{'stochastic':>14}{'fixed batch':>14}")
for k,lab in [("cos_to_final","mean cos(step, to-final)"),("perp","% perpendicular"),
              ("cos_consec","mean cos(d_t, d_t+1)"),("l1","sum|net|/sum|path|"),
              ("chordpath","chord / path"),("path","total path length")]:
    print(f"  {lab:>34}{A[k]:>14.4f}{B[k]:>14.4f}")
print(f"\n  L1 cancellation: stochastic {100*(1-A['l1']):.1f}%   fixed {100*(1-B['l1']):.1f}%")
print(f"  fixed-batch final val (on held-out) = {vf:.4f}")
print("\n  If the fixed-batch column matches the stochastic one, the transverse")
print("  motion is landscape geometry and no 'sampling' is involved.")
print(f"\n  time {time.time()-t0:.0f}s")
