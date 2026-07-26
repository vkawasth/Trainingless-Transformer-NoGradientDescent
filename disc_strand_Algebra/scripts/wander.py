"""
WHAT SETTLES WANDERING, AND IS IT NECESSARY?
 (A) sliding-window cancellation net/path through training, with |d|, ||g||, cos(d_t,d_{t+1})
 (B) hindsight culling: rerun from the same init, projecting each update onto
     the KNOWN chord direction u (rank 1) or the trajectory subspace (rank 10).
     Control: project OUT of u (keep only the wandering).
"""
import time, gc, copy, numpy as np, torch
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
# ---------- PASS 1 ----------
model.load_state_dict(torch.load("init.pt")); th0=flat(); torch.manual_seed(17)
opt=newopt(); W=20
prev=th0.clone(); wnet=torch.zeros_like(th0); wpath=torch.zeros_like(th0)
rows=[]; snaps=[th0.clone()]; gs=[]; dprev=None
for s in range(1,201):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    gn=float(torch.cat([p.grad.flatten() for _,p in named]).norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); d=cur-prev; prev=cur
    wnet+=d; wpath+=d.abs(); gs.append(gn)
    c1=float(d@dprev/(d.norm()*dprev.norm())) if dprev is not None else float('nan')
    rows.append((s,float(l),gn,float(d.norm()),c1)); dprev=d
    if s%W==0:
        rows[-1]=rows[-1]+(float(wnet.abs().sum()/wpath.sum()),)
        wnet.zero_(); wpath.zero_()
    if s%20==0: snaps.append(cur.clone())
th200=flat(); D=th200-th0; u=D/D.norm(); v_full=V()
print("="*80); print("  (A) DOES CANCELLATION CHANGE AS THE TRAJECTORY SETTLES?"); print("="*80)
print(f"  {'window':>10}{'loss':>9}{'||g||':>8}{'|d|':>8}{'cos(d,d+1)':>12}{'net/path':>10}")
for r in rows:
    if len(r)==6:
        print(f"  {str(r[0]-W+1)+'-'+str(r[0]):>10}{r[1]:>9.4f}{r[2]:>8.4f}{r[3]:>8.4f}{r[4]:>12.3f}{r[5]:>10.4f}")
print(f"\n  full-run net/path = {float(D.abs().sum()):.0f} / path  -> chord/step-sum "
      f"= {float(D.norm())/sum(r[3] for r in rows):.4f}")
# rank-10 trajectory basis
B=torch.stack([snaps[i+1]-snaps[i] for i in range(len(snaps)-1)])
Q,_=torch.linalg.qr(B.T); del B; gc.collect()
print(f"  trajectory subspace: rank {Q.shape[1]}   ||P_Q D||/||D|| = "
      f"{float((Q@(Q.T@D)).norm()/D.norm()):.4f}   |u.d| share of chord = 1.000")
del snaps; gc.collect()
# ---------- PASS 2: culled reruns ----------
def rerun(mode):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
    o=newopt(); prev=flat()
    for s in range(200):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        cur=flat(); d=cur-prev
        if mode=="u":      d2=(d@u)*u
        elif mode=="perp": d2=d-(d@u)*u
        elif mode=="Q":    d2=Q@(Q.T@d)
        elif mode=="Qperp":d2=d-Q@(Q.T@d)
        else:              d2=d
        setflat(prev+d2); prev=flat()
    return V()
print("\n"+"="*80); print("  (B) HINDSIGHT CULLING: project each update, then continue"); print("="*80)
print(f"  {'arm':>34}{'val':>10}{'% of improvement':>19}")
v0=4.4657
def pct(v): return 100*(v0-v)/max(v0-v_full,1e-12)
print(f"  {'full GD (baseline)':>34}{v_full:>10.4f}{100.0:>18.1f}%", flush=True)
import sys
ARMS=sys.argv[1:] if len(sys.argv)>1 else ["u"]
LABS={"u":"keep only chord dir u (rank 1)","perp":"keep only wandering (u removed)","Q":"keep rank-10 trajectory subspace","Qperp":"remove rank-10 subspace"}
for mode in ARMS:
    lab=LABS[mode]
    v=rerun(mode); print(f"  {lab:>34}{v:>10.4f}{pct(v):>18.1f}%", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
