"""
(1) WHAT IS THE OFF-AXIS COMPUTE?  Decompose every step into the component along
    the direction-to-final and the perpendicular remainder.
(2) Does the perpendicular part correspond to the cancelled motion?
(3) COUPLING between weight-tiles and update-tiles at matched resolution.
"""
import re, time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
def lay(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
GI={}
for n,_ in named: GI.setdefault(lay(n),[]).append(torch.arange(*SPAN[n]))
GI={k:torch.cat(v) for k,v in GI.items()}; KEYS=["EMB"]+[f"L{i}" for i in range(6)]
NT=64
BND={}
for k in KEYS:
    nel=len(GI[k]); e=np.linspace(0,nel,NT+1).astype(int); BND[k]=e
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
TH=[flat()]; STEPS=[]
Wt={k:[] for k in KEYS}; Ut={k:[] for k in KEYS}
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); opt.step(); af=flat(); d=af-b4
    STEPS.append(d.clone() if s%1==0 else None)
    for k in KEYS:
        idx=GI[k]; e=BND[k]
        w=af[idx]; u=d[idx].abs()
        Wt[k].append(np.array([float(w[e[i]:e[i+1]].abs().mean()) for i in range(NT)]))
        Ut[k].append(np.array([float(u[e[i]:e[i+1]].mean()) for i in range(NT)]))
    if s%50==49: TH.append(af.clone())
th0=TH[0]; thF=flat(); Dg=thF-th0
print(f"  collected ({time.time()-t0:.0f}s)", flush=True)
print("="*80); print("  (1) OFF-AXIS COMPUTE: each step vs the direction to the final point"); print("="*80)
print(f"  {'window':>10}{'|d|':>8}{'cos(d,u_glob)':>15}{'cos(d,u_local)':>16}{'% par':>8}{'% perp':>9}")
th=th0.clone(); ug=Dg/Dg.norm()
rows=[]
for w0 in range(0,200,25):
    cs=[];cl=[];nn=[]
    for s in range(w0,w0+25):
        d=STEPS[s]; nn.append(float(d.norm()))
        cs.append(float(d@ug/(d.norm()*ug.norm())))
        rem=thF-th; ul=rem/max(float(rem.norm()),1e-30)
        cl.append(float(d@ul/(d.norm())))
        th=th+d
    c=np.mean(cs); cl_=np.mean(cl)
    rows.append((w0,np.mean(nn),c,cl_))
    print(f"  {str(w0+1)+'-'+str(w0+25):>10}{np.mean(nn):>8.4f}{c:>15.3f}{cl_:>16.3f}"
          f"{100*cl_:>7.1f}%{100*np.sqrt(max(1-cl_**2,0)):>8.1f}%")
tot=sum(float(d.norm()) for d in STEPS)
print(f"\n  total path (sum |d|)        = {tot:.1f}")
print(f"  chord ||theta_200-theta_0|| = {float(Dg.norm()):.1f}")
print(f"  chord / path                = {float(Dg.norm())/tot:.4f}   "
      f"=> {100*(1-float(Dg.norm())/tot):.1f}% of the L2 path is off-axis")
par=sum(abs(float(d@ug)) for d in STEPS)
print(f"  sum |d.u_glob| / path       = {par/tot:.4f}")
print("\n"+"="*80); print("  (2) IS OFF-AXIS MOTION THE CANCELLED MOTION?"); print("="*80)
net=torch.zeros_like(th0); path=torch.zeros_like(th0)
for d in STEPS: net+=d; path+=d.abs()
print(f"  per-parameter: sum|net|/sum path = {float(net.abs().sum()/path.sum()):.4f}"
      f"   (L1 cancellation {100*(1-float(net.abs().sum()/path.sum())):.1f}%)")
print(f"  L2 view      : chord/path        = {float(Dg.norm())/tot:.4f}")
print("  the two differ: L1 cancellation is per-coordinate reversal, L2 loss is")
print("  reversal PLUS rotation. Their gap is the genuinely perpendicular part.")
print("\n"+"="*80); print(f"  (3) WEIGHT-TILE vs UPDATE-TILE COUPLING  ({NT} tiles/layer)"); print("="*80)
print(f"  {'layer':>7}{'corr(|w|,|u|) across tiles':>29}{'corr over time (mean)':>24}")
for k in KEYS:
    W=np.stack(Wt[k]); U=np.stack(Ut[k])
    cs=[np.corrcoef(W[t],U[t])[0,1] for t in range(0,200,10)]
    ct=[np.corrcoef(W[:,i],U[:,i])[0,1] for i in range(NT)]
    print(f"  {k:>7}{np.mean(cs):>29.3f}{np.nanmean(ct):>24.3f}")
print(f"\n  time {time.time()-t0:.0f}s")
