import re, time, gc, numpy as np, torch
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
NT=64; BND={k:np.linspace(0,len(GI[k]),NT+1).astype(int) for k in KEYS}
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
# pass 1
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
th0=flat()
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
thF=flat(); Dg=thF-th0; ug=Dg/Dg.norm()
print(f"  pass1 done, chord {float(Dg.norm()):.2f} ({time.time()-t0:.0f}s)", flush=True)
# pass 2
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
prev=flat(); net=torch.zeros_like(prev); path=torch.zeros_like(prev)
tot=0.0; parsum=0.0; rows=[]; Wt={k:[] for k in KEYS}; Ut={k:[] for k in KEYS}
cg=[]; cl=[]; nn=[]
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    nd=float(d.norm()); tot+=nd; net+=d; path+=d.abs()
    c1=float(d@ug)/max(nd,1e-30); parsum+=abs(float(d@ug))
    rem=thF-b4; c2=float(d@rem)/max(nd*float(rem.norm()),1e-30)
    cg.append(c1); cl.append(c2); nn.append(nd)
    for k in KEYS:
        idx=GI[k]; e=BND[k]; w=af[idx].abs(); u=d[idx].abs()
        Wt[k].append(np.array([float(w[e[i]:e[i+1]].mean()) for i in range(NT)]))
        Ut[k].append(np.array([float(u[e[i]:e[i+1]].mean()) for i in range(NT)]))
    prev=af; del b4,af,d; 
    if s%50==49: gc.collect()
cg=np.array(cg); cl=np.array(cl); nn=np.array(nn)
print("="*80); print("  (1) OFF-AXIS COMPUTE"); print("="*80)
print(f"  {'window':>10}{'|d|':>8}{'cos(d,chord)':>15}{'cos(d,to-final)':>18}{'% perp':>9}")
for w0 in range(0,200,25):
    sl=slice(w0,w0+25); c=cl[sl].mean()
    print(f"  {str(w0+1)+'-'+str(w0+25):>10}{nn[sl].mean():>8.4f}{cg[sl].mean():>15.3f}"
          f"{c:>18.3f}{100*np.sqrt(max(1-c**2,0)):>8.1f}%")
print(f"\n  total path {tot:.1f}   chord {float(Dg.norm()):.1f}   chord/path {float(Dg.norm())/tot:.4f}")
print(f"  sum|d.chord|/path = {parsum/tot:.4f}   => {100*(1-parsum/tot):.1f}% of motion is off the chord axis")
print("\n"+"="*80); print("  (2) L1 REVERSAL vs L2 ROTATION"); print("="*80)
l1=float(net.abs().sum()/path.sum())
print(f"  per-coordinate  sum|net|/sum|path| = {l1:.4f}   ({100*(1-l1):.1f}% reversal)")
print(f"  whole-vector    chord/path         = {float(Dg.norm())/tot:.4f}   ({100*(1-float(Dg.norm())/tot):.1f}% L2 loss)")
print(f"  gap = rotation not explained by per-coordinate reversal: "
      f"{100*(l1-float(Dg.norm())/tot):.1f} points")
print("\n"+"="*80); print(f"  (3) WEIGHT-TILE vs UPDATE-TILE COUPLING ({NT} tiles/layer)"); print("="*80)
print(f"  {'layer':>7}{'corr across tiles':>21}{'corr over time':>18}")
for k in KEYS:
    W=np.stack(Wt[k]); U=np.stack(Ut[k])
    cs=[np.corrcoef(W[t],U[t])[0,1] for t in range(0,200,10)]
    ct=[np.corrcoef(W[:,i],U[:,i])[0,1] for i in range(NT)]
    print(f"  {k:>7}{np.mean(cs):>21.3f}{np.nanmean(ct):>18.3f}")
print(f"\n  time {time.time()-t0:.0f}s")
