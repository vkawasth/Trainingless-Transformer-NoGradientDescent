"""
CEILING TEST FOR ANY beta-DRIVEN SCHEME.
beta is a per-tile MAGNITUDE field: it carries no within-tile direction.
Arms, all applied to the true update d at a checkpoint:
  (a) TRUE          : theta += d
  (b) MAG-ONLY      : per-tile |d| profile kept exactly, directions randomised in-tile
  (c) DIR-ONLY      : directions kept, magnitudes flattened to one scalar per layer
  (d) SIGN-ONLY     : keep sign pattern only (magnitude flat globally)
  (e) RANDOM        : random vector, global norm matched
"""
import re, time, copy, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
LAY=["L0","L1","L2","L3","L4","L5"]; BWD={"L0":1024,"L1":512,"L2":256,"L3":64,"L4":32,"L5":16}
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
P=off
def lay_of(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
LIDX={}
for n,_ in named: LIDX.setdefault(lay_of(n),[]).append(torch.arange(*SPAN[n]))
LIDX={k:torch.cat(v) for k,v in LIDX.items()}
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def V(n=16): return float(eval_val(model,n=n))
def tiles(nel,nt,ov=0.0):
    nt=min(nt,nel); st=nel/nt
    a=np.clip(np.floor(np.arange(nt)*st).astype(int),0,nel-1)
    z=np.clip(np.ceil((np.arange(nt)+1)*st).astype(int),1,nel)
    return a,np.maximum(z,a+1)
rng=torch.Generator().manual_seed(5)
print("="*80); print("  MAGNITUDE vs DIRECTION:  what carries the value of an update?"); print("="*80)
for CK in (80,120,160):
    ck=torch.load(f"J{CK}.pt"); th=ck["th"]
    model.load_state_dict(copy.deepcopy(ck["sd"]))
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    opt.load_state_dict(copy.deepcopy(ck["od"]))
    setflat(th); v0=V()
    # accumulate the true 20-step update
    torch.manual_seed(700+CK)
    for _ in range(20):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    d=flat()-th; vT=V()
    res={"TRUE":vT}
    # (b) magnitude-only, per tile at the L-specific backward resolution
    db=torch.zeros_like(d)
    for L,idx in LIDX.items():
        nt=BWD.get(L,16); sub=d[idx]; nel=len(sub)
        a,z=tiles(nel,nt)
        r=torch.randn(nel,generator=rng)
        for i in range(len(a)):
            seg=slice(a[i],z[i]); rs=r[seg]
            db[idx[seg]]=rs/ (rs.norm()+1e-30) * sub[seg].norm()
    setflat(th+db); res["MAG-ONLY"]=V()
    # (c) direction-only: unit direction per layer, magnitude flattened within layer
    dc=torch.zeros_like(d)
    for L,idx in LIDX.items():
        sub=d[idx]; s=torch.sign(sub); dc[idx]=s*(sub.abs().mean())
    setflat(th+dc); res["DIR-ONLY(flat mag)"]=V()
    # (d) sign only, one global scale
    dd=torch.sign(d)*d.abs().mean()
    setflat(th+dd); res["SIGN-ONLY"]=V()
    # (e) random, norm matched
    r=torch.randn(P,generator=rng); r=r/r.norm()*d.norm()
    setflat(th+r); res["RANDOM"]=V()
    setflat(th)
    print(f"\n  checkpoint {CK}:  val before {v0:.4f}   true 20-step -> {vT:.4f}")
    print(f"  {'arm':>22}{'val':>10}{'progress kept':>16}")
    for k in ["TRUE","MAG-ONLY","DIR-ONLY(flat mag)","SIGN-ONLY","RANDOM"]:
        frac=100*(v0-res[k])/max(v0-vT,1e-12)
        print(f"  {k:>22}{res[k]:>10.4f}{frac:>15.1f}%", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
