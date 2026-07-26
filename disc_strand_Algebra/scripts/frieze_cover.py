"""
FRIEZE (PERIODIC) COVER: tile i = { j : (j mod NT) in [i, i+w) mod NT }.
Adjacent tiles overlap; tile NT-1 overlaps tile 0  ->  the nerve is a CYCLE.
 (a) control: over-dispersion of sign flips, contiguous vs strided vs random
 (b) cellular sheaf on the CYCLE: restriction maps, holonomy around the loop,
     H^0 and H^1 (which are no longer forced to be 2 and 0).
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
P=flat().numel(); NT=256; W=2
res=torch.arange(P)%NT
CONT=[torch.arange(i*(P//NT),(i+1)*(P//NT)) for i in range(NT)]
FRZ=[torch.nonzero(((res-i)%NT)<W,as_tuple=True)[0] for i in range(NT)]   # periodic, overlapping
STR=[torch.nonzero(res==i,as_tuple=True)[0] for i in range(NT)]           # strided, disjoint
g=torch.Generator().manual_seed(4); pm=torch.randperm(P,generator=g)
RND=[pm[i*(P//NT):(i+1)*(P//NT)] for i in range(NT)]
print(f"  P={P:,}  NT={NT}  frieze window W={W} -> {len(FRZ[0]):,} params/tile,"
      f" overlap {100*(W-1)/W:.0f}%", flush=True)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
prev=flat(); sp=None; OD={"contig":[],"frieze":[],"strided":[],"random":[]}
FS={k:[] for k in ["frieze"]}; LOSS=[]
for s in range(1,101):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; sg=torch.sign(d); LOSS.append(float(l))
    FS["frieze"].append(np.array([float(d[I].abs().mean()) for I in FRZ],dtype=np.float32))
    if sp is not None and s>10:
        fl=(sg!=sp).double(); pbar=float(fl.mean())
        for nm,C in [("contig",CONT),("frieze",FRZ),("strided",STR),("random",RND)]:
            pt=np.array([float(fl[I].mean()) for I in C]); n=len(C[0])
            OD[nm].append(pt.std()/max(np.sqrt(pbar*(1-pbar)/n),1e-30))
    sp=sg; prev=af; del b4,af,d
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*80); print("  (a) IS FLIP-CLUSTERING ABOUT CONTIGUITY?"); print("="*80)
print(f"  {'cover':>12}{'over-dispersion vs binomial':>30}")
for nm in ["contig","frieze","strided","random"]:
    a=np.array(OD[nm]); print(f"  {nm:>12}{a.mean():>24.2f}x   (min {a.min():.2f}, max {a.max():.2f})")
print("  1.0 = independent flips.  random is the null by construction.")
# ---- (b) sheaf on the cycle ----
H=0.062; L=np.array(LOSS); ls=L-H; ok=ls>1e-6
xx=np.log(ls[ok]); xm=xx-xx.mean(); den=float((xm*xm).sum())
A=np.stack(FS["frieze"])[ok]
Y=np.log(np.maximum(A,1e-30))
al=(Y-Y.mean(0)).T@xm/den; be=Y.mean(0)-al*xx.mean()
X=np.stack([al,be],1)
print("\n"+"="*80); print("  (b) CELLULAR SHEAF ON THE CYCLE NERVE"); print("="*80)
M=[]
for i in range(NT):
    j=(i+1)%NT
    a=X[i][:,None]; b=X[j][:,None]
    M.append(np.array([[b[0,0]/a[0,0] if a[0,0]!=0 else 1.0,0],[0,b[1,0]/a[1,0] if a[1,0]!=0 else 1.0]]))
# global least-squares transition (shared) and the true per-edge holonomy
Xi=X[np.arange(NT)]; Xj=X[(np.arange(NT)+1)%NT]
Msh,_,_,_=np.linalg.lstsq(Xi,Xj,rcond=None); Msh=Msh.T
r=np.linalg.norm(Xi@Msh.T-Xj)/max(np.linalg.norm(Xj),1e-30)
print(f"  shared transition M (tile i -> i+1): residual {r:.4f}")
print(f"    M = [[{Msh[0,0]:.4f},{Msh[0,1]:.4f}],[{Msh[1,0]:.4f},{Msh[1,1]:.4f}]]"
      f"  ev {np.round(np.abs(np.linalg.eigvals(Msh)),4)}")
Hol=np.linalg.matrix_power(Msh,NT)
print(f"\n  HOLONOMY around the cycle = M^{NT}:")
print(f"    ||M^{NT} - I|| / ||I|| = {np.linalg.norm(Hol-np.eye(2))/np.sqrt(2):.4e}")
print(f"    eigenvalues |.| = {np.round(np.abs(np.linalg.eigvals(Hol)),6)}")
d0=np.zeros((2*NT,2*NT))
for i in range(NT):
    j=(i+1)%NT
    d0[2*i:2*i+2,2*i:2*i+2]=Msh; d0[2*i:2*i+2,2*j:2*j+2]=-np.eye(2)
sv=np.linalg.svd(d0,compute_uv=False); tol=1e-8*sv[0]; rk=int((sv>tol).sum())
print(f"\n  coboundary on the cycle: 2V={2*NT}  2E={2*NT}  rank={rk}")
print(f"    dim H^0 = {2*NT-rk}      dim H^1 = {2*NT-rk}")
print(f"    (path nerve forced H^0=2, H^1=0; a cycle does not)")
print(f"\n  time {time.time()-t0:.0f}s")
