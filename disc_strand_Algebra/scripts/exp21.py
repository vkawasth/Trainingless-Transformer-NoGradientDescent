import time, itertools, re, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
model=model.double(); named=list(model.named_parameters())
def lay(n):
    m=re.match(r"blocks\.(\d+)\.",n)
    return f"L{m.group(1)}" if m else "EMB"
GRP=["EMB"]+[f"L{i}" for i in range(6)]
sl={g:[] for g in GRP}; off=0
for n,p in named: sl[lay(n)].append((off,off+p.numel())); off+=p.numel()
P=off
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in GRP]
NP=[len(I) for I in IDX]; NG=len(GRP); NULL=NG/P
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def G(v):
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
def mk(v,i):
    o=torch.zeros_like(v); o[IDX[i]]=v[IDX[i]]; return o
H=1e-4
th=torch.load("J160.pt")["th"].double()
g0=G(th); nV=[float(g0[IDX[i]].norm()) for i in range(NG)]
Dg=[((G(th+H*(-mk(g0,i))/nV[i])-g0)/H) for i in range(NG)]
print("="*76); print("  LAYER-GRADED LIE STRUCTURE  (step 160)"); print("="*76)
print(f"  params per group: "+" ".join(f"{GRP[i]}:{NP[i]/1e3:.0f}k" for i in range(NG)))
print(f"  ||V_g||:          "+" ".join(f"{GRP[i]}:{nV[i]:.3g}" for i in range(NG)))
# ---- CULLING: effective rank of the response space inside each layer ----
print("\n  CULLING TEST: eff-rank of {P_l Dg_m : m=0..6} inside layer l  (max 7)")
print(f"  {'layer':>7}{'eff-rank':>10}{'rank(5%)':>10}   normalised spectrum")
er=lambda w:(np.clip(w,0,None).sum()**2)/max((np.clip(w,0,None)**2).sum(),1e-300)
for l in range(NG):
    Gm=np.array([[float(Dg[a][IDX[l]]@Dg[b][IDX[l]]) for b in range(NG)] for a in range(NG)])
    w=np.linalg.eigvalsh(Gm)[::-1]
    print(f"  {GRP[l]:>7}{er(w):>10.2f}{int((np.clip(w,0,None)>0.05*w.max()).sum()):>10}"
          f"   {' '.join(f'{x:.2f}' for x in (np.clip(w,0,None)/w.max())[:5])}")
# ---- brackets and closure vs layer distance ----
pairs=list(itertools.combinations(range(NG),2))
def inspan(b):
    num=0.0
    for j in range(NG):
        vj=-g0[IDX[j]]; d=float(vj@vj)
        num+=float(b[IDX[j]]@vj)**2/max(d,1e-300)
    return num/max(float(b@b),1e-300)
rows=[]
for (i,j) in pairs:
    b=-nV[i]*mk(Dg[i],j)+nV[j]*mk(Dg[j],i)
    rows.append((i,j,float(b.norm()),inspan(b)))
    del b
print("\n  BRACKET CLOSURE BY LAYER PAIR   (null = %.6f%%)"%(100*NULL))
print("       "+"".join(f"{g:>8}" for g in GRP))
M=np.full((NG,NG),np.nan)
for (i,j,n_,f_) in rows: M[i,j]=M[j,i]=100*f_
for i in range(NG):
    print(f"  {GRP[i]:>4} "+"".join("     .  " if i==j else f"{M[i,j]:>7.1f}%" for j in range(NG)))
print("\n  CLOSURE AND MAGNITUDE vs LAYER DISTANCE (block pairs only, EMB excluded)")
print(f"  {'|l-m|':>7}{'n':>4}{'mean in-span':>15}{'mean ||[.,.]||':>17}")
for d in range(1,6):
    sel=[(n_,f_) for (i,j,n_,f_) in rows if i>0 and j>0 and abs(i-j)==d]
    if sel:
        print(f"  {d:>7}{len(sel):>4}{100*np.mean([s[1] for s in sel]):>14.1f}%"
              f"{np.mean([s[0] for s in sel]):>17.4g}")
sel=[(n_,f_) for (i,j,n_,f_) in rows if i==0]
print(f"  {'EMB-*':>7}{len(sel):>4}{100*np.mean([s[1] for s in sel]):>14.1f}%"
      f"{np.mean([s[0] for s in sel]):>17.4g}")
print("\n  CLOSURE vs DEPTH (mean over partners, blocks only)")
print(f"  {'layer':>7}{'mean in-span':>15}{'mean ||[.,.]||':>17}")
for i in range(1,NG):
    sel=[(n_,f_) for (a,b_,n_,f_) in rows if (a==i or b_==i) and a>0 and b_>0]
    print(f"  {GRP[i]:>7}{100*np.mean([s[1] for s in sel]):>14.1f}%{np.mean([s[0] for s in sel]):>17.4g}")
allf=np.array([f_ for (_,_,_,f_) in rows])
print(f"\n  overall mean in-span {100*allf.mean():.1f}%  = {allf.mean()/NULL:.0f}x null")
print("  time %.0fs"%(time.time()-t0))
