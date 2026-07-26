import time, itertools, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
NODES=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
model=model.double(); named=list(model.named_parameters())
sl={g:[] for g in NODES}; off=0
for n,p in named: sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
P=off
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES]
NP=[len(I) for I in IDX]
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
H=1e-4; pairs=list(itertools.combinations(range(7),2)); NULL=7.0/P
CKS=[40,80,120,160]
GS={}
for ck in CKS:
    GS[ck]=G(torch.load(f"J{ck}.pt")["th"].double()).float(); gc.collect()
print(f"  field sets from {CKS} built ({time.time()-t0:.0f}s)", flush=True)
rg=torch.Generator().manual_seed(99); grand=torch.zeros(P)
for i in range(7):
    r=torch.randn(NP[i],generator=rg); grand[IDX[i]]=r/r.norm()*float(GS[120][IDX[i]].norm())
GS["rand"]=grand
th=torch.load("J120.pt")["th"].double()
g0=G(th); nV=[float(g0[IDX[i]].norm()) for i in range(7)]
Dg=[((G(th+H*(-mk(g0,i))/nV[i])-g0)/H).float() for i in range(7)]
del th; gc.collect()
print(f"  level-1 data at 120 built ({time.time()-t0:.0f}s)", flush=True)
KEYS=CKS+["rand"]
acc={k:[] for k in KEYS}
for (i,j) in pairs:
    b=(-nV[i]*mk(Dg[i],j)+nV[j]*mk(Dg[j],i))
    nb=float(b.double()@b.double())
    for k in KEYS:
        gk=GS[k]; num=0.0
        for jj in range(7):
            bj=b[IDX[jj]].double(); vj=gk[IDX[jj]].double(); d=float(vj@vj)
            num+=float(bj@vj)**2/max(d,1e-300)
        acc[k].append(num/max(nb,1e-300))
    del b
print("="*74); print("  PERSISTENCE OF LIE CLOSURE ACROSS CHECKPOINTS"); print("="*74)
print(f"  brackets computed at step 120;  null = {100*NULL:.6f}%\n")
own=np.mean(acc[120])
print(f"  {'fields from':<22}{'mean':>9}{'min':>8}{'max':>8}{'vs null':>10}{'x own':>8}")
for k in KEYS:
    fr=np.array(acc[k]); lab = "RANDOM node fields" if k=="rand" else f"step {k}" + (" (own)" if k==120 else "")
    print(f"  {lab:<22}{100*fr.mean():>8.2f}%{100*fr.min():>7.2f}{100*fr.max():>7.2f}"
          f"{fr.mean()/NULL:>9.0f}x{fr.mean()/own:>8.3f}")
print("\n  time %.0fs"%(time.time()-t0))
