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
# --- brackets at 120, stored float32 ---
th=torch.load("J120.pt")["th"].double()
g0=G(th); nV=[float(g0[IDX[i]].norm()) for i in range(7)]
Dg=[(G(th+H*(-mk(g0,i))/nV[i])-g0)/H for i in range(7)]
B=torch.stack([(-nV[i]*mk(Dg[i],j)+nV[j]*mk(Dg[j],i)).float() for (i,j) in pairs])
Bn=(B.double()*B.double()).sum(1).numpy()
del Dg; gc.collect()
def fields_at(ck):
    t=torch.load(f"J{ck}.pt")["th"].double(); gg=G(t)
    return [(-mk(gg,i)).float() for i in range(7)]
def inspan(Vt):
    num=np.zeros(len(pairs))
    for j in range(7):
        vj=Vt[j][IDX[j]].double(); d=float(vj@vj)
        c=(B[:,IDX[j]].double()@vj).numpy()
        num+=c**2/max(d,1e-300)
    return num/np.clip(Bn,1e-300,None)
print("="*74); print("  PERSISTENCE OF LIE CLOSURE ACROSS CHECKPOINTS"); print("="*74)
print(f"  brackets computed at step 120; null = {100*NULL:.6f}%\n")
print(f"  {'fields from':<22}{'mean':>9}{'min':>8}{'max':>8}{'vs null':>11}")
V120=[(-mk(g0,i)).float() for i in range(7)]
for lab,Vt in [("step 120 (own)",V120)]:
    fr=inspan(Vt); print(f"  {lab:<22}{100*fr.mean():>8.2f}%{100*fr.min():>7.2f}{100*fr.max():>7.2f}{fr.mean()/NULL:>10.0f}x", flush=True)
own=inspan(V120).mean()
for ck in (160,200,80,40):
    try:
        Vt=fields_at(ck); fr=inspan(Vt)
        print(f"  {'step '+str(ck):<22}{100*fr.mean():>8.2f}%{100*fr.min():>7.2f}{100*fr.max():>7.2f}"
              f"{fr.mean()/NULL:>10.0f}x   ({fr.mean()/own:.2f} x own)", flush=True)
        del Vt; gc.collect()
    except Exception as e: print("  skip",ck,e)
rg=torch.Generator().manual_seed(99); Vr=[]
for i in range(7):
    z=torch.zeros(P); r=torch.randn(NP[i],generator=rg); z[IDX[i]]=r/r.norm()*float(V120[i].norm()); Vr.append(z)
fr=inspan(Vr)
print(f"  {'RANDOM node fields':<22}{100*fr.mean():>8.2f}%{100*fr.min():>7.2f}{100*fr.max():>7.2f}"
      f"{fr.mean()/NULL:>10.0f}x   ({fr.mean()/own:.3f} x own)")
print("\n  time %.0fs"%(time.time()-t0))
