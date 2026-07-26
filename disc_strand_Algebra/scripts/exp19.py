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
H=1e-4
pairs=list(itertools.combinations(range(7),2))
def build(ck):
    th=torch.load(f"J{ck}.pt")["th"].double()
    g0=G(th); nV=[float(g0[IDX[i]].norm()) for i in range(7)]
    Dg=[(G(th+H*(-mk(g0,i))/nV[i])-g0)/H for i in range(7)]
    B=[(-nV[i]*mk(Dg[i],j)+nV[j]*mk(Dg[j],i)) for (i,j) in pairs]
    V=[-mk(g0,i) for i in range(7)]
    del Dg,g0; gc.collect()
    return V,B
def inspan(Z,V):
    n=sum(float(Z[IDX[j]]@V[j][IDX[j]])**2/max(float(V[j][IDX[j]]@V[j][IDX[j]]),1e-300)
          for j in range(7))
    return n/max(float(Z@Z),1e-300)
print("="*76); print("  PERSISTENCE: are brackets aligned with the fields at OTHER checkpoints?"); print("="*76)
V120,B120=build(120); print(f"  ck120 built ({time.time()-t0:.0f}s)", flush=True)
V160,_   =build(160); print(f"  ck160 built ({time.time()-t0:.0f}s)", flush=True)
V80 ,_   =build(80);  print(f"  ck80  built ({time.time()-t0:.0f}s)", flush=True)
NULL=7.0/P
rows=[]
for lab,Vt in [("own (120)",V120),("later (160)",V160),("earlier (80)",V80)]:
    fr=np.array([inspan(b,Vt) for b in B120])
    rows.append((lab,fr))
    print(f"  brackets@120 vs fields@{lab:<14} mean {100*fr.mean():>6.2f}%  "
          f"(min {100*fr.min():.2f}, max {100*fr.max():.2f})   {fr.mean()/NULL:>8.0f}x null", flush=True)
rg=torch.Generator().manual_seed(99)
Vr=[]
for i in range(7):
    z=torch.zeros(P,dtype=torch.float64); r=torch.randn(NP[i],generator=rg,dtype=torch.float64)
    z[IDX[i]]=r/r.norm()*float(V120[i].norm()); Vr.append(z)
fr=np.array([inspan(b,Vr) for b in B120])
print(f"  brackets@120 vs RANDOM node fields      mean {100*fr.mean():>6.2f}%"
      f"                        {fr.mean()/NULL:>8.0f}x null")
print(f"\n  cos between field sets (per node): "
      + " ".join(f"{NODES[i]}:{float(V120[i]@V160[i]/(V120[i].norm()*V160[i].norm())):.2f}" for i in range(7)))
print(f"\n  READING: if 'later/earlier' stay within ~2x of 'own', the closure is a")
print(f"  persistent structure of the phase, not a property of the base point.")
print(f"\n  time %.0fs"%(time.time()-t0))
