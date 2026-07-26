import time, itertools, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
named=list(model.named_parameters())
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
sl={g:[] for g in NODES}; off=0
for n,p in named: sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def sub(v,gs):
    m=torch.zeros_like(v)
    for g in gs:
        for a,b in sl[g]: m[a:b]=v[a:b]
    return m
seg=torch.load("seg.pt"); th=seg["th120"]; DT=seg["DT"]
torch.manual_seed(3); XB=[get_batch() for _ in range(3)]
def logits(vec):
    setflat(vec); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
L0=logits(th); D={g: sub(DT,[g]) for g in NODES}
V=[]                       # 7 grade-1 then 21 grade-2, float32
for g in NODES: V.append(logits(th+D[g])-L0)
pairs=list(itertools.combinations(range(7),2))
for i,j in pairs:
    V.append((logits(th+D[NODES[i]]+D[NODES[j]])-L0)-V[i]-V[j])
n=len(V); print(f"  built {n} vectors of dim {V[0].numel():,}  ({time.time()-t0:.0f}s)", flush=True)
Gm=np.zeros((n,n))
for a in range(n):
    for b in range(a,n):
        Gm[a,b]=Gm[b,a]=float(V[a].double()@V[b].double())
del V; gc.collect()
def erank(w):
    w=np.clip(w,0,None); return (w.sum()**2)/max((w**2).sum(),1e-30)
def rk(w,tol=0.05):
    w=np.clip(w,0,None); return int((w>tol*w.max()).sum())
GA=Gm[:7,:7]; GX=Gm[7:,7:]; GAX=Gm[:7,7:]
wa=np.linalg.eigvalsh(GA)[::-1]; wx=np.linalg.eigvalsh(GX)[::-1]; wt=np.linalg.eigvalsh(Gm)[::-1]
GXp=GX-GAX.T@np.linalg.pinv(GA)@GAX          # grade-2 with grade-1 projected out
wp=np.linalg.eigvalsh(GXp)[::-1]
print("="*72); print("  DOES THE ALGEBRA CLOSE ON A LARGER SPACE?"); print("="*72)
print(f"  grade-1 (7 node actions)   rank(5%) {rk(wa):>2}   eff-rank {erank(wa):.2f}")
print(f"  grade-2 (21 cross terms)   rank(5%) {rk(wx):>2}   eff-rank {erank(wx):.2f}")
print(f"  grade-2 modulo grade-1     rank(5%) {rk(wp):>2}   eff-rank {erank(wp):.2f}")
print(f"    energy of grade-2 outside grade-1: {100*np.trace(GXp)/np.trace(GX):.1f}%")
print(f"  grade-1 + grade-2 together rank(5%) {rk(wt):>2} of 28   eff-rank {erank(wt):.2f}")
print(f"    normalised spectrum: {' '.join(f'{x:.3f}' for x in (wt/wt[0])[:14])}")
c=np.cumsum(np.clip(wp,0,None))/np.clip(wp,0,None).sum()
print(f"    new directions for 50% / 90% of the out-of-span energy: "
      f"{int(np.searchsorted(c,0.5))+1} / {int(np.searchsorted(c,0.9))+1}")
print("\n  time %.1fs"%(time.time()-t0))
