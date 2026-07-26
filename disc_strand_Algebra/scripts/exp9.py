import time, itertools, numpy as np, torch
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
torch.manual_seed(3); XB=[get_batch() for _ in range(6)]
def logits(vec):
    setflat(vec); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
L0=logits(th); D={g: sub(DT,[g]) for g in NODES}
df=[(logits(th+D[g])-L0).double() for g in NODES]
Xs=[]; pairs=[]
for i,j in itertools.combinations(range(7),2):
    Xs.append((logits(th+D[NODES[i]]+D[NODES[j]])-L0).double()-df[i]-df[j]); pairs.append((i,j))
A=torch.stack(df).numpy(); Xm=torch.stack(Xs).numpy()
def erank(s): l=s**2; return (l.sum()**2)/((l**2).sum())
sa=np.linalg.svd(A,compute_uv=False)
sx=np.linalg.svd(Xm,compute_uv=False)
print("="*74); print("  DOES THE ALGEBRA CLOSE ON A LARGER SPACE?"); print("="*74)
print(f"  grade-1 (7 node actions)  : rank(5%) {int((sa>0.05*sa[0]).sum())}  eff-rank {erank(sa):.2f}")
print(f"  grade-2 (21 cross terms)  : rank(5%) {int((sx>0.05*sx[0]).sum())}  eff-rank {erank(sx):.2f}")
Q,_=np.linalg.qr(A.T)                      # orthonormal basis of grade-1
Xp=Xm-(Xm@Q)@Q.T                            # cross terms with grade-1 removed
sp=np.linalg.svd(Xp,compute_uv=False)
print(f"  grade-2 modulo grade-1    : rank(5%) {int((sp>0.05*sp[0]).sum())}  eff-rank {erank(sp):.2f}")
print(f"    energy of grade-2 outside grade-1: {100*float((Xp**2).sum()/(Xm**2).sum()):.1f}%")
tot=np.concatenate([A,Xm],0); st=np.linalg.svd(tot,compute_uv=False)
print(f"  grade-1 + grade-2 together: rank(5%) {int((st>0.05*st[0]).sum())} of 28  eff-rank {erank(st):.2f}")
print(f"    normalised spectrum: {' '.join(f'{x:.3f}' for x in (st/st[0])[:14])}")
# how many new directions carry most of the new energy?
c=np.cumsum(sp**2)/ (sp**2).sum()
k50=int(np.searchsorted(c,0.5))+1; k90=int(np.searchsorted(c,0.9))+1
print(f"    new directions carrying 50% / 90% of out-of-span energy: {k50} / {k90}")
print("\n  time %.1fs"%(time.time()-t0))
