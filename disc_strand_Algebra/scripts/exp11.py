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
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
def logits_nograd(vec):
    setflat(vec); model.eval(); out=[]
    with torch.no_grad():
        for x,y in XB: lo,_=model(x,y); out.append(lo.reshape(-1).clone())
    return torch.cat(out)
def Jt(v):                      # J^T v at base point th : vector-Jacobian product
    setflat(th); model.eval(); model.zero_grad(set_to_none=True)
    i=0
    for x,y in XB:
        lo,_=model(x,y); n=lo.numel()
        (lo.reshape(-1)*v[i:i+n]).sum().backward(); i+=n
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
D=[sub(DT,[g]) for g in NODES]
Dm=torch.stack(D)                       # (7,P) grade-1 directions IN PARAMETER SPACE
GD=(Dm@Dm.T).double().numpy(); GDi=np.linalg.pinv(GD)
L0=logits_nograd(th); f1=[logits_nograd(th+D[i])-L0 for i in range(7)]
pairs=list(itertools.combinations(range(7),2))
print(f"  base done ({time.time()-t0:.0f}s)", flush=True)
rows=[]; NRM=[]; PROJ=[]
Gp=np.zeros((len(pairs),len(pairs)))
Gvecs=[]
for k,(i,j) in enumerate(pairs):
    m_ij = logits_nograd(th+D[i]+D[j])-L0-f1[i]-f1[j]     # in F
    g = Jt(m_ij)                                          # pulled back to P
    Gvecs.append(g)
    c = (Dm@g).double().numpy()
    proj = float(c@GDi@c); tot=float(g.double()@g.double())
    PROJ.append(proj/max(tot,1e-30)); NRM.append(float(g.norm()))
    rows.append((NODES[i],NODES[j],float(g.norm()),100*proj/max(tot,1e-30)))
print(f"  21 pullbacks done ({time.time()-t0:.0f}s)", flush=True)
for a in range(len(pairs)):
    for b in range(a,len(pairs)):
        Gp[a,b]=Gp[b,a]=float(Gvecs[a].double()@Gvecs[b].double())
del Gvecs; gc.collect()
print("\n"+"="*74)
print("  PULLED-BACK PRODUCT  m~ = J* o m : P (x) P -> P    (index lowered by J*)")
print("="*74)
print("  does m~(i,j) close on span{d_1..d_7} in PARAMETER space?")
print(f"  {'pair':>14}{'||m~||':>12}{'in-span %':>12}")
for a,b,n_,p_ in sorted(rows,key=lambda r:-r[3])[:5]: print(f"  {a+'-'+b:>14}{n_:>12.3g}{p_:>11.1f}%")
print("  ...")
for a,b,n_,p_ in sorted(rows,key=lambda r:-r[3])[-3:]: print(f"  {a+'-'+b:>14}{n_:>12.3g}{p_:>11.1f}%")
P=np.array(PROJ)
print(f"\n  mean in-span fraction: {100*P.mean():.1f}%   (min {100*P.min():.1f}, max {100*P.max():.1f})")
w=np.linalg.eigvalsh(Gp)[::-1]; wc=np.clip(w,0,None)
print(f"  rank(5%) of the 21 pulled-back products: {int((wc>0.05*wc.max()).sum())}"
      f"   eff-rank {(wc.sum()**2)/max((wc**2).sum(),1e-30):.2f}")
gd=np.linalg.eigvalsh(GD)[::-1]
print(f"  for comparison, grade-1 in P: rank(5%) {int((gd>0.05*gd.max()).sum())}"
      f"   eff-rank {(gd.sum()**2)/(gd**2).sum():.2f}")
print(f"  ||m~|| / ||d|| scale ratio: {np.mean(NRM)/float(Dm.norm(dim=1).mean()):.3g}")
print("\n  time %.1fs"%(time.time()-t0))
