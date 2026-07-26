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
P=off; NPAR=[sum(b-a for a,b in sl[g]) for g in NODES]
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES]
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
SUBS=[("ALL 7",list(range(7))),("skeleton EMB,LN,FF",[0,1,2]),
      ("attention Q,K,V,O",[3,4,5,6]),("Q,K only",[3,4]),("V,O only",[5,6])]
res={}
for ck in (80,160):
    th=torch.load(f"J{ck}.pt")["th"].double()
    g0=G(th); V=[-mk(g0,i) for i in range(7)]; nV=[float(v.norm()) for v in V]
    h=1e-4; Dg=[((G(th+h*V[i]/nV[i])-g0)/h) for i in range(7)]
    def brk(i,j): return -nV[i]*mk(Dg[i],j)+nV[j]*mk(Dg[j],i)
    print("="*76); print(f"  CHECKPOINT {ck}    ||grad||={float(g0.norm()):.4g}"); print("="*76)
    print(f"  ||V_i||: "+" ".join(f"{NODES[i]}:{nV[i]:.3g}" for i in range(7)))
    print(f"\n  {'subalgebra':<22}{'mean |[.,.]|':>14}{'in-span':>10}{'null':>12}{'vs null':>11}")
    for lab,S in SUBS:
        Vm=torch.stack([V[i] for i in S]); GV=(Vm@Vm.T).numpy(); GVi=np.linalg.pinv(GV)
        fr=[]; nb=[]; nul=[]
        for (i,j) in itertools.combinations(S,2):
            b=brk(i,j); c=(Vm@b).numpy()
            fr.append(float(c@GVi@c)/max(float(b@b),1e-300)); nb.append(float(b.norm()))
            nul.append(len(S)/(NPAR[i]+NPAR[j]))
        fr=np.array(fr); nul=np.array(nul)
        print(f"  {lab:<22}{np.mean(nb):>14.4g}{100*fr.mean():>9.1f}%{100*nul.mean():>11.5f}%"
              f"{np.mean(fr/nul):>10.0f}x")
        res[(ck,lab)]=(np.mean(nb),fr.mean())
    # QK detail
    i,j=3,4; b=brk(i,j)
    Vm=torch.stack([V[3],V[4]]); GV=(Vm@Vm.T).numpy()
    c=(Vm@b).numpy(); a=np.linalg.pinv(GV)@c
    print(f"\n  Q/K DETAIL:  [V_Q,V_K] = {a[0]:+.4f} V_Q {a[1]:+.4f} V_K + r,"
          f"   ||r||/||[.,.]|| = {np.sqrt(max(1-float(c@np.linalg.pinv(GV)@c)/float(b@b),0)):.3f}")
    print(f"    ||[V_Q,V_K]|| / (||V_Q|| ||V_K||) = {float(b.norm())/(nV[3]*nV[4]):.4g}")
    del V,Dg,g0; gc.collect()
print("="*76); print("  SKELETON -> DENSE TRANSITION"); print("="*76)
print(f"  {'subalgebra':<22}{'|[.,.]| @80':>14}{'|[.,.]| @160':>14}{'ratio':>9}"
      f"{'in-span 80':>12}{'in-span 160':>13}")
for lab,S in SUBS:
    a=res[(80,lab)]; b=res[(160,lab)]
    print(f"  {lab:<22}{a[0]:>14.4g}{b[0]:>14.4g}{b[0]/max(a[0],1e-30):>9.3f}"
          f"{100*a[1]:>11.1f}%{100*b[1]:>12.1f}%")
print("\n  time %.0fs"%(time.time()-t0))
