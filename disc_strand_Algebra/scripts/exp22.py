import time, re, copy, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
named=list(model.named_parameters())
def keyL(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
def keyLT(n):
    m=re.match(r"blocks\.(\d+)\.",n)
    if not m: return "EMB"
    l=m.group(1); nl=n.lower()
    t=("WQ" if "wq" in nl else "WK" if "wk" in nl else "WV" if "wv" in nl else
       "WO" if ".op." in nl else "LN" if (".ln." in nl or ".n." in nl) else "FF")
    return f"L{l}.{t}"
def build(keyf):
    ks=[]; sl={}
    off=0
    for n,p in named:
        k=keyf(n)
        if k not in sl: sl[k]=[]; ks.append(k)
        sl[k].append((off,off+p.numel())); off+=p.numel()
    idx=[torch.cat([torch.arange(a,b) for a,b in sl[k]]) for k in ks]
    return ks,idx,off
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def G(v):
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
def V(n=12): return float(eval_val(model,n=n))
ck=torch.load("J160.pt"); th=ck["th"]; H=1e-4
print("="*76); print("  EXPLOITATION: RANK-1-PER-BLOCK REDUCED NEWTON STEP"); print("="*76)
print("  Within block l the curvature response is rank 1: P_l H v ~ u_l (w_l . v).")
print("  So g + H d = 0 collapses to a K x K system on the block coefficients.\n")
for name,keyf in [("layer only (K=7)",keyL),("layer x type (K=43)",keyLT)]:
    ks,IDX,P=build(keyf); K=len(ks)
    g0=G(th); nV=[float(g0[I].norm()) for I in IDX]
    A=[]                                   # A[m] = H * Vhat_m
    for m in range(K):
        d=torch.zeros_like(th); d[IDX[m]]=-g0[IDX[m]]/max(nV[m],1e-30)
        A.append((G(th+H*d)-g0)/H)
    u=[]; resid=[]
    for l in range(K):
        Mt=torch.stack([A[m][IDX[l]] for m in range(K)])
        U,S,Vt=torch.linalg.svd(Mt,full_matrices=False)
        u.append(Vt[0]); 
        pg=-g0[IDX[l]]
        resid.append(float((pg-(pg@Vt[0])*Vt[0]).norm()/max(pg.norm(),1e-30)))
    C=np.array([[float(A[m][IDX[l]]@u[l]) for m in range(K)] for l in range(K)])
    b=np.array([float((-g0[IDX[l]])@u[l]) for l in range(K)])
    a=np.linalg.lstsq(C,-b,rcond=1e-6)[0]
    d=torch.zeros_like(th)
    for m in range(K): d[IDX[m]]=-g0[IDX[m]]/max(nV[m],1e-30)*float(a[m])
    setflat(th); v0=V()
    print(f"  {name}:  ||g||={float(g0.norm()):.4g}  mean out-of-u residual "
          f"{100*np.mean(resid):.1f}%   cond(C)={np.linalg.cond(C):.3g}")
    best=(1e9,None)
    for s in [0.05,0.1,0.25,0.5,1.0]:
        setflat(th+s*d); vv=V()
        if vv<best[0]: best=(vv,s)
        print(f"      scale {s:>5.2f}   ||step||={float(s*d.norm()):>8.4f}   val {vv:.4f}"
              f"{'   <-- best' if vv==best[0] else ''}", flush=True)
    # baseline: plain AdamW steps from the same point
    setflat(th)
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    opt.load_state_dict(copy.deepcopy(ck["od"])); torch.manual_seed(5)
    base=[]
    for k in range(1,13):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if k in (1,2,4,8,12): base.append((k,V()))
    eq=next((k for k,vv in base if vv<=best[0]), None)
    print(f"      baseline from val {v0:.4f}: "+"  ".join(f"{k}:{vv:.4f}" for k,vv in base))
    print(f"      >>> reduced-Newton val {best[0]:.4f} at scale {best[1]}"
          f"  ==  {eq if eq else '>12'} AdamW steps, cost {K+1} gradient evals\n", flush=True)
print("  time %.0fs"%(time.time()-t0))
