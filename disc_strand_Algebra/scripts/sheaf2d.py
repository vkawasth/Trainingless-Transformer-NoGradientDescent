"""
2-D metric cellular sheaf over the tile tree.
 stalk   : (alpha,beta) from log-power-law  log|u| = alpha*log(L-H) + beta   -> R^2 (vector space)
 metric  : diagonal Fisher aggregated per tile -> inner product on each stalk
 adjacency (2-D):
    horizontal : adjacent tiles within a layer  (overlap)
    vertical   : refinement parent -> children  (COPRODUCT, one-in / r-out)
 pants   : coproduct Delta_l : S(p) -> S(c1)+...+S(cr)   product mu_l : (+)S(c) -> S(p)
"""
import pickle, numpy as np
np.set_printoptions(precision=3,suppress=True)
D=pickle.load(open("tiletree.pkl","rb")); S=D["series"]; META=D["meta"]
FIS=pickle.load(open("fisher.pkl","rb"))
H=0.062
loss=np.array([m["loss"] for m in META]); steps=np.array([m["step"] for m in META])
LAY=["L0","L1","L2","L3","L4","L5"]
FWD={"L0":16,"L1":32,"L2":64,"L3":256,"L4":512,"L5":1024}
BWD={"L0":1024,"L1":512,"L2":256,"L3":64,"L4":32,"L5":16}

def fit_group(gk, side):
    """returns per-layer arrays: X (ntiles,2) = (alpha,beta), R2, W (fisher weight)"""
    res={}
    for L in LAY:
        k=f"{L}|{gk}|{side}"
        if k not in S: continue
        A=np.abs(S[k]["max"]).astype(np.float64)      # positive channel
        f0=S[k]["first_step"]; n=S[k]["nt"]
        st=np.arange(f0, f0+A.shape[0])
        m=(st<=steps.max())
        ls=loss[st[m]-1]-H
        ok=(ls>1e-6)
        x=np.log(ls[ok]); Y=np.log(np.maximum(A[m][ok],1e-30))
        xm=x-x.mean(); den=float((xm*xm).sum())
        al=(Y-Y.mean(0)).T@xm/den                       # slope per tile
        be=Y.mean(0)-al*x.mean()
        pred=np.outer(x,al)+be
        ssr=((Y-pred)**2).sum(0); sst=((Y-Y.mean(0))**2).sum(0)
        R2=1-ssr/np.maximum(sst,1e-30)
        # fisher weight per tile: aggregate diag Fisher over the tile's index span
        fv=FIS[f"{L}|{gk}"]; nel=len(fv)
        edges=np.linspace(0,nel,n+1).astype(int)
        w=np.array([fv[edges[i]:edges[i+1]].sum() for i in range(n)])
        res[L]=dict(X=np.stack([al,be],1), R2=R2, W=w/max(w.sum(),1e-30), n=n)
    return res

def refine_maps(res, order):
    """fit 2x2 restriction M_l : parent stalk -> child stalk, Fisher-weighted LS"""
    Ms={}; info={}
    for a,b in zip(order[:-1],order[1:]):
        if a not in res or b not in res: continue
        na,nb=res[a]["n"],res[b]["n"]
        if nb % na: continue
        r=nb//na
        Xp=np.repeat(res[a]["X"],r,axis=0); Xc=res[b]["X"]
        w=np.repeat(res[a]["W"],r)*res[b]["W"]; w=w/w.sum()
        Aw=Xp*np.sqrt(w)[:,None]; Bw=Xc*np.sqrt(w)[:,None]
        M,_,_,_=np.linalg.lstsq(Aw,Bw,rcond=None)      # Xc ~ Xp @ M
        pred=Xp@M
        rel=np.sqrt((w[:,None]*(Xc-pred)**2).sum()/max((w[:,None]*Xc**2).sum(),1e-30))
        Ms[(a,b)]=M.T; info[(a,b)]=dict(r=r, rel=rel)
    return Ms, info

print("="*84); print("  METRIC CELLULAR SHEAF ON THE TILE TREE   stalk = (alpha,beta) in R^2"); print("="*84)
print(f"  power law: log|u_tile| = alpha*log(L-{H}) + beta      (log => stalk is a vector space)")
for gk in ("FF","LN","W_K"):
    for side in ("fwd","bwd"):
        res=fit_group(gk,side)
        if not res: continue
        order=LAY if side=="fwd" else LAY[::-1]
        print(f"\n  --- {gk} {side}   (coarse -> fine order: {' -> '.join(order)})")
        print(f"  {'layer':>6}{'ntiles':>8}{'alpha':>10}{'sd(a)':>8}{'beta':>10}{'R2':>8}{'fisher%':>9}")
        for L in order:
            if L not in res: continue
            r=res[L]; a=r["X"][:,0]; b=r["X"][:,1]
            print(f"  {L:>6}{r['n']:>8}{a.mean():>10.3f}{a.std():>8.3f}{b.mean():>10.3f}"
                  f"{np.median(r['R2']):>8.3f}{100*FIS[f'{L}|{gk}'].sum()/sum(FIS[f'{x}|{gk}'].sum() for x in LAY):>8.1f}%")
        Ms,info=refine_maps(res,order)
        if Ms:
            print(f"  restriction maps  M : S(parent) -> S(child)   [COPRODUCT, 1 -> r]")
            for (a,b),M in Ms.items():
                ev=np.linalg.eigvals(M)
                print(f"    {a}->{b}  r={info[(a,b)]['r']}  M=[[{M[0,0]:.3f},{M[0,1]:.3f}],"
                      f"[{M[1,0]:.3f},{M[1,1]:.3f}]]  |ev|={np.abs(ev)}  rel.resid {info[(a,b)]['rel']:.3f}")
        globals().setdefault("STORE",{})[(gk,side)]=(res,Ms,info,order)
pickle.dump({k:(v[0],v[1],v[2],v[3]) for k,v in STORE.items()}, open("sheaf2d.pkl","wb"))
print("\n  saved sheaf2d.pkl")
