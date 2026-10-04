import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
os.environ.setdefault("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
import run as R
def ard(X, y, iters=300):
    """ARD linear regression, MacKay/Tipping evidence fixed point (closed-form updates)."""
    m=y.mean(); yc=y-m; n,p=X.shape; a=np.ones(p); beta=1/np.var(yc); XtX=X.T@X; Xty=X.T@yc
    for _ in range(iters):
        S=np.linalg.inv(beta*XtX+np.diag(a)); mu=beta*S@Xty; g=1-a*np.diag(S)
        an=np.minimum(g/(mu**2+1e-12),1e8); r=yc-X@mu; beta=(n-g.sum())/(r@r)
        if np.max(np.abs(np.log(an)-np.log(a)))<1e-4: a=an; break
        a=an
    return m, mu, a
def krr_loo(Ktr,y,Kte,lams=np.logspace(-2,4,25)):
    m=y.mean(); yc=y-m; ev,V=np.linalg.eigh(Ktr); ev=np.maximum(ev,0); yt=V.T@yc; best=None
    for lam in lams:
        d=ev/(ev+lam); loo=(yc-V@(d*yt))/(1-(V**2)@d); e=np.mean(loo**2)
        if best is None or e<best[0]: best=(e,lam)
    return best[0], m+Kte@(V@(yt/(ev+best[1])))
res={}

for seed in range(10):
    W=R.world(seed); c,q=W["ctx"],W["qry"]; X=W["X"]; T=W["T"][c]; Y=W["Y"][c]
    Lo,Uo=R.manski(W["q"].mean(1)); y=(Y==T).astype(float); out={}
    out["const"]=np.full(len(q),y.mean())
    _,p=krr_loo(X[c]@X[c].T,y,X[q]@X[c].T); out["krr-lin"]=p
    m,mu,a=ard(X[c],y); out["ARD"]=m+X[q]@mu
    keep=np.where(a<1e3)[0]
    # relevance-scaled kernels on kept coords
    w=np.sqrt(1/a[keep]); w=w/w.max(); Xs=X[c][:,keep]*w; Xqs=X[q][:,keep]*w
    for g in (0.1,0.3,1.0):
        def K(A,B):
            d2=(A**2).sum(1)[:,None]+(B**2).sum(1)[None,:]-2*A@B.T; return A@B.T+np.exp(-g*d2/ max(len(keep),1))*len(keep)
        e,p=krr_loo(K(Xs,Xs),y,K(Xqs,Xs)); out[f"ARD+lin+rbf{g}"]=(e,p)
    ks=[k for k in out if k.startswith("ARD+")]; kb=min(ks,key=lambda k:out[k][0]); out["ARD+kernel(LOO)"]=out[kb][1]
    for k in ks: out[k]=out[k][1]
    sc={k:float(np.sqrt(np.mean((np.clip(v,0,1)-Uo)**2))) for k,v in out.items()}
    for k,v in sc.items(): res.setdefault(k,[]).append(v)
    print(seed,W["family"][:10],"kept",len(keep),{k:round(v,4) for k,v in sc.items()},flush=True)
print({k:round(np.mean(v),4) for k,v in res.items()})
