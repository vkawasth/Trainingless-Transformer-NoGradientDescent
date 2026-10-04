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
