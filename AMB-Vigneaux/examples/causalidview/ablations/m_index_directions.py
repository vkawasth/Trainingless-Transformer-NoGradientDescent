import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
os.environ.setdefault("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
import run as R
from ard_lib import ard, krr_loo
def rbf(A,B,g):
    d2=(A**2).sum(1)[:,None]+(B**2).sum(1)[None,:]-2*A@B.T; return np.exp(-g*d2)
res={}
NW=int(os.environ.get("NW",10))
for seed in range(NW):
    W=R.world(seed); c,q=W["ctx"],W["qry"]; X=W["X"]; T=W["T"][c].astype(float); Y=W["Y"][c].astype(float); I=W["I"][c].astype(float)
    Lo,Uo=R.manski(W["q"].mean(1)); y=(Y==T).astype(float); out={}
    _,p=krr_loo(X[c]@X[c].T,y,X[q]@X[c].T); out["krr-lin"]=(0,p)
    Zc=np.hstack([R.np.ones((len(c),1)),X[c],I[:,None]])
    bT=R._cv_logistic(Zc,T); dT=bT[1:-1]
    dirs={"T":dT}
    for t in (0,1):
        m,mu,a=ard(X[c][T==t],Y[T==t]); dirs[f"Y{t}"]=mu
    m,mu,a=ard(X[c],y); dirs["s"]=mu
    for name,keys in {"T,Y0,Y1":["T","Y0","Y1"],"T,Y0,Y1,s":["T","Y0","Y1","s"],"T,s":["T","s"]}.items():
        Dm=np.stack([dirs[k]/np.linalg.norm(dirs[k]) for k in keys],1)
        Fc=X[c]@Dm; Fq=X[q]@Dm; sd=Fc.std(0); Fc/=sd; Fq/=sd
        best=None
        for g in (0.05,0.1,0.2,0.4,0.8):
            for wl in (0.0,1.0):
                Kc=rbf(Fc,Fc,g)*len(keys)+wl*X[c]@X[c].T/ X.shape[1]*len(keys); Kq=rbf(Fq,Fc,g)*len(keys)+wl*X[q]@X[c].T/X.shape[1]*len(keys)
                e,p=krr_loo(Kc,y,Kq)
                if best is None or e<best[0]: best=(e,p,g,wl)
        out["idx["+name+"]"]=(best[0],best[1])
    sc={k:float(np.sqrt(np.mean((np.clip(v[1],0,1)-Uo)**2))) for k,v in out.items()}
    for k,v in sc.items(): res.setdefault(k,[]).append(v)
    print(seed,W["family"][:10],{k:round(v,4) for k,v in sc.items()},flush=True)
print({k:round(np.mean(v),4) for k,v in res.items()})
