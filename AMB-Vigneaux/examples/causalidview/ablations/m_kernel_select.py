import os, sys, numpy as np, time
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
os.environ.setdefault("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
import run as R
LAMS = np.logspace(-2, 4, 25)
def kernels(Xa, Xb, D):
    G = Xa @ Xb.T; na = (Xa**2).sum(1); nb = (Xb**2).sum(1); d2 = na[:,None]+nb[None,:]-2*G
    out = {"lin": G}
    for g in (0.25, 0.5, 1.0):
        out[f"rbf{g}"] = np.exp(-g*d2/(2*D))*D   # scaled to comparable trace
        out[f"lin+rbf{g}"] = G + np.exp(-g*d2/(2*D))*D
    out["poly2"] = (1+G/D)**2*D
    return out
def krr_loo(Ktr, y, Kte, sel=None):
    m=y.mean(); yc=y-m; ev,V=np.linalg.eigh(Ktr); ev=np.maximum(ev,0); yt=V.T@yc; best=None
    for lam in LAMS:
        d=ev/(ev+lam); loo=(yc-V@(d*yt))/(1-(V**2)@d); e=np.mean(loo**2)
        if best is None or e<best[0]: best=(e,lam)
    a=V@(yt/(ev+best[1])); return best[0], m+Kte@a
res={}
for seed in range(8):
    W=R.world(seed); c,q=W["ctx"],W["qry"]; X=W["X"]; T=W["T"][c]; Y=W["Y"][c]
    Lo,Uo=R.manski(W["q"].mean(1)); s_or=Uo
    Ktr=kernels(X[c],X[c],50); Kte=kernels(X[q],X[c],50); y=(Y==T).astype(float)
    sc={}
    for k in Ktr:
        e,p=krr_loo(Ktr[k],y,Kte[k]); p=np.clip(p,0,1); sc[k]=(e,np.sqrt(np.mean((p-s_or)**2)))
    kb=min(sc,key=lambda k:sc[k][0]); sc["LOO-select"]=(sc[kb][0],sc[kb][1])
    # LOO-weighted average
    for k,v in sc.items(): res.setdefault(k,[]).append(v[1])
    print(seed,W["family"],{k:round(v[1],4) for k,v in sc.items()},"pick",kb,flush=True)
print({k:round(np.mean(v),4) for k,v in res.items()})
