"""GAUGE FRACTION WITHOUT THE 17 GB BASIS.

The gauge subspace is G = {(Q X, -K X^T) : X in gl(dh) per head}. Building an
explicit orthonormal basis needs a 131072 x 16384 QR. It is unnecessary: the
projection of C = (dQ, dK) onto G is the least-squares solution of

    min_X || (Q X - dQ, -K X^T - dK) ||^2

which, per head, is a linear problem in the dh^2 unknowns of X with normal
equations

    (Q^T Q) X + X (K^T K) = Q^T dQ - (K^T dK)^T          (a Sylvester equation)

solved by diagonalising Q^T Q = U a U^T and K^T K = V b V^T, giving
X = U [ (U^T R V) / (a_i + b_j) ] V^T.  Exact, and dh x dh throughout.

The gauge fraction is then ||(QX, -K X^T)||^2 / ||C||^2, and the null is the
same quantity for a random C of the same shape -- the fraction of a random
vector that any subspace of this dimension captures.
"""
import torch, numpy as np
d=torch.load("wq_snap.pt",weights_only=False)
CH=d["ch"]; W40=d["w40"]; H=4
def gauge_frac(Q,K,dQ,dK):
    D=Q.shape[0]; dh=D//H; num=0.0
    for h in range(H):
        s=slice(h*dh,(h+1)*dh)
        q=Q[:,s].double(); k=K[:,s].double()
        dq=dQ[:,s].double(); dk=dK[:,s].double()
        A=q.T@q; B=k.T@k
        R=q.T@dq-(k.T@dk).T
        a,U=torch.linalg.eigh(A); b,V=torch.linalg.eigh(B)
        Rt=U.T@R@V
        X=U@(Rt/(a[:,None]+b[None,:]+1e-12))@V.T
        num+=float(((q@X)**2).sum()+((k@X.T)**2).sum())
    return num
print(f"  {'block':<24}{'gauge':>9}{'null':>9}{'excess':>9}")
gs=[];ns=[]
for kq in sorted(CH):
    kk=kq.replace("WQ","WK")
    if kk not in W40: continue
    Q,K=W40[kq],W40[kk]
    dQ=CH[kq]; dK=CH.get(kk,torch.zeros_like(dQ))
    den=float((dQ.double()**2).sum()+(dK.double()**2).sum())
    gf=gauge_frac(Q,K,dQ,dK)/max(den,1e-30)
    g=torch.Generator().manual_seed(7); nl=[]
    for _ in range(5):
        rq=torch.randn(dQ.shape,generator=g); rk=torch.randn(dK.shape,generator=g)
        dn=float((rq.double()**2).sum()+(rk.double()**2).sum())
        nl.append(gauge_frac(Q,K,rq,rk)/max(dn,1e-30))
    mu=float(np.mean(nl)); gs.append(gf); ns.append(mu)
    print(f"  {kq:<24}{gf:>9.4f}{mu:>9.4f}{gf-mu:>+9.4f}",flush=True)
print(f"\n  mean gauge {np.mean(gs):.4f}   mean null {np.mean(ns):.4f}"
      f"   excess {np.mean(gs)-np.mean(ns):+.4f}")
print(f"  dim G / ambient = {H*(256//H)**2}/{2*256*256} = {H*(256//H)**2/(2*256*256):.4f}")
