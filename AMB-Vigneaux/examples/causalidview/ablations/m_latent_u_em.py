import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/helpers.py').read())
def wlr(Z, y, w, lam, b0=None, iters=30):
    n,p=Z.shape; b=np.zeros(p) if b0 is None else b0.copy(); P=np.eye(p)*lam; P[0,0]=0
    for _ in range(iters):
        mu=sigm(Z@b); W=w*mu*(1-mu)+1e-9; g=Z.T@(w*(y-mu))-P@b; H=(Z*W[:,None]).T@Z+P; st=np.linalg.solve(H,g); b+=st
        if np.abs(st).max()<1e-7: break
    return b
def lcm_fit(X,T,Y,I,lam=30.0,iters=60,seed=0):
    """latent U in {-1,+1}: T ~ logit(a0 + X a + g u + d I); Y ~ logit(c0 + X c + k u + T (h0 + X h)). EM with IRLS M-steps."""
    n,d=X.shape; one=np.ones((n,1)); rng=np.random.default_rng(seed)
    ZT=lambda u: np.hstack([one,X,np.full((n,1),u),I[:,None]])
    ZY=lambda u: np.hstack([one,X,np.full((n,1),u),T[:,None],X*T[:,None]])
    aT=np.zeros(d+3); aT[d+1]=0.5*rng.choice([-1,1]); aY=np.zeros(2*d+3); aY[d+1]=0.5
    for it in range(iters):
        ll=[]
        for u in (-1,1):
            pT=sigm(ZT(u)@aT); pY=sigm(ZY(u)@aY)
            ll.append(np.log(0.5)+T*np.log(pT+1e-12)+(1-T)*np.log(1-pT+1e-12)+Y*np.log(pY+1e-12)+(1-Y)*np.log(1-pY+1e-12))
        ll=np.stack(ll); m=ll.max(0); R=np.exp(ll-m); R/=R.sum(0)                  # posterior of u
        Zt=np.vstack([ZT(-1),ZT(1)]); Zy=np.vstack([ZY(-1),ZY(1)]); w=np.concatenate([R[0],R[1]])
        aT=wlr(Zt,np.concatenate([T,T]),w,lam,aT); aY=wlr(Zy,np.concatenate([Y,Y]),w,lam,aY)
    return aT,aY
def lcm_cells(aT,aY,Xq):
    n=len(Xq); one=np.ones((n,1)); q=np.zeros((n,2,2,2))
    for u in (-1,1):
        for j in (0,1):
            e=sigm(np.hstack([one,Xq,np.full((n,1),u),np.full((n,1),j)])@aT)
            for t in (0,1):
                py=sigm(np.hstack([one,Xq,np.full((n,1),u),np.full((n,1),t),Xq*t])@aY); pt=e if t else 1-e
                q[:,j,t,1]+=0.5*pt*py; q[:,j,t,0]+=0.5*pt*(1-py)
    return q
res=[]
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c].astype(float); Y=W['Y'][c].astype(float); I=W['I'][c].astype(float)
    qm=W['q'].mean(1); s_or=qm[:,1,1]+qm[:,0,0]; rm=lambda s: np.sqrt(np.mean((s-s_or)**2)); nq=len(q)
    ivo=np.array([r.iv_bounds(W['q'][i]) for i in range(nq)])
    out=[]
    for lam in (10,30,100):
        aT,aY=lcm_fit(X[c],T,Y,I,lam=lam); qc=lcm_cells(aT,aY,X[q])
        sm=qc.mean(1)[:,1,1]+qc.mean(1)[:,0,0]
        iv=np.array([r.iv_bounds(qc[i]) for i in range(nq)])
        out+= [rm(sm), np.sqrt(0.5*np.mean((iv[:,0]-ivo[:,0])**2+(iv[:,1]-ivo[:,1])**2))]
    res.append(out); print(seed,W['family'][:10],' '.join(f"lam{l}: M {out[2*i]:.3f} IV {out[2*i+1]:.3f}" for i,l in enumerate((10,30,100))),flush=True)
print('mean',np.round(np.mean(res,0),4))
