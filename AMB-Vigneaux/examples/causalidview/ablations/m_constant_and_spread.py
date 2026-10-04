import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
exec(open('/tmp/claude-0/-home-claude/bc7a5435-f2c5-5a55-9ebf-abe395f9a702/scratchpad/proto.py').read().split('res=[]')[0])
res=[]
LAMS=(10,30,100,300,1000,3000,10000)
for seed in range(8):
    W=r.world(seed); c,q=W['ctx'],W['qry']; X=W['X']; T=W['T'][c]; Y=W['Y'][c]
    s_or=(W['q'].mean(1)[:,1,1]+W['q'].mean(1)[:,0,0]); yb=(Y==T).astype(float)
    rm=lambda s: np.sqrt(np.mean((s-s_or)**2))
    const=np.full(len(q),yb.mean())
    Z1=np.hstack([np.ones((len(c),1)),X[c]]); Zq=np.hstack([np.ones((len(q),1)),X[q]])
    b1,l1=cv_fit(Z1,yb,lams=LAMS); s1=sigm(Zq@b1)
    # oracle-free spread check: sd of the oracle s
    res.append((rm(const),rm(s1),s_or.std())); print(seed,[round(v,3) for v in res[-1]],l1,flush=True)
print('mean',np.mean(res,0))
