"""O2 checks on the true grammar of Corpus B:
 (1) reachable-symbol ranks per level (unreachable symbols removed);
 (2) root collapse: held-out log-lik with the 8-symbol root vs ONE root symbol with row mu,
     and the complete-data Bregman merge cost of that collapse (= I(B_root; children));
 (3) 8->2 root merges (the paper's level-4 result): exact observed loss vs Bregman cost."""
import json, itertools, numpy as np
M=json.load(open('cH/rhm_meta.json')); V,A,L=M['nsym'],M['nleaf'],M['depth']
P={}
for l in range(1,L+1):
    c=A if l==1 else V; t=np.zeros((V,c,c))
    for s,r in M['rules_all'][str(l)].items():
        for x,y in r: t[int(s),x,y]+=1/len(r)
    P[l]=t
pi=np.full(V,1/V)
# reachability
reach={L:np.ones(V,bool)}
for l in range(L,1,-1):
    w=np.einsum('p,pxy->xy',reach[l].astype(float),P[l]); reach[l-1]=(w.sum(0)+w.sum(1))>0
for l in range(1,L+1): print(f"level {l}: reachable {reach[l].sum()}/{V}  unreachable {np.where(~reach[l])[0].tolist()}")
X=np.array(json.load(open('cH/val_ids.json'))).reshape(-1,2**L)
def inside(x, rootrow=None):
    beta=np.eye(A)[x]                       # (n_leaves, A)
    for l in range(1,L+1):
        pairs=beta.reshape(-1,2,beta.shape[1])
        beta=np.einsum('bxy,nx,ny->nb',P[l],pairs[:,0],pairs[:,1])
    return beta[0]                          # P(x | root=b)
lik=np.array([inside(x) for x in X])        # (N, V)
ll_full=np.log(lik@pi).mean()
mu=np.einsum('b,bxy->xy',pi,P[L])
# collapsed root: one symbol with row mu; recompute directly from level-(L-1) betas
def top_beta(x):
    beta=np.eye(A)[x]
    for l in range(1,L):
        pairs=beta.reshape(-1,2,beta.shape[1]); beta=np.einsum('bxy,nx,ny->nb',P[l],pairs[:,0],pairs[:,1])
    return beta
ll_coll=np.mean([np.log(np.einsum('xy,x,y->',mu,*top_beta(x))) for x in X])
def KL(p,q): m=p>0; return (p[m]*np.log(p[m]/q[m])).sum()
cost_all=sum(pi[b]*KL(P[L][b].ravel(),mu.ravel()) for b in range(V))
print(f"\nroot collapse 8->1: held-out ll/seq full {ll_full:.6f}  collapsed {ll_coll:.6f}  diff {ll_coll-ll_full:.2e}")
print(f"  complete-data Bregman cost = I(B_root; children) = {cost_all:.4f} nats per root node")
# all 2-block partitions of the root: observed loss vs Bregman cost (both are lossless observationally)
res=[]
for mask in range(1,2**(V-1)):
    blocks=[[b for b in range(V) if (mask>>b)&1],[b for b in range(V) if not (mask>>b)&1]]
    cost=0; ll=0
    newpi=[]; rows=[]
    for Bk in blocks:
        w=pi[Bk].sum(); r=np.einsum('b,bxy->xy',pi[Bk]/w,P[L][Bk]); newpi.append(w); rows.append(r)
        cost+=sum(pi[b]*KL(P[L][b].ravel(),r.ravel()) for b in Bk)
    res.append(cost)
print(f"  8->2 root partitions: {len(res)}; Bregman cost min {min(res):.4f} max {max(res):.4f} nats/root node;"
      " observed-data loss 0 for every one (same marginal mu)")
