"""
IS THE r-CONTROLLED HOLONOMY NON-ABELIAN?
CSCT needs [M_i, M_j] != 0 (ordered transport matters).
If the M_i share an eigenbasis they commute, holonomy is an abelian phase,
and the "monodromy from noncommutativity" mechanism is trivial.
Reuse the frieze edge maps; measure commutator norms and eigenbasis spread.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def state(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
P=flat().numel(); NT=256; res=torch.arange(P)%NT; W=2
FRZ=[torch.nonzero(((res-i)%NT)<W,as_tuple=True)[0] for i in range(NT)]
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
XT=[]; prev=flat()
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt(); r=m/(v+1e-12)
    XT.append(np.array([[float(torch.log(d[I].abs().mean()+1e-30)),
                         float(torch.log(r[I].mean()+1e-30))] for I in FRZ]))
    prev=af; del b4,af,d,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
X=np.stack(XT)
def fitM(A,B): M,_,_,_=np.linalg.lstsq(A,B,rcond=None); return M.T
Ms=np.array([fitM(X[:,i,:], X[:,(i+1)%NT,:]) for i in range(NT)])
print("\n"+"="*74); print("  ARE THE EDGE MAPS NON-COMMUTING?"); print("="*74)
# commutator norms relative to the maps
comm=[]; rng=np.random.default_rng(0)
for _ in range(2000):
    i,j=rng.integers(0,NT,2)
    Mi,Mj=Ms[i],Ms[j]
    c=np.linalg.norm(Mi@Mj-Mj@Mi)
    scale=np.linalg.norm(Mi)*np.linalg.norm(Mj)
    comm.append(c/max(scale,1e-30))
comm=np.array(comm)
print(f"  ||[M_i,M_j]|| / (||M_i||||M_j||):  mean {comm.mean():.4f}  median {np.median(comm):.4f}"
      f"  max {comm.max():.4f}")
# eigenbasis spread: angle of the dominant/second eigenvector across edges
V1=[]; V2=[]
for i in range(NT):
    w,V=np.linalg.eig(Ms[i]); od=np.argsort(-np.abs(w))
    v1=np.real(V[:,od[0]]); v2=np.real(V[:,od[1]])
    V1.append(v1/np.linalg.norm(v1)); V2.append(v2/np.linalg.norm(v2))
V1=np.array(V1); V2=np.array(V2)
def spread(Vs):
    Vs=Vs*np.sign(Vs[:,[0]]+1e-12)
    m=Vs.mean(0); m/=np.linalg.norm(m)
    ang=np.degrees(np.arccos(np.clip(np.abs(Vs@m),0,1)))
    return ang.mean(), ang.std()
a1=spread(V1); a2=spread(V2)
print(f"\n  eigenvector spread across the 256 edges (deg from mean):")
print(f"    dominant (beta) eigenvector: {a1[0]:.1f} +/- {a1[1]:.1f}")
print(f"    second   (alpha) eigenvector: {a2[0]:.1f} +/- {a2[1]:.1f}")
# abelian test: does prod of eigenvalues (order-free) reproduce holonomy eigenvalues?
He=np.eye(2)
for i in range(NT): He=Ms[i]@He
prod_ev=np.prod([np.linalg.eigvals(Ms[i]) for i in range(NT)],axis=0)
print(f"\n  holonomy |ev|                 = {np.round(np.sort(np.abs(np.linalg.eigvals(He)))[::-1],4)}")
print(f"  order-free product of |ev|    = {np.round(np.sort(np.abs(prod_ev))[::-1],4)}")
print(f"  (if equal, the holonomy is abelian: ordering did not matter)")
print(f"\n  VERDICT:")
if comm.mean()<0.02:
    print("   commutators ~0 => maps commute => holonomy is an ABELIAN phase.")
    print("   The 'monodromy from noncommutativity' mechanism is trivial.")
else:
    print("   commutators nonzero => genuine non-abelian transport => CSCT has footing.")
print(f"\n  time {time.time()-t0:.0f}s")
