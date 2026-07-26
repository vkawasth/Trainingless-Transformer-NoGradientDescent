"""
(A) DISENTANGLE AXES: for one weight matrix, compare flip-clustering under
    row (output/neuron), column (input feature), contiguous, and random covers
    -- all at MATCHED tile count and size.
(B) PER-EDGE HOLONOMY on the frieze cycle: fit each M_i independently from its
    own time series, then compare prod(M_i) with M_shared^NT.
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
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
P=off
TEN={"FF g (512,256)":("blocks.2.ff.g.weight",512,256),
     "W_K (256,256)":("blocks.2.attn.WK.weight",256,256)}
COV={}
for lab,(nm,R,C) in TEN.items():
    a0,_=SPAN[nm]; M=torch.arange(a0,a0+R*C).view(R,C)
    nt=C                                   # matched: C tiles of R elements
    rng=np.random.default_rng(0); perm=torch.tensor(rng.permutation(R*C))
    COV[lab]={
      "column (input axis)":[M[:,j].clone() for j in range(C)],
      "row (output/neuron)":[M[i,:].clone() for i in range(R)][:C] if R>=C else None,
      "contiguous memory"  :[M.flatten()[j*R:(j+1)*R].clone() for j in range(C)],
      "random"             :[(a0+perm[j*R:(j+1)*R]).clone() for j in range(C)],
    }
    COV[lab]={k:v for k,v in COV[lab].items() if v is not None}
NT=256; res=torch.arange(P)%NT; W=2
FRZ=[torch.nonzero(((res-i)%NT)<W,as_tuple=True)[0] for i in range(NT)]
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
OD={lab:{k:[] for k in COV[lab]} for lab in COV}
XT=[]   # frieze per-tile 2-vector over time
prev=flat(); sp=None
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; sg=torch.sign(d)
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt(); r=m/(v+1e-12)
    XT.append(np.array([[float(torch.log(d[I].abs().mean()+1e-30)),
                         float(torch.log(r[I].mean()+1e-30))] for I in FRZ]))
    if sp is not None and s>10:
        fl=(sg!=sp).float()
        for lab in COV:
            pb=float(fl[torch.cat(COV[lab]["random"])].mean())
            for k,C in COV[lab].items():
                pt=np.array([float(fl[I].mean()) for I in C]); n=len(C[0])
                OD[lab][k].append(pt.std()/max(np.sqrt(pb*(1-pb)/n),1e-30))
    sp=sg; del b4,af,d,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*82); print("  (A) WHAT IS THE FLIP TOPOLOGY ATTACHED TO?"); print("="*82)
for lab in COV:
    print(f"\n  {lab}   (matched tile count and size)")
    print(f"  {'cover':>24}{'over-dispersion':>18}")
    for k in COV[lab]:
        a=np.array(OD[lab][k]); print(f"  {k:>24}{a.mean():>16.2f}x")
X=np.stack(XT)                       # (T, NT, 2)
print("\n"+"="*82); print("  (B) PER-EDGE vs SHARED HOLONOMY ON THE CYCLE"); print("="*82)
Ms=[]
for i in range(NT):
    j=(i+1)%NT
    A=X[:,i,:]; B=X[:,j,:]
    Mi,_,_,_=np.linalg.lstsq(A,B,rcond=None); Ms.append(Mi.T)
Ms=np.array(Ms)
Xi=X[:, np.arange(NT),:].reshape(-1,2); Xj=X[:, (np.arange(NT)+1)%NT,:].reshape(-1,2)
Msh,_,_,_=np.linalg.lstsq(Xi,Xj,rcond=None); Msh=Msh.T
He=np.eye(2)
for i in range(NT): He=Ms[i]@He
Hs=np.linalg.matrix_power(Msh,NT)
def rep(nm,H):
    w=np.abs(np.linalg.eigvals(H))
    print(f"  {nm:>22}  ||H-I||={np.linalg.norm(H-np.eye(2))/np.sqrt(2):>10.4g}"
          f"   |ev|={np.round(w,5)}   det={np.linalg.det(H):>10.4g}")
rep("shared  M^256", Hs); rep("per-edge prod(M_i)", He)
print(f"\n  spread of per-edge maps: ||M_i - M_shared||/||M_shared|| = "
      f"{np.mean([np.linalg.norm(Ms[i]-Msh)/np.linalg.norm(Msh) for i in range(NT)]):.4f}")
print(f"  mean |det M_i| = {np.mean(np.abs(np.linalg.det(Ms))):.4f}   "
      f"det M_shared = {np.linalg.det(Msh):.4f}")
print("\n  Case1 edge product also collapses -> geometric.  Case2 ~I -> shared was an artifact.")
print(f"\n  time {time.time()-t0:.0f}s")
