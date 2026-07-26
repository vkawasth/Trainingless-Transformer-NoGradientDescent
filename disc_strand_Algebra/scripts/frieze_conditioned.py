"""
DOES THE FRIEZE HOLONOMY SURVIVE CONDITIONING ON r?
Null: the transport maps M_i are fully determined by local r, so the holonomy
      is normalized-SNR wearing a bundle costume.
Test: (1) fit each edge map M_i, get holonomy H = prod M_i (baseline).
      (2) predict M_i entries from local r features (r_i, r_{i+1}, |grad r|).
      (3) form residual maps M_i^res = M_i (r-prediction removed), transport those.
      If residual holonomy ~ I, r explains the connection. If it persists,
      there is structure beyond r.
Also: nested entropy of the flip law, r vs r+layer+tensor.
"""
import re, time, gc, numpy as np, torch
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
XT=[]; RT=[]; LOSS=[]
prev=flat()
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; LOSS.append(float(l))
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt(); r=m/(v+1e-12)
    XT.append(np.array([[float(torch.log(d[I].abs().mean()+1e-30)),
                         float(torch.log(r[I].mean()+1e-30))] for I in FRZ]))
    RT.append(np.array([float(r[I].mean()) for I in FRZ]))
    prev=af; del b4,af,d,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
X=np.stack(XT); R=np.stack(RT).mean(0)          # (T,NT,2), (NT,)
def fitM(A,B): M,_,_,_=np.linalg.lstsq(A,B,rcond=None); return M.T
# baseline per-edge holonomy
Ms=np.array([fitM(X[:,i,:], X[:,(i+1)%NT,:]) for i in range(NT)])
He=np.eye(2)
for i in range(NT): He=Ms[i]@He
print("\n"+"="*74); print("  BASELINE PER-EDGE HOLONOMY"); print("="*74)
w=np.abs(np.linalg.eigvals(He))
print(f"  ||H-I|| = {np.linalg.norm(He-np.eye(2))/np.sqrt(2):.4f}   |ev| = {np.round(w,4)}"
      f"   det = {np.linalg.det(He):.4f}")
# predict each M-entry (4 of them) from local r features across the 256 edges
rl=R; rr=np.roll(R,-1); gr=np.abs(rr-rl)
Fedge=np.stack([np.log(rl+1e-30), np.log(rr+1e-30), np.log(gr+1e-30), np.ones(NT)],1)
Ment=Ms.reshape(NT,4)
Wpred,_,_,_=np.linalg.lstsq(Fedge,Ment,rcond=None)
pred=Fedge@Wpred
r2=[1-((Ment[:,k]-pred[:,k])**2).sum()/max(((Ment[:,k]-Ment[:,k].mean())**2).sum(),1e-30) for k in range(4)]
print("\n"+"="*74); print("  CAN LOCAL r PREDICT THE EDGE MAPS?"); print("="*74)
print(f"  R^2 of predicting M-entries from (r_i, r_i+1, |grad r|):")
print(f"    M00 {r2[0]:+.3f}   M01 {r2[1]:+.3f}   M10 {r2[2]:+.3f}   M11 {r2[3]:+.3f}")
# residual holonomy: transport the r-predicted maps, and the residual maps
Mpred=pred.reshape(NT,2,2)
Hpred=np.eye(2)
for i in range(NT): Hpred=Mpred[i]@Hpred
# residual = actual minus r-explained, added back to identity
Mres=np.array([np.eye(2)+(Ms[i]-Mpred[i]) for i in range(NT)])
Hres=np.eye(2)
for i in range(NT): Hres=Mres[i]@Hres
print("\n"+"="*74); print("  HOLONOMY DECOMPOSITION"); print("="*74)
def rep(nm,H):
    w=np.abs(np.linalg.eigvals(H))
    print(f"  {nm:>28}  ||H-I||={np.linalg.norm(H-np.eye(2))/np.sqrt(2):>8.4f}"
          f"   |ev|={np.round(w,4)}   det={np.linalg.det(H):>8.4f}")
rep("full (baseline)", He)
rep("r-predicted maps only", Hpred)
rep("residual (r removed)", Hres)
frac=1 - np.linalg.norm(Hres-np.eye(2))/max(np.linalg.norm(He-np.eye(2)),1e-30)
print(f"\n  fraction of holonomy explained by local r: {100*frac:.1f}%")
print("  if residual ~ I => r fully explains the connection (normalized SNR).")
print("  if residual holonomy persists => structure beyond r.")
# nested entropy of flip law
print("\n"+"="*74); print("  NESTED ENTROPY OF THE FLIP LAW"); print("="*74)
print("  (from earlier run: r alone 0.428, r+neuron 0.405 bits -- neuron adds 5%)")
print(f"\n  time {time.time()-t0:.0f}s")
