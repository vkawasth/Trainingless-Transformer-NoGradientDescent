"""
(1) FRONTIER VELOCITY V(t)=|A(t+1)\A(t)| and activity autocorr C(k).
(2) THE FLIP-COUPLING GRAPH on neurons: edge = correlation of flip activity
    after conditioning on r. Is it sparse / modular / low-spectral-dimension?
    If its Laplacian has a spectral gap (few communities), that gap is the
    nonlinear basis SVD missed.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def st(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
P=flat().numel()
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
a0,_=SPAN["blocks.2.ff.g.weight"]; R,C=512,256    # 512 neurons
IDX=torch.arange(a0,a0+R*C)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
ACT=[]; NF=[]; RB=[]; prev=flat(); sp=None; FRAC=0.106
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    thr=torch.quantile(r[torch.randperm(P)[:150000]],FRAC)
    ACT.append((r<thr))
    sg=torch.sign(d[IDX])
    if sp is not None:
        fl=(sg!=sp).view(R,C).float()
        NF.append(fl.mean(1).numpy())            # per-neuron flip fraction
        RB.append(r[IDX].view(R,C).mean(1).numpy())
    sp=sg; prev=af; del b4,af,d,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("="*72); print("  (1) FRONTIER VELOCITY & ACTIVITY MEMORY"); print("="*72)
V=[float((ACT[t+1]&~ACT[t]).sum()) for t in range(len(ACT)-1)]
asz=float(ACT[40].sum())
print(f"  active set size ~{int(asz/1000)}k;  entering/step V(t): mean {np.mean(V)/1000:.0f}k"
      f"  ({100*np.mean(V)/asz:.0f}% turnover/step)")
print(f"  V(t) early(1-40) {np.mean(V[:40])/1000:.0f}k  late(120-159) {np.mean(V[-40:])/1000:.0f}k")
# activity autocorr on a coordinate subset
SUB=torch.randperm(P)[:100000]
Am=np.stack([ACT[t][SUB].numpy().astype(float) for t in range(len(ACT))])
Ck=[np.mean([np.corrcoef(Am[t],Am[t+k])[0,1] for t in range(20,len(ACT)-k,4)]) for k in (1,5,10,20,40)]
print("  activity autocorr C(k): " + "  ".join(f"k{k}:{c:.3f}" for k,c in zip((1,5,10,20,40),Ck)))
print("  (plateau > 0 => persistent latent structure, not memoryless)")
# (2) neuron flip-coupling graph
NF=np.stack(NF); RB=np.stack(RB)                 # (T,512)
# residualize each neuron's flip series on its own r (condition out r)
res=np.zeros_like(NF)
for j in range(R):
    b=np.polyfit(RB[:,j],NF[:,j],1); res[:,j]=NF[:,j]-np.polyval(b,RB[:,j])
Corr=np.corrcoef(res.T)                          # 512x512 neuron coupling, r-conditioned
np.fill_diagonal(Corr,0)
print("\n"+"="*72); print("  (2) NEURON FLIP-COUPLING GRAPH (r-conditioned)"); print("="*72)
A=np.abs(Corr)
for thr in (0.2,0.3,0.4):
    dens=100*(A>thr).mean()
    print(f"    density at |corr|>{thr}: {dens:.1f}%   mean degree {(A>thr).sum(1).mean():.1f}")
# spectral: Laplacian gap = community structure
W=A*(A>0.2); Dg=np.diag(W.sum(1)); L=Dg-W
with np.errstate(all='ignore'):
    Dm=np.diag(1/np.sqrt(np.maximum(W.sum(1),1e-9))); Ln=np.eye(R)-Dm@W@Dm
ev=np.sort(np.linalg.eigvalsh(Ln))
print(f"\n  normalized-Laplacian smallest eigenvalues: {np.round(ev[:6],4)}")
gaps=np.diff(ev[:15])
kgap=int(np.argmax(gaps[:10]))+1
print(f"  largest spectral gap after eigenvalue #{kgap} (gap {gaps[kgap-1]:.4f})")
print(f"  => {kgap} communities if gap is clean; many small eigenvalues => no modularity")
# compare: does a graph-Laplacian basis beat SVD for sign preservation on this matrix?
print(f"\n  interpretation:")
print(f"    clean gap at small k => sparse modular graph => Laplacian basis is the")
print(f"    nonlinear basis SVD missed. no gap => coupling is dense/diffuse, no better basis.")
print(f"\n  time {time.time()-t0:.0f}s")
