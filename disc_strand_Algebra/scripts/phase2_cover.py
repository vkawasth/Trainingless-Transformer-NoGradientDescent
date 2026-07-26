"""
PHASE II: IS beta INTRINSIC OR AN ARTIFACT OF THE COVER?
Ground truth = beta computed per PARAMETER (cover by singletons).
Then ask how well each cover's regional beta predicts the singleton beta.
Block: L2 FF g.weight, first 128 rows -> (128,256) = 32768 params.
Covers: rows(=neurons), cols(=input dims), contiguous tiles, random partition.
"""
import re, time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
model.load_state_dict(torch.load("init.pt"))
named=list(model.named_parameters())
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
TARGET="blocks.2.ff.g.weight"; a0,_=SPAN[TARGET]
R,C=128,256
sel=np.concatenate([np.arange(a0+r*256, a0+r*256+C) for r in range(R)])
sel_t=torch.tensor(sel)
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
U=[]; LOSS=[]
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); opt.step(); af=flat()
    U.append((af-b4)[sel_t].abs().numpy().astype(np.float32)); LOSS.append(float(l))
    del b4,af
U=np.stack(U); LOSS=np.array(LOSS); H=0.062
print(f"  collected {U.shape} ({time.time()-t0:.0f}s)", flush=True)
ls=LOSS-H; ok=ls>1e-6; x=np.log(ls[ok]); xm=x-x.mean(); den=float((xm*xm).sum())
def ab(mat):
    Y=np.log(np.maximum(mat[ok],1e-30))
    al=(Y-Y.mean(0)).T@xm/den
    return al, Y.mean(0)-al*x.mean()
al_s,be_s=ab(U)                                   # singleton ground truth
print(f"  singleton beta: mean {be_s.mean():.3f}  sd {be_s.std():.3f}"
      f"   alpha: mean {al_s.mean():.4f} sd {al_s.std():.4f}")
rng=np.random.default_rng(0)
def cover_beta(groups):
    """regional beta from the region's mean |u|; broadcast back to parameters"""
    out=np.empty(U.shape[1]); outa=np.empty(U.shape[1])
    for gi in groups:
        m=U[:,gi].mean(1,keepdims=True)
        a,b=ab(m); out[gi]=b[0]; outa[gi]=a[0]
    return out,outa
idx=np.arange(R*C)
COVERS={
 "rows (neurons), 128":       [idx[r*C:(r+1)*C] for r in range(R)],
 "cols (inputs), 256":        [idx.reshape(R,C)[:,c].copy() for c in range(C)],
 "contig tiles, 64":          [idx[i*(R*C//64):(i+1)*(R*C//64)] for i in range(64)],
 "contig tiles, 512":         [idx[i*(R*C//512):(i+1)*(R*C//512)] for i in range(512)],
 "RANDOM partition, 128":     np.array_split(rng.permutation(idx),128),
 "RANDOM partition, 512":     np.array_split(rng.permutation(idx),512),
}
print("\n"+"="*80); print("  PHASE II: DOES A COVER'S beta PREDICT THE PER-PARAMETER beta?"); print("="*80)
print(f"  {'cover':>26}{'regions':>9}{'R2(beta)':>11}{'corr(beta)':>12}{'R2(alpha)':>11}")
B={}
for name,groups in COVERS.items():
    b,a=cover_beta(groups); B[name]=b
    r2=1-((be_s-b)**2).sum()/((be_s-be_s.mean())**2).sum()
    r2a=1-((al_s-a)**2).sum()/((al_s-al_s.mean())**2).sum()
    print(f"  {name:>26}{len(groups):>9}{r2:>11.3f}{np.corrcoef(be_s,b)[0,1]:>12.3f}{r2a:>11.3f}")
print("\n  AGREEMENT BETWEEN COVERS (corr of per-parameter beta assignment):")
ks=list(COVERS)
print(f"  {'':>26}" + "".join(f"{k.split(',')[0][:9]:>10}" for k in ks))
for k in ks:
    print(f"  {k:>26}" + "".join(f"{np.corrcoef(B[k],B[j])[0,1]:>10.3f}" for j in ks))
print("\n  variance of regional beta across regions (structure the cover resolves):")
for name,groups in COVERS.items():
    vals=np.array([B[name][gi[0]] for gi in groups])
    print(f"    {name:>26}  sd(beta_region) = {vals.std():.4f}")
print(f"\n  sd of singleton beta = {be_s.std():.4f}  (ceiling for any cover)")
print(f"\n  time {time.time()-t0:.0f}s")
