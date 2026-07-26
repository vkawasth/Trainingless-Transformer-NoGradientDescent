"""
DOES K_r HAVE SPECTRAL STRUCTURE BEYOND THE INDEPENDENT PRODUCT CHAIN?
For a small block of coordinates, estimate the empirical transition operator of
the sign process and compare its spectrum to the factorized prediction
(eigenvalues 1-2h(r_i)).  A collective mode = an eigenvalue not explained by any
single coordinate's 1-2h.  That is the only nontrivial spectral content.
Also: locate r_c where the per-coordinate gap 1-2h(r) closes (the 'transition').
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
a0,_=SPAN["blocks.2.ff.g.weight"]; R,C=512,256
# a single neuron = one row of C=256 weights; track its sign process
NROW=8
IDX=torch.cat([torch.arange(a0+i*C, a0+i*C+16) for i in range(NROW)])  # 8 neurons x 16 coords
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
S=[]; Rr=[]; prev=flat()
for s in range(1,201):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    m=state(o,"exp_avg").abs()[IDX]; v=state(o,"exp_avg_sq").sqrt()[IDX]
    S.append(torch.sign(d[IDX]).numpy()); Rr.append((m/(v+1e-12)).numpy())
    prev=af; del b4,af,d,m,v
S=np.stack(S); Rr=np.stack(Rr)             # (200, 128)
# per-coordinate flip prob and its 1-2h eigenvalue
flips=(S[1:]!=S[:-1]).mean(0)
lam_ind=1-2*flips
# hazard curve h(r) and where gap closes
print("="*72); print("  (A) PER-COORDINATE GAP 1-2h(r): where does it close?"); print("="*72)
rall=Rr[:-1].flatten(); fall=(S[1:]!=S[:-1]).flatten()
qs=np.quantile(rall,np.linspace(0,1,9)[1:-1])
b=np.digitize(rall,qs)
print(f"  {'r octile':>10}{'mean r':>10}{'h(r)':>9}{'gap 1-2h':>11}")
for i in range(8):
    m=b==i
    if m.sum()>10:
        print(f"  {i+1:>10}{rall[m].mean():>10.3f}{fall[m].mean():>9.3f}{1-2*fall[m].mean():>11.3f}")
# empirical joint transition operator on the 128-dim sign state is 2^128 - too big.
# Instead: covariance/correlation spectrum of the flip process -> collective modes
print("\n"+"="*72); print("  (B) COLLECTIVE MODES: flip-correlation spectrum vs independent"); print("="*72)
F=(S[1:]!=S[:-1]).astype(float)            # (199,128) flip events
Fc=F-F.mean(0)
Cov=Fc.T@Fc/len(Fc)
w=np.linalg.eigvalsh(Cov)[::-1]
# independent null: shuffle each coordinate's flip series in time
rng=np.random.default_rng(0); wn=[]
for _ in range(20):
    Fs=np.stack([rng.permutation(F[:,k]) for k in range(F.shape[1])],1)
    Fsc=Fs-Fs.mean(0); wn.append(np.linalg.eigvalsh(Fsc.T@Fsc/len(Fs))[::-1])
wn=np.array(wn); wnm=wn.mean(0); wns=wn.std(0)
print(f"  top flip-covariance eigenvalues (data vs shuffled-independent null):")
print(f"  {'mode':>5}{'data':>10}{'null mean':>11}{'null sd':>9}{'z-score':>9}")
for i in range(6):
    z=(w[i]-wnm[i])/max(wns[i],1e-12)
    print(f"  {i+1:>5}{w[i]:>10.4f}{wnm[i]:>11.4f}{wns[i]:>9.4f}{z:>9.1f}")
nmodes=int((w>wnm+3*wns).sum())
print(f"\n  modes exceeding independent null by 3 sigma: {nmodes} of {len(w)}")
print("  >0 collective modes => K_r has spectral content beyond the product chain.")
print("  ~0 => spectrum is fully the independent 1-2h(r_i) set (trivial).")
print(f"\n  time {time.time()-t0:.0f}s")
