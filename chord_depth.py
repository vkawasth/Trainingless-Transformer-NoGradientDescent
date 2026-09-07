"""CHORD STRUCTURE VS TRANSFORMER DEPTH.

    python3 chord_depth.py <N_STU> [seed]      e.g.  python3 chord_depth.py 24

Needs compiler_geometri_patched_86.py and build_corpus.py alongside; builds the
corpus itself if /tmp/*.json are missing. Runs Phase 1/2 then 120 Phase-3 steps
of the shipped optimiser (m,v at 4 bits/coord, absorbed alpha_t), takes the
chord theta_120 - theta_40, and reports its architectural structure.

MEASURED AT D=128, one seed each:

                              6 layers      12 layers
  per-param energy by block   flat          monotone +25% (blocks 1->11)
  non-block share             21.7%         14.5%
  W_Q chord PR                3.4           2.8      (random baseline 63.6)
  rank ordering               ff.g > pe > W_V > te > W_Q, both depths
  W_V head structure          z = +4.7      z = +8.0
  W_K head structure          z = +2.8      z = -0.0
  adjacent-block cosine       <= 0.055      <= 0.024

REGISTERED PREDICTIONS FOR 24 LAYERS, written before running it:
  1  depth gradient strengthens further, +30-50% across the stack
  2  W_Q PR stays near 3 -- suppression is a property of the 128x128 matrix,
     not of the network
  3  rank ordering unchanged
  4  W_V head structure strengthens (z > 8)
  5  W_K head structure stays absent
  6  non-block share falls to about 8%

Predictions 1 and 5 are the least secure: at 6 layers the energy was flat and
W_K had structure, so both reversed once already and rest on a single draw at
each depth. Running the same depth at two seeds is the cheaper check.

NOTE: N_STU=24 exceeded a 3 GB container inside the Phase 1/2 prelude, before
any training. It needs more memory, not a code change.
"""

import io, contextlib, subprocess, sys, json, math, collections
import numpy as np, torch, torch.nn.functional as F
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
R=open("compiler_geometri_patched_86.py").read()
SRC=R[:R.find("# \u2500\u2500 PHASE 3")]
import sys as _s
NL   = int(_s.argv[1]) if len(_s.argv)>1 else 6
SEED = int(_s.argv[2]) if len(_s.argv)>2 else 17
for o,n in [("D=256; N_HEADS=4; N_STU=6;",f"D=128; N_HEADS=4; N_STU={NL};"),
            ("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
             "    if False:")]:
    assert SRC.count(o)==1; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)
B1,B2,EPS,WD,LRM=0.9,0.95,1e-8,0.1,5.0
torch.manual_seed(1234); np.random.seed(1234)
G={}; _b=io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC,G)
tm=G["model"]; gb=G["get_batch"]; lr0=G["LR"]; NH=G.get("N_HEADS",4)
tn=[(n,p) for n,p in tm.named_parameters() if p.requires_grad]
torch.manual_seed(SEED)
BATCH=[gb() for _ in range(120)]
def q4(x):
    lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
    if hi-lo<1e-12: return x
    s=(hi-lo)/15.0
    return torch.exp(torch.round((lv-lo)/s).clamp(0,15)*s+lo)
snap={}
m={a:torch.zeros_like(p) for a,p in tn}; v={a:torch.zeros_like(p) for a,p in tn}
for t,(x,y) in enumerate(BATCH,1):
    _,l=tm(x,y); tm.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
    b1c,b2c=1-B1**t,1-B2**t
    al=lr0*LRM*(t/10 if t<=10 else 1)*math.sqrt(b2c)/b1c; et=EPS*math.sqrt(b2c)
    with torch.no_grad():
        for a,p in tn:
            g=p.grad if p.grad is not None else torch.zeros_like(p)
            m[a].mul_(B1).add_(g,alpha=1-B1); v[a].mul_(B2).addcmul_(g,g,value=1-B2)
            vv,mm=v[a],m[a]
            if p.dim()==2 and p.shape[0]>1:
                vv=q4(vv); mm=torch.sign(mm)*q4(mm.abs())
            p.data.add_(-al*mm/(vv.sqrt()+et) - lr0*LRM*WD*p.data)
    if t in (40,120):
        # memory: keep only what the analysis reads. Two full copies at
        # N_STU=24 exceeded the container.
        snap[t]={a:p.data.clone() for a,p in tn
                 if a.endswith((".WQ.weight",".WK.weight",".WV.weight",
                                "ff.g.weight","te.weight","pe.weight"))}
CH={a:(snap[120][a]-snap[40][a]) for a in snap[120]}
tot=sum(float((c**2).sum()) for c in CH.values())
rng=np.random.default_rng(0)
print(f"  N_STU={NL}  seed={SEED}   chord ||.||^2 = {tot:.4f}\n")
print("  DEPTH  share of chord energy by block")
byb=collections.defaultdict(float)
for a,c in CH.items():
    if a.startswith("blocks."): byb[int(a.split(".")[1])]+=float((c**2).sum())
    else: byb[-1]+=float((c**2).sum())
npar=collections.defaultdict(int)
for a,p in tn:
    if a in CH: npar[int(a.split(".")[1]) if a.startswith("blocks.") else -1]+=p.numel()
for b in sorted(byb):
    lab="non-block" if b<0 else f"block {b}"
    print(f"    {lab:<10} {byb[b]/tot*100:5.1f}%   per-param {byb[b]/npar[b]:.3e}")
print("\n  HEADS  per-head chord energy inside W_Q/W_K/W_V, and the")
print("         between-head spread vs random equal-size column groups")
for nm in ("WQ","WK","WV"):
    ks=[a for a in CH if f".{nm}." in a and CH[a].dim()==2]
    if not ks: continue
    E=torch.cat([CH[k] for k in ks],0).numpy()
    D=E.shape[1]; dh=D//NH
    he=np.array([np.linalg.norm(E[:,h*dh:(h+1)*dh])**2 for h in range(NH)])
    he=he/he.sum(); cv=float(np.std(he)/np.mean(he))
    nul=[]
    for _ in range(200):
        pm=rng.permutation(D)
        g=np.array([np.linalg.norm(E[:,pm[h*dh:(h+1)*dh]])**2 for h in range(NH)])
        g=g/g.sum(); nul.append(float(np.std(g)/np.mean(g)))
    print(f"    {nm}  shares {np.round(he,3)}  CV {cv:.3f}   "
          f"null {np.mean(nul):.3f}+-{np.std(nul):.3f}   z {(cv-np.mean(nul))/max(np.std(nul),1e-9):+.1f}")
print("\n  RESIDUAL  cos between the chord of block l and block l+1 (same matrix)")
for nm in ("attn.WQ","attn.WV","ff.g"):
    ks=sorted([a for a in CH if a.endswith(nm+".weight")],
              key=lambda z:int(z.split(".")[1]))
    for i in range(len(ks)-1):
        A=CH[ks[i]].reshape(-1); Bv=CH[ks[i+1]].reshape(-1)
        if A.shape!=Bv.shape: continue
        c=float((A@Bv)/(A.norm()*Bv.norm()+1e-30))
        nn=[float(np.corrcoef(rng.standard_normal(len(A)),rng.standard_normal(len(A)))[0,1])
            for _ in range(20)]
        print(f"    {nm} block {i}->{i+1}: cos {c:+.4f}   random {np.mean(nn):+.4f}")
        break
print("\n  WIDTH  effective rank of the chord per matrix type vs a random")
print("         matrix of the same shape (participation ratio of s^2)")
def pr(s):
    e=s**2; return float(e.sum()**2/max((e**2).sum(),1e-30))
for nm in ("te.weight","pe.weight","attn.WQ.weight","attn.WV.weight","ff.g.weight"):
    ks=[a for a in CH if a.endswith(nm)]
    if not ks: continue
    M=CH[ks[0]]
    s=torch.linalg.svdvals(M).numpy(); p=pr(s)
    rr=[pr(np.linalg.svd(rng.standard_normal(M.shape),compute_uv=False)) for _ in range(10)]
    print(f"    {nm:<16} {tuple(M.shape)}  PR {p:6.1f}  "
          f"random {np.mean(rr):6.1f}   ratio {p/np.mean(rr):.3f}")
