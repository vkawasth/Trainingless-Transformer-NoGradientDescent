"""IS THERE A NATURAL k, OR JUST A HEAVY TAIL?

SignTop retains 0.949 of the update norm at k=8191 and 0.846 at k=15. If the
tail were near-uniform in magnitude, that ratio would barely move -- the sign
field is unchanged and would carry nearly all the mass either way. It moves ten
points, so the magnitude residual is not negligible and

    u = u_S + a*s_Sc + eps_Sc     with eps NOT ~ 0

is the right model rather than u ~ a*s. That retires "near-uniform-magnitude
update" as a full structural description.

THE FALSIFICATION TEST. Measure the whole curve rather than two points:

    R(k) = |u_hat_k|^2 / |u|^2       norm retained by SignTop at budget k
    E(k) = 1 - R(k)                  the magnitude residual's share

  STRATIFIED   a small exceptional set S, after which the tail is
               approximately constant-magnitude. R(k) rises steeply then
               PLATEAUS -- a knee at an empirically meaningful k*.
  SMOOTH       magnitudes decay continuously with no structural transition.
               R(k) rises without a knee, and "stratified space" would be
               overinterpreting a generic heavy-tailed distribution.

Detected numerically rather than by eye: fit R(k) against log k, and locate the
maximum of the discrete second difference. A knee is a sharp negative peak in
d^2R/d(log k)^2 that stands above the curve's own noise. A pure power law has
no such peak.

CONTROLS, because a knee can be manufactured:
  gaussian    R(k) for i.i.d. normal entries of the same shape -- the null for
              "no structure", since a Gaussian has a smooth magnitude law
  lognormal   a deliberately heavy-tailed null, which is the alternative
              hypothesis in its purest form
  shuffled u  the real update's magnitudes reassigned to random coordinates:
              same magnitude LAW, no spatial structure. If the real curve and
              the shuffled curve have the same knee, the knee is a property of
              the magnitude distribution and not of the update's geometry.

The last control is the decisive one. A stratification claim needs the knee to
survive it.
"""
import json, subprocess, numpy as np, torch, io, contextlib, math
subprocess.run(["python3","build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
RAW=open("compiler_geometri_patched_86.py").read()
SRC=RAW[:RAW.find("# \u2500\u2500 PHASE 3")].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
SRC=SRC.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
SRC=SRC.replace("    if pc == N_STU-1:","    if False:",1)
SRC=SRC.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:",1)
torch.manual_seed(1234); np.random.seed(1234)
G={}; b=io.StringIO()
with contextlib.redirect_stdout(b): exec(SRC,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
MATS=[(n,p) for n,p in model.named_parameters() if p.dim()==2 and p.requires_grad]
def R_of_k(fl,k):
    """norm retained by SignTop at budget k, on a flat vector"""
    n=fl.numel()
    if k>=n: return 1.0
    z=torch.zeros_like(fl); m=torch.ones(n,dtype=torch.bool)
    if k>0:
        idx=torch.topk(fl.abs(),k,sorted=False).indices
        m[idx]=False; z[idx]=fl[idx]
    if m.any(): z[m]=torch.sign(fl[m])*float(fl[m].abs().mean())
    return float((z*z).sum())/max(float((fl*fl).sum()),1e-30)
def knee(ks,R):
    """sharpest negative second difference in log k, and its prominence"""
    lk=np.log(np.array(ks,dtype=float)); Rv=np.array(R)
    d2=[]
    for i in range(1,len(ks)-1):
        h1=lk[i]-lk[i-1]; h2=lk[i+1]-lk[i]
        d2.append(2*((Rv[i+1]-Rv[i])/h2-(Rv[i]-Rv[i-1])/h1)/(h1+h2))
    d2=np.array(d2)
    j=int(np.argmin(d2))
    prom=abs(d2[j])/max(np.std(d2),1e-12)
    return ks[j+1],float(d2[j]),float(prom)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
KS=[1,2,4,8,15,32,64,128,256,512,1024,2048,4096,8191,16384,32768]
step=0; rows=[]
print(f"  R(k) = |u_hat_k|^2/|u|^2 for SignTop, averaged over matrices\n")
for ck in (40,90,140):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th={n:p.data.clone() for n,p in MATS}
    x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    u={n:(p.data-th[n]).reshape(-1).clone() for n,p in MATS}
    with torch.no_grad():
        for n,p in MATS: p.data.copy_(th[n])
    g_=torch.Generator().manual_seed(ck)
    real=[]; shuf=[]; gaus=[]; logn=[]
    for k in KS:
        rr=[];ss=[];gg=[];ll=[]
        for n,p in MATS:
            fl=u[n]
            if k>=fl.numel(): continue
            rr.append(R_of_k(fl,k))
            ss.append(R_of_k(fl[torch.randperm(fl.numel(),generator=g_)],k))
            gg.append(R_of_k(torch.randn(fl.numel(),generator=g_),k))
            ll.append(R_of_k(torch.exp(torch.randn(fl.numel(),generator=g_)*1.5)
                             *torch.sign(torch.randn(fl.numel(),generator=g_)),k))
        real.append(np.mean(rr)); shuf.append(np.mean(ss))
        gaus.append(np.mean(gg)); logn.append(np.mean(ll))
    print(f"  === step {ck}")
    print(f"  {'k':>7}{'real':>9}{'shuffled':>10}{'gaussian':>10}{'lognormal':>11}")
    for i,k in enumerate(KS):
        print(f"  {k:>7}{real[i]:>9.4f}{shuf[i]:>10.4f}{gaus[i]:>10.4f}{logn[i]:>11.4f}")
    for nm,cur in (("real",real),("shuffled",shuf),("gaussian",gaus),("lognormal",logn)):
        kk,d2,pr=knee(KS[:len(cur)],cur)
        print(f"    knee({nm:>9}) at k={kk:<6} d2={d2:+.4f}  prominence {pr:.2f} sd")
    rows.append(dict(ck=ck,ks=KS[:len(real)],real=real,shuf=shuf,gaus=gaus,logn=logn))
    print()
json.dump(rows,open("res_stratify.json","w"),indent=2)
print(f"  a knee in 'real' that is ABSENT or much weaker in 'shuffled' would be")
print(f"  structure. the same knee in both means it is a property of the")
print(f"  magnitude distribution alone, and 'stratified' is overinterpretation.")
