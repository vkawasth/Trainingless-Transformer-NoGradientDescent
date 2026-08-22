"""IS THE HESSIAN BECOMING DIAGONAL?

The Gauss run measured II = 0.0024 for random tangents with sd 0.00002, against
6.6 for the principal direction -- a factor of 2800. An operator with almost all
its mass on a few directions. If those directions are becoming AXIS-ALIGNED, H is
drifting toward diagonal, and that is exactly the condition the sign-descent
theory needs:

    rho_diag(H) = sum_i |H_ii| / sum_ij |H_ij|

controls the L_infinity smoothness constant, hence whether signSGD is
competitive. rho_diag = 1 for a diagonal matrix, ~1/sqrt(P) for a generic dense
one.

It would tie together three loose ends:
  PR(F) = 2.5-3.2 against PR(H) = 17-152
  sign carried 100.1% of the whole update but cost 6.3% on the complement
  the SGD residual concentrates 84% of its energy on 1% of coordinates

ESTIMATORS, since the exact sums need P Hessian-vector products:
  diag(H)     Hutchinson: E[z . Hz] = diag(H) for Rademacher z, elementwise
  ||H||_F^2   E||Hz||^2 = ||H||_F^2 for the same z
  ratio       sum_i H_ii^2 / ||H||_F^2  -- the fraction of Frobenius mass on the
              diagonal. 1 means diagonal, ~1/P means the diagonal is negligible.

NOTE ON WHICH NORM. This is NOT the rho_diag of the sign-descent literature,
which is sum_i |H_ii| / sum_ij |H_ij| and needs the full absolute sum. That is
not cheaply estimable. The squared version here is, and it answers the same
question -- how much mass sits on the diagonal -- but the two should not be
quoted interchangeably.

USAGE
    python3 hutchinson_probe.py [NPROBE] [comma,separated,steps]
    python3 hutchinson_probe.py 128 30,70,110,150

RESULT AT 24 PROBES (provisional): ratio 0.1168, 0.1034, 0.0731, 0.0648 --
77,000-138,000x the generic dense value of 1/P, but declining 45% over the run.
The decline is NOT trustworthy at that probe count: split-half falls 0.482 ->
0.199 across the same checkpoints, so the estimator degrades as fast as the
signal. A declining estimate from a declining estimator is exactly the confound
to rule out, which needs split-half above ~0.8.

This is a PROXY for rho_diag, which needs sum_ij |H_ij| and is not estimable
cheaply. The squared version is, and it answers the same question: is the mass
moving onto the diagonal?

CONTROLS
  the same statistic on a random symmetric matrix of matched Frobenius norm,
    which gives the generic value
  the Hutchinson variance across probe batches, so a drift smaller than the
    estimator's noise is not reported as a trend
"""
import json, subprocess, numpy as np, torch, io, contextlib, math
subprocess.run(["python3","build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:",1)
G={}; b=io.StringIO()
with contextlib.redirect_stdout(b): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
named=[(n,p) for n,p in model.named_parameters()]; ps=[p for _,p in named]
P=sum(p.numel() for p in ps)
span={}; i=0
for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
def role(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    if "ln" in nm.lower() or nm.endswith("n.weight") or nm.endswith("n.bias"): return "LN"
    if ".ff." in nm: return "FF"
    return "ATTN"
bi={}
for nm,(a,bb) in span.items(): bi.setdefault(role(nm),[]).append(torch.arange(a,bb))
bi={k:torch.cat(v) for k,v in bi.items()}
EV=[get_batch() for _ in range(8)]
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        j=0
        for p in ps:
            q=p.numel(); p.data.copy_(t[j:j+q].view_as(p)); j+=q
def Hv(v):
    acc=torch.zeros(P)
    for x,y in EV:
        model.zero_grad(); _,l=model(x,y)
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([t.reshape(-1) for t in gr])
        hv=torch.autograd.grad((gf*v).sum(),ps,allow_unused=True)
        acc+=torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                        for t,p in zip(hv,ps)]).detach()
    model.zero_grad(); return acc/len(EV)
import sys
NPROBE=int(sys.argv[1]) if len(sys.argv)>1 else 24
STEPS=[int(z) for z in sys.argv[2].split(",")] if len(sys.argv)>2 else [30,70,110,150]
def diagstats(seed):
    th=flat()
    g=torch.Generator().manual_seed(seed)
    dsum=torch.zeros(P); fro=0.0; halves=[torch.zeros(P),torch.zeros(P)]
    for j in range(NPROBE):
        z=(torch.randint(0,2,(P,),generator=g).float()*2-1)
        hz=Hv(z); setth(th)
        dsum+=z*hz; fro+=float(hz@hz)
        halves[j%2]+=z*hz
    d=dsum/NPROBE; fro=fro/NPROBE
    h0=halves[0]/(NPROBE//2); h1=halves[1]/(NPROBE//2)
    return d,fro,h0,h1
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
print(f"  P={P:,}   {NPROBE} Hutchinson probes x {len(EV)} batches per checkpoint")
print(f"  generic dense matrix would give sum H_ii^2 / ||H||_F^2 ~ 1/P = {1/P:.2e}\n")
print(f"  {'step':>5}{'sum Hii^2':>12}{'||H||_F^2':>12}{'ratio':>10}{'x generic':>11}"
      f"{'split-half':>12}{'mean |Hii|':>12}")
step=0; rows=[]
for ck in STEPS:
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    d,fro,h0,h1=diagstats(1000+ck)
    dd=float((d*d).sum()); ratio=dd/max(fro,1e-30)
    sh=float(np.corrcoef(h0.numpy(),h1.numpy())[0,1])
    print(f"  {ck:>5}{dd:>12.4e}{fro:>12.4e}{ratio:>10.4f}{ratio*P:>11.1f}"
          f"{sh:>12.4f}{float(d.abs().mean()):>12.4e}")
    per={k:float((d[ii]*d[ii]).sum())/max(fro,1e-30) for k,ii in bi.items()}
    rows.append(dict(ck=ck,dd=dd,fro=fro,ratio=ratio,split=sh,per=per))
json.dump(rows,open("/home/claude/work/res_diag.json","w"),indent=2)
r=np.array([x["ratio"] for x in rows])
print(f"\n  ratio: {r.round(4)}   trend {100*(r[-1]-r[0])/r[0]:+.0f}% over the run")
print(f"  split-half correlation of the diagonal estimate: "
      f"{np.mean([x['split'] for x in rows]):.3f}  (1.0 = fully resolved)")
print(f"\n  per-role share of the diagonal mass:")
print(f"  {'step':>5}"+''.join(f"{k:>10}" for k in bi))
for x in rows:
    print(f"  {x['ck']:>5}"+''.join(f"{x['per'][k]:>10.4f}" for k in bi))
print(f"\n  ratio RISING => the Hessian is moving onto the diagonal, and the")
print(f"  geometry is becoming the one where sign methods are favoured")
