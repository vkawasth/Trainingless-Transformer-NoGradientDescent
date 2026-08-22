"""BLOCK HESSIAN MASS MAP.

Usage:  python3 block_hessian_mass.py
Needs:  compiler_geometri_patched_86.py and build_corpus.py alongside.
Writes: res_blockmass.json

MEASURED RESULT (D=128, P=1,182,080, 12 probes x 8 batches):

    change from step 30 to 150       total mass 100.4 -> 51.6  (-49%)
              EMB       ATTN         FF         LN
    EMB      -74%       -21%       -27%       -73%
    ATTN     -18%       -61%       +37%       -67%
    FF       -24%       +31%       -34%       -33%
    LN       -73%       -68%       -32%       -64%

    share of total:  EMB 0.440 -> 0.320,  FF 0.322 -> 0.449

EMB self-curvature drains 74% while its couplings to the compute blocks fall
only 21-27%. ATTN<->FF is the ONLY block anywhere that gains absolute mass on an
operator shrinking by half. Estimator asymmetry 0.006-0.016, so M[c,b] = M[b,c]
holds to about 1%.

WHERE DOES THE HESSIAN MASS LIVE, BLOCK BY BLOCK?

The diagonal share falls 0.078 -> 0.024 (bias-corrected), so the off-diagonal
SHARE rises 0.922 -> 0.976. But in absolute terms everything shrinks: total
Frobenius mass 107.8 -> 47.6, diagonal 8.45 -> 1.15, off-diagonal 99.4 -> 46.4.
The diagonal simply drains fastest. "Mass reappearing off-diagonal" is not what
happens; the share moves, the mass does not.

The question this asks is where the remaining mass sits. For blocks b and c,

    M[c,b] = ||H_{cb}||_F^2 = E || (H z_b)_c ||^2

for Rademacher z supported on block b. That is UNBIASED -- unlike the diagonal
estimator, whose bias term (F-S)/N was 30-50% of the raw value at N=24 and had
to be corrected. So the block map is better conditioned than the diagonal map
that motivated it.

WHAT WOULD SUPPORT THE REPRESENTATION-LEARNING READING
  EMB x EMB falling fastest, with EMB x ATTN and EMB x FF falling more slowly
  or rising -- the curvature moving from individual token coordinates into the
  interactions between the embedding and the rest.

WHAT WOULD REFUTE IT
  all blocks falling in proportion, so nothing is redistributing.

Reported as absolute mass AND as share of total, since those tell different
stories and conflating them is what the framing above got wrong. Symmetry
M[c,b] = M[b,c] is checked rather than assumed -- it is a property of H, and a
violation would mean the estimator is broken.
"""
import json, subprocess, numpy as np, torch, io, contextlib, math
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
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
B=["EMB","ATTN","FF","LN"]
bi={}
for nm,(a,bb) in span.items(): bi.setdefault(role(nm),[]).append(torch.arange(a,bb))
bi={k:torch.cat(v) for k,v in bi.items()}
EV=[get_batch() for _ in range(8)]
NP=12
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
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
step=0; hist=[]
print(f"  P={P:,}   M[c,b] = ||H_cb||_F^2, unbiased, {NP} probes x {len(EV)} batches\n")
for ck in (30,90,150):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat()
    M=np.zeros((len(B),len(B)))
    g=torch.Generator().manual_seed(50+ck)
    for j,bn in enumerate(B):
        ii=bi[bn]
        for _ in range(NP):
            z=torch.zeros(P)
            z[ii]=(torch.randint(0,2,(len(ii),),generator=g).float()*2-1)
            hz=Hv(z); setth(th)
            for c,cn in enumerate(B):
                M[c,j]+=float((hz[bi[cn]]**2).sum())
        M[:,j]/=NP
    tot=M.sum()
    asym=np.abs(M-M.T).sum()/max(tot,1e-30)
    print(f"  === step {ck}   total Frobenius mass {tot:.2f}   asymmetry {asym:.3f}")
    print(f"  {'':>6}"+"".join(f"{c:>11}" for c in B)+f"{'row sum':>11}")
    for c,cn in enumerate(B):
        print(f"  {cn:>6}"+"".join(f"{M[c,j]:>11.3f}" for j in range(len(B)))
              +f"{M[c].sum():>11.3f}")
    print(f"  {'share':>6}"+"".join(f"{M[c].sum()/tot:>11.3f}" for c in range(len(B))))
    hist.append(dict(ck=ck,M=M.tolist(),tot=float(tot),asym=float(asym)))
    print()
json.dump(hist,open("/home/claude/work/res_blockmass.json","w"),indent=2)
M0=np.array(hist[0]["M"]); M1=np.array(hist[-1]["M"])
print(f"  CHANGE from step {hist[0]['ck']} to {hist[-1]['ck']}")
print(f"  {'':>6}"+"".join(f"{c:>11}" for c in B))
for c,cn in enumerate(B):
    print(f"  {cn:>6}"+"".join(
        f"{100*(M1[c,j]-M0[c,j])/max(M0[c,j],1e-12):>10.0f}%" for j in range(len(B))))
print(f"\n  total mass {hist[0]['tot']:.1f} -> {hist[-1]['tot']:.1f}  "
      f"({100*(hist[-1]['tot']-hist[0]['tot'])/hist[0]['tot']:+.0f}%)")
print(f"\n  EMBxEMB falling faster than EMBxATTN / EMBxFF => curvature moving")
print(f"  from token coordinates into interactions")
print(f"  all blocks falling in proportion => nothing redistributes")
