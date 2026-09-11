"""DOES A PRECONDITIONER FLOOR STOP THE 4226x INFLATION?

Adam is scale-invariant per coordinate: scaling g by S scales m by S and
sqrt(v) by S, leaving m/sqrt(v) unchanged. So gating gradients down does not
shrink the step -- Adam re-amplifies exactly. The only mechanism that bites is
the epsilon floor, where sqrt(v) + eps is dominated by eps once sqrt(v) << eps.

Block 0's QK gradient is 2.8e-5 over 131072 coordinates, so per-coordinate
sqrt(v) is O(5e-7) -- fifty times the default eps of 1e-8. Raising eps for that
tensor alone should collapse its step without touching anything else.

Arms, identical init and batch stream, 300 steps:
    baseline          eps = 1e-8 everywhere
    eps0_1e-5         eps = 1e-5 on block 0 WQ, WK only
    eps0_1e-4         eps = 1e-4 on block 0 WQ, WK only
    zero0             block 0 attention output zeroed  (known: costs nothing)

Reported: val, and the per-step displacement of block 0 WQ, which is the
quantity the floor is meant to suppress. A working floor shows ||dWQ0|| falling
by orders of magnitude with val unchanged.
"""
import io, contextlib, math, sys
import numpy as np, torch
R=open("compiler_geometri_patched_86.py").read()
SRC=R[:R.find("# \u2500\u2500 PHASE 3")]
for o,n in [("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),("ETA_MF=0.01","ETA_MF=0.002"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
             "    if False:")]:
    assert SRC.count(o)==1,o; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)
ARM=sys.argv[1] if len(sys.argv)>1 else "baseline"
EPS0={"baseline":1e-8,"eps0_1e-5":1e-5,"eps0_1e-4":1e-4}.get(ARM,1e-8)
torch.manual_seed(1234); np.random.seed(1234)
G={}; _b=io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC,G)
model=G["model"]; gb=G["get_batch"]; LR=G["LR"]*5
named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
if ARM=="zero0":
    with torch.no_grad(): model.blocks[0].attn.op.weight.zero_()
    model.blocks[0].attn.op.weight.requires_grad_(False)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
B1,B2,WD=0.9,0.95,0.1
TGT=("blocks.0.attn.WQ.weight","blocks.0.attn.WK.weight")
m={n:torch.zeros_like(p) for n,p in named}; v={n:torch.zeros_like(p) for n,p in named}
EV=[gb() for _ in range(4)]
def vl():
    with torch.no_grad():
        return sum(float(model(a,b)[1]) for a,b in EV)/len(EV)
print(f"  arm {ARM}   eps on block-0 QK = {EPS0:g}")
print(f"  {'t':>5}{'val':>9}{'|dWQ0|':>12}{'|dFF0|':>11}")
acc=[0.0,0.0]
for t in range(1,301):
    x,y=gb(); _,l=model(x,y); model.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b1c,b2c=1-B1**t,1-B2**t
    with torch.no_grad():
        for n,p in named:
            g=p.grad if p.grad is not None else torch.zeros_like(p)
            m[n].mul_(B1).add_(g,alpha=1-B1); v[n].mul_(B2).addcmul_(g,g,value=1-B2)
            e=EPS0 if n in TGT else 1e-8
            d=-LR*((m[n]/b1c)/((v[n]/b2c).sqrt()+e)+WD*p.data)
            p.data.add_(d)
            if n==TGT[0]: acc[0]+=float(d.norm())
            if n=="blocks.0.ff.g.weight": acc[1]+=float(d.norm())
    if t in (50,100,200,300):
        print(f"  {t:>5}{vl():>9.4f}{acc[0]:>12.4f}{acc[1]:>11.4f}",flush=True)
