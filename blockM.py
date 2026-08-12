"""IS THE OFF-DIAGONAL BLOCK M NON-ZERO?

Lambda = [[H_D, M],[0, H_R]] has content only if M is non-negligible. With
e1 = P_F (the 3-dim Fisher descent channel) and e2 = I - P_F,

    M = P_F H (I - P_F)

is measured against the diagonal block P_F H P_F. Because the blocks have very
different dimensions, the comparison is made against a RANDOM 3-plane, which
shows what "coupling by dimension alone" looks like.

Also the additivity check: is dL(full) = dL(P_F u) + dL(resid)? Exact additivity
means the blocks do not interact at the level that matters for the loss, and the
triangular ring degenerates to a product.

SPLIT_F reported since P_F is estimated.
"""
import json, subprocess, numpy as np, torch, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
params=[p for _,p in model.named_parameters()]; P=sum(p.numel() for p in params)
for p in params: p.requires_grad_(True)
R=3
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
def setth(t):
    with torch.no_grad():
        i=0
        for p in params:
            q=p.numel(); p.data.copy_(t[i:i+q].view_as(p)); i+=q
def gfl():
    return torch.cat([(p.grad.flatten() if p.grad is not None else
        torch.zeros(p.numel())) for p in params]).clone()
def hvp(th,v,nb=6):
    a=torch.zeros(P)
    for _ in range(nb):
        x,y=get_batch(); model.zero_grad(); _,loss=model(x,y)
        gr=torch.autograd.grad(loss,params,create_graph=True,allow_unused=True)
        gr=[t if t is not None else torch.zeros_like(p) for t,p in zip(gr,params)]
        g2=torch.cat([t.flatten() for t in gr])
        hv=torch.autograd.grad((g2*v).sum(),params,allow_unused=True)
        hv=[t if t is not None else torch.zeros_like(p) for t,p in zip(hv,params)]
        a+=torch.cat([t.flatten() for t in hv]).detach(); setth(th)
    model.zero_grad(); return a/nb
def fvp(th,v,nb=50):
    a=torch.zeros(P)
    for _ in range(nb):
        x,y=get_batch(); model.zero_grad(); _,l=model(x,y); l.backward()
        g=gfl(); a+=g*float((g*v).sum()); setth(th)
    model.zero_grad(); return a/nb
def sheet(th,seed):
    gen=torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.stack([fvp(th,torch.randn(P,generator=gen))
                                        for _ in range(R)],1))[0]
EV=[get_batch() for _ in range(12)]
def L():
    t=0.0
    with torch.no_grad():
        for x,y in EV: _,v=model(x,y); t+=float(v)
    return t/len(EV)
step=0
for W in (140,260):
    while step<W:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); Q=sheet(th,11); Q2=sheet(th,22)
    sp=float((Q.T@Q2).pow(2).sum()/R)
    gen=torch.Generator().manual_seed(99)
    Qr=torch.linalg.qr(torch.randn(P,R,generator=gen))[0]
    print(f"\n  === step {W}, SPLIT_F {sp:.3f} ===")
    for nm,B in (("Fisher e1",Q),("random 3-plane",Qr)):
        HB=torch.stack([hvp(th,B[:,j]) for j in range(R)],1)
        dia=B.T@HB; off=HB-B@dia
        nd=float(torch.linalg.matrix_norm(dia)); no=float(off.norm())
        print(f"    {nm:>16}  ||e1 H e1|| {nd:.5f}   ||e1 H e2|| {no:.5f}   "
              f"off/diag {no/max(nd,1e-30):.2f}")
    setth(th); L0=L(); prev=flat()
    for _ in range(5):
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    u=flat()-prev; setth(th)
    pf=Q@(Q.T@u); pr=u-pf
    setth(th+pf); a=L()-L0
    setth(th+pr); b=L()-L0
    setth(th+u); c=L()-L0
    print(f"    dL(P_F u) {a:+.5f}  dL(resid) {b:+.5f}  sum {a+b:+.5f}  full {c:+.5f}")
    print(f"    non-additivity {c-(a+b):+.5f}  "
          f"({100*abs(c-(a+b))/max(abs(a)+abs(b),1e-30):.0f}% of the parts)")
