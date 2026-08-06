"""WHICH STAGE DESTROYS THE KRYLOV ALIGNMENT?

Capture of g by the Krylov space is ~99.7%; capture of dtheta is 2.6-11.8%.
Adam is the natural suspect, but "Adam" has four stages and only one of them
need be responsible. Isolating each, all against the SAME Q:

  cap(g)        raw drift                            expect ~1 (Q is built on g)
  cap(Dg)       D = diag(1/(sqrt(v)+eps)) applied    the preconditioner alone
  cap(m)        first moment                         momentum alone
  cap(Dm)       D applied to m                       both, = the Adam direction
  cap(dtheta)   the actual applied update            includes clipping, decay

If cap(Dg) collapses while cap(m) stays high, the diagonal preconditioner is the
cause. If cap(m) collapses too, momentum/averaging shares responsibility.

Also reported per layer, since capture of dtheta was measured to rise monotonically
with depth (0.32% at layer 0 to 3.53% at layer 5).
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
M,NB,ND=3,10,50
CKS=[6,12,24,44,76,120]
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]
named=[(n,p) for n,p in model.named_parameters()]; params=[p for _,p in named]
P=sum(p.numel() for p in params)
for p in params: p.requires_grad_(True)
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
def setth(t):
    with torch.no_grad():
        i=0
        for p in params:
            k=p.numel(); p.data.copy_(t[i:i+k].view_as(p)); i+=k
def gfl():
    return torch.cat([(p.grad.flatten() if p.grad is not None else
        torch.zeros(p.numel())) for p in params]).clone()
def drift(th,n=ND):
    a=torch.zeros(P)
    for _ in range(n):
        x,y=get_batch(); model.zero_grad(); _,l=model(x,y); l.backward(); a+=gfl(); setth(th)
    model.zero_grad(); return a/n
def hvp(th,v,nb=NB):
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
def kry(th):
    g=drift(th); V=[g/(g.norm()+1e-30)]
    for _ in range(M-1):
        w=hvp(th,V[-1])
        for u in V: w=w-float((w*u).sum())*u
        if float(w.norm())<1e-10: break
        V.append(w/w.norm())
    return torch.stack(V,1),g
def cap(Q,x): return float((Q.T@x).pow(2).sum()/((x*x).sum()+1e-30))
step=0; prev=flat(); rows=[]
print(f"  null = r/P = {M/P:.2e}\n")
print(f"  {'s':>5}{'val':>8}{'cap(g)':>9}{'cap(Dg)':>10}{'cap(m)':>9}"
      f"{'cap(Dm)':>10}{'cap(dth)':>10}")
for ck in CKS:
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); u=th-prev; prev=th.clone()
    Q,g=kry(th)
    v=torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
    m=torch.cat([opt.state[p]["exp_avg"].flatten() for p in params])
    D=1.0/(v.sqrt()+1e-8)
    r=dict(ck=ck,val=float(ev(model,n=5)),cg=cap(Q,g),cDg=cap(Q,D*g),
           cm=cap(Q,m),cDm=cap(Q,D*m),cdth=cap(Q,u)); model.train()
    rows.append(r)
    print(f"  {ck:>5}{r['val']:>8.3f}{r['cg']:>9.4f}{r['cDg']:>10.4f}"
          f"{r['cm']:>9.4f}{r['cDm']:>10.4f}{r['cdth']:>10.4f}",flush=True)
    json.dump(rows,open("/home/claude/work/res_precond.json","w"),indent=2)
print(f"\n  per-layer, last checkpoint:")
i=0; sp={}
for nm,p in named:
    k=p.numel()
    if nm.startswith("blocks."): sp.setdefault(nm.split(".")[1],[]).append((i,i+k))
    i+=k
Dg=D*g; Dm=D*m
for L,s in sorted(sp.items()):
    idx=torch.cat([torch.arange(a,b) for a,b in s])
    QL,_=torch.linalg.qr(Q[idx])
    print(f"    layer {L}: g {cap(QL,g[idx]):.4f}  Dg {cap(QL,Dg[idx]):.4f}  "
          f"m {cap(QL,m[idx]):.4f}  dth {cap(QL,u[idx]):.4f}")
c=rows[-1]
print(f"\n  drop g->Dg: {c['cg']:.4f} -> {c['cDg']:.4f}  "
      f"({100*(1-c['cDg']/max(c['cg'],1e-9)):.1f}% lost)")
print(f"  drop g->m : {c['cg']:.4f} -> {c['cm']:.4f}  "
      f"({100*(1-c['cm']/max(c['cg'],1e-9)):.1f}% lost)")
