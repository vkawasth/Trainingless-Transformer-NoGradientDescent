"""THE CENTREPIECE: WITHIN- vs BETWEEN-CHECKPOINT KRYLOV VARIABILITY.

Everything geometric in this programme depends on one distinction: is the Krylov
geometry a stable property of the model, or an artifact of stochastic gradients?

At each checkpoint, freeze theta and build N INDEPENDENT Krylov spaces from
disjoint gradient windows. Then:

  WITHIN   pairwise Grassmann distance among the N replicates at the SAME theta
           = sampling noise
  BETWEEN  distance between checkpoint means
           = true evolution

  between >> within  -> the geometry is intrinsic and evolves; the geometric
                        programme has an empirical foundation
  between ~ within   -> what has been called rotation is measurement variability

Instrument at the calibrated setting from Stage 2: m=3, nb=12, nd=80, where
same-theta overlap reached 0.74-0.91. Reported as principal angles, projection
overlap, and effective dimension of the union of replicates -- if the N spaces
were identical the union would have dimension m; if independent, N*m.
Checkpoints span the early excursion, the Fisher minimum, and late training.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
M,NB,ND,NREP=3,12,60,8
CKS=[4,8,40,120,200]
def build(D,mf=2):
    src=RAW[:CUT].replace("D=256; N_HEADS=4","D=%d; N_HEADS=4"%D,1)
    src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, %d):"%(mf+1),1)
    src=src.replace("    if pc == N_STU-1:","    if False:",1)
    src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                    "    if False:",1)
    G={}; buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(src,G)
    return G
G=build(128); model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]
params=[p for _,p in model.named_parameters()]; P=sum(p.numel() for p in params)
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
def drift(th,n):
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
    g=drift(th,ND); V=[g/(g.norm()+1e-30)]
    for _ in range(M-1):
        w=hvp(th,V[-1])
        for u in V: w=w-float((w*u).sum())*u
        if float(w.norm())<1e-12: break
        V.append(w/w.norm())
    return torch.stack(V,1)
def angles(A,B):
    s=np.linalg.svd((A.T@B).numpy(),compute_uv=False)
    return np.degrees(np.arccos(np.clip(s,-1,1)))
def ov(A,B):
    k=min(A.shape[1],B.shape[1])
    return float((A[:,:k].T@B[:,:k]).pow(2).sum()/k)
step=0; res=[]
print(f"  m={M} nb={NB} nd={ND} reps={NREP}, D=128\n")
print(f"  {'s':>5}{'val':>8}{'WITHIN ov':>11}{'sd':>7}{'ang1':>7}{'ang3':>7}"
      f"{'eff dim':>9}{'BETWEEN ov':>12}")
Qbar=None; prev=None
for ck in CKS:
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); Qs=[kry(th) for _ in range(NREP)]
    w=[ov(Qs[i],Qs[j]) for i in range(NREP) for j in range(i+1,NREP)]
    ang=np.array([angles(Qs[i],Qs[j]) for i in range(NREP) for j in range(i+1,NREP)])
    U=torch.cat(Qs,1)
    sv=torch.linalg.svdvals(U).numpy()**2
    eff=float(sv.sum()**2/(sv**2).sum())
    btw=ov(prev,Qs[0]) if prev is not None else float('nan')
    prev=Qs[0].clone()
    v=float(ev(model,n=5)); model.train()
    res.append(dict(ck=ck,val=v,within=float(np.mean(w)),within_sd=float(np.std(w)),
                    ang1=float(ang[:,0].mean()),ang3=float(ang[:,-1].mean()),
                    eff=eff,between=btw))
    print(f"  {ck:>5}{v:>8.3f}{np.mean(w):>11.4f}{np.std(w):>7.3f}"
          f"{ang[:,0].mean():>7.1f}{ang[:,-1].mean():>7.1f}{eff:>9.2f}{btw:>12.4f}",flush=True)
    json.dump(res,open("/home/claude/work/res_replicate.json","w"),indent=2)
print(f"\n  union of {NREP} replicates: eff dim {M} => identical, {NREP*M} => independent")
print(f"\n=== WITHIN vs BETWEEN ===")
for i in range(1,len(res)):
    r=res[i]
    print(f"  step {r['ck']:>4}: within {r['within']:.4f}+-{r['within_sd']:.3f}   "
          f"between(prev) {r['between']:.4f}   "
          f"ratio {r['between']/max(r['within'],1e-9):.3f}")
print("\n  ratio << 1 => evolution exceeds sampling noise: geometry is intrinsic")
print("  ratio ~ 1  => what looked like rotation is measurement variability")
