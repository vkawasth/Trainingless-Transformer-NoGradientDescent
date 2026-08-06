"""THE LPV TEST: DECOUPLE mu FROM TIME BY POOLING TWO CAPACITIES.

K = K(mu) and K = K(t) are indistinguishable on a single run, where mu and time
correlate at 0.971. Two capacities pass through overlapping mu-ranges at
DIFFERENT steps, so pooling them breaks the collinearity by construction.

The test: for operator pairs drawn from the two runs, does ||K_i - K_j|| track
mu-distance after controlling for time-distance?

  partial(mu | time) strongly positive -> K depends on mu: LPV
  partial(time | mu) dominant          -> K depends on t: drift only

mu is taken as (LN gain, val) since those are the slowest measured quantities
and the ones with predictive content. Reported with the collinearity so the test
can be judged rather than assumed valid.

This also produces the pooled dataset an encoder would need, at ~130 points
rather than 68.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
M_,NB,ND=3,4,15
out={}
for D_,CKS in ((96,list(range(150,354,3))),(128,list(range(200,404,3)))):
    src=RAW[:CUT].replace("D=256; N_HEADS=4","D=%d; N_HEADS=4"%D_,1)
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
    step=0; prev=flat(); rows=[]
    for ck in CKS:
        while step<ck:
            x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        th=flat(); u=th-prev; prev=th.clone()
        g=drift(th); V=[g/(g.norm()+1e-30)]
        for _ in range(M_-1):
            w=hvp(th,V[-1])
            for q in V: w=w-float((w*q).sum())*q
            if float(w.norm())<1e-10: break
            V.append(w/w.norm())
        Q=torch.stack(V,1)
        HQ=torch.stack([hvp(th,Q[:,j]) for j in range(Q.shape[1])],1)
        T=(Q.T@HQ).numpy(); T=(T+T.T)/2
        vv=torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
        Md=1.0/(vv.sqrt()+1e-8); un=u/(u.norm()+1e-30)
        ln=[float(p.data.mean()) for n,p in named if "ln" in n.lower()]
        rows.append(dict(ck=ck,val=float(ev(model,n=4)),
                         cu=[float(x) for x in (Q.T@un)],
                         sig=[float(x) for x in np.linalg.eigvalsh(T)],
                         cv=[float(x) for x in (Q.T@((Md*g)/((Md*g).norm()+1e-30)))],
                         dF=float((u*Md).norm()/(u.norm()+1e-30)),
                         ln=float(np.mean(ln)) if ln else 0.0)); model.train()
    out[str(D_)]=rows
    print(f"  D={D_}: {len(rows)} ckpts, val {rows[0]['val']:.4f} -> {rows[-1]['val']:.4f}, "
          f"LN {rows[0]['ln']:.5f} -> {rows[-1]['ln']:.5f}",flush=True)
    json.dump(out,open("/home/claude/work/res_lpvtest.json","w"),indent=2)
    del G,model,params; gc.collect()
W=24; Ks=[];mus=[];ts=[];src_=[]
for D_,rows in out.items():
    X=np.array([r["cu"]+r["sig"]+r["cv"]+[r["dF"],r["ln"]] for r in rows])
    Z=(X-X.mean(0))/(X.std(0)+1e-9)
    for s in range(0,len(rows)-W-1):
        A=Z[s:s+W]; dB=Z[s+1:s+W+1]-A
        Ks.append(np.linalg.lstsq(A,dB,rcond=None)[0].ravel())
        mus.append([np.mean([r["ln"] for r in rows[s:s+W]]),
                    np.mean([r["val"] for r in rows[s:s+W]])])
        ts.append(np.mean([r["ck"] for r in rows[s:s+W]])); src_.append(D_)
Ks=np.array(Ks); mus=np.array(mus); ts=np.array(ts); src_=np.array(src_)
mz=(mus-mus.mean(0))/(mus.std(0)+1e-12)
print(f"\n  {len(Ks)} operators pooled ({(src_=='96').sum()} + {(src_=='128').sum()})")
KD=[];TD=[];MD=[];CROSS=[]
for i in range(len(Ks)):
    for j in range(i+1,len(Ks)):
        KD.append(np.linalg.norm(Ks[i]-Ks[j])); TD.append(abs(ts[i]-ts[j]))
        MD.append(np.linalg.norm(mz[i]-mz[j])); CROSS.append(src_[i]!=src_[j])
KD,TD,MD,CROSS=map(np.array,(KD,TD,MD,CROSS))
def part(a,b,c):
    ra=a-np.polyval(np.polyfit(c,a,1),c); rb=b-np.polyval(np.polyfit(c,b,1),c)
    return float(np.corrcoef(ra,rb)[0,1])
print(f"\n  === LPV test, pooled ===")
print(f"  collinearity corr(time dist, mu dist) = {np.corrcoef(TD,MD)[0,1]:+.3f}"
      f"   (was +0.971 single-run)")
print(f"  corr(K dist, time dist)   = {np.corrcoef(KD,TD)[0,1]:+.3f}")
print(f"  corr(K dist, mu dist)     = {np.corrcoef(KD,MD)[0,1]:+.3f}")
print(f"  partial(mu | time)        = {part(MD,KD,TD):+.3f}")
print(f"  partial(time | mu)        = {part(TD,KD,MD):+.3f}")
m=CROSS
print(f"\n  cross-run pairs only (n={m.sum()}):")
print(f"    corr(K dist, mu dist)   = {np.corrcoef(KD[m],MD[m])[0,1]:+.3f}")
print(f"    partial(mu | time)      = {part(MD[m],KD[m],TD[m]):+.3f}")
