"""STAGE 3 (transport against a known floor) + STAGE 4 (does it predict d_F-dot?).

Stage 2 calibrated the instrument: m=6,nb=2 gives SPLIT=0.405, m=3,nb=12 gives
0.743, m=2,nb=16 gives 0.915. Earlier transport numbers (0.003-0.35) sat UNDER
their own floor and were unreadable. Here m=3, nb=12, nd=80, so consecutive
overlap has room to be meaningful.

Stage 3: ov(Q_t, Q_{t+1}) / SPLIT_t, with SPLIT re-measured at every checkpoint
rather than assumed constant -- it is a property of the local spectrum, not of
the settings alone.

Stage 4: the falsifiable prediction. Does principal-angle evolution of the
Krylov bundle predict the rate of change of the forward/backward Fisher
distance? Regression of d_F-dot on transport, with a shuffled-time null.
MF pinned to 2 rounds throughout.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
M,NB,ND=3,12,80
CKS=[4,8,14,22,32,45,62,85,115,155,205]
def build(D,mf=2):
    src=RAW[:CUT].replace("D=256; N_HEADS=4","D=%d; N_HEADS=4"%D,1)
    src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, %d):"%(mf+1),1)
    src=src.replace("    if pc == N_STU-1:","    if False:",1)
    src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                    "    if False:",1)
    G={}; buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(src,G)
    return G
out={}
for D in (96,128):
    G=build(D); model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]
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
        Q=torch.stack(V,1)
        HQ=torch.stack([hvp(th,Q[:,j]) for j in range(Q.shape[1])],1)
        T=(Q.T@HQ).numpy(); return Q,(T+T.T)/2
    def ov(A,B):
        k=min(A.shape[1],B.shape[1])
        return float((A[:,:k].T@B[:,:k]).pow(2).sum()/k)
    prev=flat(); Qp=None; step=0; rows=[]
    print(f"\n  === D={D}  m={M} nb={NB} nd={ND}")
    print(f"  {'s':>5}{'val':>8}{'phi':>7}{'nneg':>6}{'SPLIT':>8}{'ovQ':>7}{'ratio':>7}{'dF':>8}")
    for ck in CKS:
        while step<ck:
            x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        th=flat()
        Q,T=kry(th); Q2,_=kry(th)
        sp=ov(Q,Q2); o=ov(Qp,Q) if Qp is not None else float('nan'); Qp=Q.clone()
        e=np.linalg.eigvalsh(T)
        lam=float(T[0,0]); om=float(np.linalg.norm(T[1:,0]))
        phi=float(np.degrees(np.arctan2(om,lam)))
        u=th-prev; prev=th.clone()
        v=torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
        w=1.0/(v.sqrt()+1e-8); n=(P//1024)*1024
        Tw=th[:n].view(1024,-1)*w[:n].view(1024,-1).sqrt()
        Uw=u[:n].view(1024,-1)*w[:n].view(1024,-1).sqrt()
        tn=Tw/(Tw.norm(dim=1,keepdim=True)+1e-30); un=Uw/(Uw.norm(dim=1,keepdim=True)+1e-30)
        dF=float((tn-un).norm(dim=1).mean())
        rows.append(dict(ck=ck,val=float(ev(model,n=5)),phi=phi,nneg=int((e<0).sum()),
                         split=sp,ovQ=o,ratio=o/max(sp,1e-9),dF=dF)); model.train()
        r=rows[-1]
        print(f"  {ck:>5}{r['val']:>8.3f}{phi:>7.1f}{r['nneg']:>6}{sp:>8.3f}"
              f"{o:>7.3f}{r['ratio']:>7.3f}{dF:>8.4f}",flush=True)
    out[str(D)]=rows
    json.dump(out,open("/home/claude/work/res_stage34.json","w"),indent=2)
    del G,model,params; gc.collect()
print("\n=== STAGE 4: does transport predict d_F-dot? ===")
rg=np.random.default_rng(0)
for D,rows in out.items():
    r=np.array([x["ratio"] for x in rows[1:]]); dF=np.array([x["dF"] for x in rows])
    ck=np.array([x["ck"] for x in rows],dtype=float)
    dd=np.gradient(dF,ck)[1:]
    m=np.isfinite(r)&np.isfinite(dd)
    c=float(np.corrcoef(r[m],dd[m])[0,1]) if m.sum()>3 else float('nan')
    nul=[abs(np.corrcoef(rg.permutation(r[m]),dd[m])[0,1]) for _ in range(500)]
    print(f"  D={D}: corr(transport, dF_dot) = {c:+.3f}   "
          f"shuffled-time null |corr| mean {np.mean(nul):.3f} p95 {np.percentile(nul,95):.3f}"
          f"   {'SIGNAL' if abs(c)>np.percentile(nul,95) else 'no signal'}")
    print(f"        mean ovQ/SPLIT {np.nanmean([x['ratio'] for x in rows]):.3f}, "
          f"mean SPLIT {np.mean([x['split'] for x in rows]):.3f}")
