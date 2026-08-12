"""CHUNK TOKENISATION: WHAT SURVIVES WHEN RELATIONS ARE REMOVED?

The corpus is 1364 word tokens over a 1017-word vocabulary, carrying 2-form and
3-form relations. Re-tokenising at 3-5 word boundaries makes each chunk a single
symbol. With ~340 chunks in a 1364-token base, most chunks are UNIQUE, so each
deterministically predicts the next: H1 collapses toward zero and the task
becomes memorising a cycle rather than learning compositional structure.

The manipulation removes the relational content while keeping the same text.
Measured, at matched step (200) rather than matched loss, since the point is the
representation and not the loss:

  H0,H1,H2   corpus entropies, to confirm the manipulation did what is intended
  nnz        bigram graph density, which sets the spectral E0
  PR(I_H)    interaction rank, and the fraction of positive off-diagonals
  arc        omega vs lambda
  eff        path efficiency
  tile c_T   row-scale coherence at n=D, the strand observable
  fwd.bwd    weight-state against applied-update alignment
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib, collections, math
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
BASE=json.load(open("/tmp/train_ids.json"))[:1364]
V0=json.load(open("/tmp/vocab.json")) if __import__("os").path.exists("/tmp/vocab.json") else None
def entropies(seq,k=3):
    out=[]
    for order in range(k+1):
        if order==0:
            c=collections.Counter(seq); n=len(seq)
            out.append(-sum(v/n*math.log(v/n) for v in c.values()))
        else:
            ctx=collections.defaultdict(collections.Counter)
            for i in range(len(seq)-order):
                ctx[tuple(seq[i:i+order])][seq[i+order]]+=1
            tot=sum(sum(c.values()) for c in ctx.values()); h=0.0
            for c in ctx.values():
                n=sum(c.values())
                h+=n/tot*(-sum(v/n*math.log(v/n) for v in c.values()))
            out.append(h)
    return out
# build chunked corpus: groups of 4 words -> one symbol
CH=4
chunks=[tuple(BASE[i:i+CH]) for i in range(0,len(BASE)-CH+1,CH)]
uniq={c:i for i,c in enumerate(sorted(set(chunks)))}
cseq=[uniq[c] for c in chunks]
print(f"  word corpus : len {len(BASE)}  vocab {len(set(BASE))}")
print(f"  chunk corpus: len {len(cseq)}  vocab {len(uniq)}  "
      f"({100*len(uniq)/len(chunks):.0f}% of chunks unique)")
hw=entropies(BASE); hc=entropies(cseq)
print(f"  H0..H3 word : {np.round(hw,3)}")
print(f"  H0..H3 chunk: {np.round(hc,3)}")
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
def run(name,seq,vsz):
    n=len(json.load(open("/tmp/train_ids.json")))//len(seq)+1
    json.dump((seq*n)[:len(seq)*max(n//10,20)],open("/tmp/train_ids.json","w"))
    json.dump((seq*n)[:len(seq)*max(n//60,5)],open("/tmp/val_ids.json","w"))
    src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
    src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
    src=src.replace("    if pc == N_STU-1:","    if False:",1)
    src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                    "    if False:",1)
    G={}; buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(src,G)
    log=buf.getvalue()
    nz=[l.strip() for l in log.split("\n") if "nnz=" in l]
    model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]; D_=G["D"]
    named=[(nm,p) for nm,p in model.named_parameters()]; params=[p for _,p in named]
    P=sum(p.numel() for p in params)
    for p in params: p.requires_grad_(True)
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
    def hvp(th,v,nb=4):
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
    def grp(nm):
        if nm.startswith("te") or nm.startswith("pe"): return "EMB"
        if "ln" in nm.lower() or ".n." in nm: return "LN"
        if ".ff." in nm: return "FF"
        return "ATTN"
    GS=["EMB","LN","FF","ATTN"]; span={}; i=0
    for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
    masks={g:torch.zeros(P,dtype=torch.bool) for g in GS}
    for nm,(a,b) in span.items(): masks[grp(nm)][a:b]=True
    th0=flat(); prev=th0.clone(); pl=0.0; step=0
    while step<200:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        cur=flat(); pl+=float((cur-prev).norm()); prev=cur
    th=flat(); eff=float((th-th0).norm())/max(pl,1e-9)
    g=torch.zeros(P)
    for _ in range(25):
        x,y=get_batch(); model.zero_grad(); _,l=model(x,y); l.backward(); g+=gfl(); setth(th)
    model.zero_grad(); g/=25
    gn=g/(g.norm()+1e-30); Hv=hvp(th,gn)
    lam=float((gn*Hv).sum()); om=float((Hv-lam*gn).norm())
    prevth=th.clone()
    for _ in range(5):
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    u=flat()-prevth; setth(th)
    d={gg:u*masks[gg] for gg in GS}; Hd={gg:hvp(th,d[gg]) for gg in GS}
    I=np.array([[float((d[a]*Hd[b]).sum()) for b in GS] for a in GS]); I=(I+I.T)/2
    e=np.linalg.eigvalsh(I); aa=np.abs(e)
    off=I[~np.eye(4,dtype=bool)]
    # tile coherence at n=D
    cT=[]
    for nm,p in named:
        if p.dim()==2 and p.shape[0]==D_ and ("ff.o" in nm or "op" in nm):
            a,b=span[nm]; blk=g[a:b].view(D_,-1)
            coh=float(torch.sign(blk).mean(1).abs().mean())
            sh=torch.sign(blk.flatten()[torch.randperm(blk.numel())]).view(D_,-1)
            cT.append(coh/max(float(sh.mean(1).abs().mean()),1e-9))
    print(f"\n  === {name} ===  {nz[0] if nz else ''}")
    print(f"    val {float(ev(model,n=6)):.4f}   eff {eff:.4f}   "
          f"lam {lam:.4f}  om {om:.4f}  phi {np.degrees(np.arctan2(om,lam)):.1f}")
    print(f"    PR(I_H) {float(aa.sum()**2/(aa**2).sum()):.2f}/4   "
          f"off-diag positive {100*(off>0).mean():.0f}%   "
          f"c_T/base {np.mean(cT):.4f}   fwd.bwd {float((th*u).sum()/(th.norm()*u.norm()+1e-30)):.4f}")
    del G,model,params; gc.collect()
run("WORD tokens",BASE,1017)
run("CHUNK tokens (4 words)",cseq,len(uniq))
