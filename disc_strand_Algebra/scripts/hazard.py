"""
SIGN-FLIP HAZARD.
H1: P(flip) = f(|m|/sqrt(v))   -- Adam's own normalised momentum
H2: flips are spatially clustered (tiles approach a boundary together),
    tested as over-dispersion of the per-tile flip fraction vs binomial.
"""
import re, time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def st(o,key):
    out=[]
    for _,p in named:
        s=o.state.get(p,{})
        out.append(s[key].flatten() if key in s else torch.zeros(p.numel()))
    return torch.cat(out)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); NB=10
SUB=torch.randperm(P)[:400000]
NT=1024; e=np.linspace(0,P,NT+1).astype(int)
binflip=np.zeros(NB); bincnt=np.zeros(NB)
overd=[]; rates=[]; auc=[]
prev=flat(); sp=None
for s in range(1,161):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4; sg=torch.sign(d)
    if sp is not None and s>10:
        fl=(sg!=sp)
        rates.append(float(fl.double().mean()))
        m=st(o,"exp_avg").abs(); v=st(o,"exp_avg_sq").sqrt()
        r=(m/(v+1e-12))[SUB]; f=fl[SUB].double()
        q=torch.quantile(r, torch.linspace(0,1,NB+1)[1:-1].double().float())
        b=torch.bucketize(r,q)
        for i in range(NB):
            msk=(b==i)
            if msk.any(): binflip[i]+=float(f[msk].sum()); bincnt[i]+=float(msk.sum())
        # AUC of -r predicting flip (low ratio -> flip)
        n1=int(f.sum()); n0=len(f)-n1
        if 0<n1<len(f):
            rk=torch.argsort(torch.argsort(-r)).double()
            auc.append(float((rk[f>0].sum()-n1*(n1-1)/2)/(n1*n0)))
        # per-tile over-dispersion
        pt=np.array([float(fl[e[i]:e[i+1]].double().mean()) for i in range(NT)])
        pbar=pt.mean(); nsz=P/NT
        overd.append(pt.std()/max(np.sqrt(pbar*(1-pbar)/nsz),1e-30))
    sp=sg; prev=af; del b4,af,d
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*80); print("  H1: DOES |m|/sqrt(v) PREDICT SIGN FLIPS?"); print("="*80)
print(f"  mean flip rate {100*np.mean(rates):.1f}%")
print(f"\n  {'decile of |m|/sqrt(v)':>24}{'flip rate':>12}{'vs mean':>10}")
for i in range(NB):
    fr=binflip[i]/max(bincnt[i],1)
    print(f"  {'D'+str(i+1)+(' (lowest)' if i==0 else ' (highest)' if i==NB-1 else ''):>24}"
          f"{100*fr:>11.1f}%{fr/np.mean(rates):>10.2f}x")
print(f"\n  AUC of the ratio for predicting flip = {np.mean(auc):.3f}   (0.5 = no signal)")
print("\n"+"="*80); print("  H2: ARE FLIPS SPATIALLY CLUSTERED?"); print("="*80)
od=np.array(overd)
print(f"  per-tile flip-fraction dispersion / binomial expectation")
print(f"    mean {od.mean():.2f}x   min {od.min():.2f}x   max {od.max():.2f}x   ({NT} tiles)")
print(f"  1.0 = independent flips;  >>1 = tiles flip together")
print(f"\n  time {time.time()-t0:.0f}s")
