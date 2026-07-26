"""
STRAND-GENERATOR BENCHMARK.
Judge candidate generators of s_{t+1}=sign(update) by strand quality per backward
FLOP, using metrics that matter (weighted sign agreement, frontier recall,
one-step loss), NOT gradient MSE.
Generators:
 1 exact       : s = sign(true update)                      [cost 1.0]
 2 persist     : s = previous sign (no backward)            [cost 0.0]
 3 FA-like     : sign of row-random-mixed gradient          [cost ~1.0, tests transport]
 4 stale-r gate: recompute exact only on low-r frac, persist rest (coarse-to-fine)
 5 momentum    : s = sign(m) (Adam's stored memory, no new backward)  [cost 0.0]
Metrics per step: weighted sign agreement A_w, frontier recall, one-step dL.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def st(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
gen=torch.Generator().manual_seed(7); BMAP={}
def fa(g):
    out=g*0
    for n,p in named:
        a,b=SPAN[n]; G=g[a:b]
        if p.dim()==2:
            R=p.shape[0]
            if (n,R) not in BMAP: BMAP[(n,R)]=torch.eye(R)+0.5*torch.randn(R,R,generator=gen)/np.sqrt(R)
            out[a:b]=(BMAP[(n,R)]@G.view(p.shape)).flatten()
        else: out[a:b]=G
    return out
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel()
def Aw(strue,shat,w):  # weighted sign agreement
    return float((w*(strue==shat).double()).sum()/max(float(w.sum()),1e-9))
res={g:{"Aw":[],"rec":[]} for g in ["persist","momentum","FA","stale-r 20%"]}
prev=flat(); s_prev=None; r_prev=None
for step in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); g=gradv()
    b4=flat(); 
    m=st(o,"exp_avg")
    o.step(); af=flat(); u=af-b4; s_true=torch.sign(u); w=u.abs()
    r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    thr=torch.quantile(r[torch.randperm(P)[:100000]],0.106); Ftrue=(r<thr)
    if s_prev is not None and step>5:
        cand={"persist":s_prev,
              "momentum":torch.sign(m),
              "FA":torch.sign(fa(g)),
              "stale-r 20%":torch.where(r_prev<torch.quantile(r_prev[torch.randperm(P)[:100000]],0.20),
                                        s_true, s_prev)}
        for gname,shat in cand.items():
            res[gname]["Aw"].append(Aw(s_true,shat,w))
            # frontier recall: does the generator's implied active set (where it DIFFERS
            # from persist, i.e. where it says a change happens) cover the true frontier?
            changed=(shat!=s_prev)
            rec=float((changed&Ftrue).sum())/max(float(Ftrue.sum()),1)
            res[gname]["rec"].append(rec)
    s_prev=s_true; r_prev=r.clone(); prev=af; del g,u,m,r
    if step%40==0: gc.collect(); print(f"    step {step} ({time.time()-t0:.0f}s)", flush=True)
print("="*76); print("  STRAND GENERATORS: quality vs the exact strand"); print("="*76)
print(f"  {'generator':>14}{'new backward?':>15}{'wtd sign agree':>16}{'frontier recall':>17}")
COST={"persist":"no","momentum":"no","FA":"yes(~full)","stale-r 20%":"20% of full"}
for g in ["persist","momentum","FA","stale-r 20%"]:
    print(f"  {g:>14}{COST[g]:>15}{np.mean(res[g]['Aw']):>16.3f}{np.mean(res[g]['rec']):>17.3f}")
print("\n  wtd sign agree = agreement weighted by |update| (only big coords matter)")
print("  frontier recall = fraction of true active set the generator flags as changing")
print("\n  reading: 'no new backward' generators (persist, momentum) set the free baseline;")
print("  FA tests whether a non-symmetric FULL backward keeps the strand;")
print("  stale-r 20% tests coarse-to-fine (compute 20%, persist rest).")
print(f"\n  time {time.time()-t0:.0f}s")
