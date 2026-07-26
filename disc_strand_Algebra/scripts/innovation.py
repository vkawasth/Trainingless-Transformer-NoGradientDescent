"""
IS THE FRONTIER DRIVEN BY BATCH INNOVATION?
 (1) Fixed-batch vs fresh-batch: does the frontier stop migrating when the batch
     stops changing? Jaccard overlap of active sets under each.
 (2) Batch-surprise -> frontier entry: does KL(batch tokens || running dist)
     correlate with frontier turnover V(t)?
 (3) The innovation split: how much of g_t is E[g|H_t] (=Adam's m, predictable)
     vs epsilon_t (batch innovation)? cos(g_t, m_t) measures the predictable part.
"""
import time, gc, numpy as np, torch, json
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
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); FRAC=0.106; SUB=torch.randperm(P)[:150000]
def frontier_run(fixed):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    if fixed: FB=get_batch()
    ACT=[]; cosgm=[]
    for s in range(1,101):
        model.train(); x,y = FB if fixed else get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward()
        g=gradv(); m=st(o,"exp_avg")
        if s>5 and float(m.norm())>0: cosgm.append(float(g@m/(g.norm()*m.norm()+1e-30)))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
        thr=torch.quantile(r[SUB],FRAC); ACT.append((r[SUB]<thr))
        del g,m,r
    return ACT, np.mean(cosgm)
def jseq(ACT,k):
    js=[]
    for t in range(20,len(ACT)-k,3):
        a=ACT[t]; b=ACT[t+k]; inter=float((a&b).sum()); uni=float((a|b).sum())
        js.append(inter/max(uni,1))
    return np.mean(js)
print("="*72); print("  (1)+(3) FROZEN vs FRESH BATCH: frontier migration & innovation"); print("="*72)
for fixed,lab in [(True,"fixed batch"),(False,"fresh batch")]:
    ACT,cg=frontier_run(fixed)
    print(f"\n  {lab}:  cos(g_t, m_t) = {cg:+.3f}   (high => gradient predictable from Adam memory)")
    print(f"    frontier Jaccard  k1 {jseq(ACT,1):.3f}  k5 {jseq(ACT,5):.3f}"
          f"  k20 {jseq(ACT,20):.3f}  k40 {jseq(ACT,40):.3f}")
print("\n  if fixed-batch Jaccard stays HIGH (frontier freezes) while fresh decays,")
print("  the migration is batch-innovation driven, not intrinsic optimizer dynamics.")
# (2) batch surprise vs frontier turnover
print("\n"+"="*72); print("  (2) BATCH SURPRISE vs FRONTIER TURNOVER"); print("="*72)
ids=torch.tensor(json.load(open("/tmp/train_ids.json")),dtype=torch.long)
V=int(model.te.weight.shape[0])
run_freq=np.ones(V)                                   # running token distribution (Laplace)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
prev_act=None; KLs=[]; TOs=[]
for s in range(1,101):
    x,y=get_batch()
    bc=np.bincount(x.flatten().numpy(),minlength=V).astype(float)
    pb=bc/bc.sum(); pr=run_freq/run_freq.sum()
    kl=float((pb[pb>0]*np.log(pb[pb>0]/pr[pb>0])).sum())
    run_freq+=bc
    model.train(); _,l=model(x,y); o.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    r=(st(o,"exp_avg").abs()/(st(o,"exp_avg_sq").sqrt()+1e-12))
    thr=torch.quantile(r[SUB],FRAC); act=(r[SUB]<thr)
    if prev_act is not None and s>10:
        to=float((act&~prev_act).sum())/float(act.sum())
        KLs.append(kl); TOs.append(to)
    prev_act=act; del r
    if s%40==0: gc.collect()
KLs=np.array(KLs); TOs=np.array(TOs)
print(f"  corr( batch KL-surprise , frontier turnover ) = {np.corrcoef(KLs,TOs)[0,1]:+.3f}")
print(f"  batch KL range [{KLs.min():.3f},{KLs.max():.3f}]  turnover range [{TOs.min():.2f},{TOs.max():.2f}]")
print(f"\n  time {time.time()-t0:.0f}s")
