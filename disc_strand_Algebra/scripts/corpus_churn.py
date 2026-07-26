"""
WHY IS CORPUS COMPLEXITY ABSENT FROM CHURN?
A: corpus has no complexity (one sentence) -> churn is sampling noise, and its
   magnitude is constant, uncorrelated with any complexity measure but correlated
   with batch-composition OVERLAP between consecutive batches.
B: churn is optimizer-geometric, corpus-independent -> turnover constant even
   when we DELIBERATELY vary batch complexity.
Test: vary batch token-diversity on purpose (low-diversity: repeat few positions;
high-diversity: spread across the sentence). Does churn respond?
Plus: does consecutive-batch token overlap predict turnover (composition, not
complexity)?
"""
import time, gc, numpy as np, torch, json
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def stt(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); FRAC=0.106; SUB=torch.randperm(P)[:150000]
ids=torch.tensor(json.load(open("/tmp/train_ids.json")),dtype=torch.long)
SEQ=65; BS=8
def make_batch(bs, spread):
    N=len(ids)-SEQ
    if spread<0.5:
        base=np.random.randint(0,N); offs=(np.full(bs,base)+np.random.randint(0,int(1+spread*200),bs))%N
    else:
        offs=np.random.randint(0,N,bs)
    xs=torch.stack([ids[o:o+SEQ] for o in offs])
    return xs[:,:-1].contiguous(), xs[:,1:].contiguous()
def churn_run(spread):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    prev_act=None; TOs=[]; OVL=[]; prevtok=None
    for s in range(1,81):
        x,y=make_batch(BS,spread)
        toks=set(x.flatten().tolist())
        model.train(); _,l=model(x,y); o.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        r=(stt(o,"exp_avg").abs()/(stt(o,"exp_avg_sq").sqrt()+1e-12))
        thr=torch.quantile(r[SUB],FRAC); act=(r[SUB]<thr)
        if prev_act is not None and s>10:
            TOs.append(float((act&~prev_act).sum())/float(act.sum()))
            if prevtok is not None:
                OVL.append(len(toks&prevtok)/max(len(toks|prevtok),1))
        prev_act=act; prevtok=toks; del r
    return np.mean(TOs), np.std(TOs), (np.corrcoef(OVL[:len(TOs)-1],TOs[1:len(OVL)+1])[0,1] if len(OVL)>3 else np.nan)
print("="*70); print("  DOES CHURN RESPOND TO BATCH DIVERSITY? (Explanation A vs B)"); print("="*70)
print(f"  {'batch spread':>14}{'mean turnover':>15}{'sd':>8}{'corr(tok-overlap,turnover)':>28}")
for sp in (0.0,0.3,0.6,1.0):
    m,sd,c=churn_run(sp)
    print(f"  {sp:>14.1f}{100*m:>14.0f}%{sd:>8.3f}{c:>28.3f}", flush=True)
print("\n  A (corpus has no complexity): turnover ~constant across spread,")
print("    uncorrelated with overlap -> churn is optimizer-intrinsic sampling noise.")
print("  B response to diversity: turnover rises with spread -> corpus DOES drive")
print("    churn magnitude, we were just measuring the wrong complexity variable.")
print(f"\n  time {time.time()-t0:.0f}s")
