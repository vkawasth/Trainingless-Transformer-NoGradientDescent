"""
ADAPTIVE vs UNIFORM STALE-SIGN REFRESH, matched backward budget.
Uniform: every block refreshes (full backward) every k steps.
Adaptive: per-block refresh interval tau_i = f(r_i); stable blocks refresh
rarely, volatile often -- SAME average backward fraction as uniform.
If adaptive beats uniform -> scheduling on r helps (time-sparsity real).
If adaptive <= uniform -> refresh scheduling inherits the gating pathology.
Between refreshes a block applies sign(m)*scalar (transport), like stale-sign.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def stt(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def V(n=16): return float(eval_val(model,n=n))
P=flat().numel()
# block = neuron row (same tiling as before)
off=0; TILE=torch.zeros(P,dtype=torch.long); tid=0
for n,p in named:
    a=off; b=off+p.numel()
    if p.dim()==2:
        R,C=p.shape
        for r_ in range(R): TILE[a+r_*C:a+(r_+1)*C]=tid; tid+=1
    else: TILE[a:b]=tid; tid+=1
    off=b
NT=tid; cnt=torch.bincount(TILE,minlength=NT).float()
def run(mode, k=2):
    """mode: 'full', 'uniform' (refresh all every k), 'adaptive' (r-scheduled, matched budget)"""
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    prev=flat(); s_prev=None
    last_r=torch.zeros(NT)                      # per-block last-refresh step counter -> interval
    # adaptive: assign each block an interval in {1,2,4} by its running r rank so mean refresh = 1/k
    interval=torch.full((NT,),k)
    for step in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); b4=flat(); m=stt(o,"exp_avg")
        o.step(); af=flat(); u_true=af-b4
        if mode=="full" or s_prev is None:
            setflat(af); cur=af
        else:
            r=(stt(o,"exp_avg").abs()/(stt(o,"exp_avg_sq").sqrt()+1e-12))
            tr=torch.zeros(NT).index_add_(0,TILE,r)/cnt
            if mode=="adaptive":
                # low-r (volatile) -> interval 1; high-r (stable) -> interval 4; mid -> 2
                # tuned so mean(1/interval) == 1/k
                q1,q2=torch.quantile(tr,torch.tensor([0.5,0.83]))
                interval=torch.where(tr<q1, torch.tensor(1),
                          torch.where(tr<q2, torch.tensor(3), torch.tensor(9)))
            refresh_block = ((step % interval)==0) if mode=="uniform" else \
                            (torch.remainder(torch.tensor(step), interval)==0)
            if mode=="uniform":
                do_full = (step % k == 0)
                if do_full: setflat(af); cur=af
                else:
                    a_now=float(u_true.abs().mean()); setflat(b4+s_prev*a_now); cur=flat()
            else:  # adaptive per-block
                rb=refresh_block[TILE]
                a_now=float(u_true.abs().mean())
                u=torch.where(rb, u_true, s_prev*a_now)
                setflat(b4+u); cur=flat()
        s_prev=torch.sign(cur-b4); prev=cur; del b4,af,u_true,m
        if step%40==39: gc.collect()
    return V()
# report backward fraction used
print("="*66); print("  UNIFORM vs ADAPTIVE REFRESH (matched ~1/2 backward budget)"); print("="*66)
vb=run("full")
vu=run("uniform",2)
va=run("adaptive",2)
# adaptive budget check
q_share = 0.5*1/1 + 0.33*1/3 + 0.17*1/9  # mean refresh frequency
print(f"  {'full backward':>22}: val {vb:.4f}   (100% backward)")
print(f"  {'uniform k=2':>22}: val {vu:.4f}   (50% backward)")
print(f"  {'adaptive r-scheduled':>22}: val {va:.4f}   (~{100*q_share:.0f}% backward)")
print(f"\n  uniform beats full at matched compute (prior result). Question: adaptive vs uniform.")
if va < vu*0.98: print("  adaptive BEATS uniform => r-scheduling helps, time-sparsity is real.")
elif va > vu*1.02: print("  adaptive WORSE than uniform => scheduling inherits the gating pathology.")
else: print("  adaptive ~ uniform => r-scheduling adds nothing over uniform staleness.")
print(f"\n  time {time.time()-t0:.0f}s")
