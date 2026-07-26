"""
IS THE STRAND (SIGN FIELD) COMPRESSIBLE?
Measure the conditional entropy of the update sign, in bits per parameter per step:
  H(s_t)                      marginal
  H(s_t | s_{t-1})            temporal (= H of the flip rate)
  H(s_t | s_{t-1}, row ctx)   + neuron-level context
  H(s_t | s_{t-1}, r_i)       + Adam's normalised update (the hazard variable)
Lower bound on any lossless sign codec.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def state(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
TN="blocks.2.ff.g.weight"; R,C=512,256
a0,_=SPAN[TN]; IDX=torch.arange(a0,a0+R*C)
def H(p):
    p=np.clip(p,1e-12,1-1e-12); return float(-(p*np.log2(p)+(1-p)*np.log2(1-p)))
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
prev=flat(); sp=None
acc={"marg":[], "flip":[], "rowH":[], "rH":[], "bothH":[]}
NBR=6; NBH=6
for s in range(1,121):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    sg=(torch.sign(d)[IDX]>0).view(R,C)
    m=state(o,"exp_avg").abs()[IDX]; v=state(o,"exp_avg_sq").sqrt()[IDX]
    r=(m/(v+1e-12)).view(R,C)
    if sp is not None and s>10:
        fl=(sg!=sp)
        acc["marg"].append(H(float(sg.double().mean())))
        pf=float(fl.double().mean()); acc["flip"].append(H(pf))
        # condition on row flip fraction (neuron context)
        rf=fl.double().mean(1)                       # per-row flip fraction
        q=torch.quantile(rf,torch.linspace(0,1,NBR+1)[1:-1].double())
        b=torch.bucketize(rf,q)
        hh=0.0
        for i in range(NBR):
            msk=(b==i)
            if msk.any():
                w=float(msk.sum())/R; p=float(fl[msk].double().mean()); hh+=w*H(p)
        acc["rowH"].append(hh)
        # condition on r (hazard variable), per parameter
        rr=r.flatten(); ff=fl.flatten().double()
        qq=torch.quantile(rr[torch.randperm(len(rr))[:60000]],torch.linspace(0,1,NBH+1)[1:-1])
        bb=torch.bucketize(rr,qq); hr=0.0
        for i in range(NBH):
            msk=(bb==i)
            if msk.any():
                w=float(msk.sum())/len(rr); p=float(ff[msk].mean()); hr+=w*H(p)
        acc["rH"].append(hr)
        # joint: r bin x row-flip bin
        hj=0.0; brow=b.repeat_interleave(C)
        for i in range(NBH):
            for j in range(NBR):
                msk=(bb==i)&(brow==j)
                if msk.sum()>50:
                    w=float(msk.sum())/len(rr); p=float(ff[msk].mean()); hj+=w*H(p)
        acc["bothH"].append(hj)
    sp=sg; del b4,af,d,m,v,r
    if s%40==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*80); print(f"  SIGN-FIELD ENTROPY  ({TN}, {R*C:,} params)"); print("="*80)
M=lambda k: np.mean(acc[k])
print(f"  {'conditioning':>42}{'bits/param/step':>18}{'vs raw':>10}")
rows=[("none (raw sign field)", M("marg")),
      ("previous step's sign", M("flip")),
      ("prev sign + neuron (row) flip context", M("rowH")),
      ("prev sign + Adam r = |m|/sqrt(v)", M("rH")),
      ("prev sign + r + neuron context", M("bothH"))]
for lab,h in rows:
    print(f"  {lab:>42}{h:>18.4f}{h/rows[0][1]:>10.3f}")
best=rows[-1][1]
print(f"\n  raw sign payload   : {R*C:,} bits/step = {R*C/8/1024:.1f} KB")
print(f"  best conditional   : {R*C*best/8/1024:.1f} KB/step   "
      f"({100*(1-best/rows[0][1]):.1f}% reduction)")
print(f"  full fp32 update   : {R*C*32/8/1024:.1f} KB/step")
print(f"  => sign+scalar vs fp32: {32/best:.1f}x   sign alone vs fp32: 32.0x")
print(f"\n  time {time.time()-t0:.0f}s")
