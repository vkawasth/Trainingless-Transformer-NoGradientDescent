"""
IS THE COUPLING epsilon SMALL ACROSS TENSOR TYPES, LAYERS, AND TRAINING PHASE?
epsilon = conditioned adjacent-pair joint-flip excess over independence,
measured per tensor type and in an early vs late training window.
If epsilon stays << 1 everywhere, the CST theorem's premise is established.
"""
import re, time, gc, numpy as np, torch
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
# representative tensors, each reshaped to (rows, cols) so adjacency = same-row neighbours
TN={"FF g (L2)":("blocks.2.ff.g.weight",512,256),
    "FF o (L2)":("blocks.2.ff.o.weight",256,512),
    "W_Q (L2)":("blocks.2.attn.WQ.weight",256,256),
    "W_V (L2)":("blocks.2.attn.WV.weight",256,256),
    "W_O (L2)":("blocks.2.attn.op.weight",256,256),
    "FF g (L0)":("blocks.0.ff.g.weight",512,256),
    "FF g (L5)":("blocks.5.ff.g.weight",512,256),
    "embed":("te.weight",1017,128)}
NB=6
def eps_for(SG,RB):
    # SG,RB : (T, rows, cols).  epsilon = weighted joint-flip excess over independence,
    # conditioned on both coords' r-bin, for horizontally-adjacent pairs.
    T=SG.shape[0]
    fl=(SG[1:]!=SG[:-1]).astype(np.int8)          # (T-1,rows,cols)
    rb=RB[:-1]
    edges=np.quantile(rb.flatten(), np.linspace(0,1,NB+1)[1:-1])
    binr=np.digitize(rb,edges)
    A=fl[:,:,:-1]; B=fl[:,:,1:]; ba=binr[:,:,:-1]; bb=binr[:,:,1:]
    exc=[]; wt=[]
    for b1 in range(NB):
        for b2 in range(NB):
            m=(ba==b1)&(bb==b2)
            if m.sum()<1500: continue
            pj=(A[m]&B[m]).mean(); pa=A[m].mean(); pb=B[m].mean()
            exc.append(pj-pa*pb); wt.append(m.sum())
    if not exc: return np.nan, np.nan
    exc=np.array(exc); wt=np.array(wt)
    ex=np.average(exc,weights=wt)
    base=np.average([ (A[(ba==b1)&(bb==b2)].mean()*B[(ba==b1)&(bb==b2)].mean())
                      for b1 in range(NB) for b2 in range(NB)
                      if ((ba==b1)&(bb==b2)).sum()>=1500], weights=wt)
    return ex, ex/max(base,1e-9)
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
o=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
REC={k:{"S":[],"R":[]} for k in TN}
prev=flat()
for s in range(1,201):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); o.step(); af=flat(); d=af-b4
    m=state(o,"exp_avg").abs(); v=state(o,"exp_avg_sq").sqrt(); r=m/(v+1e-12)
    for k,(nm,R,C) in TN.items():
        a0,_=SPAN[nm]; idx=torch.arange(a0,a0+R*C)
        REC[k]["S"].append(torch.sign(d[idx]).view(R,C).numpy().astype(np.int8))
        REC[k]["R"].append(r[idx].view(R,C).numpy().astype(np.float32))
    prev=af; del b4,af,d,m,v,r
    if s%50==0: gc.collect(); print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*80); print("  COUPLING epsilon ACROSS TENSORS AND PHASE"); print("="*80)
print(f"  {'tensor':>12}{'eps early (1-100)':>20}{'eps late (101-200)':>21}{'rel late':>10}")
allrel=[]
for k in TN:
    S=np.stack(REC[k]["S"]); R=np.stack(REC[k]["R"])
    e_e,r_e=eps_for(S[:100],R[:100]); e_l,r_l=eps_for(S[100:],R[100:])
    allrel += [r_e, r_l]
    print(f"  {k:>12}{e_e:>13.5f}({100*r_e:>4.0f}%){e_l:>14.5f}({100*r_l:>4.0f}%){100*r_l:>9.0f}%")
allrel=np.array([a for a in allrel if not np.isnan(a)])
print(f"\n  relative excess across all tensors/phases: mean {100*allrel.mean():.0f}%"
      f"  range [{100*allrel.min():.0f}%, {100*allrel.max():.0f}%]")
print(f"  absolute epsilon stays a perturbation (rel < 1) in "
      f"{100*np.mean(allrel<1):.0f}% of cases")
print(f"\n  time {time.time()-t0:.0f}s")
