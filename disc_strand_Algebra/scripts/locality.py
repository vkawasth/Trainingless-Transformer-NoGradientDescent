"""
WHERE DOES THE NECESSARY ~36% LIVE?
 necessary set N = top 50% of parameters by |net|  (keeps 99.8% of the improvement)
 (0.441 net/path) x (0.814 of |net| in top half) = 35.9% of all motion.
Questions: how is N distributed over strands? uniform, or a fat chunk?
           does the embedding part of N track the corpus?
"""
import re, json, time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
model.load_state_dict(torch.load("init.pt")); th0=flat(); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
D=(flat()-th0).abs(); P=D.numel()
thr=torch.sort(D,descending=True).values[P//2]
NEC=(D>=thr)
print(f"  net computed, |N| = {int(NEC.sum()):,} of {P:,}  ({time.time()-t0:.0f}s)\n", flush=True)
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
print("="*84); print("  (1) WHERE THE NECESSARY MASS SITS, BY COMPONENT"); print("="*84)
def comp(n):
    m=re.match(r"blocks\.(\d+)\.",n)
    if not m: return "EMB/head/lnf"
    return f"L{m.group(1)}"
agg={}
for n,_ in named:
    a,b=SPAN[n]; c=comp(n)
    d=agg.setdefault(c,[0.0,0,0])
    d[0]+=float(D[a:b].sum()); d[1]+=int(NEC[a:b].sum()); d[2]+=b-a
tot=sum(v[0] for v in agg.values())
print(f"  {'component':>14}{'params':>11}{'% of |net|':>12}{'% in N':>9}{'density':>10}")
for c in sorted(agg):
    m,nn_,np_=agg[c]
    print(f"  {c:>14}{np_:>11,}{100*m/tot:>11.1f}%{100*nn_/np_:>8.1f}%{(m/np_)/(tot/P):>10.2f}")
# ---- embedding vs corpus ----
te_a,te_b=SPAN["te.weight"]; Vv=model.te.weight.shape[0]
tokmass=D[te_a:te_b].view(Vv,-1).sum(1).numpy()
tokN=NEC[te_a:te_b].view(Vv,-1).float().mean(1).numpy()
try:
    ids=np.array(json.load(open("/tmp/train_ids.json")),dtype=np.int64)
    cnt=np.bincount(ids,minlength=Vv).astype(float)
    big=np.zeros(Vv); np.add.at(big,ids[:-1],1.0)
    uniq=np.zeros(Vv)
    seen={}
    for a,b in zip(ids[:-1],ids[1:]): seen.setdefault(a,set()).add(b)
    for k,v in seen.items(): uniq[k]=len(v)
    print("\n"+"="*84); print("  (2) DOES THE EMBEDDING PART OF N TRACK THE CORPUS?"); print("="*84)
    ok=cnt>0
    print(f"  tokens present: {int(ok.sum())} of {Vv}")
    print(f"  corr( |net| per token , token frequency )      = "
          f"{np.corrcoef(tokmass[ok],cnt[ok])[0,1]:+.3f}")
    print(f"  corr( |net| per token , log frequency )        = "
          f"{np.corrcoef(tokmass[ok],np.log(cnt[ok]))[0,1]:+.3f}")
    print(f"  corr( |net| per token , # distinct successors ) = "
          f"{np.corrcoef(tokmass[ok],uniq[ok])[0,1]:+.3f}")
    print(f"  corr( fraction-in-N , log frequency )          = "
          f"{np.corrcoef(tokN[ok],np.log(cnt[ok]))[0,1]:+.3f}")
    q=np.argsort(cnt[ok])[::-1]; mm=tokmass[ok][q]
    print(f"\n  {'token freq decile':>20}{'mean |net|/token':>19}{'mean frac in N':>17}")
    for d in range(0,10,3):
        sl=slice(d*len(q)//10,(d+1)*len(q)//10)
        print(f"  {'decile '+str(d+1):>20}{mm[sl].mean():>19.4f}{tokN[ok][q][sl].mean():>17.3f}")
except Exception as e: print("  corpus file unavailable:",e)
# ---- strand locality (blocks only) ----
LAY=["L0","L1","L2","L3","L4","L5"]
GI={}
for n,_ in named:
    m=re.match(r"blocks\.(\d+)\.",n)
    if m: GI.setdefault(f"L{m.group(1)}",[]).append(torch.arange(*SPAN[n]))
GI={k:torch.cat(v) for k,v in GI.items()}
def strand_mass(ladder):
    S=np.zeros(1024)
    for L,nt in zip(LAY,ladder):
        idx=GI[L]; nel=len(idx); e=np.linspace(0,nel,nt+1).astype(int)
        tm=np.array([float(D[idx[e[i]:e[i+1]]].sum()) for i in range(nt)])
        share=1024//nt
        for i in range(nt): S[i*share:(i+1)*share]+=tm[i]/share
    return S
print("\n"+"="*84); print("  (3) STRAND LOCALITY OF THE NECESSARY MASS"); print("="*84)
for nm,lad in [("bwd ladder",[1024,512,256,64,32,16]),("fwd ladder",[16,32,64,256,512,1024])]:
    S=strand_mass(lad); s=np.sort(S)[::-1]; c=np.cumsum(s)/s.sum()
    gini=1-2*np.trapezoid(np.cumsum(np.sort(S))/S.sum(),dx=1/len(S))
    ac=np.corrcoef(S[:-1],S[1:])[0,1]
    print(f"\n  {nm}:  Gini {gini:.3f}   lag-1 autocorr {ac:+.3f}   CV {S.std()/S.mean():.3f}")
    print(f"  {'top k% strands':>16}{'% of strand mass':>19}")
    for k in (1,5,10,25,50):
        print(f"  {k:>15}%{100*c[int(1024*k/100)-1]:>18.1f}%")
print(f"\n  time {time.time()-t0:.0f}s")
