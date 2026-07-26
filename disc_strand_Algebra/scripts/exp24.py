import time, re, copy, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
CKS=list(range(20,221,20)); TH={}; VAL={}
for s in range(1,max(CKS)+1):
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if s in CKS: TH[s]=flat(); VAL[s]=float(eval_val(model,n=8))
print(f"  trained, {len(CKS)} checkpoints ({time.time()-t0:.0f}s)", flush=True)
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
model=model.double(); named=list(model.named_parameters())
def keyL(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
ks=[];sl={};off=0
for n,p in named:
    k=keyL(n)
    if k not in sl: sl[k]=[];ks.append(k)
    sl[k].append((off,off+p.numel())); off+=p.numel()
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[k]]) for k in ks]; K=len(ks)
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def G(v):
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
H=1e-4; REC={}; Us={}
for s in CKS:
    th=TH[s].double(); g0=G(th); nV=[float(g0[I].norm()) for I in IDX]
    A=[]
    for m in range(K):
        d=torch.zeros_like(th); d[IDX[m]]=-g0[IDX[m]]/max(nV[m],1e-30)
        A.append((G(th+H*d)-g0)/H)
    us=[]; res=[]; er=[]
    for l in range(K):
        Mt=torch.stack([A[m][IDX[l]] for m in range(K)])
        _,S,Vt=torch.linalg.svd(Mt,full_matrices=False)
        u=Vt[0]; us.append(u.float())
        pg=-g0[IDX[l]]
        res.append(float((pg-(pg@u)*u).norm()/max(pg.norm(),1e-30)))
        sv=S.numpy()**2; er.append(float((sv.sum()**2)/max((sv**2).sum(),1e-300)))
    C=np.array([[float(A[m][IDX[l]]@us[l].double()) for m in range(K)] for l in range(K)])
    REC[s]=dict(res=np.array(res), er=np.mean(er), cond=np.linalg.cond(C),
                gn=float(g0.norm()), val=VAL[s])
    Us[s]=us
    del A,g0; gc.collect()
    print(f"    ck {s:>3} done ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*80)
print("  FORWARD/BACKWARD ALIGNMENT:  fraction of the gradient OUTSIDE the reachable image")
print("="*80)
print(f"  {'step':>6}{'val':>9}{'||g||':>9}{'mean resid':>12}{'EMB':>8}{'L0':>7}{'L2':>7}{'L5':>7}"
      f"{'eff-rank':>10}{'cond C':>10}")
for s in CKS:
    r=REC[s]
    print(f"  {s:>6}{r['val']:>9.4f}{r['gn']:>9.4f}{100*r['res'].mean():>11.1f}%"
          f"{100*r['res'][0]:>7.0f}%{100*r['res'][1]:>6.0f}%{100*r['res'][3]:>6.0f}%"
          f"{100*r['res'][6]:>6.0f}%{r['er']:>10.2f}{r['cond']:>10.3g}")
rr=np.array([REC[s]['res'].mean() for s in CKS])
print(f"\n  trend: first half {100*rr[:5].mean():.1f}%   second half {100*rr[5:].mean():.1f}%")
sl_=np.polyfit(np.array(CKS,dtype=float), rr, 1)
print(f"  linear fit slope {sl_[0]*100:+.4f}% per step   (extrapolated zero at step "
      f"{-sl_[1]/sl_[0]:.0f})" if sl_[0]<0 else f"  linear fit slope {sl_[0]*100:+.4f}% per step (rising)")
print("\n"+"="*80)
print("  TRANSPORT OF THE BLOCK DIRECTIONS u_l  (unitary rotation vs deformation)")
print("="*80)
print(f"  {'step->step':>14}" + "".join(f"{k:>8}" for k in ks))
for a,b in zip(CKS[:-1],CKS[1:]):
    row=[abs(float(Us[a][l].double()@Us[b][l].double())) for l in range(K)]
    print(f"  {str(a)+'->'+str(b):>14}" + "".join(f"{x:>8.3f}" for x in row))
print("\n  time %.0fs"%(time.time()-t0))
