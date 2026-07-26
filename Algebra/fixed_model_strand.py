"""
FIXED MODEL, CHANGING BATCHES: does the corpus select among a finite set of
strand modes, or is each batch's strand independent?
Freeze theta. Compute strand s(B_i)=sign(grad) for many batches. Measure:
 (1) pairwise strand correlation between batches (high => shared modes)
 (2) effective dimension of the strand set (participation ratio of SVD spectrum)
 (3) does batch token-overlap predict strand similarity? (corpus selects mode)
Run at two checkpoints (early J40, late) to see if modes sharpen with training.
"""
import time, gc, numpy as np, torch, json
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
P=flat().numel(); SUB=torch.randperm(P)[:120000]
def collect_strands(nbatch=40):
    S=[]; TOK=[]
    theta=flat()
    for _ in range(nbatch):
        x,y=get_batch()
        model.zero_grad(); _,l=model(x,y); l.backward()
        S.append(torch.sign(gradv()[SUB]).numpy().astype(np.int8))
        TOK.append(set(x.flatten().tolist()))
        import torch as _t
        with _t.no_grad():
            i=0
            for _,p in named:
                k=p.numel(); p.data.copy_(theta[i:i+k].view_as(p)); i+=k   # restore theta (defensive)
    return np.stack(S).astype(float), TOK
def analyse(tag):
    S,TOK=collect_strands(40)
    # pairwise strand correlation
    Sc=S-S.mean(0)
    C=np.corrcoef(S)                          # batch x batch
    off=C[np.triu_indices(len(C),1)]
    # effective dimension: participation ratio of singular values of S
    sv=np.linalg.svd(S-S.mean(0),compute_uv=False); sv2=sv**2
    pr=(sv2.sum()**2)/(sv2**2).sum()
    # token overlap vs strand similarity
    n=len(TOK); ov=[]; ss=[]
    for i in range(n):
        for j in range(i+1,n):
            jo=len(TOK[i]&TOK[j])/max(len(TOK[i]|TOK[j]),1)
            sim=float((S[i]==S[j]).mean())
            ov.append(jo); ss.append(sim)
    print(f"\n  [{tag}]")
    print(f"    mean pairwise strand correlation : {off.mean():+.3f}  (0=independent,1=identical)")
    print(f"    mean pairwise sign agreement     : {np.mean(ss):.3f}")
    print(f"    effective strand dimension (PR)  : {pr:.1f} of {len(S)} batches")
    print(f"    corr(token-overlap, strand-sim)  : {np.corrcoef(ov,ss)[0,1]:+.3f}")
    return off.mean(), pr
print("="*70); print("  FIXED MODEL, VARYING BATCH: finite strand modes or independent?"); print("="*70)
# early checkpoint
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
for _ in range(40):
    x,y=get_batch(); _,l=model(x,y); o.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
analyse("after 40 steps")
# later checkpoint
for _ in range(80):
    x,y=get_batch(); _,l=model(x,y); o.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
analyse("after 120 steps")
print("\n  high correlation / low PR / positive overlap-sim corr => corpus selects")
print("  among few shared strand modes (dictionary exists). low corr / high PR =>")
print("  each batch is an independent strand (no finite dictionary).")
print(f"\n  time {time.time()-t0:.0f}s")
