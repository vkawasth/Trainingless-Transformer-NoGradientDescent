"""
DOES GD COST TRACK CORPUS INFORMATION CONTENT?
Hold architecture, length, LR, batch fixed. Vary corpus complexity only.
For each corpus: measure its own bigram entropy floor H, then count steps to
reach H+eps.  If steps ~ H, cost tracks what is extracted.  If steps ~ const,
cost is unrelated to information content.
"""
import time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
t0=time.time()
torch.manual_seed(0); rng=np.random.default_rng(0)
BASE_LEN=1364; LOOPS=60; SEQ=64; BATCH=8; DM=128; NH=4; NL=4; STEPS=400
def make_corpus(V, mode):
    if mode=="const":  base=np.zeros(BASE_LEN,dtype=np.int64)
    elif mode=="cycle": base=(np.arange(BASE_LEN)%V).astype(np.int64)
    else:               base=rng.integers(0,V,BASE_LEN).astype(np.int64)
    ids=np.tile(base,LOOPS)
    C=np.zeros((V,V))
    np.add.at(C,(ids[:-1],ids[1:]),1.0)
    P=C/np.maximum(C.sum(1,keepdims=True),1e-30)
    row=C.sum(1)/C.sum()
    Hb=float(-(row[:,None]*P*np.log(np.maximum(P,1e-30))).sum())
    return torch.tensor(ids), Hb
class Blk(nn.Module):
    def __init__(s):
        super().__init__(); s.dh=DM//NH
        for n_ in ["WQ","WK","WV","op"]: setattr(s,n_,nn.Linear(DM,DM,bias=False))
        s.ln=nn.LayerNorm(DM); s.g=nn.Linear(DM,2*DM,bias=False)
        s.o=nn.Linear(2*DM,DM,bias=False); s.n=nn.LayerNorm(DM)
        for w in [s.WQ,s.WK,s.WV,s.op,s.g,s.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(s,h):
        B,L,_=h.shape
        q,k,v=[getattr(s,n_)(h).view(B,L,NH,s.dh).transpose(1,2) for n_ in ["WQ","WK","WV"]]
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        h=s.ln(h+s.op(o.transpose(1,2).reshape(B,L,DM)))
        return s.n(h+s.o(F.gelu(s.g(h))))
class LM(nn.Module):
    def __init__(s,V):
        super().__init__(); s.te=nn.Embedding(V,DM); s.pe=nn.Embedding(SEQ,DM)
        s.bl=nn.ModuleList([Blk() for _ in range(NL)]); s.lnf=nn.LayerNorm(DM)
        s.head=nn.Linear(DM,V,bias=False)
        nn.init.normal_(s.te.weight,std=0.02); nn.init.normal_(s.pe.weight,std=0.02)
        s.V=V
    def forward(s,x,y=None):
        h=s.te(x)+s.pe(torch.arange(x.shape[1]))
        for b in s.bl: h=b(h)
        lo=s.head(s.lnf(h))
        return lo,(F.cross_entropy(lo.reshape(-1,s.V),y.reshape(-1)) if y is not None else None)
print("="*84); print("  DOES GD COST TRACK CORPUS INFORMATION?  (arch, length, LR, batch fixed)")
print("="*84)
print(f"  {'corpus':>18}{'V':>6}{'H_bigram':>10}{'floor hit':>11}{'steps to H+.05':>16}"
      f"{'steps to H+.01':>16}{'final':>9}")
rows=[]
for mode,V in [("const",2),("cycle",2),("cycle",8),("cycle",64),("rand",8),("rand",64),("rand",1017)]:
    ids,Hb=make_corpus(V,mode)
    torch.manual_seed(1); m=LM(V)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.95),weight_decay=0.1)
    n=len(ids)//SEQ-1
    def batch():
        i=torch.randint(0,n,(BATCH,))*SEQ
        return (torch.stack([ids[a:a+SEQ] for a in i]),
                torch.stack([ids[a+1:a+SEQ+1] for a in i]))
    s05=s01=None; hist=[]
    for s in range(1,STEPS+1):
        m.train(); x,y=batch(); _,l=m(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        lv=float(l); hist.append(lv)
        if s05 is None and lv<=Hb+0.05: s05=s
        if s01 is None and lv<=Hb+0.01: s01=s
    lab=f"{mode} V={V}"
    print(f"  {lab:>18}{V:>6}{Hb:>10.4f}{'yes' if s01 else 'no':>11}"
          f"{(s05 if s05 else -1):>16}{(s01 if s01 else -1):>16}{hist[-1]:>9.4f}", flush=True)
    rows.append((lab,Hb,s05,s01,hist[-1]))
H=np.array([r[1] for r in rows]); S=np.array([r[2] if r[2] else STEPS for r in rows],dtype=float)
print(f"\n  corr(H_bigram, steps to H+0.05) = {np.corrcoef(H,S)[0,1]:+.3f}")
print(f"  steps range: {S.min():.0f} to {S.max():.0f}   ({S.max()/max(S.min(),1):.1f}x)")
print(f"  entropy range: {H.min():.4f} to {H.max():.4f} nats")
print("\n  scaling: steps per nat of corpus entropy")
for lab,Hb,s05,s01,fin in rows:
    if s05: print(f"    {lab:>18}  H={Hb:.4f}  steps={s05:>4}  steps/nat={s05/max(Hb,1e-6):>9.1f}")
print(f"\n  time {time.time()-t0:.0f}s")
