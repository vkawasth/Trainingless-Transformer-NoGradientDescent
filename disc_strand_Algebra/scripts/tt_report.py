import pickle, numpy as np
D=pickle.load(open("tiletree.pkl","rb")); S=D["series"]; M=D["meta"]
LAY=["EMB","L0","L1","L2","L3","L4","L5"]; GRP=["LN","FF","W_Q","W_K","W_V","W_O","EMB"]
print("="*84); print("  TILE TREE  --  STRUCTURE"); print("="*84)
print(f"  P = {D['P']:,}   steps = {len(M)}   series = {len(S)}")
ov=[m["overlap"] for m in M]; sw=next((m["step"] for m in M if m["overlap"]>0.15), None)
print(f"  overlap: 10% for steps 1-{sw-1 if sw else len(M)}, 25% from {sw}"
      f"   (E(64) crossed 0.05)")
print(f"\n  {'layer':>6}{'fwd tiles':>11}{'bwd tiles':>11}   groups present")
for L in LAY:
    f=[k for k in S if k.startswith(L+"|") and k.endswith("|fwd")]
    if not f: continue
    nf=S[f[0]]["nt"]; b=[k for k in S if k.startswith(L+"|") and k.endswith("|bwd")]
    gs=sorted({k.split("|")[1] for k in f})
    print(f"  {L:>6}{nf:>11}{S[b[0]]['nt']:>11}   {' '.join(gs)}")
print("\n"+"="*84); print("  FWD = weights after step  |  BWD = applied update  (tile mean & |max|)"); print("="*84)
def show(gk, side, stat, rows=LAY, t=(0,-1)):
    print(f"\n  --- {gk}  {side}  tile-{stat}   (across tiles: mean of tile values)")
    print(f"  {'layer':>6}{'ntiles':>8}{'step 1':>12}{'step 60':>12}{'step 120':>12}{'step 199':>12}{'range':>10}")
    for L in rows:
        k=f"{L}|{gk}|{side}"
        if k not in S: continue
        A=S[k][stat]; n=S[k]["nt"]; f0=S[k]["first_step"]
        idx=lambda st: max(0,min(A.shape[0]-1, st-f0))
        v=[A[idx(st)].mean() for st in (1,60,120,199)]
        sp=A.mean(1)
        print(f"  {L:>6}{n:>8}"+"".join(f"{x:>12.3e}" for x in v)+f"{sp.max()-sp.min():>10.2e}")
for gk in ("LN","FF"):
    show(gk,"fwd","mean"); show(gk,"bwd","max")
show("W_K","fwd","mean"); show("W_K","bwd","max")
print("\n"+"="*84); print("  TILE HETEROGENEITY  --  is structure visible across tiles within a layer?"); print("="*84)
print("  CV across tiles of the tile-|max| (bwd, applied update), at 3 steps")
print(f"  {'layer':>6}{'group':>7}{'ntiles':>8}{'CV@60':>10}{'CV@120':>10}{'CV@199':>10}")
for L in LAY:
    for gk in ("LN","FF","W_K"):
        k=f"{L}|{gk}|bwd"
        if k not in S: continue
        A=S[k]["max"]; f0=S[k]["first_step"]
        def cv(st):
            i=max(0,min(A.shape[0]-1,st-f0)); r=A[i]
            return r.std()/(r.mean()+1e-30)
        print(f"  {L:>6}{gk:>7}{S[k]['nt']:>8}{cv(60):>10.3f}{cv(120):>10.3f}{cv(199):>10.3f}")
