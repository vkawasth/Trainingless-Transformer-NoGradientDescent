import numpy as np, json
NODES=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
SK=np.load("SK.npy")            # (7, 80, 512) sketched per-step displacements
T=SK.shape[1]; CELLS=[(0,20),(20,40),(40,60),(60,80)]; LAB=["120-140","140-160","160-180","180-200"]
G=np.concatenate([SK[i] for i in range(7)],axis=1)   # (80, 3584) global sketch

def eff_rank(S):
    l=S**2; return (l.sum()**2)/((l**2).sum()+1e-30)
def numrank(S,tol=0.05): return int((S> tol*S[0]).sum())

print("="*76); print("  CELL SPECTRA  (stalk dimension per cell)"); print("="*76)
print(f"  {'cell':>9}{'||d||':>9}{'r_eff':>8}{'rank(5%)':>10}{'rank(10%)':>11}   top-5 sing. (norm.)")
U={}; SVs={}
for (a,b),lab in zip(CELLS,LAB):
    X=G[a:b]; u,s,vt=np.linalg.svd(X,full_matrices=False)
    U[lab]=vt; SVs[lab]=s
    print(f"  {lab:>9}{np.linalg.norm(X):>9.2f}{eff_rank(s):>8.2f}{numrank(s,0.05):>10}{numrank(s,0.10):>11}"
          f"   {' '.join(f'{x:.2f}' for x in (s/s[0])[:5])}")
print("\n  per-node effective rank of the 20-step displacement block")
print(f"  {'cell':>9}"+"".join(f"{g:>8}" for g in NODES))
RK=np.zeros((4,7))
for ci,((a,b),lab) in enumerate(zip(CELLS,LAB)):
    row=[]
    for i in range(7):
        s=np.linalg.svd(SK[i][a:b],compute_uv=False); row.append(eff_rank(s)); RK[ci,i]=eff_rank(s)
    print(f"  {lab:>9}"+"".join(f"{v:>8.2f}" for v in row))

K=6
print("\n"+"="*76); print(f"  RESTRICTION MAPS: principal angles between adjacent cells (k={K})"); print("="*76)
def basis(lab,k=K): return U[lab][:k].T          # (3584,k)
for i in range(3):
    A=basis(LAB[i]); B=basis(LAB[i+1])
    sv=np.linalg.svd(A.T@B,compute_uv=False)
    ang=np.degrees(np.arccos(np.clip(sv,-1,1)))
    print(f"  {LAB[i]} -> {LAB[i+1]}   cos: {' '.join(f'{x:.3f}' for x in sv)}")
    print(f"  {'':>21}deg: {' '.join(f'{x:5.1f}' for x in ang)}   overlap={np.sum(sv**2)/K:.3f}")

print("\n"+"="*76); print("  CELLULAR SHEAF ON THE 4-CELL PATH: H^0 and H^1"); print("="*76)
print("  vertices = cells, edges = adjacencies, stalk = R^k (local subspace coords)")
V=len(LAB); E=V-1
delta=np.zeros((E*K, V*K))
for e in range(E):
    A=basis(LAB[e]); B=basis(LAB[e+1])
    Q,_=np.linalg.qr(np.concatenate([A,B],axis=1))   # edge basis
    W=Q[:,:K]
    Fa=W.T@A; Fb=W.T@B
    delta[e*K:(e+1)*K, e*K:(e+1)*K]     = -Fa
    delta[e*K:(e+1)*K, (e+1)*K:(e+2)*K] =  Fb
s=np.linalg.svd(delta,compute_uv=False)
print(f"  delta singular values: {' '.join(f'{x:.3f}' for x in s)}")
print(f"\n  {'tol':>6}{'rank(delta)':>13}{'dim H^0':>10}{'dim H^1':>10}   reading")
for tol in [0.5,0.2,0.1,0.05,0.01]:
    r=int((s>tol*s[0]).sum()); h0=V*K-r; h1=E*K-r
    read = "all local laws glue" if h1==0 else f"{h1} obstructed direction(s)"
    print(f"  {tol:>6.2f}{r:>13}{h0:>10}{h1:>10}   {read}")
print("\n  (H^0 = globally consistent directions; H^1 = local-to-global obstruction.")
print("   Path graph is contractible, so H^1>0 can only come from restriction maps")
print("   dropping rank -- i.e. genuine failure of local dynamics to glue.)")

SYN=np.load("SYN.npy"); R1=json.load(open("R1.json"))
print("\n"+"="*76); print("  GRAPH SUMMARY"); print("="*76)
tot=sum(R1.values())
print(f"  sum of single-node recoveries = {tot:.1f}%   joint = 100%  -> "
      f"{'SUB-additive (redundant)' if tot>110 else 'SUPER-additive (synergistic)'}")
off=SYN[np.triu_indices(7,1)]
print(f"  edges: {int((off>5).sum())} strongly positive, {int((off<-5).sum())} strongly negative, "
      f"{int((np.abs(off)<=5).sum())} near-zero")
deg=SYN.sum(1)
print(f"\n  {'node':>6}{'solo rec':>10}{'sum edge':>10}   role")
for i,g in enumerate(NODES):
    role = "connector (weak solo, positive edges)" if R1[g]<10 and deg[i]>0 else \
           "carrier (strong solo)" if R1[g]>50 else \
           "redundant (negative edges)" if deg[i]<-20 else "mixed"
    print(f"  {g:>6}{R1[g]:>9.1f}%{deg[i]:>10.1f}   {role}")
