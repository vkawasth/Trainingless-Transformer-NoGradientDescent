"""
COMBINED 2-D COMPLEX: horizontal (overlap) + vertical (refinement) edges.
Strands = root-to-leaf paths; two strands are adjacent iff their tiles overlap
at some layer. Compare transport: overlap-only vs tree vs combined.
"""
import pickle, numpy as np, itertools
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
D=pickle.load(open("sheaf_overlap.pkl","rb")); ST=D["ST"]; FLR=D["FLR"]
LAY=["L0","L1","L2","L3","L4","L5"]; BWD=[1024,512,256,64,32,16]
print("="*84); print("  COMBINED COMPLEX: horizontal overlaps + vertical refinement"); print("="*84)
for g in ("FF","LN"):
    # global vertex numbering
    offs={}; tot=0
    for L,n in zip(LAY,BWD):
        offs[L]=tot; tot+=n
    V=tot
    A=lil_matrix((V,V))
    # horizontal: adjacent tiles (overlap exists)
    for L,n in zip(LAY,BWD):
        for i in range(n-1):
            A[offs[L]+i, offs[L]+i+1]=1; A[offs[L]+i+1, offs[L]+i]=1
    # vertical: refinement parent-child
    for (La,na),(Lb,nb) in zip(zip(LAY,BWD), zip(LAY[1:],BWD[1:])):
        big,small=(La,na),(Lb,nb)
        if na<nb: big,small=(Lb,nb),(La,na)
        r=big[1]//small[1]
        for c in range(big[1]):
            A[offs[big[0]]+c, offs[small[0]]+c//r]=1
            A[offs[small[0]]+c//r, offs[big[0]]+c]=1
    A=csr_matrix(A)
    deg=np.asarray(A.sum(1)).ravel()
    def fiedler(M):
        d=np.asarray(M.sum(1)).ravel(); d[d==0]=1
        Dm=csr_matrix((1/np.sqrt(d),(range(len(d)),range(len(d)))),shape=M.shape)
        Lap=csr_matrix(np.eye(M.shape[0]))-Dm@M@Dm
        w=eigsh(Lap,k=3,which='SM',return_eigenvectors=False,maxiter=5000)
        return np.sort(w)[1]
    # horizontal-only graph
    Ah=lil_matrix((V,V))
    for L,n in zip(LAY,BWD):
        for i in range(n-1):
            Ah[offs[L]+i,offs[L]+i+1]=1; Ah[offs[L]+i+1,offs[L]+i]=1
    Ah=csr_matrix(Ah)
    print(f"\n  {g}: V={V}, horizontal edges={int(Ah.sum()//2)}, "
          f"total edges={int(A.sum()//2)}")
    try:
        fh=fiedler(Ah[:BWD[0],:BWD[0]])   # finest layer path only (connected)
        print(f"    Fiedler, finest layer path only        : {fh:.3e}   (1/n^2 = {1/BWD[0]**2:.3e})")
    except Exception as e: print("    horizontal fiedler failed",e)
    try:
        fc=fiedler(A)
        print(f"    Fiedler, combined complex              : {fc:.3e}")
        print(f"    ratio combined/horizontal              : {fc/fh:.1f}x")
    except Exception as e: print("    combined fiedler failed",e)
    # graph distances: sample
    import collections
    def bfs_diam(M,src,N):
        d=-np.ones(N,dtype=int); d[src]=0; q=collections.deque([src])
        while q:
            u=q.popleft()
            for v in M.indices[M.indptr[u]:M.indptr[u+1]]:
                if d[v]<0: d[v]=d[u]+1; q.append(v)
        return d
    dh=bfs_diam(Ah,0,V); dc=bfs_diam(A,0,V)
    print(f"    ecc(tile 0) horizontal-only            : {dh[dh>=0].max()}")
    print(f"    ecc(tile 0) combined                   : {dc.max()}")
    print(f"    transport speedup from vertical edges  : {dh[dh>=0].max()/max(dc.max(),1):.0f}x")
print("\n"+"="*84); print("  STRANDS ON THE COMBINED COMPLEX"); print("="*84)
for g in ("FF","LN"):
    fine=BWD[0]
    S=[]
    for L,n in zip(LAY,BWD):
        X=ST[(g,L)]["T"].copy(); X=(X-X.mean(0))/(X.std(0)+1e-30)
        S.append(X[(np.arange(fine)*n)//fine])
    S=np.concatenate(S,1)
    # strand adjacency: leaves i,j adjacent iff tiles overlap at SOME layer
    # (adjacent leaf indices at the finest layer, or same tile at a coarser one)
    ds=np.linalg.norm(S[:-1]-S[1:],axis=1)          # neighbour strand distance
    rng=np.random.default_rng(0)
    pr=rng.integers(0,fine,(4000,2)); pr=pr[pr[:,0]!=pr[:,1]]
    dr=np.linalg.norm(S[pr[:,0]]-S[pr[:,1]],axis=1)
    print(f"\n  {g}: strands={fine}, dim={S.shape[1]}")
    print(f"    mean distance, overlap-adjacent strands : {ds.mean():.3f} +/- {ds.std():.3f}")
    print(f"    mean distance, random strand pairs      : {dr.mean():.3f} +/- {dr.std():.3f}")
    print(f"    ratio                                   : {dr.mean()/ds.mean():.2f}x")
    for k in (1,2,4,16,64,256):
        d=np.linalg.norm(S[:-k]-S[k:],axis=1)
        print(f"      lag {k:>4}: mean {d.mean():.3f}")
