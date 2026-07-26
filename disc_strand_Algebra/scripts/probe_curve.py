import numpy as np
SK=np.load("SK.npy"); G=np.concatenate([SK[i] for i in range(7)],axis=1)  # (80,3584)
chord=G.sum(0); chord=chord/np.linalg.norm(chord)
print("="*70); print("  HOW WELL DOES AN n-STEP PROBE ESTIMATE THE CHORD? (120->200)"); print("="*70)
print(f"  {'n steps':>9}{'cos(probe,chord)':>19}{'1/sqrt(n) model':>17}")
for n in [1,2,5,10,20,30,40,60,79]:
    p=G[:n].sum(0); c=float(p@chord/np.linalg.norm(p))
    print(f"  {n:>9}{c:>19.4f}{np.sqrt(n/80.):>17.3f}")
print("\n  (n=80 is the chord itself, cos=1 by construction; the question is the approach)")
# same but from a random start offset, averaged - unbiased estimate of a *local* probe
print("\n  averaged over 8 random start offsets (a genuinely local probe):")
rng=np.random.default_rng(0)
print(f"  {'n steps':>9}{'mean cos':>12}{'std':>8}")
for n in [5,10,20,30,40]:
    cs=[]
    for st in rng.integers(0,80-n,size=8):
        p=G[st:st+n].sum(0); cs.append(float(p@chord/np.linalg.norm(p)))
    print(f"  {n:>9}{np.mean(cs):>12.4f}{np.std(cs):>8.4f}")
