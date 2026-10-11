"""Exploratory (not pre-registered): laxness of floor summarisation (a closure) and of policy imposition (an interior) under transport. Output: results/lax_composition.json"""
import os, sys, json, numpy as np; H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import support as SP
from scipy.optimize import linprog
rng=np.random.default_rng(99)
def min_over_P(c,f,g,lower):
    n,m=len(f),len(g); r=linprog(c,A_eq=SP._transport_eq(n,m),b_eq=np.concatenate([f,g]),bounds=[(l,None) for l in lower.ravel()],method='highs'); return r.fun
def floor_image(K,f,g,lower): return np.array([min_over_P(K[t],f,g,lower) for t in range(K.shape[0])])
def floor_box_push(K,fl):          # floor of K applied to the box {pi >= fl, sum 1}: fl.K_row + (1 - sum fl) * min_j K_row
    return K@fl + (1-fl.sum())*K.min(1)
viol1=strict1=viol2=strict2=0; N=300; cells=0; cells2=0
for _ in range(N):
    n,m=int(rng.integers(2,4)),int(rng.integers(2,4)); f,g=rng.dirichlet(np.ones(n)*2),rng.dirichlet(np.ones(m)*2); M=n*m
    k1,k2=int(rng.integers(2,M+1)),int(rng.integers(2,5)); K1=rng.dirichlet(np.ones(k1)*.5,M).T; K2=rng.dirichlet(np.ones(k2)*.5,k1).T
    z=np.zeros((n,m))
    composite=floor_image(K2@K1,f,g,z)                    # floor of the image of S under gf
    stepwise=floor_box_push(K2,floor_image(K1,f,g,z))      # summarise to floors after f, then push the box through g
    viol1+=int((stepwise>composite+1e-9).any()); strict1+=int((stepwise<composite-1e-9).sum()); cells+=k2
    tau=np.full((n,m),rng.uniform(0,.5)*min(f.min()/m,g.min()/n))
    imposed=floor_image(K1,f,g,tau)                       # impose policy, then transport (tight)
    pushed=np.maximum(floor_image(K1,f,g,z),K1@tau.ravel()) # transport, then impose the pushed floor
    viol2+=int((pushed>imposed+1e-9).any()); strict2+=int((pushed<imposed-1e-9).sum()); cells2+=k1
print(f'summarise-then-transport never exceeds the composite floor: violations {viol1}/{N}; strictly lower in {strict1}/{cells} target cells')
print(f'impose-then-transport >= transport-then-impose: violations {viol2}/{N}; strictly higher in {strict2}/{cells2} target cells')
json.dump(dict(instances=N, summarise_violations=viol1, summarise_strict_cells=strict1, summarise_cells=cells, impose_violations=viol2, impose_strict_cells=strict2, impose_cells=cells2),
          open(os.path.join(H, 'results', 'lax_composition.json'), 'w'), indent=1)
