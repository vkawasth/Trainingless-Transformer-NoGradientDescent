"""Imperfect-coordination variants of the efficient coalition (sec:watch): python watchtoy_variants.py partial|lagged.
partial: R1 is not recruited and stays honest; lagged: R2 follows the coalition three steps late. Writes watchtoy_variant_<which>.json."""
import sys, json, numpy as np; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce import watchtoy as WT, bridgetoy as BT, stalkbank as SB
W=WT.setup(); P=W['P']; gB=W['gB']; which=sys.argv[1]
orig=WT.history
def hist_variant(name,W):
    H=orig('efficient',W); out=[]
    for t,(cells,tv) in enumerate(H):
        new=[]
        for c in cells:
            if which=='partial' and c[0]=='R1': c=('R1',c[1],BT.table(P,(2,3)))            # R1 not recruited: stays honest
            if which=='lagged' and c[0]=='R2':                                            # R2 follows 3 steps late
                lam=min(max(t-3,0)/10,1); c=('R2',c[1],BT.table((1-lam)*P+lam*W['p1'],(2,3)))
            new.append(c)
        out.append((new,tv))
    return out
WT.history=hist_variant
rows=WT.run('efficient',W)
for r in rows: r['s']['anchor_excess']=max(0.0,r['s']['anchor_residual']-r['s']['residual']) if r['t']>=15 else 0.0
json.dump(rows,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),f'watchtoy_variant_{which}.json'),'w'),default=float); print(which,'done')
