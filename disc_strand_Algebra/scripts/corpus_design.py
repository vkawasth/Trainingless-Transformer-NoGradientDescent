"""
Build 4 corpora spanning a structure axis, matched on vocab, seq len, token count.
Then run the SAME intrinsic-dimension protocol on each:
  freeze theta -> sweep batches -> strand s_B=sign(grad)
  measure: linear vs nonlinear vs kNN prediction of s_B from batch embedding,
           pairwise strand correlation, effective dim (PR), dist->sim corr.
The discriminator: does nonlinear BEAT linear as structure rises? If yes, the
high-dimensionality of the single-sentence result was corpus-triviality, not law.
"""
import numpy as np
V=256; SEQ=64; NTOK=80000   # matched across all corpora
rng=np.random.default_rng(0)

def c_repeat():
    "C0: one repeated sentence (the original degenerate case), length ~120"
    sent=rng.integers(0,V,120)
    reps=NTOK//len(sent)+1
    return np.tile(sent,reps)[:NTOK]

def c_markov():
    "C1: low structure -- order-1 Markov chain (local dependency only)"
    T=rng.dirichlet(np.ones(V)*0.3,size=V)   # sparse-ish transitions
    out=[rng.integers(0,V)]
    for _ in range(NTOK-1): out.append(rng.choice(V,p=T[out[-1]]))
    return np.array(out)

def c_template():
    "C2: mid structure -- 'scientific' templated: fixed frames with slot fillers"
    # frames = fixed token skeletons; slots drawn from topic-specific pools
    nframe=8; frame_len=16
    frames=[rng.integers(0,V,frame_len) for _ in range(nframe)]
    slots=[rng.integers(0,V,frame_len)<0 for _ in range(nframe)]  # placeholder
    pools=[rng.integers(0,V,20) for _ in range(nframe)]           # per-frame vocab pool
    slotmask=[rng.random(frame_len)<0.4 for _ in range(nframe)]
    out=[]
    while len(out)<NTOK:
        f=rng.integers(0,nframe); s=frames[f].copy()
        s[slotmask[f]]=rng.choice(pools[f],slotmask[f].sum())
        out.extend(s.tolist())
    return np.array(out[:NTOK])

def c_hier():
    "C3: high structure -- hierarchical/recursive grammar (nested composition)"
    # simple PCFG: symbols expand to sequences of symbols/terminals
    NT=16; term_start=NT
    rules={s:[ (rng.integers(0,NT+40,rng.integers(2,4))) for _ in range(2)] for s in range(NT)}
    def expand(sym,depth):
        if depth<=0 or sym>=NT: return [ (sym%V) if sym>=NT else (term_start+sym)%V ]
        r=rules[sym][rng.integers(0,2)]; out=[]
        for x in r: out.extend(expand(int(x),depth-1))
        return out
    out=[]
    while len(out)<NTOK:
        out.extend(expand(rng.integers(0,NT),4))
    return np.array([t%V for t in out[:NTOK]])

import json,os
os.makedirs("/home/claude/w/corpora",exist_ok=True)
info={}
for name,fn in [("C0_repeat",c_repeat),("C1_markov",c_markov),
                ("C2_template",c_template),("C3_hier",c_hier)]:
    ids=fn().astype(int).tolist()
    json.dump(ids, open(f"/home/claude/w/corpora/{name}.json","w"))
    a=np.array(ids)
    # structure proxies: unique tokens, bigram entropy, type-token ratio
    big=list(zip(a[:-1],a[1:]))
    from collections import Counter
    bc=Counter(big); p=np.array(list(bc.values()),float); p/=p.sum()
    bent=-(p*np.log2(p)).sum()
    info[name]=dict(ntok=len(a),vocab=int(len(set(ids))),
                    bigram_entropy=round(float(bent),2),
                    ttr=round(len(set(ids))/len(ids),4))
print(f"{'corpus':>14}{'ntok':>8}{'vocab':>7}{'bigramH':>10}{'TTR':>9}")
for k,v in info.items():
    print(f"{k:>14}{v['ntok']:>8}{v['vocab']:>7}{v['bigram_entropy']:>10}{v['ttr']:>9}")
print("\n  structure axis: C0 (repeat) < C1 (markov) < C2 (template) < C3 (hierarchical)")
print("  matched: vocab space V=256, seq=64, ntok=80k. differ in compositional structure.")
