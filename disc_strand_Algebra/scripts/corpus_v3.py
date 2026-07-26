"""
Fix C3 grammar (must produce long-range MI with variety) and add a real-text
corpus (Python source from the stdlib -- genuine hierarchical structure).
Re-measure the structure axis.
"""
import numpy as np, json, os, glob
from collections import Counter
from math import log2
V=256; NTOK=80000
rng=np.random.default_rng(2)

def metrics(a):
    a=np.asarray(a)
    grams=[tuple(a[i:i+8]) for i in range(0,len(a)-8,4)]
    c=Counter(grams); reuse=sum(v for v in c.values() if v>1)/len(grams)
    uniq_reused=sum(1 for v in c.values() if v>1)/len(c)   # VARIETY of reused grams
    lag=16; xt=a[:-lag].tolist(); xl=a[lag:].tolist(); n=len(xt)
    jc=Counter(zip(xt,xl)); px=Counter(xt); py=Counter(xl); mi=0.0
    for (u,w),c2 in jc.items():
        pj=c2/n; mi+=pj*log2(pj/((px[u]/n)*(py[w]/n)))
    return reuse, uniq_reused, mi

# C3 fixed: PCFG where non-terminals reliably reproduce terminal subsequences
def c_hier():
    NT=20; TERM=range(NT,NT+80)
    # each NT -> 2 fixed expansions of other NTs/terminals (deterministic reuse + choice)
    rules={s:[[int(x) for x in rng.integers(0,NT+80,3)] for _ in range(2)] for s in range(NT)}
    def exp(sym,d):
        if sym>=NT: return [sym-NT]                 # terminal -> token id 0..79
        if d<=0:    return [rng.integers(0,80)]
        return [t for x in rules[sym][rng.integers(0,2)] for t in exp(x,d-1)]
    out=[]
    while len(out)<NTOK: out.extend(exp(rng.integers(0,NT),5))
    return np.array([t%V for t in out[:NTOK]])

# C4 real: tokenize python stdlib source at byte level (genuine structure)
def c_real():
    txt=""
    for f in sorted(glob.glob("/usr/lib/python3.12/*.py"))[:60]:
        try: txt+=open(f,encoding="utf-8",errors="ignore").read()
        except: pass
        if len(txt)>NTOK*2: break
    b=np.frombuffer(txt.encode("utf-8","ignore"),dtype=np.uint8).astype(int)
    return (b%V)[:NTOK]

os.makedirs("/home/claude/w/corpora",exist_ok=True)
print(f"{'corpus':>14}{'vocab':>7}{'reuse':>8}{'reused-variety':>15}{'longMI':>9}")
built={}
for name,fn in [("C3_hier",c_hier),("C4_real",c_real)]:
    a=fn().astype(int); json.dump(a.tolist(),open(f"/home/claude/w/corpora/{name}.json","w"))
    r,uv,mi=metrics(a); built[name]=(r,uv,mi)
    print(f"{name:>14}{len(set(a.tolist())):>7}{r:>8.3f}{uv:>15.3f}{mi:>9.3f}")
# re-print the others for the full axis
for name in ["C1_iid","C0_repeat","C2_template"]:
    a=np.array(json.load(open(f"/home/claude/w/corpora/{name}.json")))
    r,uv,mi=metrics(a)
    print(f"{name:>14}{len(set(a.tolist())):>7}{r:>8.3f}{uv:>15.3f}{mi:>9.3f}")
print("\n  KEY axis = long-range MI (compositional structure with variety):")
print("  C1_iid ~ C0_repeat(no variety) < C2 < C3_hier < C4_real  <- if this holds, axis is clean")
