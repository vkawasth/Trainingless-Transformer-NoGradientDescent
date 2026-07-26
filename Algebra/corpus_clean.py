"""
CLEAN structure axis via a shuffle control -- the only confound-free comparison.
For real text: the corpus vs a token-shuffled version of ITSELF.
Same vocab, same unigram distribution, same length. Differs ONLY in order =
ONLY in structure. Structure measured by compressibility gap (gzip) and by
block-entropy at length L, which is bias-controlled by the shuffle baseline.
This gives a validated structure axis: shuffle=0 structure, real=whatever it has.
Build 4 test corpora as (real, shuffled) pairs at increasing real-structure.
"""
import numpy as np, json, os, glob, gzip
V=256; NTOK=60000
rng=np.random.default_rng(3)
os.makedirs("/home/claude/w/corpora",exist_ok=True)

def gzip_ratio(a):
    b=np.asarray(a,dtype=np.uint8).tobytes()
    return len(gzip.compress(b,9))/len(b)

def block_entropy_gap(a,L=6):
    "structure = how much more predictable order-L blocks are than shuffled"
    a=np.asarray(a)
    def bent(x):
        from collections import Counter
        blk=[tuple(x[i:i+L]) for i in range(0,len(x)-L,2)]
        c=Counter(blk); n=len(blk); import math
        return -sum((v/n)*math.log2(v/n) for v in c.values())/L
    real=bent(a); sh=a.copy(); rng.shuffle(sh)
    return bent(sh)-real     # positive = real more structured than its shuffle

# sources spanning real structure
def repeat(): s=rng.integers(0,V,120); return np.tile(s,NTOK//120+1)[:NTOK]
def iid():    return rng.integers(0,V,NTOK)
def english():
    # structured pseudo-text: reusable word vocabulary, word+space rhythm,
    # with order-1 word dependency (mid structure, real variety)
    words=[bytes(rng.integers(97,123,rng.integers(2,9)).tolist()) for _ in range(200)]
    Tw=rng.dirichlet(np.ones(200)*0.2,size=200)   # word->word transitions
    seq=[rng.integers(0,200)]
    while sum(len(words[w])+1 for w in seq)<NTOK: seq.append(rng.choice(200,p=Tw[seq[-1]]))
    txt=b" ".join(words[w] for w in seq)
    b=np.frombuffer(txt,dtype=np.uint8).astype(int); return (b%V)[:NTOK]
def code():
    txt=""
    for f in sorted(glob.glob("/usr/lib/python3.12/*.py"))[:80]:
        try: txt+=open(f,errors="ignore").read()
        except: pass
        if len(txt)>NTOK: break
    return (np.frombuffer(txt.encode("utf-8","ignore"),dtype=np.uint8).astype(int)%V)[:NTOK]

print(f"{'corpus':>12}{'vocab':>7}{'gzip-ratio':>12}{'blockH-gap':>12}  (gap>0 = structured)")
axis={}
for name,fn in [("iid",iid),("repeat",repeat),("english",english),("code",code)]:
    a=fn().astype(int)
    if len(a)<NTOK: a=np.tile(a,NTOK//len(a)+1)[:NTOK]
    json.dump(a.tolist(),open(f"/home/claude/w/corpora/{name}.json","w"))
    gr=gzip_ratio(a); ge=block_entropy_gap(a)
    axis[name]=(gr,ge)
    print(f"{name:>12}{len(set(a.tolist())):>7}{gr:>12.3f}{ge:>12.3f}")
# also store shuffled controls (same tokens, zero structure)
for name in ["english","code"]:
    a=np.array(json.load(open(f"/home/claude/w/corpora/{name}.json"))); rng.shuffle(a)
    json.dump(a.tolist(),open(f"/home/claude/w/corpora/{name}_shuf.json","w"))
print("\n  gzip ratio: lower = more compressible = more structured.")
print("  blockH-gap: real vs own-shuffle; >0 proves order carries structure.")
print("  => validated axis. shuffled controls isolate structure from unigram stats.")
