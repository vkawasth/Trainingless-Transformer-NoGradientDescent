#!/usr/bin/env python3
"""HOLONOMY -- the obstruction that survives: path-dependence of known operations.

    python3 holonomy.py

THE SETTING
-----------
Light enters a room through several windows and reaches an observer along
different paths, each crossing surfaces that absorb, reflect and shift it. The
operation a surface applies is KNOWN; what is not known is whether the paths
agree. If two paths from a source to a point compose to different results,
there is no consistent colour field on the room, and the failure is measured by
the holonomy around loops.

WHY THIS ESCAPES EVERY OBSTACLE THE OTHER CONSTRUCTIONS HIT
------------------------------------------------------------
  not a marginal problem   Nothing is estimated from a sample, so the
                           single-source theorem does not apply: there is no
                           empirical joint to serve as a witness. The witness
                           would have to be a colour assignment satisfying
                           every edge at once, and it may not exist.

  genuine coefficients     Z/n is an abelian group, so delta exists. Rounding
                           to integers, the probability simplex, and the
                           support indicator all failed exactly here.

  not possibilistic        The class is a sum of known operations around a
                           loop. It is continuous in them and has no support
                           to blur, so it does not die under smoothing the way
                           the AMB obstruction does at eps = 1e-9.

WHAT IS COMPUTED
----------------
Positions in the room are vertices, light paths are edges, and each edge
carries its known operation g_e in Z/n. A colour field is c : V -> Z/n with
c(b) - c(a) = g_{ab} on every edge. It exists iff the holonomy

    h(L) = sum of +/- g_e around L

vanishes for every loop L, so the obstruction is the class [g] in
H^1(graph, Z/n) = (Z/n)^{beta_1}, computed on a cycle basis and checked here
against brute-force construction of the field.
"""
import numpy as np, itertools
N = 4096          # colour band
# positions in the room (vertices) and light paths between them (edges).
# Each edge carries a KNOWN operation: the shift a surface applies.
EDGES = [(0,1),(1,2),(2,3),(3,0),(0,2)]     # a room with two independent loops
V = 4
def cycle_basis(edges, V):
    """spanning tree, then one independent loop per non-tree edge"""
    par=list(range(V)); tree=[]
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    extra=[]
    for e in edges:
        a,b=find(e[0]),find(e[1])
        if a!=b: par[a]=b; tree.append(e)
        else: extra.append(e)
    # path in the tree between the endpoints of each extra edge
    adj={v:[] for v in range(V)}
    for i,(a,b) in enumerate(tree): adj[a].append((b,i,1)); adj[b].append((a,i,-1))
    loops=[]
    for (a,b) in extra:
        prev={a:None}; st=[a]
        while st:
            u=st.pop()
            for (w,i,s) in adj[u]:
                if w not in prev: prev[w]=(u,i,s); st.append(w)
        path=[]; cur=b
        while prev[cur] is not None:
            u,i,s=prev[cur]; path.append((tree[i],-s)); cur=u
        loops.append([( (a,b), 1)] + path)
    return loops, extra
def holonomy(ops, loop):
    return sum(s*ops[e] for e,s in loop) % N
loops, extra = cycle_basis(EDGES, V)
print(f"  room: {V} positions, {len(EDGES)} light paths, "
      f"beta_1 = {len(EDGES)-V+1} independent loops")
print(f"  colour band Z/{N}\n")
print(f"  H^1(room, Z/{N}) = (Z/{N})^beta_1 -- the obstruction has "
      f"{len(loops)} independent components\n")
for name, ops in (("consistent  (a colour field exists)",
                   {(0,1):100,(1,2):250,(2,3):-90,(3,0):-260,(0,2):350}),
                  ("one loop broken",
                   {(0,1):100,(1,2):250,(2,3):-90,(3,0):-260,(0,2):351}),
                  ("both loops broken",
                   {(0,1):100,(1,2):250,(2,3):-88,(3,0):-260,(0,2):351})):
    h=[holonomy(ops,l) for l in loops]
    glob = all(x==0 for x in h)
    # brute-force check: does an assignment c(v) with c(b)-c(a)=op exist?
    ok=False
    for c0 in [0]:
        c={0:c0}; st=[0]; good=True
        seen=set()
        while st:
            u=st.pop()
            for (a,b),g in ops.items():
                if a==u and b not in c: c[b]=(c[u]+g)%N; st.append(b)
                if b==u and a not in c: c[a]=(c[u]-g)%N; st.append(a)
        for (a,b),g in ops.items():
            if (c.get(b,0)-c.get(a,0))%N != g%N: good=False
        ok=good
    print(f"  {name}")
    print(f"    holonomy per independent loop: {h}")
    print(f"    class zero: {glob}     brute-force colour field exists: {ok}")
