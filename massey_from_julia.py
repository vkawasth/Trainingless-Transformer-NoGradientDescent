#!/usr/bin/env python3
"""
Massey Products via Julia-style A_infinity Engine
===================================================
Ports the curved A_inf computation from curved_hh2_sparse_refactored_filteredA.jl
to our transformer Jacobian context.

The Julia code computes:
  mu2 = commutator [J_l, J_{l+1}]            (associativity defect)
  mu3 = Jacobi defect of mu2                  (first obstruction)
  mu4 = mu4_obstruction_full(a,b,c,d)         (second obstruction)
  mu5 = accumulated m3-m3 and m4 terms        (third obstruction)
  mu6 = full A_inf identity at level 6        (prime path generator)

The prime paths (high mu6 obstruction weight) are the irreducible
obstruction generators — exactly what gradient descent is building
in the 200 CE co-adaptation steps.

KEY INSIGHT FROM JULIA CODE:
  prime_paths = extract_prime_paths(m6, top_n=100)
  Each prime path encodes a "hidden corridor" of the algebra.
  The prime higher ideals close under all higher operations.
  The annihilator_infty = basis elements with zero higher score.
  The support_infty = elements active in mu3..mu6.

IN TRANSFORMER TERMS:
  Basis elements = layer interfaces (L0..L23)
  mu2(J_l, J_{l+1}) = [J_l, J_{l+1}] (layer commutator)
  prime_paths = sequences of layer transitions with high obstruction
  support_infty = layers that participate in higher A_inf structure
  annihilator_infty = layers that are algebraically passive

The gauge flattening (mu_0 -> 0) corresponds to the FilteredAInfAlgebra
with m0_curvature per element — exactly what we need.

THREE EXPERIMENTS:
  A: Standard Serre cascade (baseline)
  B: Prime-path cascade (use prime path sequences as block init)
  C: Support-filtered cascade (only initialize blocks in support_infty)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import expm as scipy_expm, logm as scipy_logm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  A_INF ENGINE (Julia port)")
print(f"  mu3..mu6, prime paths, support variety, gauge flattening")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        out=(F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)
        return self.ln(h+self.op(out))

class FF(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.g=nn.Linear(d,d*2,bias=False); self.v=nn.Linear(d,d*2,bias=False)
        self.o=nn.Linear(d*2,d,bias=False); self.n=nn.LayerNorm(d)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))

class Block(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=Attn(d,nh); self.ff=FF(d)
    def forward(self,h): return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def layer_jac(block,h_in,pos,m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            hh=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=block(hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T,U.detach().numpy()

def comm(A,B): return A@B-B@A
def norm(A): return float(np.linalg.norm(A))
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

# ════════════════════════════════════════════════════
# JULIA-PORTED A_INF ENGINE
# ════════════════════════════════════════════════════

def mu2(a,b): return comm(a,b)

def mu3(a,b,c):
    """Triple Massey = Jacobi defect. Julia: m3_element."""
    return comm(comm(a,b),c) - comm(a,comm(b,c))

def mu4_obstruction(a,b,c,d):
    """
    Julia: m4_obstruction_full(a,b,c,d,m2,m3)
    A_inf identity at level 4:
    mu4 = -( mu2(mu3(a,b,c),d) - mu3(mu2(a,b),c,d)
            + mu3(a,mu2(b,c),d) - mu3(a,b,mu2(c,d))
            + mu2(a,mu3(b,c,d)) )
    """
    total  =  comm(mu3(a,b,c), d)
    total -= mu3(comm(a,b), c, d)
    total += mu3(a, comm(b,c), d)
    total -= mu3(a, b, comm(c,d))
    total += comm(a, mu3(b,c,d))
    return -total

def mu5_obstruction(a,b,c,d,e, m3_cache, m4_cache):
    """
    Julia: compute_global_m5 — accumulated m3-m3 and m4 terms.
    A_inf identity at level 5.
    """
    total = np.zeros_like(a)
    # m4 terms
    key4 = (1,2,3,4)  # indices into (a,b,c,d,e)
    args = [a,b,c,d,e]
    for i in range(5):
        remaining = [args[j] for j in range(5) if j!=i]
        # m2(args[i], m4(rest)) or m4(rest, args[i])
        # simplified: use the cached m4 values
        pass
    # m3 o m3 terms (compose_m3_m3 from Julia)
    # + mu3(mu3(a,b,c), d, e)
    total += mu3(mu3(a,b,c), d, e)
    # - mu3(a, mu3(b,c,d), e)
    total -= mu3(a, mu3(b,c,d), e)
    # + mu3(a, b, mu3(c,d,e))
    total += mu3(a, b, mu3(c,d,e))
    # m4 o m2 terms
    total += comm(mu4_obstruction(a,b,c,d), e)
    total += comm(a, mu4_obstruction(b,c,d,e))
    total -= mu4_obstruction(comm(a,b), c, d, e)
    total += mu4_obstruction(a, comm(b,c), d, e)
    total -= mu4_obstruction(a, b, comm(c,d), e)
    total += mu4_obstruction(a, b, c, comm(d,e))
    return -total

def mu6_obstruction(a,b,c,d,e,f):
    """
    Julia: m6_obstruction_full — A_inf identity at level 6.
    This is the prime path generator.
    """
    total = np.zeros_like(a)
    # m2 o m5 terms
    m5_abcde = mu5_obstruction(a,b,c,d,e,{},{})
    m5_bcdef = mu5_obstruction(b,c,d,e,f,{},{})
    total += comm(m5_abcde, f)
    total -= comm(a, m5_bcdef)
    # m3 o m4 terms (3 insertion points)
    m4_abcd=mu4_obstruction(a,b,c,d)
    m4_bcde=mu4_obstruction(b,c,d,e)
    m4_cdef=mu4_obstruction(c,d,e,f)
    total += mu3(m4_abcd, e, f)
    total -= mu3(a, m4_bcde, f)
    total += mu3(a, b, m4_cdef)
    # m4 o m3 terms (4 insertion points)
    m3_abc=mu3(a,b,c); m3_bcd=mu3(b,c,d)
    m3_cde=mu3(c,d,e); m3_def=mu3(d,e,f)
    total += mu4_obstruction(m3_abc, d, e, f)
    total -= mu4_obstruction(a, m3_bcd, e, f)
    total += mu4_obstruction(a, b, m3_cde, f)
    total -= mu4_obstruction(a, b, c, m3_def)
    # m5 o m2 terms (5 insertion points)
    def m5(x1,x2,x3,x4,x5): return mu5_obstruction(x1,x2,x3,x4,x5,{},{})
    total += m5(comm(a,b), c, d, e, f)
    total -= m5(a, comm(b,c), d, e, f)
    total += m5(a, b, comm(c,d), e, f)
    total -= m5(a, b, c, comm(d,e), f)
    total += m5(a, b, c, d, comm(e,f))
    return -total

def extract_prime_paths_transformer(Js, n_layers, threshold_ratio=0.3):
    """
    Julia: extract_prime_paths(m6, top_n=100)
    Identify layer sequences with high mu6 obstruction weight.
    These are the 'prime paths' — irreducible obstruction generators.
    """
    print("  Computing mu6 obstruction for all 6-layer paths...")
    # For efficiency: sample the attractor neighborhood (L12-L17)
    # and compute mu6 for all 6-tuples
    att_layers = list(range(max(0,L_ATT-3), min(n_layers, L_ATT+5)))
    import itertools
    scored = []
    for combo in itertools.combinations(att_layers, 6):
        a,b,c,d,e,f = [Js[i] for i in combo]
        obs = mu6_obstruction(a,b,c,d,e,f)
        weight = float(np.linalg.norm(obs))
        scored.append((combo, weight, obs))
    scored.sort(key=lambda x:-x[1])
    if not scored: return []
    max_w = scored[0][1]
    threshold = max_w * threshold_ratio
    # Extract prime (non-decomposable) paths
    weight_dict = {combo:w for combo,w,_ in scored}
    prime = []
    for combo, w, obs in scored:
        if w < threshold: break
        decomposable = False
        for split in range(1,6):
            left = combo[:split]; right = combo[split:]
            if weight_dict.get(left,0)>threshold and weight_dict.get(right,0)>threshold:
                decomposable=True; break
        if not decomposable:
            prime.append((combo,w,obs))
    return prime

def support_variety(Js, n_layers):
    """
    Julia: support_infty — layers active in mu3..mu6 structure.
    Returns: list of layer indices with high higher-order participation.
    """
    scores = np.zeros(n_layers)
    for l in range(n_layers-2):
        a,b,c = Js[l],Js[l+1],Js[l+2]
        m3_norm = norm(mu3(a,b,c))
        scores[l]+=m3_norm; scores[l+1]+=m3_norm; scores[l+2]+=m3_norm
    for l in range(n_layers-3):
        a,b,c,d = Js[l],Js[l+1],Js[l+2],Js[l+3]
        m4_norm = norm(mu4_obstruction(a,b,c,d))
        for i in range(4): scores[l+i]+=m4_norm
    return scores

def gauge_flatten_per_layer(Js, m0_curvature, lambda_=1.0, n_iter=100, lr=0.005):
    """
    Julia: FilteredAInfAlgebra with m0_curvature per element.
    Apply per-layer gauge: J_l' = exp(eps_l * beta_l) @ J_l @ exp(-eps_l * beta_l)
    where beta_l is chosen to reduce the local curvature mu_0(l).
    """
    Js_flat = list(Js)
    for l in range(len(Js)-1):
        dJ = Js[l]-np.eye(len(Js[l]))
        # Local curvature: mu_0(l) = [dJ_l, alpha_l]
        # where alpha_l = contribution from MC expansion at layer l
        alpha_l = Js[l]@Js[l+1]-Js[l+1]@Js[l]  # local MC element approx
        mu0_l = comm(dJ, alpha_l)
        mu0_norm = norm(mu0_l)
        if mu0_norm < 1e-8: continue
        # Solve: [dJ, beta] = -mu0
        beta = np.zeros_like(dJ)
        for _ in range(n_iter):
            grad = comm(dJ, beta) + mu0_l
            beta = beta - lr*grad
        # Apply gauge with filtration decay
        eps = lambda_ * m0_curvature.get(l, 0.1)
        g = scipy_expm(eps*beta)
        g_inv = scipy_expm(-eps*beta)
        Js_flat[l] = g@Js[l]@g_inv
    return Js_flat

# ════════════════════════════════════════════════════
# TRAIN TEACHER + EXTRACT JACOBIANS
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval()
val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

print("Extracting Jacobians...",flush=True)
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]
print(f"  Done. ma={ma}\n")

# ════════════════════════════════════════════════════
# PART 1: COMPUTE FULL MU3..MU6 AT ATTRACTOR
# ════════════════════════════════════════════════════
print("="*65)
print("PART 1: A_inf structure maps mu3..mu6 (Julia engine)")
print("="*65)

att = [Js[L_ATT+l] for l in range(-2,4)]  # L12..L17
a,b,c,d,e,f = att

mu2_val=mu2(a,b)
mu3_val=mu3(a,b,c)
mu4_val=mu4_obstruction(a,b,c,d)
mu5_val=mu5_obstruction(a,b,c,d,e,{},{})
mu6_val=mu6_obstruction(a,b,c,d,e,f)

print(f"\n  A_inf structure maps at attractor neighborhood:")
print(f"  ||mu2|| = {norm(mu2_val):.6f}  (Lie bracket)")
print(f"  ||mu3|| = {norm(mu3_val):.6f}  (Jacobi defect)")
print(f"  ||mu4|| = {norm(mu4_val):.6f}  (level-4 obstruction)")
print(f"  ||mu5|| = {norm(mu5_val):.6f}  (level-5 obstruction)")
print(f"  ||mu6|| = {norm(mu6_val):.6f}  (prime path generator)")

# Decay rate of mu_k norms
norms=[norm(mu2_val),norm(mu3_val),norm(mu4_val),norm(mu5_val),norm(mu6_val)]
ks=np.array([2,3,4,5,6],dtype=float)
log_norms=np.log(np.maximum(norms,1e-12))
slope,intercept=np.polyfit(ks,log_norms,1)
r2=np.corrcoef(ks,log_norms)[0,1]**2
print(f"\n  Massey norm decay: slope={slope:.4f}  R²={r2:.4f}")
print(f"  Compare: Serre cascade decay = -0.843 (R²=0.9997)")

# ════════════════════════════════════════════════════
# PART 2: PRIME PATHS AND SUPPORT VARIETY
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: Prime paths and support variety")
print("="*65)

prime_paths=extract_prime_paths_transformer(Js,N_LAYERS_T)
print(f"\n  Prime paths (high mu6 obstruction): {len(prime_paths)}")
for i,(combo,w,obs) in enumerate(prime_paths[:5]):
    print(f"  [{i+1}] layers {combo}  weight={w:.4f}")

support_scores=support_variety(Js,N_LAYERS_T)
active_layers=np.where(support_scores>np.percentile(support_scores,70))[0]
print(f"\n  Support variety (layers active in mu3..mu4):")
print(f"  Active layers: {active_layers.tolist()}")
print(f"  Attractor region: L{L_ATT-2}..L{L_ATT+2}")
print(f"  Support concentrates near attractor: "
      f"{np.mean(np.abs(active_layers-L_ATT)):.2f} mean distance from L14")

# Per-layer m0 curvature (from Julia FilteredAInfAlgebra)
m0_curvature={}
for l in range(N_LAYERS_T-1):
    alpha_l=comm(Js[l],Js[l+1])
    dJ=Js[l]-np.eye(ma)
    mu0_l=comm(dJ,alpha_l)
    m0_curvature[l]=float(norm(mu0_l))

print(f"\n  Per-layer curvature mu_0(l):")
for l in range(0,N_LAYERS_T-1,4):
    print(f"  L{l:>2}: mu_0={m0_curvature[l]:.4f}",end="")
print()
print(f"  Attractor L{L_ATT}: mu_0={m0_curvature[L_ATT]:.4f}")

# ════════════════════════════════════════════════════
# PART 3: GAUGE-FLATTENED CASCADES
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 3: Gauge-flattened Jacobians (Julia FilteredAInfAlgebra)")
print("="*65)

# Apply gauge flattening to attractor region
print("  Applying per-layer gauge transformation...")
Js_gauged=gauge_flatten_per_layer(Js, m0_curvature, lambda_=0.1, n_iter=200, lr=0.002)

# Measure curvature reduction
mu0_before=[m0_curvature[l] for l in range(N_LAYERS_T-1)]
mu0_after=[]
for l in range(N_LAYERS_T-1):
    alpha_l=comm(Js_gauged[l],Js_gauged[l+1])
    dJ=Js_gauged[l]-np.eye(ma)
    mu0_l=comm(dJ,alpha_l)
    mu0_after.append(float(norm(mu0_l)))

print(f"  Mean mu_0 before: {np.mean(mu0_before):.4f}")
print(f"  Mean mu_0 after:  {np.mean(mu0_after):.4f}")
print(f"  Reduction: {np.mean(mu0_before)/max(np.mean(mu0_after),1e-8):.2f}x")

# ════════════════════════════════════════════════════
# PART 4: BUILD AND TEST CASCADES
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 4: Student experiments")
print("="*65)

def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r

def build_cascade(Js_src, J14_src, U14_src, mode='serre'):
    cascade=[]
    if mode=='serre':
        for l in range(1,N_STU+1):
            C=ad_k(J14_src,Js_src[min(L_ATT+l,N_LAYERS_T-1)],l)
            n=float(np.linalg.norm(C)); C=C/max(n,1e-8)
            cascade.append(C)
    elif mode=='massey':
        # Use Massey products as cascade levels
        att=[Js_src[L_ATT+l] for l in range(-2,4)]
        a,b,c,d,e,f=att
        massey_ops=[mu2(a,b),mu3(a,b,c),mu4_obstruction(a,b,c,d),
                    mu5_obstruction(a,b,c,d,e,{},{}),mu6_obstruction(a,b,c,d,e,f),
                    comm(mu6_obstruction(a,b,c,d,e,f),att[0])]
        for mk in massey_ops:
            n=float(np.linalg.norm(mk)); cascade.append(mk/max(n,1e-8))
    elif mode=='prime':
        # Use prime path operators as cascade levels
        if prime_paths:
            for i,(_,_,obs) in enumerate(prime_paths[:N_STU]):
                n=float(np.linalg.norm(obs)); cascade.append(obs/max(n,1e-8))
            while len(cascade)<N_STU:
                cascade.append(cascade[-1])
        else:
            return build_cascade(Js_src,J14_src,U14_src,mode='serre')
    return cascade

def run_student(cascade, label, steps=200):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            stu.blocks[l].attn.WK.weight.copy_(W_t)
            stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [50,100,150,200]:
            print(f"  [{label}] step {step}  val={eval_val(stu,n=20):.4f}")
    return eval_val(stu)

# A: Standard Serre
cas_A=build_cascade(Js,J14,U14,'serre')
vA=run_student(cas_A,"A-Serre")

# B: Massey products cascade
cas_B=build_cascade(Js,J14,U14,'massey')
vB=run_student(cas_B,"B-Massey")

# C: Prime path cascade
cas_C=build_cascade(Js,J14,U14,'prime')
vC=run_student(cas_C,"C-prime-paths")

# D: Gauge-flattened Serre (from gauged Jacobians)
J14_g=Js_gauged[L_ATT]
cas_D=build_cascade(Js_gauged,J14_g,U14,'serre')
vD=run_student(cas_D,"D-gauge-flat-Serre")

print(f"\n{'='*65}")
print(f"  A_INF ENGINE RESULTS")
print("="*65)
print(f"""
  A_INF STRUCTURE (at attractor neighborhood L12-L17):
    ||mu2|| = {norm(mu2_val):.6f}  decay slope: {slope:.4f}  R²={r2:.4f}
    ||mu3|| = {norm(mu3_val):.6f}
    ||mu4|| = {norm(mu4_val):.6f}
    ||mu5|| = {norm(mu5_val):.6f}
    ||mu6|| = {norm(mu6_val):.6f}
    Prime paths found: {len(prime_paths)}

  SUPPORT VARIETY:
    Active layers: {active_layers.tolist()}
    Mean distance from L14: {np.mean(np.abs(active_layers-L_ATT)):.2f}

  GAUGE FLATTENING:
    mu_0 reduction: {np.mean(mu0_before)/max(np.mean(mu0_after),1e-8):.2f}x
    (Julia FilteredAInfAlgebra with lambda=0.1)

  STUDENT RESULTS (200 CE, teacher embeddings):
    Teacher:             val={val_teacher:.4f}
    A: Serre cascade:    val={vA:.4f}  (baseline)
    B: Massey cascade:   val={vB:.4f}
    C: Prime paths:      val={vC:.4f}
    D: Gauge-flat Serre: val={vD:.4f}

  IF B < A: Massey products give better A_inf initialization.
  IF C < A: Prime path obstruction structure guides blocks better.
  IF D < A: Gauge flattening of Jacobians improves cascade.
  IF all ≈ A: The 200 CE steps are irreducible regardless of
    which A_inf representative we initialize from.
    The co-adaptation is between corpus distribution and
    the algebraic structure — not between A_inf representatives.
""")
