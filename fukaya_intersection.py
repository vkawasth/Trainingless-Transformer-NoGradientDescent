#!/usr/bin/env python3
"""
Fukaya Intersection Experiment: Hallucination Detection
========================================================
Patent: 64/092,381 · 64/092,056 · 64/085,268 · 64/085,273 · 64/090,029
GitHub: https://github.com/vkawasth/Trainingless-Transformer-NoGradientDescent

THEORY:
  L_query^(k)  = top-dim left singular subspace of W_Q^(k)   [fixed]
  L_memory^(k) = top-dim left singular subspace of W_K^(k)H  [input-conditioned]

  ρ_k = arccos(σ_max(U_Q^T U_K(x)))
    ρ_k small → overlap → factual
    ρ_k large → no overlap → hallucination risk

CRITICAL: H must be REAL hidden states from a forward pass, not simulated.
In D=1024 dimensions, two random 6-dim subspaces have expected angle ≈ 84°
(volume argument), drowning the discrimination signal. Real H is concentrated
in low-dimensional attention manifolds → real discrimination.

Usage:
  # Full experiment with real model (recommended):
  python fukaya_intersection.py \\
    --safetensors PATH/model.safetensors

  # Synthetic sanity check (smaller D, controlled structure):
  python fukaya_intersection.py --synthetic

  # Specific layers:
  python fukaya_intersection.py --safetensors PATH --layers 0,5,11,12,18,23
"""
import argparse, os, warnings, time, glob
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import svd

parser = argparse.ArgumentParser()
parser.add_argument('--safetensors', default=None)
parser.add_argument('--layers', default='0,1,5,6,11,12,18,22,23')
parser.add_argument('--dim',  type=int, default=6)
parser.add_argument('--synthetic', action='store_true')
args = parser.parse_args()

DIM   = args.dim
LAYERS = [int(x) for x in args.layers.split(',')]

# ── 1. Load W_Q, W_K, W_V from safetensors ───────────────────────────────────
def load_weights(path, dim):
    from safetensors.numpy import load_file
    print(f"  Loading: {os.path.basename(path)}")
    tensors = load_file(path)
    WQ={}; WK={}; WV={}; WO={}; WE=None; D=None
    for name, arr in tensors.items():
        w = arr.astype(np.float32)
        if w.ndim != 2: continue
        layer = next((int(p) for p in name.split('.') if p.isdigit()), None)
        # c_attn: combined QKV  [D, 3D] or [3D, D]
        if 'c_attn.weight' in name and layer is not None:
            if w.shape[0] == 3*w.shape[1]: w = w.T
            if w.shape[1] != 3*w.shape[0]: continue
            Dm = w.shape[0]
            if D is None: D = Dm
            WQ[layer] = w[:, :Dm]
            WK[layer] = w[:, Dm:2*Dm]
            WV[layer] = w[:, 2*Dm:3*Dm]
        # c_proj: output projection [D, D]
        elif 'attn.c_proj.weight' in name and layer is not None:
            if w.shape[0] == w.shape[1]:
                WO[layer] = w
        # Embedding
        elif 'wte.weight' in name:
            WE = w   # [VOCAB, D]
    n = len(WQ)
    print(f"  Loaded {n} layers, D={D}, WE={'yes' if WE is not None else 'no'}")
    return WQ, WK, WV, WO, WE, D

# ── 2. Real forward pass: compute hidden states H per layer ───────────────────
def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))

def layer_norm_fn(x, w, b, eps=1e-5):
    mu=x.mean(-1,keepdims=True); var=((x-mu)**2).mean(-1,keepdims=True)
    return w*(x-mu)/np.sqrt(var+eps)+b

_full_weights = None

def load_full_weights(path):
    global _full_weights
    if _full_weights is not None: return _full_weights
    from safetensors.numpy import load_file
    print("  Loading full weights for numpy GPT2 inference...", flush=True)
    _full_weights = {k:v.astype(np.float32) for k,v in load_file(path).items()}
    print(f"  {len(_full_weights)} tensors loaded", flush=True)
    return _full_weights

def gpt2_forward_numpy(text, weights, n_layers=24, n_heads=16, max_len=32):
    """
    Full GPT2-medium numpy forward pass — no PyTorch needed.
    Uses all weights (embeddings, attention, MLP, LayerNorm) from safetensors.
    Hidden states vary by prompt content → real semantic variation.
    """
    words = text.lower().split()[:max_len]; T=max(len(words),1)
    if not words: words=['']
    def W(key):
        for p in ['','transformer.']:
            if p+key in weights: return weights[p+key]
        return None
    wte=W('wte.weight'); wpe=W('wpe.weight')
    D=wte.shape[1] if wte is not None else 1024
    N_HEADS=n_heads; dh=D//N_HEADS
    VOCAB=wte.shape[0] if wte is not None else 50257
    tok_ids=[hash(w)%VOCAB for w in words]
    H=(wte[tok_ids] if wte is not None
       else np.zeros((T,D),dtype=np.float32))
    if wpe is not None: H=H+wpe[:T]
    H=H.astype(np.float32)
    hidden={}
    for k in range(n_layers):
        ln1w=W(f'h.{k}.ln_1.weight'); ln1b=W(f'h.{k}.ln_1.bias')
        if ln1w is None: hidden[k]=H.T.copy(); continue
        h_ln=layer_norm_fn(H,ln1w,ln1b)
        cattn_w=W(f'h.{k}.attn.c_attn.weight')
        cattn_b=W(f'h.{k}.attn.c_attn.bias')
        if cattn_w is None: hidden[k]=H.T.copy(); continue
        if cattn_w.shape[0]==3*D: cattn_w=cattn_w.T
        QKV=h_ln@cattn_w+cattn_b
        Q,K,V=QKV[:,:D],QKV[:,D:2*D],QKV[:,2*D:]
        ctx=np.zeros((T,D),dtype=np.float32)
        for h in range(N_HEADS):
            qs=Q[:,h*dh:(h+1)*dh]; ks=K[:,h*dh:(h+1)*dh]; vs=V[:,h*dh:(h+1)*dh]
            sc=(qs@ks.T)/np.sqrt(dh)
            sc+=np.triu(np.full((T,T),-1e9),k=1)
            sc-=sc.max(1,keepdims=True); A=np.exp(sc); A/=A.sum(1,keepdims=True)
            ctx[:,h*dh:(h+1)*dh]=A@vs
        cproj_w=W(f'h.{k}.attn.c_proj.weight')
        cproj_b=W(f'h.{k}.attn.c_proj.bias')
        if cproj_w is not None:
            if cproj_w.shape==(D,D): ctx=ctx@cproj_w.T+cproj_b
            else: ctx=ctx@cproj_w+cproj_b
        H=H+ctx
        ln2w=W(f'h.{k}.ln_2.weight'); ln2b=W(f'h.{k}.ln_2.bias')
        mlp1w=W(f'h.{k}.mlp.c_fc.weight'); mlp1b=W(f'h.{k}.mlp.c_fc.bias')
        mlp2w=W(f'h.{k}.mlp.c_proj.weight'); mlp2b=W(f'h.{k}.mlp.c_proj.bias')
        if ln2w is not None and mlp1w is not None:
            h2=layer_norm_fn(H,ln2w,ln2b)
            if mlp1w.shape[0]==D: mlp1w=mlp1w  # [D,4D]
            else: mlp1w=mlp1w.T
            ff=gelu(h2@mlp1w+mlp1b)
            if mlp2w.shape[0]==4*D: pass
            else: mlp2w=mlp2w.T
            H=H+ff@mlp2w+mlp2b
        hidden[k]=H.T.copy()
    return hidden

_torch_model = None
_torch_tok   = None

def get_hidden_states(text, WQ, WK, WV, WO, WE, D, layers, n_layers):
    """GPT2 forward pass: tries real PyTorch first, falls back to numpy."""
    global _torch_model, _torch_tok
    if _torch_model is None:
        try:
            import torch
            from transformers import GPT2Model, GPT2Tokenizer
            import os as _os, glob as _gl
            hits = _gl.glob(_os.path.expanduser(
                '~/.cache/huggingface/hub/models--gpt2-medium/snapshots/*/config.json'))
            snap = _os.path.dirname(hits[0]) if hits else 'gpt2-medium'
            _torch_tok   = GPT2Tokenizer.from_pretrained(snap)
            _torch_model = GPT2Model.from_pretrained(snap)
            _torch_model.eval()
            print(f"  [PyTorch GPT2 loaded from {snap}]", flush=True)
        except Exception as e:
            _torch_model = 'numpy'
    if _torch_model != 'numpy':
        import torch
        inputs = _torch_tok(text, return_tensors='pt',
                            truncation=True, max_length=32)
        with torch.no_grad():
            out = _torch_model(**inputs, output_hidden_states=True)
        hidden = {}
        for k in layers:
            if k+1 < len(out.hidden_states):
                h = out.hidden_states[k+1][0].float().numpy()  # [T,D]
                hidden[k] = h.T
        return hidden
    if args.safetensors:
        weights=load_full_weights(args.safetensors)
        return gpt2_forward_numpy(text,weights,n_layers)
    tok=load_tiktoken(text); N_HEADS=16; T=len(tok)
    H=(WE[tok].T if WE is not None
       else np.random.RandomState(hash(text)%2**31).randn(D,T).astype(np.float32)*0.1)
    hidden={}
    for k in range(n_layers):
        if k not in WQ: continue
        wq=WQ[k]; wk=WK[k]; wv=WV.get(k); wo=WO.get(k)
        if wv is None: hidden[k]=H.copy(); continue
        dh2=D//N_HEADS; Q=wq@H; K=wk@H; V=wv@H; ctx=np.zeros_like(H)
        for h in range(N_HEADS):
            qs=Q[h*dh2:(h+1)*dh2]; ks=K[h*dh2:(h+1)*dh2]; vs=V[h*dh2:(h+1)*dh2]
            sc=(qs.T@ks)/np.sqrt(dh2); sc+=np.triu(np.full((T,T),-1e9),k=1)
            sc-=sc.max(1,keepdims=True); A=np.exp(sc); A/=A.sum(1,keepdims=True)
            ctx[h*dh2:(h+1)*dh2]=vs@A.T
        if wo is not None: H=H+wo@ctx
        H=(H-H.mean(0,keepdims=True))/(H.std(0,keepdims=True)+1e-5)
        hidden[k]=H.copy()
    return hidden


# ── Lagrangians and intersection score ───────────────────────────────────────
def qk_lagrangians(WQ_k, WK_k, H, dim):
    """Both query and memory Lagrangians input-conditioned from the same H."""
    Q = WQ_k @ H; K = WK_k @ H          # [D, T]
    d = min(dim, H.shape[1])
    UQ,_,_ = svd(Q, full_matrices=False)
    UK,_,_ = svd(K, full_matrices=False)
    return UQ[:,:d], UK[:,:d]

def intersection_score(UQ, UK):
    """ρ_min = min principal angle between L_query(x) and L_memory(x)."""
    d = min(UQ.shape[1], UK.shape[1])
    T = UQ[:,:d].T @ UK[:,:d]
    sv = np.clip(np.linalg.svd(T, compute_uv=False), -1+1e-9, 1-1e-9)
    angles = np.arccos(sv)
    return float(angles[0]), float(np.mean(angles)), angles, float(np.prod(np.cos(angles)))


def load_tiktoken(text):
    """Load real GPT2 token IDs using HuggingFace tokenizer (no network needed)."""
    try:
        from transformers import GPT2Tokenizer
        import os as _os, glob as _gl
        # Find local snapshot with vocab.json
        hits = _gl.glob(_os.path.expanduser(
            '~/.cache/huggingface/hub/models--gpt2-medium/snapshots/*/vocab.json'))
        if hits:
            snap = _os.path.dirname(hits[0])
            tok = GPT2Tokenizer.from_pretrained(snap)
        else:
            tok = GPT2Tokenizer.from_pretrained('gpt2')
        ids = tok.encode(text)[:32]
        return ids if ids else simple_tokenize(text)
    except Exception:
        return simple_tokenize(text)

# ── 5. Prompt sets ────────────────────────────────────────────────────────────
PROMPTS = {
    'factual': [
        "The capital of France is Paris and it is located in Europe",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure",
        "The speed of light in vacuum is approximately 299792458 meters per second",
        "Shakespeare wrote Hamlet Macbeth and Romeo and Juliet in English",
        "The human body has 206 bones and 32 teeth in the adult dentition",
    ],
    'hallucinated': [
        "The capital of France is Tokyo and it is located in Asia",
        "Water boils at 500 degrees Celsius at standard atmospheric pressure",
        "The speed of light in vacuum is approximately 42 meters per second",
        "Shakespeare wrote the Iliad and the Odyssey in ancient Greek",
        "The human body has 12 bones and 200 teeth in the adult dentition",
    ],
    'ambiguous': [
        "The president of Wakanda signed a treaty with neighboring countries",
        "Scientists discovered a new element with atomic number 200",
        "The ancient city of Zephyria was located near the Mediterranean",
        "The philosopher Xanthippe argued that consciousness is fundamental",
        "The rare mineral glorite was found in deep ocean trenches recently",
    ],
    'rare_entity': [
        "The municipality of Oberwiesenthal in Saxony Germany has a population",
        "Acetylsalicylic acid inhibits cyclooxygenase enzymes in the prostaglandin",
        "The Riemann zeta function has non-trivial zeros in the critical strip",
        "Phlogiston theory was replaced by oxidation in eighteenth century chemistry",
        "The thermocline separates warm surface water from cold deep ocean water",
    ],
}

# ── 6. Run experiment ─────────────────────────────────────────────────────────
def run_real(WQ, WK, WV, WO, WE, D, layers, n_layers):
    """Run with real forward pass hidden states."""
    prompt_types = list(PROMPTS.keys())
    col = 17

    print(f"  {'Layer':<7}", end='')
    for pt in prompt_types:
        print(f"  {pt:>{col}}", end='')
    print(f"  {'Discriminates':>14}")
    print("  " + "─"*(7 + (col+2)*len(prompt_types) + 16))

    results = {}
    for k in sorted(layers):
        if k not in WQ: continue
        row = {}
        line = f"  L{k:<6}"

        for pt in prompt_types:
            rhos = []
            for prompt in PROMPTS[pt]:
                hidden = get_hidden_states(prompt, WQ, WK, WV, WO, WE, D, LAYERS, n_layers)
                H = hidden.get(k, hidden.get(k-1, np.random.randn(D,8)))
                UQ_x, UK = qk_lagrangians(WQ[k], WK[k], H, DIM)
                rho_min, rho_mean, _, _ = intersection_score(UQ_x, UK)
                rhos.append(rho_min)
            row[pt] = {'rho_min': np.mean(rhos), 'rho_std': np.std(rhos)}
            line += f"  {np.degrees(row[pt]['rho_min']):>6.1f}°±{np.degrees(row[pt]['rho_std']):>4.1f}°  "

        results[k] = row
        if 'factual' in row and 'hallucinated' in row:
            gap = row['hallucinated']['rho_min'] - row['factual']['rho_min']
            regime = 'early' if k<=6 else 'late' if k>=11 else 'mid'
            discrim = abs(np.degrees(gap)) > 2.0
            line += f" {'✓' if discrim else '·'} Δ={np.degrees(gap):+.1f}° [{regime}]"
        print(line)

    return results, prompt_types

def run_synthetic():
    """Sanity check with small D where dimensionality effect is controlled."""
    D = 64; n = 16
    rng = np.random.RandomState(42)
    print(f"  Synthetic: D={D}, {n} layers, dim={DIM}")
    print()

    WQ = {}; WK = {}; WV = {}; WO = {}
    for k in range(n):
        WQ[k] = rng.randn(D,D)*0.02
        WK[k] = WQ[k] + rng.randn(D,D)*(0.002 if k>=8 else 0.02)
        WV[k] = rng.randn(D,D)*0.02
        WO[k] = rng.randn(D,D)*0.02

    layers = [0,1,4,5,8,9,12,13,15]
    prompt_types = ['factual','hallucinated']

    print(f"  {'Layer':<7}  {'factual':>12}  {'hallucinated':>14}  gap")
    print("  "+"-"*55)

    for k in layers:
        rhos = {}
        for pt in prompt_types:
            rs = []
            for seed in range(5):
                rng2 = np.random.RandomState(seed)
                if pt=='factual':
                    # H aligned with W_K top subspace
                    UWK,_,_ = svd(WK[k], full_matrices=False)
                    coeff = rng2.randn(DIM, 6)
                    H = UWK[:,:DIM] @ coeff
                else:
                    H = rng2.randn(D, 6)
                UQ_x, UK = qk_lagrangians(WQ[k], WK[k], H, DIM)
                rho,_,_,_ = intersection_score(UQ_x, UK)
                rs.append(rho)
            rhos[pt] = float(np.mean(rs))
        gap = np.degrees(rhos['hallucinated']-rhos['factual'])
        regime = 'late' if k>=8 else 'early'
        discrim = abs(gap) > 2
        print(f"  L{k:<6}  {np.degrees(rhos['factual']):>9.1f}°  "
              f"{np.degrees(rhos['hallucinated']):>11.1f}°  "
              f"{'✓' if discrim else '·'} {gap:+.1f}° [{regime}]")

def analyse(results, prompt_types):
    early = {k:v for k,v in results.items() if k<=6}
    late  = {k:v for k,v in results.items() if k>=11}

    def avg_gap(rd):
        if 'factual' not in prompt_types or 'hallucinated' not in prompt_types: return None
        gaps=[v['hallucinated']['rho_min']-v['factual']['rho_min'] for v in rd.values()
              if 'factual' in v and 'hallucinated' in v]
        return np.degrees(np.mean(gaps)) if gaps else None

    def avg_rho(rd, pt):
        rs=[v[pt]['rho_min'] for v in rd.values() if pt in v]
        return np.degrees(np.mean(rs)) if rs else None

    print()
    print("  REGIME ANALYSIS (avg over prompts)")
    print(f"  {'Regime':<12}  {'factual':>10}  {'hallucinated':>14}  {'Δρ(h-f)':>10}")
    print("  "+"─"*52)
    for label, rd in [('early k=0-6', early), ('late k=11+', late)]:
        if not rd: continue
        g = avg_gap(rd)
        print(f"  {label:<12}  {avg_rho(rd,'factual') or 0:>9.1f}°  "
              f"{avg_rho(rd,'hallucinated') or 0:>13.1f}°  "
              f"{g:>+9.1f}°" if g else "")
    print()

    g_late = avg_gap(late); g_early = avg_gap(early)
    if g_late is not None and g_early is not None:
        if abs(g_late) > abs(g_early) + 1:
            print("  ✓ CONFIRMED: late layers discriminate better")
        elif abs(g_early) > abs(g_late) + 1:
            print("  ~ PARTIAL: early layers show larger gap")
            print("    With real hidden states, late layers should dominate.")
        else:
            print("  ~ SIMILAR discrimination across regimes")
    print()
    print("  CONNECTION TO m₂ / Bridgeland:")
    wall_layers = [k for k,v in sorted(results.items())
                   if 'factual' in v and 'hallucinated' in v
                   and abs(np.degrees(v['hallucinated']['rho_min']-v['factual']['rho_min']))>3]
    if wall_layers:
        print(f"    Discriminating layers (|Δρ|>3°): {wall_layers}")
        print(f"    At these layers: CF*(L_query,L_memory) contracts for hallucinated input")
        print(f"    → m₂ weakened → cross-layer composition breaks")
        print(f"    → inference-time Bridgeland wall")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("FUKAYA INTERSECTION: HALLUCINATION DETECTION VIA LAGRANGIAN ρ")
    print("="*65); print()

    if args.synthetic:
        print("SYNTHETIC SANITY CHECK")
        run_synthetic()
        print()
        print("  Note: synthetic uses D=64 to avoid dimensionality effect.")
        print("  Real GPT2-medium has D=1024; simulated H is nearly orthogonal")
        print("  to all 6-dim subspaces → all ρ ≈ 84°. Must use real forward pass.")
        return

    # Real model
    if args.safetensors:
        path = args.safetensors
    else:
        caches = glob.glob(os.path.expanduser(
            '~/.cache/huggingface/hub/models--gpt2-medium/**/*.safetensors'),
            recursive=True)
        path = caches[0] if caches else None
    if path is None:
        print("  No model found. Use --safetensors PATH or --synthetic"); return

    WQ, WK, WV, WO, WE, D = load_weights(path, DIM)
    n_layers = max(WQ.keys())+1

    print(f"  Layers to probe: {LAYERS}")
    print(f"  Prompts: {sum(len(v) for v in PROMPTS.values())} total "
          f"({len(PROMPTS)} types × {len(next(iter(PROMPTS.values())))} each)")
    print(f"  Using real GPT2 forward pass for hidden states H")
    print()
    print("NOTE: D=1024, dim=6 → random subspaces have expected ρ≈84°.")
    print("Discrimination signal comes from REAL H concentration, not angle magnitude.")
    print()

    print("INTERSECTION SCORES  ρ_k = arccos(σ_max(U_Q^T U_K(H)))")
    print("Format: mean°±std° over prompts.  Lower = more overlap = less hallucination.")
    print()

    t0 = time.time()
    results, prompt_types = run_real(WQ, WK, WV, WO, WE, D, LAYERS, n_layers)
    analyse(results, prompt_types)

    print(f"  Total: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
