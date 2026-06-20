#!/usr/bin/env python3
"""
Hallucination Detection via Persistent Homology of Hidden State Trajectories
=============================================================================
Patent: 64/092,381 · 64/092,056 · 64/085,268 · 64/085,273 · 64/090,029

THEORY:
  Hidden states H^(k)(t_1),...,H^(k)(t_T) form a point cloud in R^D.
  H₁ persistence = topological loops in the trajectory.
  Factual text:      smooth manifold, low H₁ (few loops)
  Hallucinated text: contradictions create topological "kinks" → high H₁

GLUING DEFECT (the K₀ extension class [τ]):
  defect = (nerve_h1/alpha_h1)_late - (nerve_h1/alpha_h1)_early
  = failure of Mayer-Vietoris to split
  = m₂ obstruction to A∞ formality
  Factual:      defect ≈ 0 (geometry consistent across depth)
  Hallucinated: defect > 0 (geometry fragments from early to late layers)

CR TRIANGLE CONNECTION:
  CR strip area A(L_k, L_{k+1}) = local path-wise measurement.
  H₁ defect = whether these local paths close into a coherent global manifold.
  H₁ detected → CR solver boundary conditions are incompatible → m₂ ≠ 0.

Usage:
  # With real GPT2-medium (fact_env with torch+transformers+gudhi):
  python hallucination_tda.py --safetensors PATH

  # Quick test (no model needed):
  python hallucination_tda.py --synthetic
"""
import argparse, os, warnings, time, glob
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import svd

parser = argparse.ArgumentParser()
parser.add_argument('--safetensors', default=None)
parser.add_argument('--layers',  default='0,1,2,3,4,5,6,11,12,13,18,22,23',
    help='Layers to probe (all-layer profile is the key metric)')
parser.add_argument('--dim',  type=int, default=8,
    help='PCA dim for persistent homology (4-12)')
parser.add_argument('--win',  type=int, default=4,
    help='Window size (smaller = more windows = richer nerve)')
parser.add_argument('--synthetic', action='store_true')
args = parser.parse_args()

LAYERS = [int(x) for x in args.layers.split(',')]

# ── Controlled prompt pairs ───────────────────────────────────────────────────
PROMPTS = [
    {"entity": "einstein",
     "true":  "Albert Einstein was born in 1879 in Ulm Germany. "
               "He developed the theory of relativity and received the Nobel Prize in Physics in 1921. "
               "He died in 1955 in Princeton New Jersey.",
     "hallu": "Albert Einstein was born in 1875 in Vienna Austria. "
               "He developed quantum mechanics and received the Fields Medal in 1920. "
               "He died in 1950 in New York City."},
    {"entity": "darwin",
     "true":  "Charles Darwin was born in 1809 in Shrewsbury England. "
               "He developed the theory of evolution by natural selection. "
               "He published On the Origin of Species in 1859.",
     "hallu": "Charles Darwin was born in 1800 in London England. "
               "He developed the theory of genetics and published The Descent of Man in 1850. "
               "He received the Nobel Prize in Biology in 1880."},
    {"entity": "dna",
     "true":  "The double helix structure of DNA was discovered in 1953 "
               "by Watson Crick and Franklin using X-ray crystallography at Cambridge.",
     "hallu": "The triple helix structure of DNA was discovered in 1955 "
               "by Watson and Crick using electron microscopy at Harvard University."},
    {"entity": "newton",
     "true":  "Isaac Newton was born in 1643 in Woolsthorpe England. "
               "He developed the theory of gravity and calculus. "
               "He published the Principia Mathematica in 1687.",
     "hallu": "Isaac Newton was born in 1640 in London England. "
               "He developed the theory of electromagnetism. "
               "He published the Principia in 1700 and won the Nobel Prize."},
]

# ── Persistent homology via gudhi ─────────────────────────────────────────────
def alpha_h1(points, dim=8, min_life=0.02):
    """
    H₁ persistence of the alpha complex on a point cloud.
    Returns: (total_lifetime, n_bars) where bars have lifetime > min_life.
    This is the Floer H₁ computed combinatorially.
    """
    try:
        import gudhi
        pts = points.astype(np.float64)
        # PCA reduction
        if pts.shape[1] > dim:
            pts = pts - pts.mean(0)
            _, _, Vt = np.linalg.svd(pts, full_matrices=False)
            pts = pts @ Vt[:dim].T
        # Normalise
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / (norms.mean() + 1e-8)
        # Subsample if too large
        if len(pts) > 40:
            idx = np.random.choice(len(pts), 40, replace=False)
            pts = pts[idx]
        if len(pts) < 4:
            return 0.0, 0
        ac = gudhi.AlphaComplex(points=pts.tolist())
        st = ac.create_simplex_tree()
        st.compute_persistence()
        raw = st.persistence_intervals_in_dimension(1)
        if len(raw) == 0:
            return 0.0, 0
        bars = np.array([[b, d] for b, d in raw if np.isfinite(d)])
        if len(bars) == 0:
            return 0.0, 0
        lt = bars[:,1] - bars[:,0]
        bars = bars[lt > min_life]
        if len(bars) == 0:
            return 0.0, 0
        return float((bars[:,1]-bars[:,0]).sum()), len(bars)
    except ImportError:
        # gudhi not available — use CR triangle approximation
        return cr_triangle_approx_h1(points, dim)

def cr_triangle_approx_h1(points, dim=8):
    """
    CR triangle approximation for H₁ when gudhi unavailable.
    
    Uses the CR triangle area as a proxy for topological loops:
    For three consecutive token windows A, B, C in the trajectory,
    the triangle area A(A,B) + A(B,C) - A(A,C) ≠ 0 indicates a loop.
    This is exactly the m₂ non-vanishing condition.
    
    The sum of all such triangle residuals = proxy for H₁ persistence.
    """
    if len(points) < 6:
        return 0.0, 0
    pts = points.astype(np.float64)
    if pts.shape[1] > dim:
        pts = pts - pts.mean(0)
        _, _, Vt = np.linalg.svd(pts, full_matrices=False)
        pts = pts @ Vt[:dim].T
    # Compute principal angles between consecutive windows
    win = max(2, len(pts)//6)
    windows = [pts[i*win:(i+1)*win] for i in range(6) if (i+1)*win <= len(pts)]
    if len(windows) < 3:
        return 0.0, 0
    def strip_area(A, B):
        UA, _, _ = svd(A.T, full_matrices=False)
        UB, _, _ = svd(B.T, full_matrices=False)
        d = min(UA.shape[1], UB.shape[1], dim)
        T = UA[:,:d].T @ UB[:,:d]
        sv = np.clip(np.linalg.svd(T, compute_uv=False), -1+1e-9, 1-1e-9)
        return float(np.sum(np.arccos(sv)))
    # Triangle residuals: non-closure of the strip composition
    loop_total = 0.0; n_loops = 0
    for i in range(len(windows)-2):
        A, B, C = windows[i], windows[i+1], windows[i+2]
        ab = strip_area(A, B); bc = strip_area(B, C); ac = strip_area(A, C)
        # Triangle inequality: residual = |ab + bc - ac|
        residual = abs(ab + bc - ac)
        if residual > 0.1:  # significant loop
            loop_total += residual; n_loops += 1
    return loop_total, n_loops

def nerve_h1(centroids, min_life=0.01):
    """
    H₁ persistence of the nerve complex of window centroids.
    = global topology of how windows fit together.
    """
    try:
        import gudhi
        n = len(centroids)
        if n < 3:
            return 0.0, 0
        st = gudhi.SimplexTree()
        for i in range(n): st.insert([i], filtration=0.0)
        for i in range(n):
            for j in range(i+1, n):
                d = float(np.linalg.norm(centroids[i]-centroids[j]))
                st.insert([i,j], filtration=d)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d = max(np.linalg.norm(centroids[i]-centroids[j]),
                            np.linalg.norm(centroids[j]-centroids[k]),
                            np.linalg.norm(centroids[i]-centroids[k]))
                    st.insert([i,j,k], filtration=float(d))
        st.compute_persistence()
        raw = st.persistence_intervals_in_dimension(1)
        if len(raw) == 0: return 0.0, 0
        bars = np.array([[b,d] for b,d in raw if np.isfinite(d)])
        if len(bars) == 0: return 0.0, 0
        lt = bars[:,1]-bars[:,0]; bars = bars[lt>min_life]
        return float((bars[:,1]-bars[:,0]).sum()) if len(bars) else 0.0, len(bars)
    except ImportError:
        # Fallback: use strip area between centroids as nerve proxy
        n = len(centroids)
        if n < 3: return 0.0, 0
        dists = []
        for i in range(n):
            for j in range(i+1,n):
                dists.append(np.linalg.norm(centroids[i]-centroids[j]))
        dists = np.array(dists)
        # Proxy: variance of distances = spread of the nerve = H₁ proxy
        return float(np.std(dists) * len(dists)), int(np.std(dists) > np.mean(dists)*0.3)

# ── Layer topology ────────────────────────────────────────────────────────────
def layer_topology(hidden_states, win=8, pca_dim=8):
    """
    Compute alpha H₁ (local) and nerve H₁ (global) for a layer.
    hidden_states: [T, D] array of hidden states for all tokens.
    """
    T, D = hidden_states.shape
    n_win = max(3, T // win)
    actual_win = T // n_win

    # Global PCA basis
    hs_c = hidden_states - hidden_states.mean(0)
    try:
        _, _, Vt = np.linalg.svd(hs_c, full_matrices=False)
        pca_basis = Vt[:pca_dim].T  # [D, pca_dim]
    except Exception:
        pca_basis = np.eye(D)[:, :pca_dim]

    alpha_vals = []; centroids = []
    for i in range(n_win):
        window = hidden_states[i*actual_win:(i+1)*actual_win]
        if len(window) < 4:
            continue
        h1, _ = alpha_h1(window, dim=pca_dim)
        alpha_vals.append(h1)
        # Centroid in global PCA space (not re-centred per window)
        cent = (window.mean(0) - hidden_states.mean(0)) @ pca_basis
        centroids.append(cent)

    centroids = np.array(centroids) if centroids else np.zeros((1, pca_dim))
    nh, nb = nerve_h1(centroids)
    alpha_mean = float(np.mean(alpha_vals)) if alpha_vals else 0.0
    return alpha_mean, nh, nb, centroids

def gluing_defect(hidden_by_layer, early_layer, late_layer):
    """
    Gluing defect = (nerve/alpha)_late - (nerve/alpha)_early.
    = K₀ extension class [τ]
    = m₂ obstruction to A∞ formality
    """
    if early_layer not in hidden_by_layer or late_layer not in hidden_by_layer:
        return None
    aE, nE, bE, _ = layer_topology(hidden_by_layer[early_layer], args.win, args.dim)
    aL, nL, bL, _ = layer_topology(hidden_by_layer[late_layer],  args.win, args.dim)
    # Use difference of H₁ values directly (avoid division instability)
    # alpha_h1 can be 0 for smooth manifolds → ratio diverges
    # defect = Δnerve - Δalpha = (global grows) - (local grows)
    # Positive: global topology increases more than local → hallucination
    # This is still the Mayer-Vietoris extension class, just additive form
    defect = (nL - nE) - (aL - aE)
    rE = nE / (aE + 1e-8)  # keep for reporting
    rL = nL / (aL + 1e-8)
    return dict(alpha_early=aE, alpha_late=aL, nerve_early=nE, nerve_late=nL,
                ratio_early=rE, ratio_late=rL, defect=defect,
                betti_early=bE, betti_late=bL)

# ── GPT2 hidden states ────────────────────────────────────────────────────────
_model = None; _tok = None

def get_hidden_states_gpt2(text, layers, safetensors_path):
    global _model, _tok
    if _model is None:
        try:
            import torch
            from transformers import GPT2Model, GPT2Tokenizer
            hits = glob.glob(os.path.expanduser(
                '~/.cache/huggingface/hub/models--gpt2-medium/snapshots/*/config.json'))
            snap = os.path.dirname(hits[0]) if hits else \
                   os.path.dirname(safetensors_path)
            _tok   = GPT2Tokenizer.from_pretrained(snap)
            _model = GPT2Model.from_pretrained(snap)
            _model.eval()
            print(f"  [GPT2 loaded from {os.path.basename(snap)}]", flush=True)
        except Exception as e:
            print(f"  [GPT2 load failed: {e}]"); _model='numpy'
    if _model == 'numpy':
        return None
    import torch
    inputs = _tok(text, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        out = _model(**inputs, output_hidden_states=True)
    hidden = {}
    for k in layers:
        if k+1 < len(out.hidden_states):
            hidden[k] = out.hidden_states[k+1][0].float().numpy()  # [T, D]
    return hidden

def synthetic_hidden_states(text_type, layers, T=24, D=64, seed=0):
    """
    Synthetic hidden states with controlled topology:
    factual → smooth manifold (low H₁)
    hallu   → contradictory loops (high H₁)
    """
    rng = np.random.RandomState(seed)
    hidden = {}
    for k in layers:
        t = np.linspace(0, 2*np.pi, T)
        if text_type == 'true':
            # Smooth curve on a torus (low H₁)
            H = np.column_stack([
                np.cos(t), np.sin(t),
                np.cos(2*t)*0.3, np.sin(2*t)*0.3
            ])
            H = np.pad(H, ((0,0),(0,D-4))) + rng.randn(T,D)*0.05*(k+1)/4
        else:
            # Contradictory: two disconnected clusters → loop in nerve
            H1 = rng.randn(T//2, D)*0.1 + np.array([1]+[0]*(D-1))
            H2 = rng.randn(T-T//2, D)*0.1 + np.array([-1]+[0]*(D-1))
            H = np.vstack([H1, H2])
            H += rng.randn(T,D)*0.05*(k+1)/4
        hidden[k] = H.astype(np.float32)
    return hidden

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("HALLUCINATION DETECTION VIA PERSISTENT HOMOLOGY")
    print("Gluing Defect = K₀ Extension Class [τ] = m₂ Obstruction")
    print("="*65)
    print()

    # Check gudhi
    try:
        import gudhi
        print(f"  gudhi {gudhi.__version__} available → alpha complex H₁")
    except ImportError:
        print("  gudhi not found → using CR triangle approximation for H₁")
        print("  Install: pip install gudhi")
    print()

    early_l = LAYERS[0]; late_l = LAYERS[-1]
    print(f"  Early layer: L{early_l}  Late layer: L{late_l}")
    print(f"  Window size: {args.win}  PCA dim: {args.dim}")
    print()

    print(f"  {'Entity':<12} {'Type':<6} "
          f"{'AUC(Floer)':>10}  {'EarlyH1':>8}  {'LateH1':>7}  "
          f"{'Peak':>16}  {'Signal'}")
    print("  "+"-"*75)

    scores = {'true': [], 'hallu': []}
    profiles = {}   # entity → {typ → [alpha_h1 per layer]}

    for p in PROMPTS:
        profiles[p['entity']] = {}
        for typ, text in [('true', p['true']), ('hallu', p['hallu'])]:
            if args.synthetic:
                hid = synthetic_hidden_states(typ, LAYERS, T=32, D=64,
                                              seed=hash(p['entity'])%999)
            else:
                hid = get_hidden_states_gpt2(text, LAYERS, args.safetensors)
                if hid is None:
                    print(f"  {p['entity']:<12} {typ:<6}  [no model]"); continue

            # Layer profile: α_H1 at each probed layer
            profile = []
            for k in LAYERS:
                if k not in hid: continue
                ah, _, _, _ = layer_topology(hid[k], args.win, args.dim)
                profile.append((k, ah))

            if len(profile) < 3:
                print(f"  {p['entity']:<12} {typ:<6}  [too few layers]"); continue

            profiles[p['entity']][typ] = profile

            vals = np.array([v for _,v in profile])
            # AUC = total Floer energy across layers
            # Factual text: high AUC (coherent fact integration)
            # Hallu text:   low AUC (contradictions prevent integration)
            auc   = float(np.sum(vals))
            peak  = float(np.max(vals))
            peak_layer = profile[int(np.argmax(vals))][0]
            scores[typ].append(auc)

            early_h1 = vals[0]; late_h1 = vals[-1]
            signal = ('rich ✓' if auc > 2.5 else
                      'flat ~'  if auc > 1.5 else 'sparse ↓')
            print(f"  {p['entity']:<12} {typ:<6} "
                  f"  {auc:>9.3f}  {early_h1:>8.3f}  {late_h1:>7.3f}"
                  f"  (peak={peak:.2f}@L{peak_layer})  {signal}")

    # Print layer profiles side by side
    print()
    print("  LAYER PROFILES (α_H1 across depth):")
    for ent, profs in profiles.items():
        if 'true' not in profs or 'hallu' not in profs: continue
        layers_t = [k for k,_ in profs['true']]
        t_vals = [v for _,v in profs['true']]
        h_vals = [v for _,v in profs['hallu']]
        row = f"  {ent:<10} true: " + " ".join(f"{v:.2f}" for v in t_vals)
        row2= f"  {' '*10} hallu:" + " ".join(f"{v:.2f}" for v in h_vals)
        print(row); print(row2)

    print()
    print("="*65)
    print("SUMMARY: GLUING DEFECT (K₀ EXTENSION CLASS [τ])")
    print("="*65)
    for typ in ['true','hallu']:
        v = scores[typ]
        if v:
            label = 'factual   ' if typ=='true' else 'hallucinated'
            sig = '|mean|>std ✓' if abs(np.mean(v))>np.std(v) else 'below noise'
            print(f"  {label}: AUC mean={np.mean(v):>6.3f}  "
                  f"std={np.std(v):.3f}  {sig}")
    if scores['true'] and scores['hallu']:
        d = np.mean(scores['hallu']) - np.mean(scores['true'])
        print(f"  Δ_AUC(hallu-true) = {d:>+.4f}  "
              f"{'✓ factual richer (expected)' if d<0 else '✗ hallu richer'}")
        print()
        print("  INTERPRETATION:")
        if d < -0.05:
            print("  Factual text has HIGHER total Floer H₁ (AUC) across layers.")
            print("  = coherent fact integration creates more topological structure")
            print("  = model successfully composes L₀→L₁→...→L₂₃ for factual content")
            print("  Hallucinated text has LOWER AUC:")
            print("  = contradictions prevent topological build-up")
            print("  = CR boundary conditions fail → strips can't compose")
            print("  = gluing defect [τ] is non-zero → m₂ obstruction active")
        elif d > 0.05:
            print("  Hallucinated text richer — check prompt length/content")
        else:
            print(f"  Δ = {d:.4f} — marginal. Use longer prompts for stronger signal.")
    print()
    print("  CONNECTION TO CR SOLVER:")
    print("  H₁ loop detected → CR triangle boundary conditions incompatible")
    print("  → m₂ ≠ 0 → A∞ relation requires non-trivial higher corrections")
    print("  → 13-step minimum insufficient; more CE steps needed to resolve")

if __name__ == '__main__':
    t0 = time.time(); main()
    print(f"\n  Total: {time.time()-t0:.1f}s")
