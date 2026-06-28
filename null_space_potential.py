"""
null_space_potential.py
========================
Evaluates V(u) = L(w₀ + P·u) on a 2D grid in the null eigenspace
to classify the local catastrophe type.

The null eigenspace E₀ = ker(H) at step 64 has dimension 5.
We restrict to the 2D subspace spanned by the two most-null directions:
  v₀ (λ=-0.22): most-null, "least curved" direction
  v₁ (λ=+4.53): second most-null

V(x,y) = L(w₀ + x·v₀ + y·v₁) on a grid [-r,r]² 

Catastrophe classification from V level sets:
  Fold:   V ~ x³ + μx       → asymmetric level sets, one cusp
  Cusp:   V ~ x⁴ + ax² + bx → symmetric, two cusps
  Higher: V ~ x⁵ + ...      → more structure

Also computes the Taylor coefficients of V at the origin to
directly compare with Thom's normal forms.

Usage
-----
  python null_space_potential.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --compiler compiler_geometric.py \\
      --radius 0.5 --grid_n 15 --n_eval 16 \\
      --output null_space_report.json
"""

import argparse, json, math, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64',  default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--compiler', default='compiler_geometric.py')
    p.add_argument('--radius',   type=float, default=0.3,
                   help='Grid half-width in null-space coordinates')
    p.add_argument('--grid_n',   type=int,   default=11,
                   help='Grid points per axis (total = grid_n²)')
    p.add_argument('--rank',     type=int,   default=6)
    p.add_argument('--k_lanczos',type=int,   default=20)
    p.add_argument('--n_eval',   type=int,   default=16,
                   help='Batches per loss evaluation')
    p.add_argument('--output',   default='null_space_report.json')
    return p.parse_args()


def import_compiler(path):
    globs = {'__name__':'__compiler__','__file__':path,'__builtins__':__builtins__}
    src = open(path).read()
    src = src.replace('if __name__ == "__main__":','if False:')
    src = src.replace("if __name__ == '__main__':","if False:")
    import re
    src = re.sub(r'torch\.save\(','_NOSAVE(',src)
    globs['_NOSAVE'] = lambda *a,**kw: None
    exec(compile(src,path,'exec'),globs)
    return types.SimpleNamespace(**{k:v for k,v in globs.items() if not k.startswith('__')})


def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        state = ckpt
    return {k: v.clone() for k, v in state.items() if isinstance(v, torch.Tensor)}


def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor
    return [wk[i].detach().float() for i in sorted(wk)]


class SymplecticProxy(nn.Module):
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank = rank
        self.theta = nn.Parameter(torch.cat([w.reshape(-1) for w in wk_list]).clone())
        self.shapes = [w.shape for w in wk_list]
        self.splits = [w.numel() for w in wk_list]
    def wk_matrices(self):
        return [p.reshape(s) for p,s in zip(torch.split(self.theta,self.splits),self.shapes)]
    def forward(self):
        wks = self.wk_matrices()
        loss = torch.tensor(0.)
        for k in range(len(wks)-1):
            Ua = torch.linalg.svd(wks[k], full_matrices=False)[0][:,:self.rank]
            Ub = torch.linalg.svd(wks[k+1], full_matrices=False)[0][:,:self.rank]
            sv = torch.linalg.svdvals(Ua.T@Ub).clamp(1e-6,1-1e-6)
            loss = loss + torch.arccos(sv).sum()
        return loss

def hvp(model, v):
    loss = model()
    grad = torch.autograd.grad(loss, model.theta, create_graph=True)[0]
    return torch.autograd.grad((grad*v).sum(), model.theta)[0].detach()

def lanczos_null(model, k, n_null, seed=42):
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k+1)
    alpha_c = torch.zeros(k); beta_c = torch.zeros(k-1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:,0]=v; prev_b=0.; kk=k
    for j in range(kk):
        w = hvp(model, V[:,j])
        if j>0: w = w - prev_b*V[:,j-1]
        a = (w*V[:,j]).sum().item(); alpha_c[j]=a
        w = w - a*V[:,j]
        for i in range(j+1): w = w-(w@V[:,i])*V[:,i]
        if j<kk-1:
            b=w.norm().item()
            if b<1e-10: kk=j+1; break
            prev_b=b; beta_c[j]=b; V[:,j+1]=w/b
    Hk = (np.diag(alpha_c[:kk].numpy())+np.diag(beta_c[:kk-1].numpy(),1)+
          np.diag(beta_c[:kk-1].numpy(),-1))
    ev,ritz = np.linalg.eigh(Hk)
    evecs = V[:,:kk].numpy()@ritz
    idx = np.argsort(np.abs(ev))
    return ev[idx[:n_null]], evecs[:,idx[:n_null]]


@torch.no_grad()
def eval_loss(model, get_batch, n=16):
    model.eval()
    return float(np.mean([model(*(get_batch()))[1].item() for _ in range(n)]))


def eval_V(state, LM, get_batch, v0_flat, v1_flat, x, y, n_eval):
    """V(x,y) = L(w₀ + x·v₀ + y·v₁)"""
    state_new = {}
    # Build parameter offset: need to know which keys are WK and their positions
    wk_keys = []
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            wk_keys.append(name)
    wk_keys = sorted(wk_keys, key=lambda k: int([p for p in k.split('.') if p.isdigit()][0]))

    # Compute offsets in the concatenated parameter vector
    splits = [state[k].numel() for k in wk_keys]
    offsets = list(np.cumsum([0]+splits))

    for name, tensor in state.items():
        state_new[name] = tensor.clone()

    for i, name in enumerate(wk_keys):
        s, e = offsets[i], offsets[i+1]
        dv0 = torch.tensor(v0_flat[s:e], dtype=torch.float32).reshape(state[name].shape)
        dv1 = torch.tensor(v1_flat[s:e], dtype=torch.float32).reshape(state[name].shape)
        state_new[name] = state[name] + x * dv0 + y * dv1

    model = LM()
    model.load_state_dict(state_new, strict=False)
    return eval_loss(model, get_batch, n_eval)


def taylor_coefficients(V_grid, xs, ys):
    """
    Fit V(x,y) = c00 + c10*x + c01*y + c20*x² + c11*xy + c02*y²
               + c30*x³ + c21*x²y + c12*xy² + c03*y³
               + c40*x⁴ + ...
    Returns coefficient dict.
    """
    n = len(xs)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    x_flat = X.ravel()
    y_flat = Y.ravel()
    V_flat = V_grid.ravel()

    # Build polynomial feature matrix up to degree 4
    features = {}
    for i in range(5):
        for j in range(5-i):
            features[f'x{i}y{j}'] = (x_flat**i) * (y_flat**j)

    names = list(features.keys())
    A = np.column_stack([features[n] for n in names])
    coeffs, _, _, _ = np.linalg.lstsq(A, V_flat, rcond=None)

    return {names[i]: float(coeffs[i]) for i in range(len(names))}


def classify_catastrophe(coeffs, V0):
    """
    Compare Taylor coefficients with Thom normal forms.
    V(x,y) relative to V(0,0).

    Relevant coefficients (normalized):
      c10, c01: linear (should be ≈0 at critical point)
      c20, c11, c02: quadratic (Hessian)
      c30, c21, c12, c03: cubic
      c40, c31, c22, c13, c04: quartic
    """
    c10 = coeffs.get('x1y0', 0)
    c01 = coeffs.get('x0y1', 0)
    c20 = coeffs.get('x2y0', 0)
    c02 = coeffs.get('x0y2', 0)
    c11 = coeffs.get('x1y1', 0)
    c30 = coeffs.get('x3y0', 0)
    c03 = coeffs.get('x0y3', 0)
    c40 = coeffs.get('x4y0', 0)
    c04 = coeffs.get('x0y4', 0)

    grad_norm = math.sqrt(c10**2 + c01**2)
    hess_trace = c20 + c02
    hess_det   = c20*c02 - c11**2
    cubic_norm = math.sqrt(c30**2 + c03**2)
    quartic_norm = math.sqrt(c40**2 + c04**2)

    print(f"\n  Taylor coefficients of V(x,y) - V(0,0):")
    print(f"    Linear:   c10={c10:+.4f}  c01={c01:+.4f}  |grad|={grad_norm:.4f}")
    print(f"    Quadratic: c20={c20:+.4f}  c11={c11:+.4f}  c02={c02:+.4f}")
    print(f"    det(Hess)={hess_det:+.4f}  tr(Hess)={hess_trace:+.4f}")
    print(f"    Cubic:    c30={c30:+.4f}  c03={c03:+.4f}  |cubic|={cubic_norm:.4f}")
    print(f"    Quartic:  c40={c40:+.4f}  c04={c04:+.4f}  |quartic|={quartic_norm:.4f}")

    # Classification
    if grad_norm > 0.1:
        cls = "NOT_AT_CRITICAL_POINT"
        msg = f"Gradient ≠ 0 (|∇V|={grad_norm:.3f}). Not at a critical point."
    elif hess_det > 0.01:
        cls = "NON_DEGENERATE_MINIMUM"
        msg = f"det(H)={hess_det:.3f}>0, tr(H)={hess_trace:.3f}. Non-degenerate min/max."
    elif abs(hess_det) < 0.01 and cubic_norm > 0.01:
        cls = "FOLD_A2"
        msg = (f"Degenerate Hessian (det≈0), cubic ≠ 0 (|c3|={cubic_norm:.3f}). "
               "Consistent with fold catastrophe (A₂: x³ + μx).")
    elif abs(hess_det) < 0.01 and cubic_norm < 0.01 and quartic_norm > 0.01:
        cls = "CUSP_A3"
        msg = (f"Degenerate Hessian, cubic≈0, quartic ≠ 0 (|c4|={quartic_norm:.3f}). "
               "Consistent with cusp catastrophe (A₃: x⁴ + ax² + bx).")
    else:
        cls = "HIGHER_ORDER_OR_UNCLEAR"
        msg = "Higher-order or mixed structure. Need more directions."

    print(f"\n  Classification: {cls}")
    print(f"  {msg}")
    return cls, msg


def main():
    args = parse_args()
    print("="*60)
    print("  NULL-SPACE POTENTIAL V(u) = L(w₀ + Pu)")
    print("  Catastrophe classification via Taylor expansion")
    print("="*60)

    comp = import_compiler(args.compiler)
    LM, get_batch = comp.LM, comp.get_batch

    state = load_state(args.spike64)
    wk_list = extract_wk(state)

    print(f"\n  Computing null eigenvectors (Lanczos) ...")
    model = SymplecticProxy(wk_list, args.rank)
    evals, evecs = lanczos_null(model, args.k_lanczos, 5)
    print(f"  Null eigenvalues: " + ", ".join(f"{e:.4f}" for e in evals))
    print(f"  Using v₀ (λ={evals[0]:.4f}) and v₁ (λ={evals[1]:.4f})")

    v0 = evecs[:, 0]   # most null
    v1 = evecs[:, 1]   # second most null

    # Baseline loss
    model0 = LM()
    model0.load_state_dict(state, strict=False)
    V0 = eval_loss(model0, get_batch, args.n_eval)
    print(f"  Baseline V(0,0) = L(w₀) = {V0:.4f}")

    # 2D grid
    xs = np.linspace(-args.radius, args.radius, args.grid_n)
    ys = np.linspace(-args.radius, args.radius, args.grid_n)
    N  = args.grid_n
    total = N * N

    print(f"\n  Evaluating V on {N}×{N} grid, radius={args.radius} ...")
    print(f"  Total evaluations: {total}  (n_eval={args.n_eval} per point)")

    V_grid = np.zeros((N, N))
    t0 = time.time()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            V_grid[i, j] = eval_V(state, LM, get_batch, v0, v1, x, y, args.n_eval)
        pct = (i+1)/N*100
        elapsed = time.time()-t0
        eta = elapsed/(i+1)*(N-i-1)
        print(f"  Row {i+1}/{N}  ({pct:.0f}%)  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    V_rel = V_grid - V0

    print(f"\n  V(x,y) range: [{V_rel.min():.4f}, {V_rel.max():.4f}]")
    print(f"  V(0,0) = 0 by construction")

    # Taylor coefficients
    coeffs = taylor_coefficients(V_rel, xs, ys)
    cls, msg = classify_catastrophe(coeffs, V0)

    # ASCII contour plot
    print(f"\n  V(x,y) - V(0,0) level structure:")
    print(f"  (+ = V above baseline, - = V below baseline, 0 = near baseline)")
    print(f"  x axis: v₀ ∈ [{-args.radius:.2f}, {args.radius:.2f}]")
    print(f"  y axis: v₁ ∈ [{-args.radius:.2f}, {args.radius:.2f}]")
    thresh = max(abs(V_rel.max()), abs(V_rel.min())) * 0.2
    for j in range(N-1, -1, -1):
        row = ""
        for i in range(N):
            v = V_rel[i, j]
            if abs(v) < thresh: row += "·"
            elif v > 0:         row += "+"
            else:               row += "-"
        print(f"  {ys[j]:+.2f} |{row}|")
    print(f"        " + "".join(f"{x:+.1f}"[0] for x in xs[::max(1,N//8)]))

    report = {
        'baseline_V0': float(V0),
        'grid_xs': xs.tolist(),
        'grid_ys': ys.tolist(),
        'V_grid': V_grid.tolist(),
        'V_rel': V_rel.tolist(),
        'null_evals': evals.tolist(),
        'taylor_coeffs': coeffs,
        'classification': cls,
        'classification_msg': msg,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
