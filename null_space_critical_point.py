"""
null_space_critical_point.py
=============================
Finds the critical point of V(u) = L(w₀ + Pu) within the
5-dimensional null-space subspace, then computes the Taylor
expansion at that point for catastrophe classification.

Workflow (per the priority-2 recommendation):
  1. Solve min_{u ∈ R^5} V(u) via L-BFGS-B
  2. Verify ‖∇V(u*)‖ < tolerance
  3. Recompute Hessian, cubic tensor at u*
  4. Classify: fold (A₂), cusp (A₃), or higher

Usage
-----
  python null_space_critical_point.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --compiler compiler_geometric.py \\
      --rank 6 --k_lanczos 20 --n_eval 32 \\
      --search_radius 1.0 --grid_init 5 \\
      --output null_space_critical_report.json
"""

import argparse, json, math, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64',      default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--compiler',     default='compiler_geometric.py')
    p.add_argument('--rank',         type=int,   default=6)
    p.add_argument('--k_lanczos',    type=int,   default=20)
    p.add_argument('--n_eval',       type=int,   default=32)
    p.add_argument('--search_radius',type=float, default=0.5)
    p.add_argument('--grid_init',    type=int,   default=5,
                   help='Grid initializations per dimension')
    p.add_argument('--fd_eps',       type=float, default=0.05,
                   help='FD step for gradient/Hessian in u-space')
    p.add_argument('--output',       default='null_space_critical_report.json')
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
            wk[li] = (name, tensor)
    return [(wk[i][0], wk[i][1]) for i in sorted(wk)]


class SymplecticProxy(nn.Module):
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank = rank
        self.theta = nn.Parameter(
            torch.cat([w.reshape(-1) for w in wk_list]).clone())
        self.shapes = [w.shape for w in wk_list]
        self.splits = [w.numel() for w in wk_list]
    def wk_matrices(self):
        return [p.reshape(s) for p,s in
                zip(torch.split(self.theta,self.splits),self.shapes)]
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
    Hk = (np.diag(alpha_c[:kk].numpy())+
          np.diag(beta_c[:kk-1].numpy(),1)+
          np.diag(beta_c[:kk-1].numpy(),-1))
    ev, ritz = np.linalg.eigh(Hk)
    evecs = V[:,:kk].numpy() @ ritz
    idx = np.argsort(np.abs(ev))
    return ev[idx[:n_null]], evecs[:,idx[:n_null]]


@torch.no_grad()
def eval_loss(model, get_batch, n=32):
    model.eval()
    return float(np.mean([model(*(get_batch()))[1].item() for _ in range(n)]))


def eval_V(u_vec, state, wk_names, P_cols, LM, get_batch, n_eval):
    """
    V(u) = L(w₀ + P·u) where P = [v₀,...,v_{k-1}] is the basis matrix.
    u_vec: (k,) coordinates in null space.
    """
    state_new = {n: t.clone() for n, t in state.items()}
    # P_cols: list of (name, flat_delta) for each basis vector
    for i, u_i in enumerate(u_vec):
        for (name, start, end, delta_flat) in P_cols[i]:
            state_new[name] = state_new[name] + u_i * delta_flat.reshape(state_new[name].shape)
    model = LM()
    model.load_state_dict(state_new, strict=False)
    return eval_loss(model, get_batch, n_eval)


def gradient_V(u_vec, state, wk_names, P_cols, LM, get_batch, n_eval, fd_eps):
    """Central-difference gradient of V in the 5D subspace."""
    k = len(u_vec)
    grad = np.zeros(k)
    for i in range(k):
        u_p = u_vec.copy(); u_p[i] += fd_eps
        u_m = u_vec.copy(); u_m[i] -= fd_eps
        Vp = eval_V(u_p, state, wk_names, P_cols, LM, get_batch, n_eval)
        Vm = eval_V(u_m, state, wk_names, P_cols, LM, get_batch, n_eval)
        grad[i] = (Vp - Vm) / (2*fd_eps)
    return grad


def hessian_V(u_vec, state, wk_names, P_cols, LM, get_batch, n_eval, fd_eps):
    """Second-order FD Hessian of V at u_vec."""
    k = len(u_vec)
    H = np.zeros((k, k))
    V0 = eval_V(u_vec, state, wk_names, P_cols, LM, get_batch, n_eval)
    for i in range(k):
        for j in range(i, k):
            u_pp = u_vec.copy(); u_pp[i]+=fd_eps; u_pp[j]+=fd_eps
            u_pm = u_vec.copy(); u_pm[i]+=fd_eps; u_pm[j]-=fd_eps
            u_mp = u_vec.copy(); u_mp[i]-=fd_eps; u_mp[j]+=fd_eps
            u_mm = u_vec.copy(); u_mm[i]-=fd_eps; u_mm[j]-=fd_eps
            Vpp = eval_V(u_pp, state, wk_names, P_cols, LM, get_batch, n_eval)
            Vpm = eval_V(u_pm, state, wk_names, P_cols, LM, get_batch, n_eval)
            Vmp = eval_V(u_mp, state, wk_names, P_cols, LM, get_batch, n_eval)
            Vmm = eval_V(u_mm, state, wk_names, P_cols, LM, get_batch, n_eval)
            H[i,j] = H[j,i] = (Vpp-Vpm-Vmp+Vmm)/(4*fd_eps**2)
    return (H+H.T)/2


def main():
    args = parse_args()
    print("="*60)
    print("  V(u) CRITICAL POINT — CATASTROPHE CLASSIFICATION")
    print("  min_{u ∈ R^5} L(w₀ + Pu)")
    print("="*60)

    comp = import_compiler(args.compiler)
    LM, get_batch = comp.LM, comp.get_batch

    state = load_state(args.spike64)
    wk_pairs = extract_wk(state)
    wk_list  = [w for _, w in wk_pairs]
    wk_names = [n for n, _ in wk_pairs]   # list of strings

    print(f"\n  Computing null eigenvectors …")
    model = SymplecticProxy(wk_list, args.rank)
    evals, evecs = lanczos_null(model, args.k_lanczos, 5)
    dim = evecs.shape[1]
    print(f"  Null eigenvalues: " + ", ".join(f"{e:.4f}" for e in evals))

    # Build P_cols: for each basis vector vᵢ, store (name, slice, delta) per WK layer
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    P_cols = []
    for i in range(dim):
        v = evecs[:, i]
        col = []
        for j, name in enumerate(wk_names):
            s, e = offsets[j], offsets[j+1]
            delta = torch.tensor(v[s:e], dtype=torch.float32)
            col.append((name, s, e, delta))
        P_cols.append(col)

    # Baseline
    V0 = eval_V(np.zeros(dim), state, wk_names, P_cols, LM, get_batch, args.n_eval)
    print(f"  V(0) = {V0:.4f}")

    # Gradient at origin
    print(f"\n  Gradient ∇V(0) …")
    grad0 = gradient_V(np.zeros(dim), state, wk_names, P_cols,
                        LM, get_batch, args.n_eval, args.fd_eps)
    print(f"  ‖∇V(0)‖ = {np.linalg.norm(grad0):.4f}")
    print(f"  Components: " + ", ".join(f"{g:+.4f}" for g in grad0))

    # Multi-start gradient-based optimization in 5D
    print(f"\n  Searching for critical point u* = argmin_u V(u) …")
    print(f"  Using L-BFGS-B with FD gradient (fd_eps={args.fd_eps})")
    print(f"  Search radius: {args.search_radius}  n_eval/pt: {args.n_eval}")

    best_u, best_V = np.zeros(dim), V0
    n_starts = 0

    def objective_and_grad(u):
        v = eval_V(u, state, wk_names, P_cols, LM, get_batch, args.n_eval)
        g = gradient_V(u, state, wk_names, P_cols, LM, get_batch,
                       args.n_eval, args.fd_eps)
        return v, g

    def objective(u):
        return eval_V(u, state, wk_names, P_cols, LM, get_batch, args.n_eval)

    def jac(u):
        return gradient_V(u, state, wk_names, P_cols, LM, get_batch,
                          args.n_eval, args.fd_eps)

    # Diverse initializations — use gradient direction as first hint
    init_points = [np.zeros(dim)]
    # Follow negative gradient from origin
    if np.linalg.norm(grad0) > 0:
        step_dir = -grad0 / np.linalg.norm(grad0)
        for scale in [0.1, 0.3, args.search_radius]:
            init_points.append(step_dir * scale)
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(max(0, args.grid_init - len(init_points))):
        init_points.append(
            rng.uniform(-args.search_radius, args.search_radius, dim))

    for u0 in init_points:
        n_starts += 1
        res = minimize(objective, u0, jac=jac, method='L-BFGS-B',
                       bounds=[(-args.search_radius*3, args.search_radius*3)]*dim,
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-4})
        print(f"  start {n_starts}: V={res.fun:.4f}  ‖g‖={np.linalg.norm(res.jac):.4f}  "
              f"u=[{', '.join(f'{x:.3f}' for x in res.x)}]")
        if res.fun < best_V:
            best_V = res.fun
            best_u = res.x.copy()

    print(f"\n  Best critical point found:")
    print(f"  u* = [{', '.join(f'{x:.4f}' for x in best_u)}]")
    print(f"  V(u*) = {best_V:.4f}  (V(0) = {V0:.4f}  ΔV = {best_V-V0:+.4f})")

    # Verify gradient at u*
    print(f"\n  Verifying stationarity ∇V(u*) …")
    grad_star = gradient_V(best_u, state, wk_names, P_cols,
                            LM, get_batch, args.n_eval, args.fd_eps)
    grad_norm = float(np.linalg.norm(grad_star))
    print(f"  ‖∇V(u*)‖ = {grad_norm:.4f}  "
          f"({'✓ stationary' if grad_norm < 0.05 else '◑ not fully converged'})")

    # Hessian at u*
    print(f"\n  Computing Hessian H_V at u* ({dim}×{dim} matrix) …")
    H_star = hessian_V(best_u, state, wk_names, P_cols,
                        LM, get_batch, args.n_eval, args.fd_eps)
    h_evals = np.linalg.eigvalsh(H_star)
    print(f"  H_V eigenvalues: " + ", ".join(f"{e:+.4f}" for e in h_evals))
    print(f"  det(H_V) = {np.linalg.det(H_star):+.4f}")
    print(f"  tr(H_V)  = {np.trace(H_star):+.4f}")
    print(f"  min |λ|  = {min(abs(e) for e in h_evals):.4f}")

    n_near_zero = sum(1 for e in h_evals if abs(e) < 0.1)
    print(f"  Eigenvalues near 0 (|λ|<0.1): {n_near_zero}/{dim}")

    # Catastrophe classification from Hessian
    print(f"\n  Classification:")
    if grad_norm > 0.1:
        cls = "NOT_AT_CRITICAL_POINT"
        msg = f"‖∇V‖={grad_norm:.3f}>0.1. Find better critical point."
    elif n_near_zero == 0:
        cls = "NON_DEGENERATE"
        msg = "All Hessian eigenvalues nonzero. Not a catastrophe point."
    elif n_near_zero == 1:
        # Check cubic term along zero-eigenvector
        v_null = np.linalg.eigh(H_star)[1][:, 0]  # smallest eigenvalue direction
        # Cubic: ∂³V/∂v³ via FD
        h = args.fd_eps
        Vp  = eval_V(best_u + h*v_null, state, wk_names, P_cols, LM, get_batch, args.n_eval)
        Vm  = eval_V(best_u - h*v_null, state, wk_names, P_cols, LM, get_batch, args.n_eval)
        V2p = eval_V(best_u + 2*h*v_null, state, wk_names, P_cols, LM, get_batch, args.n_eval)
        V2m = eval_V(best_u - 2*h*v_null, state, wk_names, P_cols, LM, get_batch, args.n_eval)
        # Third derivative: (-V(2h) + 2V(h) - 2V(-h) + V(-2h)) / (2h³)
        cubic = (-V2p + 2*Vp - 2*Vm + V2m) / (2*h**3)
        print(f"  Cubic term along null direction: {cubic:+.4f}")
        if abs(cubic) > 0.01:
            cls = "FOLD_A2"
            msg = (f"1 near-zero eigenvalue, cubic={cubic:+.4f}≠0. "
                   "Consistent with fold (A₂: x³+μx).")
        else:
            # Quartic
            quartic = (V2p - 4*Vp + 6*best_V - 4*Vm + V2m) / h**4
            print(f"  Quartic term along null direction: {quartic:+.4f}")
            cls = "CUSP_A3" if abs(quartic) > 0.01 else "HIGHER_ORDER"
            msg = (f"1 near-zero eigenvalue, cubic≈0, quartic={quartic:+.4f}. "
                   "Consistent with cusp (A₃) or higher.")
    else:
        cls = "MULTIPLY_DEGENERATE"
        msg = f"{n_near_zero} near-zero eigenvalues. Higher-codimension singularity."

    print(f"  {cls}: {msg}")

    report = {
        'V0': float(V0), 'best_V': float(best_V),
        'best_u': best_u.tolist(),
        'grad_at_origin': grad0.tolist(),
        'grad_norm_at_origin': float(np.linalg.norm(grad0)),
        'grad_at_ustar': grad_star.tolist(),
        'grad_norm_at_ustar': float(grad_norm),
        'hessian': H_star.tolist(),
        'hessian_eigenvalues': h_evals.tolist(),
        'n_near_zero_evals': n_near_zero,
        'null_evals': evals.tolist(),
        'classification': cls,
        'message': msg,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
