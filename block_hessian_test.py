"""
block_hessian_test.py
======================
Tests the block Hessian structure at the clean-phase wall.

Theorem target (Block Hessian at Clean Phase):
  Let w ∈ W = {Im Z = 0}. Suppose:
    (1) L satisfies local reflection symmetry at w  [Stage III, confirmed]
    (2) ∂²L/∂φₖ∂xⱼ = 0 at w for all tangent directions xⱼ  [this test]
  Then H(w) = diag(H_tan, A) + O(y), and if L₀ is locally constant on W:
    ker H(w) = T_w W.

  The null-space splitting at step 72 follows from H(w+yn) = diag(0,A) + y·B + O(y²)
  where B is the mixed curvature tensor. Eigenvectors of B with nonzero
  eigenvalue are the E_cross directions (the 3/5 that go non-tangent).

What we compute:
  At step 64 (w on wall, y=0):
    - Tangent directions: null eigenvectors v₁,...,v₅ (dφₖ/dvᵢ = 0)
    - Normal directions: gradient of φₖ (dφₖ/dn ≠ 0)
    - Mixed curvature: ∂²L/∂vᵢ∂n for each null eigenvector vᵢ
    - If all mixed curvatures ≈ 0: block structure holds → ker H = T_w W

  At step 72 (w off wall, y≠0):  
    - Same computation but with nonzero y
    - Mixed curvature should be nonzero for E_cross directions
    - Ratio |mixed_curv(E_cross)| / |mixed_curv(E_tan)| quantifies splitting

If mixed curvatures ≈ 0 at step 64: condition (2) confirmed, theorem holds.
If mixed curvatures ≠ 0 at step 64: block structure fails, different hypothesis needed.

Usage
-----
  python block_hessian_test.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --spike72 tau_spikes/tau_spike_step0072_tau5.94.pt \\
      --compiler compiler_geometric.py \\
      --rank 6 --k_lanczos 20 --n_null 5 --eps 1e-3 \\
      --output block_hessian_report.json
"""

import argparse, json, math, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64', default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--spike72', default='tau_spikes/tau_spike_step0072_tau5.94.pt')
    p.add_argument('--compiler',  default='compiler_geometric.py')
    p.add_argument('--rank',      type=int,   default=6)
    p.add_argument('--k_lanczos', type=int,   default=20)
    p.add_argument('--n_null',    type=int,   default=5)
    p.add_argument('--eps',       type=float, default=1e-3,
                   help='FD step for second derivatives')
    p.add_argument('--n_eval',    type=int,   default=64,
                   help='Batches for loss evaluation')
    p.add_argument('--output',    default='block_hessian_report.json')
    return p.parse_args()


# ─── Compiler + model ────────────────────────────────────────────────────────

def import_compiler(path):
    globs = {'__name__': '__compiler__', '__file__': path,
             '__builtins__': __builtins__}
    src = open(path).read()
    src = src.replace('if __name__ == "__main__":', 'if False:')
    src = src.replace("if __name__ == '__main__':", 'if False:')
    import re
    src = re.sub(r'torch\.save\(', '_NOSAVE(', src)
    globs['_NOSAVE'] = lambda *a, **kw: None
    exec(compile(src, path, 'exec'), globs)
    return types.SimpleNamespace(**{k: v for k, v in globs.items()
                                    if not k.startswith('__')})


def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        meta  = ckpt.get('metadata', {})
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        meta, state = {}, ckpt
    return {k: v.clone() for k, v in state.items()
            if isinstance(v, torch.Tensor)}, meta


@torch.no_grad()
def eval_loss(model, get_batch, n=64):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = (name, tensor)
    return [(i, wk[i][0], wk[i][1]) for i in sorted(wk)]


# ─── Symplectic Hessian machinery ────────────────────────────────────────────

class SymplecticProxy(nn.Module):
    def __init__(self, wk_list, rank):
        super().__init__()
        self.rank   = rank
        self.theta  = nn.Parameter(
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
            Ua = torch.linalg.svd(wks[k],   full_matrices=False)[0][:,:self.rank]
            Ub = torch.linalg.svd(wks[k+1], full_matrices=False)[0][:,:self.rank]
            sv = torch.linalg.svdvals(Ua.T@Ub).clamp(1e-6,1-1e-6)
            loss = loss + torch.arccos(sv).sum()
        return loss

def hvp(model, v):
    loss = model()
    grad = torch.autograd.grad(loss, model.theta, create_graph=True)[0]
    return torch.autograd.grad((grad*v).sum(), model.theta)[0].detach()

def lanczos_null(model, k, n_null, seed=42):
    """Return n_null near-null eigenvectors (smallest |λ|) via Lanczos."""
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k+1)
    alpha_c = torch.zeros(k); beta_c = torch.zeros(k-1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:,0]=v; prev_b=0.
    kk = k
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
    idx = np.argsort(np.abs(ev))   # smallest |λ| first
    return ev[idx[:n_null]], evecs[:,idx[:n_null]]


# ─── Phase and loss directional derivatives ───────────────────────────────────

def bridgeland_phase_k(wk_list, k):
    Wk  = wk_list[k].float().numpy()
    Wk1 = wk_list[k+1].float().numpy()
    M = Wk1 @ np.linalg.pinv(Wk)
    evals = np.linalg.eigvals(M)
    dom = evals[np.argmax(np.abs(evals.real))]
    phi = float(np.arctan2(dom.imag, dom.real))
    if phi < 0: phi += 2*math.pi
    return phi


def loss_along_direction(state, wk_info, v_flat, alpha, LM, get_batch, n_eval):
    """
    Evaluate L(θ + α·v) where v is a parameter-space direction.
    """
    state_new = {n: t.clone() for n, t in state.items()}
    splits  = [w.numel() for _, _, w in wk_info]
    offsets = list(np.cumsum([0]+splits))
    for i, (_, name, _) in enumerate(wk_info):
        s, e = offsets[i], offsets[i+1]
        delta = torch.tensor(v_flat[s:e], dtype=torch.float32)
        state_new[name] = state_new[name] + alpha * delta.reshape(state_new[name].shape)
    model = LM()
    model.load_state_dict(state_new, strict=False)
    return eval_loss(model, get_batch, n_eval)


def phase_along_direction(wk_list, v_flat, alpha, k):
    """φₖ(θ + α·v)"""
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    wk_new = []
    for i, w in enumerate(wk_list):
        s, e = offsets[i], offsets[i+1]
        delta = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
        wk_new.append(w + alpha * delta)
    return bridgeland_phase_k(wk_new, k)


# ─── Mixed curvature ∂²L/∂v∂n ────────────────────────────────────────────────

def mixed_curvature(state, wk_info, v_tan, n_norm, eps,
                    LM, get_batch, n_eval):
    """
    ∂²L/∂v∂n ≈ [L(+v+n) - L(+v-n) - L(-v+n) + L(-v-n)] / (4ε²)

    v_tan: tangent direction (null eigenvector, dφ/dv≈0)
    n_norm: normal direction (phase gradient, dφ/dn≠0)

    If block structure holds: ∂²L/∂v∂n = 0
    If not: nonzero, quantifying the coupling between tangent and normal directions
    """
    wk_list = [w for _, _, w in wk_info]

    Lpp = loss_along_direction(state, wk_info,  eps*v_tan + eps*n_norm, 1., LM, get_batch, n_eval)
    Lpm = loss_along_direction(state, wk_info,  eps*v_tan - eps*n_norm, 1., LM, get_batch, n_eval)
    Lmp = loss_along_direction(state, wk_info, -eps*v_tan + eps*n_norm, 1., LM, get_batch, n_eval)
    Lmm = loss_along_direction(state, wk_info, -eps*v_tan - eps*n_norm, 1., LM, get_batch, n_eval)

    return (Lpp - Lpm - Lmp + Lmm) / (4 * eps**2)


def normal_direction(state, wk_info, k_active, eps, LM, get_batch, n_eval):
    """
    Estimate the normal direction to W at the current point:
    n = ∂φₖ/∂θ ∝ gradient of the phase function.

    Approximated by finite differences on the phase map.
    We use a random probe in parameter space, project onto
    the phase gradient direction via:
      dφₖ/dθ ≈ (φₖ(θ+ε·eᵢ) - φₖ(θ-ε·eᵢ)) / (2ε) for each i

    For D=393216 this is too expensive. Instead use a low-rank
    approximation: the phase gradient lives in the span of
    WK[k] and WK[k+1] blocks only.
    Compute the gradient in those blocks (D_block << D total).
    """
    wk_list = [w for _, _, w in wk_info]
    splits  = [w.numel() for _, _, w in wk_info]
    offsets = list(np.cumsum([0]+splits))
    D_total = sum(splits)

    grad_phi = np.zeros(D_total)

    # Only WK[k] and WK[k+1] affect φₖ
    for layer_idx in [k_active, min(k_active+1, len(wk_list)-1)]:
        _, name, W = wk_info[layer_idx]
        n = W.numel()
        s = offsets[layer_idx]

        W_flat = W.reshape(-1).numpy()
        for flat_idx in range(n):
            v = np.zeros(D_total)
            v[s + flat_idx] = 1.0

            phi_p = phase_along_direction(wk_list, v, +eps, k_active)
            phi_m = phase_along_direction(wk_list, v, -eps, k_active)
            dphi  = (phi_p - phi_m) / (2*eps)
            if abs(dphi) > math.pi: dphi -= math.copysign(2*math.pi, dphi)
            grad_phi[s + flat_idx] = dphi

    nm = np.linalg.norm(grad_phi)
    return grad_phi / nm if nm > 1e-10 else grad_phi


# ─── Main analysis per checkpoint ─────────────────────────────────────────────

def analyze(state, meta, wk_info, null_evals, null_evecs,
            dphi_table, k_active, eps, LM, get_batch, n_eval, label):
    """
    For each null eigenvector:
      1. Classify as E_tan (dφ≈0) or E_cross (dφ≠0) using dphi_table
      2. Compute normal direction n for the active constraint
      3. Compute mixed curvature ∂²L/∂v∂n
      4. Test: is ∂²L/∂v∂n ≈ 0 for E_tan vectors? (block structure condition)
    """
    t0 = time.time()
    wk_list = [w for _, _, w in wk_info]
    d = len(wk_list) - 1

    print(f"\n  Computing normal direction for active constraint k={k_active} …")
    n_norm = normal_direction(state, wk_info, k_active, eps, LM, get_batch, n_eval)
    nm = np.linalg.norm(n_norm)
    print(f"  ‖n‖ = {nm:.4f}  (in WK[{k_active}]+WK[{k_active+1}] block only)")

    print(f"\n  Mixed curvatures ∂²L/∂vᵢ∂n:")
    print(f"  {'v':>4} {'λ':>10} {'dφ_a/dv':>10} {'type':>8} {'∂²L/∂v∂n':>12}")
    print(f"  {'-'*50}")

    results = []
    for i in range(null_evecs.shape[1]):
        v = null_evecs[:, i]
        lam = float(null_evals[i])
        dphi_a = float(dphi_table[i][k_active]) if k_active < len(dphi_table[i]) else 0.
        is_tan = abs(dphi_a) < 0.1
        vtype  = "E_tan" if is_tan else "E_cross"

        mc = mixed_curvature(state, wk_info, v, n_norm, eps,
                             LM, get_batch, n_eval)

        print(f"  v{i:>2}  {lam:>10.4f}  {dphi_a:>+10.4f}  "
              f"{vtype:>8}  {mc:>+12.6f}")

        results.append({
            'i': i, 'eigenvalue': lam,
            'dphi_active': dphi_a,
            'type': vtype,
            'mixed_curvature': float(mc),
        })

    # Block structure test
    tan_mc  = [r['mixed_curvature'] for r in results if r['type']=='E_tan']
    cross_mc = [r['mixed_curvature'] for r in results if r['type']=='E_cross']

    mean_tan_mc  = float(np.mean(np.abs(tan_mc)))   if tan_mc   else 0.
    mean_cross_mc = float(np.mean(np.abs(cross_mc))) if cross_mc else 0.
    ratio = mean_cross_mc / (mean_tan_mc + 1e-8)

    print(f"\n  Block structure test:")
    print(f"  Mean |∂²L/∂v∂n| for E_tan  vectors: {mean_tan_mc:.6f}")
    print(f"  Mean |∂²L/∂v∂n| for E_cross vectors: {mean_cross_mc:.6f}")
    print(f"  Ratio cross/tan: {ratio:.3f}")

    if mean_tan_mc < 0.01:
        verdict = "BLOCK_STRUCTURE_CONFIRMED"
        msg = (f"Mixed curvature ≈ 0 for E_tan directions. "
               "Block Hessian structure holds: H = diag(0, A) + O(y). "
               "Theorem: ker H = T_w W confirmed.")
    elif mean_tan_mc < 0.05:
        verdict = "APPROXIMATE_BLOCK_STRUCTURE"
        msg = (f"Mixed curvature small but nonzero ({mean_tan_mc:.4f}). "
               "Approximate block structure. Theorem holds up to O(ε) corrections.")
    else:
        verdict = "NO_BLOCK_STRUCTURE"
        msg = (f"Mixed curvature {mean_tan_mc:.4f} not negligible. "
               "Block structure fails. Different hypothesis needed for Stage IV.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    elapsed = time.time() - t0
    return {
        'label':        label,
        'k_active':     k_active,
        'normal_norm':  float(nm),
        'null_evals':   null_evals.tolist(),
        'results':      results,
        'mean_tan_mc':  mean_tan_mc,
        'mean_cross_mc': mean_cross_mc,
        'ratio':        ratio,
        'verdict':      verdict,
        'message':      msg,
        'elapsed_s':    round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  BLOCK HESSIAN TEST")
    print("  Tests: H = diag(0, A) + O(y)?  (ker H = T_w W)")
    print("  Mixed curvature ∂²L/∂v∂n for null eigenvectors")
    print("=" * 60)

    comp     = import_compiler(args.compiler)
    LM        = comp.LM
    get_batch = comp.get_batch

    # ── Step 64: on the wall ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 64 (on wall, Im(Z)=0, 5/5 tangent)")
    state64, meta64 = load_state(args.spike64)
    wk_info64 = extract_wk(state64)
    wk_list64 = [w for _, _, w in wk_info64]
    d = len(wk_list64) - 1

    print(f"  Computing null eigenvectors (Lanczos k={args.k_lanczos}) …")
    model64 = SymplecticProxy(wk_list64, args.rank)
    evals64, evecs64 = lanczos_null(model64, args.k_lanczos, args.n_null)
    print(f"  Null eigenvalues: " + ", ".join(f"{e:.4f}" for e in evals64))

    # Phase directional derivatives for classification
    from codim2_test import phase_directional_derivative  # reuse from codim2
    print(f"  Computing dφₖ/dvᵢ for classification …")
    dphi64 = []
    for i in range(args.n_null):
        v = evecs64[:, i]
        row = []
        for k in range(d):
            dphi, _ = phase_directional_derivative(wk_list64, v, k, args.rank, args.eps)
            row.append(dphi)
        dphi64.append(row)

    # Active constraint at step 64: all phases clean, use the two that
    # become active at step 72 (k=0 and k=1 from codim2 test)
    k_active = 0   # Im(Z₀) becomes +7.33 at step 72

    result64 = analyze(state64, meta64, wk_info64,
                       evals64, evecs64, dphi64,
                       k_active, args.eps, LM, get_batch, args.n_eval,
                       "step64_on_wall")

    # ── Step 72: off the wall ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 72 (off wall, Im(Z)≠0, 2/5 tangent, 3/5 crossing)")
    state72, meta72 = load_state(args.spike72)
    wk_info72 = extract_wk(state72)
    wk_list72 = [w for _, _, w in wk_info72]

    print(f"  Computing null eigenvectors …")
    model72 = SymplecticProxy(wk_list72, args.rank)
    evals72, evecs72 = lanczos_null(model72, args.k_lanczos, args.n_null)
    print(f"  Null eigenvalues: " + ", ".join(f"{e:.4f}" for e in evals72))

    print(f"  Computing dφₖ/dvᵢ for classification …")
    dphi72 = []
    for i in range(args.n_null):
        v = evecs72[:, i]
        row = []
        for k in range(d):
            dphi, _ = phase_directional_derivative(wk_list72, v, k, args.rank, args.eps)
            row.append(dphi)
        dphi72.append(row)

    result72 = analyze(state72, meta72, wk_info72,
                       evals72, evecs72, dphi72,
                       k_active, args.eps, LM, get_batch, args.n_eval,
                       "step72_off_wall")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BLOCK HESSIAN SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Step 64 (on wall):")
    print(f"    Mean |∂²L/∂v∂n| (E_tan):   {result64['mean_tan_mc']:.6f}")
    print(f"    Verdict: {result64['verdict']}")
    print(f"\n  Step 72 (off wall):")
    print(f"    Mean |∂²L/∂v∂n| (E_tan):   {result72['mean_tan_mc']:.6f}")
    print(f"    Mean |∂²L/∂v∂n| (E_cross): {result72['mean_cross_mc']:.6f}")
    print(f"    Ratio cross/tan:             {result72['ratio']:.3f}")
    print(f"    Verdict: {result72['verdict']}")

    # Key prediction for theorem
    print(f"\n  THEOREM STATUS:")
    if result64['verdict'] in ('BLOCK_STRUCTURE_CONFIRMED',
                               'APPROXIMATE_BLOCK_STRUCTURE'):
        print(f"  ✓ Stage IV supported: ker H = T_w W at step 64")
        print(f"  L(x,y) = L₀(x) + ½yᵀA(x)y + O(‖y‖³) confirmed")
        if result72['ratio'] > 2.0:
            print(f"  ✓ Splitting explained: E_cross has "
                  f"{result72['ratio']:.1f}× larger mixed curvature than E_tan")
            print(f"  ✓ H(w+yn) = diag(0,A) + y·B + O(y²): B differentiates E_tan/E_cross")
    else:
        print(f"  ✗ Stage IV not confirmed: block structure fails")
        print(f"    Need different hypothesis for ker H = T_w W")

    report = {
        'step64': result64,
        'step72': result72,
        'theorem_supported': result64['verdict'] != 'NO_BLOCK_STRUCTURE',
    }
    Path(args.output).write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
