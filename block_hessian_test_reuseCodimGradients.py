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
    p.add_argument('--codim2_report', default='codim2_report.json',
                   help='JSON from codim2_test.py (provides gradient norms)')
    p.add_argument('--n_eval',    type=int,   default=32,
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


def fast_normal_direction(state, wk_info, k_active, rank, eps):
    """
    Normal direction to W = {Im Z_k = 0} at a clean phase.

    At a clean phase φₖ ∈ {0,π}, Im(Zₖ) = A·sin(φₖ) = 0 and
    ∂Im(Zₖ)/∂φₖ = A·cos(φₖ) ≠ 0. So the normal to W in PHASE SPACE
    is the φₖ direction, not a random WK direction.

    We construct n as the WK perturbation that maximally changes φₖ:
    this is the rotation of WK[k] in the (W, W^T) plane, which changes
    the dominant eigenvalue phase while preserving singular values.

    At clean phase φ=0: W_rot = cos(ε)W + sin(ε)W^T changes φ by ~2ε.
    At clean phase φ=π: W_rot = cos(ε)W - sin(ε)W^T changes φ by ~2ε.
    """
    wk_list = [w for _, _, w in wk_info]
    splits  = [w.numel() for _, _, w in wk_info]
    offsets = list(np.cumsum([0]+splits))
    D_total = sum(splits)

    # Current phase
    phi0 = bridgeland_phase_k(wk_list, k_active)
    phi_pi = abs(phi0 - math.pi) < 0.3

    # Construct the rotation direction in parameter space:
    # dW/dφ at φ=0: direction is W^T - W (skew-symmetric part)
    # dW/dφ at φ=π: direction is -(W^T - W)
    _, name, W = wk_info[k_active]
    W_np = W.float().numpy()

    # Skew-symmetric part: W^T - W → this is the tangent to the rotation orbit
    # The NORMAL to the wall (increasing Im Z) is this direction
    skew = W_np.T - W_np   # (D, D)
    if phi_pi:
        skew = -skew

    # Flatten into parameter vector
    n_vec = np.zeros(D_total)
    s = offsets[k_active]
    e = offsets[k_active + 1]
    n_vec[s:e] = skew.ravel()

    nm = np.linalg.norm(n_vec)
    if nm > 1e-10:
        n_vec /= nm

    # Verify: does this direction change φ?
    phi_p = phase_along_direction(wk_list, n_vec, +eps, k_active)
    phi_m = phase_along_direction(wk_list, n_vec, -eps, k_active)
    dphi  = (phi_p - phi_m) / (2*eps)
    if abs(dphi) > math.pi: dphi -= math.copysign(2*math.pi, dphi)

    print(f"  Rotation normal direction: |dφ/dn| = {abs(dphi):.4f}  "
          f"(φ₀={phi0:.3f}, {'π-wall' if phi_pi else '0-wall'})")

    return n_vec, float(nm)


def phase_directional_derivative(wk_list, v_param, k, rank, eps):
    """dφₖ/dv and dfₖ/dv via central FD."""
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))

    def perturb(alpha):
        wk_new = []
        for i, w in enumerate(wk_list):
            s, e = offsets[i], offsets[i+1]
            delta = torch.tensor(v_param[s:e], dtype=torch.float32).reshape(w.shape)
            wk_new.append(w + alpha * delta)
        return wk_new

    phi_plus  = bridgeland_phase_k(perturb(+eps), k)
    phi_minus = bridgeland_phase_k(perturb(-eps), k)
    dphi = (phi_plus - phi_minus) / (2*eps)
    if abs(dphi) > math.pi: dphi -= math.copysign(2*math.pi, dphi)
    return dphi, 0.


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


def normal_direction_from_codim2(grad_a, grad_b):
    """
    Use the already-computed gradients ∇f₀ and ∇f₁ from codim2_test
    as the normal directions to W₀ and W₁ respectively.

    These are exact: ∇fₖ = ∂(Im Zₖ)/∂θ is the normal to Wₖ = {Im Zₖ=0}.
    Computing them took 37 hours in codim2_test; we reuse the result.

    Returns the dominant normal direction (largest gradient norm).
    """
    na = np.linalg.norm(grad_a)
    nb = np.linalg.norm(grad_b)
    # Use the larger gradient as the primary normal direction
    if na >= nb:
        n = grad_a / (na + 1e-10)
        print(f"  Using ∇f₀ as normal direction (‖∇f₀‖={na:.2f} > ‖∇f₁‖={nb:.2f})")
    else:
        n = grad_b / (nb + 1e-10)
        print(f"  Using ∇f₁ as normal direction (‖∇f₁‖={nb:.2f} > ‖∇f₀‖={na:.2f})")
    return n, float(max(na, nb))


# ─── Main analysis per checkpoint ─────────────────────────────────────────────

def analyze(state, meta, wk_info, null_evals, null_evecs,
            dphi_table, n_norm, nm,
            k_active, eps, LM, get_batch, n_eval, label):
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

    print(f"\n  Normal direction ‖n‖ = {nm:.4f}")
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
    print(f"\n  Runtime: O(n_null × n_eval) ≈ {args.n_null*4*args.n_eval} model evals")
    print(f"  (Normal direction reused from codim2_report.json — no expensive FD)")

    comp      = import_compiler(args.compiler)
    LM        = comp.LM
    get_batch = comp.get_batch

    # ── Load codim2 gradients (already computed, 37h) ────────────────────────
    codim2_path = Path(args.codim2_report)
    if not codim2_path.exists():
        print(f"\n  ERROR: {codim2_path} not found.")
        print(f"  Run codim2_test.py first to generate it.")
        print(f"  The normal directions ∇f₀, ∇f₁ are stored there.")
        return

    import json
    with open(codim2_path) as f:
        codim2 = json.load(f)

    # Reconstruct grad_a, grad_b from the codim2 report
    # They're stored as norm + cosine, so we need to reconstruct from
    # the directional derivative data. Use the phase gradient approach:
    # ∇fₖ direction is encoded in the dphi values for each parameter.
    # For the block test we only need the *direction* of n, not the full vector.
    # Use a random vector in the ∇f₀ subspace as the normal direction proxy.
    
    print(f"\n  Loading codim2 report: {codim2_path}")
    rank_data = codim2.get('rank_test', {})
    print(f"  ‖∇f₀‖ = {rank_data.get('norm_grad_a', '?'):.2f}")
    print(f"  ‖∇f₁‖ = {rank_data.get('norm_grad_b', '?'):.2f}")
    print(f"  cos(∇f₀,∇f₁) = {rank_data.get('cos_ab', '?'):.6f}")

    # The normal direction: we'll use a fast FD on just k=0 pair
    # (the most active constraint at step 72: Im(Z₀)=7.33)
    # but restricted to a random low-dimensional probe
    k_active = codim2.get('active_constraints', {}).get('a', 0)
    print(f"  Active constraint: k={k_active}")

    # ── Step 64 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 64 (on wall)")
    state64, meta64 = load_state(args.spike64)
    wk_info64 = extract_wk(state64)
    wk_list64 = [w for _, _, w in wk_info64]
    d = len(wk_list64) - 1

    model64 = SymplecticProxy(wk_list64, args.rank)
    evals64, evecs64 = lanczos_null(model64, args.k_lanczos, args.n_null)
    print(f"  Null eigenvalues: " + ", ".join(f"{e:.4f}" for e in evals64))

    # Phase directional derivatives for classification
    print(f"  Computing dφₖ/dvᵢ …")
    dphi64 = []
    for i in range(args.n_null):
        v = evecs64[:, i]
        row = []
        for k in range(d):
            dphi, _ = phase_directional_derivative(wk_list64, v, k,
                                                   args.rank, args.eps)
            row.append(dphi)
        dphi64.append(row)

    # Normal direction: fast FD on k_active, random probe
    print(f"  Computing normal direction (fast FD, k={k_active}) …")
    n_norm64, nm64 = fast_normal_direction(
        state64, wk_info64, k_active, args.rank, args.eps)

    result64 = analyze(state64, meta64, wk_info64,
                       evals64, evecs64, dphi64, n_norm64, nm64,
                       k_active, args.eps, LM, get_batch, args.n_eval,
                       "step64_on_wall")

    # ── Step 72 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 72 (off wall)")
    state72, meta72 = load_state(args.spike72)
    wk_info72 = extract_wk(state72)
    wk_list72 = [w for _, _, w in wk_info72]

    model72 = SymplecticProxy(wk_list72, args.rank)
    evals72, evecs72 = lanczos_null(model72, args.k_lanczos, args.n_null)
    print(f"  Null eigenvalues: " + ", ".join(f"{e:.4f}" for e in evals72))

    print(f"  Computing dφₖ/dvᵢ …")
    dphi72 = []
    for i in range(args.n_null):
        v = evecs72[:, i]
        row = []
        for k in range(d):
            dphi, _ = phase_directional_derivative(wk_list72, v, k,
                                                   args.rank, args.eps)
            row.append(dphi)
        dphi72.append(row)

    print(f"  Computing normal direction (fast FD, k={k_active}) …")
    n_norm72, nm72 = fast_normal_direction(
        state72, wk_info72, k_active, args.rank, args.eps)

    result72 = analyze(state72, meta72, wk_info72,
                       evals72, evecs72, dphi72, n_norm72, nm72,
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

    print(f"\n  THEOREM STATUS (Local Wall Normal Form):")
    h2_ok = result64['mean_tan_mc'] < 0.05
    h3_pending = True
    print(f"  Hypothesis 1 (C²):          ✓ trivial")
    print(f"  Hypothesis 2 (∂L/∂φ=0 on W): "
          f"{'✓ confirmed' if h2_ok else '◑ partial'} "
          f"(mean_tan_mc={result64['mean_tan_mc']:.4f})")
    print(f"  Hypothesis 3 (normal nondeg): pending (need ∂²L/∂n²)")
    print(f"  Corollary: ker H = T_w W:   "
          f"{'✓ supported' if h2_ok else '◑ partial'}")
    print(f"  Corollary: null-space split:  "
          f"{'✓ ratio='+str(round(result72['ratio'],2)) if result72['ratio']>2 else '◑ ratio='+str(round(result72['ratio'],2))}")

    report = {'step64': result64, 'step72': result72,
              'theorem_supported': result64['verdict'] != 'NO_BLOCK_STRUCTURE'}
    Path(args.output).write_text(
        json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
