"""
phase_equivariance_test.py
===========================
Tests whether the loss L factors through the central charges Z_k,
i.e., whether L = L(Z_1,...,Z_n).

This is the hypothesis required for the Morse-Bott theorem (Stage IV):

  Theorem (Phase Morse-Bott): If L is phase-equivariant and Morse-Bott
  along phase orbits, then ker H = T(phase orbit).

Phase equivariance means: rotating φ_k at fixed strip area leaves L unchanged.
  L(Z_k) = L(|Z_k| e^{iφ_k}) does not depend on φ_k if equivariant.

Three tests:
  A) Phase rotation: perturb W_K^k → R(ε)·W_K^k (rotation at fixed area)
     If equivariant: ΔL ≈ 0 (loss unchanged)
     If not: ΔL > 0

  B) Area perturbation: perturb W_K^k → (1+ε)·W_K^k (scale at fixed phase)
     Must give ΔL > 0 (loss depends on area — sanity check)

  C) Ratio test: |ΔL_phase| / |ΔL_area|
     → 0 as ε→0: L is asymptotically phase-equivariant
     → 1: L treats phase and area equally (not equivariant)
     
If ratio → 0: Morse-Bott hypothesis supported → Stage IV theorem applies.
If ratio ≈ 1: need different symmetry argument for Stage IV.

Also computes:
  - The phase gradient ∂L/∂φ_k at the current checkpoint
  - The area gradient ∂L/∂A_k
  - Ratio ‖∂L/∂φ‖ / ‖∂L/∂A‖ → 0 if phase-equivariant

Usage
-----
  python phase_equivariance_test.py \\
      --compiler compiler_geometric.py \\
      --checkpoints \\
          tau_spikes/tau_spike_step0064_tau5.90.pt \\
          tau_spikes/tau_spike_step0072_tau5.94.pt \\
          basin_state.pt \\
      --eps 0.01 0.001 0.0001 \\
      --rank 6 \\
      --output phase_equivariance_report.json
"""

import argparse, json, math, time, types
from pathlib import Path

import numpy as np
import torch
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
    p.add_argument('--checkpoints', nargs='+', default=['basin_state.pt'])
    p.add_argument('--compiler',    default='compiler_geometric.py')
    p.add_argument('--eps',         nargs='+', type=float,
                   default=[0.1, 0.01, 0.001],
                   help='Rotation angles / scale factors to test')
    p.add_argument('--rank',        type=int, default=6)
    p.add_argument('--n_eval',      type=int, default=32,
                   help='Batches for val evaluation')
    p.add_argument('--output',      default='phase_equivariance_report.json')
    return p.parse_args()


# ─── Compiler import ──────────────────────────────────────────────────────────

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


# ─── Checkpoint / model ───────────────────────────────────────────────────────

def load_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        meta  = ckpt.get('metadata', {})
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    else:
        meta, state = {}, ckpt
    return {k: v.clone() for k, v in state.items()
            if isinstance(v, torch.Tensor)}, meta


def build_model_from_state(LM, state):
    model = LM()
    model.load_state_dict(state, strict=False)
    return model


# ─── Phase rotation operator ──────────────────────────────────────────────────

def rotation_matrix_2d(theta, D):
    """
    Rotation in the (W_K, W_K^T) 2D subspace by angle theta.
    For a D×D weight matrix W, the rotation acts as:
      W → cos(θ)·W + sin(θ)·W^T  (rotation between W and its transpose)
    This changes φ_k while approximately preserving the singular values
    (hence approximately preserving the strip area A_k).

    For small θ: W → W + θ·(W^T - W) + O(θ²)
    The skew-symmetric part (W^T - W) is the "phase direction."
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    def rotate(W):
        return cos_t * W + sin_t * W.T
    return rotate


def area_scale(scale, D):
    """Scale W → scale·W. Changes strip area but not phase."""
    def scale_fn(W):
        return scale * W
    return scale_fn


def bridgeland_phase(W0, W1):
    """φ_k = arg(λ_dom(W_{k+1} W_k^{-1}))"""
    M = W1.float().numpy() @ np.linalg.pinv(W0.float().numpy())
    evals = np.linalg.eigvals(M)
    dom = evals[np.argmax(np.abs(evals.real))]
    phi = float(np.arctan2(dom.imag, dom.real))
    if phi < 0: phi += 2*math.pi
    return phi


def strip_area(W0, W1, rank):
    Uk  = torch.linalg.svd(W0.float(), full_matrices=False)[0][:, :rank]
    Uk1 = torch.linalg.svd(W1.float(), full_matrices=False)[0][:, :rank]
    sv  = torch.linalg.svdvals(Uk.T @ Uk1).clamp(-1+1e-6, 1-1e-6)
    return float(torch.arccos(sv).sum().item())


# ─── WK extraction ───────────────────────────────────────────────────────────

def extract_wk_names(state):
    """Return list of (layer_idx, name) for WK matrices."""
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = name
    return [(i, wk[i]) for i in sorted(wk)]


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, get_batch, n=32):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


# ─── Core test ────────────────────────────────────────────────────────────────

def test_phase_equivariance(state, meta, LM, get_batch, eval_val,
                             wk_names, rank, eps_values, n_eval, label):
    """
    For each layer k and each eps:
      1. Rotate W_K^k by eps (phase perturbation, approximately area-preserving)
      2. Scale W_K^k by (1+eps) (area perturbation, phase-preserving)
      3. Measure ΔL_phase and ΔL_area
      4. Compute ratio = |ΔL_phase| / |ΔL_area|
    """
    t0 = time.time()

    # Baseline
    model0 = build_model_from_state(LM, state)
    L0 = eval_loss(model0, get_batch, n_eval)
    print(f"  Baseline loss: {L0:.4f}")

    # Get WK matrices
    wk_list = []
    for _, name in wk_names:
        wk_list.append(state[name].float())
    d = len(wk_list) - 1

    # Current phases and areas
    phases0 = [bridgeland_phase(wk_list[k], wk_list[k+1])
                for k in range(d)]
    areas0  = [strip_area(wk_list[k], wk_list[k+1], rank)
                for k in range(d)]

    print(f"  Phases: " + ", ".join(f"{p:.3f}" for p in phases0))
    print(f"  Areas:  " + ", ".join(f"{a:.3f}" for a in areas0))

    results = []
    print(f"\n  {'k':>3} {'eps':>8} {'ΔL_phase':>12} {'ΔL_area':>12} "
          f"{'ratio':>8} {'Δφ_k':>8} {'ΔA_k':>8}")
    print(f"  {'-'*65}")

    for k in range(min(d, 3)):   # test first 3 pairs (speed)
        _, wk_name = wk_names[k]
        W_orig = state[wk_name].float().clone()

        for eps in eps_values:
            # ── Phase rotation ────────────────────────────────────────────
            state_rot = {n: v.clone() for n, v in state.items()}
            rotate = rotation_matrix_2d(eps, W_orig.shape[0])
            W_rot = rotate(W_orig)
            state_rot[wk_name] = W_rot

            phi_rot = bridgeland_phase(
                state_rot[wk_names[k][1]],
                state_rot[wk_names[k+1][1]] if k+1 < len(wk_names)
                else wk_list[k+1])
            A_rot = strip_area(
                state_rot[wk_names[k][1]],
                wk_list[k+1], rank)

            model_rot = build_model_from_state(LM, state_rot)
            L_rot = eval_loss(model_rot, get_batch, n_eval)
            dL_phase = L_rot - L0
            dphi = phi_rot - phases0[k]
            if abs(dphi) > math.pi: dphi -= math.copysign(2*math.pi, dphi)
            dA_phase = A_rot - areas0[k]

            # ── Area scale ────────────────────────────────────────────────
            state_scl = {n: v.clone() for n, v in state.items()}
            W_scl = (1 + eps) * W_orig
            state_scl[wk_name] = W_scl

            phi_scl = bridgeland_phase(
                state_scl[wk_names[k][1]],
                state_scl[wk_names[k+1][1]] if k+1 < len(wk_names)
                else wk_list[k+1])
            A_scl = strip_area(
                state_scl[wk_names[k][1]],
                wk_list[k+1], rank)

            model_scl = build_model_from_state(LM, state_scl)
            L_scl = eval_loss(model_scl, get_batch, n_eval)
            dL_area = L_scl - L0
            dA_area = A_scl - areas0[k]

            # ── Ratio ─────────────────────────────────────────────────────
            ratio = abs(dL_phase) / (abs(dL_area) + 1e-8)

            print(f"  {k:>3} {eps:>8.4f} {dL_phase:>+12.6f} {dL_area:>+12.6f} "
                  f"{ratio:>8.4f} {dphi:>+8.4f} {dA_phase:>+8.4f}")

            results.append({
                'k': k, 'eps': float(eps),
                'L0': float(L0), 'L_rot': float(L_rot), 'L_scl': float(L_scl),
                'dL_phase': float(dL_phase), 'dL_area': float(dL_area),
                'ratio': float(ratio),
                'dphi': float(dphi), 'dA_phase': float(dA_phase),
                'dA_area': float(dA_area - areas0[k]),
            })

    # ── Phase and area gradients via FD ──────────────────────────────────────
    print(f"\n  Phase gradient ∂L/∂φ_k and area gradient ∂L/∂A_k:")
    grad_phi, grad_A = [], []
    eps0 = eps_values[0]

    for k in range(d):
        _, wk_name = wk_names[k]
        W_orig = state[wk_name].float().clone()

        # dL/dφ_k via rotation
        state_p = {n: v.clone() for n, v in state.items()}
        state_m = {n: v.clone() for n, v in state.items()}
        rot_p = rotation_matrix_2d(+eps0, W_orig.shape[0])
        rot_m = rotation_matrix_2d(-eps0, W_orig.shape[0])
        state_p[wk_name] = rot_p(W_orig)
        state_m[wk_name] = rot_m(W_orig)

        Lp = eval_loss(build_model_from_state(LM, state_p), get_batch, n_eval)
        Lm = eval_loss(build_model_from_state(LM, state_m), get_batch, n_eval)
        dL_dphi = (Lp - Lm) / (2 * eps0)
        grad_phi.append(dL_dphi)

        # dL/dA_k via scaling
        state_sp = {n: v.clone() for n, v in state.items()}
        state_sm = {n: v.clone() for n, v in state.items()}
        state_sp[wk_name] = (1 + eps0) * W_orig
        state_sm[wk_name] = (1 - eps0) * W_orig

        Lsp = eval_loss(build_model_from_state(LM, state_sp), get_batch, n_eval)
        Lsm = eval_loss(build_model_from_state(LM, state_sm), get_batch, n_eval)
        dL_dA = (Lsp - Lsm) / (2 * eps0)
        grad_A.append(dL_dA)

        print(f"    k={k}: ∂L/∂φ_{k} = {dL_dphi:+.6f}  "
              f"∂L/∂A_{k} = {dL_dA:+.6f}  "
              f"ratio = {abs(dL_dphi)/(abs(dL_dA)+1e-8):.4f}")

    norm_phi = float(np.linalg.norm(grad_phi))
    norm_A   = float(np.linalg.norm(grad_A))
    global_ratio = norm_phi / (norm_A + 1e-8)

    print(f"\n  ‖∂L/∂φ‖ = {norm_phi:.6f}")
    print(f"  ‖∂L/∂A‖ = {norm_A:.6f}")
    print(f"  Global ratio = {global_ratio:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  VERDICT:")
    if global_ratio < 0.05:
        verdict = "PHASE_EQUIVARIANT"
        msg = (f"‖∂L/∂φ‖/‖∂L/∂A‖ = {global_ratio:.4f} < 0.05. "
               "Loss is approximately phase-equivariant. "
               "Morse-Bott hypothesis (Stage IV) is supported. "
               "ker(H) = T(phase orbit) can be derived as a theorem.")
    elif global_ratio < 0.2:
        verdict = "WEAKLY_EQUIVARIANT"
        msg = (f"‖∂L/∂φ‖/‖∂L/∂A‖ = {global_ratio:.4f} ∈ (0.05, 0.2). "
               "Weak phase equivariance. Morse-Bott holds approximately. "
               "Theorem requires additional symmetry argument.")
    else:
        verdict = "NOT_EQUIVARIANT"
        msg = (f"‖∂L/∂φ‖/‖∂L/∂A‖ = {global_ratio:.4f} > 0.2. "
               "Loss is not phase-equivariant at this checkpoint. "
               "Need different symmetry for Stage IV. "
               "Check if equivariance holds at other checkpoints or "
               "in the basin-entry phase.")

    print(f"  {verdict}: {msg}")

    elapsed = time.time() - t0
    return {
        'label':         label,
        'meta':          meta,
        'L0':            float(L0),
        'phases0':       phases0,
        'areas0':        areas0,
        'perturbations': results,
        'grad_phi':      grad_phi,
        'grad_A':        grad_A,
        'norm_phi':      float(norm_phi),
        'norm_A':        float(norm_A),
        'global_ratio':  float(global_ratio),
        'verdict':       verdict,
        'message':       msg,
        'elapsed_s':     round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  PHASE EQUIVARIANCE TEST")
    print("  Tests: L = L(Z_1,...,Z_n)?  (Morse-Bott hypothesis)")
    print("=" * 60)
    print(f"  eps values: {args.eps}")
    print(f"  n_eval: {args.n_eval} batches per evaluation")

    # Import compiler
    print(f"\n  Importing compiler …")
    comp     = import_compiler(args.compiler)
    LM        = comp.LM
    get_batch = comp.get_batch
    eval_val  = comp.eval_val

    all_results = []

    for ckpt_path in args.checkpoints:
        print(f"\n{'─'*60}\n  {ckpt_path}")
        state, meta = load_state(ckpt_path)
        wk_names = extract_wk_names(state)
        print(f"  WK layers: {[n for _, n in wk_names]}")
        print(f"  meta: step={meta.get('step','?')} "
              f"tau={meta.get('tau','?')} val={meta.get('val','?')}")

        result = test_phase_equivariance(
            state, meta, LM, get_batch, eval_val,
            wk_names, args.rank, args.eps, args.n_eval,
            Path(ckpt_path).stem)
        all_results.append(result)

    # Cross-checkpoint summary
    print(f"\n{'='*60}")
    print(f"  CROSS-CHECKPOINT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Checkpoint':<35} {'ratio':>8} {'Verdict'}")
    print(f"  {'-'*60}")
    for r in all_results:
        name = r['label'][:34]
        print(f"  {name:<35} {r['global_ratio']:>8.4f}  {r['verdict']}")

    print(f"\n  Interpretation:")
    print(f"  ratio → 0 as ε→0: L phase-equivariant → Morse-Bott applies")
    print(f"  ratio → 1: L equally sensitive to phase and area → need different Stage IV")

    # Check if ratio decreases with eps (asymptotic equivariance)
    for r in all_results:
        if len(r['perturbations']) >= 2:
            ratios_by_eps = {}
            for p in r['perturbations']:
                e = p['eps']
                ratios_by_eps.setdefault(e, []).append(p['ratio'])
            mean_ratios = {e: np.mean(v) for e, v in ratios_by_eps.items()}
            eps_sorted = sorted(mean_ratios)
            ratio_vals = [mean_ratios[e] for e in eps_sorted]
            decreasing = all(ratio_vals[i] >= ratio_vals[i+1]
                             for i in range(len(ratio_vals)-1))
            print(f"\n  {r['label']}: ratios by eps = "
                  f"{dict(zip(eps_sorted, [round(x,4) for x in ratio_vals]))}")
            print(f"  Ratio decreasing as eps→0: "
                  f"{'✓ asymptotically equivariant' if decreasing else '✗ not monotone'}")

    Path(args.output).write_text(
        json.dumps({'results': all_results}, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
