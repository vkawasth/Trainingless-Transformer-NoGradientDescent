"""
three_geometry_correlations.py
===============================
Priority 3: Relate the three geometric objects:
  (A) Symplectic Hessian (Hessian of strip-area functional)
  (B) Loss Hessian (Hessian of V(u) = L(w₀ + Pu) restricted to null space)
  (C) Transfer-matrix spectral geometry (Im(λ), collision scale, phase tangency)

Three correlations:

  Test 1: Symplectic curvature vs eigenvalue collision scale
    For each null eigenvector vᵢ, measure:
      - |λ_symp(vᵢ)|: symplectic curvature (from Lanczos)
      - ε_crit(vᵢ): scale at which Im(λ(w+εvᵢ)) first becomes nonzero
    Question: Do more-null symplectic directions have larger ε_crit?
    (More symplectically flat → more collision-resistant?)

  Test 2: Phase-tangent directions vs loss Hessian curvature
    At step 72, vᵢ classified as E_tan (dφ/dv≈0) or E_cross (dφ/dv≠0).
    Measure loss Hessian curvature in each direction: d²V/dvᵢ².
    Question: Do E_tan directions have smaller loss curvature than E_cross?

  Test 3: Loss gradient vs real-breaking directions
    The loss gradient ∇L at step 64 points toward lower loss.
    The 16% real-breaking directions (from jet_order_test) can break Im(λ)=0.
    Question: Does ∇L align with the real-breaking subspace?
    (Does gradient descent push the trajectory off Σ_R?)

Usage
-----
  python three_geometry_correlations.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --spike72 tau_spikes/tau_spike_step0072_tau5.94.pt \\
      --compiler compiler_geometric.py \\
      --rank 6 --k_lanczos 20 --n_null 5 \\
      --n_random 50 --n_eval 32 --fd_eps 0.05 \\
      --output three_geometry_report.json
"""

import argparse, json, math, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64',   default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--spike72',   default='tau_spikes/tau_spike_step0072_tau5.94.pt')
    p.add_argument('--compiler',  default='compiler_geometric.py')
    p.add_argument('--rank',      type=int, default=6)
    p.add_argument('--k_lanczos', type=int, default=20)
    p.add_argument('--n_null',    type=int, default=5)
    p.add_argument('--n_random',  type=int, default=50,
                   help='Random directions for Test 3')
    p.add_argument('--n_eval',    type=int, default=32)
    p.add_argument('--fd_eps',    type=float, default=0.05)
    p.add_argument('--k_pair',    type=int, default=0)
    p.add_argument('--output',    default='three_geometry_report.json')
    return p.parse_args()


# ─── Shared infrastructure ────────────────────────────────────────────────────

def import_compiler(path):
    globs = {'__name__':'__compiler__','__file__':path,'__builtins__':__builtins__}
    src = open(path).read()
    src = src.replace('if __name__ == "__main__":','if False:')
    src = src.replace("if __name__ == '__main__':","if False:")
    import re
    src = re.sub(r'torch\.save\(','_NOSAVE(',src)
    globs['_NOSAVE'] = lambda *a,**kw: None
    exec(compile(src,path,'exec'),globs)
    return types.SimpleNamespace(**{k:v for k,v in globs.items()
                                    if not k.startswith('__')})


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
    def forward(self):
        wks = [p.reshape(s) for p,s in
               zip(torch.split(self.theta,self.splits),self.shapes)]
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


def eval_loss_at(state, wk_pairs, v_flat, eps, LM, get_batch, n_eval):
    """L(w₀ + eps·v)"""
    state_new = {n: t.clone() for n, t in state.items()}
    splits  = [w.numel() for _, w in wk_pairs]
    offsets = list(np.cumsum([0]+splits))
    for i, (name, _) in enumerate(wk_pairs):
        s, e = offsets[i], offsets[i+1]
        delta = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(
            state_new[name].shape)
        state_new[name] = state_new[name] + eps * delta
    model = LM()
    model.load_state_dict(state_new, strict=False)
    return eval_loss(model, get_batch, n_eval)


def im_lambda_at(wk_pairs, v_flat, eps, k):
    """Im(λ_dom(M(w₀+εv)))"""
    splits  = [w.numel() for _, w in wk_pairs]
    offsets = list(np.cumsum([0]+splits))
    wk_new = [w.clone() for _, w in wk_pairs]
    for i, (_, w) in enumerate(wk_pairs):
        s, e = offsets[i], offsets[i+1]
        delta = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
        wk_new[i] = w + eps * delta
    M = wk_new[k+1].numpy() @ np.linalg.pinv(wk_new[k].numpy())
    evals = np.linalg.eigvals(M)
    return float(evals[np.argmax(np.abs(evals.real))].imag)


def phase_deriv(wk_pairs, v_flat, k, fd_eps=0.01):
    """dφ_k/dv via central FD."""
    def phi(eps):
        splits  = [w.numel() for _, w in wk_pairs]
        offsets = list(np.cumsum([0]+splits))
        wk_new = [w.clone() for _, w in wk_pairs]
        for i, (_, w) in enumerate(wk_pairs):
            s, e = offsets[i], offsets[i+1]
            delta = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
            wk_new[i] = w + eps * delta
        M = wk_new[k+1].numpy() @ np.linalg.pinv(wk_new[k].numpy())
        evals = np.linalg.eigvals(M)
        lam = evals[np.argmax(np.abs(evals.real))]
        p = float(np.arctan2(lam.imag, lam.real))
        return p
    dp = (phi(+fd_eps) - phi(-fd_eps)) / (2*fd_eps)
    if abs(dp) > math.pi: dp -= math.copysign(2*math.pi, dp)
    return dp


# ─── Test 1: Symplectic curvature vs collision scale ─────────────────────────

def test1_symp_vs_collision(wk_pairs, null_evals, null_evecs, k, label):
    """
    For each null eigenvector vᵢ:
      |λ_symp|: symplectic curvature (smaller = more null)
      ε_crit: first ε where Im(λ(w+εv)) ≠ 0 (larger = more collision-resistant)
    Correlation: Spearman ρ(|λ_symp|, ε_crit)
    """
    D = sum(w.numel() for _, w in wk_pairs)
    eps_vals = np.arange(0.02, 0.52, 0.02)

    print(f"\n  Test 1 ({label}): Symplectic curvature vs collision scale")
    print(f"  {'vᵢ':>4} {'|λ_symp|':>10} {'ε_crit':>8} {'type'}")
    print(f"  {'-'*38}")

    symp_curvs, eps_crits = [], []
    for i in range(null_evecs.shape[1]):
        v = null_evecs[:, i]
        lam_symp = float(abs(null_evals[i]))

        # Find ε_crit: first ε where |Im(λ)| > 0.5
        eps_crit = None
        for eps in eps_vals:
            im = abs(im_lambda_at(wk_pairs, v, eps, k))
            if im > 0.5:
                eps_crit = float(eps)
                break

        eps_str = f"{eps_crit:.3f}" if eps_crit else ">0.5"
        etype = "A(stay real)" if eps_crit is None else f"B/C(crit@{eps_str})"
        print(f"  v{i:>3} {lam_symp:>10.4f} {eps_str:>8} {etype}")

        symp_curvs.append(lam_symp)
        eps_crits.append(eps_crit if eps_crit else 0.52)

    symp_arr  = np.array(symp_curvs)
    eps_arr   = np.array(eps_crits)
    rho, pval = spearmanr(symp_arr, eps_arr)

    print(f"\n  Spearman ρ(|λ_symp|, ε_crit) = {rho:.4f}  p={pval:.3f}")
    if rho < -0.5:
        interp = "ANTI-CORRELATED: more-null directions hit collision earlier"
    elif rho > 0.5:
        interp = "CORRELATED: more-null directions are more collision-resistant"
    else:
        interp = f"WEAK/NO correlation (ρ={rho:.2f})"
    print(f"  {interp}")

    return {
        'symp_curvatures': symp_curvs,
        'eps_crits': eps_crits,
        'spearman_rho': float(rho),
        'spearman_p': float(pval),
        'interpretation': interp,
    }


# ─── Test 2: Phase tangency vs loss Hessian curvature ────────────────────────

def test2_phase_tangency_vs_loss_hessian(state72, wk_pairs72, null_evals72,
                                          null_evecs72, k, LM, get_batch,
                                          n_eval, fd_eps, label):
    """
    At step 72: for each null eigenvector vᵢ:
      dφ/dv: phase derivative (≈0 → E_tan, ≠0 → E_cross)
      d²L/dv²: loss Hessian curvature in direction vᵢ
    Question: Do E_tan directions have smaller loss curvature?
    """
    D = sum(w.numel() for _, w in wk_pairs72)
    n = null_evecs72.shape[1]

    print(f"\n  Test 2 ({label}): Phase tangency vs loss Hessian curvature")
    print(f"  {'vᵢ':>4} {'|dφ/dv|':>10} {'d²L/dv²':>12} {'type':>10}")
    print(f"  {'-'*42}")

    dphis, d2Ls = [], []
    for i in range(n):
        v = null_evecs72[:, i]

        # Phase derivative
        dphi = abs(phase_deriv(wk_pairs72, v, k, fd_eps=0.01))

        # Loss Hessian curvature: (L(+εv) + L(-εv) - 2L(0)) / ε²
        L0  = eval_loss_at(state72, wk_pairs72, np.zeros(D), 1.,
                           LM, get_batch, n_eval)
        Lp  = eval_loss_at(state72, wk_pairs72, v, +fd_eps, LM, get_batch, n_eval)
        Lm  = eval_loss_at(state72, wk_pairs72, v, -fd_eps, LM, get_batch, n_eval)
        d2L = (Lp + Lm - 2*L0) / (fd_eps**2)

        etype = "E_tan" if dphi < 0.1 else "E_cross"
        print(f"  v{i:>3} {dphi:>10.4f} {d2L:>+12.4f} {etype:>10}")
        dphis.append(dphi)
        d2Ls.append(d2L)

    dphi_arr = np.array(dphis)
    d2L_arr  = np.array(d2Ls)
    rho, pval = spearmanr(dphi_arr, d2L_arr)

    tan_mask   = dphi_arr < 0.1
    cross_mask = dphi_arr >= 0.1

    print(f"\n  E_tan  mean d²L/dv²: {d2L_arr[tan_mask].mean():.4f}" if tan_mask.any() else "")
    print(f"  E_cross mean d²L/dv²: {d2L_arr[cross_mask].mean():.4f}" if cross_mask.any() else "")
    print(f"  Spearman ρ(|dφ/dv|, d²L/dv²) = {rho:.4f}  p={pval:.3f}")

    if rho > 0.5:
        interp = "CORRELATED: larger phase sensitivity → more curved loss"
    elif rho < -0.5:
        interp = "ANTI-CORRELATED: larger phase sensitivity → flatter loss"
    else:
        interp = f"WEAK/NO correlation (ρ={rho:.2f})"
    print(f"  {interp}")

    return {
        'dphis': dphis, 'd2Ls': d2Ls,
        'tan_mean_d2L': float(d2L_arr[tan_mask].mean()) if tan_mask.any() else None,
        'cross_mean_d2L': float(d2L_arr[cross_mask].mean()) if cross_mask.any() else None,
        'spearman_rho': float(rho), 'spearman_p': float(pval),
        'interpretation': interp,
    }


# ─── Test 3: Loss gradient vs real-breaking subspace ─────────────────────────

def test3_gradient_vs_real_breaking(state64, wk_pairs64, k,
                                     LM, get_batch, n_eval, fd_eps,
                                     n_random, label):
    """
    At step 64:
      - Estimate ∇L in the WK block using FD on n_random basis vectors
      - For each basis vector eᵢ, compute Im(ℓᵀ·dM·r): real-breaking component
      - Correlation: ρ(∂L/∂eᵢ, |Im(ℓᵀ·dM·r)|)
    Question: Does the loss gradient point into the real-breaking subspace?
    """
    D = sum(w.numel() for _, w in wk_pairs64)
    splits  = [w.numel() for _, w in wk_pairs64]
    offsets = list(np.cumsum([0]+splits))
    s0, e1  = offsets[k], offsets[min(k+2, len(wk_pairs64))]
    block_size = e1 - s0

    # Dominant eigenvectors at step 64
    wk_list64 = [w for _, w in wk_pairs64]
    M0 = wk_list64[k+1].numpy() @ np.linalg.pinv(wk_list64[k].numpy())
    evals0, evecs0 = np.linalg.eig(M0)
    dom = np.argmax(np.abs(evals0.real))
    r0  = evecs0[:, dom].real
    evals_l, evecs_l = np.linalg.eig(M0.T)
    dom_l = np.argmin(np.abs(evals_l - evals0[dom]))
    ell0  = evecs_l[:, dom_l].real
    norm  = ell0 @ r0
    if abs(norm) > 1e-10: ell0 /= norm

    print(f"\n  Test 3 ({label}): Loss gradient vs real-breaking subspace")
    print(f"  Using {n_random} random directions in WK[{k}]+WK[{k+1}] block")

    rng = np.random.default_rng(77)
    grad_comps  = []   # ∂L/∂v for each random direction
    im_comps    = []   # |Im(ℓᵀ·dM·r)| for each direction

    for trial in range(n_random):
        v = np.zeros(D)
        v_block = rng.standard_normal(block_size)
        v_block /= np.linalg.norm(v_block) + 1e-10
        v[s0:e1] = v_block

        # Loss gradient component: (L(+ε) - L(-ε)) / 2ε
        Lp = eval_loss_at(state64, wk_pairs64, v, +fd_eps, LM, get_batch, n_eval)
        Lm = eval_loss_at(state64, wk_pairs64, v, -fd_eps, LM, get_batch, n_eval)
        dL = (Lp - Lm) / (2*fd_eps)

        # Real-breaking component: Im(ℓᵀ·dM·r) via FD on M
        eps_M = 1e-4
        splits_wk = [w.numel() for w in wk_list64]
        offsets_wk = list(np.cumsum([0]+splits_wk))
        wk_p = [w.clone() for w in wk_list64]
        wk_m = [w.clone() for w in wk_list64]
        for i, w in enumerate(wk_list64):
            ss, ee = offsets_wk[i], offsets_wk[i+1]
            # map from full v to wk-only v
            full_s = offsets[i] if i < len(offsets)-1 else 0
            dv = torch.tensor(v[full_s:full_s+w.numel()],
                              dtype=torch.float32).reshape(w.shape)
            wk_p[i] = w + eps_M * dv
            wk_m[i] = w - eps_M * dv
        Mp = wk_p[k+1].numpy() @ np.linalg.pinv(wk_p[k].numpy())
        Mm = wk_m[k+1].numpy() @ np.linalg.pinv(wk_m[k].numpy())
        dM = (Mp - Mm) / (2*eps_M)
        im_dlam = float(abs((ell0.conj() @ dM @ r0).imag))

        grad_comps.append(dL)
        im_comps.append(im_dlam)

    grad_arr = np.array(grad_comps)
    im_arr   = np.array(im_comps)

    rho, pval = spearmanr(np.abs(grad_arr), im_arr)

    # Fraction of large-gradient directions that are real-breaking
    high_grad = np.abs(grad_arr) > np.percentile(np.abs(grad_arr), 75)
    frac_breaking_high = float((im_arr[high_grad] > 0.01).mean())
    frac_breaking_all  = float((im_arr > 0.01).mean())

    print(f"  Mean |∂L/∂v|: {np.abs(grad_arr).mean():.4f}")
    print(f"  Mean |Im(δλ)|: {im_arr.mean():.6f}")
    print(f"  Spearman ρ(|∂L/∂v|, |Im(δλ)|) = {rho:.4f}  p={pval:.3f}")
    print(f"  Frac real-breaking (all dirs): {frac_breaking_all:.3f}")
    print(f"  Frac real-breaking (high-|grad| dirs): {frac_breaking_high:.3f}")

    if frac_breaking_high > frac_breaking_all + 0.1 and rho > 0.1:
        interp = "GRADIENT ALIGNS WITH REAL-BREAKING: gradient flow pushes off Σ_R"
    elif frac_breaking_high < frac_breaking_all - 0.1:
        interp = "GRADIENT AVOIDS REAL-BREAKING: gradient flow stays on Σ_R"
    elif im_arr.mean() < 1e-5:
        interp = "ALL DIRECTIONS REAL-PRESERVING (algebraic locking dominates)"
    else:
        interp = f"MIXED/NO ALIGNMENT (ρ={rho:.2f})"
    print(f"  {interp}")

    return {
        'grad_components': grad_comps[:20],
        'im_components': im_comps[:20],
        'mean_abs_grad': float(np.abs(grad_arr).mean()),
        'mean_im_dlam': float(im_arr.mean()),
        'spearman_rho': float(rho),
        'spearman_p': float(pval),
        'frac_breaking_all': frac_breaking_all,
        'frac_breaking_high_grad': frac_breaking_high,
        'interpretation': interp,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("="*60)
    print("  THREE GEOMETRY CORRELATIONS")
    print("  Symplectic ↔ Loss ↔ Spectral")
    print("="*60)

    comp = import_compiler(args.compiler)
    LM, get_batch = comp.LM, comp.get_batch

    # Load both checkpoints
    state64 = load_state(args.spike64)
    state72 = load_state(args.spike72)
    wk_pairs64 = extract_wk(state64)
    wk_pairs72 = extract_wk(state72)
    wk_list64  = [w for _, w in wk_pairs64]
    wk_list72  = [w for _, w in wk_pairs72]

    # Null eigenvectors at both checkpoints
    print(f"\n  Computing null eigenvectors …")
    model64 = SymplecticProxy(wk_list64, args.rank)
    evals64, evecs64 = lanczos_null(model64, args.k_lanczos, args.n_null)
    model72 = SymplecticProxy(wk_list72, args.rank)
    evals72, evecs72 = lanczos_null(model72, args.k_lanczos, args.n_null)
    print(f"  Step 64 null λ: {', '.join(f'{e:.3f}' for e in evals64)}")
    print(f"  Step 72 null λ: {', '.join(f'{e:.3f}' for e in evals72)}")

    # ── Test 1 ────────────────────────────────────────────────────────────────
    t1 = test1_symp_vs_collision(
        wk_pairs64, evals64, evecs64, args.k_pair,
        "step64")

    # ── Test 2 ────────────────────────────────────────────────────────────────
    t2 = test2_phase_tangency_vs_loss_hessian(
        state72, wk_pairs72, evals72, evecs72, args.k_pair,
        LM, get_batch, args.n_eval, args.fd_eps,
        "step72")

    # ── Test 3 ────────────────────────────────────────────────────────────────
    t3 = test3_gradient_vs_real_breaking(
        state64, wk_pairs64, args.k_pair,
        LM, get_batch, args.n_eval, args.fd_eps,
        args.n_random, "step64")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  THREE GEOMETRY SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Test 1 (Symplectic curvature vs collision scale):")
    print(f"    Spearman ρ = {t1['spearman_rho']:+.4f}  p={t1['spearman_p']:.3f}")
    print(f"    {t1['interpretation']}")

    print(f"\n  Test 2 (Phase tangency vs loss Hessian curvature):")
    print(f"    Spearman ρ = {t2['spearman_rho']:+.4f}  p={t2['spearman_p']:.3f}")
    print(f"    {t2['interpretation']}")
    if t2['tan_mean_d2L'] is not None:
        print(f"    E_tan mean d²L/dv²  = {t2['tan_mean_d2L']:+.4f}")
    if t2['cross_mean_d2L'] is not None:
        print(f"    E_cross mean d²L/dv² = {t2['cross_mean_d2L']:+.4f}")

    print(f"\n  Test 3 (Loss gradient vs real-breaking subspace):")
    print(f"    Spearman ρ = {t3['spearman_rho']:+.4f}  p={t3['spearman_p']:.3f}")
    print(f"    Frac real-breaking (all):       {t3['frac_breaking_all']:.3f}")
    print(f"    Frac real-breaking (high grad): {t3['frac_breaking_high_grad']:.3f}")
    print(f"    {t3['interpretation']}")

    print(f"\n  Central question — are the three geometries related?")
    corrs = [abs(t1['spearman_rho']), abs(t2['spearman_rho']),
             abs(t3['spearman_rho'])]
    if max(corrs) > 0.7:
        print(f"  STRONG correlation found (max |ρ|={max(corrs):.3f}): "
              "geometries are quantitatively linked.")
    elif max(corrs) > 0.4:
        print(f"  MODERATE correlation (max |ρ|={max(corrs):.3f}): "
              "partial geometric alignment.")
    else:
        print(f"  WEAK/NO correlation (max |ρ|={max(corrs):.3f}): "
              "three geometries appear independent.")

    report = {'test1': t1, 'test2': t2, 'test3': t3}
    Path(args.output).write_text(
        json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
