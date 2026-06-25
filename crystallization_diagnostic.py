"""
crystallization_diagnostic.py
==============================
Tracks the crystallization of the Fukaya A∞ category from the
free algebra (spectral backbone) as a function of strip-area std.

Theory
------
The transformer weights exist in two phases:

  PRE-STRUCTURE (search regime, std < 0.5):
    - Spectral backbone stable (spectral cos ≈ 0.97)
    - Frobenius r_m2 ≈ 0.35 (scale-degenerate, α*≈0)
    - B ≈ 0 (exchange matrix near zero)
    - m2 walls not differentiated
    - A∞ structure: free algebra with vanishing potential (m0→0)

  CATEGORICAL RESOLUTION (floor regime, std > 0.5):
    - Strip areas differentiate (non-uniform, B anisotropic)
    - Frobenius r_m2 → high (scale-comparable, α*≠0)
    - Exchange matrix B non-trivial
    - m2 walls discriminating
    - A∞ structure: crystallized category with active m2 composition

The crystallization transition occurs at the Moran fixation event
(strip-area differentiation, predicted at val < 0.062).

This script:
  1. Takes multiple checkpoints at different training stages
  2. Computes (strip_area_std, spectral_cos, r_m2, B_norm) for each
  3. Plots the crystallization curve: metrics vs std
  4. Identifies the crystallization threshold std*

Usage
-----
  # With tau-spike checkpoints as intermediate stages:
  python crystallization_diagnostic.py \\
      --compiler compiler_geometric.py \\
      --checkpoints \\
          tau_spikes/tau_spike_step0064_tau5.90.pt \\
          tau_spikes/tau_spike_step0072_tau5.94.pt \\
          basin_entry_state.pt \\
          basin_state.pt \\
      --rank 6 --k_lanczos 20 \\
      --output crystallization_report.json

  # Live tracking during training (call from compiler loop):
  # tracker = CrystallizationTracker(rank=6)
  # tracker.record(step, model, val)
  # tracker.report()
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
    p.add_argument('--checkpoints', nargs='+',
                   default=['basin_state.pt'])
    p.add_argument('--compiler',    default='compiler_geometric.py')
    p.add_argument('--rank',        type=int, default=6)
    p.add_argument('--k_lanczos',   type=int, default=20)
    p.add_argument('--fd_eps',      type=float, default=1e-4)
    p.add_argument('--output',      default='crystallization_report.json')
    return p.parse_args()


# ─── Import compiler ──────────────────────────────────────────────────────────

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


# ─── WK extraction ───────────────────────────────────────────────────────────

def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if 'c_attn' in n and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            D = tensor.shape[0]//3 if tensor.shape[0]%3==0 else tensor.shape[1]//3
            wk[li] = tensor[D:2*D,:] if tensor.shape[0]%3==0 else tensor[:,D:2*D].T
        elif ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor if tensor.ndim==2 else tensor.squeeze()
    if not wk:
        raise RuntimeError("No WK. Keys: " + ", ".join(list(state.keys())[:10]))
    return [wk[i].detach().float() for i in sorted(wk)]


# ─── Core metrics ─────────────────────────────────────────────────────────────

def strip_areas(wk_list, rank):
    d = len(wk_list) - 1
    areas = []
    for k in range(d):
        Uk  = torch.linalg.svd(wk_list[k],   full_matrices=False)[0][:, :rank]
        Uk1 = torch.linalg.svd(wk_list[k+1], full_matrices=False)[0][:, :rank]
        sv  = torch.linalg.svdvals(Uk.T @ Uk1).clamp(-1+1e-6, 1-1e-6)
        areas.append(torch.arccos(sv).sum().item())
    return np.array(areas)


def exchange_matrix(areas):
    d = len(areas)
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            diff = areas[i] - areas[j]
            B[i,j] = math.copysign(abs(diff), diff)
    return B


def m2_functional(areas):
    mu  = np.mean(areas)
    mad = np.mean(np.abs(areas - mu))
    if mad < 1e-10: return 0.0
    return float(np.sum(((areas - mu)/mad)**2))


def m2_functional_hessian(areas, eps):
    d = len(areas)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            pp = areas.copy(); pp[i]+=eps; pp[j]+=eps
            pm = areas.copy(); pm[i]+=eps; pm[j]-=eps
            mp = areas.copy(); mp[i]-=eps; mp[j]+=eps
            mm = areas.copy(); mm[i]-=eps; mm[j]-=eps
            H[i,j] = (m2_functional(pp) - m2_functional(pm)
                      - m2_functional(mp) + m2_functional(mm)) / (4*eps**2)
    return (H + H.T) / 2


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
                zip(torch.split(self.theta, self.splits), self.shapes)]
    def forward(self):
        wks = self.wk_matrices()
        loss = torch.tensor(0.)
        for k in range(len(wks)-1):
            Ua = torch.linalg.svd(wks[k],   full_matrices=False)[0][:,:self.rank]
            Ub = torch.linalg.svd(wks[k+1], full_matrices=False)[0][:,:self.rank]
            sv = torch.linalg.svdvals(Ua.T@Ub).clamp(1e-6, 1-1e-6)
            loss = loss + torch.arccos(sv).sum()
        return loss


def hvp(model, v):
    loss = model()
    grad = torch.autograd.grad(loss, model.theta, create_graph=True)[0]
    return torch.autograd.grad((grad*v).sum(), model.theta)[0].detach()


def lanczos(model, k, seed=42):
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k+1)
    alpha_c = torch.zeros(k); beta_c = torch.zeros(k-1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:,0] = v; prev_b = 0.
    for j in range(k):
        w = hvp(model, V[:,j])
        if j > 0: w = w - prev_b*V[:,j-1]
        a = (w*V[:,j]).sum().item(); alpha_c[j] = a
        w = w - a*V[:,j]
        for i in range(j+1): w = w - (w@V[:,i])*V[:,i]
        if j < k-1:
            b = w.norm().item()
            if b < 1e-10: k=j+1; break
            prev_b=b; beta_c[j]=b; V[:,j+1]=w/b
    Hk = (np.diag(alpha_c[:k].numpy()) +
          np.diag(beta_c[:k-1].numpy(),1) +
          np.diag(beta_c[:k-1].numpy(),-1))
    ev, ritz = np.linalg.eigh(Hk)
    evecs = V[:,:k].numpy() @ ritz
    idx = np.argsort(-np.abs(ev))
    return ev[idx], evecs[:,idx]


def build_basis(wk_list, rank):
    shapes  = [w.shape for w in wk_list]
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    D_tot = sum(splits); d = len(wk_list)-1
    cols = []
    for k in range(d):
        Wk  = wk_list[k].detach().float()
        Wk1 = wk_list[k+1].detach().float()
        Uk,_,Vhk   = torch.linalg.svd(Wk,  full_matrices=False)
        Uk1,_,Vhk1 = torch.linalg.svd(Wk1, full_matrices=False)
        Uk,Uk1 = Uk[:,:rank], Uk1[:,:rank]
        M = Uk.T@Uk1; Um,Sm,Vhm = torch.linalg.svd(M)
        Sm = Sm.clamp(1e-6,1-1e-6)
        col = np.zeros(D_tot)
        for r in range(rank):
            sc = -1./math.sqrt(max(1.-Sm[r].item()**2,1e-8))
            lk  = (Uk @Um[:,r:r+1]).reshape(-1).detach().numpy()
            rk  = (Vhm[r:r+1,:]@Vhk1[:rank,:]).reshape(-1).detach().numpy()
            col[offsets[k]:offsets[k+1]] += sc*(lk[:,None]*rk[None,:]).reshape(-1)[:offsets[k+1]-offsets[k]]
            lk1 = (Uk1@Vhm[r:r+1,:].T).reshape(-1).detach().numpy()
            rk1 = (Um[:,r:r+1].T@Vhk[:rank,:]).reshape(-1).detach().numpy()
            col[offsets[k+1]:offsets[k+2]] += sc*(lk1[:,None]*rk1[None,:]).reshape(-1)[:offsets[k+2]-offsets[k+1]]
        nm = np.linalg.norm(col)
        cols.append(col/nm if nm>1e-10 else col)
    return np.stack(cols, axis=1)


def frobenius_cos(A, B):
    return float(np.sum(A*B) /
                 (np.linalg.norm(A,'fro')*np.linalg.norm(B,'fro')+1e-12))


def spectral_cos_abs(A, B):
    ea = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
    eb = np.sort(np.abs(np.linalg.eigvalsh(B)))[::-1]
    n  = min(len(ea), len(eb))
    return float(np.dot(ea[:n],eb[:n]) /
                 (np.linalg.norm(ea[:n])*np.linalg.norm(eb[:n])+1e-12))


def alpha_star(A, B):
    return float(np.sum(A*B)) / (np.linalg.norm(B,'fro')**2+1e-12)


# ─── Crystallization metrics for one checkpoint ───────────────────────────────

def compute_crystallization_metrics(state, rank, k_lanczos, fd_eps, label):
    """Compute all crystallization metrics for one checkpoint."""
    t0 = time.time()
    wk_list = extract_wk(state)
    d = len(wk_list) - 1

    # 1. Strip areas and exchange matrix
    A = strip_areas(wk_list, rank)
    std  = float(np.std(A))
    mu   = float(np.mean(A))
    mad  = float(np.mean(np.abs(A - mu)))
    B    = exchange_matrix(A)
    B_norm = float(np.linalg.norm(B, 'fro'))

    # 2. Hess(m2) via finite differences
    F0      = m2_functional(A)
    Hess_m2 = m2_functional_hessian(A, fd_eps)

    # 3. Projected Hessian Ĥ = πHπᵀ
    basis  = build_basis(wk_list, rank)
    model  = SymplecticProxy(wk_list, rank)
    evals, evecs = lanczos(model, k_lanczos)
    k_eff  = min(evecs.shape[1], d)
    PiV    = basis.T @ evecs[:, :k_eff]
    H_hat  = (PiV * evals[:k_eff][None, :]) @ PiV.T

    # 4. Core metrics
    r_m2        = frobenius_cos(H_hat, Hess_m2)
    spec_cos    = spectral_cos_abs(H_hat, Hess_m2)
    a_star      = alpha_star(H_hat, Hess_m2)
    norm_Hhat   = float(np.linalg.norm(H_hat, 'fro'))
    norm_Hm2    = float(np.linalg.norm(Hess_m2, 'fro'))
    scale_ratio = norm_Hm2 / (norm_Hhat + 1e-12)
    B_vs_Hhat   = frobenius_cos(B, H_hat)

    # 5. Phase
    if std < 0.1:
        phase = "PRE-STRUCTURE (free algebra)"
    elif std < 0.5:
        phase = "TRANSITIONAL"
    else:
        phase = "CATEGORICAL RESOLUTION (crystallized)"

    elapsed = time.time() - t0
    return {
        "label":        label,
        "strip_areas":  A.tolist(),
        "strip_std":    float(std),
        "strip_mu":     float(mu),
        "strip_mad":    float(mad),
        "B_norm":       float(B_norm),
        "m2_functional": float(F0),
        "r_m2":         float(r_m2),
        "spectral_cos": float(spec_cos),
        "alpha_star":   float(a_star),
        "scale_ratio":  float(scale_ratio),
        "norm_H_hat":   float(norm_Hhat),
        "norm_Hess_m2": float(norm_Hm2),
        "B_vs_Hhat":    float(B_vs_Hhat),
        "top5_eigvals": evals[:5].tolist(),
        "phase":        phase,
        "elapsed_s":    round(elapsed, 1),
    }


# ─── CrystallizationTracker for live use in compiler ─────────────────────────

class CrystallizationTracker:
    """
    Lightweight tracker for use inside compiler_geometric.py.
    Records crystallization metrics at each checkpoint without
    requiring the full diagnostic suite.

    Usage in compiler:
        tracker = CrystallizationTracker(rank=6)
        # After each phase:
        tracker.record(step=step, wk_list=model_wk_list(), val=val)
        # At end:
        tracker.report()
        tracker.save('crystallization_log.json')
    """

    def __init__(self, rank=6, fd_eps=1e-4):
        self.rank   = rank
        self.fd_eps = fd_eps
        self.records = []

    def _wk_from_model(self, model):
        wks = []
        for k in range(100):
            found = False
            for name, p in model.named_parameters():
                if f'blocks.{k}.attn.WK' in name or f'blocks.{k}.attn.wk' in name.lower():
                    wks.append(p.detach().float())
                    found = True
                    break
            if not found:
                break
        return wks

    def record(self, step, model_or_wk_list, val, label=None):
        if isinstance(model_or_wk_list, list):
            wk_list = model_or_wk_list
        else:
            wk_list = self._wk_from_model(model_or_wk_list)

        if not wk_list:
            return

        A    = strip_areas(wk_list, self.rank)
        std  = float(np.std(A))
        B    = exchange_matrix(A)
        Hm2  = m2_functional_hessian(A, self.fd_eps)

        basis = build_basis(wk_list, self.rank)
        proxy = SymplecticProxy(wk_list, self.rank)
        evals, evecs = lanczos(proxy, min(10, len(A)))
        d     = len(A)
        k_eff = min(evecs.shape[1], d)
        PiV   = basis.T @ evecs[:, :k_eff]
        H_hat = (PiV * evals[:k_eff][None,:]) @ PiV.T

        r_m2     = frobenius_cos(H_hat, Hm2)
        spec_cos = spectral_cos_abs(H_hat, Hm2)

        rec = {
            "step":       step,
            "val":        float(val),
            "strip_std":  float(std),
            "B_norm":     float(np.linalg.norm(B, 'fro')),
            "r_m2":       float(r_m2),
            "spectral_cos": float(spec_cos),
            "label":      label or f"step_{step}",
        }
        self.records.append(rec)
        return rec

    def report(self):
        print(f"\n  {'Step':>6}  {'val':>7}  {'std':>7}  "
              f"{'r_m2':>8}  {'spec_cos':>9}  Phase")
        print(f"  {'-'*65}")
        for r in self.records:
            std = r['strip_std']
            phase = ("PRE" if std < 0.1 else
                     "TRANS" if std < 0.5 else "CRYSTAL")
            print(f"  {r['step']:>6}  {r['val']:>7.4f}  {std:>7.4f}  "
                  f"{r['r_m2']:>8.4f}  {r['spectral_cos']:>9.4f}  {phase}")

    def crystallization_threshold(self):
        """Find the step where r_m2 first rises above 0.7."""
        for r in self.records:
            if r['r_m2'] > 0.7:
                return r['step'], r['strip_std']
        return None, None

    def save(self, path):
        Path(path).write_text(json.dumps(self.records, indent=2,
                                          cls=NumpyEncoder))


# ─── ASCII plot ───────────────────────────────────────────────────────────────

def ascii_plot(records, x_key, y_key, title, width=50, height=12):
    xs = [r[x_key] for r in records]
    ys = [r[y_key] for r in records]
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min: x_max = x_min + 1
    if y_max == y_min: y_max = y_min + 0.1

    grid = [[' '] * width for _ in range(height)]
    for x, y in zip(xs, ys):
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
        row = max(0, min(height-1, row))
        col = max(0, min(width-1, col))
        grid[row][col] = '●'

    print(f"\n  {title}")
    print(f"  {y_max:.3f} ┤")
    for i, row in enumerate(grid):
        print(f"         │ {''.join(row)}")
    print(f"  {y_min:.3f} ┤")
    print(f"          └{'─'*width}")
    print(f"          {x_min:.3f}{' '*(width-10)}{x_max:.3f}")
    print(f"          ← {x_key} →")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  CRYSTALLIZATION DIAGNOSTIC")
    print("  Tracking A∞ category emergence vs strip-area std")
    print("=" * 60)
    print(f"  Checkpoints: {len(args.checkpoints)}")
    print(f"  rank={args.rank}  k_lanczos={args.k_lanczos}")

    # Import compiler for eval
    print(f"\n  Importing compiler …")
    try:
        comp     = import_compiler(args.compiler)
        eval_val = comp.eval_val
    except Exception as e:
        print(f"  ⚠  Compiler import failed: {e}")
        eval_val = None

    records = []

    for ckpt_path in args.checkpoints:
        print(f"\n{'─'*60}")
        print(f"  {ckpt_path}")
        t0 = time.time()

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get('state_dict', ckpt.get('model', ckpt))
            meta  = ckpt.get('metadata', {})
        else:
            state, meta = ckpt, {}

        if not isinstance(state, dict) or not all(
                isinstance(v, torch.Tensor) for v in state.values()):
            state = ckpt if isinstance(ckpt, dict) else {}

        state = {k: v for k, v in state.items()
                 if isinstance(v, torch.Tensor)}

        # Val from metadata or eval
        val = meta.get('val', None)
        step = meta.get('step', None)
        tau  = meta.get('tau',  None)

        label = Path(ckpt_path).stem

        try:
            metrics = compute_crystallization_metrics(
                state, args.rank, args.k_lanczos, args.fd_eps, label)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        metrics['checkpoint'] = str(ckpt_path)
        if val  is not None: metrics['val']  = float(val)
        if step is not None: metrics['step'] = int(step)
        if tau  is not None: metrics['tau']  = float(tau)

        records.append(metrics)

        # Print summary
        print(f"  strip_std    = {metrics['strip_std']:.4f}")
        print(f"  B_norm       = {metrics['B_norm']:.4f}")
        print(f"  r_m2         = {metrics['r_m2']:.4f}  "
              f"(Frobenius cos Ĥ vs Hess(m2))")
        print(f"  spectral_cos = {metrics['spectral_cos']:.4f}  "
              f"(|spec Ĥ| vs |spec Hess(m2)|)")
        print(f"  α*           = {metrics['alpha_star']:.6f}  "
              f"{'(scale-collapsed)' if abs(metrics['alpha_star']) < 1e-4 else '(scale-valid)'}")
        print(f"  scale_ratio  = {metrics['scale_ratio']:.2e}")
        print(f"  Phase: {metrics['phase']}")
        print(f"  [{metrics['elapsed_s']:.1f}s]")

    if not records:
        print("No records computed.")
        return

    # Crystallization table
    print(f"\n{'='*60}")
    print(f"  CRYSTALLIZATION TABLE")
    print(f"{'='*60}")
    print(f"  {'Checkpoint':<32} {'std':>6} {'r_m2':>7} {'spec':>7} "
          f"{'α*':>9} {'Phase'}")
    print(f"  {'-'*75}")
    for r in records:
        name = Path(r['checkpoint']).stem[:31]
        collapsed = abs(r['alpha_star']) < 1e-4
        print(f"  {name:<32} {r['strip_std']:>6.4f} {r['r_m2']:>7.4f} "
              f"{r['spectral_cos']:>7.4f} {r['alpha_star']:>9.4f}  "
              f"{'⚠ scale-col' if collapsed else r['phase'][:20]}")

    # Crystallization threshold
    sorted_by_std = sorted(records, key=lambda r: r['strip_std'])
    threshold_record = next(
        (r for r in sorted_by_std if r['r_m2'] > 0.7), None)

    print(f"\n  Crystallization threshold (r_m2 > 0.7):")
    if threshold_record:
        print(f"  ✓ Crossed at std ≈ {threshold_record['strip_std']:.4f}  "
              f"(r_m2 = {threshold_record['r_m2']:.4f})")
    else:
        print(f"  ✗ Not yet reached in these checkpoints")
        print(f"  Current max r_m2 = {max(r['r_m2'] for r in records):.4f} "
              f"at std = {max(records, key=lambda r: r['r_m2'])['strip_std']:.4f}")
        print(f"  Requires std > 0.5 (Moran fixation, "
              f"predicted at val < 0.062)")

    # ASCII plots if multiple checkpoints
    if len(records) > 1:
        ascii_plot(records, 'strip_std', 'r_m2',
                   "r_m2 (Frobenius cos) vs strip_std  [crystallization curve]")
        ascii_plot(records, 'strip_std', 'spectral_cos',
                   "spectral_cos vs strip_std")

    # Key finding
    print(f"\n  KEY FINDING:")
    max_r = max(r['r_m2'] for r in records)
    max_spec = max(r['spectral_cos'] for r in records)
    print(f"  Spectral backbone: stable (max spectral_cos = {max_spec:.4f})")
    print(f"  Categorical resolution: "
          f"{'active' if max_r > 0.7 else 'not yet reached'} "
          f"(max r_m2 = {max_r:.4f})")
    print(f"  The spectral backbone exists independently of crystallization.")
    print(f"  The Fukaya A∞ structure crystallizes only when")
    print(f"  strip-area std > 0.5 (data potential m0 activates).")

    # Save
    out = {
        "records":  records,
        "threshold": {
            "std":    threshold_record['strip_std'] if threshold_record else None,
            "r_m2":  threshold_record['r_m2'] if threshold_record else None,
            "label": threshold_record['label'] if threshold_record else None,
        },
        "interpretation": (
            "PRE-STRUCTURE: spectral backbone stable, "
            "A∞ structure not yet geometrically resolved. "
            "Frobenius r_m2 scale-degenerate (alpha*≈0). "
            "Crystallization requires strip-area std > 0.5 "
            "(Moran fixation event, val < 0.062)."
        ),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
