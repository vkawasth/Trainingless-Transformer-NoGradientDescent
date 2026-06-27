"""
tqft_frobenius_diagnostic.py  (v2 — quantitative residuals)
=============================================================
Two diagnostics:

A) Strip-area-weighted r_m2:
   ⟨A,B⟩_σ = Σᵢⱼ Aᵢⱼ Bᵢⱼ / (σᵢ σⱼ)
   Replaces Frobenius in non-uniform regime.
   Ratio weighted/Frobenius ≈ 1 when uniform; > 1 when std > 0.5.

B) Frobenius algebra residuals on the persistent spectral subspace:
   H = span of top-k Lanczos eigenvectors projected to R^d.

   Four quantitative residuals (following Atiyah-Kock classification):
     R_assoc   = ‖μ(μ⊗I) − μ(I⊗μ)‖       (associativity of ∇)
     R_coassoc = ‖(Δ⊗I)Δ − (I⊗Δ)Δ‖       (coassociativity of Δ)
     R_frob    = ‖(∇⊗I)(I⊗Δ) − Δ∘∇‖       (Frobenius relation)
     R_unit    = ‖∇(η⊗I) − id‖             (unit axiom)

   Each is a real number; smaller = more TQFT-like.
   Track how residuals change across training stages.

   Implementation:
     ∇(u,v) = normalize(PiV @ diag(evals) @ PiV^T (u⊗v) projected to H)
            = normalize(H_hat @ (u*v))  [H_hat = projected Hessian]
     Δ(u)   = (H_hat u) ⊗ (H_hat u) / ‖H_hat u‖²
     η      = e₁ (first basis vector of H, unit strip-area direction)

The falsifiable hypothesis (per Atiyah-Kock):
  If residuals decrease monotonically as models become more capable,
  or remain stable under training/pruning/quantization,
  that is strong evidence for an underlying categorical structure.

Usage
-----
  python tqft_frobenius_diagnostic.py \\
      --compiler compiler_geometric.py \\
      --checkpoints \\
          tau_spikes/tau_spike_step0064_tau5.90.pt \\
          tau_spikes/tau_spike_step0072_tau5.94.pt \\
          basin_entry_state.pt \\
          basin_state.pt \\
      --rank 6 --k_lanczos 20 --n_vectors 5 \\
      --output tqft_report.json
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
    p.add_argument('--checkpoints', nargs='+', default=['basin_state.pt'])
    p.add_argument('--compiler',    default='compiler_geometric.py')
    p.add_argument('--rank',        type=int, default=6)
    p.add_argument('--k_lanczos',   type=int, default=20)
    p.add_argument('--fd_eps',      type=float, default=1e-4)
    p.add_argument('--n_vectors',   type=int, default=5)
    p.add_argument('--output',      default='tqft_report.json')
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


# ─── WK extraction ───────────────────────────────────────────────────────────

def extract_wk(state):
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if 'c_attn' in n and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            D = tensor.shape[0]//3
            wk[li] = tensor[D:2*D,:]
        elif ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor if tensor.ndim==2 else tensor.squeeze()
    if not wk:
        raise RuntimeError("No WK. Keys: " + str(list(state.keys())[:8]))
    return [wk[i].detach().float() for i in sorted(wk)]


# ─── Strip areas + Hessian machinery ─────────────────────────────────────────

def strip_areas(wk_list, rank):
    areas = []
    for k in range(len(wk_list)-1):
        Uk  = torch.linalg.svd(wk_list[k],   full_matrices=False)[0][:,:rank]
        Uk1 = torch.linalg.svd(wk_list[k+1], full_matrices=False)[0][:,:rank]
        sv  = torch.linalg.svdvals(Uk.T@Uk1).clamp(-1+1e-6, 1-1e-6)
        areas.append(torch.arccos(sv).sum().item())
    return np.array(areas)


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

def lanczos(model, k, seed=42):
    torch.manual_seed(seed)
    n = model.theta.numel()
    V = torch.zeros(n, k+1)
    alpha_c = torch.zeros(k); beta_c = torch.zeros(k-1)
    v = F.normalize(torch.randn(n), dim=0)
    V[:,0]=v; prev_b=0.
    for j in range(k):
        w = hvp(model, V[:,j])
        if j>0: w = w - prev_b*V[:,j-1]
        a = (w*V[:,j]).sum().item(); alpha_c[j]=a
        w = w - a*V[:,j]
        for i in range(j+1): w = w-(w@V[:,i])*V[:,i]
        if j<k-1:
            b=w.norm().item()
            if b<1e-10: k=j+1; break
            prev_b=b; beta_c[j]=b; V[:,j+1]=w/b
    Hk = (np.diag(alpha_c[:k].numpy())+
          np.diag(beta_c[:k-1].numpy(),1)+
          np.diag(beta_c[:k-1].numpy(),-1))
    ev,ritz = np.linalg.eigh(Hk)
    evecs = V[:,:k].numpy()@ritz
    idx = np.argsort(-np.abs(ev))
    return ev[idx], evecs[:,idx]

def build_basis(wk_list, rank):
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
            lk  = (Uk@Um[:,r:r+1]).reshape(-1).detach().numpy()
            rk  = (Vhm[r:r+1,:]@Vhk1[:rank,:]).reshape(-1).detach().numpy()
            col[offsets[k]:offsets[k+1]] += sc*(lk[:,None]*rk[None,:]).reshape(-1)[:offsets[k+1]-offsets[k]]
            lk1 = (Uk1@Vhm[r:r+1,:].T).reshape(-1).detach().numpy()
            rk1 = (Um[:,r:r+1].T@Vhk[:rank,:]).reshape(-1).detach().numpy()
            col[offsets[k+1]:offsets[k+2]] += sc*(lk1[:,None]*rk1[None,:]).reshape(-1)[:offsets[k+2]-offsets[k+1]]
        nm = np.linalg.norm(col)
        cols.append(col/nm if nm>1e-10 else col)
    return np.stack(cols, axis=1)

def m2_functional(areas):
    mu=np.mean(areas); mad=np.mean(np.abs(areas-mu))
    if mad<1e-10: return 0.
    return float(np.sum(((areas-mu)/mad)**2))

def m2_hessian_fd(areas, eps):
    d=len(areas); H=np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            pp=areas.copy(); pp[i]+=eps; pp[j]+=eps
            pm=areas.copy(); pm[i]+=eps; pm[j]-=eps
            mp=areas.copy(); mp[i]-=eps; mp[j]+=eps
            mm=areas.copy(); mm[i]-=eps; mm[j]-=eps
            H[i,j]=(m2_functional(pp)-m2_functional(pm)
                    -m2_functional(mp)+m2_functional(mm))/(4*eps**2)
    return (H+H.T)/2


# ─── A) Weighted r_m2 ────────────────────────────────────────────────────────

def weighted_frobenius_cos(A, B, sigma):
    """⟨A,B⟩_σ = Σᵢⱼ Aᵢⱼ Bᵢⱼ / (σᵢ σⱼ)"""
    W = np.outer(1./(sigma+1e-8), 1./(sigma+1e-8))
    inner   = float(np.sum(A*B*W))
    norm_A  = math.sqrt(max(float(np.sum(A*A*W)), 1e-12))
    norm_B  = math.sqrt(max(float(np.sum(B*B*W)), 1e-12))
    return inner / (norm_A * norm_B)


# ─── B) Frobenius algebra residuals ──────────────────────────────────────────

class FrobeniusAlgebra:
    """
    Approximate commutative Frobenius algebra on H = R^d.

    Operators defined via the projected Hessian H_hat:
      ∇(u,v) = normalize(H_hat(u⊙v))     [product: Hadamard then project]
      Δ(u)   = H_hat(u) ⊗ H_hat(u) / ‖H_hat(u)‖²  [coproduct: project then outer]
      η      = e₁                          [unit: first strip-angle direction]
      ε(u)   = ⟨u, η⟩_σ                  [counit: σ-weighted pairing with η]

    The four Atiyah-Kock residuals:
      R_assoc:   ‖∇(∇(u,v),w) − ∇(u,∇(v,w))‖  (associativity)
      R_coassoc: ‖(Δ⊗I)Δu − (I⊗Δ)Δu‖           (coassociativity)
      R_frob:    ‖(∇⊗I)(I⊗Δ)u,v − Δ(∇(u,v))‖   (Frobenius relation)
      R_unit:    ‖∇(η,u) − u‖                    (unit axiom)
    """

    def __init__(self, H_hat, sigma):
        self.H = H_hat          # (d,d) projected Hessian
        self.sigma = sigma      # (d,) strip areas
        self.d = len(sigma)
        # Unit: normalize first basis vector weighted by sigma
        e1 = np.zeros(self.d); e1[0] = 1.0
        self.eta = e1 / (np.linalg.norm(e1) + 1e-8)

    def _proj(self, u):
        """Project u through H_hat and normalize."""
        Hu = self.H @ u
        nm = np.linalg.norm(Hu)
        return Hu / nm if nm > 1e-8 else Hu

    def product(self, u, v):
        """∇(u,v) = normalize(H_hat(u⊙v))"""
        uv = u * v
        return self._proj(uv)

    def coproduct(self, u):
        """Δ(u) = H_hat(u) ⊗ H_hat(u) / ‖H_hat(u)‖²  → (d,d) tensor"""
        Hu = self.H @ u
        nm2 = float(np.dot(Hu, Hu))
        if nm2 < 1e-12: return np.zeros((self.d, self.d))
        return np.outer(Hu, Hu) / nm2

    def counit(self, u):
        """ε(u) = ⟨u, η⟩_σ (σ-weighted)"""
        return float(np.sum(u * self.eta / (self.sigma**2 + 1e-8)))

    # ── Four residuals ────────────────────────────────────────────────────────

    def R_assoc(self, vectors):
        """
        R_assoc = (1/N³) Σᵤᵥw ‖∇(∇(u,v),w) − ∇(u,∇(v,w))‖
        Tests associativity: μ(μ⊗I) = μ(I⊗μ)
        """
        total = 0.; count = 0
        for u in vectors:
            for v in vectors:
                for w in vectors:
                    lhs = self.product(self.product(u,v), w)
                    rhs = self.product(u, self.product(v,w))
                    total += np.linalg.norm(lhs - rhs)
                    count += 1
        return total / max(count, 1)

    def R_coassoc(self, vectors):
        """
        R_coassoc = (1/N) Σᵤ ‖(Δ⊗I)Δu − (I⊗Δ)Δu‖_F
        Tests coassociativity: (Δ⊗I)Δ = (I⊗Δ)Δ

        Δu ∈ R^{d×d}; (Δ⊗I)Δu ∈ R^{d×d×d}:
          LHS[i,j,k] = Δ(Δu[:,j])[i,k] * Δu[?,j] ... 
        
        Simplified to matrix level:
          (Δ⊗I)Δ: apply Δ to left column of Δu
          (I⊗Δ)Δ: apply Δ to right column of Δu
        For rank-1 Δu = hu⊗hu: LHS = Δ(hu)⊗hu, RHS = hu⊗Δ(hu)
        In Frobenius norm: ‖Δ(hu)⊗hu − hu⊗Δ(hu)‖_F
        """
        total = 0.; count = 0
        for u in vectors:
            Hu = self.H @ u
            nm = np.linalg.norm(Hu)
            if nm < 1e-8: continue
            hu = Hu / nm
            # Δu = hu ⊗ hu (rank-1)
            # LHS: Δ(hu) ⊗ hu
            Delta_hu = self.coproduct(hu)   # (d,d)
            LHS = np.einsum('ij,k->ijk', Delta_hu, hu)   # (d,d,d)
            # RHS: hu ⊗ Δ(hu)
            RHS = np.einsum('i,jk->ijk', hu, Delta_hu)   # (d,d,d)
            total += float(np.linalg.norm((LHS - RHS).ravel()))
            count += 1
        return total / max(count, 1)

    def R_frob(self, vectors):
        """
        R_frob = (1/N²) Σᵤᵥ ‖(∇⊗I)(I⊗Δ)(u,v) − Δ(∇(u,v))‖_F
        Tests Frobenius relation: (∇⊗I)(I⊗Δ) = Δ∘∇

        LHS: (I⊗Δ)(u,v) = (u, Δv); then (∇⊗I): (∇(u,Δv_left), Δv_right)
           = ∇(u, H_hat(v)/‖H_hat(v)‖) ⊗ H_hat(v)/‖H_hat(v)‖
           [taking left factor of Δv as the projected v]

        RHS: Δ(∇(u,v)) = coproduct of the product
        """
        total = 0.; count = 0
        for u in vectors:
            for v in vectors:
                # RHS: Δ(∇(u,v))
                nabla_uv = self.product(u, v)
                RHS = self.coproduct(nabla_uv)  # (d,d)

                # LHS: ∇(u, H_hat(v)/‖H_hat(v)‖) ⊗ H_hat(v)/‖H_hat(v)‖
                Hv = self.H @ v
                nm_Hv = np.linalg.norm(Hv)
                if nm_Hv < 1e-8: continue
                hv = Hv / nm_Hv
                nabla_u_hv = self.product(u, hv)
                LHS = np.outer(nabla_u_hv, hv)  # (d,d)

                total += float(np.linalg.norm((LHS - RHS).ravel()))
                count += 1
        return total / max(count, 1)

    def R_unit(self, vectors):
        """
        R_unit = (1/N) Σᵤ ‖∇(η, u) − u‖
        Tests unit axiom: ∇(η, ·) = id
        """
        total = 0.; count = 0
        for u in vectors:
            nabla_eta_u = self.product(self.eta, u)
            total += np.linalg.norm(nabla_eta_u - u)
            count += 1
        return total / max(count, 1)

    def compute_all(self, vectors, label=""):
        """Compute all four residuals. Returns dict."""
        print(f"\n    Frobenius algebra residuals ({label}):")
        Ra = self.R_assoc(vectors)
        Rc = self.R_coassoc(vectors)
        Rf = self.R_frob(vectors)
        Ru = self.R_unit(vectors)

        # Normalize by typical vector norm for comparability
        norms = [np.linalg.norm(v) for v in vectors]
        scale = np.mean(norms) if norms else 1.

        print(f"      R_assoc    = {Ra:.4f}  (assoc. defect; 0=exact)")
        print(f"      R_coassoc  = {Rc:.4f}  (coassoc. defect)")
        print(f"      R_frob     = {Rf:.4f}  (Frobenius relation defect)")
        print(f"      R_unit     = {Ru:.4f}  (unit axiom defect)")
        total = Ra + Rc + Rf + Ru
        print(f"      Total      = {total:.4f}  "
              f"{'≈ Frobenius algebra' if total < 0.5 else 'not Frobenius yet'}")

        return {
            'R_assoc':   float(Ra),
            'R_coassoc': float(Rc),
            'R_frob':    float(Rf),
            'R_unit':    float(Ru),
            'R_total':   float(total),
            'is_frobenius': bool(total < 0.5),
        }


# ─── Per-checkpoint analysis ──────────────────────────────────────────────────

def analyze_checkpoint(state, meta, rank, k_lanczos, fd_eps, n_vectors, label):
    t0 = time.time()
    wk_list = extract_wk(state)
    d = len(wk_list) - 1

    sigma = strip_areas(wk_list, rank)
    std   = float(np.std(sigma))
    print(f"\n  strip_std={std:.4f}  σ={np.round(sigma,3)}")

    Hm2  = m2_hessian_fd(sigma, fd_eps)
    basis = build_basis(wk_list, rank)
    model = SymplecticProxy(wk_list, rank)
    evals, evecs = lanczos(model, k_lanczos)
    k_eff = min(evecs.shape[1], d)
    PiV   = basis.T @ evecs[:, :k_eff]
    H_hat = (PiV * evals[:k_eff][None,:]) @ PiV.T

    # A) Weighted r_m2
    r_frob = float(np.sum(H_hat*Hm2) /
                   (np.linalg.norm(H_hat,'fro')*np.linalg.norm(Hm2,'fro')+1e-12))
    r_wgt  = weighted_frobenius_cos(H_hat, Hm2, sigma)
    ratio  = r_wgt / (abs(r_frob) + 1e-8)

    print(f"\n  A) r_m2:  Frobenius={r_frob:+.4f}  weighted={r_wgt:+.4f}  "
          f"ratio={ratio:.3f}")

    # B) Frobenius residuals
    # Backbone vectors: top-k eigenvectors projected to R^d, normalized
    vecs = []
    vlabels = []
    for i in range(min(n_vectors, k_eff)):
        v = PiV[:, i]
        nm = np.linalg.norm(v)
        if nm > 1e-6:
            vecs.append(v / nm)
            vlabels.append(f"e{i}")

    print(f"\n  B) {len(vecs)} backbone vectors in R^{d}")
    fa = FrobeniusAlgebra(H_hat, sigma)
    residuals = fa.compute_all(vecs, label=label)

    elapsed = time.time() - t0
    return {
        'label':          label,
        'step':           meta.get('step'),
        'tau':            meta.get('tau'),
        'val':            meta.get('val'),
        'strip_std':      float(std),
        'r_m2_frobenius': float(r_frob),
        'r_m2_weighted':  float(r_wgt),
        'ratio_w_f':      float(ratio),
        'top5_eigvals':   evals[:5].tolist(),
        'frobenius_residuals': residuals,
        'elapsed_s':      round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  TQFT FROBENIUS DIAGNOSTIC  (v2 — quantitative residuals)")
    print("  A) Strip-area-weighted r_m2")
    print("  B) Frobenius algebra residuals (Atiyah-Kock)")
    print("=" * 60)

    try:
        import_compiler(args.compiler)
    except Exception as e:
        print(f"  Compiler import: {e} (non-fatal)")

    results = []
    for ckpt_path in args.checkpoints:
        print(f"\n{'─'*60}\n  {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            meta  = ckpt.get('metadata', {})
            state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        else:
            meta, state = {}, ckpt
        state = {k:v for k,v in state.items() if isinstance(v,torch.Tensor)}
        try:
            r = analyze_checkpoint(state, meta, args.rank, args.k_lanczos,
                                   args.fd_eps, args.n_vectors,
                                   Path(ckpt_path).stem)
            results.append(r)
        except Exception as e:
            import traceback; print(f"  ERROR: {e}"); traceback.print_exc()

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"  {'Checkpoint':<28} {'std':>6} {'r_frob':>8} {'r_wgt':>8} "
          f"{'R_tot':>7} {'Frob?':>6}")
    print(f"  {'-'*68}")
    for r in results:
        name = r['label'][:27]
        res = r['frobenius_residuals']
        frob = '✓' if res['is_frobenius'] else '✗'
        print(f"  {name:<28} {r['strip_std']:>6.4f} "
              f"{r['r_m2_frobenius']:>+8.4f} {r['r_m2_weighted']:>+8.4f} "
              f"{res['R_total']:>7.4f} {frob:>6}")

    print(f"\n  Key predictions:")
    print(f"  ratio_w_f ≈ 1.0  in uniform regime  (all 4 checkpoints: expected)")
    print(f"  ratio_w_f > 1.0  at std > 0.5        (Moran fixation: pending)")
    print(f"  R_total decreasing across training    (Frobenius crystallization signal)")

    # Check monotonicity of R_total
    if len(results) > 1:
        r_totals = [r['frobenius_residuals']['R_total'] for r in results]
        stds     = [r['strip_std'] for r in results]
        print(f"\n  R_total across checkpoints: {[round(x,4) for x in r_totals]}")
        diffs = [r_totals[i+1]-r_totals[i] for i in range(len(r_totals)-1)]
        decreasing = all(d <= 0 for d in diffs)
        print(f"  Monotone decreasing: {'✓ YES — Frobenius crystallization signal' if decreasing else '✗ no — not monotone'}")

    Path(args.output).write_text(json.dumps(
        {'results': results,
         'interpretation': {
             'weighted_r_m2': 'ratio≈1 in uniform regime; >1 at Moran fixation',
             'R_total': 'decreasing = Frobenius crystallization; stable = free algebra',
             'R_assoc': 'associativity defect of strip composition',
             'R_coassoc': 'coassociativity defect of strip splitting',
             'R_frob': 'Frobenius relation defect (merging vs splitting)',
             'R_unit': 'unit axiom defect (trivial strip identity)',
         }},
        indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
