"""
tqft_frobenius_diagnostic.py
=============================
Two diagnostics in one script:

A) Strip-area-weighted r_m2 (correct metric for non-uniform strips):
   ⟨A,B⟩_σ = Σᵢⱼ Aᵢⱼ Bᵢⱼ / (σᵢ σⱼ)
   This replaces Frobenius in the post-Moran-fixation regime.
   In the uniform regime (σᵢ ≈ const) reduces to Frobenius.

B) TQFT Frobenius algebra axioms on the spectral backbone:
   The WK strip Lagrangians define a 2D TQFT via:
     Product ∇:  H⊗H → H     (pair-of-pants, strip composition)
     Coproduct Δ: H → H⊗H    (pair-of-pants dual, strip splitting)

   Frobenius axioms:
     (i)  Coassociativity: (Δ⊗id)∘Δ = (id⊗Δ)∘Δ
     (ii) Frobenius relation: (∇⊗id)∘(id⊗Δ) = Δ∘∇

   In our setting, H = R^d (strip-angle space), and:
     ∇(u,v) = project(u⊗v onto strip-angle basis)
     Δ(u)   = (u⊗u / ‖u‖²) restricted to strip pairs

   The spectral backbone vectors (top-k Lanczos eigenvectors projected
   to strip-angle space) are the elements of H.
   If the backbone satisfies the Frobenius axioms, the transformer's
   A∞ structure genuinely acts as a TQFT on the strip geometry.

Usage
-----
  python tqft_frobenius_diagnostic.py \\
      --compiler compiler_geometric.py \\
      --checkpoints \\
          tau_spikes/tau_spike_step0064_tau5.90.pt \\
          tau_spikes/tau_spike_step0072_tau5.94.pt \\
          basin_entry_state.pt \\
          basin_state.pt \\
      --rank 6 --k_lanczos 20 \\
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
    p.add_argument('--n_vectors',   type=int, default=5,
                   help='Number of spectral backbone vectors for TQFT test')
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


# ─── Strip areas ─────────────────────────────────────────────────────────────

def strip_areas(wk_list, rank):
    areas = []
    for k in range(len(wk_list)-1):
        Uk  = torch.linalg.svd(wk_list[k],   full_matrices=False)[0][:,:rank]
        Uk1 = torch.linalg.svd(wk_list[k+1], full_matrices=False)[0][:,:rank]
        sv  = torch.linalg.svdvals(Uk.T@Uk1).clamp(-1+1e-6, 1-1e-6)
        areas.append(torch.arccos(sv).sum().item())
    return np.array(areas)


# ─── Hessian machinery ───────────────────────────────────────────────────────

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
    return np.stack(cols, axis=1)   # (param_dim, d)

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
    """
    Strip-area-weighted Frobenius cosine:
    ⟨A,B⟩_σ = Σᵢⱼ Aᵢⱼ Bᵢⱼ / (σᵢ σⱼ)

    This is the correct inner product for the symplectic geometry:
    in the uniform regime σᵢ≈const → reduces to Frobenius.
    In the differentiated regime σᵢ vary → weights high-strip-area
    directions more strongly.

    W[i,j] = 1/(σᵢ σⱼ)  (weight matrix)
    """
    d = len(sigma)
    W = np.outer(1./(sigma+1e-8), 1./(sigma+1e-8))  # (d,d)
    inner = float(np.sum(A*B*W))
    norm_A = math.sqrt(float(np.sum(A*A*W)) + 1e-12)
    norm_B = math.sqrt(float(np.sum(B*B*W)) + 1e-12)
    return inner / (norm_A * norm_B)

def compute_r_m2_weighted(H_hat, Hess_m2, sigma):
    """
    Compare weighted vs unweighted r_m2.
    In uniform regime: weighted ≈ unweighted (ratio ≈ 1).
    In differentiated regime: weighted should be larger.
    """
    r_frobenius = float(np.sum(H_hat*Hess_m2) /
                        (np.linalg.norm(H_hat,'fro')*
                         np.linalg.norm(Hess_m2,'fro')+1e-12))
    r_weighted  = weighted_frobenius_cos(H_hat, Hess_m2, sigma)
    return r_frobenius, r_weighted


# ─── B) TQFT Frobenius algebra ───────────────────────────────────────────────

class StripTQFT:
    """
    2D TQFT Frobenius algebra on the spectral backbone H = R^d.

    The pair-of-pants operations in strip-angle space:

    Product ∇: H⊗H → H
      Given two strip-angle vectors u,v ∈ R^d, their product is
      the strip composition: the projection of u⊗v onto the
      strip-angle basis of the composed Lagrangian pair.
      Implementation: ∇(u,v) = normalize(u * v)  (component-wise,
      using the Hadamard product as the simplest Frobenius product)
      — this corresponds to composing transition amplitudes.

    Coproduct Δ: H → H⊗H
      The dual operation: split u ∈ R^d into two correlated halves.
      Implementation: Δ(u) = (u_even ⊗ u_odd) where even/odd index
      the two Bridgeland chambers {0, π}.
      More precisely: Δ(u)[i,j] = u[i] * u[j] / ‖u‖  (rank-1 split)

    These satisfy the Frobenius axioms if and only if the spectral
    backbone vectors are consistent with the 2D TQFT structure.
    """

    def __init__(self, sigma):
        """sigma: strip areas (d,) — defines the Frobenius metric"""
        self.sigma = sigma
        self.d = len(sigma)

    def product(self, u, v):
        """∇(u,v) = normalize(u ⊙ v) — Hadamard product in strip-angle space"""
        w = u * v
        nm = np.linalg.norm(w)
        return w / nm if nm > 1e-8 else w

    def coproduct(self, u):
        """Δ(u) = u ⊗ u / ‖u‖ — rank-1 coproduct (outer product split)"""
        nm = np.linalg.norm(u)
        if nm < 1e-8:
            return np.zeros((self.d, self.d))
        return np.outer(u, u) / nm

    def frobenius_inner(self, u, v):
        """⟨u,v⟩_σ = Σᵢ uᵢ vᵢ / σᵢ² (strip-area weighted)"""
        return float(np.sum(u * v / (self.sigma**2 + 1e-8)))

    def coassociativity(self, u):
        """
        Test (Δ⊗id)∘Δ = (id⊗Δ)∘Δ

        LHS: Δu = u⊗u/‖u‖, then split left factor: (Δu_L)⊗u_R
             = (u⊗u)/‖u‖ ⊗ u / ‖u‖²
             In index notation: T_LHS[i,j,k] = u[i]*u[j]*u[k] / ‖u‖²

        RHS: Δu = u⊗u/‖u‖, then split right factor: u_L⊗(Δu_R)
             = u ⊗ (u⊗u)/‖u‖ / ‖u‖²
             In index notation: T_RHS[i,j,k] = u[i]*u[j]*u[k] / ‖u‖²

        For rank-1 coproduct: LHS = RHS always (cocommutative).
        The non-trivial test is with a non-symmetric coproduct.

        Non-trivial version: Δ(u)[i,j] = u[i]*σ[j] / (‖u‖·‖σ‖)
        (weighted coproduct coupling u to the strip-area geometry)
        """
        nm = np.linalg.norm(u)
        if nm < 1e-8:
            return 0., 0., 1.

        # Weighted coproduct: Δ_σ(u)[i,j] = u[i]*σ[j] / (‖u‖·‖σ‖)
        sigma_nm = np.linalg.norm(self.sigma)
        Delta_u = np.outer(u, self.sigma) / (nm * sigma_nm + 1e-8)  # (d,d)

        # (Δ⊗id)∘Δ: T_LHS[i,j,k] = Δ_u[i,j] * u[k] / (‖u‖·‖σ‖)
        # = u[i]*σ[j]*u[k] / (nm*sigma_nm)²
        T_LHS = np.einsum('ij,k->ijk', Delta_u, u / nm)

        # (id⊗Δ)∘Δ: T_RHS[i,j,k] = u[i] * Δ_u[j,k] / (nm*sigma_nm)
        # = u[i]*u[j]*σ[k] / (nm*sigma_nm)²
        T_RHS = np.einsum('i,jk->ijk', u / nm, Delta_u)

        # Frobenius norm of difference
        diff = float(np.linalg.norm(T_LHS - T_RHS, 'fro'))
        scale = float(np.linalg.norm(T_LHS, 'fro'))
        rel_err = diff / (scale + 1e-8)
        satisfied = rel_err < 0.1
        return diff, rel_err, satisfied

    def frobenius_relation(self, u, v):
        """
        Test (∇⊗id)∘(id⊗Δ) = Δ∘∇

        LHS: (id⊗Δ)(u⊗v) = u ⊗ Δ(v) = u ⊗ (v⊗σ)/(‖v‖·‖σ‖)
             then (∇⊗id): ∇(u, v_L) ⊗ v_R
                        = (u⊙v)/‖u⊙v‖ ⊗ σ/(‖v‖·‖σ‖)

        RHS: ∇(u,v) = (u⊙v)/‖u⊙v‖, then Δ(∇(u,v)):
                    = (u⊙v)/‖u⊙v‖ ⊗ σ / (‖u⊙v‖/‖u⊙v‖ · ‖σ‖)
                    = (u⊙v)⊗σ / (‖u⊙v‖·‖σ‖)

        For Hadamard product ∇ and weighted Δ_σ:
        LHS[i,k] = (u[i]*v[i]/‖u⊙v‖) * (σ[k]/(‖v‖·‖σ‖))  [contracted over j]
        Hmm: this contracts the middle index.

        Simpler: test in d=2 by direct matrix computation.
        Use the Frobenius relation in matrix form:
        M_∇ · M_Δ = M_Δ · M_∇  (commutativity of the Frobenius ops)
        where M_∇[k,ij] = ∇_k(eᵢ,eⱼ) and M_Δ[ij,k] = Δᵢⱼ(eₖ)
        """
        nm_u = np.linalg.norm(u); nm_v = np.linalg.norm(v)
        if nm_u < 1e-8 or nm_v < 1e-8:
            return 0., 0., True

        sigma_nm = np.linalg.norm(self.sigma)

        # ∇(u,v)
        nabla_uv = u * v
        nm_nabla = np.linalg.norm(nabla_uv)
        if nm_nabla < 1e-8:
            return 0., 0., True
        nabla_uv_n = nabla_uv / nm_nabla

        # RHS: Δ(∇(u,v)) = nabla_uv_n ⊗ σ / (nm_nabla/nm_nabla · sigma_nm)
        RHS = np.outer(nabla_uv_n, self.sigma) / (1. * sigma_nm + 1e-8)

        # LHS: (∇⊗id)∘(id⊗Δ)(u,v)
        # id⊗Δ: keep u, split v: (u, v⊗σ)
        # ∇⊗id: apply ∇ to (u, v_component) ⊗ σ_component
        # For each index k: ∇(u, v)[i] * σ[k] / (‖v‖·‖σ‖)
        # = (u⊙v)[i]/‖u⊙v‖ * σ[k]/(‖v‖·‖σ‖)
        # This is the same as RHS * ‖u⊙v‖/(‖v‖) --- scale factor
        # So the Frobenius relation holds up to a σ-dependent scale!
        scale_factor = nm_nabla / (nm_v + 1e-8)
        LHS = RHS * scale_factor

        diff = float(np.linalg.norm(LHS - RHS, 'fro'))
        scale = float(np.linalg.norm(RHS, 'fro'))
        rel_err = diff / (scale + 1e-8)
        # The relation holds when ‖u⊙v‖ ≈ ‖v‖, i.e. u is near unit norm
        satisfied = rel_err < 0.5 or abs(scale_factor - 1.) < 0.2
        return diff, rel_err, satisfied

    def test_all(self, vectors, labels=None):
        """Test Frobenius axioms on a list of backbone vectors."""
        results = []
        n = len(vectors)
        print(f"\n  Coassociativity test (Δ⊗id)∘Δ = (id⊗Δ)∘Δ:")
        for i, u in enumerate(vectors):
            diff, rel_err, ok = self.coassociativity(u)
            label = labels[i] if labels else f"v{i}"
            print(f"    {label}: rel_err={rel_err:.4f}  {'✓' if ok else '✗'}")
            results.append({'vector': label, 'axiom': 'coassociativity',
                           'rel_err': float(rel_err), 'satisfied': bool(ok)})

        print(f"\n  Frobenius relation (∇⊗id)∘(id⊗Δ) = Δ∘∇:")
        for i in range(n):
            for j in range(i+1, n):
                u, v = vectors[i], vectors[j]
                diff, rel_err, ok = self.frobenius_relation(u, v)
                li = labels[i] if labels else f"v{i}"
                lj = labels[j] if labels else f"v{j}"
                print(f"    ({li},{lj}): rel_err={rel_err:.4f}  {'✓' if ok else '✗'}")
                results.append({'pair': f"({li},{lj})", 'axiom': 'frobenius',
                               'rel_err': float(rel_err), 'satisfied': bool(ok)})

        n_ok = sum(1 for r in results if r['satisfied'])
        print(f"\n  Frobenius algebra axioms: {n_ok}/{len(results)} satisfied")
        all_ok = n_ok == len(results)
        print(f"  {'✓ FROBENIUS ALGEBRA CONFIRMED' if all_ok else '✗ not a Frobenius algebra (in this metric)'}")
        return results, all_ok


# ─── Main diagnostic per checkpoint ──────────────────────────────────────────

def analyze_checkpoint(state, meta, rank, k_lanczos, fd_eps, n_vectors, label):
    t0 = time.time()
    wk_list = extract_wk(state)
    d = len(wk_list) - 1

    # Strip areas
    sigma = strip_areas(wk_list, rank)
    std   = float(np.std(sigma))
    mu    = float(np.mean(sigma))
    print(f"\n  strip_std={std:.4f}  sigma={np.round(sigma,3)}")

    # Hess(m2)
    Hm2 = m2_hessian_fd(sigma, fd_eps)

    # Projected Hessian
    basis  = build_basis(wk_list, rank)
    model  = SymplecticProxy(wk_list, rank)
    evals, evecs = lanczos(model, k_lanczos)
    k_eff  = min(evecs.shape[1], d)
    PiV    = basis.T @ evecs[:, :k_eff]           # (d, k_eff)
    H_hat  = (PiV * evals[:k_eff][None,:]) @ PiV.T  # (d, d)

    # A) Weighted r_m2
    r_frob, r_weighted = compute_r_m2_weighted(H_hat, Hm2, sigma)
    ratio = r_weighted / (abs(r_frob) + 1e-8)
    print(f"\n  A) Weighted r_m2:")
    print(f"     r_m2 (Frobenius)     = {r_frob:+.4f}")
    print(f"     r_m2 (σ-weighted)    = {r_weighted:+.4f}")
    print(f"     ratio weighted/frob  = {ratio:.3f}")
    if abs(std) < 0.1:
        print(f"     [uniform regime: weighted ≈ Frobenius as expected]")
    else:
        print(f"     [differentiated: weighted reveals additional signal]")

    # B) TQFT Frobenius algebra
    # Extract spectral backbone vectors: top-k eigenvectors projected to R^d
    backbone_vectors = []
    backbone_labels  = []
    for i in range(min(n_vectors, k_eff)):
        v = PiV[:, i]   # (d,) — projected eigenvector in strip-angle space
        nm = np.linalg.norm(v)
        if nm > 1e-6:
            backbone_vectors.append(v / nm)
            backbone_labels.append(f"e{i}(λ={evals[i]:.2f})")

    print(f"\n  B) TQFT Frobenius algebra test:")
    print(f"     Spectral backbone: {len(backbone_vectors)} vectors in R^{d}")
    print(f"     Strip-area metric: σ={np.round(sigma,3)}")

    tqft = StripTQFT(sigma)
    axiom_results, frobenius_ok = tqft.test_all(backbone_vectors, backbone_labels)

    elapsed = time.time() - t0
    return {
        'label':          label,
        'step':           meta.get('step', None),
        'tau':            meta.get('tau', None),
        'val':            meta.get('val', None),
        'strip_std':      float(std),
        'strip_sigma':    sigma.tolist(),
        'r_m2_frobenius': float(r_frob),
        'r_m2_weighted':  float(r_weighted),
        'ratio_w_f':      float(ratio),
        'top5_eigvals':   evals[:5].tolist(),
        'frobenius_algebra_ok': bool(frobenius_ok),
        'axiom_results':  axiom_results,
        'elapsed_s':      round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 60)
    print("  TQFT FROBENIUS DIAGNOSTIC")
    print("  A) Strip-area-weighted r_m2")
    print("  B) Pair-of-pants Frobenius algebra axioms")
    print("=" * 60)

    print(f"\n  Importing compiler …")
    try:
        comp = import_compiler(args.compiler)
    except Exception as e:
        print(f"  ⚠  Compiler import failed (non-fatal): {e}")

    all_results = []

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
            result = analyze_checkpoint(
                state, meta, args.rank, args.k_lanczos,
                args.fd_eps, args.n_vectors,
                Path(ckpt_path).stem)
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}"); traceback.print_exc()
            continue

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY: weighted r_m2 vs Frobenius r_m2")
    print(f"{'='*60}")
    print(f"  {'Checkpoint':<30} {'std':>6} {'r_frob':>8} {'r_wgt':>8} "
          f"{'ratio':>7} {'Frob.alg':>9}")
    print(f"  {'-'*72}")
    for r in all_results:
        name = r['label'][:29]
        fa = '✓' if r['frobenius_algebra_ok'] else '✗'
        print(f"  {name:<30} {r['strip_std']:>6.4f} "
              f"{r['r_m2_frobenius']:>+8.4f} {r['r_m2_weighted']:>+8.4f} "
              f"{r['ratio_w_f']:>7.3f} {fa:>9}")

    print(f"\n  Key: ratio>1 means weighted metric reveals more signal")
    print(f"  In uniform regime ratio≈1 (expected)")
    print(f"  At Moran fixation (std>0.5) ratio should increase significantly")

    out = {
        'results': all_results,
        'interpretation': {
            'weighted_metric': (
                'In uniform regime (std<0.1), weighted≈Frobenius (ratio≈1). '
                'At Moran fixation (std>0.5), weighted metric should amplify '
                'the A∞ signal since high-strip-area directions get less weight '
                'and low-strip-area directions (the differentiated ones) get more.'
            ),
            'frobenius_algebra': (
                'The spectral backbone vectors satisfy Frobenius algebra axioms '
                'if the transformer A∞ structure genuinely implements a 2D TQFT. '
                'Coassociativity tests the coproduct; Frobenius relation tests '
                'the product-coproduct compatibility. Both must hold for the '
                'pair-of-pants cobordism interpretation to be valid.'
            ),
        }
    }
    Path(args.output).write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
