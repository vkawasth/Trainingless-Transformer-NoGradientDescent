"""
branch_tracking_test.py
========================
Documents the branch-switching that caused the apparent k≈1 in
jet_order_test.py at large ε.

Method: eigenvector overlap tracking.
At ε=0, the dominant eigenvalue λ₀ has right eigenvector r₀.
At ε=ε', the "dominant" eigenvalue (argmax |Re(λ)|) has eigenvector r(ε').
If |r₀ · r(ε')| ≈ 1: same branch (eigenvalue identity preserved).
If |r₀ · r(ε')| ≈ 0: branch switch (different eigenvalue is dominant).

This directly tests whether Im(λ)|_{ε=0.5} ≠ 0 is from the original
branch going complex, or from a different eigenvalue taking over.

Usage
-----
  python branch_tracking_test.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --k_pair 0 --n_dirs 10 \\
      --output branch_tracking_report.json
"""

import argparse, json, math
from pathlib import Path

import numpy as np
import torch


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
    p.add_argument('--k_pair',  type=int, default=0)
    p.add_argument('--n_dirs',  type=int, default=10)
    p.add_argument('--eps_vals', nargs='+', type=float,
                   default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.5])
    p.add_argument('--output',  default='branch_tracking_report.json')
    return p.parse_args()


def load_wk(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor.detach().float()
    return [wk[i] for i in sorted(wk)]


def compute_all_eigenstuff(W0, W1):
    """Returns all eigenvalues and right eigenvectors of M = W1 @ pinv(W0)."""
    M = W1.numpy() @ np.linalg.pinv(W0.numpy())
    evals, evecs = np.linalg.eig(M)
    return evals, evecs, M


def dominant_idx(evals):
    """Index of eigenvalue with largest |Re(λ)|."""
    return int(np.argmax(np.abs(evals.real)))


def track_branch(wk_list, k, v_flat, eps_vals):
    """
    Track the eigenvalue branch as ε increases.
    Returns for each ε:
      - λ_dom: the argmax eigenvalue (may switch branches)
      - λ_orig: the analytic continuation of the ε=0 branch
      - overlap: |r₀ · r_dom(ε)| (1=same branch, 0=switched)
      - branch_switched: bool
    """
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))

    # Baseline: ε=0
    evals0, evecs0, M0 = compute_all_eigenstuff(wk_list[k], wk_list[k+1])
    dom0 = dominant_idx(evals0)
    lam0 = evals0[dom0]
    r0   = evecs0[:, dom0]
    r0   = r0 / (np.linalg.norm(r0) + 1e-10)

    results = []
    for eps in eps_vals:
        # Perturbed weight matrices
        wk_new = []
        for i, w in enumerate(wk_list):
            s, e = offsets[i], offsets[i+1]
            dv = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
            wk_new.append(w + eps * dv)

        evals_e, evecs_e, M_e = compute_all_eigenstuff(wk_new[k], wk_new[k+1])

        # Dominant branch at ε (argmax)
        dom_e = dominant_idx(evals_e)
        lam_dom = evals_e[dom_e]
        r_dom   = evecs_e[:, dom_e]
        r_dom   = r_dom / (np.linalg.norm(r_dom) + 1e-10)

        # Overlap of ε=0 eigenvector with ε eigenspace
        overlap_dom = float(abs(r0.conj() @ r_dom))

        # Nearest-neighbor continuation: find eigenvalue closest to λ₀
        dists = np.abs(evals_e - lam0)
        nn_idx = np.argmin(dists)
        lam_nn = evals_e[nn_idx]
        r_nn   = evecs_e[:, nn_idx]
        r_nn   = r_nn / (np.linalg.norm(r_nn) + 1e-10)
        overlap_nn = float(abs(r0.conj() @ r_nn))

        branch_switched = (dom_e != 0) or (overlap_dom < 0.5)
        # More precise: compare which eigenvalue has highest overlap with r0
        overlaps_all = np.abs(evecs_e.T.conj() @ r0)
        best_match = int(np.argmax(overlaps_all))
        best_overlap = float(overlaps_all[best_match])
        lam_best = evals_e[best_match]

        switched = (best_match != dom_e)

        results.append({
            'eps':           float(eps),
            'lam_dom':       complex(lam_dom),
            'im_lam_dom':    float(lam_dom.imag),
            'lam_best_match': complex(lam_best),
            'im_lam_best':   float(lam_best.imag),
            'overlap_dom':   overlap_dom,
            'overlap_nn':    overlap_nn,
            'best_overlap':  best_overlap,
            'branch_switched': bool(switched),
        })

    return lam0, r0, results


def main():
    args = parse_args()
    print("="*60)
    print("  BRANCH TRACKING TEST")
    print("  Documenting eigenvalue branch switches at large ε")
    print("  Method: eigenvector overlap tracking")
    print("="*60)

    wk_list = load_wk(args.spike64)
    D = sum(w.numel() for w in wk_list)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    k = args.k_pair
    s0, e1 = offsets[k], offsets[min(k+2, len(wk_list))]
    block_size = e1 - s0

    print(f"\n  k_pair={k}  block_size={block_size}  n_dirs={args.n_dirs}")
    print(f"  eps_vals={args.eps_vals}")

    rng = np.random.default_rng(42)
    all_results = []

    # Print header
    print(f"\n  {'dir':>4} {'eps':>6} {'Im(λ_dom)':>12} {'Im(λ_best)':>12} "
          f"{'overlap':>9} {'switched?':>10}")
    print(f"  {'-'*60}")

    for trial in range(args.n_dirs):
        v = np.zeros(D)
        v_block = rng.standard_normal(block_size)
        v_block /= np.linalg.norm(v_block) + 1e-10
        v[s0:e1] = v_block

        lam0, r0, track = track_branch(wk_list, k, v, args.eps_vals)

        dir_results = {'trial': trial, 'lam0': complex(lam0), 'track': []}

        for t in track:
            if trial < 5:  # print first 5 directions
                sw = '✓ SWITCH' if t['branch_switched'] else '  same'
                print(f"  {trial:>4} {t['eps']:>6.3f} {t['im_lam_dom']:>+12.4f} "
                      f"{t['im_lam_best']:>+12.4f} {t['best_overlap']:>9.4f} "
                      f"{sw:>10}")
            dir_results['track'].append(t)

        if trial < 5:
            print()

        all_results.append(dir_results)

    # Summary: at which ε do branches typically switch?
    print(f"\n  SUMMARY: Fraction of directions with branch switch at each ε")
    print(f"  {'eps':>8} {'frac_switched':>15} {'mean_overlap':>13} "
          f"{'mean_Im(λ_dom)':>16} {'mean_Im(λ_best)':>17}")
    print(f"  {'-'*72}")

    for i, eps in enumerate(args.eps_vals):
        tracks_at_eps = [r['track'][i] for r in all_results]
        frac_sw = np.mean([t['branch_switched'] for t in tracks_at_eps])
        mean_ov = np.mean([t['best_overlap']    for t in tracks_at_eps])
        mean_im_dom  = np.mean([abs(t['im_lam_dom'])  for t in tracks_at_eps])
        mean_im_best = np.mean([abs(t['im_lam_best']) for t in tracks_at_eps])
        print(f"  {eps:>8.3f} {frac_sw:>15.3f} {mean_ov:>13.4f} "
              f"{mean_im_dom:>+16.4f} {mean_im_best:>+17.6f}")

    print(f"\n  INTERPRETATION:")
    print(f"  - 'overlap' = |r₀ · r_best(ε)|: how much the ε=0 eigenvector")
    print(f"    matches the best-matching eigenvector at ε")
    print(f"  - overlap≈1: same branch (no switch)")
    print(f"  - overlap≈0: different eigenvalue has taken over")
    print(f"  - Im(λ_dom): imaginary part of argmax eigenvalue (may have switched)")
    print(f"  - Im(λ_best): imaginary part of ε=0 branch continuation (should stay 0)")
    print(f"\n  If Im(λ_best)=0 but Im(λ_dom)≠0 at large ε:")
    print(f"  → The original branch remains real (algebraic locking holds)")
    print(f"  → The argmax switched to a different complex eigenvalue")
    print(f"  → This confirms the Algebraic Real-Locking theorem")

    Path(args.output).write_text(
        json.dumps({'results': all_results, 'eps_vals': args.eps_vals},
                   indent=2, cls=NumpyEncoder, default=str))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
