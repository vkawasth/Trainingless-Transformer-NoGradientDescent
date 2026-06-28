"""
spectral_evolution.py
======================
Dense branch tracking to resolve the eigenvalue collision question.

For each direction v, samples ε ∈ [0, 0.5] in increments of 0.01,
computes all eigenvalues, and connects branches continuously using
eigenvector overlap (Hungarian assignment between successive steps).

Distinguishes three cases:
  Type A: branch remains real while another becomes dominant (switching)
  Type B: two real branches collide → complex-conjugate pair (collision)
  Type C: tracking ambiguity (overlap drops without clear collision)

Outputs:
  - ASCII plots of Re(λᵢ) and Im(λᵢ) vs ε for each direction
  - Classification table
  - JSON report with full spectral evolution

Usage
-----
  python spectral_evolution.py \\
      --spike64 tau_spikes/tau_spike_step0064_tau5.90.pt \\
      --k_pair 0 --n_dirs 5 \\
      --eps_max 0.5 --eps_step 0.01 \\
      --output spectral_evolution_report.json
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
        if isinstance(obj, complex):     return [obj.real, obj.imag]
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spike64',   default='tau_spikes/tau_spike_step0064_tau5.90.pt')
    p.add_argument('--k_pair',    type=int,   default=0)
    p.add_argument('--n_dirs',    type=int,   default=5)
    p.add_argument('--eps_max',   type=float, default=0.5)
    p.add_argument('--eps_step',  type=float, default=0.01)
    p.add_argument('--n_track',   type=int,   default=4,
                   help='Number of eigenvalue branches to track')
    p.add_argument('--output',    default='spectral_evolution_report.json')
    return p.parse_args()


def load_wk(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = (ckpt.get('state_dict', ckpt.get('model', ckpt))
             if isinstance(ckpt, dict) else ckpt)
    wk = {}
    for name, tensor in state.items():
        if tensor.ndim < 2: continue
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = tensor.detach().float()
    return [wk[i] for i in sorted(wk)]


def compute_M(wk_list, k):
    return wk_list[k+1].numpy() @ np.linalg.pinv(wk_list[k].numpy())


def perturb_M(wk_list, k, v_flat, eps):
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    wk_new = []
    for i, w in enumerate(wk_list):
        s, e = offsets[i], offsets[i+1]
        dv = torch.tensor(v_flat[s:e], dtype=torch.float32).reshape(w.shape)
        wk_new.append(w + eps * dv)
    return wk_new[k+1].numpy() @ np.linalg.pinv(wk_new[k].numpy())


def hungarian_assignment(evecs_prev, evecs_curr):
    """
    Assign eigenvalue indices at ε_curr to those at ε_prev using
    maximum eigenvector overlap (greedy Hungarian-style).
    Returns: permutation array such that evecs_curr[:, perm[i]] best
    matches evecs_prev[:, i].
    """
    n = evecs_prev.shape[1]
    overlap = np.abs(evecs_prev.T.conj() @ evecs_curr)  # (n, n)
    used = set()
    perm = [-1] * n
    for i in range(n):
        best_j, best_ov = -1, -1
        for j in range(n):
            if j not in used and overlap[i, j] > best_ov:
                best_ov = overlap[i, j]
                best_j = j
        perm[i] = best_j
        used.add(best_j)
    return np.array(perm)


def track_spectral_evolution(wk_list, k, v_flat, eps_vals, n_track):
    """
    Track n_track eigenvalue branches continuously across ε values.
    Returns: dict with arrays of shape (n_eps, n_track) for Re, Im, overlap.
    """
    n_eps = len(eps_vals)

    # ε=0 baseline: identify n_track eigenvalues closest to the dominant
    M0 = compute_M(wk_list, k)
    evals0, evecs0 = np.linalg.eig(M0)
    dom_idx = np.argmax(np.abs(evals0.real))
    lam_dom = evals0[dom_idx]

    # Sort all eigenvalues by distance to λ_dom, take n_track closest
    dists = np.abs(evals0 - lam_dom)
    sorted_idx = np.argsort(dists)[:n_track]
    track_evals = evals0[sorted_idx]
    track_evecs = evecs0[:, sorted_idx]   # (D, n_track)

    # Arrays to store evolution
    Re_arr = np.zeros((n_eps, n_track))
    Im_arr = np.zeros((n_eps, n_track))
    Ov_arr = np.ones((n_eps, n_track))    # overlap with ε=0 eigenvectors

    # Store ε=0 values
    Re_arr[0] = track_evals.real
    Im_arr[0] = track_evals.imag
    Ov_arr[0] = 1.0

    prev_evecs = track_evecs.copy()
    prev_evals = track_evals.copy()
    init_evecs = track_evecs.copy()  # always compare to ε=0

    for step, eps in enumerate(eps_vals[1:], 1):
        M_eps = perturb_M(wk_list, k, v_flat, eps)
        evals_e, evecs_e = np.linalg.eig(M_eps)

        # Find n_track eigenvalues closest to previous ones
        new_track_evals = np.zeros(n_track, dtype=complex)
        new_track_evecs = np.zeros((evecs_e.shape[0], n_track), dtype=complex)
        used = set()

        for i in range(n_track):
            # Find closest unmatched eigenvalue to prev_evals[i]
            dists_i = np.abs(evals_e - prev_evals[i])
            for j_sorted in np.argsort(dists_i):
                if j_sorted not in used:
                    new_track_evals[i] = evals_e[j_sorted]
                    new_track_evecs[:, i] = evecs_e[:, j_sorted]
                    used.add(j_sorted)
                    break

        # Overlap with ε=0 eigenvectors
        for i in range(n_track):
            ov = abs(init_evecs[:, i].conj() @ new_track_evecs[:, i])
            ov /= (np.linalg.norm(init_evecs[:, i]) *
                   np.linalg.norm(new_track_evecs[:, i]) + 1e-10)
            Ov_arr[step, i] = float(ov)

        Re_arr[step] = new_track_evals.real
        Im_arr[step] = new_track_evals.imag

        prev_evals = new_track_evals.copy()
        prev_evecs = new_track_evecs.copy()

    return Re_arr, Im_arr, Ov_arr, track_evals


def classify_evolution(Im_arr, Ov_arr, eps_vals):
    """
    Classify based on the dominant branch (index 0, closest to λ_dom):

    Type A (branch switch): Im_arr[:,0] stays near 0, but overlap drops
      because a different (complex) eigenvalue became "dominant"
      while THIS branch remains real.

    Type B (collision): Im_arr[:,0] was 0, then becomes nonzero,
      AND another branch has converging Re with opposite Im sign
      (complex conjugate pair forming).

    Type C (ambiguous): Im_arr[:,0] becomes nonzero but conjugate
      pairing is unclear.
    """
    branch0_real = np.abs(Im_arr[:, 0]) < 0.1   # (n_eps,)
    eps_first_complex = None
    for i, r in enumerate(branch0_real):
        if not r:
            eps_first_complex = eps_vals[i]
            break

    if eps_first_complex is None:
        return "TYPE_A_OR_REAL", None, "Branch 0 stays real throughout."

    # Check for conjugate pair: at the first complex step,
    # is there another branch with same Re and opposite Im?
    step_complex = list(eps_vals).index(eps_first_complex)
    re0 = Re_arr[step_complex, 0]
    im0 = Im_arr[step_complex, 0]

    conjugate_found = False
    for i in range(1, Im_arr.shape[1]):
        re_i = Re_arr[step_complex, i]
        im_i = Im_arr[step_complex, i]
        if (abs(re_i - re0) < 0.5 * abs(re0) and
                abs(im_i + im0) < 0.5 * abs(im0) and
                abs(im0) > 0.01):
            conjugate_found = True
            break

    if conjugate_found:
        cls = "TYPE_B_COLLISION"
        msg = (f"Branch 0 becomes complex at ε={eps_first_complex:.3f} "
               f"with conjugate partner → eigenvalue collision.")
    else:
        cls = "TYPE_C_AMBIGUOUS"
        msg = (f"Branch 0 becomes complex at ε={eps_first_complex:.3f} "
               f"but no clear conjugate partner → ambiguous.")

    return cls, eps_first_complex, msg


def ascii_plot_evolution(Re_arr, Im_arr, eps_vals, title, n_show=2, width=60):
    """ASCII plot of Im(λ) vs ε for the first n_show branches."""
    print(f"\n  {title}")
    all_im = Im_arr[:, :n_show].ravel()
    y_min, y_max = all_im.min(), all_im.max()
    if y_max - y_min < 0.01:
        y_max = y_min + 0.01

    height = 8
    chars = ['●', '○', '▲', '△']
    grid  = [[' '] * width for _ in range(height)]

    for branch in range(n_show):
        for step, eps in enumerate(eps_vals):
            col = int(eps / max(eps_vals) * (width-1))
            val = Im_arr[step, branch]
            row = height-1-int((val-y_min)/(y_max-y_min)*(height-1))
            row = max(0, min(height-1, row))
            col = max(0, min(width-1, col))
            grid[row][col] = chars[branch]

    print(f"  Im(λ) {y_max:+.2f} ┤")
    for row in grid:
        print(f"         │ {''.join(row)}")
    print(f"         {y_min:+.2f} ┤")
    print(f"          └{'─'*width}")
    print(f"          0{' '*(width-8)}{max(eps_vals):.1f}")
    print(f"          ← ε →   (● branch0  ○ branch1)")


def main():
    args = parse_args()
    print("="*60)
    print("  SPECTRAL EVOLUTION TRACKER")
    print("  Dense branch tracking: Type A / B / C classification")
    print("="*60)

    wk_list = load_wk(args.spike64)
    D = sum(w.numel() for w in wk_list)
    splits  = [w.numel() for w in wk_list]
    offsets = list(np.cumsum([0]+splits))
    k = args.k_pair
    s0, e1 = offsets[k], offsets[min(k+2, len(wk_list))]
    block_size = e1 - s0

    eps_vals = np.arange(0, args.eps_max + args.eps_step/2, args.eps_step)
    print(f"  ε ∈ [0, {args.eps_max}] step {args.eps_step}  ({len(eps_vals)} points)")
    print(f"  n_dirs={args.n_dirs}  n_track={args.n_track}  k_pair={k}")

    # Use the same directions as branch_tracking_test for comparison
    rng = np.random.default_rng(42)
    all_results = []

    for trial in range(args.n_dirs):
        v = np.zeros(D)
        v_block = rng.standard_normal(block_size)
        v_block /= np.linalg.norm(v_block) + 1e-10
        v[s0:e1] = v_block

        print(f"\n{'─'*60}")
        print(f"  Direction {trial}")

        global Re_arr  # needed by classify_evolution
        Re_arr, Im_arr, Ov_arr, init_evals = track_spectral_evolution(
            wk_list, k, v, eps_vals, args.n_track)

        # Print key ε values
        print(f"  Initial eigenvalues (ε=0): " +
              ", ".join(f"{e:.3f}+{e.imag:.3f}j" for e in init_evals[:args.n_track]))

        print(f"\n  ε      Im(branch0)  Im(branch1)  overlap0  overlap1  note")
        print(f"  {'-'*65}")
        for step in [0, 5, 10, 20, 30, 40, 50]:
            if step >= len(eps_vals): continue
            eps = eps_vals[step]
            im0 = Im_arr[step, 0]
            im1 = Im_arr[step, 1] if args.n_track > 1 else 0
            ov0 = Ov_arr[step, 0]
            ov1 = Ov_arr[step, 1] if args.n_track > 1 else 1

            # Note special events
            note = ""
            if abs(im0) > 0.1 and abs(Im_arr[max(0,step-1), 0]) < 0.1:
                note = "← branch0 goes complex"
            elif abs(im0) > 0.1 and abs(im1) > 0.1:
                if abs(im0 + im1) < 0.1 * abs(im0):
                    note = "← conjugate pair?"
            print(f"  {eps:.3f}  {im0:>+12.4f} {im1:>+12.4f}  "
                  f"{ov0:>8.4f}  {ov1:>8.4f}  {note}")

        # ASCII plot
        ascii_plot_evolution(Re_arr, Im_arr, eps_vals,
                            f"Im(λ) vs ε — direction {trial}",
                            n_show=min(2, args.n_track))

        # Classify
        cls, eps_crit, msg = classify_evolution(Im_arr, Ov_arr, eps_vals)
        print(f"\n  Classification: {cls}")
        print(f"  {msg}")

        all_results.append({
            'trial': trial,
            'classification': cls,
            'eps_critical': float(eps_crit) if eps_crit else None,
            'message': msg,
            'init_evals_real': [float(e.real) for e in init_evals[:args.n_track]],
            'im_at_eps': {
                f'eps_{eps_vals[step]:.3f}': Im_arr[step, :args.n_track].tolist()
                for step in [0, 10, 20, 30, 40, 50]
                if step < len(eps_vals)
            },
            'overlap_at_eps': {
                f'eps_{eps_vals[step]:.3f}': Ov_arr[step, :args.n_track].tolist()
                for step in [0, 10, 20, 30, 40, 50]
                if step < len(eps_vals)
            },
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    from collections import Counter
    counts = Counter(r['classification'] for r in all_results)
    for cls, n in counts.items():
        print(f"  {cls}: {n}/{args.n_dirs}")

    print(f"\n  Type A (branch switch, original stays real): "
          f"{counts.get('TYPE_A_OR_REAL', 0)}")
    print(f"  Type B (eigenvalue collision → complex pair): "
          f"{counts.get('TYPE_B_COLLISION', 0)}")
    print(f"  Type C (ambiguous):                           "
          f"{counts.get('TYPE_C_AMBIGUOUS', 0)}")

    print(f"\n  Critical ε values (where branch first goes complex):")
    for r in all_results:
        if r['eps_critical']:
            print(f"    dir {r['trial']}: ε_crit = {r['eps_critical']:.3f}  "
                  f"({r['classification']})")

    Path(args.output).write_text(
        json.dumps({'results': all_results, 'eps_vals': eps_vals.tolist()},
                   indent=2, cls=NumpyEncoder))
    print(f"\n  Report → {args.output}")


if __name__ == '__main__':
    main()
