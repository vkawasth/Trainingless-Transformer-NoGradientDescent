"""
loss_function_experiment.py
============================
Tests augmented loss functions to see if Phase 3 can converge faster.

Four candidates:
A) Pure CE (baseline)
B) CE + phase coherence penalty λ·Σ sin²(φ_k)
C) CE + τ regularization λ·(τ-τ*)²  
D) CE + r_m2^σ alignment -λ·r_m2^σ

Run from basin entry (post-MF2) state.
Measure: steps to val < 0.15, wall crossings, final val.

Usage: call run_loss_experiment(model, get_batch, eval_val, ...) 
       from within the compiler after MF pump.
"""

import math, time, copy
import numpy as np
import torch


def compute_phase_penalty(model, lambda_phase=0.1):
    """
    Phase coherence penalty: λ·Σ_k sin²(φ_k)
    
    At clean phase (φ_k ∈ {0,π}): sin²(φ_k) = 0, no penalty.
    Off-wall: penalty pushes phases toward {0,π}.
    
    Computable via WK matrices without eigendecomposition:
    sin²(φ_k) ≈ (Im λ_k)² / |λ_k|²  where λ_k = dominant eigenvalue of M_k
    """
    wk = {}
    for name, param in model.named_parameters():
        n = name.lower()
        if ('key' in n or 'wk' in n) and 'weight' in n and param.ndim >= 2:
            try: li = int([p for p in name.split('.') if p.isdigit()][0])
            except: li = len(wk)
            wk[li] = param
    wk_list = [wk[i] for i in sorted(wk)]

    penalty = torch.tensor(0.0)
    for k in range(len(wk_list)-1):
        W0 = wk_list[k]; W1 = wk_list[k+1]
        # Proxy: ‖W1 - W0‖² / ‖W1‖² measures misalignment
        # (true penalty requires eigendecomposition, not differentiable easily)
        # Better proxy: ‖W1·W0^{-1} - W1·W0^{-1}.T‖_F (asymmetry = phase)
        try:
            M = W1 @ torch.linalg.solve(W0.T, torch.eye(W0.shape[0],
                                          device=W0.device))
            asymmetry = (M - M.T).pow(2).sum() / (M.pow(2).sum() + 1e-8)
            penalty = penalty + asymmetry
        except Exception:
            pass
    return lambda_phase * penalty


def compute_tau_penalty(model, tau_target=5.3, lambda_tau=0.01):
    """
    τ regularization: λ·(τ - τ*)²
    
    τ = ‖∇_FF L‖ / ‖∇_Emb L‖ — requires gradient norms.
    Proxy using weight norms instead (correlated with gradient norms):
    τ_proxy = ‖W_FF‖_F / ‖W_Emb‖_F
    """
    ff_norm  = torch.tensor(0.0)
    emb_norm = torch.tensor(0.0)
    for name, param in model.named_parameters():
        n = name.lower()
        if 'ff' in n or 'mlp' in n or 'feedforward' in n:
            ff_norm = ff_norm + param.pow(2).sum()
        elif 'emb' in n or 'embed' in n or 'wte' in n:
            emb_norm = emb_norm + param.pow(2).sum()
    tau_proxy = (ff_norm.sqrt() / (emb_norm.sqrt() + 1e-8))
    return lambda_tau * (tau_proxy - tau_target).pow(2)


def run_loss_experiment(model, get_batch, eval_val,
                         phi_clean_fn, gluing_defect_fn, LR,
                         n_steps=120, rank=6):
    """
    A/B/C/D experiment: four loss functions from same starting point.
    """
    state_0 = copy.deepcopy(model.state_dict())

    configs = [
        ('A: Pure CE (baseline)',        0.0,    0.0,   False),
        ('B: CE + phase penalty',        0.05,   0.0,   False),
        ('C: CE + τ regularization',     0.0,    0.01,  False),
        ('D: CE + phase + τ',            0.05,   0.01,  False),
    ]

    results = []
    for label, lam_phase, lam_tau, use_rm2 in configs:
        model.load_state_dict(state_0)
        opt = torch.optim.AdamW(model.parameters(), lr=LR*5,
                                 betas=(0.9,0.95), weight_decay=0.1)
        val_history = [eval_val(model, n=4)]
        step_converged = n_steps
        wall_crossings = 0
        prev_phases = None

        print(f"\n  {label}")
        print(f"  {'step':>5} {'val':>8} {'Φ_cl':>6} {'τ':>6} {'walls':>6}")
        print(f"  {'─'*38}")

        for step in range(1, n_steps+1):
            if step <= 10:
                for pg in opt.param_groups: pg['lr'] = LR*5*step/10

            model.train()
            x, y = get_batch()
            _, ce_loss = model(x, y)

            # Augmented loss
            total_loss = ce_loss
            if lam_phase > 0:
                total_loss = total_loss + compute_phase_penalty(
                    model, lam_phase)
            if lam_tau > 0:
                total_loss = total_loss + compute_tau_penalty(
                    model, 5.3, lam_tau)

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 8 == 0:
                v   = eval_val(model, n=8)
                pc  = phi_clean_fn(model)
                tau = gluing_defect_fn(model, n=4)
                delta = abs(v - val_history[-1]) / 8
                val_history.append(v)

                # Count wall crossings
                wk_pairs = []
                for name, param in model.named_parameters():
                    n = name.lower()
                    if ('key' in n or 'wk' in n) and 'weight' in n:
                        try: li = int([p for p in name.split('.') if p.isdigit()][0])
                        except: li = 0
                        wk_pairs.append((li, param.detach().float().cpu().numpy()))
                wk_pairs.sort(key=lambda x: x[0])
                phases_now = []
                for k in range(len(wk_pairs)-1):
                    W0 = wk_pairs[k][1].astype(complex)
                    W1 = wk_pairs[k+1][1].astype(complex)
                    try:
                        M = W1 @ np.linalg.pinv(W0)
                        ev = np.linalg.eigvals(M)
                        lam = ev[np.argmax(np.abs(ev.real))]
                        phases_now.append(float(np.arctan2(lam.imag, lam.real)))
                    except Exception:
                        phases_now.append(0.0)

                if prev_phases is not None:
                    wc = sum(1 for p1, p2 in zip(prev_phases, phases_now)
                            if np.sign(p1) != np.sign(p2)
                            and abs(p1) > 0.15 and abs(p2) > 0.15)
                    wall_crossings += wc
                prev_phases = phases_now

                print(f"  {step:>5} {v:>8.4f} {pc:>5}/5 {tau:>6.2f} {wall_crossings:>6}")

                if delta < 0.003 and v < 0.20:
                    print(f"  ✓ Converged at step {step}")
                    step_converged = step; break
                if v < 0.15:
                    print(f"  ✓ val < 0.15 at step {step}")
                    step_converged = step; break

        results.append({
            'label': label,
            'step_converged': step_converged,
            'final_val': eval_val(model),
            'wall_crossings': wall_crossings,
            'val_history': val_history,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  LOSS FUNCTION EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"  {'Config':>30} {'Steps':>7} {'Val':>8} {'Walls':>7}")
    print(f"  {'─'*55}")
    for r in results:
        print(f"  {r['label']:>30} {r['step_converged']:>7} "
              f"{r['final_val']:>8.4f} {r['wall_crossings']:>7}")

    best = min(results, key=lambda r: r['step_converged'])
    print(f"\n  Best: {best['label']} ({best['step_converged']} steps)")

    if best['step_converged'] < results[0]['step_converged']:
        saving = results[0]['step_converged'] - best['step_converged']
        print(f"  Phase 3 saving vs baseline: {saving} CE steps")
    else:
        print(f"  No improvement over baseline CE loss")
        print(f"  Basin IS featureless — geometric penalties don't help")

    return results


if __name__ == '__main__':
    print("Run from within compiler after MF pump:")
    print("  from loss_function_experiment import run_loss_experiment")
    print("  results = run_loss_experiment(model, get_batch, eval_val,")
    print("                                phi_clean, gluing_defect, LR)")
