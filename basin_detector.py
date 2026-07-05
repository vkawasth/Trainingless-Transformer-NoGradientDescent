"""
basin_detector.py
==================
Detects whether Phase 3 basin is featureless at basin entry.
Featureless basin → use LR×50 for fast descent.
Structured basin → use LR×5 with geometric guidance.

Three detectors, increasing cost:
  D1: Offline (0 forward passes) — from architecture stats
  D2: Hessian isotropy (10 forward passes, ~0.1s)
  D3: Probe descent (16 gradient steps on copy, ~2s)

Usage:
  from basin_detector import detect_basin, recommended_lr
  basin_type, lr_mult = detect_basin(model, get_batch, LR, rank=6)
  print(f"Basin: {basin_type}, use LR×{lr_mult}")
"""

import math, copy
import numpy as np
import torch


# ── D1: Offline detector ──────────────────────────────────────────────────────

def detect_offline(rank=6, D=256, N_heads=4):
    """
    Offline detector from architecture stats alone.
    
    Key signal: rank mismatch r_used/r_natural
    r_natural = d_head/4 = (D/N_heads)/4
    
    If r_used/r_natural < 0.5: weak spectral gap → featureless basin
    If r_used/r_natural > 0.8: clean spectral gap → may have structure
    
    Cost: 0 forward passes. O(1).
    """
    d_head = D // N_heads
    r_natural = max(1, d_head // 4)
    ratio = rank / r_natural
    
    featureless = ratio < 0.5
    confidence = 1.0 - ratio  # higher = more confident featureless
    
    return {
        'detector': 'offline',
        'r_used': rank,
        'r_natural': r_natural,
        'ratio': ratio,
        'featureless': featureless,
        'confidence': confidence,
        'reason': (f"r_used/r_natural = {rank}/{r_natural} = {ratio:.2f} "
                   f"({'< 0.5 → featureless' if featureless else '> 0.5 → may have structure'})")
    }


# ── D2: Hessian isotropy detector ────────────────────────────────────────────

def detect_hessian_isotropy(model, get_batch, n_directions=5, eps=1e-2):
    """
    Detect Hessian isotropy via finite differences in random directions.
    
    κ_i = (L(θ+εv_i) - 2L(θ) + L(θ-εv_i)) / ε²
    
    If max(κ)/min(κ) < 10: isotropic → featureless → LR×50
    If max(κ)/min(κ) > 100: anisotropic → structured → LR×5
    
    Cost: 2*n_directions + 1 forward passes ≈ 11 forward passes, ~0.1s
    """
    model.eval()
    
    # Get a batch for consistent measurement
    with torch.no_grad():
        x, y = get_batch()
        _, loss0 = model(x, y)
        L0 = loss0.item()
    
    curvatures = []
    params = list(model.parameters())
    
    # Save original params
    saved = [p.data.clone() for p in params]
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_directions):
        # Random direction
        directions = [torch.tensor(rng.standard_normal(p.shape),
                                   dtype=p.dtype, device=p.device)
                     for p in params]
        # Normalize
        norm = math.sqrt(sum((d**2).sum().item() for d in directions))
        directions = [d / (norm + 1e-10) for d in directions]
        
        # Forward perturb
        with torch.no_grad():
            for p, d in zip(params, directions):
                p.data.add_(d, alpha=eps)
        with torch.no_grad():
            _, loss_p = model(x, y)
            Lp = loss_p.item()
        
        # Restore
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.data.copy_(s)
        
        # Backward perturb
        with torch.no_grad():
            for p, d in zip(params, directions):
                p.data.add_(d, alpha=-eps)
        with torch.no_grad():
            _, loss_m = model(x, y)
            Lm = loss_m.item()
        
        # Restore
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.data.copy_(s)
        
        kappa = (Lp - 2*L0 + Lm) / (eps**2)
        curvatures.append(abs(kappa))
    
    curvatures = [c for c in curvatures if c > 1e-10]
    if not curvatures:
        return {'detector': 'hessian', 'featureless': True,
                'confidence': 0.5, 'reason': 'no curvature detected'}
    
    kappa_max = max(curvatures)
    kappa_min = min(curvatures)
    condition = kappa_max / (kappa_min + 1e-10)
    
    featureless = condition < 10
    confidence = 1.0 / (1.0 + math.log10(condition + 1))
    
    return {
        'detector': 'hessian',
        'curvatures': curvatures,
        'condition_number': condition,
        'featureless': featureless,
        'confidence': confidence,
        'reason': (f"Hessian condition κ_max/κ_min = {condition:.1f} "
                   f"({'< 10 → isotropic → featureless' if featureless else '> 10 → anisotropic → structured'})")
    }


# ── D3: Probe descent detector ────────────────────────────────────────────────

def detect_probe_descent(model, get_batch, eval_val, LR,
                          n_probe_steps=8, phi_clean_fn=None,
                          gluing_defect_fn=None):
    """
    Run 8 gradient steps at LR×50 on a copy of the model.
    
    If val drops below 2.0 with Φ_cl ≥ 3/5: featureless → LR×50 is safe
    If val stays above 5.0 or Φ_cl drops to 0/5: structured/fragile → LR×5
    
    Cost: 8 gradient steps on a model copy ≈ 2s
    """
    model_copy = copy.deepcopy(model)
    opt = torch.optim.AdamW(model_copy.parameters(), lr=LR*50,
                             betas=(0.9, 0.95), weight_decay=0.1)
    
    val_start = eval_val(model, n=4)
    val_after = val_start
    phi_ok = True
    
    for step in range(n_probe_steps):
        model_copy.train()
        x, y = get_batch()
        _, loss = model_copy(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
        opt.step()
    
    val_after = eval_val(model_copy, n=4)
    
    # Check orbit stability
    if phi_clean_fn is not None:
        pc = phi_clean_fn(model_copy)
        phi_ok = pc >= 3
    
    if gluing_defect_fn is not None:
        tau = gluing_defect_fn(model_copy, n=4)
        tau_ok = tau < 8.0  # tau spike < 8 is acceptable
    else:
        tau_ok = True
    
    descent_ratio = val_start / (val_after + 1e-10)
    featureless = (val_after < 2.0 and phi_ok and tau_ok)
    confidence = min(1.0, descent_ratio / 5.0)  # 5× descent = high confidence
    
    del model_copy  # free memory
    
    return {
        'detector': 'probe',
        'val_start': val_start,
        'val_after': val_after,
        'descent_ratio': descent_ratio,
        'phi_ok': phi_ok,
        'tau_ok': tau_ok,
        'featureless': featureless,
        'confidence': confidence,
        'reason': (f"Probe LR×50: val {val_start:.2f}→{val_after:.2f} "
                   f"(ratio={descent_ratio:.1f}x), "
                   f"orbit={'stable' if phi_ok else 'broken'}")
    }


# ── Combined detector ─────────────────────────────────────────────────────────

def detect_basin(model, get_batch, eval_val, LR,
                 rank=6, D=256, N_heads=4,
                 phi_clean_fn=None, gluing_defect_fn=None,
                 use_probe=True):
    """
    Combined basin detector. Uses D1+D2 cheaply, D3 only if needed.
    
    Returns: (basin_type, lr_multiplier, confidence, details)
    basin_type: 'featureless' or 'structured'
    lr_multiplier: 50 or 5
    """
    results = {}
    
    # D1: Offline (always run, free)
    d1 = detect_offline(rank, D, N_heads)
    results['D1_offline'] = d1
    print(f"  D1 (offline): {d1['reason']}")
    
    # D2: Hessian isotropy (fast, ~0.1s)
    d2 = detect_hessian_isotropy(model, get_batch)
    results['D2_hessian'] = d2
    print(f"  D2 (hessian): {d2['reason']}")
    
    # Vote from D1 and D2
    votes_featureless = sum([d1['featureless'], d2['featureless']])
    
    if votes_featureless == 2:
        # Both agree: featureless
        if use_probe:
            # Confirm with probe (2s)
            d3 = detect_probe_descent(model, get_batch, eval_val, LR,
                                       phi_clean_fn=phi_clean_fn,
                                       gluing_defect_fn=gluing_defect_fn)
            results['D3_probe'] = d3
            print(f"  D3 (probe):   {d3['reason']}")
            final_featureless = d3['featureless']
            confidence = (d1['confidence'] + d2['confidence'] +
                         d3['confidence']) / 3
        else:
            final_featureless = True
            confidence = (d1['confidence'] + d2['confidence']) / 2
    elif votes_featureless == 0:
        # Both agree: structured
        final_featureless = False
        confidence = ((1-d1['confidence']) + (1-d2['confidence'])) / 2
    else:
        # Disagreement: run probe to decide
        if use_probe:
            d3 = detect_probe_descent(model, get_batch, eval_val, LR,
                                       phi_clean_fn=phi_clean_fn,
                                       gluing_defect_fn=gluing_defect_fn)
            results['D3_probe'] = d3
            print(f"  D3 (probe):   {d3['reason']}")
            final_featureless = d3['featureless']
            confidence = d3['confidence']
        else:
            # Default to featureless (safer: LR×50 with orbit monitoring)
            final_featureless = True
            confidence = 0.5
    
    basin_type = 'featureless' if final_featureless else 'structured'
    lr_mult = 50 if final_featureless else 5
    
    return basin_type, lr_mult, confidence, results


def recommended_lr(model, get_batch, eval_val, LR,
                   rank=6, D=256, N_heads=4,
                   phi_clean_fn=None, gluing_defect_fn=None):
    """
    One-line interface: returns recommended LR multiplier for Phase 3.
    """
    print("  Detecting basin type...")
    basin_type, lr_mult, conf, _ = detect_basin(
        model, get_batch, eval_val, LR, rank, D, N_heads,
        phi_clean_fn, gluing_defect_fn)
    print(f"  Basin: {basin_type} (confidence={conf:.2f}) → use LR×{lr_mult}")
    return lr_mult


if __name__ == '__main__':
    print("Basin detector for Phase 3 LR selection.")
    print("Detectors:")
    print("  D1 (offline, 0 passes): rank mismatch r_used/r_natural")
    print("  D2 (hessian, 11 passes): Hessian condition number")
    print("  D3 (probe, 8 steps): test LR×50 on model copy")
    print()
    print("Usage:")
    print("  from basin_detector import recommended_lr")
    print("  lr_mult = recommended_lr(model, get_batch, eval_val, LR, rank=6)")
    print("  # Use lr_mult in Phase 3 AdamW")
    print()
    print("Expected output for our setting:")
    d1 = detect_offline(rank=6, D=256, N_heads=4)
    print(f"  D1: {d1['reason']}")
    print(f"  → Featureless basin expected → LR×50 recommended")
