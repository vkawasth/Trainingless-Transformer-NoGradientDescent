"""
geometric_stopping_criterion.py
=================================
Determines when to stop basin settle early using geometric sensors,
potentially reducing 162 CE steps to ~80 CE steps.

Current stopping rule: |Δval|/step < 0.005 over 8 steps (loss plateau)
Proposed stopping rule: geometric convergence in Stab(F)

Geometric convergence = three simultaneous conditions:
  G1: Φ_cl ≥ threshold (orbit established)
  G2: τ in stable range (K₀ gluing defect settled)
  G3: r_m2^σ > threshold (Hessian-strip alignment restored)

If G1 ∧ G2 ∧ G3 before loss plateau:
  → stop early, proceed to TopoGate
  → saves CE steps = (loss_plateau_step - geometric_convergence_step)

From the data:
  Basin settle logs show Φ_cl=5/5 at step 48 (val=1.28) or 64 (val=0.65)
  τ > 5 maintained from step 56 onward
  r_m2^σ ≈ 0.59-0.70 throughout (computed post-hoc)

The question: does geometric convergence precede loss plateau?
If yes: early stopping saves steps.
If no: geometric sensors are not early enough — need a different signal.

This script estimates the stopping step for each criterion and
computes the potential CE savings.

Usage (analysis only — no model needed):
  python geometric_stopping_criterion.py
"""

import numpy as np

# ─── Data from basin settle logs ─────────────────────────────────────────────
# From the compiler output across multiple runs

BASIN_SETTLE_LOG = [
    # (step, val, Phi_cl, tau)
    ( 8,  9.49, 3, 4.48),
    (16,  4.84, 4, 4.23),
    (24,  3.54, 2, 4.58),
    (32,  2.77, 3, 10.15),
    (40,  1.93, 4, 5.89),
    (48,  1.28, 5, 5.38),
    (56,  0.91, 4, 5.86),
    (64,  0.65, 4, 5.58),
    (72,  0.53, 3, 5.41),
    (80,  0.41, 3, 5.38),
    (88,  0.30, 4, 5.77),
    (96,  0.25, 2, 5.33),
    (104, 0.18, 3, 5.65),
    (112, 0.16, 4, 5.54),
    (120, 0.14, 3, 5.20),  # < 0.15 → stop (current rule)
]

# r_m2^σ computed at available checkpoints (from weighted_rm2_diagnostic)
# step 64: +0.697, step 72: +0.693, basin_entry(~step 162): +0.584
# Interpolate for intermediate steps
def interp_rm2_sigma(step):
    if step <= 64:  return 0.697
    if step <= 72:  return 0.693
    if step <= 120: return 0.693 + (0.584 - 0.693) * (step - 72) / (120 - 72)
    return 0.584


def loss_plateau_step(log, window=8, threshold=0.005):
    """Current stopping rule: |Δval|/step < threshold over window steps."""
    for i in range(window, len(log)):
        vals = [log[j][1] for j in range(i-window, i+1)]
        rates = [abs(vals[k+1]-vals[k])/8 for k in range(len(vals)-1)]
        if max(rates) < threshold:
            return log[i][0], log[i][1]
    return log[-1][0], log[-1][1]


def geometric_convergence_step(log, phi_threshold=4, tau_lo=4.5, tau_hi=6.5,
                                rm2_threshold=0.60, require_all=True):
    """
    Geometric stopping: find first step where ALL conditions hold:
      G1: Phi_cl >= phi_threshold
      G2: tau_lo <= tau <= tau_hi
      G3: r_m2^σ >= rm2_threshold
    """
    for step, val, phi_cl, tau in log:
        rm2 = interp_rm2_sigma(step)
        g1 = phi_cl >= phi_threshold
        g2 = tau_lo <= tau <= tau_hi
        g3 = rm2 >= rm2_threshold
        if require_all and g1 and g2 and g3:
            return step, val, {'Phi_cl': phi_cl, 'tau': tau,
                               'rm2_sigma': rm2}
        # Also check: any two of three (softer criterion)
    return None, None, None


def main():
    print("="*60)
    print("  GEOMETRIC EARLY STOPPING ANALYSIS")
    print("  Can geometric sensors replace loss plateau?")
    print("="*60)

    # Current stopping rule
    loss_step, loss_val = loss_plateau_step(BASIN_SETTLE_LOG)
    print(f"\n  Current stopping (loss plateau): step {loss_step}, val={loss_val:.4f}")
    print(f"  Total basin settle CE: {loss_step}")

    print(f"\n  Geometric convergence analysis:")
    print(f"  {'Criterion':>35} {'Step':>6} {'val':>8} {'Savings':>8}")
    print(f"  {'-'*60}")

    # Test different threshold combinations
    configs = [
        ("Phi≥4, τ∈[4.5,6.5], r_m2σ≥0.60", 4, 4.5, 6.5, 0.60),
        ("Phi≥4, τ∈[4.5,6.5], r_m2σ≥0.65", 4, 4.5, 6.5, 0.65),
        ("Phi≥4, τ∈[5.0,7.0], r_m2σ≥0.60", 4, 5.0, 7.0, 0.60),
        ("Phi≥5, τ∈[4.5,6.5], r_m2σ≥0.60", 5, 4.5, 6.5, 0.60),
        ("Phi≥4 only (no r_m2σ)",            4, 4.5, 7.0, 0.00),
        ("τ∈[5,7] only",                      0, 5.0, 7.0, 0.00),
    ]

    for name, phi_t, tau_lo, tau_hi, rm2_t in configs:
        geo_step, geo_val, geo_state = geometric_convergence_step(
            BASIN_SETTLE_LOG, phi_t, tau_lo, tau_hi, rm2_t)
        if geo_step is not None:
            savings = loss_step - geo_step
            print(f"  {name:>35} {geo_step:>6} {geo_val:>8.4f} {savings:>8} CE")
        else:
            print(f"  {name:>35} {'never':>6} {'':>8} {'0':>8}")

    # Key finding
    print(f"\n  ANALYSIS:")
    print(f"  The basin settle runs {loss_step} CE steps (current).")
    print(f"  Φ_cl=5/5 first appears at step 48 (val=1.28).")
    print(f"  τ > 5 maintained from step 56 onward.")
    print(f"  r_m2^σ ≈ 0.69 from step 64 onward.")
    print()
    print(f"  If geometric criterion (Φ≥4, τ∈[4.5,6.5], r_m2σ≥0.60) is used:")

    geo_step, geo_val, _ = geometric_convergence_step(
        BASIN_SETTLE_LOG, 4, 4.5, 6.5, 0.60)
    if geo_step:
        savings = loss_step - geo_step
        total_current = 187
        total_new = total_current - savings
        print(f"  → Stop at step {geo_step} (val={geo_val:.4f})")
        print(f"  → Save {savings} CE steps")
        print(f"  → Total CE: {total_current} → {total_new} "
              f"({savings/total_current*100:.0f}% reduction)")
    print()
    print(f"  CAVEAT: val={geo_val:.4f} at geometric stop is higher than")
    print(f"  val={loss_val:.4f} at loss plateau.")
    print(f"  The τ-retry (50 CE) would still be needed to reach the floor.")
    print(f"  But the τ-retry starts from a better geometric position.")
    print()
    print(f"  BETTER STRATEGY: Use geometric criterion to SKIP the τ-retry")
    print(f"  altogether by stopping basin settle at the right stability")
    print(f"  condition, then going directly to TopoGate.")
    print()
    print(f"  Target: Φ≥4/5, τ∈[5,6], r_m2σ>0.65 simultaneously")
    print(f"  → Indicates the stability condition is already well-settled")
    print(f"  → TopoGate can proceed without τ-retry")

    # Estimate: how many steps could be saved total?
    print(f"\n  STEP BUDGET BREAKDOWN (current 187 CE):")
    phases = [
        ("Saddle exit",     1,   "fixed"),
        ("MF pump ×2",      10,  "fixed"),
        ("Basin settle",    120, f"→ {geo_step} with geo stopping"),
        ("τ-retry",         50,  "→ 0 if geo stopping works"),
        ("TopoGate",        5,   "fixed"),
        ("K₀ + joint CE",   25,  "fixed"),
        ("Lanczos",         8,   "fixed"),
    ]
    total_min = 0
    print(f"  {'Phase':>20} {'Current':>8} {'Optimized':>10}")
    print(f"  {'-'*42}")
    for name, current, note in phases:
        if "geo" in note:
            if "Basin" in name:
                opt = geo_step
            else:
                opt = 0
        else:
            opt = current
        total_min += opt
        print(f"  {name:>20} {current:>8} {opt:>10}  {note}")
    print(f"  {'TOTAL':>20} {187:>8} {total_min:>10}")
    print(f"\n  Potential reduction: 187 → {total_min} CE "
          f"({(187-total_min)/187*100:.0f}% fewer steps)")


if __name__ == '__main__':
    main()
