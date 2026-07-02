"""
PATCH: compiler_geometric_geo_stop.py
Replace Phase 3 basin settle block in compiler_geometric.py with this version.

Changes:
  1. Add rm2_sigma computation at each 8-step checkpoint
  2. Geometric stopping: Φ_cl≥4 AND τ∈[5,7] AND rm2σ≥0.65 for 2 consecutive checks
  3. Skip τ-retry if geo-stop triggered (rm2σ already settled)
  4. Use LR×10 for 30CE after geo-stop instead of LR×5 for 120CE + LR×2 for 50CE

Expected: 40-48 CE (geo-stop) + 30 CE (fast descent) + skipped τ-retry
         = ~75-80 CE vs current 120-170 CE
"""

import math, torch, numpy as np

# ── Weighted r_m2^σ computation (inline, no external import) ──────────────────

def compute_rm2_sigma_inline(model, rank=6):
    """
    Compute strip-area-weighted Frobenius correlation r_m2^σ inline.
    Uses WK singular values as the Hessian proxy.
    Returns float in [-1, +1].
    """
    wk_list = []
    for name, param in model.named_parameters():
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n and param.ndim >= 2:
            wk_list.append(param.detach().float().cpu().numpy())
    if len(wk_list) < 2:
        return 0.0

    wk_list.sort(key=lambda w: w.shape[0])  # sort by layer size
    rm2_vals = []

    for k in range(len(wk_list)-1):
        W0, W1 = wk_list[k], wk_list[k+1]
        try:
            U0, s0, _ = np.linalg.svd(W0, full_matrices=False)
            U1, s1, _ = np.linalg.svd(W1, full_matrices=False)
            r = min(rank, U0.shape[1], U1.shape[1])
            Ur0, Ur1 = U0[:, :r], U1[:, :r]
            sv = np.linalg.svd(Ur0.T @ Ur1, compute_uv=False)
            sv = np.clip(sv, 1e-6, 1-1e-6)

            # Hessian of arccos at σᵢ
            h_strip = sv / (1 - sv**2)**1.5
            h_loss  = s0[:r] / (np.linalg.norm(s0[:r]) + 1e-10)
            h_strip = h_strip / (np.linalg.norm(h_strip) + 1e-10)

            weights = 1.0 / (sv**2 + 1e-6)
            num  = np.dot(h_loss * weights, h_strip)
            den  = (np.sqrt(np.dot(h_loss**2, weights)) *
                    np.sqrt(np.dot(h_strip**2, weights)) + 1e-10)
            rm2_vals.append(float(num / den))
        except Exception:
            pass

    return float(np.mean(rm2_vals)) if rm2_vals else 0.0


# ── PHASE 3: BASIN SETTLE (geometric early stopping) ─────────────────────────
# EXPERIMENT: replace loss-plateau stop with geometric convergence stop
# Hypothesis: stopping at Φ_cl≥4 + τ∈[5,7] + rm2σ≥0.65 (2 consecutive)
# saves ~70-80 CE steps vs loss-plateau at step 120

print("━━━ PHASE 3: BASIN SETTLE (GEO-STOP EXPERIMENT) ━━━━━━━━")
print("  Geometric stopping: Φ_cl≥4 + τ∈[5,7] + rm2σ≥0.65 (×2 checks)")
print("  Hypothesis: orbit geometry converges before loss plateaus")

opt_b = torch.optim.AdamW(model.parameters(), lr=LR*5,
                           betas=(0.9,0.95), weight_decay=0.1)
val_history = [v_mf]
step = 0
geo_stop_count = 0
geo_stopped = False
geo_stop_step = None

for step in range(1, 151):
    # Warmup first 10 steps
    if step <= 10:
        for pg in opt_b.param_groups:
            pg['lr'] = LR*5*step/10
    model.train(); x, y = get_batch(); _, l = model(x, y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:
        v = eval_val(model, n=8)
        delta = abs(v - val_history[-1]) / 8
        val_history.append(v)
        pc  = phi_clean(model)
        tau = gluing_defect(model, n=4)
        rm2 = compute_rm2_sigma_inline(model)

        print(f"  step {step:3d}: val={v:.4f}  Δ={delta:.4f}  "
              f"Φ_cl={pc}/5  τ={tau:.2f}  rm2σ={rm2:+.3f}")

        # Original stopping conditions (kept as fallback)
        if delta < 0.003:
            print(f"  ✓ Plateau (loss)"); break
        if v < 0.15:
            print(f"  ✓ val={v:.4f} < 0.15"); break

        # NEW: geometric stopping condition
        geo_ok = (pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65)
        if geo_ok:
            geo_stop_count += 1
            print(f"  ○ GEO-STOP candidate ({geo_stop_count}/2): "
                  f"Φ={pc}/5 τ={tau:.2f} rm2σ={rm2:.3f}")
            if geo_stop_count >= 2:
                print(f"  ✓ GEO-STOP confirmed at step {step}")
                geo_stopped = True
                geo_stop_step = step
                break
        else:
            geo_stop_count = 0  # reset if conditions not met

step_basin = step
v_basin = eval_val(model); pc_b = phi_clean(model); tau_b = gluing_defect(model)
rm2_b = compute_rm2_sigma_inline(model)
print(f"  After {step}CE: val={v_basin:.4f}  Φ_cl={pc_b}/5  "
      f"τ={tau_b:.2f}  rm2σ={rm2_b:+.3f}")
print(f"  Geo-stop: {'YES at step '+str(geo_stop_step) if geo_stopped else 'NO (loss plateau)'}")

# Extension if Φ_cl < 3 (unchanged from original)
if pc_b < 3:
    print(f"  ⚠ Φ_cl={pc_b}/5 — extending 16CE")
    for _ in range(16):
        model.train(); x, y = get_batch(); _, l = model(x, y)
        opt_b.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_b.step()
    v_basin = eval_val(model); pc_b = phi_clean(model)
    tau_b = gluing_defect(model); rm2_b = compute_rm2_sigma_inline(model)
    step_basin += 16
    print(f"  After extension: val={v_basin:.4f}  Φ_cl={pc_b}/5")

torch.save(model.state_dict(), 'basin_entry_state.pt')
print(f"  Saved basin_entry_state.pt (val={v_basin:.4f})")

# ── τ-retry: SKIP if geo-stopped, run fast descent instead ───────────────────
if geo_stopped:
    # Geo-stop means stability condition is settled — run aggressive descent
    # instead of slow τ-retry to reach the loss floor
    print(f"  ○ GEO-STOP: skipping τ-retry, running 30CE fast descent @LR×10")
    n_fast = 30
    opt_fast = torch.optim.AdamW(model.parameters(), lr=LR*10,
                                  betas=(0.9,0.95), weight_decay=0.1)
    for _s in range(n_fast):
        # Cosine anneal from LR×10 to LR×2
        lr_s = LR*2 + (LR*10 - LR*2) * 0.5 * (1 + math.cos(math.pi*_s/n_fast))
        for pg in opt_fast.param_groups: pg['lr'] = lr_s
        model.train(); x, y = get_batch(); _, l = model(x, y)
        opt_fast.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_fast.step()
    v_basin = eval_val(model); pc_b = phi_clean(model)
    tau_b = gluing_defect(model); rm2_b = compute_rm2_sigma_inline(model)
    step_basin += n_fast
    print(f"  After fast descent ({n_fast}CE@LR×10→2): val={v_basin:.4f}  "
          f"Φ_cl={pc_b}/5  τ={tau_b:.2f}  rm2σ={rm2_b:+.3f}")

elif tau_b > 5:
    # Original τ-retry (only if geo-stop didn't trigger)
    n_retry = 25 if pc_b >= 5 else 75 if pc_b <= 2 else 50
    print(f"  ⚠ HIGH τ={tau_b:.2f}  Φ_cl={pc_b}/5 → τ-retry {n_retry}CE@LR×2")
    opt_retry = torch.optim.AdamW(model.parameters(), lr=LR*2,
                                   betas=(0.9,0.95), weight_decay=0.1)
    for _s in range(n_retry):
        lr_s = LR*2*0.5*(1+math.cos(math.pi*_s/n_retry))
        for pg in opt_retry.param_groups: pg['lr'] = lr_s
        model.train(); x, y = get_batch(); _, l = model(x, y)
        opt_retry.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_retry.step()
    v_basin = eval_val(model); pc_b = phi_clean(model)
    tau_b = gluing_defect(model)
    step_basin += n_retry
    print(f"  After τ-retry ({n_retry}CE@LR×2): val={v_basin:.4f}  "
          f"Φ_cl={pc_b}/5  τ={tau_b:.2f}")

print()
# ── END PHASE 3 ───────────────────────────────────────────────────────────────
# Compare: step_basin (new) vs ~170 (original with τ-retry)
print(f"  Phase 3 total CE: {step_basin}  "
      f"({'GEO-STOP' if geo_stopped else 'LOSS-PLATEAU'})")
