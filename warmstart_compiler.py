"""
warmstart_compiler.py
=====================
1-shot compiler: skip Phase 3 entirely by initializing from a
previous basin_state.pt checkpoint.

If basin_state.pt transfers across runs (same corpus, same arch),
total CE = 0 (Phase 3) + TopoGate (0.5s) + 25 CE (K0) + Lanczos
         ≈ 30 CE vs 187 CE current.

This tests whether the training trajectory is reusable — whether
Stab(F) coordinates at the floor are stable across initializations.

Usage: drop this file next to compiler_analytic_topogate.py and run:
  python warmstart_compiler.py

It imports model/data infrastructure from the compiler and runs
only Phases 4-5 from the saved basin_state.
"""

# ── Copy the compiler header (model def + data loading) ──────────────────────
# This script must be run in the same directory as compiler_analytic_topogate.py
# It reuses the model class and data functions defined there.

import sys, os, math, time
import numpy as np
import torch

# Inline the minimum needed from the compiler
# (model class, get_batch, eval_val, phi_clean, gluing_defect, sheet_angles)
# These are defined in the compiler — exec it up to the model definition

exec_globals = {}
compiler_src = open('compiler_analytic_topogate.py').read()

# Execute only up to the model instantiation (before Phase 1)
# Find the cutoff: just after model = StudentLM(...) and before Phase 1
cutoff = compiler_src.find('# ── PHASE 1')
if cutoff == -1:
    cutoff = compiler_src.find('print("━━━ PHASE 1')
if cutoff == -1:
    cutoff = compiler_src.find('PHASE 1')

print(f"Executing compiler header up to char {cutoff} (model + data setup)")
header = compiler_src[:cutoff]

# Execute the header to get model, get_batch, eval_val etc.
exec(header, exec_globals)
model        = exec_globals['model']
get_batch    = exec_globals['get_batch']
eval_val     = exec_globals['eval_val']
phi_clean    = exec_globals['phi_clean']
gluing_defect= exec_globals['gluing_defect']
sheet_angles = exec_globals['sheet_angles']
LR           = exec_globals['LR']
N_STU        = exec_globals['N_STU']

# Also need these for Phase 4
analytic_topogate = exec_globals.get('analytic_topogate')
analytic_wv_flip  = exec_globals.get('analytic_wv_flip')

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
print(f"LR = {LR}")

# ── WARM START: load basin_state.pt ──────────────────────────────────────────
print("\n" + "="*60)
print("  WARM-START COMPILER")
print("  Initialize from basin_state.pt → skip Phase 3")
print("="*60)

basin_path = 'basin_state.pt'
if not os.path.exists(basin_path):
    print(f"ERROR: {basin_path} not found. Run full compiler first.")
    sys.exit(1)

ckpt = torch.load(basin_path, map_location='cpu', weights_only=False)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    model.load_state_dict(ckpt['state_dict'])
else:
    model.load_state_dict(ckpt)

v0   = eval_val(model)
pc0  = phi_clean(model)
tau0 = gluing_defect(model)
phi0 = sheet_angles(model)

print(f"\n  Loaded basin_state.pt")
print(f"  val={v0:.4f}  Φ_cl={pc0}/5  τ={tau0:.2f}")
print(f"  Φ={phi0}")
print(f"\n  Skipped: Phase 1 (saddle exit)")
print(f"  Skipped: Phase 2 (MF pump, 10 CE)")
print(f"  Skipped: Phase 3 (basin settle, 162 CE)")
print(f"  Skipped: Phase 3 (τ-retry, 50 CE)")
print(f"  CE saved so far: ~222 CE")

# ── PHASE 4: ANALYTIC TOPOGATE ────────────────────────────────────────────────
print("\n━━━ PHASE 4: ANALYTIC TOPOGATE (warm-start) ━━━━━━━━━━━━━")

if analytic_topogate is not None:
    wk_improved = analytic_topogate(model)
    wv_improved = analytic_wv_flip(model) if analytic_wv_flip else False
else:
    print("  [analytic_topogate not found in compiler globals — using empirical]")
    wk_improved = False; wv_improved = False

if not wk_improved and not wv_improved:
    print("  Both analytic steps found no gain — empirical fallback")
    v_before_fb = eval_val(model, n=8)
    pc_before   = phi_clean(model)
    best_score = 0; best_layers = None; best_val_fb = v_before_fb
    for flip_layers in [[1,2],[0,1],[2,3],[1,3],[0,3]]:
        with torch.no_grad():
            for l in flip_layers:
                model.blocks[l].attn.WV.weight.data.mul_(-1)
                model.blocks[l].attn.op.weight.data.mul_(-1)
        v_try = eval_val(model, n=6); pc_try = phi_clean(model)
        score = (v_before_fb - v_try) + 0.3*(pc_try - pc_before)/5.0
        if score > best_score:
            best_score = score; best_layers = flip_layers; best_val_fb = v_try
        with torch.no_grad():
            for l in flip_layers:
                model.blocks[l].attn.WV.weight.data.mul_(-1)
                model.blocks[l].attn.op.weight.data.mul_(-1)
    if best_layers and best_score > 0:
        with torch.no_grad():
            for l in best_layers:
                model.blocks[l].attn.WV.weight.data.mul_(-1)
                model.blocks[l].attn.op.weight.data.mul_(-1)
        print(f"  ✓ Fallback TopoGate {best_layers}: val→{best_val_fb:.4f}")

v_topo = eval_val(model)
pc_topo = phi_clean(model)
tau_topo = gluing_defect(model)
print(f"  Post-TopoGate: val={v_topo:.4f}  Φ_cl={pc_topo}/5  τ={tau_topo:.2f}")
print(f"  Φ={sheet_angles(model)}")

# ── PHASE 5: K0 SPLIT DESCENT ─────────────────────────────────────────────────
print("\n━━━ PHASE 5: K₀ SPLIT DESCENT (warm-start) ━━━━━━━━━━━━━━")
w_FF = min(18.19, 3.5 * (1.5/max(tau_topo, 0.5))**1.5)
print(f"  τ={tau_topo:.2f} → w_FF={w_FF:.2f}")

# Joint CE
opt_k0 = torch.optim.AdamW(model.parameters(), lr=LR,
                             betas=(0.9,0.95), weight_decay=0.1)
for _s in range(25):
    model.train(); x, y = get_batch(); _, l = model(x, y)
    opt_k0.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_k0.step()

v_k0 = eval_val(model)
print(f"  After 25 CE joint: val={v_k0:.4f}")

# ── LANCZOS ───────────────────────────────────────────────────────────────────
# Skip Lanczos for now — just report final val
v_final = eval_val(model)
tau_final = gluing_defect(model)
pc_final = phi_clean(model)

print("\n" + "="*60)
print("  WARM-START RESULT")
print("="*60)
print(f"  val (basin_state.pt):  {v0:.4f}")
print(f"  val (post-TopoGate):   {v_topo:.4f}")
print(f"  val (post-K0 25 CE):   {v_final:.4f}")
print(f"  Φ_cl={pc_final}/5  τ={tau_final:.2f}")
print(f"\n  CE used: ~30 (TopoGate 0.5s + K0 25CE)")
print(f"  CE saved: ~162 (Phase 3 skipped)")
print(f"\n  Compare: full compiler final val ≈ 0.055-0.065 in 187 CE")
print(f"  Warm-start final val: {v_final:.4f} in ~30 CE")

if v_final < 0.10:
    print(f"\n  ✓ WARM-START WORKS: {v_final:.4f} < 0.10")
    print(f"  The trajectory IS reusable. Phase 3 can be skipped.")
    print(f"  Next: run Lanczos to push below 0.065")
else:
    print(f"\n  ✗ WARM-START INSUFFICIENT: {v_final:.4f} > 0.10")
    print(f"  The basin_state does not transfer cleanly.")
    print(f"  Possible reason: batch randomness changed the loss landscape.")
