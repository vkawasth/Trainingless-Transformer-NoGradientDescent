#!/usr/bin/env bash
# compare_modes.sh — coordinate-driven basin comparison, using the paper's exact
# failure-mode boundaries (Section 8.3 falsification table, p.26).
#
# The runtime coordinate system is (Phi, sigma, tau) -- verified against the
# source (sheet_angles/phi_clean, compute_rm2_sigma_inline, gluing_defect). We
# do NOT compare final val (a Level-2 shadow) and we do NOT copy RNG (that would
# reduce the compiler to gradient descent). Instead we classify each mode's
# ENDPOINT by the geometric coordinates and check both land in the same basin.
#
# Failure-mode boundaries (from the paper):
#   tau < 0.5        -> WRONG BASIN     (gradient-norm ratio collapsed)
#   tau > 8.0        -> ORBIT SHATTERED (tau spike)
#   phi < 3          -> WRONG SHEET     (off-wall phases, not crystallized)
#   otherwise (phi>=4, tau moderate) -> FLOOR / correct basin
#
# Usage:  ./compare_modes.sh [seed] [corpus]

set -uo pipefail
SEED="${1:-99}"
CORPUS="${2:-real}"
PY="${VENV_PYTHON:-$(command -v python3)}"
OPS="../vgpuc/sched/geo.ops"

if [[ ! -x ./build/run_distributed ]]; then
  echo "ERROR: build run_distributed first (cmake --build build)" >&2; exit 1
fi

# Run distributed TWICE with the same seed, different checkpoint dirs. Because
# RNG is deliberately NOT threaded, the two runs take different random paths.
# The coordinate-driven claim: they still land in the same basin.
echo "=== RUN A (distributed, seed=$SEED) ==="
mkdir -p "ckpt_A_$SEED"
VENV_PYTHON="$PY" ./build/run_distributed "$OPS" py "ckpt_A_$SEED" "$SEED" "$CORPUS" 2>&1 \
  | grep -E "basin|snapper|final_val|floor" || true

echo
echo "=== RUN B (distributed, same seed, different random path) ==="
mkdir -p "ckpt_B_$SEED"
VENV_PYTHON="$PY" ./build/run_distributed "$OPS" py "ckpt_B_$SEED" "$SEED" "$CORPUS" 2>&1 \
  | grep -E "basin|snapper|final_val|floor" || true

echo
echo "=== COORDINATE-DRIVEN BASIN COMPARISON (paper's failure boundaries) ==="
"$PY" - "ckpt_A_$SEED/results.json" "ckpt_B_$SEED/results.json" << 'PYEOF'
import json, sys

def classify(phi, tau):
    # paper's failure-mode boundaries
    if tau is None or phi is None: return "UNKNOWN"
    if tau < 0.5:  return "WRONG_BASIN (tau collapsed)"
    if tau > 8.0:  return "ORBIT_SHATTERED (tau spike)"
    if phi < 3:    return "WRONG_SHEET (off-wall phases)"
    return "FLOOR/correct basin"

def endpoint(path):
    with open(path) as f: r = json.load(f)
    phases = r.get("phases", [])
    # last phase that recorded geometry coordinates
    phi = tau = sig = None
    for p in phases:
        if "geom_phi" in p:   phi = p["geom_phi"]
        if "geom_tau" in p:   tau = p["geom_tau"]
        if "geom_sigma" in p: sig = p["geom_sigma"]
    return r.get("final_val"), phi, sig, tau

a_path, b_path = sys.argv[1], sys.argv[2]
try:
    av, ap, asg, at = endpoint(a_path)
    bv, bp, bsg, bt = endpoint(b_path)
except Exception as e:
    print("  could not read results:", e); raise SystemExit(1)

def fmt(x, f="{:.3f}"):
    return "  n/a" if x is None else f.format(x)

print(f"  {'':10}{'run A':>12}{'run B':>12}")
print(f"  {'final val':10}{fmt(av):>12}{fmt(bv):>12}")
print(f"  {'phi':10}{fmt(ap,'{:d}') if ap is not None else '  n/a':>12}{fmt(bp,'{:d}') if bp is not None else '  n/a':>12}")
print(f"  {'sigma':10}{fmt(asg):>12}{fmt(bsg):>12}")
print(f"  {'tau':10}{fmt(at):>12}{fmt(bt):>12}")
print()
ca, cb = classify(ap, at), classify(bp, bt)
print(f"  run A basin: {ca}")
print(f"  run B basin: {cb}")
print()
same = (ca == cb) and ca.startswith("FLOOR")
if same:
    print("  RESULT: SAME BASIN via different random paths.")
    print("    The endpoint geometry (Phi, tau) matches despite no shared RNG.")
    print("    -> coordinate-driven replaceability supported for this seed.")
elif ca == cb:
    print(f"  RESULT: both classified '{ca}' -- consistent, but not the floor basin.")
else:
    print("  RESULT: DIFFERENT basins -- the two random paths diverged geometrically.")
    print("    Inspect per-phase geom_tau / geom_phi in each results.json.")
PYEOF
echo
echo "note: strip energy E is recorded as null pending its definition;"
echo "      classification uses (Phi, tau) per the paper's failure boundaries."
