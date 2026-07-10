#!/usr/bin/env bash
# run_variance.sh — run the full geometry pipeline N times with different seeds
# and summarize the final val distribution. Run from geo_bridge/.
#
#   ./run_variance.sh          # 5 runs, seeds 99..103
#   ./run_variance.sh 8        # 8 runs, seeds 99..106
#
# Requires the bridge already built (./build/run_schedule) and VENV_PYTHON set
# or resolvable via `which python3`.

set -uo pipefail

N="${1:-5}"
PY="${VENV_PYTHON:-$(command -v python3)}"
OPS="../vgpuc/sched/geo.ops"
MODULE="geo_phases"

if [[ ! -x ./build/run_schedule ]]; then
  echo "ERROR: ./build/run_schedule not found. Build first (build_all.sh)." >&2
  exit 1
fi

echo "running $N times (seeds 99..$((99 + N - 1)))"
echo "seed   floor    final_val   gap_vs_floor   basin_val   snapper_val   phase5"
echo "----   -----    ---------   ------------   ---------   -----------   ------"

vals=()
floors=()
for ((i = 0; i < N; i++)); do
  seed=$((99 + i))
  out="$(SCHED_SEED=$seed VENV_PYTHON="$PY" ./build/run_schedule "$OPS" py "$MODULE" 2>&1)"

  floor=$(echo "$out"  | awk '/\[floor\]/ {print $NF; exit}')
  final=$(echo "$out"   | awk '/eval\(15\)/ {v=$NF} END{gsub(/val=/,"",v); print v}')
  basin=$(echo "$out"   | awk '/basin    ->/ {gsub(/val=/,"",$3); print $3; exit}')
  snap=$(echo "$out"    | awk '/snapper  ->/ {gsub(/val=/,"",$3); print $3; exit}')
  if echo "$out" | grep -q "k0_split -"; then p5="k0_split"
  elif echo "$out" | grep -q "joint_ce -"; then p5="joint_ce"
  elif echo "$out" | grep -q "REACHED FLOOR"; then p5="snapper_floor"
  else p5="?"; fi

  gap="?"
  if [[ -n "${final:-}" && -n "${floor:-}" ]]; then
    gap=$(awk -v f="$final" -v fl="$floor" 'BEGIN{printf "%+.4f", f-fl}')
  fi

  printf "%-4s   %-6s   %-9s   %-12s   %-9s   %-11s   %s\n" \
    "$seed" "${floor:-?}" "${final:-ERR}" "$gap" "${basin:-?}" "${snap:-?}" "$p5"
  [[ -n "${final:-}" ]] && vals+=("$final")
  [[ -n "${floor:-}" ]] && floors+=("$floor")
done

if [[ ${#vals[@]} -gt 0 ]]; then
  echo "----"
  { printf '%s\n' "${vals[@]}"; echo "FLOORS"; printf '%s\n' "${floors[@]}"; } | "$PY" -c '
import sys
lines = sys.stdin.read().split()
i = lines.index("FLOORS")
xs = [float(x) for x in lines[:i]]
fs = [float(x) for x in lines[i+1:]]
xs.sort(); n = len(xs)
mean = sum(xs)/n
med = xs[n//2] if n%2 else (xs[n//2-1]+xs[n//2])/2
floor = sum(fs)/len(fs) if fs else float("nan")
print(f"n={n}  min={min(xs):.4f}  median={med:.4f}  mean={mean:.4f}  max={max(xs):.4f}")
print(f"corpus floor (mean) = {floor:.4f}")
print(f"beat floor: {sum(1 for x in xs if x < floor)}/{n} runs")
print(f"beat GD-400 (0.0917): {sum(1 for x in xs if x < 0.0917)}/{n} runs")
'
fi
