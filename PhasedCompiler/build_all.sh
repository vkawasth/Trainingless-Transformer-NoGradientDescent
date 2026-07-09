#!/usr/bin/env bash
# build_all.sh — compile the schedule compiler, compile geo.sched -> geo.ops,
# and build the C++/pybind11 bridge. Run from the directory that CONTAINS both
# vgpuc/ and geo_bridge/ (that's how the tarballs extract).
#
#   ./build_all.sh              # build everything
#   ./build_all.sh run          # build everything, then run the full geo pipeline
#
# Prereqs: a C++23 clang++ (LLVM 16+ / libc++), and a Python venv (fact_env)
# with torch, numpy, scipy, and pybind11 installed.

set -euo pipefail   # stop on first error, treat unset vars as errors

# ---- resolve tools -------------------------------------------------------
# Prefer whatever clang++ is on PATH (your Homebrew LLVM 22). Override by
# exporting CXX before running, e.g. CXX=/path/to/clang++ ./build_all.sh
CXX="${CXX:-$(command -v clang++)}"
PY="$(command -v python3)"

if [[ -z "$CXX" ]]; then
  echo "ERROR: no clang++ found on PATH. Set CXX=/full/path/to/clang++" >&2
  exit 1
fi

echo "==> using C++ compiler: $CXX"
"$CXX" --version | head -1
echo "==> using Python:       $PY"
"$PY" --version

# Sanity: confirm the Python has the packages the bridge needs.
echo "==> checking Python packages (torch/numpy/scipy/pybind11)"
"$PY" - <<'PYCHECK'
import importlib, sys
missing = []
for m in ("torch", "numpy", "scipy", "pybind11"):
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({e})")
if missing:
    print("  MISSING:", ", ".join(missing)); sys.exit(1)
print("  all present")
PYCHECK

# ---- locate projects -----------------------------------------------------
ROOT="$(pwd)"
if [[ ! -d "$ROOT/vgpuc/sched" || ! -d "$ROOT/geo_bridge" ]]; then
  echo "ERROR: run this from the directory containing both vgpuc/ and geo_bridge/" >&2
  echo "       (found: $(ls -d "$ROOT"/*/ 2>/dev/null | tr '\n' ' '))" >&2
  exit 1
fi

# ---- 1) build the schedule compiler --------------------------------------
echo
echo "==> [1/3] building schedule compiler (schedc)"
cd "$ROOT/vgpuc/sched"
"$CXX" -std=c++23 -O2 -I../include -Wall -Wextra schedc.cpp -o schedc
echo "    schedc built"

# ---- 2) compile the schedules to op streams ------------------------------
echo
echo "==> [2/3] compiling schedules -> op streams"
./schedc geo.sched   geo.ops
./schedc train.sched train.ops
echo "    geo.ops and train.ops written"
echo "    (geo.ops phases:)"
grep -E 'SADDLE|MFPUMP|BASIN|SNAPPER|TOPOGATE|K0_SPLIT|JOINT_CE|LANCZOS' geo.ops \
  | sed 's/^/      /' || true

# ---- 3) build the bridge (CMake + pybind11) ------------------------------
echo
echo "==> [3/3] building the C++/pybind11 bridge"
cd "$ROOT/geo_bridge"
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DPython3_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
cmake --build build -j
echo "    bridge built: geo_bridge/build/{geo_bridge,run_schedule}"

echo
echo "=========================================================="
echo "BUILD COMPLETE"
echo "Run the full geometry pipeline with:"
echo "  cd geo_bridge"
echo "  VENV_PYTHON=\$(which python3) ./build/run_schedule ../vgpuc/sched/geo.ops py geo_phases"
echo "=========================================================="

# ---- optional: run it ----------------------------------------------------
if [[ "${1:-}" == "run" ]]; then
  echo
  echo "==> running full geometry pipeline (this builds spectral E_init +"
  echo "    floor gradient first, so expect a minute of silence)..."
  cd "$ROOT/geo_bridge"
  VENV_PYTHON="$PY" ./build/run_schedule ../vgpuc/sched/geo.ops py geo_phases
fi
