#!/usr/bin/env bash
# Runs every check script for "Shadows, Fibres and Secants of the Posterior Tower" and reports the totals.
set -u
cd "$(dirname "$0")"
fail=0
for s in check_shadows.py check_tropical.py check_cohomology.py check_hallucination.py check_barr.py check_twocat.py check_infogeom.py check_places.py companion/check_agents.py check_bc_test.py; do
  echo "== $s"
  out=$(python3 "$s" 2>&1); echo "$out" | grep -E "^(PASS|FAIL)" | sed 's/   \[.*//'
  echo "$out" | grep -q "^FAIL" && fail=1
  echo "$out" | tail -1
done
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
