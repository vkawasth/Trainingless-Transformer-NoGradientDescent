#!/usr/bin/env bash
# Runs every check script for "Shadows, Fibres and Secants of the Posterior Tower" and reports the totals.
set -u
cd "$(dirname "$0")"
fail=0
for s in check_shadows.py check_tropical.py check_cohomology.py check_hallucination.py check_barr.py check_twocat.py check_infogeom.py check_places.py companion/check_agents.py engine/check_pce.py engine/check_icl.py engine/check_integrate.py engine/check_complex.py engine/check_cohomology_engine.py engine/check_lift.py engine/check_gs.py engine/check_places_engine.py engine/check_overlap.py engine/check_overlap2.py engine/check_cover.py engine/check_perturb.py engine/check_strata.py engine/check_cbd.py engine/check_toric.py engine/check_updates.py engine/check_persist.py engine/check_persist2.py engine/check_level2.py engine/check_scheme.py engine/check_ladder.py engine/check_noncomm.py engine/check_masked.py engine/check_masked_gauge.py engine/check_qglue.py engine/check_npa.py engine/check_fingerprint.py engine/check_cbdfrac.py engine/check_ball.py engine/check_coxeter.py engine/check_birth.py check_grid.py check_bc_test.py; do
  echo "== $s"
  out=$(python3 "$s" 2>&1); echo "$out" | grep -E "^(PASS|FAIL)" | sed 's/   \[.*//'
  echo "$out" | grep -q "^FAIL" && fail=1
  echo "$out" | tail -1
done
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
