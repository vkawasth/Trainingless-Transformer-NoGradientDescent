# Phase 4: geometry as a forecaster, restricted selection, structural events, and a generalized tower

`PREANALYSIS_PHASE4.md` is the plan. Its hash is in `FROZEN_PLAN.txt`; the results are hashed in `FROZEN_RESULTS.txt`.

## Data
`load_physionet.py` builds per-stay hourly MAP × HR band tables and the lactate band. It writes a patient-level cache *outside* this folder (`$PHYSIONET_DIR/cache`). Never share that cache.

## Work packages
Each script runs its planted check with `synth` and the real analysis with `run`.
- `geom.py`: vertex counts of 3×3 coupling polytopes (checked against `bridges/fukaya/nef33.py`), wall signs, tear margin, breaks and steepness.
- `wp1_clinical.py synth|run`: does per-stay ICU geometry forecast death or shock onset beyond the margins?
- `wp1_gss.py synth|run`: does GSS item-pair geometry forecast the next chamber change?
- `wp2_selection.py synth|run`: restricted selection models with model-specific nulls (PhysioNet A and B, Diabetes 130, C19PRC UK).
- `structural_events.py`: blow-ups and blow-downs, chamber flips and breaks, tested against death, partisanship and LLM error shifts.
- `wp4_tower.py synth|run`: the tower with three reliability floors (fixed, evidence rate, family of floors) on arms R0–R4.

## Headline results
- **Forecasting.** Geometry does not forecast ICU death or shock beyond the margins; 98.8% of per-stay systems are broken. It does forecast GSS chamber changes (gain 0.041, interval 0.020–0.062).
- **Selection.** Restricted models fit UK attrition (depression-only, demographics, or both). They fail for ICU lactate and diabetic HbA1c.
- **Structural events.** No ICU structural event is associated with death. GSS partisanship gives Spearman 0.35, p = 0.11.
- **Tower.** All three hypotheses hold:
  - the evidence floor replicates in 2 of 3 shift arms;
  - the family of floors earns its keep in 3 of 3;
  - no variant harms the same-population arm.

## Deviations
- The WP2 planted external design was changed before the real run.
- The WP3 real run happened in the same command as its first planted check, whose planted confounding was too weak. The check was then redone and passed.
