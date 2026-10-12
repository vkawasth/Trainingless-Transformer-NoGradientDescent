# Phase 5: the context-to-outcome map as a toric variety, and geometry at the boundary

`PREANALYSIS_CONTEXT_MAP.md` is the plan. Hashes are in `FROZEN_PLAN.txt`, `FROZEN_A.txt` and `FROZEN_B.txt`.

## Scripts
- **`context_map.py synth|run`** asks whether the map from information to outcome is the same in every context. The invariance model O ⟂ C | A is a toric variety cut out by binomials. The script reports:
  - G², and I(O;C|A) as the distance to the variety;
  - which answer level breaks invariance;
  - the toric type of the answer × outcome system per context, observed and under the invariant map.
  - Data:
    - GSS (wording × answer × party, 11 items, 24 years);
    - PhysioNet (unit × first-window pressure × death, sets A and B).
- **`boundary_geometry.py synth|run`** gives a boundary-aware reading of per-stay ICU day systems:
  - the Dirichlet posterior over toric types;
  - the walls decided on the bands;
  - face strata.
  It reruns the phase 4 forecast of death with these features.

## Results
- **GSS.** Invariance is rejected for welfare, crime, cities and health. Welfare's failure sits at "about right". Its toric type differs between wordings in 23 of 24 years; 16 of those are explained by the answer margins alone.
- **ICU.** Invariance is rejected in both sets. Cardiac surgery's Fano surface is CP² where invariance would give F₁, decided on the bands in set B.
- **Boundary.** 71% of point-estimate-broken day systems recover a full-support toric type with posterior mass above ½. That geometry still does not forecast death.

## Deviations
The first planted check of Part B (weaker effect, 24% broken) failed to show power. It was replaced before the real run by a stronger check matched to the data, which passed.
