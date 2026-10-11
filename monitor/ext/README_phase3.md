# Phase III: positive control and six-wave composition

- `snowqueen/snowqueen.py`: Snow Queen (Cervantes & Dzhafarov 2018) recomputed from the published tables with the programme's CbD code. It reproduces the published statistic and interval.
- `c19/`: the C19PRC panels.
  - Plans: `PREANALYSIS_AMENDMENT_C19PRC_UK.md` (UK primary) and `PREANALYSIS_AMENDMENT_C19PRC_IE.md` (Irish replication), both dated 2026-10-10 and written before any data values were read.
  - `latentT.py` is the latent Markov model over T waves with a bootstrap G². `synth_latentT.py` checks it: a planted Markov chain fits and planted second-order memory is rejected.
  - `synth_c19.py` builds synthetic data in both SPSS layouts. `c19.py uk|ie` runs the analyses.
  - Results are frozen in `results/frozen/`, with SHA-256 hashes in `results/FROZEN_UK.txt`; the comparison is in `results/results_c19_compare.json`.
  - Deviation, a loader fix: `WAVE_1` and `WAVE_2` are empty in the UK W1W2 file, so a wave's respondents are taken to be those with that wave's scores.
- Data are not included. They are the user's OSF downloads (`osf.io/v2zur` for the UK strand; the Irish Mental Health ASAP archive).

## After the independent review (2026-10-10)
- `snowqueen/cbd_split_power.py`: the Snow Queen control checks the rank-n cyclic criterion only. This script plants contextuality in two-context 3×3 systems and measures the size and power of the pipeline behind the clinical and panel non-detections (`cbd_split` with the boundary null). Size ≤ 0.075; power ≤ 0.10 for deficits ≤ 0.04 and about 0.5 for deficits 0.08–0.16. Output: `snowqueen/results_cbd_split_power.json`.
- `c19/b2_equal_starts.py uk|ie|power`: reruns the four-wave test with 8 EM starts for the observed table and for every bootstrap refit (the frozen run used 20 and 4). The p-values are unchanged (UK 0.886, free 0.723; Ireland 0.448, free 0.317). Power against planted second-order memory: 3/8 at n = 771, 0/8 at n = 327. `latentT.test` now takes `refit_starts`.
- Selection (analysis E): weighting on the (X, Y) cells makes the three tables compatible by construction, so `cf_after_ipw = 0` always. Only the available-case test is evidence. This also applies to the MIMIC-IV kit's analysis E.
- Further deviations, all disclosed in the paper's §"Deviations from the plans":
  - Irish W5 recontact is defined as taking part at W1.
  - The UK strand was made primary before any Irish values were examined.
  - Heart Disease: `ca` is clipped to 0/1/2+ and age is banded <50/50–59/60+.
- `synth_latentT.py` now runs its check only as a script (it used to exit on import).
- `c19/raw/` holds the downloaded data for reruns. Never package or share it.

## Banding interchange test (2026-10-10, post-hoc amendment)
- `c19/PREANALYSIS_BANDING_INTERCHANGE.md` is the plan; `c19/banding_interchange.py uk|ie` runs the test.
- `synth_banding.py` and `synth_banding_calib.py` are the adapter check and the calibration run on planted raw chains.
- The results are frozen in `results/FROZEN_BANDING.txt`.
- The interchange law (Algebras Prop. 6.1), with banding as the middle merge, splits the banded composition failure exactly into a raw-chain part and an interchange defect.
- Banding is not sufficient in either strand, and the monitor's banded Markov null is 41–51% too small.
- The failure still survives the corrected null: p = 0.0025 in the UK and 0.022 in Ireland.
