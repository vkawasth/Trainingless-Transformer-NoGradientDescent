# Study 26 — results (one end-to-end example)

Frozen plan: PREREG_study26.md and world26.py (FROZEN_PREREG_study26.txt, 2026-11-11 16:20 UTC, before any table was generated).
Run: run_study26.py -> run_study26_output.txt, results_study26.json. Deviations: DEVIATIONS_study26.md (D1, D2, D3).
Checks: engine/check_study26.py (XS1-XS4: freeze intact, run reproducible, identities hold, every failure recorded).

Pre-registered checks: 13 of 19 passed (plus XE4b, a deviation check).
Failed: XE2 (flagging caught 2/10 other-population sources), XE3 and XE4 (estimated fibre empty with or without flagged sources),
XE9 (operationalisation: L * distance -> 0, convergence faster than 1/L), XE10 (no selection wall within +-40 counts; the advisory
switched at 7 counts), XE18 (rho_q = 0.035 < 2 eta = 0.180; the coalition is not flagged).

Main numbers
- Fibre: dim 9 (32 - 23); with {M,C,R,Y}: 6; Fisher kernels 9 and 6.
- Paths given M = 1 (exact data): pi1 [0.523, 0.680], pi2 0.575, pi3 [0.300, 0.488], union [0.638, 0.680].
- Ladder Delta(K): none 0.168; C=0 0.151; C=0,R=1 [-0.178, 0.517]; C=0,R=1,H=0 [-0.389, 1]; truth 0.200; decision (tau = 0.10) open.
- VOI of R: [0, 0.080] on development data; 0.020 after {M,C,R,Y}; Theorem 92's criterion agrees both times.
- Selection: f = (0.317, 0.683), g = (0.448, 0.351, 0.201); leximin levels 0.1006 (bright column), 0.1084, 0.2427, 0.3393.
- Composition: maximum entropy 3e-17, leximin 0.034. System independence: 7e-18 and 0.003.
- Cycle: consistent 1.211 (glues), twisted 2.314 (SE 0.036, no global law), path glues; NCF 0.980 vs 0.843; 16 global sections each.
- History: path independence 2e-16; foreign {M,R,Y} table TV 0.089, trace residuals 6e-15 (MRY) and 1.6 (MHY); honest +0.102,
  history B +0.228, {M,R,Y} slice [-0.123, 0.466] (crosses tau), fibre [-0.389, 1].
- Protection: g(P*) 0.168, tau2 0.208, pooled g 0.295; rho_q 0.035, eta 0.090, Delta_q honest 0.011, coalition 0.063.
- Holdout calibration: 16/16 cells covered (median width 0.178).
