# Study 8 — Origin fingerprint of the Snow Queen data (pre-registration)

Registered 2026-10-25, before any response frequencies were seen. Only the design (Cervantes & Dzhafarov, arXiv:1711.00418v8,
"Snow Queen is evil and beautiful") was read: one story, four contexts, a Bell-type cyclic system of rank 4, about 450 respondents per
context, joint proportion tables in Table 2 (correct responses only) and Table 3 (all responses), sample sizes in Table 1. The paper's
title states that the authors found contextuality; that conclusion is known before this registration and is recorded here as prior
knowledge. Nothing patient-level or credentialed is involved: the inputs are published aggregated proportions.

## Data and coding
- Variables: A_x (character choice) for x = 1 (Gerda / Troll), 2 (Snow Queen / Old Finn Woman); B_y (characteristic) for
  y = 1 (Beautiful / Unattractive), 2 (Kind / Evil). Coding: the first-listed option is +1 (Gerda, Snow Queen, Beautiful, Kind).
- Contexts (x, y): (1,1), (1,2), (2,1), (2,2), each a 2x2 joint table of proportions; counts = proportion x n from Table 1.
- Two datasets: D2 = Table 2 (correct only), D3 = Table 3 (all responses).
- Transcription: tables are taken from the paper text; each table must sum to 1 within rounding (|sum - 1| <= 0.01) and its stated
  marginals must match the cells; otherwise it is re-read, and any unresolved discrepancy is logged in DEVIATIONS.md.

## Quantities (engine, frozen code: pce/fingerprint.py, pce/cbd.py, pce/npa.py, pce/integrate.py)
For each dataset: per-variable direct influence s_q (TV between the two copies' marginals) and their sum Delta; the linear gap; the
Contextuality-by-Default degree c (> 0: contextual beyond direct influence) and the Kujala-Dzhafarov criterion; the classical fraction
lam*; the dimension-free quantum fraction lam_Q (NPA level 1+AB, computed directly on the tables, also when they signal); the origin
label of the fingerprint. Uncertainty: parametric bootstrap, 2000 replicates, multinomial per context with the Table 1 sample size; 95%
percentile intervals.

## Predictions (made now)
- P1 (structural, not empirical): on D2 every context is supported on two cells with parities (+,+,+,-), so lam* = 0 and lam_Q = 0
  exactly. Confirmation is a consistency check of the transcription and code, not evidence.
- P2: D2 shows direct influence: the 95% bootstrap interval of Delta excludes 0.
- P3: on D2 our CbD degree c and the Kujala-Dzhafarov criterion agree in sign, and the fingerprint's label agrees with the authors'
  stated conclusion (contextual beyond direct influence).
- P4 (the empirical test): on D3, the fingerprint label is "contextual beyond direct influence" iff the 95% bootstrap lower bound of c is
  > 0; we predict it is. If the lower bound is <= 0 the result is recorded as "not shown beyond direct influence".
- P5: on D3, lam* < lam_Q (quantum saves strictly more than classical) with the bootstrap interval of lam_Q - lam* excluding 0. No
  directional prediction about the size.
- P6: all quantities except label-dependent signs are invariant under recoding any variable (+1 <-> -1); checked on both datasets.

## What each outcome means
- P4 confirmed: the fingerprint locates the data above the direct-influence rung, on the contextual rung, in agreement with CbD.
- P4 not confirmed: incoherence in D3 is explained by direct influence (the bottom rung), consistent with most behavioural data.
- P5 tells whether a quantum-cone model explains more of the evidence than a classical one; it is not evidence of quantum mechanics in
  cognition (any model class whose cone contains the quantum set would do as well).

## Deviations
Any change after data are seen is logged in DEVIATIONS.md with reason and date; the frozen plan is not edited.
