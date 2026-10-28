# Study 11: deviations from the frozen plan

The frozen script is kept as `run_study11_as_frozen.py.txt`. Its SHA-256 matches `FROZEN_PREREG_study11.txt`.

## DV1 (2026-10-27): empty fibre for pooled pair tables (11d)
- **What happened.** The frozen script crashed in 11d. For some truths, the pooled pair tables (a 50/50 mixture of the target with a
  benign product population) admit no two-type explanation: the fitted v gives an empty segment. The plan had not anticipated this.
  11a-11c had completed and were not affected.
- **Rule adopted.** When the reported evidence has no two-type explanation, both the naive and the skeptical regulator find nothing
  good enough and reject (q_hi = q_mid = infinity). This follows from robust satisficing on an empty ambiguity set. It was fixed before
  any 11d outcome had been computed, since the crash happened while the inputs were being built.
- **Effect on the predictions.** Only strategy Q (pooled) is affected. D4 asks whether pooling can still cause false approvals under
  skepticism; truths with an empty pooled fibre can now only be rejected. The share of such truths is reported
  (pooled_empty_share).
- **Rerun.** The full script was rerun. It is seeded, so 11a-11c reproduce identically.
