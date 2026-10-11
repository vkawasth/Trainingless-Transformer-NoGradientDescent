# SciFact: the decision layer on real scientific claims

**Data.** SciFact (Wadden et al., EMNLP 2020), supplied by the user as `data.tar.gz` (SHA-256 11c62128…76be). It is kept outside the code archive at `/home/claude/monitor_data_sf/data` (licence CC BY-NC 2.0). Set `SCIFACT_DIR` to use another copy.

**Setting.** Oracle abstracts: 919 training and 340 dev claim–abstract pairs. Three weak classical verifiers are trained on the training split only. Pretrained NLI models could not be downloaded in this environment.

## Analysis 1 (`scifact_verify.py`, plan `PREANALYSIS_SCIFACT.md`, `FROZEN_RESULTS.txt`)
Channel estimation by cross-validation, then credal decision rules.

| | result |
|---|---|
| H1 | **holds**. The joint-channel credal rule is 88.7% accurate when acting, on 21% of pairs; naive Bayes is 66.8% on all pairs. |
| H2 | **fails**. Assuming independent errors did not measurably cost accuracy. |
| H3 | **holds**. The margin κ* ranks correctness, AUC 0.59. |
| H4 | **fails**. Ranking by κ* gives a worse risk–coverage curve than naive Bayes confidence (0.249 vs 0.192). |

`explore.py` (exploratory): at the same coverage, the credal gate selects the same pairs as confidence thresholding, and it only ever asserts NEI.

## Analysis 2 (`conformal/`, plan `PREANALYSIS_SCIFACT_CONFORMAL.md` with amendment 1)
Each verifier's reliability is a conformal quantile, and decisions come from `phase7/decide.py`. This gives a guarantee: P(act and wrong) ≤ 0.05.

| | result |
|---|---|
| P1 | **holds**. No wrong act; sources missed the truth on 3.5% of pairs. |
| P2 | **fails**. The layer acts on only 4 of 340 pairs. |
| P3, P4, P5 | **hold**. P4 rests on 4 pairs. |

The risk–coverage curve stays within every guarantee level. Accuracy when acting is 57% at level 0.20.

## Disclosure
Analysis 2's plan was written without knowledge of analysis 1, and its plan file overwrote analysis 1's. The original was restored byte-identically from the session transcript, and all of analysis 1's hashes verify. Analysis 2 was then moved to `conformal/`, and two file paths were changed before its single run. Both are recorded in its `FROZEN_CODE.txt`.
