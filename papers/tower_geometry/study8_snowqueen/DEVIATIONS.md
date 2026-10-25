# Study 8 deviations log

1. 2026-10-25, after the tables were seen, before any analysis output on them. The frozen script bootstrapped both datasets with the
   total sample sizes of Table 1; the plan says counts come from Table 1, and for D2 (correct responses only) the right counts are the
   numbers of correct responses (425, 429, 410, 388). One line changed: `bootstrap(T, D.get("n_" + ds, D["n"]), ...)`. The frozen original is
   kept as analyze_study8_frozen.py (hash in FROZEN_PREREG_study8.txt); point estimates are unaffected, D2 intervals widen slightly.
2. Transcription note (not a plan change). The text extraction of Table 3, context 1, Gerda row was garbled. The cells were fixed from the
   printed marginals (Gerda 0.864, Beautiful 0.872, Unattractive 0.128) and the Troll row (0.029, 0.107): Gerda/Beautiful 0.843 and
   Gerda/Unattractive 0.020. Check: with these values the Kujala-Dzhafarov statistic reproduces the authors' published values exactly,
   0.452 (Table 2) and 0.279 (Table 3). Gerda/Unattractive = 0.021 would give 0.277. Table 3 context 1 sums to 0.999, and Table 2 context 4
   prints an Evil marginal of 0.627 against cells summing to 0.626; both are rounding, within the plan's 0.01 tolerance, and tables are renormalised.
