# Study 27b — deviations from the frozen specification (SHA-256 b90c6804...)

1. (before the run; implementation impossibility) Section 1 records the braid key (N = 2) as a secondary encoding. Computing it needs the
   Artin action on the free group, whose words grow exponentially with the history: on full 27b histories the computation ran out of
   memory in a code test. The braid key is therefore computed only on safe-pad pairs with at most 4 events (up to 200 pairs) and reported
   as such. R2 itself (the reduced RAAG normal form) is unaffected, and no hypothesis uses the braid key.

Code tests before the run: run_study27b.py has a smoke mode (SMOKE27B=1: seeds offset by 100, 40 histories per regime, 3 runs per
scenario, results written to /dev/null). It was used only to find crashes; its numbers were not read.

2. (implementation detail, recorded) Section 3 verifies transformations by the activation map of the trace. In the decision arm the
   adaptive coalition's candidates were also required to keep source 3's activation round under the recorded rounds, because the
   decision trajectory depends on it (memo checklist: "verify that each transformation preserves ... the decision trajectory").
3. (recorded) Toy WB1-WB2 hold by construction (C1 and C2 are given the same event word and the same source reports); the code records
   them rather than testing them. WB3-WB5 and WB7 are computed.
4. (after the run, exploratory, not pre-registered; no verdict changed) H1 failed. study27b/exploratory_h1.py refits the same learner on
   the same histories with the frozen R1 features plus counts of the covering pairs of the dependence order (trace invariants, checked
   invariant under independent swaps): E2 error 0.094, against R0 0.140 and frozen R1 0.194. So the failure reflects the frozen R1
   feature map (layer indicators without dependence pairs), not information the trace lacks. This is exploratory and is reported as such.
