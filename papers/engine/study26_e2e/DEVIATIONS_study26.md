# Study 26 — deviations from the frozen plan

Written after the first run of run_study26.py stopped at the robust fibre, and before any downstream result was computed.

## What happened

- XE2 failed: the leave-one-out flagging rule flagged 5 sources: 2 of the 10 other-population sources, 2 of the 15 coalition sources and
  1 honest source. The other population has P(M=1) = 0.80 against 0.68 in the truth; each source's p-value was between 6e-5 and 0.15,
  so one at a time most of them look like sampling noise, while ten of them together move the pooled {M,C} table.
- XE3 failed in its second half and XE4 failed: the robust fibre J_z (z = 3) is empty both with and without the flagged sources.
- The plan did not say what to do when J_z is empty; the script stopped (a linear-fractional programme over an empty set).

These are recorded as findings. No threshold was changed.

## Deviation D1 (fallback when J_z is empty)

1. Elastic attribution: minimise sum_C s_C over laws P and scalars s_C >= 0, subject to |π_C(P) − p̂_C| ≤ tol_C (1 + s_C) cellwise, with
   tol_C the z = 3 tolerance. Report s_C per context. This uses only the pooled tables, not the field 'kind'.
2. Report z*, the smallest z at which J_z is non-empty (bisection), for the pool with flagged sources removed.
3. If the elastic slack is carried by one context alone, run the downstream estimated ranges on J_z with that context's pooled table
   removed (call it J_z^-), and say so wherever those ranges are reported. Exact statements on the oracle fibre J* are unaffected.
4. XE4 is reported as failed in its pre-registered form; its question (does the coherent coalition pass compatibility?) is re-asked
   on J_z^- as XE4b, labelled as a deviation check.

## Deviation D2 (numerical safeguard in the slice search of XE17)

The Nelder–Mead search over traces on {M,R,Y} reached trace values so large that the starting law had cells near zero, and the
I-projection's convergence guard stopped the run. The search coordinates are now mapped through 12 tanh(t/12), so each trace
coefficient is bounded by 12 in absolute value, and a projection that does not converge is treated as infeasible. The slice range
reported is therefore an inner estimate of the slice's range for traces bounded by 12, which is what XE17 compares with the fibre range.

## Recorded during the same run (no change made)

- XE10 failed: the wall scan k = −40..40 crossed no selection wall; the advisory decision switched at k = −7.
- XE9 failed in its pre-registered operationalisation: L · distance did not converge to a non-zero constant (1e-4 at L = 1e3, smaller
  at 1e4) because here the toric point approaches the leximin coupling faster than 1/L. The theorem's O(1/L) bound holds; the test
  assumed a non-zero 1/L coefficient. Recorded as a finding.
- XE18 failed: rho_q = 0.035 < 2 eta = 0.180, so Theorem 122(iii) gives no guarantee, and the coalition's anchor distance 0.063 is below
  kappa = 0.090: with ten inspection tables (3000 episodes) the coalition is not flagged. Recorded as a finding.

## Deviation D3 (post hoc diagnostic, added after seeing XE4b)

D1's rule removed the context carrying the elastic slack, which turned out to be {M,C,Y}: the coalition's own context. So XE4b, as
specified, cannot answer XE4's question (it removes the coalition along with the context). A diagnostic is added: remove {M,C} instead,
keep {M,C,Y} with the coalition, and test J_z. It is reported, labelled post hoc, and not used downstream; the downstream estimated
ranges stay on J_z^- as D1 specified. That the elastic attribution named {M,C,Y} rather than {M,C} (the context the other population
actually entered) is itself recorded: compatibility locates a conflict between tables, not which table is wrong.

## Classification of the failures (added 2026-11-11, after review; no result changed)

| check | kind of failure | what it shows |
|---|---|---|
| XE2 | detection (estimation) | a population shift spread over ten sources is too thin for a per-source test |
| XE3 | consequence of XE2 | with the shifted sources pooled, the estimated fibre is empty |
| XE4 | not testable as specified; answered post hoc (D3) | the coherent coalition passes compatibility, as predicted |
| D1 (elastic step) | attribution | incompatibility shows that tables conflict, not which table is wrong |
| coalition (XE4, XE18) | adversarial coherence | a coordinated coalition flips a decision while every compatibility test passes |
| XE9 | test operationalisation | the theorem bounds the error by O(1/L); the test assumed a non-zero 1/L term |
| XE10 | prediction about the data | the selection walls lay farther than 40 counts; the decision hyperplane passed 0.0006 from the selected point |
| XE18 | under-powered design | the theorem's hypothesis rho_q > 2 eta failed, so its guarantee never applied |

No failure exposes a limit of a theorem. The identities that hold to 1e-16 or 1e-17 check that the implementation agrees with the proofs;
they are not independent evidence for the theorems.
