# Study 91 AMEND3 (procedural; registered 2026-11-24 before the check is resumed)
The single check invocation (check91.py, started 2026-11-23 23:55) was stopped by the environment's 30-minute limit for a background
command while QF5 was running. QF1-QF4 had been decided and written (check91_output_inv1.txt, check91_run_inv1.log); they stand as
recorded and are NOT recomputed. Timing on a non-event segment of a held-out walk (no QF quantity computed): 1.26 s per tracking step
of all 96 strands, so one QF5 loop (400 steps) takes ~8 min and QF5 ~4 h on one process, beyond any single command limit here.
Amendment (procedure only): the remainder runs as check91_resume.py. Its QF5, QF6 and R7 code is copied verbatim from check91.py
(same events selected, same loop radius 1e-3, n = 400, same tolerances and pass conditions). QF5 is split into six jobs
(path 0, 1, 2 x caustic / off-tie), each writing its windings and pass flags to study91_3x4/qf5/; the final step combines them into
the single QF5 line exactly as check91.py would, then runs QF6 and R7 and appends to check91_output.txt. No threshold, event,
selection or tolerance changes. Thread count fixed at 1 per job.

## AMEND3 erratum (2026-11-24, before the remaining jobs)
The first job (path 0, caustic) completed all its loops but crashed when saving: the windings were numpy longdouble, which json cannot
write. Fixed by casting the winding and pass flag to Python float/bool before saving (check91_resume.py); nothing computed changes.
For the path 0 jobs, whose loops had already run, the json is rebuilt from the job log, which prints (t, winding/pi, pass) for every
loop selected (qf5/rebuild_from_log.py); the loops are not rerun.
