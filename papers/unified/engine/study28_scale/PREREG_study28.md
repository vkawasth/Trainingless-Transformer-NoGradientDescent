# Study 28: scaling of the stratified ledger (sec:ledger), pre-registered 2026-11-13 before any run

Question. For exact-margin credal sets, does the expensive work (linear programmes) grow with the number of tables T or with the
number of distinct basis regions encountered, and how do region-construction cost and per-table cost compare across shapes?

Shapes. 2 x n for n = 2..8; 3 x n for n = 3..7; 4 x n for n = 4..6 (15 shapes).
Populations. For each shape, nondegenerate margins (f, g), integers over den = 1009 (prime), Dirichlet(2) as in check_stratledger.py,
seed 20261200 + 10 r + c. T in {1000, 3000, 10000}, nested (the first 1000 and 3000 of the 10000).
Decision quantity (unchanged from sec:ledger). P(cell (0,0) | rows {0,1} x columns {0,1}), defined when f0 + f1 + g0 + g1 > 1;
upper and lower ends before evidence (hi0, lo0) and after the evidence weights L(i,j) = 1 + ((i + 2j) mod 3)/2 (hi1, lo1).
Linear control. Three integer objectives in [-20, 20]^N per shape (seed 7 + shape index), all T = 10000 tables.
Order. Tables are processed in a fixed random order (seed 1).

Measurements (per shape, per T): defined tables; LP solves per band end; library size; regions used; wall-clock time split into
LP time and certification time; median per-table time of one direct LP (200 tables) and of one library membership check after
the build (the vectorised certify over all tables divided by T, summed over the library); memory of the ledger (library inverses
as float64 plus one int32 basis id per table and band end) against storing the bands alone (float64 per table and end); growth
curve (solves after each table).

Checks (decided now; thresholds fixed).
  LS1 IDENTITY  every defined table receives a certificate at all four band ends, at every shape and T; and an exact rational
                re-check (SL.exact_check) of 300 random defined tables per shape and band end at T = 10000 passes
  LS2 IDENTITY  linear control: solves per objective <= C(r + c - 2, r - 1) (prop:basiscount) at T = 10000, every shape
  LS3 IDENTITY  solves <= regions used (each solve creates the region of the table that triggered it), every shape, T and end
  LS4 HYPOTHESIS  sublinear growth: for every shape and end, solves(T = 10000) < 10 x solves(T = 1000) / 2, i.e. a tenfold
                increase in T at most quintuples the solves
  LS5 HYPOTHESIS  the ledger is cheaper than direct solving: at T = 10000, total build time (LP + certification, all four ends) is
                below T x 4 x the median direct LP time, for every shape
  LS6 HYPOTHESIS  the per-table membership check is at least 10 times cheaper than one direct LP, for every shape
  LS7 RECORD    regions used against T and against the tree bound r^(c-1) c^(r-1); fraction of solves in the second half of the
                order at T = 10000; memory ratio ledger / bands; a log-log slope of solves against T per shape

Reporting. Hypotheses that fail are reported as failures; thresholds are not changed after the run. First-run failures and any code
fixes are recorded in the script docstring. Output: check_ledgerscale_output.txt and ledgerscale.json (aggregates only; all data
synthetic).
