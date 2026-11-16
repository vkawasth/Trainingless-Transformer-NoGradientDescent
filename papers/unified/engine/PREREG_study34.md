# Study 34: the structural state under sequential evidence (sec:dynamics). Pre-registered 2026-11-14 before any code for it ran.
Proof-first: every structural statement is a proposition proved in the paper; checks DX1-DX6 are IDENTITY checks of those proofs on
concrete streams. DX7 is the only hypothesis (cost), DX8 a record.

Setting. The 3 x 3 x 2 triangle population of Studies 29-33 (seed 20261129, counts over 600). Catalog of contexts: AB, BC, AC.
Query: g(P) = P(A = 0, C = 0 | B = 0) (upper and lower end), threshold tau = midpoint of its band over the first record's fibre.
Streams: 40 streams of 60 steps (seed 34). A true law drifts (each step: P <- normalise(P * exp(0.05 Z)), Z standard normal per cell).
Each step is one update: with probability 0.6 a revision (every present context's table is replaced by the true law's), 0.2 an
addition of an absent context, 0.2 a withdrawal of a present context (never below one context). Start: all three contexts present.
Checks.
  DX1 IDENTITY prop:fibremotion: under revisions the fibre moves by translation inside a fixed affine family: the direction space
               (kernel of the context map) is unchanged by revisions and changes only at additions (shrinks) and withdrawals (grows)
  DX2 IDENTITY prop:certstate: the summary (context set, data, one certifying basis per query end) answers every query exactly at every
               step (1e-9 against a fresh solve); an old basis is reused whenever it still certifies, else a fresh programme is solved;
               values never depend on the path; the canonical basis (lexicographically smallest optimal basis) is path independent
  DX3 IDENTITY prop:crossing: at each revision, the dual certificate of n - tau d at the old data gives a time t* such that along the
               straight path from old to new data the upper end stays below tau (or the lower end above) for all t < t* (101 points)
  DX4 IDENTITY prop:hidden: with data fixed, a law moving inside the fibre can move the decision (g vs tau) iff the fibre straddles tau
               (checked at every step: straddle <=> min < tau < max; non-straddle => the decision is the same at all vertices sampled
               by 20 random fibre points)
  DX5 IDENTITY prop:separate: for each single-cell anchor e (observing P(cell e) exactly) and each step's fibre (first 5 steps of each
               stream): for t on a 21-point grid of the cell's range over the fibre, the new upper end equals the old iff t lies in the
               cell's range over the optimal face of the upper end (1e-9)
  DX6 IDENTITY prop:historyfunctor: the KL updater (I-projection steps, refit by IPF) equals the recomputer (maximum entropy) along
               add-only and revision-only paths (1e-10); after publishing and retracting a foreign full joint table, the defect obeys
               KL(updater || recomputer) = H(recomputer) - H(updater) (1e-10)
  DX7 HYPOTHESIS cost: at least 50% of query-end updates at revision steps reuse the previous certificate (no programme solved)
  DX8 RECORD   reuse rates by update type; number of chamber (basis) changes; straddle share; empty-fibre events (none expected)
Output: check_dynamics_output.txt, study34_dynamics/dynamics.json.
