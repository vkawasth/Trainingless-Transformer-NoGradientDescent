# Study 35: leximin and Theorem 196 at 4 x 6; cheap exact structure; coarsening. Pre-registered 2026-11-14 before any code for it ran.
Proof-first: LZ1-LZ3 check propositions proved in the paper (prop:twowaycomplete, prop:coarsen, max-ent merge consistency).
Population: 2000 nondegenerate 4 x 6 margins, integers over 240, Dirichlet(2) (seed 20261135); the fibre has dimension 15.
Leximin by the recursion of lem:lexunique over the facet cells, floating-point programmes (tolerance 1e-9) for the census; a cell whose
Frechet upper bound min(f_i, g_j) does not exceed the current level is declared blocked without a programme (pruning).
Merges: rows {0,1},{2,3}; columns {0,1},{2,3},{4,5} (coarse table 2 x 3).
Checks.
  LZ1 IDENTITY prop:twowaycomplete on 500 tables: for row set S = {0,1}, the set of achievable column vectors s_j = sum_{i in S} x_ij is
               the box [max(0, g_j - f(S^c)), min(g_j, f(S))] cut by sum s = f(S): 20 random directions per table, support function of the
               box-and-plane (closed form) equals the linear programme's maximum (1e-9)
  LZ2 IDENTITY prop:coarsen on 500 tables: for every coarse cell (I, J), the band of P(column group J | row group I) on the fine fibre
               equals the band on the coarse fibre with merged margins (1e-9)
  LZ3 IDENTITY maximum entropy commutes with merging: merge(f g^T) = merged f (merged g)^T (1e-15), all 2000
  LZ4 RECORD   census: leximin level patterns (r_j), number of levels, linear programmes used and saved by pruning
  LZ5 IDENTITY Theorem 196 at 4 x 6: on the first census table of each of the (up to) 8 most frequent r-patterns, with levels certified in
               exact rational arithmetic (certify_max), the scale valuations of Q/W at the positive critical point (L = 60, 120, 240,
               extrapolated in 1/L) equal h_j - h_1 with multiplicity r_j (tolerance 0.01)
  LZ6 RECORD   leximin coarsening defect: max |merge(leximin(fine)) - leximin(coarse)| per table; share of tables with defect <= 1e-9
  LZ7 HYPOTHESIS leximin is not merge consistent in general: defect > 1e-9 in at least 50% of the census
Output: check_four6_output.txt, study35_four6/four6.json.
