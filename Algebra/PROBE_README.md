# Intrinsic-Dimension Probe — files requested

## Corpora (validated structure axis)
iid < english < code < repeat   (increasing compositional structure)
+ english_shuf, code_shuf = shuffle controls (same unigram stats, ZERO structure).
Axis validated by gzip-ratio + block-entropy-gap in scripts/corpus_clean.py:
  iid    gzip 1.001  blockH-gap 0.000   (null, no structure)
  english gzip 0.282 blockH-gap 0.362   (mid)
  code   gzip 0.259  blockH-gap 0.436   (high)
  repeat gzip 0.007  blockH-gap 1.494   (degenerate = the original corpus)

## Scripts
- corpus_clean.py        : builds corpora + the gzip/shuffle-gap axis validation
- strand_manifold.py     : the batch->strand intrinsic-dim probe (linear vs
                           nonlinear-random-feature vs kNN, PR, dist->sim).
                           s_B = sign(grad)  -- the SECOND strand sense (surviving
                           object), NOT the refuted refinement-ladder strand.
- fixed_model_strand.py  : freeze-theta batch sweep; pairwise strand corr + PR.

## Compiler
compiler_geometri_patched_86.py : loaded via
  exec(src[:src.find("# PHASE 1")]) to get model/get_batch/etc.

## Two open decisions (make against the code, not from memory)
1. Vocab mismatch: model expects 1017, corpora are V=256. Remap corpora into
   model vocab (keeps checkpoint, scrambles token stats the axis relies on) vs.
   adapt model dims (keeps corpora/axis, discards checkpoint + prior-number
   comparability). This changes what the probe measures.
2. Embedding confound (from the epsilon table): embedding is non-perturbative
   because it memorizes a single repeated corpus; structured corpora change that
   regime directly. Run probe BLOCK-PARAMETERS-ONLY as primary; embedding as a
   separate arm. An embedding-inclusive number may move for reasons unrelated to
   the batch->strand map.

## Discriminator
Does nonlinear BEAT linear as structure rises (iid->english->code)?
  YES -> strand field low-dim on real corpora -> dictionary compression reopens,
         sparsity refutation was corpus-specific.
  NO  -> high-dimensionality is a law of learning -> wall holds in generality.
Shuffle controls (english vs english_shuf, code vs code_shuf) isolate structure
from unigram statistics -- this is the whole point.
