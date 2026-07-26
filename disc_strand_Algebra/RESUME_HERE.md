# Disc/Strand Program — Resume Point

## What this is
Empirical investigation of whether geometric/algebraic structure in a small
transformer's optimization can replace or cheapen gradient descent. Terminal
finding: it cannot (sparsity refuted 9 ways), but the OUTPUT of backprop admits
a complete compressed description (disc + strand). Full theory in the LaTeX.

## Key files
- `phase3_algebraic_structure.tex` — the complete writeup (4 sections + figure).
  Read `\subsection{sec:cst}` and `\subsection{sec:backward}` for the terminal
  statement. Section labels: sec:hazard, sec:frieze, sec:signentropy,
  sec:residual, sec:cst, sec:backward.
- `disc_strand_figure.png` — the theory diagram.
- Full prior transcript: `/mnt/transcripts/` (see compaction summary at top of
  the most recent one; `journal.txt` catalogs earlier transcripts).
- `scripts/` — all 114 experiment scripts.
- `corpora/` — validated test corpora for the NEXT experiment.

## The one open question (the reason to continue)
Everything was measured on ONE degenerate corpus (a single sentence looped ~300x).
The central finding — the strand field is high-dimensional, so it does not
compress across corpus states — may be an ARTIFACT of that triviality. A
structured corpus might make the batch->strand map low-dimensional, which would
REOPEN dictionary compression and overturn the sparsity refutation.

## The decisive next experiment (NOT yet run)
Intrinsic-dimension probe across a validated structure axis:
  corpora/  iid < english < code < repeat   (structure axis, validated by
  gzip-ratio and block-entropy-gap; see scripts/corpus_clean.py)
  Shuffle controls: english_shuf, code_shuf (same unigram stats, zero structure).

Protocol (see scripts/strand_manifold.py and fixed_model_strand.py for the
pattern):
  for each corpus:
    train transformer to a checkpoint
    freeze theta
    sweep many batches -> strand s_B = sign(grad)
    measure: linear vs nonlinear(random-feature) vs kNN prediction of s_B from
             batch embedding; effective dim (participation ratio);
             corr(batch distance, strand similarity)
  Discriminator: does nonlinear BEAT linear as structure rises?
    YES -> strand field is low-dim on real corpora -> dictionary reopens,
           sparsity refutation was corpus-specific.
    NO  -> high-dimensionality is a law of learning -> wall holds in generality.

## Compatibility note
The model in scripts expects vocab=1017; the corpora use V=256. Either remap
corpora into the model's vocab or adapt the model dims before the probe.
The compiler prefix is loaded from the uploaded
compiler_geometri_patched_86.py via exec of src[:src.find("# PHASE 1")].

## Stance to re-inhabit
Every claim in this program was checked against a control; many reversed sign
after one. Read the transcript as record, not as settled judgment — re-verify
before building on any result. The rigor was the point.
